import os
import sqlite3

import pandas as pd
from langchain_community.vectorstores import FAISS
from openai import RateLimitError

import llm.analysis_prompt
import llm.sql_prompt
from analysis_visualization_agent.executor import CodeExecutor
from analysis_visualization_agent.llm import llm_explain, llm_visualize
from analysis_visualization_agent.visualization_prompt import get_explaination_prompt, get_user_prompt, system_prompt
from engine.sql_executor import analyze_sqlite_db, execute_sql, get_schema
from llm.agent import break_request, llm_analysis, llm_sql
from llm.sql_prompt import break_request_prompt
from rag.pipeline import Rag
from utils import clean_sql_text, serialize_dataframe


def sql_flow(user_input: str, memory_context: list) -> dict:
    result_dict = {"user_prompt": user_input, "sql": None, "results": None, "analysis": None}
    database_path = "database_info.json"
    rag_instance = Rag(database_path)
    rag_instance.vectorize()
    vector_store = FAISS.load_local("vector_store", rag_instance.embedding_model, allow_dangerous_deserialization=True)

    database_schema = get_schema(user_input, rag_instance, vector_store)

    try:
        response = llm_sql(
            system_message=llm.sql_prompt.get_system_message(database_schema, memory_context),
            user_prompt=user_input,
            stream=False,
        )
    except RateLimitError:
        result_dict["analysis"] = "OpenAI API rate limit reached while generating SQL. Please try again later."
        return result_dict

    json_response = clean_sql_text(response)
    if not isinstance(json_response, dict):
        result_dict["analysis"] = "Error: SQL generation did not return a valid dictionary. Please try again."
        return result_dict

    sql_query = json_response.get("sql")
    if sql_query is None:
        result_dict["analysis"] = json_response.get("explanation", "No SQL generated and no explanation provided.")
        return result_dict

    try:
        results = execute_sql(sql_query)
    except (sqlite3.OperationalError, pd.errors.DatabaseError) as e:
        if "no such column" in str(e):
            full_schema = analyze_sqlite_db()
            response = llm_sql(
                system_message=llm.sql_prompt.get_system_message(full_schema, memory_context),
                user_prompt=user_input,
                stream=False,
            )
            json_response = clean_sql_text(response)
            sql_query = json_response.get("sql")
            results = execute_sql(sql_query)
        else:
            raise e

    results = serialize_dataframe(results)

    full_response = ""
    try:
        for response_chunk in llm_analysis(
            llm.analysis_prompt.system_prompt,
            llm.analysis_prompt.get_user_prompt(user_input, sql_query, results, memory_context),
            stream=True,
        ):
            full_response += response_chunk
    except RateLimitError:
        result_dict["analysis"] = "OpenAI API rate limit reached while analyzing SQL. Please try again later."
        return result_dict

    result_dict["sql"] = sql_query
    result_dict["results"] = results
    result_dict["analysis"] = full_response

    return result_dict


def python_flow(user_input: str):
    result_dict = {"user_prompt": user_input, "code": None, "results": None, "analysis": None}
    database_path = "database_info.json"
    rag_instance = Rag(database_path)
    rag_instance.vectorize()
    vector_store = FAISS.load_local("vector_store", rag_instance.embedding_model, allow_dangerous_deserialization=True)

    dataset_paths = [
        "data/brands.csv",
        "data/categories.csv",
        "data/customers.csv",
        "data/order_items.csv",
        "data/orders.csv",
        "data/products.csv",
        "data/staffs.csv",
        "data/stocks.csv",
        "data/stores.csv",
    ]
    df_dict: dict[str, pd.DataFrame] = {}
    for dataset_path in dataset_paths:
        fn = os.path.splitext(os.path.basename(dataset_path))[0]
        df_dict[fn] = pd.read_csv(dataset_path)

    df_summary = get_schema(user_input, rag_instance, vector_store)
    user_prompt = get_user_prompt(user_input)

    with open("requirements.txt", "r") as f:
        libraries = f.read()
    libraries = libraries.replace(">", "")

    try:
        response = llm_visualize(system_prompt.format(df_info=df_summary, libraries=libraries), user_prompt)
    except RateLimitError:
        result_dict["analysis"] = "OpenAI API rate limit reached while generating code. Please try again later."
        return result_dict

    if response.code is None:
        result_dict["analysis"] = "Error: No code was generated."
        return result_dict

    result_dict["code"] = response.code
    print("generated code", response.code)

    executor = CodeExecutor()
    try:
        final_df, clean_code = executor.execute_code(response.code)
    except Exception as e:
        result_dict["analysis"] = f"Error while executing code: {e}"
        return result_dict

    results = serialize_dataframe(final_df)
    result_dict["results"] = results

    full_response = ""
    try:
        explanation_prompt = get_explaination_prompt(results, user_input, clean_code)
        for response_chunk in llm_explain(explanation_prompt):
            print(response_chunk, end="")
            full_response += response_chunk
    except RateLimitError:
        result_dict["analysis"] = "OpenAI API rate limit reached while analyzing python code. Please try again later."
        return result_dict

    result_dict["analysis"] = full_response
    return result_dict


def general_flow(user_input: str, memory_context: list):
    result_dict = {"user_prompt": user_input, "code": None, "results": None, "analysis": None}
    database_path = "database_info.json"
    rag_instance = Rag(database_path)
    rag_instance.vectorize()
    vector_store = FAISS.load_local("vector_store", rag_instance.embedding_model, allow_dangerous_deserialization=True)

    database_schema = get_schema(user_input, rag_instance, vector_store)
    try:
        full_response = ""
        for response_chunk in llm_analysis(
            f"""Answer the user's question according to provided schema {database_schema}. If question is not relevant simply deny the request with reason. latest messages in the list need to be assigned more priority.
 This is the provided memory context: {memory_context}""",
            user_input,
            stream=True,
        ):
            full_response += response_chunk
            print(response_chunk, end="")
        result_dict["analysis"] = full_response
    except RateLimitError:
        print("OpenAI API rate limit reached while generating response. Please try again later.")
        result_dict["analysis"] = "OpenAI API rate limit reached while generating response. Please try again later."
        return result_dict


def multi_flow(user_input: str):
    multi_context = []
    database_path = "database_info.json"
    rag_instance = Rag(database_path)
    rag_instance.vectorize()
    vector_store = FAISS.load_local("vector_store", rag_instance.embedding_model, allow_dangerous_deserialization=True)

    database_schema = get_schema(user_input, rag_instance, vector_store)
    sql_list = break_request(user_input, break_request_prompt.format(context=multi_context, schema=database_schema))
    sql_list_cleaned = clean_sql_text(sql_list)

    queries = sql_list_cleaned.get("sql", [])

    if not queries:
        return {
            "user_prompt": user_input,
            "multi_context": [
                {
                    "user_prompt": user_input,
                    "sql": None,
                    "results": None,
                    "analysis": "No SQL queries were generated from the input.",
                }
            ],
        }

    for query in queries:
        try:
            results = execute_sql(query)
            results = serialize_dataframe(results)
        except Exception as e:
            multi_context.append(
                {
                    "user_prompt": user_input,
                    "sql": query,
                    "results": None,
                    "analysis": f"Error while executing query: {e}",
                }
            )
            continue

        full_response = ""
        try:
            for response_chunk in llm_analysis(
                llm.analysis_prompt.system_prompt,
                llm.analysis_prompt.get_user_prompt(user_input, query, results, multi_context),
                stream=True,
            ):
                full_response += response_chunk
        except RateLimitError:
            return {
                "user_prompt": user_input,
                "multi_context": [
                    {
                        "user_prompt": user_input,
                        "sql": query,
                        "results": results,
                        "analysis": "OpenAI API rate limit reached while analyzing SQL. Please try again later.",
                    }
                ],
            }

        multi_context.append(
            {
                "user_prompt": user_input,
                "sql": query,
                "results": results,
                "analysis": full_response,
            }
        )

    return {"user_prompt": user_input, "multi_context": multi_context}
