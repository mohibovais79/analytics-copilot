import os
import sqlite3
import time

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


class AgentFlow:
    def __init__(self, db_path: str):
        self.memory_context = []
        self.rag_instance = Rag(db_info_path=db_path)
        self.rag_instance.vectorize()
        self.vector_store = FAISS.load_local(
            "vector_store", self.rag_instance.embedding_model, allow_dangerous_deserialization=True
        )

    def sql_flow(self, user_input: str):
        query_execute_time = 0
        analysis_time = 0

        database_schema = get_schema(user_input, self.rag_instance, self.vector_store)
        print("\n")

        query_gen_start = time.time()
        try:
            response = llm_sql(
                system_message=llm.sql_prompt.get_system_message(database_schema, self.memory_context),
                user_prompt=user_input,
                stream=False,
            )
        except RateLimitError:
            print("OpenAI API rate limit reached while generating sql. Please try again later.")
            return
        query_gen_end = time.time()
        query_time = round(query_gen_end - query_gen_start, 2)
        json_response = clean_sql_text(response)

        print(json_response)
        if not isinstance(json_response, dict):
            print("Error, Please try again.")
            return

        print("\n")

        for key, value in json_response.items():
            if key == "sql":
                if value is not None:
                    print("Executing SQL Query")
                    query_execute_start = time.time()
                    try:
                        results = execute_sql(value)
                    except (sqlite3.OperationalError, pd.errors.DatabaseError) as e:
                        if "no such column" in str(e):
                            print("Retrying sql query")
                            full_schema = analyze_sqlite_db()

                            response = llm_sql(
                                llm.sql_prompt.get_system_message(full_schema, self.memory_context),
                                user_input,
                                stream=False,
                            )
                            json_response = clean_sql_text(response)
                            value = json_response.get("sql")
                            results = execute_sql(value)
                        else:
                            raise e
                        query_execute_end = time.time()
                        query_execute_time = round(query_execute_end - query_execute_start, 2)

                        results = serialize_dataframe(results)

                    print("\nExecuted SQL Query:")
                    print(value)
                    print("\nResults:")
                    print(results)
                    print("\n")

                    analysis_start = time.time()
                    full_response = ""
                    try:
                        for response_chunk in llm_analysis(
                            llm.analysis_prompt.system_prompt,
                            llm.analysis_prompt.get_user_prompt(user_input, value, results, self.memory_context),
                            stream=True,
                        ):
                            print(response_chunk, end="")
                            full_response += response_chunk
                    except RateLimitError:
                        print("OpenAI API rate limit reached while analyzing sql. Please try again later.")
                        return
                    print("\n")
                    analysis_end = time.time()
                    analysis_time = round(analysis_end - analysis_start, 2)

                    if (
                        user_input is not None
                        and value is not None
                        and results is not None
                        and full_response is not None
                    ):
                        self.memory_context.append(
                            {
                                "user_prompt": user_input,
                                "sql": value,
                                "results": results,
                                "analysis": full_response,
                            }
                        )
                elif value is None and json_response.get("explanation") is not None:
                    print(json_response.get("explanation"))

                print(
                    f"\nquery generation time = {query_time} seconds.\n query execution time = {query_execute_time} seconds.\nAnalysis time = {analysis_time} seconds \n"
                )

    def python_flow(self, user_input: str):
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
        df_summary = get_schema(user_input, self.rag_instance, self.vector_store)
        user_prompt = get_user_prompt(user_input)
        with open("requirements.txt", "r") as f:
            libraries = f.read()
        libraries = libraries.replace(">", "")
        try:
            response = llm_visualize(system_prompt.format(df_info=df_summary, libraries=libraries), user_prompt)
        except RateLimitError:
            print("OpenAI API rate limit reached while generating code. Please try again later.")

        if response.code is not None:
            print("generated code", response.code)
            executor = CodeExecutor()
            try:
                final_df, clean_code = executor.execute_code(response.code)
            except Exception as e:
                print("Error while executing code:", e)
                return
            results = serialize_dataframe(final_df)

            full_response = ""
            try:
                explanation_prompt = get_explaination_prompt(results, user_input, clean_code)
                for response_chunk in llm_explain(explanation_prompt):
                    print(response_chunk, end="")
                    full_response += response_chunk
            except RateLimitError:
                print("OpenAI API rate limit reached while analyzing python code. Please try again later.")
                return
            self.memory_context.append(
                {
                    "user_prompt": user_input,
                    "code": clean_code,
                    "results": results,
                    "analysis": full_response,
                }
            )

    def general_flow(self, user_input: str):
        database_schema = get_schema(user_input, self.rag_instance, self.vector_store)
        try:
            for response_chunk in llm_analysis(
                f"""Answer the user's question according to provided schema {database_schema}. If question is not relevant simply deny the request with reason. latest messages in the list need to be assigned more priority.
 This is the provided memory context: {self.memory_context}""",
                user_input,
                stream=True,
            ):
                print(response_chunk, end="")
        except RateLimitError:
            print("OpenAI API rate limit reached while generating response. Please try again later.")
            self.memory_context.append(
                {"user_prompt": user_input, "code": None, "results": None, "analysis": response_chunk}
            )

    def multi_flow(self, user_input: str):
        multi_context = []
        database_schema = get_schema(user_input, self.rag_instance, self.vector_store)
        sql_list = break_request(user_input, break_request_prompt.format(context=multi_context, schema=database_schema))
        print(sql_list)
        sql_list_cleaned = clean_sql_text(sql_list)

        queries = sql_list_cleaned.get("sql", [])
        for query in queries:
            print("Executing SQL Query")
            print(query)
            results = execute_sql(query)
            results = serialize_dataframe(results)
            print("Results:")
            print(results)
            print("\n")

            full_response = ""
            try:
                for response_chunk in llm_analysis(
                    llm.analysis_prompt.system_prompt,
                    llm.analysis_prompt.get_user_prompt(user_input, query, results, multi_context),
                    stream=True,
                ):
                    print(response_chunk, end="")
                    full_response += response_chunk
            except RateLimitError:
                print("OpenAI API rate limit reached while analyzing sql. Please try again later.")
                return
            multi_context.append(
                {
                    "user_prompt": user_input,
                    "sql": query,
                    "results": results,
                    "analysis": "",
                }
            )
