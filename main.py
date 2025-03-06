import json
import os
import re
import sqlite3
import time

import pandas as pd
from langchain_community.vectorstores import FAISS

import llm
import llm.analysis_prompt
import llm.sql_prompt
from analysis_visualization_agent.executor import CodeExecutor
from analysis_visualization_agent.llm import llm_explain, llm_visualize
from analysis_visualization_agent.utils import dataframe_to_markdown
from analysis_visualization_agent.visualization_prompt import get_explaination_prompt, get_user_prompt, system_prompt
from engine.sql_executor import analyze_sqlite_db, execute_sql, get_schema
from llm.agent import client, llm_analysis, llm_sql, planner_llm
from rag.pipeline import Rag


def clean_sql_text(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_str = match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None


if __name__ == "__main__":
    memory_context = []
    db_info_path = "database_info.json"
    rag_instance = Rag(db_info_path=db_info_path)
    rag_instance.vectorize()
    vector_store = FAISS.load_local("vector_store", rag_instance.embedding_model, allow_dangerous_deserialization=True)
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        planner_response = planner_llm(client, user_input, stream=False)
        print(planner_response.lower())
        if planner_response.lower() == "sql":
            query_execute_time = 0
            analysis_time = 0

            database_schema = get_schema(user_input, rag_instance, vector_store)
            print("\n")
            print("relevant schema", database_schema)

            query_gen_start = time.time()
            response = llm_sql(
                llm.sql_prompt.get_system_message(database_schema, memory_context), client, user_input, stream=False
            )
            query_gen_end = time.time()
            query_time = round(query_gen_end - query_gen_start, 2)
            json_response = clean_sql_text(response)
            print("\n")

            for key, value in json_response.items():
                if key == "sql":
                    if value is not None:
                        query_execute_start = time.time()
                        try:
                            results = execute_sql(value)
                        except (sqlite3.OperationalError, pd.errors.DatabaseError) as e:
                            if "no such column" in str(e):
                                print("Retrying sql query")
                                full_schema = analyze_sqlite_db()
                                response = llm_sql(
                                    llm.sql_prompt.get_system_message(full_schema, memory_context),
                                    client,
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

                        lines = results.split("\n")
                        if len(lines) > 2:
                            if lines[1].strip().startswith("|:"):
                                header = lines[:2]
                                data_rows = lines[2:]
                            else:
                                header = [lines[0]]
                                data_rows = lines[1:]
                            if len(data_rows) > 20:
                                data_rows = data_rows[:20]
                            results = "\n".join(header + data_rows)

                        print("\nExecuted SQL Query:")
                        print(value)
                        print("\nResults:")
                        print(results)
                        print("\n")

                        analysis_start = time.time()
                        full_response = ""
                        for response_chunk in llm_analysis(
                            llm.analysis_prompt.system_prompt,
                            client,
                            llm.analysis_prompt.get_user_prompt(user_input, value, results, memory_context),
                            stream=True,
                        ):
                            print(response_chunk, end="")
                            full_response += response_chunk
                        analysis_end = time.time()
                        analysis_time = round(analysis_end - analysis_start, 2)

                        if (
                            user_input is not None
                            and value is not None
                            and results is not None
                            and full_response is not None
                        ):
                            memory_context.append(
                                {
                                    "user_prompt": user_input,
                                    "sql": value,
                                    "results": results,
                                    "analysis": full_response,
                                }
                            )

                elif key == "refusal" and value is not None:
                    print(value)

            if not json_response.get("sql") and not json_response.get("refusal") and json_response.get("explanation"):
                print(json_response["explanation"])

            print(
                f"\nquery generation time = {query_time} seconds.\n query execution time = {query_execute_time} seconds.\nAnalysis time = {analysis_time} seconds \n"
            )
        elif planner_response.lower() == "python":
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
            df_summary = dataframe_to_markdown(df_dict)
            user_prompt = get_user_prompt(df_summary, user_input)
            response = llm_visualize(system_prompt, user_prompt)

            if response.code is not None:
                print(response.code)
                executor = CodeExecutor(df_dict)
                final_df, clean_code = executor.execute_code(response.code)
                final_df_md = final_df.head(10).to_markdown()

                explanation_prompt = get_explaination_prompt(final_df_md, user_input, clean_code)
                for response_chunk in llm_explain(explanation_prompt):
                    print(response_chunk, end="")

            else:
                print(response.refusal)

        else:
            print("Invalid request")
