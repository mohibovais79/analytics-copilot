import json
import re
import time

import llm
import llm.analysis_prompt
import llm.sql_prompt
from engine.sql_executor import execute_sql
from llm.agent import client, llm_analysis, llm_sql


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
    db_info = execute_sql("SELECT name, sql FROM sqlite_master WHERE type='table';")
    user_prompt = "can you list actors name  starting with d only 5 "
    query_gen_start = time.time()
    response = llm_sql(llm.sql_prompt.get_system_message(db_info), client, user_prompt, stream=False)
    query_gen_end = time.time()
    query_time = round(query_gen_end - query_gen_start, 2)
    json_response = clean_sql_text(response)
    for key, value in json_response.items():
        if key == "sql":
            if value is not None:
                query_execute_start = time.time()

                results = execute_sql(value)
                query_execute_end = time.time()
                query_execute_time = round(query_execute_end - query_execute_start, 2)

                print("\n")
                print(value)
                print("\n")
                print(results)
                print("\n")
                analysis_start = time.time()
                for response_chunk in llm_analysis(
                    llm.analysis_prompt.system_prompt,
                    client,
                    llm.analysis_prompt.get_user_prompt(user_prompt, value, results),
                    stream=True,
                ):
                    print(response_chunk, end="")
                analysis_end = time.time()
                analysis_time = round(analysis_end - analysis_start, 2)
        elif key == "refusal" and key is not None:
            print(value)
    print(
        f"\nquery generation time = {query_time} seconds.\n query execution time = {query_execute_time} seconds.\nAnalysis time = {analysis_time} seconds \n"
    )
