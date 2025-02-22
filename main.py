import json
import re
import time

import llm
import llm.analysis_prompt
import llm.sql_prompt
from engine.sql_executor import analyze_sqlite_db, execute_sql
from llm.agent import client, llm_analysis, llm_sql
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
    user_prompt = "find me movie titles starting with B order by number of votes in ascending order only top 3"

    db_info = analyze_sqlite_db()
    rag = Rag(db_info)

    vector_store = rag.vectorize()

    response = rag.llm_response(user_prompt, vector_store)

    print(response["answer"])

    table_names = response["answer"]

    db_info = analyze_sqlite_db(table_names=table_names)

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
