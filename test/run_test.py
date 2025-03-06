import asyncio
import json
import logging
import random
import time

import pandas as pd
import tiktoken
from openai import RateLimitError

import test.logging_config
from engine.sql_executor import analyze_sqlite_db, execute_sql, get_schema
from llm.agent import llm_sql
from llm.sql_prompt import get_system_message
from main import clean_sql_text
from rag.pipeline import Rag
from test.test_utils import get_prompt, make_async_call
from utils import db_conn, load_params


def prompt_length(prompt):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(prompt))
    return num_tokens


def get_random_value(cursor, table, column, condition=None, db_conn=db_conn()):
    query = f"SELECT {column} FROM {table}"
    if condition:
        query += f" WHERE {condition}"
    query += " ORDER BY RANDOM() LIMIT 1;"
    cursor = db_conn.cursor()
    cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    return result[0] if result else None


def fetch_random_parameters(test_case, db_conn):
    params = {}
    cursor = db_conn.cursor()

    for param, param_type in test_case["parameters"].items():
        if param == "N" and param_type == "integer":
            params[param] = random.randint(5, 10)
        elif param == "min_votes" and param_type == "integer":
            params[param] = get_random_value(cursor, "title_ratings", "numVotes", "numVotes > 1000")
            if params[param] is None:
                params[param] = random.randint(1000, 5000)
        elif param == "title_id" and param_type == "string":
            params[param] = get_random_value(cursor, "title_basics", "tconst")
        elif param == "region" and param_type == "string":
            params[param] = get_random_value(cursor, "title_akas", "region")
        elif param == "genre" and param_type == "string":
            cursor.execute("SELECT DISTINCT genres FROM title_basics WHERE genres IS NOT NULL;")
            genre_rows = cursor.fetchall()
            genres_set = set()
            for row in genre_rows:
                if row[0]:
                    for g in row[0].split(","):
                        genres_set.add(g.strip())
            if genres_set:
                params[param] = random.choice(list(genres_set))
            else:
                params[param] = "Drama"
        elif param == "year" and param_type == "integer":
            params[param] = get_random_value(cursor, "title_basics", "startYear", "startYear IS NOT NULL")
            if params[param] is None:
                params[param] = random.randint(1900, 2025)
        elif param == "person_id" and param_type == "string":
            params[param] = get_random_value(cursor, "name_basics", "nconst")
        elif param == "birth_year" and param_type == "integer":
            params[param] = get_random_value(
                cursor,
                "name_basics",
                "birthYear",
                "deathYear = 0 AND (primaryProfession LIKE '%actor%' OR primaryProfession LIKE '%actress%')",
            )
            if params[param] is None:
                params[param] = random.randint(1950, 2000)
        elif param == "tconst" and param_type == "string":
            params[param] = get_random_value(cursor, "title_basics", "tconst")
        elif param == "max_runtime" and param_type == "integer":
            params[param] = random.randint(60, 180)
        elif param == "min_movies" and param_type == "integer":
            params[param] = random.randint(1, 10)

    cursor.close()
    return params


def substitute_parameters(template: str, parameters: dict) -> str:
    result = template
    for key, value in parameters.items():
        result = result.replace("{" + key + "}", str(value))
    return result


async def main():
    start_time = time.time()
    results = []
    with open("test/test_cases.json", "r") as f:
        test_cases_json = json.load(f)

    db_info = get_schema()
    output_file = "test/test_results_2.csv"

    for test_case in test_cases_json["test_cases"]:
        params = fetch_random_parameters(test_case, db_conn())
        test_question = substitute_parameters(test_case["question"], params)
        test_sql = substitute_parameters(test_case["query"], params)
        logging.info(f"Test case: {test_case['question']}")
        logging.info(f"Test SQL: {test_sql}")

        try:
            print("system prompt length", prompt_length(get_system_message(db_info=db_info)))

            response = await make_async_call(
                user_prompt=test_question,
                system_prompt=get_system_message(db_info=db_info),
            )
        except RateLimitError:
            logging.info("Rate limit exceeded. Please try again later.")
            await asyncio.sleep(5)
            continue

        cleaned_response = clean_sql_text(response)
        if "sql" in cleaned_response and cleaned_response["sql"] is not None:
            generated_sql = cleaned_response["sql"]
            logging.info(f"Generated SQL: {generated_sql}")
            try:
                test_result = execute_sql(test_sql)
            except Exception as e:
                logging.info(f"Error executing test SQL: {e}")
                continue

            try:
                generated_result = execute_sql(generated_sql)
            except Exception as e:
                logging.info(f"Error executing generated SQL: {e}")
                continue

            if test_result is not None:
                lines = test_result.split("\n")
                if len(lines) > 2:
                    if lines[1].strip().startswith("|:"):
                        header = lines[:2]
                        data_rows = lines[2:]
                    else:
                        header = [lines[0]]
                        data_rows = lines[1:]
                    if len(data_rows) > 20:
                        data_rows = data_rows[:20]
                    test_result = "\n".join(header + data_rows)

            if generated_result is not None:
                lines = generated_result.split("\n")
                if len(lines) > 2:
                    if lines[1].strip().startswith("|:"):
                        header = lines[:2]
                        data_rows = lines[2:]
                    else:
                        header = [lines[0]]
                        data_rows = lines[1:]
                    if len(data_rows) > 20:
                        data_rows = data_rows[:20]
                    generated_result = "\n".join(header + data_rows)
        else:
            generated_sql = None
            test_result = execute_sql(test_sql)
            lines = test_result.split("\n")
            if len(lines) > 2:
                if lines[1].strip().startswith("|:"):
                    header = lines[:2]
                    data_rows = lines[2:]
                else:
                    header = [lines[0]]
                    data_rows = lines[1:]
                if len(data_rows) > 20:
                    data_rows = data_rows[:20]
                test_result = "\n".join(header + data_rows)

            generated_result = None

        prompt = get_prompt(test_result, generated_result)
        similarity_output = ""
        try:
            logging.info(f"Prompt: {prompt}")
            print("user prompt length", prompt_length(prompt))
            if prompt_length(prompt) > 6000:
                print(test_sql)
                print(generated_sql)
                break
                print(prompt)
            similarity_output = await make_async_call(user_prompt=prompt)
            logging.info(f"Similarity output: {similarity_output}")
        except RateLimitError:
            logging.info("Rate limit exceeded during similarity check. Please try again later.")
            await asyncio.sleep(5)
            continue

        try:
            relevance_score_str, results_similarity_score_str = similarity_output.split(",")
            relevance_score = float(relevance_score_str.strip())
            results_similarity_score = float(results_similarity_score_str.strip())
        except Exception as e:
            logging.info(f"Error parsing similarity output: {e}")
            relevance_score = None
            results_similarity_score = None

        results.append(
            {
                "User Question": test_question,
                "Test SQL": test_sql,
                "Generated SQL": generated_sql,
                "Relevance Score": relevance_score,
                "Results Similarity Score": results_similarity_score,
            }
        )

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    logging.info(f"\nResults written to {output_file}")
    end_time = time.time()
    logging.info(f"Test completed in {round(end_time - start_time, 2)} seconds.")


if __name__ == "__main__":
    asyncio.run(main())
