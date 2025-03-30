from typing import Optional

import backoff
from openai import AsyncOpenAI, RateLimitError

from utils import load_params


def get_prompt(llm_sql_result_1: str, test_sql_result_2: str) -> str:
    prompt = f"""
    You are a SQL query result comparison tool. Your task is to compare the results of two SQL queries and validate it against user question and calculate a similarity score between them. The score should be a value between 0 and 1, where 1 means the results are identical and 0 means they are completely different.

    Perform the following steps:
    1. Compare the user question with the generated SQL query results.
    2. Compare the two results row by row and column by column.
    3. Calculate the simmilarity between generated sql query resuylts and user question.
    4. Calculate the similarity score based on the percentage of matching rows and columns.
    5. Return only the similarity for both comparisons as a float value between 0 and 1.

    Do not include any explanations or additional text. Only return the similarity score for both comparisons.

    Example output:
    0.5,0.79

    Here are the two results to compare:

    1. **Result from LLM-generated SQL query:**
    {llm_sql_result_1}


    2. **Result from Test SQL query:**
    {test_sql_result_2}

    """
    return prompt


def get_llm_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url="https://api.groq.com/openai/v1")


client = get_llm_client()


@backoff.on_exception(backoff.expo, RateLimitError)
async def make_async_call(user_prompt: str, system_prompt: Optional[str] = None):
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        messages = [{"role": "user", "content": user_prompt}]
    response = await client.chat.completions.create(
        model=load_params("model_name"),
        messages=messages,
        temperature=0,
    )

    return response.choices[0].message.content
