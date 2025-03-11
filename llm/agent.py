import json
from typing import Generator

from dotenv import load_dotenv
from openai import OpenAI

from llm.sql_prompt import planner_prompt
from utils import load_params

load_dotenv(override=True)


def get_llm_client():
    return OpenAI(base_url="https://api.groq.com/openai/v1")


client = get_llm_client()


def llm_sql(system_message: str, client: OpenAI, user_prompt: str, stream: bool) -> str:
    completion = client.chat.completions.create(
        model=load_params("model_name"),
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        stream=stream,
    )
    if not stream:
        print(completion.usage)

    return completion.choices[0].message.content


def planner_llm(client: OpenAI, user_request: str, db_info: str, stream: bool) -> str:
    completion = client.chat.completions.create(
        model=load_params("planner_model_name"),
        messages=[
            {
                "role": "system",
                "content": planner_prompt.format(user_request=user_request, database_schema=db_info),
            }
        ],
        stream=stream,
    )
    return completion.choices[0].message.content


def llm_analysis(system_message: str, client: OpenAI, user_prompt: str, stream: bool) -> Generator:
    completion = client.chat.completions.create(
        model=load_params("model_name"),
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        stream=stream,
    )

    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            yield content
