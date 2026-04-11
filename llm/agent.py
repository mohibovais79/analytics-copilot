import os
from json import load
from typing import Generator

from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from sqlalchemy import over

from llm.models import PlannerResponse
from llm.sql_prompt import planner_prompt
from utils import load_params

load_dotenv(override=True)


def get_llm_client():
    return OpenAI(base_url=load_params("base_url"))


def llm_sql(
    system_message: str,
    user_prompt: str,
    stream: bool,
    client: OpenAI = get_llm_client(),
) -> str:
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


def planner_llm(user_request: str, db_info: str, memory_context: list) -> PlannerResponse:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    system_prompt = planner_prompt.format(database_schema=db_info)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"""system: {system_prompt}
        user: {user_request}
        **Memory Context** (latest messages have higher priority):
        {memory_context}
        """,
        config={
            "response_mime_type": "application/json",
            "response_schema": PlannerResponse,
        },
    )
    if response.parsed:
        return response.parsed
    else:
        return PlannerResponse()


def break_request(user_input: str, system_prompt, client: OpenAI = get_llm_client()) -> str:
    completion = client.chat.completions.create(
        model=load_params("model_name"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
    )
    return completion.choices[0].message.content


def llm_analysis(
    system_message: str,
    user_prompt: str,
    stream: bool,
    client: OpenAI = get_llm_client(),
) -> Generator:
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
