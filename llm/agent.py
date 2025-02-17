import json
from typing import Generator

from dotenv import load_dotenv
from openai import OpenAI

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
    with open("llm_responses.json", "a") as f:
        json.dump(completion.model_dump(), f, indent=4)
        f.write("\n")

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
    full_response = []

    for chunk in completion:
        full_response.append(chunk.model_dump())

        content = chunk.choices[0].delta.content
        if content:
            yield content

    with open("llm_responses.json", "a") as f:
        json.dump(full_response, f, indent=4)
        f.write("\n")
