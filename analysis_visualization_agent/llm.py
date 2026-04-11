from typing import Generator

import instructor
from openai import OpenAI

from analysis_visualization_agent.models import VizResponse
from analysis_visualization_agent.utils import load_params


def get_llm_client():
    return OpenAI(base_url=load_params("base_url"))


def llm_visualize(system_message: str, user_prompt: str) -> VizResponse:
    client = instructor.from_openai(get_llm_client(), mode=instructor.Mode.JSON)

    completion = client.chat.completions.create(
        model=load_params("model_name"),
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        response_model=VizResponse,
    )
    return completion


def llm_explain(user_prompt: str) -> Generator:
    client = get_llm_client()

    completion = client.chat.completions.create(
        model=load_params("model_name"),
        messages=[
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )

    for chunk in completion:
        content = chunk.choices[0].delta.content
        if content:
            yield content
