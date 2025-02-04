import instructor
from models import VizResponse
from openai import OpenAI

from utils import load_params


def get_llm_client():
    return OpenAI(base_url="https://api.groq.com/openai/v1")


def llm_visualize(system_message: str, user_prompt: str) -> VizResponse:
    client = instructor.from_openai(get_llm_client(), mode=instructor.Mode.JSON_SCHEMA)

    completion = client.chat.completions.create(
        model=load_params("model_name"),
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        response_model=VizResponse,
    )
    return completion
