from dotenv import load_dotenv
from openai import OpenAI
from llm.prompt import get_system_message

load_dotenv(override=True)


def get_llm_client():
    return OpenAI(base_url="https://api.groq.com/openai/v1")


client = get_llm_client()


def llm_sql(system_message: str, client: OpenAI, user_prompt) -> str:
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content


def llm_analysis(client: OpenAI, user_prompt, sql_query, result) -> str:
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "you are a data analyst"},
            {
                "role": "user",
                "content": f"this is user query {user_prompt} and this sqlite3 query {sql_query} is executed to answer user question and this is result {result} of query now using this result provide a summarize answer.",
            },
        ],
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    db_info = ""
    user_prompt = ""
    print(
        llm_sql(
            get_system_message(db_info),
            client,
            user_prompt,
        )
    )
