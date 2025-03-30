import asyncio
import json
import time

import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.templating import Jinja2Templates
from openai import RateLimitError
from sse_starlette.sse import EventSourceResponse

from engine.sql_executor import analyze_sqlite_db, execute_sql
from llm.agent import client, llm_analysis, llm_sql
from llm.analysis_prompt import get_user_prompt, system_prompt
from llm.sql_prompt import get_system_message
from main import clean_sql_text
from utils import db_conn

app = FastAPI(debug=True)
templates = Jinja2Templates(directory="templates")


async def run_llm_analysis(system_message: str, client, user_prompt: str, stream: bool):
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()

    def run_generator():
        try:
            for chunk in llm_analysis(system_message, client, user_prompt, stream):
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    loop.run_in_executor(None, run_generator)

    while True:
        item = await queue.get()
        if item is None:
            break
        yield item


async def event_stream(user_prompt: str):
    db_info = analyze_sqlite_db(db_conn())

    query_gen_start = time.time()
    try:
        response = llm_sql(
            get_system_message(db_info), client, user_prompt, stream=False
        )
    except RateLimitError:
        response = "Rate limit exceeded. Please try again later."
    query_gen_end = time.time()
    query_time = round(query_gen_end - query_gen_start, 2)

    json_response = clean_sql_text(response)
    analysis_time = 0
    query_execute_time = 0
    for key, value in json_response.items():
        if key == "sql" and value is not None:
            yield f"data: {json.dumps({'type': 'sql', 'value': value})}\n\n"
            await asyncio.sleep(0)

            query_execute_start = time.time()
            results = execute_sql(value)
            query_execute_end = time.time()
            query_execute_time = round(query_execute_end - query_execute_start, 2)

            yield f"data: {json.dumps({'type': 'results', 'value': results})}\n\n"
            await asyncio.sleep(0)

            analysis_start = time.time()
            try:
                async for response_chunk in run_llm_analysis(
                    system_prompt,
                    client,
                    get_user_prompt(user_prompt, value, results),
                    stream=True,
                ):
                    yield f"data: {json.dumps({'type': 'analysis', 'value': response_chunk})}\n\n"
                    await asyncio.sleep(0)
                analysis_end = time.time()
                analysis_time = round(analysis_end - analysis_start, 2)
            except RateLimitError:
                analysis_time = "Rate limit exceeded. Please try again later."

            analysis_end = time.time()
            analysis_time = round(analysis_end - analysis_start, 2)
        elif key == "refusal" and value is not None:
            yield f"data: {json.dumps({'type': 'refusal', 'value': value})}\n\n"
            await asyncio.sleep(0)

    yield f"data: {json.dumps({'type': 'timings', 'query_time': query_time, 'query_execute_time': query_execute_time, 'analysis_time': analysis_time})}\n\n"


@app.get("/stream")
async def stream(
    request: Request, prompt: str = Query(..., description="User prompt for the query")
):
    """
    SSE endpoint using sse_starlette's EventSourceResponse.
    Clients connect here to receive streaming messages.
    """
    return EventSourceResponse(event_stream(prompt))


@app.get("/")
async def index(request: Request):
    """
    Renders the chatbot frontend using a Jinja2 template.
    """
    return templates.TemplateResponse("chat.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
