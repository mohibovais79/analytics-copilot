import json
import os
import re

from langchain_community.vectorstores import FAISS

from engine.sql_executor import get_schema
from llm.agent import planner_llm
from llm.flow import AgentFlow
from rag.pipeline import Rag

if __name__ == "__main__":
    db_info_path = "database_info.json"
    rag_instance = Rag(db_info_path=db_info_path)
    rag_instance.vectorize()
    vector_store = FAISS.load_local("vector_store", rag_instance.embedding_model, allow_dangerous_deserialization=True)
    agent_flow = AgentFlow(db_info_path)
    while True:
        
        user_input = input("User: ")
        os.system("cls")
        print("User: ", user_input)

        if user_input.lower() == "exit":
            break
        database_schema = get_schema(user_input, rag_instance, vector_store)

        planner_response = planner_llm(user_input, database_schema,agent_flow.memory_context)
        print(planner_response)
        planner_pattern = r"\{[^{}]*\}"

        if planner_response.mode.lower() == "sql":
            agent_flow.sql_flow(user_input)

        elif planner_response.mode.lower() == "python":
            agent_flow.python_flow(user_input)

        elif planner_response.mode.lower() == "general":
            agent_flow.general_flow(user_input)
        elif planner_response.mode.lower() == "multi":
            agent_flow.multi_flow(user_input)

        elif planner_response.refusal is not None:
            print(planner_response.refusal)

        print(len(agent_flow.memory_context))
