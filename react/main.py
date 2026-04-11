import json
import os
from pyexpat import model
from typing import Any, Dict, List, Optional

import instructor
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field

from engine.sql_executor import get_schema
from rag.pipeline import Rag
from react.tools import general_flow, multi_flow, python_flow, sql_flow

load_dotenv(override=True)

tools = [
    {
        "type": "function",
        "function": {
            "name": "sql_flow",
            "description": "Generates and executes SQL based on the user's question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_input": {"type": "string", "description": "User analytical question"},
                    "memory_context": {"type": "array", "description": "Past interactions or context from the user."},
                },
                "required": ["user_input", "memory_context"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_flow",
            "description": "Generates code based on a user question, executes the code, and returns a dictionary containing user_prompt, code, results, and analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "User question on which code generation and execution need to be performed.",
                    }
                },
                "required": ["user_input"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "general_flow",
            "description": "Processes a user question along with memory context to provide a general analysis using a predefined schema. Returns a dictionary containing user_prompt, code, results, and analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_input": {"type": "string", "description": "User question to be analyzed."},
                    "memory_context": {
                        "type": "array",
                        "description": "Past interactions or context to be considered in the analysis.",
                    },
                },
                "required": ["user_input", "memory_context"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multi_flow",
            "description": "Generates and executes multiple SQL queries from a user question. Returns a dictionary containing user_prompt and multi_context (a list of objects with user_prompt, sql, results, and analysis for each query).",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "User question from which multiple SQL queries are generated and executed.",
                    }
                },
                "required": ["user_input"],
            },
        },
    },
]


class ReasoningStep(BaseModel):
    thought: str = Field(description="The agent's reasoning about what to do next")
    should_use_tool: bool = Field(description="Whether a tool should be used")


class ToolCallStep(BaseModel):
    tool_name: str = Field(description="The name of the tool to call")
    tool_parameters: Dict[str, Any] = Field(description="Parameters for the tool call")


class AgentDecision(BaseModel):
    reasoning: ReasoningStep = Field(description="The agent's reasoning about what to do")
    tool_call: Optional[ToolCallStep] = Field(None, description="Tool to call if needed")
    is_complete: bool = Field(description="Whether the reasoning-action cycle is complete")
    final_answer: Optional[str] = Field(None, description="Final answer to provide to the user if complete")


class IterativeReActAgent:
    def __init__(self, model_name="llama-3.3-70b-versatile", max_iterations=5):
        self.client = instructor.from_groq(Groq(), mode=instructor.Mode.JSON)
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.conversation_history = []
        self.memory_context = []

    def _add_to_history(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})

    def _add_to_memory(self, interaction: Dict[str, Any]):
        if "final_answer" in interaction:
            simplified_interaction = {
                "user_prompt": interaction["user_prompt"],
                "final_answer": interaction["final_answer"],
            }
            self.memory_context.append(simplified_interaction)
        elif "tool" in interaction:
            simplified_interaction = {"user_prompt": interaction["user_prompt"], "tool_used": interaction["tool"]}
            if "result" in interaction and "analysis" in interaction["result"]:
                simplified_interaction["analysis"] = interaction["result"]["analysis"]

            self.memory_context.append(simplified_interaction)

        if len(self.memory_context) > 10:
            self.memory_context = self.memory_context[-10:]

    def _format_tool_response(self, tool_name: str, response: Dict[str, Any]) -> str:
        if tool_name == "sql_flow":
            result = f"SQL Query: ```sql\n{response.get('sql', 'No SQL generated')}\n```\n\n"
            if response.get("results"):
                result += f"Results:\n```json\n{json.dumps(response.get('results'), indent=2)[:500]}...\n```\n\n"
            result += f"Analysis: {response.get('analysis', 'No analysis available')}"
            return result

        elif tool_name == "python_flow":
            result = f"Generated Code: ```python\n{response.get('code', 'No code generated')}\n```\n\n"
            if response.get("results"):
                result += f"Results:\n```json\n{json.dumps(response.get('results'), indent=2)[:500]}...\n```\n\n"
            result += f"Analysis: {response.get('analysis', 'No analysis available')}"
            return result

        elif tool_name == "multi_flow":
            result = "Multiple Queries Analysis:\n\n"
            for i, context in enumerate(response.get("multi_context", [])):
                result += f"Query {i + 1}: ```sql\n{context.get('sql', 'No SQL generated')}\n```\n"
                result += f"Analysis: {context.get('analysis', 'No analysis available')}\n\n"
            return result

        elif tool_name == "general_flow":
            return f"Analysis: {response.get('analysis', 'No analysis available')}"

        return f"Raw response: {json.dumps(response, indent=2)}"

    def _prepare_tool_parameters(self, tool_name: str, user_input: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        prepared_params = {}

        prepared_params["user_input"] = user_input

        if tool_name in ["sql_flow", "general_flow"]:
            prepared_params["memory_context"] = self.memory_context

        return prepared_params

    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        tool_function = {
            "sql_flow": sql_flow,
            "python_flow": python_flow,
            "general_flow": general_flow,
            "multi_flow": multi_flow,
        }.get(tool_name)

        if not tool_function:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return tool_function(**parameters)
        except Exception as e:
            return {"error": f"Error executing {tool_name}: {str(e)}"}

    def process_user_input(self, user_input: str, database_schema: str) -> str:
        self._add_to_history("user", user_input)

        system_message = f"""You are a ReAct (Reasoning and Acting) agent that helps with data analysis. 
Follow these steps for each request:

1. THINK: Reason about what the user is asking and determine:
   - What the user wants to accomplish
   - What data is needed
   - Which tool would be most appropriate
   - Identify parameters needed to call relevant tool from tool description
   - Whether the request is valid according to system policies

2. ACT: Select and call the most appropriate tool based on the user's request.
   IMPORTANT: You MUST set is_complete to false when you need to use a tool.
   Only set is_complete to true when you have a final answer based on tool results.

DO NOT include SQL queries or Python code in your tool parameters. The tools will generate the appropriate code based on the user input.
ONLY include the required parameters as specified in the tool descriptions. For most tools, this is just 'user_input'.

   Choose tools based on these guidelines:
   - Use 'sql_flow' for data retrieval, filtering, or aggregation requests
   - Use 'python_flow' for visualization requests (charts, graphs, plots) or calculations intended for visual output
   - Use 'multi_flow' for complex questions that require multiple SQL queries
   - Use 'general_flow' for general questions about the data or system

   Refuse requests that:
   - Involve write operations (INSERT, UPDATE, DELETE)
   - Request data beyond the provided database schema
   - Show malicious intent
   - Involve sensitive or explicit content
   - Request file I/O operations
   - Use libraries other than Seaborn/Matplotlib for visualization and pandas/numpy for analysis
   - Ask for actions outside data analytics
   - Have unclear requirements

3. REFLECT: Analyze the results and decide if you need more information or can provide a final answer.

Before selecting a tool, verify that every table, column, or field mentioned in the request exists in the provided database schema. If any element is missing, explain the issue to the user.

The database schema includes: {database_schema}



Once you have gathered all the information needed, provide a final answer and set is_complete to true.
        """

        messages = [
            {"role": "system", "content": system_message},
        ]

        relevant_history = (
            self.conversation_history[-6:] if len(self.conversation_history) > 6 else self.conversation_history
        )
        messages.extend(relevant_history)

        workspace = {"query": user_input, "steps": [], "current_context": ""}

        for iteration in range(self.max_iterations):
            try:
                # Add current context to messages
                iteration_messages = messages.copy()
                if workspace["current_context"]:
                    iteration_messages.append(
                        {
                            "role": "assistant",
                            "content": f"Previous reasoning steps and tool results:\n{str(workspace['current_context'])}",
                        }
                    )

                response = self.client.chat.completions.create(
                    model=self.model_name, response_model=AgentDecision, messages=iteration_messages
                )

                if response.is_complete and not workspace["steps"]:
                    print("WARNING: Agent marked completion without tool execution. Overriding decision.")
                    response.is_complete = False
                    if not response.reasoning.should_use_tool:
                        response.reasoning.should_use_tool = True
                        if not response.tool_call:
                            default_tool = "sql_flow"
                            if (
                                "visual" in user_input.lower()
                                or "chart" in user_input.lower()
                                or "plot" in user_input.lower()
                            ):
                                default_tool = "python_flow"
                            if "all" in user_input.lower() and "queries" in user_input.lower():
                                default_tool = "multi_flow"

                            response.tool_call = ToolCallStep(
                                tool_name=default_tool, tool_parameters={"user_input": user_input}
                            )
                            print(f"Added default tool call to {default_tool}")

                step_record = f"Step {iteration + 1}:\nThought: {response.reasoning.thought}\n"
                print(f"\n Thought {iteration + 1}: {response.reasoning.thought}")

                if response.is_complete and response.final_answer:
                    step_record += f"Final Answer: {response.final_answer}\n"
                    workspace["steps"].append(step_record)
                    workspace["current_context"] += step_record

                    self._add_to_memory({"user_prompt": user_input, "final_answer": response.final_answer})

                    self._add_to_history("assistant", response.final_answer)
                    print(f"\n Final Answer: {response.final_answer}")

                    return response.final_answer

                if response.reasoning.should_use_tool and response.tool_call:
                    tool_name = response.tool_call.tool_name
                    parameters = self._prepare_tool_parameters(
                        tool_name, user_input, response.tool_call.tool_parameters
                    )

                    step_record += f"Action: Using tool '{tool_name}' with parameters: {json.dumps(parameters)}\n"

                    tool_result = self._execute_tool(tool_name, parameters)

                    formatted_result = self._format_tool_response(tool_name, tool_result)
                    step_record += f"Tool Result: {formatted_result}\n"
                    print(f" Tool Result:\n{formatted_result}")

                    self._add_to_memory(
                        {
                            "user_prompt": user_input,
                            "tool": tool_name,
                            "result": {"analysis": tool_result.get("analysis", "")},
                        }
                    )
                else:
                    step_record += "Action: No tool used in this step.\n"

                workspace["steps"].append(step_record)
                print(step_record)
                workspace["current_context"] += step_record
                print(f"\n{'─' * 50} [Step {iteration + 1} Complete] {'─' * 50}\n")

            except Exception as e:
                error_message = f"An error occurred during iteration {iteration + 1}: {str(e)}"
                print(f"ERROR: {error_message}")
                print(f"Error details: {type(e).__name__}")
                self._add_to_history("assistant", error_message)
                return error_message

        fallback_response = (
            "I've attempted to analyze your request but reached the maximum number of reasoning steps. "
            "Here's what I've gathered so far:\n\n"
            + workspace["current_context"]
            + "\n\nBased on the information collected, I can say that: "
            + "\n\nTo get more complete results, consider breaking your question into smaller, more specific parts."
        )

        self._add_to_memory(
            {
                "user_prompt": user_input,
                "final_answer": "Request exceeded maximum reasoning steps, incomplete analysis provided.",
            }
        )

        self._add_to_history("assistant", fallback_response)
        return fallback_response

    def run_conversation(self, initial_prompt: str, database_schema: str) -> List[Dict[str, str]]:
        response = self.process_user_input(initial_prompt, database_schema)
        print("User:", initial_prompt)
        print("Assistant:", response)
        print("\n" + "-" * 50 + "\n")

        return self.conversation_history


if __name__ == "__main__":
    agent = IterativeReActAgent()
    rag_instance = Rag("database_info.json")
    rag_instance.vectorize()
    vector_store = FAISS.load_local("vector_store", rag_instance.embedding_model, allow_dangerous_deserialization=True)

    while True:
        user_input = input("User: ")
        os.system("cls" if os.name == "nt" else "clear")  # Use clear for Unix-based systems
        print("User: ", user_input)

        if user_input.lower() == "exit":
            break
        database_schema = get_schema(user_input, rag_instance, vector_store)
        agent.run_conversation(user_input, database_schema)
