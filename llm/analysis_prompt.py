system_prompt = """
 **Role**: Data Analysis Interpreter
**Instructions**:
1. Analyze ONLY the provided SQL query results
2. Never reference external knowledge or training data
3. Format response clearly for non-technical users
4. State limitations of the data explicitly
5. Never speculate beyond the provided data
6. If results are empty/insufficient, explicitly state this
7. Maintain complete neutrality in interpretations

**Response Format**:
- Plain text (NO markdown)
- 1-3 concise paragraphs
- Structured as:
  Summary: [Main conclusion]
  Key Findings: [Bullet points]
  Limitations: [Data constraints]
"""

def get_user_prompt(user_prompt: str, sql_query: str, result: str, memory_context: list) -> str:
    memory_str = ""
    if memory_context:
        memory_str += "**Previous Interactions:**\n"
        for idx, entry in enumerate(memory_context, start=1):
            memory_str += (
                f"{idx}. User Question: {entry.get('user_prompt', 'N/A')}\n"
                f"   Executed SQL Query: {entry.get('sql', 'N/A')}\n"
                f"   Query Results: {entry.get('results', 'N/A')}\n"
                f"   Analysis: {entry.get('analysis', 'N/A')}\n\n"
            )
    else:
        memory_str = "None"

    prompt = f"""
**Analysis Requirements**:
1. Create a factual summary BASED ONLY ON THE RESULTS PROVIDED AT THE END.
2. Highlight significant patterns/numbers FROM THE DATA.
3. If results are empty/insufficient, explain this clearly.
4. Never add assumptions or external context.
5. State if query doesn't fully answer the original question.

**Response Rules**:
- Use simple business language.
- Numbers must match exactly with results.
- Comparisons require explicit data support.
- Cite specific figures from results.
- Avoid technical SQL terminology.
- Do not explain actual SQL code; only discuss the results.


**Memory Context**:
latest messages in the list need to be assigned more priority.

{memory_str}

**User Question**:
{user_prompt}
**Executed SQL Query**:
{sql_query}
**Query Results** (Markdown format):
{result}
"""
    return prompt
