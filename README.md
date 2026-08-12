# Analytics Copilot

An LLM-powered, agentic data-analytics copilot that turns natural-language
questions into SQL queries, Python visualizations, and plain-English analysis —
all running against a local SQLite database with strong guardrails, sandboxed
code execution, retrieval-augmented schema selection, and an **LLM-as-a-Judge**
evaluation harness.

Built around a planner / router architecture, the system supports both a
**normal (planner-routed) flow** and a fully **agentic ReAct flow**, so you can
pick the right trade-off between determinism and autonomous reasoning for your
workload.

---

## Table of Contents

- [Top Features](#top-features)
- [Architecture Overview](#architecture-overview)
- [Agentic vs Normal Flow](#agentic-vs-normal-flow)
- [Guardrails](#guardrails)
- [Sandboxing \& Safe Code Execution](#sandboxing--safe-code-execution)
- [Retrieval-Augmented Schema Selection (RAG)](#retrieval-augmented-schema-selection-rag)
- [Memory \& Conversation Context](#memory--conversation-context)
- [Error Handling \& Resilience](#error-handling--resilience)
- [LLM-as-a-Judge Test Harness](#llm-as-a-judge-test-harness)
- [Streaming Web UI (FastAPI + SSE)](#streaming-web-ui-fastapi--sse)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Configuration](#configuration)
- [Usage](#usage)
- [Extending the System](#extending-the-system)

---

## Top Features

1. **Multi-mode planner router** — a Gemini-powered planner classifies each
   request into `sql`, `python`, `general`, or `multi` (or refuses it) using a
   structured Pydantic response schema, so the right pipeline handles the right
   question.
2. **Agentic ReAct loop** — an iterative Reason → Act → Reflect agent
   (`react/main.py`) that picks tools autonomously, with a fallback override
   that forces a tool call if the model tries to "complete" without acting.
3. **Retrieval-augmented schema selection** — only the most schema-relevant
   tables/columns are pulled from a FAISS vector store (FastEmbed `bge-small-en`)
   and injected into the prompt, keeping token usage low and accuracy high.
4. **Sandboxed Python execution** — LLM-generated visualization code is parsed
   with `ast`, dedented, prefixed with a fixed import block, and executed in a
   controlled namespace (`CodeExecutor`) that only exposes `db_conn`,
   `final_df`, and `chart`. No arbitrary imports, no `plt.show()`/`savefig()`.
5. **Strict guardrails** — write operations (`INSERT`/`UPDATE`/`DELETE`) are
   blocked at the executor, the planner refuses out-of-scope, malicious,
   sensitive, file-IO, and ambiguous requests, and only Seaborn/Matplotlib +
   pandas/numpy are allowed for analysis.
6. **Self-healing SQL** — when a generated query fails with `no such column`,
   the system re-pulls the full DB schema and regenerates the query once.
7. **LLM-as-a-Judge evaluation harness** — `test/run_test.py` runs templated
   test cases with randomized parameters, executes both the gold and generated
   SQL, and asks the LLM to score **relevance** and **result similarity**
   (0–1), persisting results to CSV for offline analysis.
8. **Streaming web UI** — a FastAPI backend with `sse_starlette` streams
   `sql` → `results` → `analysis` → `timings` events to a Jinja2 chat
   frontend, so users see progress incrementally.
9. **Conversation memory** — recent interactions (user prompt, SQL/code,
   results, analysis) are kept in a rolling context window with
   latest-message priority, enabling follow-up questions.
10. **Resilient error handling** — `RateLimitError` is caught at every LLM
    call with graceful user-facing messages; `backoff` + exponential retry is
    used in the async test client; per-step timings are reported.

---

## Architecture Overview

```
                ┌──────────────────────────┐
   User Query ─▶│  Planner LLM (Gemini)    │  structured Pydantic response
                │  → mode | refusal         │
                └─────────────┬────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐          ┌──────────┐          ┌──────────┐
   │ SQL     │          │ Python   │          │ General  │
   │ flow    │          │ flow     │          │ flow     │
   └────┬────┘          └────┬─────┘          └────┬─────┘
        │                    │                     │
        ▼                    ▼                     ▼
   RAG schema          CodeExecutor          Direct LLM
   SQL gen             (sandboxed exec)      answer
   self-heal           viz + analysis
   analysis
        │
        ▼
   Memory context (rolling, latest-priority)
```

The **ReAct agent** (`react/main.py`) replaces the planner with an iterative
loop that calls the same underlying flows (`sql_flow`, `python_flow`,
`multi_flow`, `general_flow`) as tools, reasoning between steps and reflecting
on tool output before producing a final answer.

---

## Agentic vs Normal Flow

| Aspect | **Normal (Planner) Flow** — `main.py` | **Agentic (ReAct) Flow** — `react/main.py` |
|---|---|---|
| Routing | Single LLM call classifies mode → dispatches to one flow | Iterative Reason → Act → Reflect loop, multiple tool calls allowed |
| Tools | Fixed `sql_flow` / `python_flow` / `general_flow` / `multi_flow` | Same flows exposed as callable tools with JSON schemas |
| Iterations | 1 (with one self-heal retry for SQL) | Up to `max_iterations=5` |
| Memory | `AgentFlow.memory_context` (full SQL/results/analysis) | `IterativeReActAgent.memory_context` (simplified, capped at 10) |
| Determinism | High — one classification, one pipeline | Lower — agent decides tool order and may combine tools |
| Best for | Simple, well-scoped questions; lowest latency | Multi-step questions needing tool chaining or reflection |
| Fallback | Refusal message from planner | Override that forces a default tool if the agent tries to "complete" without acting; final fallback message after `max_iterations` |

Both flows share the same underlying pipelines, guardrails, RAG schema
retrieval, and sandboxed executor, so behavior is consistent regardless of
which entry point you use.

---

## Guardrails

Guardrails are enforced at **two layers**:

### 1. Planner / Router level (`llm/sql_prompt.py` `planner_prompt`)

The planner is instructed to refuse with a specific reason when the request:

- Involves **write operations** (`INSERT`/`UPDATE`/`DELETE`) → *"Write operations are not permitted"*
- Asks for **external/unrelated data** beyond the schema → *"This request is beyond the system's capabilities"*
- Shows **malicious intent** → *"Query violates security policies"*
- Involves **sensitive or explicit content** → *"Content restrictions prevent compliance"*
- References **tables/columns not in the schema** → *"Required data not available in tables"*
- Requests **file I/O** → *"File operations are prohibited"*
- Asks for **non-allowed libraries** → *"Only Seaborn/Matplotlib allowed for visualization and pandas/numpy for analysis."*
- Is **outside data analytics** → *"Strictly stay relevant to Data Analytics"*
- Has **ambiguous plot requirements** → *"Ambiguous visualization request"*

The same refusal policy is mirrored in the ReAct agent's system prompt.

### 2. Executor level (`engine/sql_executor.py`)

`execute_sql` performs a defensive pre-check on the query string and refuses
to run anything starting with `insert`, `update`, or `delete`, returning
`None` and printing `write operation not allowed`. This means even if a
prompt-injection attempt bypasses the planner, the executor still blocks it.

---

## Sandboxing & Safe Code Execution

The `analysis_visualization_agent/executor.py` `CodeExecutor` controls how
LLM-generated Python is run:

- **AST parsing** — generated code is parsed with `ast.parse`; `IndentationError`
  and `SyntaxError` are caught and reported instead of crashing.
- **Dedenting** — `textwrap.dedent` normalizes indentation before execution.
- **Fixed import block** — the executor prepends only the approved imports
  (`seaborn`, `matplotlib.pyplot`, `pandas`, `numpy`) and forbids the LLM from
  emitting its own imports or `plt.show()`/`plt.savefig()`/`plt.figure()`.
- **Controlled namespace** — `exec` runs with a restricted `locals` dict
  exposing only `db_conn`, `final_df`, and `chart`. The LLM is instructed to
  assign its result dataframe to `final_df` and the matplotlib figure to
  `chart`, which the executor then saves to
  `analysis_visualization_agent/outputs/output_<timestamp>.{png,csv}`.
- **Persisted code artifacts** — every executed script is also written to
  `outputs/code_<timestamp>.py` for audit/debugging.

> Note: execution uses `exec` in-process. For production hardening, consider
> moving `CodeExecutor` into a subprocess or container with resource limits.

---

## Retrieval-Augmented Schema Selection (RAG)

`rag/pipeline.py` (`Rag` class) builds a FAISS vector store from a
`database_info.json` file describing each table (description, sample
questions, columns). At query time:

1. `Rag.vectorize()` checks an MD5 hash of `database_info.json` against
   `vector_store/hash.txt` and **skips re-embedding** if unchanged.
2. `get_schema()` (in `engine/sql_executor.py`) lists tables/columns via
   `PRAGMA table_info`, then runs `vector_search(question, top_k=3)` to attach
   the most relevant table descriptions to the prompt.
3. Embeddings use `BAAI/bge-small-en-v1.5` via FastEmbed (CPU-friendly).

This keeps prompts small and focused, which materially improves SQL accuracy
on wide schemas.

---

## Memory & Conversation Context

Both flows maintain a `memory_context` list of past interactions
(`user_prompt`, `sql`/`code`, `results`, `analysis`). Prompts explicitly
instruct the LLM that **latest messages have higher priority**, enabling
natural follow-ups like *"and what about last month?"*. The ReAct agent
trims memory to the last 10 entries to bound context growth.

---

## Error Handling & Resilience

- **`openai.RateLimitError`** is caught at every LLM call site (SQL gen,
  analysis, visualization, explanation, general flow) with a user-facing
  message and early return — no uncaught crashes.
- **SQL execution errors** — `sqlite3.OperationalError` and
  `pd.errors.DatabaseError` are caught; on `no such column` the system
  re-pulls the full schema via `analyze_sqlite_db()` and regenerates the
  query once (self-healing).
- **Code execution errors** — `CodeExecutor.execute_code` wraps `exec` in a
  `try/except`, printing the error and traceback instead of propagating.
- **Async test client** (`test/test_utils.py`) uses `@backoff.on_exception`
  with exponential backoff on `RateLimitError`, and the test runner sleeps
  and retries on rate limits.
- **ReAct loop** — each iteration is wrapped in `try/except`; on failure the
  error is recorded into conversation history and returned to the user.
  A fallback message is produced if `max_iterations` is exceeded.
- **Malformed LLM JSON** — `utils.clean_sql_text` tries `json.loads` first,
  then falls back to `ast.literal_eval`, then returns `{}` so downstream
  code can detect missing keys gracefully.
- **Per-step timings** (`query_time`, `query_execute_time`, `analysis_time`)
  are reported for observability.

---

## LLM-as-a-Judge Test Harness

`test/run_test.py` is an end-to-end evaluation harness:

1. Loads templated test cases from `test/test_cases.json` (each with a
   parameterized `question` and gold `query`).
2. **Randomizes parameters** (`fetch_random_parameters`) by sampling real
   values from the DB (e.g., random `tconst`, `genre`, `year`, `person_id`)
   so each run tests different concrete inputs.
3. Sends the substituted question through the SQL generation pipeline.
4. Executes both the **gold SQL** and the **generated SQL** and truncates
   large result sets to 20 rows for fair comparison.
5. Calls the LLM with a comparison prompt (`test/test_utils.py:get_prompt`)
   that asks for two scores in `relevance,results_similarity` format
   (0–1 floats).
6. Parses the scores (with error fallback to `None`) and appends a row to
   `test/test_results_2.csv` with `User Question`, `Test SQL`, `Generated SQL`,
   `Relevance Score`, `Results Similarity Score`.
7. Logs everything to `test/logs/test_<timestamp>.log` via
   `test/logging_config.py`.

This gives a reproducible, automated quality signal for prompt or model
changes — the LLM acts as the judge comparing generated output against the
gold answer and the user intent.

---

## Streaming Web UI (FastAPI + SSE)

`backend/app.py` exposes:

- `GET /` — renders `templates/chat.html` (Jinja2) chat UI.
- `GET /stream?prompt=...` — `EventSourceResponse` streaming Server-Sent
  Events with typed payloads: `sql`, `results`, `analysis` (chunked),
  `refusal`, and `timings`. The frontend appends analysis chunks to a
  single growing bubble and renders other types as separate colored
  message blocks.

A thread → asyncio bridge (`run_llm_analysis`) wraps the synchronous
streaming generator so it can be consumed asynchronously by the SSE
response.

---

## Project Structure

```
analytics-copilot/
├── main.py                      # Normal (planner-routed) CLI entry point
├── config.json                  # DB name, model names, base_url
├── utils.py                     # load_params, db_conn, serialize_dataframe, clean_sql_text
├── requirements.txt             # Pinned deps for the sandboxed code env
├── pyproject.toml               # Project + dev deps (uv-managed)
│
├── llm/                         # LLM clients + prompts
│   ├── agent.py                 # llm_sql, llm_analysis, planner_llm, break_request
│   ├── flow.py                  # AgentFlow: sql/python/general/multi flows (normal)
│   ├── models.py                # PlannerResponse pydantic model
│   ├── sql_prompt.py            # SQL gen + break-request + planner prompts
│   └── analysis_prompt.py       # Result-analysis system + user prompts
│
├── react/                       # Agentic ReAct flow
│   ├── main.py                  # IterativeReActAgent (Reason → Act → Reflect)
│   └── tools.py                 # sql_flow / python_flow / general_flow / multi_flow as tools
│
├── engine/
│   └── sql_executor.py          # execute_sql (write-block), analyze_sqlite_db, get_schema (RAG)
│
├── rag/
│   └── pipeline.py              # FastEmbed + FAISS vector store, hash-based skip
│
├── analysis_visualization_agent/# Python/viz sub-agent
│   ├── executor.py              # CodeExecutor (sandboxed exec, AST parse, fixed imports)
│   ├── llm.py                   # llm_visualize (instructor JSON), llm_explain (stream)
│   ├── models.py                # VizResponse pydantic model (code/explanation/refusal)
│   ├── visualization_prompt.py  # Code-gen system prompt + user/explanation prompts
│   ├── utils.py                 # dataframe_to_markdown, local load_params
│   └── main.py                  # Standalone viz agent CLI
│
├── backend/
│   └── app.py                   # FastAPI + SSE streaming endpoint
├── templates/
│   └── chat.html                # Chat UI consuming /stream
│
├── data/
│   └── loader.py                # CSVToSQLite: bulk-load CSVs into the SQLite DB
│
└── test/
    ├── run_test.py              # LLM-as-a-Judge evaluation harness
    ├── test_utils.py            # async client w/ backoff, judge prompt
    └── logging_config.py        # File logging setup
```

---

## Setup Instructions

### Prerequisites

- Python **3.12.4+** (see `.python-version`)
- A Groq API key (for `llama-3.3-70b-versatile`) — set as `GROQ_API_KEY`
- A Google Gemini API key (for the planner) — set as `GEMINI_API_KEY`
- Your own CSV datasets placed in `data/` and a matching
  `database_info.json` describing each table (the repo does not ship data)

### Install

Using `uv` (recommended, matches `uv.lock`):

```bash
uv sync
```

Or with plain pip:

```bash
python -m venv .venv
. .venv/Scripts/activate    # Windows
# source .venv/bin/activate # macOS/Linux
pip install -r requirements.txt
pip install langchain langchain-community langchain-huggingface faiss-cpu \
            fastembed google-genai instructor pydantic-ai python-dotenv \
            fastapi uvicorn sse-starlette jinja2 backoff tiktoken tabulate
```

### Environment

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
```

### Prepare the database

1. Drop CSV files into `data/` (e.g. `customers.csv`, `orders.csv`, ...).
2. Create a `database_info.json` at the project root describing each table:

   ```json
   {
     "customers": {
       "description": "Customer master records",
       "sample_questions": ["How many customers per state?"],
       "columns": {
         "customer_id": "integer primary key",
         "state": "text"
       }
     }
   }
   ```

3. Load CSVs into SQLite:

   ```bash
   python data/loader.py
   ```

   This creates `bicycle.db` (name from `config.json`) with one table per CSV.

4. (Re)build the vector store (also auto-runs on first launch, but useful to
   do explicitly after changing `database_info.json`):

   ```bash
   python rag/pipeline.py
   ```

---

## Configuration

`config.json` (project root):

```json
{
  "params": {
    "db_name": "bicycle.db",
    "model_name": "llama-3.3-70b-versatile",
    "planner_model_name": "gemini-2.0-flash",
    "base_url": "https://api.groq.com/openai/v1"
  }
}
```

- `db_name` — SQLite file used by `utils.db_conn` and `data/loader.py`.
- `model_name` — main generation model (SQL, analysis, visualization, ReAct).
- `planner_model_name` — used by the planner (currently hardcoded to
  `gemini-2.0-flash` in `llm/agent.py`).
- `base_url` — OpenAI-compatible endpoint (Groq by default).

`analysis_visualization_agent/config.json` lists `allowed_modules`
(`numpy`, `pandas`, `seaborn`, `matplotlib.pyplot`) for the sandbox allowlist.

---

## Usage

### Normal (planner-routed) CLI

```bash
python main.py
```

Type questions at the `User:` prompt; type `exit` to quit. The planner
classifies each question and dispatches to the appropriate flow.

### Agentic ReAct CLI

```bash
python react/main.py
```

The ReAct agent reasons, picks a tool, observes the result, and iterates up
to 5 times before producing a final answer.

### Streaming web UI

```bash
python backend/app.py
```

Open <http://localhost:8000>, type a prompt, and watch `sql` → `results` →
`analysis` stream in via SSE.

### Standalone visualization agent

```bash
python analysis_visualization_agent/main.py
```

Bypasses the planner and goes straight to code generation + sandboxed
execution for a hardcoded prompt (edit `main.py` to customize).

### Run the LLM-as-a-Judge evaluation

1. Create `test/test_cases.json` with templated cases (parameterized
   `question` + gold `query`).
2. Run:

   ```bash
   python test/run_test.py
   ```

3. Inspect `test/test_results_2.csv` for per-case relevance and result
   similarity scores, and `test/logs/test_<timestamp>.log` for full traces.

---

## Extending the System

- **Add a new tool to the ReAct agent** — define the function in
  `react/tools.py`, add its JSON schema to the `tools` list in
  `react/main.py`, and add a branch in `_execute_tool` / `_format_tool_response`.
- **Swap models** — edit `config.json` (and the planner model in
  `llm/agent.py`). Any OpenAI-compatible endpoint works via `base_url`.
- **Tighten the sandbox** — uncomment the AST-based import filtering in
  `analysis_visualization_agent/executor.py` (uses
  `allowed_modules` from that subproject's `config.json`), or move `exec`
  into a subprocess/container.
- **Add guardrail categories** — extend the refusal conditions in
  `llm/sql_prompt.py` `planner_prompt` and the mirrored list in
  `react/main.py`'s system message.
- **Custom datasets** — replace `data/*.csv`, regenerate `bicycle.db` via
  `data/loader.py`, and update `database_info.json` + rebuild the vector
  store.
