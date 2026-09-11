# Pool Assistant (MVP)

An AI-powered chat assistant for pool water chemistry, equipment, and maintenance questions. It combines a Neo4j knowledge graph, a hybrid (dense + sparse) Qdrant vector store, and a team of Gemini-backed LangGraph agents behind a Streamlit chat UI.

## How it works

A user message flows through a compiled [LangGraph](https://github.com/langchain-ai/langgraph) graph ([src/agent/graph.py](src/agent/graph.py)):

```
START → build_context_node → [summarize_memory_node] → planner ─┬→ general ──→ synthesizer → END
                                                                ├→ oos ──────→ synthesizer → END
                                                                └→ orchestrator ⟲ run_step
                                                                         │
                                                                         └→ synthesizer ─→ END
                                                                         └→ suggester  ─→ END
```

- **build_context_node** estimates token usage, resets the per-turn channels, and routes to summarization when the conversation gets long.
- **summarize_memory_node** compresses older messages into a rolling summary that then rides along into the planner and every specialist.
- **planner** classifies the request, detects the language, and produces an `execution_plan` of up to 5 steps, each assigned to a sub-agent. A single-step plan for `general` or `oos` short-circuits straight to that node.
- **orchestrator** works out which steps are *ready* (their `depends_on` already succeeded) and dispatches them with `Send`, so **independent steps run in parallel in one superstep**. It re-evaluates on every return, cascades skips when a dependency fails, and trips a circuit breaker on provider errors.
- **run_step** executes exactly one step and merges its result through `merge_agent_results`, a reducer with a `None` sentinel so parallel writes don't overwrite each other.
- **synthesizer** and **suggester** fan out together: the first turns the raw sub-agent output into the user-facing reply, the second proposes follow-up chips. Both branches reach `END`, so the turn closes when the slower one finishes.

### Sub-agents ([src/agent/agents.py](src/agent/agents.py))

Eleven specialists share the same three retrieval tools (`vector_search`,
`search_seed_nodes`, `expand_subgraph`), each with its own system prompt,
archetype and tool budget. The full roster is `AgentName` in
[src/agent/agent_names.py](src/agent/agent_names.py) — the single source of
truth the planner is constrained to.

| Agent | Responsibility |
|---|---|
| `chemistry` | Water chemistry of a specific pool: symptoms, test results, corrective action |
| `equipment` | Faulty, worn or fouled components; service procedures; parts |
| `hydraulics` | Flow rate, turnover, head loss, pump operating point |
| `operations` | Schedules, preventive maintenance programmes, routines |
| `compliance` | What US/Canadian codes require, permit or inspect |
| `contamination` | Fecal/vomit/blood incidents, recreational water illness |
| `facility_design` | New builds and renovations (the system does not exist yet) |
| `safety` | Prevention, supervision, PPE, emergency preparedness |
| `recovery` | Flood, storm, sewage backup, prolonged abandonment |
| `records` | Log structure, retention, inspection packages |
| `math` | Pure numeric computation, over a deterministic YAML catalogue ([src/tools_math/](src/tools_math/)) — the only agent with a calculator |
| `general` | Greetings, capability questions, and clarification requests when numeric inputs are missing |
| `oos` | Politely declines out-of-scope requests |

A `langgraph_supervisor`-based supervisor is also built, but the main
pipeline does not use it: `run_step` resolves agents by name through
`get_agent_by_name`.

### Data layer

- **Neo4j** ([src/agent/tools.py](src/agent/tools.py)) stores the symptom → parameter → chemical → equipment relationship graph, queried via Cypher.
- **Qdrant** ([src/qdrant_vector_store.py](src/qdrant_vector_store.py)) stores unstructured manual/troubleshooting content as a hybrid index: dense vectors from **FastEmbed** (`BAAI/bge-small-en-v1.5`, computed locally — not Gemini) plus sparse `Qdrant/bm25`. Set `QDRANT_URL` to use a server; without it the client falls back to an embedded index under `src/data/qdrant_pool_db`, which takes an exclusive file lock and does not survive more than one Streamlit worker.
- **Langfuse** provides tracing and user feedback scoring for the Streamlit UI (optional — the app degrades gracefully if credentials are missing).

## Prerequisites

- Python 3.11 or 3.12
- A Neo4j instance (e.g. [Neo4j Aura](https://console.neo4j.io/) free tier) pre-loaded with the pool chemistry graph
- A Google Gemini API key
- (Optional) A Langfuse project for tracing/feedback

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
```

Poetry also works — `poetry install` resolves the project's `.venv`, and
`poetry run pytest` / `poetry run streamlit run app.py` behave the same.

> **Two manifests disagree, and only one is deployed.** Streamlit Cloud and
> [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json) both
> install `requirements.txt` (`pip3 install --user -r requirements.txt`) and
> never read `pyproject.toml`. The two also specify different floors —
> `langchain >=0.3` vs `>=1.3` — so a local Poetry environment and a
> deployed one can resolve to different versions of the same library.
> `requirements.txt` is what ships; keep it authoritative, or generate it
> from `pyproject.toml`, but do not maintain both by hand.
>
> `requirements-dev.txt` adds `pytest` on top and is never installed by a
> deployment.

Copy the environment template and fill in your own credentials — never commit `.env`:

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Google Gemini LLM + embeddings |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` | Knowledge graph connection |
| `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_BASE_URL` | Optional observability/feedback tracking |

### Vector store

The Qdrant hybrid index is built from a CSV of manual/troubleshooting chunks via `inicializar_vector_store()` in [src/qdrant_vector_store.py](src/qdrant_vector_store.py). Point it at your chunks CSV and run once to populate `src/data/qdrant_pool_db`:

```bash
.venv/bin/python -c "from src.qdrant_vector_store import inicializar_vector_store; inicializar_vector_store('path/to/pool_manual_chunks.csv')"
```

Subsequent runs load the existing local index via `cargar_vector_store()`.

> **Note:** Neo4j Aura's free tier auto-pauses after inactivity. The app detects this and prompts you to resume the instance from the Aura console — it cannot be unpaused programmatically without the Aura API.

## Running the app

```bash
.venv/bin/streamlit run app.py
```

This opens the chat UI, where the sidebar shows knowledge-graph connectivity, session info, and a conversation reset button. Each response can be rated 👍/👎, which is logged to Langfuse when configured.

## Running tests

```bash
.venv/bin/python -m pytest
```

Tests live under [test/](test/) and mirror the `src/` layout (`test/agent`,
`test/config`, `test/ui`). The whole suite runs offline: the `langfuse` module
is stubbed in [test/conftest.py](test/conftest.py) and every LLM is mocked, so
no API key, database or network access is needed. Verified by running it with
`GEMINI_API_KEY`, `NEO4J_URI`, `NEO4J_PASSWORD`, `QDRANT_URL` and
`LANGFUSE_SECRET_KEY` unset.

## Evaluation

An LLM-as-judge evaluation harness using [Giskard](https://github.com/Giskard-AI/giskard) is available at [eval/gizkard/eval_giskard.py](eval/gizkard/eval_giskard.py) for scanning the compiled graph for hallucination, robustness, and other quality issues.

## Project structure

```
app.py                       # Streamlit chat UI + turn loop (streams the synthesizer)
src/
  agent/
    graph.py                 # LangGraph topology and compilation
    nodes.py                 # Context, planner, orchestrator, run_step, synthesizer, suggester
    agents.py                # Sub-agent definitions and lazy initialization
    agent_names.py           # AgentName + slugs — single source of truth, no deps
    chains.py                # Planner structured-output chain
    gates.py                 # Deterministic pre-flight gates (cheaper than a ReAct loop)
    middleware.py            # Drops budget-exhausted tools before the model asks for them
    state.py                 # Shared graph state, reducers and pydantic models
    tools.py                 # The three retrieval tools over Neo4j + Qdrant
  graph_context/
    response_contracts.py    # Archetypes and the synthesizer's output contract
    response_validator.py    # Deterministic enforcement of that contract
    suggestions.py           # Suggestion model + the gates that filter chips
    turn_cache.py            # Nodes touched by retrieval during one turn
  tools_math/                # Deterministic calculator over pool_math_catalog_v2.yaml
  prompts/                   # System prompts, per agent and per archetype
  ui/
    theme.py                 # Styling and the phone frame
    streaming.py             # Pulls `answer` out of the synthesizer's partial JSON
    turns.py, copy.py, messages.py
  config/llm.py              # Gemini factories, one per role, with time budgets
  qdrant_vector_store.py     # Hybrid vector store creation/loading
  data/neo4j_indexes.cypher  # Index script — see the notes inside before running
test/                        # Pytest suite mirroring src/ — offline, no credentials
eval/gizkard/                # Giskard evaluation harness
```
