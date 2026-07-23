# Pool Assistant (MVP)

An AI-powered chat assistant for pool water chemistry, equipment, and maintenance questions. It combines a Neo4j knowledge graph, a hybrid (dense + sparse) Qdrant vector store, and a team of Gemini-backed LangGraph agents behind a Streamlit chat UI.

## How it works

A user message flows through a compiled [LangGraph](https://github.com/langchain-ai/langgraph) graph ([src/agent/graph.py](src/agent/graph.py)):

```
START → build_context_node → [summarize_memory_node] → planner → orchestrator ⟲ → synthesizer → END
```

- **build_context_node** estimates token usage and routes to summarization when the conversation gets long.
- **summarize_memory_node** compresses older messages into a rolling summary to keep the context window small.
- **planner** classifies the request, detects the language, and produces a step-by-step `execution_plan`, assigning each step to a sub-agent.
- **orchestrator** executes the plan one step at a time, looping back on itself until every step has run.
- **synthesizer** merges the sub-agents' raw output into a single, language-matched, user-facing reply.

### Sub-agents ([src/agent/agents.py](src/agent/agents.py))

| Agent | Responsibility | Tools |
|---|---|---|
| `diagnosis` | Maps symptoms to causal chemistry parameters | Neo4j symptom graph, troubleshooting KB search |
| `dosage` | Calculates chemical dosing to raise/lower a parameter | Neo4j chemical actions, dosing formula search |
| `equipment` | Evaluates corrosion/scaling/degradation effects on hardware | Neo4j hardware impact, equipment manual search |
| `maintenance` | Routine/seasonal maintenance and LSI water-balance analysis | Maintenance procedure search, LSI calculator/interpreter/recommender |
| `general` | Free-form pool knowledge fallback | LLM knowledge only |
| `ooo` | Politely declines out-of-scope requests | none |

A `langgraph_supervisor`-based supervisor is also available for ad hoc agent routing outside the main graph.

### Data layer

- **Neo4j** ([src/agent/tools.py](src/agent/tools.py)) stores the symptom → parameter → chemical → equipment relationship graph, queried via Cypher.
- **Qdrant** ([src/qdrant_vector_store.py](src/qdrant_vector_store.py)) stores unstructured manual/troubleshooting content as a hybrid dense (Gemini embeddings) + sparse (BM25) index, persisted locally under `src/data/qdrant_pool_db`.
- **Langfuse** provides tracing and user feedback scoring for the Streamlit UI (optional — the app degrades gracefully if credentials are missing).

## Prerequisites

- Python 3.11 or 3.12
- [Poetry](https://python-poetry.org/) for dependency management
- A Neo4j instance (e.g. [Neo4j Aura](https://console.neo4j.io/) free tier) pre-loaded with the pool chemistry graph
- A Google Gemini API key
- (Optional) A Langfuse project for tracing/feedback

## Setup

```bash
poetry install
```

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
poetry run python -c "from src.qdrant_vector_store import inicializar_vector_store; inicializar_vector_store('path/to/pool_manual_chunks.csv')"
```

Subsequent runs load the existing local index via `cargar_vector_store()`.

> **Note:** Neo4j Aura's free tier auto-pauses after inactivity. The app detects this and prompts you to resume the instance from the Aura console — it cannot be unpaused programmatically without the Aura API.

## Running the app

```bash
poetry run streamlit run app.py
```

This opens the chat UI, where the sidebar shows knowledge-graph connectivity, session info, and a conversation reset button. Each response can be rated 👍/👎, which is logged to Langfuse when configured.

## Running tests

```bash
poetry run pytest
```

Tests live under [test/](test/) and mirror the `src/` layout (`test/agent`, `test/config`). Langfuse's `@observe` decorator is stubbed out in [test/conftest.py](test/conftest.py) so tests don't require real tracing credentials.

## Evaluation

An LLM-as-judge evaluation harness using [Giskard](https://github.com/Giskard-AI/giskard) is available at [eval/gizkard/eval_giskard.py](eval/gizkard/eval_giskard.py) for scanning the compiled graph for hallucination, robustness, and other quality issues.

## Project structure

```
app.py                      # Streamlit chat UI
src/
  agent/
    graph.py                # LangGraph topology and compilation
    nodes.py                 # Context, planner, orchestrator, synthesizer nodes
    agents.py                # Sub-agent definitions and lazy initialization
    chains.py                # Planner structured-output chain
    prompts.py                # System prompts per agent
    state.py                 # Shared graph state and pydantic models
    tools.py                 # Neo4j/Qdrant tools + deterministic LSI chemistry engine
  config/
    llm.py                   # Gemini LLM factory functions
  qdrant_vector_store.py     # Hybrid vector store creation/loading
test/                        # Pytest suite mirroring src/
eval/gizkard/                # Giskard evaluation harness
```
