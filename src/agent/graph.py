"""
graph.py
========
Compiles the PoolAgent LangGraph workflow.

Node topology
─────────────
                         ┌───────────────────┐
          START ───────► │ build_context_node │
                         └────────┬──────────┘
                                  │
                    tokens > 4000 │ tokens ≤ 4000
                                  │
               ┌──────────────────┘
               │                  │
               ▼                  │
  ┌──────────────────────┐        │
  │ summarize_memory_node│        │
  └──────────┬───────────┘        │
             │  goto="planner"    │
             └──────────┬─────────┘
                        │
                 ┌──────▼──────┐
                 │   planner   │
                 └──────┬──────┘
                        │ Command(goto="orchestrator")
                 ┌──────▼──────┐ ◄────────────────────┐
                 │ orchestrator│ Command(goto="orchestrator") (loop)
                 └──────┬──────┘
                        │ Command(goto="synthesizer")
                 ┌──────▼──────┐
                 │ synthesizer │
                 └──────┬──────┘
                        │
                       END

Routing notes
─────────────
- build_context_node returns a Command with goto, bypassing any memory_router.
- summarize_memory_node returns Command(goto="planner") after trimming messages.
- planner and orchestrator return Command objects → declared with destinations=[...].
- synthesizer returns a plain dict → simple edge to END.
- The orchestrator loops back to itself until all execution_plan steps are done,
  then routes to synthesizer.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from .state import PoolAgentState
from .nodes import (
    planner,
    orchestrator,
    synthesizer,
    build_context_node,
    summarize_memory_node,
)

# ================================================================
# BUILD GRAPH
# ================================================================

def build_graph(checkpointer=None):
    """
    Construct and compile the PoolAgent StateGraph.

    Args:
        checkpointer: Optional LangGraph checkpointer for persistence.
                      Defaults to InMemorySaver() for short-term memory.

    Returns:
        Compiled LangGraph application ready to invoke.
    """
    builder = StateGraph(PoolAgentState)

    # ── Register nodes ────────────────────────────────────────────────────────

    # Context + memory nodes use Command(goto=...) to route dynamically,
    # so their possible destinations must be declared at registration time.
    builder.add_node(
        "build_context_node",
        build_context_node,
        destinations=["summarize_memory_node", "planner"],
    )

    builder.add_node(
        "summarize_memory_node",
        summarize_memory_node,
        destinations=["planner"],
    )

    builder.add_node(
        "planner",
        planner,
        destinations=["orchestrator"],
    )

    builder.add_node(
        "orchestrator",
        orchestrator,
        destinations=["orchestrator", "synthesizer"],
    )

    builder.add_node("synthesizer", synthesizer)

    # ── Wire edges ────────────────────────────────────────────────────────────

    builder.add_edge(START, "build_context_node")   # new entry point
    builder.add_edge("synthesizer", END)             # terminal node

    # All other transitions (build_context_node → summarize_memory_node | planner,
    # summarize_memory_node → planner, planner → orchestrator,
    # orchestrator → orchestrator | synthesizer) are driven by the Command
    # objects returned inside each node — no explicit add_edge needed.

    # ── Compile ───────────────────────────────────────────────────────────────
    _checkpointer = checkpointer or InMemorySaver()

    app = builder.compile(checkpointer=_checkpointer)
    return app


# ================================================================
# DEFAULT EXPORT
# ================================================================

graph = build_graph()