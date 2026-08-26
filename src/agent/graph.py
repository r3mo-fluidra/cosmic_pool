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
                    tokens > 25000 │ tokens ≤ 25000
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
                 ┌──────▼──────┐ ◄─────────────────────────┐
                 │ orchestrator│                            │
                 └──────┬──────┘                            │
                        │ Command(goto=[Send("run_step", …), …])
                        │ fan-out: uno o más steps "ready"   │
                        │ despachados en el mismo superstep  │
                 ┌──────▼──────┐                            │
                 │  run_step   │ ── Command(goto="orchestrator") ──┘
                 └─────────────┘   (cada Send vuelve por su lado;
                                    agent_results se mergea vía
                                    operator.or_, no se pisa)
                        │
                        │ orchestrator: cuando ya no quedan
                        │ steps pendientes → Command(goto="synthesizer")
                 ┌──────▼──────┐
                 │ synthesizer │
                 └──────┬──────┘
                        │
                       END

Routing notes
─────────────
- build_context_node returns a Command with goto, bypassing any memory_router.
- summarize_memory_node returns Command(goto="planner") after trimming messages.
- planner returns Command(goto="orchestrator").
- orchestrator no longer executes agents itself: it computes which steps in
  execution_plan are "ready" (their depends_on are already present in
  agent_results) and dispatches them via Send("run_step", ...). Multiple
  Send calls in the same Command run in parallel in the same superstep.
- run_step executes exactly one ExecutionStep (whatever LangGraph handed it
  via Send) and always routes back to orchestrator with Command(goto=...).
- orchestrator re-evaluates on every return; once execution_plan has no
  pending steps left, it routes to synthesizer instead of dispatching.
- agent_results uses an operator.or_ reducer in the state (see state.py) so
  concurrent writes from parallel run_step calls merge instead of
  overwriting each other.
- synthesizer returns a plain dict → simple edge to END.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from .state import PoolAgentState
from .nodes import (
    planner,
    orchestrator,
    run_step_node,
    synthesizer,
    build_context_node,
    summarize_memory_node,
    suggester,
    general,
    oos,
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
        destinations=["orchestrator", "general", "oos"],
    )

    builder.add_node(
            "general",
            general,
            destinations=["synthesizer"],
        )

    builder.add_node(
                "oos",
                oos,
                destinations=["synthesizer"],
            )

    builder.add_node(
        "orchestrator",
        orchestrator,
        # "run_step" es el destino de fan-out (uno o más Send por invocación);
        # "synthesizer" cuando ya no quedan steps pendientes.
        destinations=["run_step", "synthesizer"],
    )

    builder.add_node(
        "run_step",
        run_step_node,
        destinations=["orchestrator"],
    )

    builder.add_node("synthesizer", synthesizer)
    builder.add_node("suggester", suggester)

    # ── Wire edges ────────────────────────────────────────────────────────────

    builder.add_edge(START, "build_context_node")   # new entry point
    builder.add_edge("synthesizer", END)             # terminal node
    builder.add_edge("suggester", END)
    # All other transitions (build_context_node → summarize_memory_node | planner,
    # summarize_memory_node → planner, planner → orchestrator,
    # orchestrator → run_step (fan-out) | synthesizer,
    # run_step → orchestrator) are driven by the Command objects returned
    # inside each node — no explicit add_edge needed.

    # ── Compile ───────────────────────────────────────────────────────────────
    _checkpointer = checkpointer or InMemorySaver()

    app = builder.compile(checkpointer=_checkpointer)
    return app


# ================================================================
# DEFAULT EXPORT
# ================================================================

graph = build_graph()