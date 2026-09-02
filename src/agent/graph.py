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
                        │
               ┌────────┼────────┐
               │        │        │
               ▼        ▼        ▼
          ┌────────┐ ┌────────┐ ┌────────┐
          │general │ │  oos   │ │orchestr│
          └───┬────┘ └───┬────┘ └───┬────┘
              │          │          │
              └──────────┼──────────┘
                         │
                         │ Command(goto=[Send("run_step", …), …])
                         │ fan-out: uno o más steps "ready"
                         │ despachados en el mismo superstep
                  ┌──────▼──────┐
                  │  run_step   │ ── Command(goto="orchestrator") ──┘
                  └─────────────┘   (cada Send vuelve por su lado;
                                    agent_results se mergea vía
                                    merge_agent_results con centinela
                                    None, no se pisa)
                         │
                         │ orchestrator: cuando ya no quedan
                         │ steps pendientes, puede hacer:
                         │ - Fan-out a ["synthesizer", "suggester"]
                         │ - O ir solo a "synthesizer"
                         │
                  ┌──────▼──────┐
                  │ synthesizer │
                  └──────┬──────┘
                         │  edge simple
                  ┌──────▼──────┐
                  │  suggester  │
                  └──────┬──────┘
                         │
                        END

Routing notes
─────────────
- build_context_node returns a Command with goto, bypassing any memory_router.
  It also resets the per-turn channels (error, planner_error, archetype,
  misroute_retries, response, validation, suggestions): the checkpointer
  persists them across turns and they have no None sentinel of their own the
  way agent_results does via merge_agent_results.
- summarize_memory_node returns Command(goto="planner") after trimming messages.
- planner returns Command(goto="orchestrator") by default, but may route to
  "general" or "oos" if execution_plan has a single step assigned to those nodes.
- orchestrator computes which steps in execution_plan are "ready" (their
  depends_on are already present in agent_results) and dispatches them via
  Send("run_step", ...). Multiple Send calls in the same Command run in
  parallel in the same superstep.
- run_step executes exactly one ExecutionStep and always routes back to
  orchestrator with Command(goto=...).
- orchestrator re-evaluates on every return; once execution_plan has no
  pending steps left, it routes to synthesizer.
- agent_results uses a custom reducer (merge_agent_results) with centinela
  None so parallel writes from run_step merge instead of overwriting.
- general and oos are direct graph nodes, not agents in AGENT_REGISTRY.
  oos may also route BACK to orchestrator on a MISROUTE.
- synthesizer returns a plain dict → edge to suggester → END.

Suggester: sequential, not parallel
───────────────────────────────────
It was designed as a fan-out branch off the orchestrator, running beside the
synthesizer so it could never delay the answer. It isn't wired that way, and
the sequential edge is the correct choice for now: suggester reads
state["response"], which only exists after the synthesizer writes it. Two of
its gates (answer_ends_with_question, and the redundancy filter in
_unconsumed_entities) have no input at all under true fan-out.

The cost is that the suggester sits on the user's critical path. That is why
_SUGGESTER_DEADLINE_S is short and why every failure inside the node degrades
to [] rather than propagating.

Going back to real fan-out means rewriting both gates against agent_results
instead of response — a deliberate trade, not a cleanup.
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

    builder.add_edge(START, "build_context_node")
    builder.add_edge("synthesizer", "suggester")   # antes: END
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