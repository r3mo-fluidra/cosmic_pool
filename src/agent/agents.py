"""
agents.py
=========
Defines the sub-agents used inside the orchestrator pipeline.
"""

import logging

from langchain_core.tools import tool
from langchain_core.runnables import RunnableWithFallbacks
from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor

from ..tools_math.tools import MATH_TOOLS
from .agent_names import AgentName
from ..config.llm import (
    create_routing_llm,
    create_synthesizer_llm,
    create_fallback_llm,
    create_specialist_llm,
    create_direct_answer_llm,
)
from ..prompts.prompts import (
    GENERAL_PROMPT,
    OOS_PROMPT,
    SUPERVISOR_PROMPT,
)
from ..prompts.prompt_archetype import build_agent_prompt
from ..prompts.prompts_sub_agents import (
    AGENT_REGISTRY,
    CHEMISTRY,
    EQUIPMENT,
    HYDRAULICS,
    OPERATIONS,
    COMPLIANCE,
    CONTAMINATION,
    FACILITY_DESIGN,
    SAFETY,
    RECOVERY,
    RECORDS,
    MATH,
)
from .tools import (
    vector_search,
    search_seed_nodes,
    expand_subgraph,
)
from .middleware import ToolBudgetMiddleware


logger = logging.getLogger(__name__)


# ================================================================
# TOOLS
# ================================================================

@tool
def pool_general_knowledge(topic: str) -> str:
    """
    Retrieve general knowledge and best practices for a pool-related topic.

    Args:
        topic: The specific pool subject to look up
               (e.g. 'saltwater pools', 'water balance', 'pool covers').
    """
    return (
        f"General pool information about '{topic}': "
        "Please provide a comprehensive, helpful explanation based on your training knowledge."
    )


# ================================================================
# SPECIALIST DECLARATION
# ================================================================

RETRIEVAL_TOOLS = [vector_search, search_seed_nodes, expand_subgraph]

# (node_name, AGENT_REGISTRY key). El node_name es el que usa el planner
# en `assigned_agent`; no se deriva de la constante para evitar drift silencioso.
SPECIALIST_SPECS: tuple[tuple[str, str], ...] = (
    ("chemistry",       CHEMISTRY),
    ("equipment",       EQUIPMENT),
    ("hydraulics",      HYDRAULICS),
    ("operations",      OPERATIONS),
    ("compliance",      COMPLIANCE),
    ("contamination",   CONTAMINATION),
    ("facility_design", FACILITY_DESIGN),
    ("safety",          SAFETY),
    ("recovery",        RECOVERY),
    ("records",         RECORDS),
)


# ================================================================
# LAZY INITIALIZATION
# ================================================================

_initialized = False
_routing_llm = None
_synthesizer_llm = None
_fallback_llm = None
_specialist_llm = None
_direct_answer_llm = None
_agents: dict[str, object] = {}
_supervisor_agents: list[object] = []
pool_supervisor = None


def _initialize():
    global _initialized, _routing_llm, _synthesizer_llm, _fallback_llm, _specialist_llm
    global _direct_answer_llm
    global _agents, _supervisor_agents, pool_supervisor

    if _initialized:
        return

    _routing_llm       = create_routing_llm()
    _synthesizer_llm   = create_synthesizer_llm()
    _fallback_llm      = create_fallback_llm()
    _specialist_llm    = create_specialist_llm()
    _direct_answer_llm = create_direct_answer_llm()

    # ---- general: primario + fallback ---------------------------------
    # El fallback tiene que ser OTRO AGENTE, no un LLM suelto: el primario
    # recibe y devuelve estado de grafo ({"messages": [...]}), un chat model
    # no acepta esa firma.
    #
    # _direct_answer_llm (thinking_budget=0) y no _synthesizer_llm.
    #
    # Hay DOS `general` en el sistema y es fácil arreglar solo uno: el NODO
    # de nodes.py, al que el planner rutea directo cuando el plan tiene un
    # único step, y este AGENTE, que corre por run_step cuando `general` es
    # un paso más de un plan mayor. Medido en el trace 6bb32a41: este
    # segundo camino gastó 471 tokens de razonamiento y 4.0s para preguntar
    # al usuario qué ácido tenía disponible — la tarea entera venía escrita
    # en el `task` del planner.
    primary_general = create_agent(
        model=_direct_answer_llm,
        tools=[pool_general_knowledge],
        name="general",
        system_prompt=GENERAL_PROMPT,
    )

    fallback_general = create_agent(
        model=_fallback_llm,
        tools=[pool_general_knowledge],
        name="general_fallback",
        system_prompt=GENERAL_PROMPT,
    )

    # RunnableWithFallbacks NO hereda el .name del runnable que envuelve.
    # Sin `name=` explícito, create_supervisor aborta con:
    #   "Please specify a name when you create your agent ..."
    general_agent = RunnableWithFallbacks(
        runnable=primary_general,
        fallbacks=[fallback_general],
        name="general",
    )

    # ---- oos ----------------------------------------------------------
    # Mismo modelo sin thinking: declinar una petición fuera de alcance es
    # la tarea menos deliberativa del sistema. El prompt ya dice qué decir.
    oos_agent = create_agent(
        model=_direct_answer_llm,
        tools=[],
        name="oos",
        system_prompt=OOS_PROMPT,
    )

    # ---- specialists (retrieval) --------------------------------------
    specialists = {
        node_name: create_agent(
            model=_specialist_llm,
            tools=RETRIEVAL_TOOLS,
            name=node_name,
            system_prompt=build_agent_prompt(AGENT_REGISTRY[registry_key], node_name),
            middleware=[ToolBudgetMiddleware(),],
        )
        for node_name, registry_key in SPECIALIST_SPECS
    }

    # ---- math (catálogo determinista) ---------------------------------
    math_agent = create_agent(
        model=_synthesizer_llm,
        tools=MATH_TOOLS,
        name="math",
        system_prompt=build_agent_prompt(AGENT_REGISTRY[MATH], "math"),
        middleware=[ToolBudgetMiddleware()],
    )

    _agents = {
        "general": general_agent,
        "oos": oos_agent,
        **specialists,
        "math": math_agent,
    }

    # El supervisor recibe el agente primario "pelado", no el wrapper de
    # fallbacks: create_supervisor inspecciona el grafo de cada nodo.
    _supervisor_agents = [primary_general, oos_agent, *specialists.values(), math_agent]

    # Los specialists ya están utilizables aunque el supervisor falle.
    _initialized = True

    _build_supervisor()


def _build_supervisor():
    """
    Construye el supervisor de forma aislada. Un fallo aquí no debe tumbar
    el pipeline principal (planner -> orchestrator -> run_step), que resuelve
    los agentes por `get_agent_by_name` y no usa el supervisor.
    """
    global pool_supervisor

    try:
        pool_supervisor = create_supervisor(
            agents=_supervisor_agents,
            model=_routing_llm,
            prompt=SUPERVISOR_PROMPT,
        ).compile()
    except Exception:
        pool_supervisor = None
        logger.exception("create_supervisor failed; supervisor path disabled")


# ================================================================
# AGENT REGISTRY
# ================================================================

def get_agent_by_name(agent_name: AgentName):
    """
    Return the compiled agent graph for *agent_name*.
    Raises ValueError if the agent is not yet registered.
    """
    _initialize()

    agent = _agents.get(agent_name)
    if agent is None:
        raise ValueError(
            f"Agent '{agent_name}' is not registered. "
            f"Available agents: {list(_agents.keys())}"
        )
    return agent


def get_supervisor():
    """Devuelve el supervisor compilado, inicializando si es necesario."""
    _initialize()
    return pool_supervisor