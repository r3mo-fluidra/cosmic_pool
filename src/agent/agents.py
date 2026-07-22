"""
agents.py
=========
Defines the sub-agents used inside the orchestrator pipeline.
"""

from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor

from .state import AgentName
from ..config.llm import create_llm, create_routing_llm, create_synthesizer_llm
from ..agent.prompts import (
    GENERAL_PROMPT, OOS_PROMPT, SUPERVISOR_PROMPT,
    DIAGNOSIS_PROMPT, DOSAGE_PROMPT, EQUIPMENT_PROMPT, MAINTENANCE_PROMPT,
)
from .tools import (
    query_symptom_graph, search_troubleshooting_kb,
    query_chemical_actions, get_dosing_formulas,
    query_hardware_impact, search_equipment_manuals,
    search_maintenance_procedures, query_maintenance_dependencies,
    calculate_lsi, interpret_lsi, recommend_lsi_correction, analyze_pool_lsi,
)

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
# LAZY INITIALIZATION
# ================================================================

_initialized = False
_llm = None
_routing_llm = None
_synthesizer_llm = None
_agents: dict[str, object] = {}
pool_supervisor = None

def _initialize():
    global _initialized, _llm, _routing_llm, _synthesizer_llm, _agents, pool_supervisor

    if _initialized:
        return

    _llm            = create_llm()
    _routing_llm    = create_routing_llm()
    _synthesizer_llm = create_synthesizer_llm()

    general_agent = create_agent(
        model=_synthesizer_llm,
        tools=[pool_general_knowledge],
        name="general",
        system_prompt=GENERAL_PROMPT,
    )

    oos_agent = create_agent(
        model=_synthesizer_llm,
        tools=[],
        name="out_of_scope",
        system_prompt=OOS_PROMPT,
    )

    diagnosis_agent = create_agent(
        model=_llm,
        tools=[query_symptom_graph, search_troubleshooting_kb],
        name="diagnosis",
        system_prompt=DIAGNOSIS_PROMPT,
    )

    dosage_agent = create_agent(
        model=_routing_llm,
        tools=[query_chemical_actions, get_dosing_formulas],
        name="dosage",
        system_prompt=DOSAGE_PROMPT,
    )

    equipment_agent = create_agent(
        model=_llm,
        tools=[query_hardware_impact, search_equipment_manuals],
        name="equipment",
        system_prompt=EQUIPMENT_PROMPT,
    )

    maintenance_agent = create_agent(
        model=_llm,
        tools=[
            search_maintenance_procedures, query_maintenance_dependencies,
            calculate_lsi, interpret_lsi, recommend_lsi_correction, analyze_pool_lsi,
        ],
        name="maintenance",
        system_prompt=MAINTENANCE_PROMPT,
    )

    _agents = {
        "general":     general_agent,
        "ooo":         oos_agent,
        "diagnosis":   diagnosis_agent,
        "dosage":      dosage_agent,
        "equipment":   equipment_agent,
        "maintenance": maintenance_agent,
    }

    pool_supervisor = create_supervisor(
        agents=list(_agents.values()),
        model=_routing_llm,
        prompt=SUPERVISOR_PROMPT,
    ).compile()

    _initialized = True

# ================================================================
# AGENT REGISTRY
# ================================================================

def get_agent_by_name(agent_name: AgentName):
    """
    Return the compiled agent graph for *agent_name*.
    Raises ValueError if the agent is not yet registered.
    """
    _initialize()  # ✅ Solo se construye cuando se necesita por primera vez

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