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
from ..prompts.prompts import (
    GENERAL_PROMPT, 
    OOS_PROMPT, 
    SUPERVISOR_PROMPT,
    build_agent_prompt
)
from ..prompts.prompts_sub_agents import (
    AGENT_REGISTRY,
    tool_instructions_AA,
    )   
from .tools import (
    vector_search,
    search_seed_nodes,
    expand_subgraph,
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

    chemistry_agent = create_agent(
        model=_synthesizer_llm,
        tools=[vector_search, search_seed_nodes, expand_subgraph],
        name="chemistry",
        system_prompt=build_agent_prompt(AGENT_REGISTRY[CHEMISTRY]),
    )

    equipment_agent = create_agent(
        model=_synthesizer_llm,
        tools=[vector_search, search_seed_nodes, expand_subgraph],  
        name="equipment",
        system_prompt=build_agent_prompt(AGENT_REGISTRY[EQUIPMENT]),
    )

    hydraulics_agent = create_agent(
        model=_synthesizer_llm,
        tools=[vector_search, search_seed_nodes, expand_subgraph],
        name="hydraulics",
        system_prompt=build_agent_prompt(AGENT_REGISTRY[HYDRAULICS]),
    )

    operations_agent = create_agent(
        model=_synthesizer_llm,
        tools=[vector_search, search_seed_nodes, expand_subgraph],
        name="operations",
        system_prompt=build_agent_prompt(AGENT_REGISTRY[OPERATIONS]),
    )

    compliance_agent = create_agent(
        model=_synthesizer_llm,
        tools=[vector_search, search_seed_nodes, expand_subgraph],
        name="compliance",
        system_prompt=build_agent_prompt(AGENT_REGISTRY[COMPLIANCE]),
    )

    contamination_agent = create_agent(
        model=_synthesizer_llm,
        tools=[vector_search, search_seed_nodes, expand_subgraph],
        name="contamination",
        system_prompt=build_agent_prompt(AGENT_REGISTRY[CONTAMINATION]),
    )

    facility_design_agent = create_agent(
        model=_synthesizer_llm,
        tools=[vector_search, search_seed_nodes, expand_subgraph],
        name="facility_design",
        system_prompt=build_agent_prompt(AGENT_REGISTRY[FACILITY_DESIGN]),
    )

    safety_agent = create_agent(
        model=_synthesizer_llm,
        tools=[vector_search, search_seed_nodes, expand_subgraph],
        name="safety",
        system_prompt=build_agent_prompt(AGENT_REGISTRY[SAFETY]),
    )

    recovery_agent = create_agent(
        model=_synthesizer_llm,
        tools=[vector_search, search_seed_nodes, expand_subgraph],
        name="recovery",
        system_prompt=build_agent_prompt(AGENT_REGISTRY[RECOVERY]),
    )

    records_agent = create_agent(
        model=_synthesizer_llm,
        tools=[vector_search, search_seed_nodes, expand_subgraph],  
        name="records",
        system_prompt=build_agent_prompt(AGENT_REGISTRY[RECORDS]),
    )

    math_agent = create_agent(
        model=_synthesizer_llm,
        tools=[vector_search, search_seed_nodes, expand_subgraph],
        name="math",
        system_prompt=build_agent_prompt(AGENT_REGISTRY[MATH]),
    )
    _agents = {
        "general":     general_agent,
        "ooo":         oos_agent,
        "chemistry":   chemistry_agent,
        "equipment":   equipment_agent,
        "hydraulics":  hydraulics_agent,
        "operations":  operations_agent,
        "compliance":  compliance_agent,
        "contamination": contamination_agent,
        "facility_design": facility_design_agent,
        "safety": safety_agent,
        "recovery": recovery_agent,
        "records": records_agent,
        "math": math_agent,
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