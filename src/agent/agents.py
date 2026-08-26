"""
agents.py
=========
Defines the sub-agents used inside the orchestrator pipeline.
"""

from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor

from ..tools_math.tools import MATH_TOOLS
from .agent_names import AgentName
from ..config.llm import (
    create_llm,
    create_routing_llm,
    create_synthesizer_llm,
    create_fallback_llm,
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
_fallback_llm = None

_agents: dict[str, object] = {}
pool_supervisor = None


# ================================================================
# AGENT FACTORY
# ================================================================

def _create_agent_with_fallback(
    *,
    name: str,
    system_prompt: str,
    tools: list,
    primary_llm,
    fallback_llm,
):
    """
    Create an agent using the primary LLM with a fallback LLM.

    If the primary model fails, LangChain will automatically retry
    the same agent execution using the fallback model.
    """

    primary_agent = create_agent(
        model=primary_llm,
        tools=tools,
        name=name,
        system_prompt=system_prompt,
    )

    fallback_agent = create_agent(
        model=fallback_llm,
        tools=tools,
        name=name,
        system_prompt=system_prompt,
    )

    return primary_agent.with_fallbacks([fallback_agent])


def _initialize():
    global (
        _initialized,
        _llm,
        _routing_llm,
        _synthesizer_llm,
        _fallback_llm,
        _agents,
        pool_supervisor,
    )

    if _initialized:
        return

    # ------------------------------------------------------------
    # LLMs
    # ------------------------------------------------------------

    _llm = create_llm()
    _routing_llm = create_routing_llm()
    _synthesizer_llm = create_synthesizer_llm()
    _fallback_llm = create_fallback_llm()

    # ------------------------------------------------------------
    # Common RAG tools
    # ------------------------------------------------------------

    RAG_TOOLS = [
        vector_search,
        search_seed_nodes,
        expand_subgraph,
    ]

    # ------------------------------------------------------------
    # GENERAL
    # ------------------------------------------------------------

    general_agent = _create_agent_with_fallback(
        name="general",
        tools=[pool_general_knowledge],
        system_prompt=GENERAL_PROMPT,
        primary_llm=_synthesizer_llm,
        fallback_llm=_fallback_llm,
    )

    # ------------------------------------------------------------
    # OUT OF SCOPE
    # ------------------------------------------------------------

    oos_agent = _create_agent_with_fallback(
        name="oos",
        tools=[],
        system_prompt=OOS_PROMPT,
        primary_llm=_synthesizer_llm,
        fallback_llm=_fallback_llm,
    )

    # ------------------------------------------------------------
    # CHEMISTRY
    # ------------------------------------------------------------

    chemistry_agent = _create_agent_with_fallback(
        name="chemistry",
        tools=RAG_TOOLS,
        system_prompt=build_agent_prompt(
            AGENT_REGISTRY[CHEMISTRY]
        ),
        primary_llm=_synthesizer_llm,
        fallback_llm=_fallback_llm,
    )

    # ------------------------------------------------------------
    # EQUIPMENT
    # ------------------------------------------------------------

    equipment_agent = _create_agent_with_fallback(
        name="equipment",
        tools=RAG_TOOLS,
        system_prompt=build_agent_prompt(
            AGENT_REGISTRY[EQUIPMENT]
        ),
        primary_llm=_synthesizer_llm,
        fallback_llm=_fallback_llm,
    )

    # ------------------------------------------------------------
    # HYDRAULICS
    # ------------------------------------------------------------

    hydraulics_agent = _create_agent_with_fallback(
        name="hydraulics",
        tools=RAG_TOOLS,
        system_prompt=build_agent_prompt(
            AGENT_REGISTRY[HYDRAULICS]
        ),
        primary_llm=_synthesizer_llm,
        fallback_llm=_fallback_llm,
    )

    # ------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------

    operations_agent = _create_agent_with_fallback(
        name="operations",
        tools=RAG_TOOLS,
        system_prompt=build_agent_prompt(
            AGENT_REGISTRY[OPERATIONS]
        ),
        primary_llm=_synthesizer_llm,
        fallback_llm=_fallback_llm,
    )

    # ------------------------------------------------------------
    # COMPLIANCE
    # ------------------------------------------------------------

    compliance_agent = _create_agent_with_fallback(
        name="compliance",
        tools=RAG_TOOLS,
        system_prompt=build_agent_prompt(
            AGENT_REGISTRY[COMPLIANCE]
        ),
        primary_llm=_synthesizer_llm,
        fallback_llm=_fallback_llm,
    )

    # ------------------------------------------------------------
    # CONTAMINATION
    # ------------------------------------------------------------

    contamination_agent = _create_agent_with_fallback(
        name="contamination",
        tools=RAG_TOOLS,
        system_prompt=build_agent_prompt(
            AGENT_REGISTRY[CONTAMINATION]
        ),
        primary_llm=_synthesizer_llm,
        fallback_llm=_fallback_llm,
    )

    # ------------------------------------------------------------
    # FACILITY DESIGN
    # ------------------------------------------------------------

    facility_design_agent = _create_agent_with_fallback(
        name="facility_design",
        tools=RAG_TOOLS,
        system_prompt=build_agent_prompt(
            AGENT_REGISTRY[FACILITY_DESIGN]
        ),
        primary_llm=_synthesizer_llm,
        fallback_llm=_fallback_llm,
    )

    # ------------------------------------------------------------
    # SAFETY
    # ------------------------------------------------------------

    safety_agent = _create_agent_with_fallback(
        name="safety",
        tools=RAG_TOOLS,
        system_prompt=build_agent_prompt(
            AGENT_REGISTRY[SAFETY]
        ),
        primary_llm=_synthesizer_llm,
        fallback_llm=_fallback_llm,
    )

    # ------------------------------------------------------------
    # RECOVERY
    # ------------------------------------------------------------

    recovery_agent = _create_agent_with_fallback(
        name="recovery",
        tools=RAG_TOOLS,
        system_prompt=build_agent_prompt(
            AGENT_REGISTRY[RECOVERY]
        ),
        primary_llm=_synthesizer_llm,
        fallback_llm=_fallback_llm,
    )

    # ------------------------------------------------------------
    # RECORDS
    # ------------------------------------------------------------

    records_agent = _create_agent_with_fallback(
        name="records",
        tools=RAG_TOOLS,
        system_prompt=build_agent_prompt(
            AGENT_REGISTRY[RECORDS]
        ),
        primary_llm=_synthesizer_llm,
        fallback_llm=_fallback_llm,
    )

    # ------------------------------------------------------------
    # MATH
    # ------------------------------------------------------------

    math_agent = _create_agent_with_fallback(
        name="math",
        tools=MATH_TOOLS,
        system_prompt=build_agent_prompt(
            AGENT_REGISTRY[MATH]
        ),
        primary_llm=_synthesizer_llm,
        fallback_llm=_fallback_llm,
    )

    # ------------------------------------------------------------
    # AGENT REGISTRY
    # ------------------------------------------------------------

    _agents = {
        "general": general_agent,
        "oos": oos_agent,
        "chemistry": chemistry_agent,
        "equipment": equipment_agent,
        "hydraulics": hydraulics_agent,
        "operations": operations_agent,
        "compliance": compliance_agent,
        "contamination": contamination_agent,
        "facility_design": facility_design_agent,
        "safety": safety_agent,
        "recovery": recovery_agent,
        "records": records_agent,
        "math": math_agent,
    }

    _initialized = True


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


# ================================================================
# SUPERVISOR
# ================================================================

def get_supervisor():
    """Return the compiled supervisor, constructing it lazily."""

    global pool_supervisor

    _initialize()

    if pool_supervisor is None:
        pool_supervisor = create_supervisor(
            agents=list(_agents.values()),
            model=_routing_llm,
            prompt=SUPERVISOR_PROMPT,
        ).compile()

    return pool_supervisor