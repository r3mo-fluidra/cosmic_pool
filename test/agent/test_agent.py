"""
Tests de src.agent.agents.

El roster de la v1 (diagnosis / dosage / maintenance) ya no existe: hoy son
los 10 especialistas de SPECIALIST_SPECS más `general`, `oos` y `math`. Los
tests se apoyan en esas constantes en vez de repetir la lista a mano, para
que agregar un especialista no deje el test mintiendo en verde.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import RunnableLambda

from src.agent import agents
from src.agent.agent_names import AGENT_NAME_SET


@pytest.fixture(autouse=True)
def reset_agents():
    """La inicialización es perezosa y global: se limpia antes y después."""
    def _reset():
        agents._initialized = False
        agents._routing_llm = None
        agents._synthesizer_llm = None
        agents._fallback_llm = None
        agents._specialist_llm = None
        agents._agents = {}
        agents._supervisor_agents = []
        agents.pool_supervisor = None

    _reset()
    yield
    _reset()


@pytest.fixture
def patched_factories():
    """Todo lo que tocaría la red, mockeado."""
    with patch("src.agent.agents.create_supervisor") as create_supervisor, \
         patch("src.agent.agents.create_agent") as create_agent, \
         patch("src.agent.agents.create_specialist_llm") as specialist_llm, \
         patch("src.agent.agents.create_fallback_llm") as fallback_llm, \
         patch("src.agent.agents.create_synthesizer_llm") as synthesizer_llm, \
         patch("src.agent.agents.create_routing_llm") as routing_llm:
        # Runnable de verdad y no MagicMock: `general` se envuelve en
        # RunnableWithFallbacks, que valida por tipo y rechaza un mock.
        create_agent.side_effect = lambda **kwargs: RunnableLambda(
            lambda state: state, name=kwargs["name"]
        )
        yield {
            "create_agent": create_agent,
            "create_supervisor": create_supervisor,
            "routing_llm": routing_llm,
            "synthesizer_llm": synthesizer_llm,
            "fallback_llm": fallback_llm,
            "specialist_llm": specialist_llm,
        }


# ============================================================
# pool_general_knowledge
# ============================================================

def test_pool_general_knowledge_echoes_the_topic():
    result = agents.pool_general_knowledge.invoke({"topic": "saltwater pools"})

    assert "saltwater pools" in result
    assert "General pool information" in result


# ============================================================
# _initialize()
# ============================================================

def test_initialize_registers_every_specialist_plus_general_oos_and_math(
    patched_factories,
):
    agents._initialize()

    expected = {"general", "oos", "math"} | {n for n, _ in agents.SPECIALIST_SPECS}
    assert set(agents._agents) == expected


def test_every_registered_node_name_is_a_valid_agent_name(patched_factories):
    """Un nodo que no está en AgentName es un step que el planner no puede pedir."""
    agents._initialize()

    assert set(agents._agents) <= AGENT_NAME_SET


def test_general_is_wrapped_with_a_fallback_that_keeps_its_name(patched_factories):
    """RunnableWithFallbacks no hereda .name, y create_supervisor lo exige."""
    agents._initialize()

    general = agents._agents["general"]
    assert general.name == "general"
    assert len(general.fallbacks) == 1


def test_specialists_get_the_retrieval_tools(patched_factories):
    agents._initialize()

    calls = {
        call.kwargs["name"]: call.kwargs
        for call in patched_factories["create_agent"].call_args_list
    }

    for node_name, _ in agents.SPECIALIST_SPECS:
        assert calls[node_name]["tools"] == agents.RETRIEVAL_TOOLS


def test_math_gets_the_deterministic_catalog_and_no_retrieval(patched_factories):
    agents._initialize()

    calls = {
        call.kwargs["name"]: call.kwargs
        for call in patched_factories["create_agent"].call_args_list
    }

    assert calls["math"]["tools"] is not agents.RETRIEVAL_TOOLS
    assert calls["math"]["tools"]


def test_initialize_runs_only_once(patched_factories):
    agents._initialize()
    call_count = patched_factories["create_agent"].call_count

    agents._initialize()

    assert patched_factories["create_agent"].call_count == call_count


def test_supervisor_receives_the_bare_general_agent_not_the_fallback_wrapper(
    patched_factories,
):
    agents._initialize()

    kwargs = patched_factories["create_supervisor"].call_args.kwargs
    assert agents._agents["general"] not in kwargs["agents"]
    assert kwargs["model"] is patched_factories["routing_llm"].return_value
    assert kwargs["prompt"] == agents.SUPERVISOR_PROMPT


def test_a_failing_supervisor_does_not_disable_the_specialists(patched_factories):
    """El pipeline real resuelve por get_agent_by_name; el supervisor es aparte."""
    patched_factories["create_supervisor"].side_effect = RuntimeError("boom")

    agents._initialize()

    assert agents.pool_supervisor is None
    assert agents._initialized is True
    assert agents._agents["chemistry"] is not None


# ============================================================
# get_agent_by_name() / get_supervisor()
# ============================================================

def test_get_agent_by_name_returns_the_registered_agent():
    fake_agent = MagicMock()
    agents._initialized = True
    agents._agents = {"general": fake_agent}

    assert agents.get_agent_by_name("general") is fake_agent


def test_get_agent_by_name_raises_for_an_unregistered_agent():
    agents._initialized = True
    agents._agents = {}

    with pytest.raises(ValueError) as exc:
        agents.get_agent_by_name("unknown")

    assert "not registered" in str(exc.value)


def test_get_supervisor_returns_the_compiled_supervisor():
    supervisor = MagicMock()
    agents._initialized = True
    agents.pool_supervisor = supervisor

    assert agents.get_supervisor() is supervisor
