"""
tests/agent/test_agents.py
==========================

Unit tests for src.agent.agents
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agent import agents


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(autouse=True)
def reset_agents():
    """Reset lazy initialization globals before every test."""

    agents._initialized = False
    agents._llm = None
    agents._routing_llm = None
    agents._synthesizer_llm = None
    agents._agents = {}
    agents.pool_supervisor = None


# ============================================================================
# pool_general_knowledge
# ============================================================================

def test_pool_general_knowledge():

    result = agents.pool_general_knowledge.invoke(
        {"topic": "saltwater pools"}
    )

    assert "saltwater pools" in result
    assert "General pool information" in result


# ============================================================================
# _initialize()
# ============================================================================

@patch("src.agent.agents.create_supervisor")
@patch("src.agent.agents.create_agent")
@patch("src.agent.agents.create_synthesizer_llm")
@patch("src.agent.agents.create_routing_llm")
@patch("src.agent.agents.create_llm")
def test_initialize(
    mock_create_llm,
    mock_create_routing_llm,
    mock_create_synthesizer_llm,
    mock_create_agent,
    mock_create_supervisor,
):
    """Verify lazy initialization creates all agents and supervisor."""

    llm = MagicMock(name="llm")
    routing_llm = MagicMock(name="routing_llm")
    synthesizer_llm = MagicMock(name="synthesizer_llm")

    mock_create_llm.return_value = llm
    mock_create_routing_llm.return_value = routing_llm
    mock_create_synthesizer_llm.return_value = synthesizer_llm

    general_agent = MagicMock(name="general_agent")
    oos_agent = MagicMock(name="oos_agent")
    diagnosis_agent = MagicMock(name="diagnosis_agent")
    dosage_agent = MagicMock(name="dosage_agent")
    equipment_agent = MagicMock(name="equipment_agent")
    maintenance_agent = MagicMock(name="maintenance_agent")

    mock_create_agent.side_effect = [
        general_agent,
        oos_agent,
        diagnosis_agent,
        dosage_agent,
        equipment_agent,
        maintenance_agent,
    ]

    compiled_supervisor = MagicMock(name="compiled_supervisor")

    mock_create_supervisor.return_value.compile.return_value = (
        compiled_supervisor
    )

    agents._initialize()

    assert agents._initialized is True

    assert agents._llm is llm
    assert agents._routing_llm is routing_llm
    assert agents._synthesizer_llm is synthesizer_llm

    assert agents._agents == {
        "general": general_agent,
        "oos": oos_agent,
        "diagnosis": diagnosis_agent,
        "dosage": dosage_agent,
        "equipment": equipment_agent,
        "maintenance": maintenance_agent,
    }

    assert agents.pool_supervisor is compiled_supervisor

    assert mock_create_agent.call_count == 6
    mock_create_supervisor.assert_called_once()


# ============================================================================
# _initialize() should only run once
# ============================================================================

@patch("src.agent.agents.create_llm")
def test_initialize_only_once(mock_create_llm):

    agents._initialized = True

    agents._initialize()

    mock_create_llm.assert_not_called()


# ============================================================================
# Verify create_agent wiring
# ============================================================================

@patch("src.agent.agents.create_supervisor")
@patch("src.agent.agents.create_agent")
@patch("src.agent.agents.create_synthesizer_llm")
@patch("src.agent.agents.create_routing_llm")
@patch("src.agent.agents.create_llm")
def test_initialize_creates_expected_agents(
    mock_create_llm,
    mock_create_routing_llm,
    mock_create_synthesizer_llm,
    mock_create_agent,
    mock_create_supervisor,
):

    mock_create_llm.return_value = MagicMock()
    mock_create_routing_llm.return_value = MagicMock()
    mock_create_synthesizer_llm.return_value = MagicMock()

    mock_create_agent.side_effect = [
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
    ]

    mock_create_supervisor.return_value.compile.return_value = MagicMock()

    agents._initialize()

    names = [
        call.kwargs["name"]
        for call in mock_create_agent.call_args_list
    ]

    assert names == [
        "general",
        "out_of_scope",
        "diagnosis",
        "dosage",
        "equipment",
        "maintenance",
    ]


# ============================================================================
# get_agent_by_name()
# ============================================================================

def test_get_agent_by_name():

    fake_agent = MagicMock()

    agents._initialized = True
    agents._agents = {
        "general": fake_agent
    }

    result = agents.get_agent_by_name("general")

    assert result is fake_agent


def test_get_agent_by_name_unknown():

    agents._initialized = True
    agents._agents = {}

    with pytest.raises(ValueError) as exc:
        agents.get_agent_by_name("unknown")

    assert "not registered" in str(exc.value)


# ============================================================================
# get_supervisor()
# ============================================================================

def test_get_supervisor():

    supervisor = MagicMock()

    agents._initialized = True
    agents.pool_supervisor = supervisor

    result = agents.get_supervisor()

    assert result is supervisor


# ============================================================================
# Supervisor creation
# ============================================================================

@patch("src.agent.agents.create_supervisor")
@patch("src.agent.agents.create_agent")
@patch("src.agent.agents.create_synthesizer_llm")
@patch("src.agent.agents.create_routing_llm")
@patch("src.agent.agents.create_llm")
def test_supervisor_receives_all_agents(
    mock_create_llm,
    mock_create_routing_llm,
    mock_create_synthesizer_llm,
    mock_create_agent,
    mock_create_supervisor,
):

    mock_create_llm.return_value = MagicMock()
    mock_create_routing_llm.return_value = MagicMock()
    mock_create_synthesizer_llm.return_value = MagicMock()

    created_agents = [MagicMock() for _ in range(6)]
    mock_create_agent.side_effect = created_agents

    compiled = MagicMock()

    mock_create_supervisor.return_value.compile.return_value = compiled

    agents._initialize()

    mock_create_supervisor.assert_called_once()

    kwargs = mock_create_supervisor.call_args.kwargs

    assert kwargs["agents"] == created_agents
    assert kwargs["model"] is mock_create_routing_llm.return_value
    assert kwargs["prompt"] == agents.SUPERVISOR_PROMPT

    assert agents.pool_supervisor is compiled