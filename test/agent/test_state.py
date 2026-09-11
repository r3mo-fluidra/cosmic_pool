"""
Tests de src/agent/state.py.

Cubre:
- ExecutionStep: campos, default de `oos`, Literal `assigned_agent`, el
  validador que ata oos<->assigned_agent, y `depends_on`.
- PlannerOutput: idiomas, techo de 5 steps, plan vacío con missing_inputs.
- AgentResult: campos requeridos, `status`, default_factory de `sources`.
- PoolAgentState: forma estructural (required vs NotRequired) y el reducer
  `add_messages` del canal `messages`.

Los nombres de agente salen de AgentName, no de una lista copiada a mano:
el roster de la v1 (diagnosis/dosage/maintenance) ya no existe y una lista
hardcodeada vuelve a quedar obsoleta en el siguiente cambio de roster.
"""
from __future__ import annotations

import typing

import pytest
from pydantic import ValidationError
from typing_extensions import NotRequired, get_args, get_origin
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages

from src.agent.agent_names import AgentName
from src.agent.state import (
    ExecutionStep,
    PlannerOutput,
    AgentResult,
    PoolAgentState,
)


VALID_AGENT_NAMES = list(get_args(AgentName))


def _resolved_notrequired_keys(typed_dict_cls) -> set[str]:
    """
    Keys de un TypedDict envueltas en `NotRequired[...]`.

    `state.py` usa `from __future__ import annotations`, así que
    `__optional_keys__` no ve los wrappers: hay que resolver las
    anotaciones con get_type_hints(include_extras=True).
    """
    hints = typing.get_type_hints(typed_dict_cls, include_extras=True)
    return {key for key, hint in hints.items() if get_origin(hint) is NotRequired}


@pytest.fixture
def valid_step_kwargs():
    return dict(
        step=1,
        task="Map green water symptom to causal chemical parameters",
        assigned_agent="chemistry",
    )


# =====================================================================
# ExecutionStep
# =====================================================================

class TestExecutionStep:
    def test_creates_with_required_fields(self, valid_step_kwargs):
        step = ExecutionStep(**valid_step_kwargs)

        assert step.step == 1
        assert step.assigned_agent == "chemistry"
        assert step.oos is False
        assert step.depends_on == []

    @pytest.mark.parametrize("agent", VALID_AGENT_NAMES)
    def test_accepts_every_name_declared_in_agent_name(self, valid_step_kwargs, agent):
        valid_step_kwargs["assigned_agent"] = agent
        if agent == "oos":
            valid_step_kwargs["oos"] = True

        assert ExecutionStep(**valid_step_kwargs).assigned_agent == agent

    def test_rejects_unknown_agent_name(self, valid_step_kwargs):
        valid_step_kwargs["assigned_agent"] = "not_a_real_agent"

        with pytest.raises(ValidationError):
            ExecutionStep(**valid_step_kwargs)

    def test_rejects_the_v1_roster(self, valid_step_kwargs):
        """diagnosis/dosage/maintenance desaparecieron con la v1."""
        for dead in ("diagnosis", "dosage", "maintenance"):
            valid_step_kwargs["assigned_agent"] = dead
            with pytest.raises(ValidationError):
                ExecutionStep(**valid_step_kwargs)

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            ExecutionStep(step=1, assigned_agent="chemistry")  # falta `task`

    def test_step_number_starts_at_one(self, valid_step_kwargs):
        valid_step_kwargs["step"] = 0

        with pytest.raises(ValidationError):
            ExecutionStep(**valid_step_kwargs)

    def test_step_rejects_non_numeric_value(self, valid_step_kwargs):
        valid_step_kwargs["step"] = "not-a-number"

        with pytest.raises(ValidationError):
            ExecutionStep(**valid_step_kwargs)

    def test_oos_flag_requires_the_oos_agent(self, valid_step_kwargs):
        valid_step_kwargs["oos"] = True
        valid_step_kwargs["assigned_agent"] = "chemistry"

        with pytest.raises(ValidationError):
            ExecutionStep(**valid_step_kwargs)

    def test_the_oos_agent_normalizes_the_flag_to_true(self, valid_step_kwargs):
        """Elegir `oos` sin marcar el flag no debe producir un step ambiguo."""
        valid_step_kwargs["assigned_agent"] = "oos"

        assert ExecutionStep(**valid_step_kwargs).oos is True

    def test_depends_on_holds_the_step_numbers_it_waits_for(self, valid_step_kwargs):
        valid_step_kwargs["step"] = 3

        step = ExecutionStep(**valid_step_kwargs, depends_on=[1, 2])

        assert step.depends_on == [1, 2]


# =====================================================================
# PlannerOutput
# =====================================================================

class TestPlannerOutput:
    @staticmethod
    def _step(n=1, agent="chemistry"):
        return ExecutionStep(step=n, task=f"task {n}", assigned_agent=agent)

    @pytest.mark.parametrize("lang", ["es", "en"])
    def test_accepts_valid_languages(self, lang):
        output = PlannerOutput(detected_language=lang, execution_plan=[self._step()])

        assert output.detected_language == lang

    def test_rejects_unsupported_language(self):
        with pytest.raises(ValidationError):
            PlannerOutput(detected_language="fr", execution_plan=[self._step()])

    def test_an_empty_plan_is_valid_when_inputs_are_missing(self):
        """Pedir datos no es un plan: el planner devuelve missing_inputs y nada más."""
        output = PlannerOutput(
            detected_language="es",
            execution_plan=[],
            missing_inputs=["pool volume", "current pH"],
        )

        assert output.execution_plan == []
        assert output.missing_inputs == ["pool volume", "current pH"]

    def test_execution_plan_accepts_up_to_five_steps(self):
        steps = [self._step(n=i) for i in range(1, 6)]

        assert len(PlannerOutput(detected_language="en", execution_plan=steps).execution_plan) == 5

    def test_execution_plan_rejects_more_than_five_steps(self):
        steps = [self._step(n=i) for i in range(1, 7)]

        with pytest.raises(ValidationError):
            PlannerOutput(detected_language="en", execution_plan=steps)

    def test_oos_query_is_a_single_oos_step(self):
        oos_step = ExecutionStep(
            step=1, task="Irrelevant request", assigned_agent="oos", oos=True
        )
        output = PlannerOutput(detected_language="en", execution_plan=[oos_step])

        assert len(output.execution_plan) == 1
        assert output.execution_plan[0].oos is True


# =====================================================================
# AgentResult
# =====================================================================

class TestAgentResult:
    def test_creates_with_required_fields_only(self):
        result = AgentResult(agent="math", step=1, output="calculated dosage")

        assert result.agent == "math"
        assert result.step == 1
        assert result.output == "calculated dosage"
        assert result.sources == []
        assert result.error is None
        assert result.status == "ok"

    @pytest.mark.parametrize("status", ["ok", "failed", "skipped"])
    def test_accepts_the_three_declared_statuses(self, status):
        result = AgentResult(agent="math", step=1, output="", status=status)

        assert result.status == status

    def test_rejects_an_undeclared_status(self):
        with pytest.raises(ValidationError):
            AgentResult(agent="math", step=1, output="", status="pending")

    def test_sources_default_factory_is_independent_per_instance(self):
        r1 = AgentResult(agent="math", step=1, output="a")
        r2 = AgentResult(agent="equipment", step=2, output="b")

        r1.sources.append("source-1")

        assert r2.sources == []

    def test_accepts_explicit_sources(self):
        result = AgentResult(
            agent="equipment",
            step=2,
            output="corrosion risk detected",
            sources=["kb://corrosion-guide"],
        )

        assert result.sources == ["kb://corrosion-guide"]

    def test_rejects_invalid_agent_literal(self):
        with pytest.raises(ValidationError):
            AgentResult(agent="not_an_agent", step=1, output="x")

    def test_missing_required_output_raises(self):
        with pytest.raises(ValidationError):
            AgentResult(agent="math", step=1)


# =====================================================================
# PoolAgentState (TypedDict)
# =====================================================================

class TestPoolAgentState:
    def test_required_keys_are_the_channels_with_no_per_turn_reset(self):
        all_keys = set(typing.get_type_hints(PoolAgentState, include_extras=True))
        required_keys = all_keys - _resolved_notrequired_keys(PoolAgentState)

        assert required_keys == {
            "messages",
            "conversation_summary",
            "turn_started_at",
            "agent_results",
            "archetype",
            "misroute_retries",
            "ignored_chip_streak",
            "error",
            "planner_error",
        }

    def test_optional_keys_include_the_planner_and_response_fields(self):
        optional = _resolved_notrequired_keys(PoolAgentState)

        assert optional == {
            "detected_language",
            "execution_plan",
            "missing_inputs",
            "current_step",
            "response",
            "validation",
            "suggestions",
        }

    @pytest.mark.parametrize(
        "channel", ["error", "planner_error", "archetype", "agent_results"]
    )
    def test_channels_written_by_more_than_one_node_declare_a_reducer(self, channel):
        """
        Sin reducer declarado, dos nodos del mismo superstep escribiendo el
        mismo canal dan InvalidUpdateError. `planner_error` no estaba
        declarado en absoluto y LangGraph lo descartaba en silencio.
        """
        hint = typing.get_type_hints(PoolAgentState, include_extras=True)[channel]

        assert getattr(hint, "__metadata__", None), f"{channel} sin reducer"

    def test_minimal_valid_instance_construction(self):
        state: PoolAgentState = {
            "messages": [HumanMessage(content="hola")],
            "conversation_summary": "",
        }

        assert state["conversation_summary"] == ""
        assert len(state["messages"]) == 1

    def test_full_instance_construction_with_all_optional_fields(self):
        step = ExecutionStep(step=1, task="diagnose", assigned_agent="chemistry")
        result = AgentResult(agent="chemistry", step=1, output="P_pH identified")

        state: PoolAgentState = {
            "messages": [HumanMessage(content="mi agua está verde")],
            "conversation_summary": "User reports green water.",
            "detected_language": "es",
            "execution_plan": [step],
            "missing_inputs": [],
            "current_step": 0,
            "agent_results": {"step_1": result},
            "error": None,
        }

        assert state["detected_language"] == "es"
        assert state["agent_results"]["step_1"].agent == "chemistry"

    def test_add_messages_reducer_appends_in_order(self):
        m1 = AIMessage(content="first", id="1")
        m2 = AIMessage(content="second", id="2")

        assert [m.content for m in add_messages([m1], [m2])] == ["first", "second"]

    def test_add_messages_reducer_replaces_on_matching_id(self):
        merged = add_messages(
            [AIMessage(content="first", id="1")], [AIMessage(content="second", id="2")]
        )
        merged_again = add_messages(
            merged, [AIMessage(content="second-updated", id="2")]
        )

        assert [m.content for m in merged_again] == ["first", "second-updated"]
