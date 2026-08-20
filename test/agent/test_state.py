"""
Tests localizados para src/agent/state.py

Cubre:
- ExecutionStep: validación de campos, default de `oos`, Literal `assigned_agent`.
- PlannerOutput: Literal `detected_language`, límites de `execution_plan` (1..5 pasos).
- AgentResult: campos requeridos, default_factory de `sources`, default de `error`.
- PoolAgentState (TypedDict): forma estructural (keys requeridas vs NotRequired) y
  el reducer `add_messages` usado en el canal `messages`.

NOTA sobre versión de Pydantic:
El código usa `Field(..., min_length=1, max_length=5)` sobre un campo `List[...]`.
Esa sintaxis de límites de longitud para listas es de **Pydantic v2**
(`ExecutionStep.min_length/max_length`). Si el proyecto corre en **Pydantic v1
puro**, esos kwargs no son los correctos para listas (en v1 serían
`min_items`/`max_items`) y es posible que la restricción NO se valide.
Los tests `test_execution_plan_requires_at_least_one_step` y
`test_execution_plan_rejects_more_than_five_steps` fallarán en ese caso — si
eso pasa, es una señal de que hay que revisar `state.py`, no el test.

Ajustá el import de abajo (`from src.agent.state import ...`) según cómo
resuelva los paquetes tu configuración de pytest (rootdir / conftest.py /
pyproject.toml). Si tu proyecto importa como `from agent.state import ...`,
cambiá esa línea.
"""
from __future__ import annotations

import typing

import pytest
from pydantic import ValidationError
from typing_extensions import NotRequired, get_origin
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages

from src.agent.state import (
    ExecutionStep,
    PlannerOutput,
    AgentResult,
    PoolAgentState,
)


def _resolved_notrequired_keys(typed_dict_cls) -> set[str]:
    """
    Devuelve las keys de un TypedDict que están envueltas en `NotRequired[...]`,
    resolviendo las anotaciones "de verdad" (no como strings).

    Necesario porque `state.py` usa `from __future__ import annotations`, lo
    que convierte las anotaciones en strings y hace que
    `PoolAgentState.__optional_keys__` / `__required_keys__` NO detecten
    correctamente los wrappers `NotRequired` (quedan como frozenset() vacío o
    incorrecto). `typing.get_type_hints(include_extras=True)` sí resuelve los
    strings a los objetos reales `NotRequired[...]`.
    """
    hints = typing.get_type_hints(typed_dict_cls, include_extras=True)
    return {key for key, hint in hints.items() if get_origin(hint) is NotRequired}


VALID_AGENT_NAMES = ["diagnosis", "dosage", "equipment", "maintenance", "oos"]


@pytest.fixture
def valid_step_kwargs():
    return dict(
        step=1,
        task="Map green water symptom to causal chemical parameters",
        assigned_agent="diagnosis",
    )


# =====================================================================
# ExecutionStep
# =====================================================================

class TestExecutionStep:
    def test_creates_with_required_fields(self, valid_step_kwargs):
        step = ExecutionStep(**valid_step_kwargs)
        assert step.step == 1
        assert step.assigned_agent == "diagnosis"
        assert step.oos is False  # default

    @pytest.mark.parametrize("agent", VALID_AGENT_NAMES)
    def test_accepts_all_defined_agent_names(self, valid_step_kwargs, agent):
        valid_step_kwargs["assigned_agent"] = agent
        step = ExecutionStep(**valid_step_kwargs)
        assert step.assigned_agent == agent

    def test_rejects_unknown_agent_name(self, valid_step_kwargs):
        valid_step_kwargs["assigned_agent"] = "not_a_real_agent"
        with pytest.raises(ValidationError):
            ExecutionStep(**valid_step_kwargs)

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            ExecutionStep(step=1, assigned_agent="diagnosis")  # falta `task`

    def test_step_rejects_non_numeric_value(self, valid_step_kwargs):
        valid_step_kwargs["step"] = "not-a-number"
        with pytest.raises(ValidationError):
            ExecutionStep(**valid_step_kwargs)

    def test_oos_can_be_explicitly_true(self, valid_step_kwargs):
        valid_step_kwargs["oos"] = True
        valid_step_kwargs["assigned_agent"] = "oos"
        step = ExecutionStep(**valid_step_kwargs)
        assert step.oos is True

    def test_oos_true_does_not_currently_enforce_oos_agent(self, valid_step_kwargs):
        """
        Documenta un GAP del modelo actual: el docstring de `oos` dice que
        `oos=True` DEBE implicar `assigned_agent='oos'`, pero no existe
        ningún validador que lo obligue. Este test hoy PASA justamente
        porque la regla no se aplica. Si en el futuro agregan un validador
        (@model_validator / @root_validator), este test debería cambiarse
        a `pytest.raises(ValidationError)`.
        """
        valid_step_kwargs["oos"] = True
        valid_step_kwargs["assigned_agent"] = "diagnosis"  # viola la regla documentada
        step = ExecutionStep(**valid_step_kwargs)
        assert step.oos is True
        assert step.assigned_agent == "diagnosis"


# =====================================================================
# PlannerOutput
# =====================================================================

class TestPlannerOutput:
    @staticmethod
    def _step(n=1, agent="diagnosis"):
        return ExecutionStep(step=n, task=f"task {n}", assigned_agent=agent)

    @pytest.mark.parametrize("lang", ["es", "en"])
    def test_accepts_valid_languages(self, lang):
        output = PlannerOutput(detected_language=lang, execution_plan=[self._step()])
        assert output.detected_language == lang

    def test_rejects_unsupported_language(self):
        with pytest.raises(ValidationError):
            PlannerOutput(detected_language="fr", execution_plan=[self._step()])

    def test_execution_plan_requires_at_least_one_step(self):
        with pytest.raises(ValidationError):
            PlannerOutput(detected_language="en", execution_plan=[])

    def test_execution_plan_accepts_up_to_five_steps(self):
        steps = [self._step(n=i) for i in range(1, 6)]
        output = PlannerOutput(detected_language="en", execution_plan=steps)
        assert len(output.execution_plan) == 5

    def test_execution_plan_rejects_more_than_five_steps(self):
        steps = [self._step(n=i) for i in range(1, 7)]
        with pytest.raises(ValidationError):
            PlannerOutput(detected_language="en", execution_plan=steps)

    def test_oos_query_single_step_shape(self):
        oos_step = ExecutionStep(
            step=1, task="Irrelevant request", assigned_agent="oos", oos=True
        )
        output = PlannerOutput(detected_language="en", execution_plan=[oos_step])
        assert len(output.execution_plan) == 1
        assert output.execution_plan[0].oos is True
        assert output.execution_plan[0].assigned_agent == "oos"


# =====================================================================
# AgentResult
# =====================================================================

class TestAgentResult:
    def test_creates_with_required_fields_only(self):
        result = AgentResult(agent="dosage", step=1, output="calculated dosage")
        assert result.agent == "dosage"
        assert result.step == 1
        assert result.output == "calculated dosage"
        assert result.sources == []   # default_factory
        assert result.error is None   # default

    def test_sources_default_factory_is_independent_per_instance(self):
        """Guarda contra el bug clásico de default mutable compartido entre instancias."""
        r1 = AgentResult(agent="dosage", step=1, output="a")
        r2 = AgentResult(agent="equipment", step=2, output="b")
        r1.sources.append("source-1")
        assert r2.sources == []

    def test_accepts_explicit_sources_and_error(self):
        result = AgentResult(
            agent="equipment",
            step=2,
            output="corrosion risk detected",
            sources=["kb://corrosion-guide"],
            error=None,
        )
        assert result.sources == ["kb://corrosion-guide"]

    def test_rejects_invalid_agent_literal(self):
        with pytest.raises(ValidationError):
            AgentResult(agent="not_an_agent", step=1, output="x")

    def test_missing_required_output_raises(self):
        with pytest.raises(ValidationError):
            AgentResult(agent="dosage", step=1)


# =====================================================================
# PoolAgentState (TypedDict)
# =====================================================================

class TestPoolAgentState:
    def test_is_typed_dict(self):
        assert hasattr(PoolAgentState, "__annotations__")
        # Ver nota en `_resolved_notrequired_keys`: estos dunders existen pero
        # NO son confiables acá porque `state.py` usa
        # `from __future__ import annotations` (anotaciones como strings).
        # Por eso el resto de los tests de esta clase usan
        # `typing.get_type_hints(..., include_extras=True)` en su lugar.
        assert hasattr(PoolAgentState, "__required_keys__")
        assert hasattr(PoolAgentState, "__optional_keys__")

    def test_required_keys_are_only_messages_and_summary(self):
        """
        NOTA: no usamos `PoolAgentState.__required_keys__` directamente porque
        `state.py` tiene `from __future__ import annotations`, lo que hace que
        ese dunder no reconozca los wrappers `NotRequired` (ver
        `_resolved_notrequired_keys`). En su lugar, derivamos "required" como
        "todas las keys anotadas MENOS las que resuelven a NotRequired".
        """
        all_keys = set(typing.get_type_hints(PoolAgentState, include_extras=True))
        optional_keys = _resolved_notrequired_keys(PoolAgentState)
        required_keys = all_keys - optional_keys

        assert required_keys == {"messages", "conversation_summary"}

    def test_optional_keys_include_expected_notrequired_fields(self):
        optional = _resolved_notrequired_keys(PoolAgentState)
        expected_optional = {
            "detected_language",
            "execution_plan",
            "current_step",
            "agent_results",
            "error",
        }
        assert optional == expected_optional

    def test_minimal_valid_instance_construction(self):
        state: PoolAgentState = {
            "messages": [HumanMessage(content="hola")],
            "conversation_summary": "",
        }
        assert state["conversation_summary"] == ""
        assert len(state["messages"]) == 1

    def test_full_instance_construction_with_all_optional_fields(self):
        step = ExecutionStep(step=1, task="diagnose", assigned_agent="diagnosis")
        result = AgentResult(agent="diagnosis", step=1, output="P_pH identified")

        state: PoolAgentState = {
            "messages": [HumanMessage(content="mi agua está verde")],
            "conversation_summary": "User reports green water.",
            "detected_language": "es",
            "execution_plan": [step],
            "current_step": 0,
            "agent_results": {"step_1": result},
            "error": None,
        }
        assert state["detected_language"] == "es"
        assert state["agent_results"]["step_1"].agent == "diagnosis"

    def test_add_messages_reducer_appends_in_order(self):
        """
        `messages` usa el reducer `add_messages` de langgraph. Debe combinar
        dos listas de mensajes agregándolos en orden.
        """
        m1 = AIMessage(content="first", id="1")
        m2 = AIMessage(content="second", id="2")
        merged = add_messages([m1], [m2])
        assert [m.content for m in merged] == ["first", "second"]

    def test_add_messages_reducer_replaces_on_matching_id(self):
        """
        Si se pasa un mensaje con un `id` ya existente, su contenido se
        REEMPLAZA en lugar de duplicarse (comportamiento estándar de
        `add_messages`).
        """
        m1 = AIMessage(content="first", id="1")
        m2 = AIMessage(content="second", id="2")
        merged = add_messages([m1], [m2])

        m2_updated = AIMessage(content="second-updated", id="2")
        merged_again = add_messages(merged, [m2_updated])
        assert [m.content for m in merged_again] == ["first", "second-updated"]