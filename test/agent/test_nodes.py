"""
Tests de los nodos "simples" de src/agent/nodes.py: helpers puros,
build_context_node, summarize_memory_node, planner y synthesizer.

El orchestrator tiene fichero propio (test_orchestrator.py): su superficie
—fan-out, dependencias, circuit breaker— no entra cómodamente acá.

Estrategia de mocking
─────────────────────
1. Los getters de LLM (`_get_llm`, `_get_synthesis_llm`, `_get_planner_chain`)
   son singletons perezosos a nivel de módulo. Se parchean con
   monkeypatch.setattr sobre el módulo `nodes`, no sobre `config.llm`: las
   funciones los resuelven como globals en runtime, así que el parche surte
   efecto sin tener que limpiar el caché global entre tests.

2. `langfuse.observe` se neutraliza en test/conftest.py, que corre antes de
   cualquier import — la decoración ocurre en tiempo de import, no de
   ejecución.

Ningún test de este fichero toca la red.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.types import Command

from src.agent import nodes
from src.agent.state import ExecutionStep, AgentResult
from src.graph_context.response_contracts import SynthesizerOutput


# ================================================================
# HELPERS / FACTORIES
# ================================================================

def make_step(step=1, agent="chemistry", task="do something", oos=False, depends_on=None):
    return ExecutionStep(
        step=step,
        task=task,
        assigned_agent=agent,
        oos=oos,
        depends_on=depends_on or [],
    )


def make_result(step=1, agent="chemistry", output="ok", error=None, status=None):
    kwargs = {} if status is None else {"status": status}
    return AgentResult(agent=agent, step=step, output=output, error=error, **kwargs)


CONFIG = {"configurable": {"thread_id": "test-thread"}}


@pytest.fixture
def fake_synthesis_llm(monkeypatch):
    """
    Sustituye `_get_synthesis_llm()`. El synthesizer llama
    `.with_structured_output(SynthesizerOutput).invoke(messages)`, así que el
    doble tiene que respetar esa cadena.

    Devuelve el mock estructurado para poder inspeccionar el prompt enviado.
    """
    structured = MagicMock()
    structured.invoke.return_value = SynthesizerOutput(
        answer="respuesta final", actions=[], safety=None, details=[]
    )

    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    monkeypatch.setattr(nodes, "_get_synthesis_llm", lambda: llm)
    return structured


def system_content_of(structured_mock) -> str:
    return structured_mock.invoke.call_args.args[0][0].content


# ================================================================
# _extract_text
# ================================================================

class TestExtractText:
    def test_plain_string_passthrough(self):
        assert nodes._extract_text("hello") == "hello"

    def test_non_string_scalar_is_stringified(self):
        assert nodes._extract_text(123) == "123"

    def test_list_of_text_blocks_joined(self):
        assert nodes._extract_text([{"text": "hello"}, {"text": "world"}]) == "hello world"

    def test_list_skips_blank_and_non_dict_items(self):
        content = [{"text": "hello"}, {"text": "   "}, "not-a-dict", {"other": "x"}]
        assert nodes._extract_text(content) == "hello"

    def test_empty_list_returns_empty_string(self):
        assert nodes._extract_text([]) == ""


# ================================================================
# _normalize_agent
# ================================================================

class TestNormalizeAgent:
    def test_none_becomes_the_empty_string(self):
        assert nodes._normalize_agent(None) == ""

    def test_trims_and_lowercases(self):
        assert nodes._normalize_agent("  Math  ") == "math"

    def test_unwraps_the_value_of_an_enum_like_object(self):
        assert nodes._normalize_agent(SimpleNamespace(value="Chemistry")) == "chemistry"


# ================================================================
# _route_from_plan
# ================================================================

class TestRouteFromPlan:
    def test_an_empty_plan_goes_to_the_orchestrator(self):
        assert nodes._route_from_plan([]) == "orchestrator"

    def test_a_single_general_step_shortcuts_to_general(self):
        assert nodes._route_from_plan([make_step(agent="general")]) == "general"

    def test_a_single_oos_step_shortcuts_to_oos(self):
        assert nodes._route_from_plan([make_step(agent="oos", oos=True)]) == "oos"

    def test_a_multi_step_plan_always_goes_to_the_orchestrator(self):
        plan = [make_step(step=1, agent="general"), make_step(step=2)]
        assert nodes._route_from_plan(plan) == "orchestrator"


# ================================================================
# _is_oos
# ================================================================

class TestIsOos:
    def test_single_step_oos_true(self):
        assert nodes._is_oos([make_step(agent="oos", oos=True)]) is True

    def test_single_step_oos_false(self):
        assert nodes._is_oos([make_step(oos=False)]) is False

    def test_multi_step_plan_is_never_oos(self):
        plan = [make_step(agent="oos", oos=True), make_step(step=2)]
        assert nodes._is_oos(plan) is False

    def test_empty_plan_is_not_oos(self):
        assert nodes._is_oos([]) is False


# ================================================================
# is_infra_error
# ================================================================

class TestIsInfraError:
    @pytest.mark.parametrize("err", ["429 quota exceeded", "503 Service Unavailable"])
    def test_provider_status_codes_are_infra(self, err):
        assert nodes.is_infra_error(err) is True

    @pytest.mark.parametrize(
        "err",
        [
            "MISSING_INPUTS: need pool volume",
            "CANNOT_COMPUTE: no formula",
            "NO_GRAPH_COVERAGE: topic absent",
            "TOOL_BUDGET_EXCEEDED: math exceeded its recursion_limit",
        ],
    )
    def test_business_contracts_are_not_infra(self, err):
        """Un gap de negocio no debe disparar el circuit breaker del turno."""
        assert nodes.is_infra_error(err) is False

    def test_no_error_is_not_infra(self):
        assert nodes.is_infra_error(None) is False


# ================================================================
# _build_raw_content
# ================================================================

class TestBuildRawContent:
    def test_empty_results_returns_empty_string(self):
        assert nodes._build_raw_content({}) == ""

    def test_successful_step_formats_output_section(self):
        results = {"step_1": make_result(step=1, agent="chemistry", output="pH is low")}
        assert nodes._build_raw_content(results) == "[Step 1 — chemistry]\npH is low"

    def test_errored_step_formats_error_section(self):
        results = {"step_1": make_result(step=1, agent="math", output="", error="boom", status="failed")}
        assert nodes._build_raw_content(results) == "[Step 1 — math] ERROR: boom"

    def test_sections_ordered_by_step_not_dict_insertion_order(self):
        raw = nodes._build_raw_content({
            "step_2": make_result(step=2, agent="math", output="second"),
            "step_1": make_result(step=1, agent="chemistry", output="first"),
        })
        assert raw.index("first") < raw.index("second")

    def test_step_with_no_output_and_no_error_is_omitted(self):
        assert nodes._build_raw_content({"step_1": make_result(output="")}) == ""

    def test_failed_steps_reach_the_synthesizer_too(self):
        """Sin el error en el raw content, el turno degrada a un saludo genérico."""
        results = {
            "step_1": make_result(step=1, output="ok"),
            "step_2": make_result(step=2, output="", error="504 timeout", status="failed"),
        }
        raw = nodes._build_raw_content(results)

        assert "504 timeout" in raw


# ================================================================
# estimated_tokens
# ================================================================

class TestEstimatedTokens:
    def test_string_content(self):
        assert nodes.estimated_tokens([HumanMessage(content="a" * 40)]) == 10

    def test_list_of_blocks_content(self):
        msgs = [AIMessage(content=[{"text": "a" * 20}, {"text": "b" * 20}])]
        assert nodes.estimated_tokens(msgs) == 10

    def test_mixed_message_shapes(self):
        msgs = [HumanMessage(content="a" * 8), AIMessage(content=[{"text": "b" * 8}])]
        assert nodes.estimated_tokens(msgs) == 4

    def test_no_messages(self):
        assert nodes.estimated_tokens([]) == 0


# ================================================================
# build_context_node
# ================================================================

class TestBuildContextNode:
    def test_short_conversation_routes_to_planner(self):
        result = nodes.build_context_node({"messages": [HumanMessage(content="hi")]})

        assert isinstance(result, Command)
        assert result.goto == "planner"

    def test_conversation_over_token_limit_routes_to_summarizer(self):
        long_text = "x" * ((nodes.TOKEN_LIMIT + 10) * 4)
        result = nodes.build_context_node({"messages": [HumanMessage(content=long_text)]})

        assert result.goto == "summarize_memory_node"

    @pytest.mark.parametrize(
        "channel", ["error", "planner_error", "archetype", "response"]
    )
    def test_clears_the_per_turn_channels(self, channel):
        """
        El checkpointer los persiste entre turnos y no tienen centinela propio.
        `error` es el crítico: should_suggest corta con cualquier error, así
        que un turno fallido dejaba el thread sin chips para siempre.
        """
        result = nodes.build_context_node({"messages": [HumanMessage(content="hi")]})

        assert result.update[channel] is None

    def test_stamps_the_turn_start_for_the_budget(self):
        result = nodes.build_context_node({"messages": [HumanMessage(content="hi")]})

        assert result.update["turn_started_at"] > 0


# ================================================================
# summarize_memory_node
# ================================================================

class TestSummarizeMemoryNode:
    def test_few_messages_skips_llm_call_and_goes_to_planner(self, monkeypatch):
        fake_llm = MagicMock()
        monkeypatch.setattr(nodes, "_get_llm", lambda: fake_llm)

        messages = [HumanMessage(content=f"m{i}") for i in range(nodes.MESSAGES_TO_KEEP)]
        result = nodes.summarize_memory_node(
            {"messages": messages, "conversation_summary": ""}
        )

        assert result.goto == "planner"
        assert result.update is None
        fake_llm.invoke.assert_not_called()

    def test_summarizes_from_scratch_when_no_previous_summary(self, monkeypatch):
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = AIMessage(content="brand new summary")
        monkeypatch.setattr(nodes, "_get_llm", lambda: fake_llm)

        messages = [
            HumanMessage(content=f"m{i}", id=f"id{i}")
            for i in range(nodes.MESSAGES_TO_KEEP + 3)
        ]
        result = nodes.summarize_memory_node(
            {"messages": messages, "conversation_summary": ""}
        )

        assert result.update["conversation_summary"] == "brand new summary"

        removed_ids = {m.id for m in result.update["messages"]}
        assert removed_ids == {m.id for m in messages[: -nodes.MESSAGES_TO_KEEP]}
        assert all(isinstance(m, RemoveMessage) for m in result.update["messages"])

        sent_prompt = fake_llm.invoke.call_args.args[0][-1].content
        assert "Summarize the following conversation" in sent_prompt

    def test_extends_previous_summary_when_present(self, monkeypatch):
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = AIMessage(content="extended summary")
        monkeypatch.setattr(nodes, "_get_llm", lambda: fake_llm)

        messages = [
            HumanMessage(content=f"m{i}", id=f"id{i}")
            for i in range(nodes.MESSAGES_TO_KEEP + 2)
        ]
        result = nodes.summarize_memory_node(
            {"messages": messages, "conversation_summary": "old summary"}
        )

        sent_prompt = fake_llm.invoke.call_args.args[0][-1].content
        assert "old summary" in sent_prompt
        assert "Extend this summary" in sent_prompt
        assert result.update["conversation_summary"] == "extended summary"


# ================================================================
# planner
# ================================================================

class TestPlanner:
    def _patch_chain(self, monkeypatch, return_value=None, side_effect=None):
        fake_chain = MagicMock()
        if side_effect is not None:
            fake_chain.invoke.side_effect = side_effect
        else:
            fake_chain.invoke.return_value = return_value
        monkeypatch.setattr(nodes, "_get_planner_chain", lambda: fake_chain)
        return fake_chain

    def test_uses_detected_language_from_plan(self, monkeypatch):
        plan = SimpleNamespace(detected_language="en", execution_plan=[make_step()])
        self._patch_chain(monkeypatch, return_value=plan)

        result = nodes.planner({"messages": [HumanMessage(content="hello")]}, CONFIG)

        assert result.goto == "orchestrator"
        assert result.update["detected_language"] == "en"
        assert result.update["execution_plan"] == plan.execution_plan

    def test_resets_agent_results_with_the_none_sentinel(self, monkeypatch):
        """
        None y no {}: `agent_results` tiene reducer, así que {} se MERGEA con
        lo del turno anterior en vez de limpiarlo. El centinela es la única
        forma de resetear el canal.
        """
        plan = SimpleNamespace(detected_language="es", execution_plan=[make_step()])
        self._patch_chain(monkeypatch, return_value=plan)

        result = nodes.planner({"messages": [HumanMessage(content="hola")]}, CONFIG)

        assert result.update["agent_results"] is None

    def test_falls_back_to_state_language_when_plan_language_missing(self, monkeypatch):
        plan = SimpleNamespace(detected_language=None, execution_plan=[make_step()])
        self._patch_chain(monkeypatch, return_value=plan)

        state = {"messages": [HumanMessage(content="hola")], "detected_language": "es"}
        result = nodes.planner(state, CONFIG)

        assert result.update["detected_language"] == "es"

    def test_falls_back_to_spanish_when_nothing_else_available(self, monkeypatch):
        plan = SimpleNamespace(detected_language=None, execution_plan=[make_step()])
        self._patch_chain(monkeypatch, return_value=plan)

        result = nodes.planner({"messages": [HumanMessage(content="hola")]}, CONFIG)

        assert result.update["detected_language"] == "es"

    def test_sends_a_plain_dict_to_the_chain_not_a_message_list(self, monkeypatch):
        """
        Una lista hacía que langchain la envolviera en {"input": <lista>}: el
        PLANNER_PROMPT viajaba dos veces por turno y el mensaje del usuario
        llegaba enterrado en un literal de Python.
        """
        chain = self._patch_chain(
            monkeypatch,
            return_value=SimpleNamespace(detected_language="es", execution_plan=[make_step()]),
        )

        nodes.planner({"messages": [HumanMessage(content="hola")]}, CONFIG)

        payload = chain.invoke.call_args.args[0]
        assert isinstance(payload, dict)
        assert set(payload) == {"input"}

    def test_includes_summary_last_marlin_message_and_user_reply(self, monkeypatch):
        chain = self._patch_chain(
            monkeypatch,
            return_value=SimpleNamespace(detected_language="es", execution_plan=[make_step()]),
        )

        state = {
            "messages": [
                HumanMessage(content="first question"),
                AIMessage(content="previous answer", name="Marlin"),
                HumanMessage(content="follow up"),
            ],
            "conversation_summary": "user has a 50 m3 pool",
        }
        nodes.planner(state, CONFIG)

        context = chain.invoke.call_args.args[0]["input"]
        assert "user has a 50 m3 pool" in context
        assert "previous answer" in context
        assert "follow up" in context

    def test_ignores_ai_messages_not_authored_by_marlin(self, monkeypatch):
        chain = self._patch_chain(
            monkeypatch,
            return_value=SimpleNamespace(detected_language="es", execution_plan=[make_step()]),
        )

        state = {
            "messages": [
                AIMessage(content="not marlin", name="OtherBot"),
                HumanMessage(content="user text"),
            ]
        }
        nodes.planner(state, CONFIG)

        assert chain.invoke.call_args.args[0]["input"] == "user text"

    def test_an_llm_failure_degrades_to_a_single_general_step(self, monkeypatch):
        self._patch_chain(monkeypatch, side_effect=RuntimeError("503 unavailable"))

        result = nodes.planner({"messages": [HumanMessage(content="hola")]}, CONFIG)

        assert result.goto == "general"
        assert len(result.update["execution_plan"]) == 1
        assert result.update["execution_plan"][0].assigned_agent == "general"
        assert "503 unavailable" in result.update["planner_error"]


# ================================================================
# synthesizer
# ================================================================

class TestSynthesizer:
    def test_includes_the_raw_content_and_the_target_language(self, fake_synthesis_llm):
        state = {
            "execution_plan": [make_step(agent="chemistry")],
            "agent_results": {"step_1": make_result(output="pH is low")},
            "detected_language": "en",
            "archetype": "assessment",
        }

        nodes.synthesizer(state)

        system = system_content_of(fake_synthesis_llm)
        assert "pH is low" in system
        assert "English" in system

    def test_unknown_language_code_falls_back_to_spanish(self, fake_synthesis_llm):
        state = {
            "execution_plan": [make_step(agent="chemistry")],
            "agent_results": {"step_1": make_result(output="algo")},
            "detected_language": "fr",
            "archetype": "assessment",
        }

        nodes.synthesizer(state)

        assert "Spanish (Latin American)" in system_content_of(fake_synthesis_llm)

    def test_the_answer_is_published_as_a_marlin_message(self, fake_synthesis_llm):
        state = {
            "execution_plan": [make_step(agent="chemistry")],
            "agent_results": {"step_1": make_result(output="algo")},
            "detected_language": "es",
            "archetype": "assessment",
        }

        result = nodes.synthesizer(state)

        assert result["messages"][0].name == "Marlin"
        assert "respuesta final" in result["messages"][0].content
        assert result["response"].answer == "respuesta final"

    def test_an_empty_turn_gets_a_greeting_and_never_calls_the_fallback(
        self, fake_synthesis_llm
    ):
        nodes.synthesizer({"execution_plan": [], "agent_results": {}, "detected_language": "es"})

        assert "warm greeting" in system_content_of(fake_synthesis_llm)

    def test_a_plan_that_produced_nothing_is_a_failure_not_a_greeting(
        self, fake_synthesis_llm
    ):
        """
        raw_content vacío con plan no vacío es un bug de escritura de estado.
        Saludar al usuario lo enmascara: se devuelve el fallback explícito.
        """
        state = {
            "execution_plan": [make_step(agent="chemistry")],
            "agent_results": {},
            "detected_language": "es",
        }

        result = nodes.synthesizer(state)

        assert result["validation"]["fallback"] == "empty_results"
        fake_synthesis_llm.invoke.assert_not_called()

    def test_the_oos_node_prose_is_not_re_synthesized(self, fake_synthesis_llm):
        """Re-redactarla cuesta una llamada entera para producir el mismo texto."""
        state = {
            "execution_plan": [make_step(agent="oos", oos=True)],
            "agent_results": {"step_1": make_result(agent="oos", output="Eso está fuera de mi alcance.")},
            "detected_language": "es",
        }

        result = nodes.synthesizer(state)

        assert result["archetype"] == "oos"
        assert result["response"].answer == "Eso está fuera de mi alcance."
        fake_synthesis_llm.invoke.assert_not_called()

    def test_a_general_clarification_keeps_its_own_prose(self, fake_synthesis_llm):
        state = {
            "execution_plan": [make_step(agent="general")],
            "agent_results": {
                "step_1": make_result(agent="general", output="¿Cuál es el volumen de tu piscina?")
            },
            "detected_language": "es",
        }

        result = nodes.synthesizer(state)

        assert result["validation"]["is_clarification"] is True
        assert result["response"].answer == "¿Cuál es el volumen de tu piscina?"
        fake_synthesis_llm.invoke.assert_not_called()

    def test_falls_back_to_a_static_payload_when_both_models_fail(
        self, fake_synthesis_llm, monkeypatch
    ):
        fake_synthesis_llm.invoke.side_effect = RuntimeError("503")
        failing_fallback = MagicMock()
        failing_fallback.invoke.side_effect = RuntimeError("429")
        monkeypatch.setattr(nodes, "_get_fallback_llm", lambda: failing_fallback)

        state = {
            "execution_plan": [make_step(agent="chemistry")],
            "agent_results": {"step_1": make_result(output="algo")},
            "detected_language": "es",
            "archetype": "assessment",
        }

        result = nodes.synthesizer(state)

        assert result["validation"]["fallback"] == "static"
        assert result["response"].answer
