"""
Tests para src/agent/nodes.py

Estrategia de mocking (por qué está hecho así):

1. `_get_llm()` y `_get_planner_chain()` son singletons perezosos definidos como
   funciones a nivel de módulo en nodes.py. En vez de mockear `create_llm` /
   `create_planner_chain` (lo que obligaría a lidiar con el caché global `_llm`
   / `_planner_chain` entre tests), se parchea directamente
   `nodes._get_llm` / `nodes._get_planner_chain` con `monkeypatch.setattr`.
   Como las funciones del módulo llaman a estos nombres como globals en tiempo
   de ejecución (no los capturan como bound methods), el parche surte efecto
   sin fugas de estado entre tests.

2. `get_agent_by_name` se importa dentro de nodes.py con
   `from .agents import get_agent_by_name`, quedando como global del módulo
   `nodes`. Se parchea igual: `monkeypatch.setattr(nodes, "get_agent_by_name", ...)`.
   Esto evita inicializar el supervisor real (LLMs, Neo4j, Qdrant, etc.).

3. `ExecutionStep` y `AgentResult` se importan tal cual del `state.py` real del
   proyecto (son modelos pydantic simples, sin dependencias pesadas), para que
   los tests validen contra los mismos tipos que usa la app en producción.

4. `langfuse.observe` decora `planner`, `orchestrator` y `synthesizer`. No se
   mockea explícitamente: se asume que `langfuse` está instalado en el entorno
   de test (como lo está en producción) y que el decorador no requiere
   credenciales para envolver la función — solo para el tracing real, que en
   test simplemente no se reporta a ningún backend.

Requisitos para correr: pytest, y que el proyecto exponga `src` como paquete
importable (por eso el bootstrap de sys.path al inicio, para que el archivo
funcione sin importar cómo esté configurado pytest.ini / pyproject.toml).
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# --- Bootstrap: garantiza que "src" sea importable sin importar la config de pytest ---
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langgraph.types import Command

from src.agent import nodes
from src.agent.state import ExecutionStep, AgentResult


# ================================================================
# HELPERS / FACTORIES
# ================================================================

def make_step(step=1, agent="diagnosis", task="do something", oos=False) -> ExecutionStep:
    """Nota: `agent` debe ser uno de los AgentName válidos:
    diagnosis | dosage | equipment | maintenance | ooo
    ("general" NO es válido para ExecutionStep aunque exista como agente registrado
    en agents.py — no usarlo aquí o pydantic lo rechazará en el proyecto real)."""
    return ExecutionStep(step=step, task=task, assigned_agent=agent, oos=oos)


def make_result(step=1, agent="diagnosis", output="ok", error=None) -> AgentResult:
    return AgentResult(agent=agent, step=step, output=output, error=error)


# ================================================================
# _extract_text
# ================================================================

class TestExtractText:
    def test_plain_string_passthrough(self):
        assert nodes._extract_text("hello") == "hello"

    def test_non_string_scalar_is_stringified(self):
        assert nodes._extract_text(123) == "123"

    def test_list_of_text_blocks_joined(self):
        content = [{"text": "hello"}, {"text": "world"}]
        assert nodes._extract_text(content) == "hello world"

    def test_list_skips_blank_and_non_dict_items(self):
        content = [{"text": "hello"}, {"text": "   "}, "not-a-dict", {"other": "x"}]
        assert nodes._extract_text(content) == "hello"

    def test_empty_list_returns_empty_string(self):
        assert nodes._extract_text([]) == ""


# ================================================================
# _is_oos
# ================================================================

class TestIsOos:
    def test_single_step_oos_true(self):
        plan = [make_step(agent="ooo", oos=True)]
        assert nodes._is_oos(plan) is True

    def test_single_step_oos_false(self):
        plan = [make_step(oos=False)]
        assert nodes._is_oos(plan) is False

    def test_multi_step_plan_is_never_oos(self):
        plan = [make_step(agent="ooo", oos=True), make_step(step=2)]
        assert nodes._is_oos(plan) is False

    def test_empty_plan_is_not_oos(self):
        assert nodes._is_oos([]) is False


# ================================================================
# _build_raw_content
# ================================================================

class TestBuildRawContent:
    def test_empty_results_returns_empty_string(self):
        assert nodes._build_raw_content({}) == ""

    def test_successful_step_formats_output_section(self):
        results = {"step_1": make_result(step=1, agent="diagnosis", output="pH is low")}
        assert nodes._build_raw_content(results) == "[Step 1 — diagnosis]\npH is low"

    def test_errored_step_formats_error_section(self):
        results = {"step_1": make_result(step=1, agent="dosage", output="", error="boom")}
        assert nodes._build_raw_content(results) == "[Step 1 — dosage] ERROR: boom"

    def test_sections_ordered_by_step_not_dict_insertion_order(self):
        second = make_result(step=2, agent="dosage", output="second")
        first = make_result(step=1, agent="diagnosis", output="first")
        raw = nodes._build_raw_content({"step_2": second, "step_1": first})
        assert raw.index("first") < raw.index("second")

    def test_step_with_no_output_and_no_error_is_omitted(self):
        results = {"step_1": make_result(step=1, output="", error=None)}
        assert nodes._build_raw_content(results) == ""


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
        state = {"messages": [HumanMessage(content="hi")]}
        result = nodes.build_context_node(state)
        assert isinstance(result, Command)
        assert result.goto == "planner"

    def test_conversation_over_token_limit_routes_to_summarizer(self):
        long_text = "x" * ((nodes.TOKEN_LIMIT + 10) * 4)
        state = {"messages": [HumanMessage(content=long_text)]}
        result = nodes.build_context_node(state)
        assert result.goto == "summarize_memory_node"


# ================================================================
# summarize_memory_node
# ================================================================

class TestSummarizeMemoryNode:
    def test_few_messages_skips_llm_call_and_goes_to_planner(self, monkeypatch):
        fake_llm = MagicMock()
        monkeypatch.setattr(nodes, "_get_llm", lambda: fake_llm)

        messages = [HumanMessage(content=f"m{i}") for i in range(nodes.MESSAGES_TO_KEEP)]
        state = {"messages": messages, "conversation_summary": ""}

        result = nodes.summarize_memory_node(state)

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
        state = {"messages": messages, "conversation_summary": ""}

        result = nodes.summarize_memory_node(state)

        assert result.goto == "planner"
        assert result.update["conversation_summary"] == "brand new summary"

        removed_ids = {m.id for m in result.update["messages"]}
        expected_removed_ids = {m.id for m in messages[: -nodes.MESSAGES_TO_KEEP]}
        assert removed_ids == expected_removed_ids
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
        state = {"messages": messages, "conversation_summary": "old summary"}

        result = nodes.summarize_memory_node(state)

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

        result = nodes.planner({"messages": [HumanMessage(content="hello")]})

        assert result.goto == "orchestrator"
        assert result.update["detected_language"] == "en"
        assert result.update["execution_plan"] == plan.execution_plan
        assert result.update["current_step"] == 0
        assert result.update["agent_results"] == {}

    def test_falls_back_to_state_language_when_plan_language_missing(self, monkeypatch):
        plan = SimpleNamespace(detected_language=None, execution_plan=[make_step()])
        self._patch_chain(monkeypatch, return_value=plan)

        state = {"messages": [HumanMessage(content="hola")], "detected_language": "es"}
        result = nodes.planner(state)

        assert result.update["detected_language"] == "es"

    def test_falls_back_to_spanish_when_nothing_else_available(self, monkeypatch):
        plan = SimpleNamespace(detected_language=None, execution_plan=[make_step()])
        self._patch_chain(monkeypatch, return_value=plan)

        result = nodes.planner({"messages": [HumanMessage(content="hola")]})

        assert result.update["detected_language"] == "es"

    def test_includes_last_izel_message_and_user_reply_in_context(self, monkeypatch):
        captured = {}

        def fake_invoke(messages):
            captured["context"] = messages[-1]["content"]
            return SimpleNamespace(detected_language="es", execution_plan=[make_step()])

        self._patch_chain(monkeypatch, side_effect=fake_invoke)

        state = {
            "messages": [
                HumanMessage(content="first question"),
                AIMessage(content="previous answer", name="Izel"),
                HumanMessage(content="follow up"),
            ]
        }
        nodes.planner(state)

        assert "previous answer" in captured["context"]
        assert "follow up" in captured["context"]

    def test_ignores_ai_messages_not_authored_by_izel(self, monkeypatch):
        captured = {}

        def fake_invoke(messages):
            captured["context"] = messages[-1]["content"]
            return SimpleNamespace(detected_language="es", execution_plan=[make_step()])

        self._patch_chain(monkeypatch, side_effect=fake_invoke)

        state = {
            "messages": [
                AIMessage(content="not izel", name="OtherBot"),
                HumanMessage(content="user text"),
            ]
        }
        nodes.planner(state)

        assert captured["context"] == "user text"


# ================================================================
# orchestrator
# ================================================================

class TestOrchestrator:
    def test_empty_execution_plan_routes_to_synthesizer_with_error(self):
        state = {"execution_plan": [], "agent_results": {}, "current_step": 0}
        result = nodes.orchestrator(state)

        assert result.goto == "synthesizer"
        assert "execution_plan is empty" in result.update["error"]

    def test_all_steps_already_done_routes_to_synthesizer(self):
        plan = [make_step(step=1)]
        state = {"execution_plan": plan, "agent_results": {}, "current_step": 1}
        result = nodes.orchestrator(state)

        assert result.goto == "synthesizer"
        assert result.update is None

    def test_runs_pending_step_and_advances_when_more_steps_remain(self, monkeypatch):
        plan = [make_step(step=1, agent="diagnosis"), make_step(step=2, agent="dosage")]
        state = {
            "messages": [HumanMessage(content="my water is green")],
            "execution_plan": plan,
            "agent_results": {},
            "current_step": 0,
        }

        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"messages": [AIMessage(content="pH is low")]}
        monkeypatch.setattr(nodes, "get_agent_by_name", lambda name: fake_agent)

        result = nodes.orchestrator(state)

        assert result.goto == "orchestrator"  # queda pendiente el step 2
        assert result.update["current_step"] == 1

        step1_result = result.update["agent_results"]["step_1"]
        assert step1_result.output == "pH is low"
        assert step1_result.agent == "diagnosis"

        agent_call_input = fake_agent.invoke.call_args.args[0]
        assert "my water is green" in agent_call_input["messages"][0].content

    def test_last_step_routes_to_synthesizer(self, monkeypatch):
        plan = [make_step(step=1, agent="diagnosis")]
        state = {
            "messages": [HumanMessage(content="hi")],
            "execution_plan": plan,
            "agent_results": {},
            "current_step": 0,
        }

        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"messages": [AIMessage(content="done")]}
        monkeypatch.setattr(nodes, "get_agent_by_name", lambda name: fake_agent)

        result = nodes.orchestrator(state)

        assert result.goto == "synthesizer"
        assert result.update["current_step"] == 1

    def test_step_exception_is_captured_as_agent_result_error(self, monkeypatch):
        plan = [make_step(step=1, agent="dosage")]
        state = {
            "messages": [HumanMessage(content="hi")],
            "execution_plan": plan,
            "agent_results": {},
            "current_step": 0,
        }

        def boom(name):
            raise RuntimeError("agent exploded")

        monkeypatch.setattr(nodes, "get_agent_by_name", boom)

        result = nodes.orchestrator(state)
        step_result = result.update["agent_results"]["step_1"]

        assert step_result.error == "agent exploded"
        assert step_result.output == ""
        assert result.goto == "synthesizer"  # el error no rompe el flujo

    def test_preserves_previously_completed_results(self, monkeypatch):
        plan = [make_step(step=1, agent="diagnosis"), make_step(step=2, agent="dosage")]
        previous_results = {
            "step_1": make_result(step=1, agent="diagnosis", output="already done")
        }
        state = {
            "messages": [HumanMessage(content="hi")],
            "execution_plan": plan,
            "agent_results": previous_results,
            "current_step": 1,
        }

        fake_agent = MagicMock()
        fake_agent.invoke.return_value = {"messages": [AIMessage(content="dosage done")]}
        monkeypatch.setattr(nodes, "get_agent_by_name", lambda name: fake_agent)

        result = nodes.orchestrator(state)

        assert result.update["agent_results"]["step_1"].output == "already done"
        assert result.update["agent_results"]["step_2"].output == "dosage done"


# ================================================================
# synthesizer
# ================================================================

class TestSynthesizer:
    def test_oos_plan_uses_oos_instruction(self, monkeypatch):
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = AIMessage(content="Lo siento, eso está fuera de mi alcance.")
        monkeypatch.setattr(nodes, "_get_llm", lambda: fake_llm)

        plan = [make_step(step=1, agent="ooo", oos=True)]
        state = {"execution_plan": plan, "agent_results": {}, "detected_language": "es"}

        result = nodes.synthesizer(state)

        system_msg = fake_llm.invoke.call_args.args[0][0]
        assert "OUT OF SCOPE" in system_msg.content
        assert result["messages"][0].name == "Izel"
        assert result["messages"][0].content == "Lo siento, eso está fuera de mi alcance."

    def test_non_oos_plan_uses_normal_instruction_and_includes_raw_content(self, monkeypatch):
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = AIMessage(content="final answer")
        monkeypatch.setattr(nodes, "_get_llm", lambda: fake_llm)

        plan = [make_step(step=1, agent="diagnosis")]
        results = {"step_1": make_result(step=1, agent="diagnosis", output="pH is low")}
        state = {"execution_plan": plan, "agent_results": results, "detected_language": "en"}

        nodes.synthesizer(state)

        system_msg = fake_llm.invoke.call_args.args[0][0]
        assert "Do not add disclaimers about scope" in system_msg.content
        assert "pH is low" in system_msg.content
        assert "English" in system_msg.content

    def test_no_prior_content_uses_greeting_placeholder(self, monkeypatch):
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = AIMessage(content="hi there")
        monkeypatch.setattr(nodes, "_get_llm", lambda: fake_llm)

        plan = [make_step(step=1, agent="maintenance")]  # no oos, sin resultado registrado
        state = {"execution_plan": plan, "agent_results": {}, "detected_language": "es"}

        nodes.synthesizer(state)

        system_msg = fake_llm.invoke.call_args.args[0][0]
        assert "generate a warm greeting" in system_msg.content

    def test_unknown_language_code_falls_back_to_spanish(self, monkeypatch):
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = AIMessage(content="respuesta")
        monkeypatch.setattr(nodes, "_get_llm", lambda: fake_llm)

        plan = [make_step(step=1, agent="maintenance")]
        state = {"execution_plan": plan, "agent_results": {}, "detected_language": "fr"}

        nodes.synthesizer(state)

        system_msg = fake_llm.invoke.call_args.args[0][0]
        assert "Spanish (Latin American)" in system_msg.content

    def test_final_message_extracts_text_from_block_list_content(self, monkeypatch):
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = AIMessage(
            content=[{"text": "part one"}, {"text": "part two"}]
        )
        monkeypatch.setattr(nodes, "_get_llm", lambda: fake_llm)

        plan = [make_step(step=1, agent="maintenance")]
        state = {"execution_plan": plan, "agent_results": {}, "detected_language": "es"}

        result = nodes.synthesizer(state)

        assert result["messages"][0].content == "part one part two"
        assert result["messages"][0].name == "Izel"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))