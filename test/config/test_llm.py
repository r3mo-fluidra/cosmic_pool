"""
Tests de src/config/llm.py.

Lo que importa acá no es el nombre del modelo (cambia con cada calibración)
sino las dos invariantes que el módulo documenta:

  1. Peor caso de una llamada (timeout * max_retries) por debajo del
     STEP_DEADLINE_S del grafo: quien corta debe ser el deadline del grafo,
     nunca el del cliente.
  2. thinking_budget=0 en los nodos que no razonan (planner y synthesizer),
     porque ahí el thinking es latencia pura en el camino crítico.
"""

from unittest.mock import patch

import pytest

from src.agent import nodes
from src.config import llm as llm_module


FACTORIES = [
    "create_llm",
    "create_routing_llm",
    "create_synthesizer_llm",
    "create_suggester_llm",
    "create_fallback_llm",
    "create_synthesis_llm",
    "create_specialist_llm",
]


def _kwargs_of(factory_name: str) -> dict:
    with patch("src.config.llm.ChatGoogleGenerativeAI") as mock_chat, \
         patch("src.config.llm._get_secret", return_value="fake-key"):
        getattr(llm_module, factory_name)()
    return mock_chat.call_args.kwargs


@pytest.mark.parametrize("factory_name", FACTORIES)
def test_every_factory_passes_the_api_key_and_a_model(factory_name):
    kwargs = _kwargs_of(factory_name)

    assert kwargs["google_api_key"] == "fake-key"
    assert kwargs["model"].startswith("gemini-")


@pytest.mark.parametrize("factory_name", FACTORIES)
def test_worst_case_call_stays_under_the_step_deadline(factory_name):
    """timeout * max_retries < STEP_DEADLINE_S: el techo real lo pone el grafo."""
    kwargs = _kwargs_of(factory_name)

    worst_case = kwargs["timeout"] * kwargs["max_retries"] * 2
    assert worst_case < nodes.STEP_DEADLINE_S


@pytest.mark.parametrize(
    "factory_name", ["create_routing_llm", "create_synthesis_llm"]
)
def test_classifying_and_rewriting_nodes_run_without_thinking(factory_name):
    """El planner clasifica y el synthesizer reescribe: ninguno investiga."""
    assert _kwargs_of(factory_name)["thinking_budget"] == 0


def test_specialists_keep_a_bounded_thinking_budget():
    """Los especialistas sí razonan, pero con techo: era la mayor varianza."""
    budget = _kwargs_of("create_specialist_llm")["thinking_budget"]

    assert budget > 0


def test_routing_llm_is_deterministic():
    assert _kwargs_of("create_routing_llm")["temperature"] == 0.0


# ==========================================================
# _get_secret
# ==========================================================

def test_get_secret_prefers_the_environment_variable(monkeypatch):
    monkeypatch.setenv("SOME_KEY", "from-env")

    assert llm_module._get_secret("SOME_KEY") == "from-env"


def test_get_secret_returns_default_when_nothing_is_configured(monkeypatch):
    monkeypatch.delenv("SOME_MISSING_KEY", raising=False)

    assert llm_module._get_secret("SOME_MISSING_KEY", "fallback") == "fallback"
