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


# =====================================================================
# Presupuesto de razonamiento por rol
# =====================================================================
# Medido en traces reales, no elegido a ojo. El thinking latente es la
# mayor fuente de latencia y de varianza del turno, y se pierde con
# facilidad: basta que un nodo nuevo reutilice la factory equivocada.
#
# Trace 84fda7b8: `general` gastó 631 tokens de razonamiento para 270
# visibles. Trace b0496bf9: 725 para 189. Trace 6bb32a41: el AGENTE
# `general` (otro camino distinto del NODO) gastó 471 más, porque el
# primer arreglo solo cubrió uno de los dos.

def _budget(factory):
    return getattr(factory(), "thinking_budget", None)


class TestPresupuestoDeRazonamiento:
    """Quién puede razonar y quién no. Un cambio acá se paga en cada turno."""

    def test_el_planner_no_razona(self):
        # Clasifica contra un schema cerrado, y es el PRIMER nodo del turno:
        # su latencia la espera el usuario mirando una pantalla vacía.
        from src.config.llm import create_routing_llm
        assert _budget(create_routing_llm) == 0

    def test_el_synthesizer_no_razona(self):
        # Reescribe material que los sub-agentes ya resolvieron.
        from src.config.llm import create_synthesis_llm
        assert _budget(create_synthesis_llm) == 0

    def test_general_y_oos_no_razonan(self):
        # Redactan desde un `task` que el planner ya acotó, o declinan.
        from src.config.llm import create_direct_answer_llm
        assert _budget(create_direct_answer_llm) == 0

    def test_los_especialistas_si_razonan_pero_acotado(self):
        # Investigan: eligen qué tool llamar y qué evidencia vale.
        from src.config.llm import create_specialist_llm
        assert _budget(create_specialist_llm) == 1024

    def test_el_summarizer_conserva_el_thinking_dinamico(self):
        # Comprimir sin perder los hechos que importan sí es deliberación,
        # y corre fuera del camino crítico (solo sobre TOKEN_LIMIT).
        from src.config.llm import create_llm
        assert _budget(create_llm) is None


class TestTechosDeTiempo:
    def test_ninguna_llamada_puede_exceder_el_deadline_del_paso(self):
        """
        El peor caso de una llamada es timeout * max_retries. Estaba en
        120*3 = 360s contra un turno de 110: el cliente no acotaba nada, y
        como future.cancel() no mata el thread, una llamada colgada ocupaba
        un worker del _STEP_POOL durante todo ese tiempo.
        """
        from src.agent.nodes import STEP_DEADLINE_S
        from src.config import llm

        factories = [
            llm.create_llm, llm.create_routing_llm, llm.create_synthesizer_llm,
            llm.create_suggester_llm, llm.create_fallback_llm,
            llm.create_synthesis_llm, llm.create_specialist_llm,
            llm.create_direct_answer_llm,
        ]
        for f in factories:
            m = f()
            peor_caso = m.timeout * (m.max_retries + 1)
            assert peor_caso <= STEP_DEADLINE_S, (
                f"{f.__name__}: {peor_caso}s > STEP_DEADLINE_S ({STEP_DEADLINE_S}s)"
            )
