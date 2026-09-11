"""
El presupuesto de tools se aplica en el SCHEMA, no en la tool.

`_gate` en tools.py rechaza una llamada agotada en microsegundos, pero para
entonces el modelo ya gastó el round trip y el razonamiento para decidirla:
ese coste no lo recupera ningún mensaje de rechazo. ToolBudgetMiddleware
existe para que la tool agotada ni siquiera esté en el schema.

Sin cobertura hasta ahora, y es difícil de verificar a ojo: la evidencia de
que falla es un ToolMessage con BUDGET_EXHAUSTED enterrado en un trace.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.middleware import ToolBudgetMiddleware, _calls_so_far
from src.agent.tools import _TOOL_BUDGETS, expand_subgraph, search_seed_nodes, vector_search

try:
    from langchain.agents.middleware.types import ModelRequest
except ImportError:  # pragma: no cover - la ruta cambió entre versiones de langchain 1.x
    from langchain.agents.middleware import ModelRequest

TOOLS = [vector_search, search_seed_nodes, expand_subgraph]


def _llamada(nombre, i):
    return [
        AIMessage(content="", tool_calls=[{"name": nombre, "args": {}, "id": str(i)}]),
        ToolMessage(content="resultado", tool_call_id=str(i)),
    ]


def _ofrecidas(historial):
    """Qué tools deja pasar el middleware dado ese historial."""
    req = ModelRequest(
        model=None, messages=historial, system_message=None, tool_choice=None,
        tools=list(TOOLS), response_format=None, state={}, runtime=None,
        model_settings={},
    )
    visto = {}

    def handler(r):
        visto["tools"] = [t.name for t in r.tools]
        return AIMessage(content="ok")

    ToolBudgetMiddleware().wrap_model_call(req, handler)
    return visto["tools"]


class TestConteoDeLlamadas:
    def test_cuenta_por_nombre_de_tool(self):
        msgs = [HumanMessage(content="q"), *_llamada("vector_search", 1)]
        assert _calls_so_far(msgs) == {"vector_search": 1}

    def test_acumula_repeticiones(self):
        msgs = [HumanMessage(content="q"), *_llamada("expand_subgraph", 1),
                *_llamada("expand_subgraph", 2)]
        assert _calls_so_far(msgs)["expand_subgraph"] == 2

    def test_un_historial_sin_tools_no_cuenta_nada(self):
        assert _calls_so_far([HumanMessage(content="q"), AIMessage(content="hola")]) == {}


class TestRetiradaDelSchema:
    def test_al_principio_estan_todas(self):
        assert set(_ofrecidas([HumanMessage(content="q")])) == {t.name for t in TOOLS}

    def test_una_tool_de_presupuesto_1_desaparece_tras_usarla(self):
        historial = [HumanMessage(content="q"), *_llamada("vector_search", 1)]
        assert "vector_search" not in _ofrecidas(historial)

    def test_una_tool_de_presupuesto_2_sobrevive_al_primer_uso(self):
        # expand_subgraph tiene 2: un intento por slug y otro sobre los seeds
        # reales tras el miss.
        historial = [HumanMessage(content="q"), *_llamada("expand_subgraph", 1)]
        assert "expand_subgraph" in _ofrecidas(historial)

    def test_y_desaparece_al_agotarse(self):
        historial = [HumanMessage(content="q"), *_llamada("expand_subgraph", 1),
                     *_llamada("expand_subgraph", 2)]
        assert "expand_subgraph" not in _ofrecidas(historial)

    def test_el_caso_del_trace_5b88b65f(self):
        """
        Regresión concreta: tras gastar vector_search y search_seed_nodes, el
        modelo volvió a pedir las dos y el gate las rechazó en 0.001s. Fueron
        dos round trips completos (1.78s y 1.38s) por tools que no deberían
        haber estado en el schema.
        """
        historial = [
            HumanMessage(content="why does chlorine lose effectiveness as pH rises"),
            *_llamada("vector_search", 1),
            *_llamada("search_seed_nodes", 2),
            *_llamada("expand_subgraph", 3),
        ]
        assert _ofrecidas(historial) == ["expand_subgraph"]

    def test_sin_tools_disponibles_el_modelo_se_queda_sin_salida(self):
        """Agotarlo todo debe forzar una respuesta, no otro intento."""
        historial = [HumanMessage(content="q"), *_llamada("vector_search", 1),
                     *_llamada("search_seed_nodes", 2), *_llamada("expand_subgraph", 3),
                     *_llamada("expand_subgraph", 4)]
        assert _ofrecidas(historial) == []


class TestPresupuestos:
    def test_son_por_tool_y_no_intercambiables(self):
        # Gastar vector_search no libera una segunda de search_seed_nodes.
        historial = [HumanMessage(content="q"), *_llamada("vector_search", 1)]
        assert "search_seed_nodes" in _ofrecidas(historial)

    @pytest.mark.parametrize("tool,esperado", [
        ("vector_search", 1), ("search_seed_nodes", 1), ("expand_subgraph", 2),
    ])
    def test_valores_declarados(self, tool, esperado):
        assert _TOOL_BUDGETS[tool] == esperado
