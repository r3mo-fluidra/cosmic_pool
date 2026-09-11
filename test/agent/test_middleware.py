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

        search_seed_nodes pasó a 2 después (un panel de siete parámetros no
        cabe en una búsqueda), así que acá se agota explícitamente.
        """
        historial = [
            HumanMessage(content="why does chlorine lose effectiveness as pH rises"),
            *_llamada("vector_search", 1),
            *_llamada("search_seed_nodes", 2),
            *_llamada("search_seed_nodes", 3),
            *_llamada("expand_subgraph", 4),
        ]
        assert _ofrecidas(historial) == ["expand_subgraph"]

    def test_sin_tools_disponibles_el_modelo_se_queda_sin_salida(self):
        """Agotarlo todo debe forzar una respuesta, no otro intento."""
        historial = [HumanMessage(content="q"), *_llamada("vector_search", 1),
                     *_llamada("search_seed_nodes", 2), *_llamada("search_seed_nodes", 3),
                     *_llamada("expand_subgraph", 4), *_llamada("expand_subgraph", 5)]
        assert _ofrecidas(historial) == []


class TestPresupuestos:
    def test_son_por_tool_y_no_intercambiables(self):
        # Gastar vector_search no libera una segunda de search_seed_nodes.
        historial = [HumanMessage(content="q"), *_llamada("vector_search", 1)]
        assert "search_seed_nodes" in _ofrecidas(historial)

    @pytest.mark.parametrize("tool,esperado", [
        ("vector_search", 1), ("search_seed_nodes", 2), ("expand_subgraph", 2),
    ])
    def test_valores_declarados(self, tool, esperado):
        assert _TOOL_BUDGETS[tool] == esperado

    def test_la_fuente_de_verdad_es_tool_budgets(self):
        """
        tools.py tenía una copia literal que PISABA el import del módulo
        canónico. Coincidían, así que no se notaba — hasta que se tocó
        tool_budgets.py y el cambio no llegó. middleware.py importa
        _TOOL_BUDGETS desde tools.py, o sea que consumía la copia.
        """
        from src.tool_budgets import RETRIEVAL_TOOL_BUDGETS
        assert _TOOL_BUDGETS is RETRIEVAL_TOOL_BUDGETS


# =====================================================================
# Techo de recursión: margen para las llamadas que el gate rechaza
# =====================================================================
# Trace 9caf725c. `chemistry` hizo 7 tool calls — 5 útiles y 2 rechazadas por
# presupuesto — y murió con TOOL_BUDGET_EXCEEDED. El usuario recibió "el
# sistema se quedó sin tiempo de proceso" teniendo cinco recuperaciones
# correctas en el historial.
#
# Cuentas: 7 tool calls son 15 pasos del grafo interno (modelo + tools por
# iteración, más la respuesta final). El límite era 14, calculado sobre el
# `tool_budget` declarado en el config (6) y no sobre la suma real de los caps
# por tool (5). Los dos se desincronizaron en cuanto se tocó uno.
#
# El presupuesto se aplica en dos capas: el middleware retira del schema, y
# `_gate` rechaza lo que aun así llegue. La segunda existe porque la primera
# no siempre alcanza — un modelo puede pedir una tool que no está en el
# esquema que se le pasó. Cada rechazo cuesta dos pasos sin aportar evidencia,
# así que el techo tiene que preverlos.

class TestTechoDeRecursion:
    def _limite(self, agente):
        from src.agent.nodes import _recursion_limit_for
        return _recursion_limit_for(agente)

    def test_el_caso_del_trace_ahora_cabe(self):
        # 7 tool calls = 7*2 + 1 = 15 pasos.
        assert self._limite("chemistry") >= 15

    def test_se_deriva_del_presupuesto_real_no_del_declarado(self):
        """
        chemistry declara tool_budget=6 y la suma de sus caps es 5. Calcular
        sobre el número equivocado fue lo que dejó el techo sin margen.
        """
        from src.prompts.prompts_sub_agents import AGENT_REGISTRY, CHEMISTRY
        from src.tool_budgets import RETRIEVAL_TOTAL
        declarado = AGENT_REGISTRY[CHEMISTRY].tool_budget
        assert self._limite("chemistry") != declarado * 2 + 2
        assert self._limite("chemistry") > RETRIEVAL_TOTAL * 2 + 1

    def test_absorbe_varias_llamadas_rechazadas(self):
        from src.agent.nodes import _REJECTED_CALL_MARGIN
        from src.tool_budgets import RETRIEVAL_TOTAL
        pasos_peor_caso = (RETRIEVAL_TOTAL + _REJECTED_CALL_MARGIN) * 2 + 1
        assert self._limite("chemistry") >= pasos_peor_caso

    def test_todos_los_especialistas_de_retrieval_comparten_techo(self):
        limites = {self._limite(a) for a in
                   ["chemistry", "equipment", "hydraulics", "safety", "records"]}
        assert len(limites) == 1, "usan las mismas tres tools, mismo techo"

    def test_math_usa_su_propio_presupuesto(self):
        # No tiene caps por tool: su catálogo es determinista y el config es
        # la única cifra disponible.
        assert self._limite("math") != self._limite("chemistry")

    def test_un_agente_desconocido_no_revienta(self):
        assert self._limite("no_existe") > 0
