"""
middleware.py
=============
Presupuesto de tools aplicado en la capa de binding, no en la tool.

`_gate` en tools.py rechaza la llamada DESPUÉS de que el modelo ya gastó
el razonamiento para decidirla. El trace 6e13ad4c lo muestra: 4.09s y 645
tokens de reasoning para pedir un `vector_search` que el gate contestó en
0ms con BUDGET_EXHAUSTED. Ese costo no lo recupera ningún mensaje de
rechazo, porque se paga antes de que la tool exista.

Este middleware filtra las tools agotadas de `request.tools` antes de cada
llamada al modelo. La tool no está en el schema, así que no se puede pedir.
Cuando no queda ninguna, el modelo no tiene otra salida que responder.

El conteo sale de `request.messages`, no del ContextVar de tools.py: el
middleware corre en el mismo thread que la llamada al modelo, así que no
depende de que el contexto cruce el _STEP_POOL de nodes.py.

Sin anotaciones de tipo a propósito: los nombres de ModelRequest/ModelResponse
se movieron entre versiones de langchain 1.x y un import roto acá tumbaría
el módulo entero de agentes.
"""

from __future__ import annotations

import logging

from langchain.agents.middleware import AgentMiddleware

from .tools import _TOOL_BUDGETS

logger = logging.getLogger(__name__)


def _calls_so_far(messages) -> dict[str, int]:
    """Cuántas veces se pidió cada tool en esta invocación del agente."""
    counts: dict[str, int] = {}
    for msg in messages:
        for call in getattr(msg, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                counts[name] = counts.get(name, 0) + 1
    return counts


class ToolBudgetMiddleware(AgentMiddleware):
    """
    Quita del scope las tools cuyo presupuesto ya se consumió.

    Una tool que no figura en _TOOL_BUDGETS no tiene límite. Deliberado:
    el middleware es inerte para MATH_TOOLS y para pool_general_knowledge,
    que llevan su propio control.
    """

    name = "tool_budget"

    def wrap_model_call(self, request, handler):
        used = _calls_so_far(request.messages)

        allowed = [
            t for t in request.tools
            if used.get(getattr(t, "name", None), 0) < _TOOL_BUDGETS.get(getattr(t, "name", None), 10**6)
        ]

        if len(allowed) != len(request.tools):
            dropped = [
                getattr(t, "name", "?") for t in request.tools if t not in allowed
            ]
            logger.info(
                "tool_budget: fuera de scope %s (usadas=%s) — quedan %d tools",
                dropped, used, len(allowed),
            )
            request = request.override(tools=allowed)

        return handler(request)