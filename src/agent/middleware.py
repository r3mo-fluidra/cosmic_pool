from __future__ import annotations

import logging

from langchain.agents.middleware import AgentMiddleware

from .tools import _TOOL_BUDGETS, _EXECUTED_TOOL_CALLS

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

    def wrap_tool_call(self, request, handler):
        """
        bloquea la ejecución de la tool si ya se gastó su presupuesto en este step.
        """
        logger.info(
            "wrap_tool_call: name=%r scope=%s",
            getattr(getattr(request, "tool", None), "name", None)
            or (getattr(request, "tool_call", None) or {}).get("name"),
            "activo" if _EXECUTED_TOOL_CALLS.get() is not None else "None",
        )
        name = getattr(getattr(request, "tool", None), "name", None) \
            or (request.tool_call or {}).get("name")

        if not name:
            return handler(request)

        budget = _TOOL_BUDGETS.get(name)
        if budget is None:
            return handler(request)          # tool sin presupuesto declarado

        executed = _EXECUTED_TOOL_CALLS.get()
        if executed is None:
            return handler(request)          # sin scope de step activo

        used = executed.get(name, 0)
        if used >= budget:
            logger.info(
                "tool_budget: %s bloqueada en ejecución (%d/%d)",
                name, used, budget,
            )
            return (
                f"BUDGET_EXHAUSTED — `{name}` already ran {used}/{budget} "
                "times in this task. This call did not execute and neither "
                "will the next one. Report what you have already established, "
                "with its assumptions. A partial result with its provenance "
                "is a valid answer; stopping with nothing is not."
            )

        executed[name] = used + 1
        return handler(request)