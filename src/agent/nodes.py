from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    HumanMessage,
    BaseMessage,
    RemoveMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send
import logging
from langfuse import observe, get_client
from typing import List, Literal

import contextvars
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
import concurrent.futures

from .state import PoolAgentState, ExecutionStep, AgentResult
from ..prompts.prompts import PLANNER_PROMPT, SYNTHESIZER_PROMPT, SUGGESTER_PROMPT
from .chains import create_planner_chain
from ..config.llm import create_llm, create_suggester_llm
from .agents import get_agent_by_name
from .gates import (
    math_inputs_present,
    missing_inputs_result
)
from ..prompts.prompts_sub_agents import MATH
from ..prompts.prompts import GENERAL_PROMPT , OOS_PROMPT
# Graph context
from ..graph_context.response_contracts import (
    SynthesizerOutput, get_contract, resolve_archetype,
    usable_results, DetailSection, agents_from_results
)
from ..graph_context.response_validator import enforce_contract, fallback_payload
from ..graph_context.suggestions import (
    SUPERNODES,
    Suggestion,
    SuggesterOutput,
    answer_ends_with_question,
    apply_gates_with_report,
    roster_text,
)
from ..graph_context.turn_cache import reset_turn
from ..graph_context.turn_cache import get_touched
# ================================================================
# CONFIGURATION
# ================================================================

logger = logging.getLogger(__name__)

TOKEN_LIMIT = 25000
MESSAGES_TO_KEEP = 6
_SUGGESTER_DEADLINE_S = 1.2


# ================================================================
# ROUTING: planner → general | oos | orchestrator
# ================================================================

GENERAL_AGENT = "general"

# Roster válido para recuperar un MISROUTE. Sin whitelist, un nombre
# alucinado por el LLM explota adentro de get_agent_by_name en run_step.
_MISROUTE_AGENTS = frozenset({
    "contamination", "safety", "chemistry", "compliance", "general",
})

_MISROUTE_RE = re.compile(r"^\s*MISROUTE:\s*([A-Za-z_]+)\s*(.*)", re.DOTALL)

def _normalize_agent(agent) -> str:
    """AgentName puede ser str, Enum o None."""
    if agent is None:
        return ""
    # Enum → value; str → str
    value = getattr(agent, "value", agent)
    return str(value).strip().lower()

def _route_from_plan(execution_plan: list[ExecutionStep]) -> str:
    if not execution_plan:
        return "orchestrator"  # el orchestrator debe mirar missing_inputs en el state

    if len(execution_plan) != 1:
        return "orchestrator"

    step = execution_plan[0]
    agent = _normalize_agent(step.assigned_agent)

    if bool(step.oos) or agent == "oos":
        return "oos"
    if agent == "general":
        return "general"
    return "orchestrator"


def _last_human_text(state: PoolAgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "type", None) == "human":
            return _extract_text(msg.content)
    return ""


def _direct_answer(state: PoolAgentState, system_prompt: str) -> tuple[str, str | None]:
    """
    Una sola llamada al LLM: sin tools, sin ReAct loop, sin ThreadPool.
    Estos dos nodos no hacen retrieval, así que el overhead de _run_step
    (create_react_agent + iteración de tool calls) es puro costo.
    """
    plan = state.get("execution_plan") or []
    user_message = _last_human_text(state)
    task = plan[0].task if plan else user_message
    language = _LANGUAGE_MAP.get(state.get("detected_language", "es"), _LANGUAGE_MAP["es"])

    try:
        raw = _get_llm().invoke([
            SystemMessage(content=f"{system_prompt}\n\nRespond in: {language}"),
            HumanMessage(content=f"Task: {task}\n\nUser context: {user_message}"),
        ])
        return _extract_text(raw.content), None
    except Exception as exc:
        err = str(exc).strip() or exc.__class__.__name__
        if exc.__class__.__name__ in _INFRA_EXC_NAMES and not _INFRA_CODE_RE.match(err):
            err = f"{exc.__class__.__name__}: {err}"
        return "", err

# ================================================================
# LAZY LLM + PLANNER CHAIN
# ================================================================

_llm = None
_planner_chain = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = create_llm()
    return _llm

def _get_planner_chain():
    global _planner_chain
    if _planner_chain is None:
        _planner_chain = create_planner_chain(_get_llm())
    return _planner_chain



# ---------------------------------------------------------------------------
# Clasificación de errores
# ---------------------------------------------------------------------------
 
STEP_DEADLINE_S = 25.0    # techo por sub-agente
TURN_DEADLINE_S = 60.0    # techo por turno completo
MIN_STEP_BUDGET_S = 8.0   # si queda menos que esto, no arranques otro paso
 
# Pool dedicado: no compartir con el executor por defecto de LangGraph.
_STEP_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="run_step")
 


_INFRA_CODE_RE = re.compile(r"^\s*(429|500|502|503|504)\b")
 
_INFRA_NAMES = (
    "DEADLINE_EXCEEDED",
    "UNAVAILABLE",
    "RESOURCE_EXHAUSTED",
    "INTERNAL",
    "STEP_DEADLINE_EXCEEDED",
    "TURN_DEADLINE_EXCEEDED",
    "UPSTREAM_INFRA_FAILURE",
)
 
_INFRA_EXC_NAMES = (
    "DeadlineExceeded",
    "ServiceUnavailable",
    "ResourceExhausted",
    "InternalServerError",
    "TooManyRequests",
    "ReadTimeout",
    "ConnectTimeout",
    "APITimeoutError",
    "APIConnectionError",
)
 
# Prefijos de error que NO son fallo del proveedor: son contratos de negocio.
# Un paso con MISSING_INPUTS "falló" pero el sistema está sano.
_SOFT_ERROR_PREFIXES = ("MISSING_INPUTS", "CANNOT_COMPUTE", "NO_GRAPH_COVERAGE")
 

# ================================================================
# HELPERS
# ================================================================

def is_infra_error(err: str | None, exc: BaseException | None = None) -> bool:
    """True solo para fallos del proveedor / timeouts, no para contratos de negocio."""
    if exc is not None and exc.__class__.__name__ in _INFRA_EXC_NAMES:
        return True
    if not err:
        return False
    if err.startswith(_SOFT_ERROR_PREFIXES):
        return False
    return bool(_INFRA_CODE_RE.match(err)) or any(n in err for n in _INFRA_NAMES)
 

def _field(result, name: str, default=None):
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)
 

def _status(result) -> str:
    """'ok' | 'failed' | 'skipped'. Usa result.status si existe, si no lo infiere."""
    explicit = _field(result, "status")
    if explicit in ("ok", "failed", "skipped"):
        return explicit
    err = _field(result, "error")
    if not err:
        return "ok" if _field(result, "output") else "failed"
    if str(err).startswith("SKIPPED_"):
        return "skipped"
    return "failed"
 
 
def _step_num(key: str) -> int | None:
    try:
        return int(str(key).split("_", 1)[1])
    except (IndexError, ValueError):
        return None
 

def _skipped_result(step, reason: str):
    from .state import AgentResult  # ajustá el import
 
    return AgentResult(
        agent=step.assigned_agent,
        step=step.step,
        output="",
        sources=[],
        error=reason,
        status="skipped",
    )
 

def _remaining_budget(state) -> float:
    started = state.get("turn_started_at")
    if not started or started < time.time() - 3600:
        # stale de un turno viejo, o nunca se seteó -> no confiar en el budget
        return TURN_DEADLINE_S
    return TURN_DEADLINE_S - (time.time() - float(started))

def _flatten(content) -> str:
    """Tu lógica actual de parseo, ahora solo para el camino de fallback."""
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") for item in content
            if isinstance(item, dict) and item.get("text", "").strip()
        ).strip()
    return str(content).strip()

_SERVICE_UNAVAILABLE_TEXT = {
    "es": "Lo siento, nuestro asistente está experimentando una interrupción temporal por alta demanda. Probá de nuevo en unos minutos.",
    "en": "Sorry, our assistant is experiencing a temporary service interruption due to high demand. Please try again in a few minutes.",
}

def static_service_unavailable_payload(output_cls, language_code: str):
    text = _SERVICE_UNAVAILABLE_TEXT.get(language_code, _SERVICE_UNAVAILABLE_TEXT["es"])
    return output_cls(
        archetype="conversational",
        answer=text,
        actions=[],
        safety=None,
        details=[],
    )

def _attach_sources(payload: SynthesizerOutput, results: list) -> None:
    seen, srcs = set(), []
    for r in results:
        for s in r.sources:
            if s not in seen:
                seen.add(s); srcs.append(s)
    if srcs:
        payload.details.append(
            DetailSection(label="Fuentes", body="\n".join(f"- {s}" for s in srcs))
        )

def _extract_text(content) -> str:
    """Normalise LLM content to plain text regardless of its shape."""
    if isinstance(content, list):
        return " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("text", "").strip()
        ).strip()
    return str(content).strip()


def _run_step(step: ExecutionStep, user_message: str) -> AgentResult:
    agent = get_agent_by_name(step.assigned_agent)

    agent_input = {
        "messages": [
            HumanMessage(
                content=(
                    f"Task: {step.task}\n\n"
                    f"User context: {user_message}"
                )
            )
        ]
    }

    result = agent.invoke(agent_input)

    output_text = ""
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            output_text = _extract_text(msg.content)
            break

    return AgentResult(
        agent=step.assigned_agent,
        step=step.step,
        output=output_text,
    )

_LANGUAGE_MAP: dict[str, str] = {
    "es": (
        "Spanish (Latin American). "
        "Every single word must be in Spanish. Translate anything that is not."
    ),
    "en": (
        "English. "
        "Every single word must be in English. Translate anything that is not."
    ),
}

_OOS_INSTRUCTION_ACTIVE = (
    "IMPORTANT — OUT OF SCOPE RESPONSE: The user's request falls outside your area of "
    "expertise as a Pool Assistant. Do NOT attempt to answer the question. Instead, "
    "acknowledge the topic briefly, explain politely that it is outside your scope, "
    "and invite the user to ask any pool or spa related question."
)

_OOS_INSTRUCTION_INACTIVE = (
    "Provide a complete, helpful, and technically accurate response based on the raw "
    "content supplied. Do not add disclaimers about scope; the content is fully on-topic."
)


def _is_oos(execution_plan: list[ExecutionStep]) -> bool:
    return len(execution_plan) == 1 and execution_plan[0].oos


def _build_raw_content(agent_results: dict[str, AgentResult]) -> str:
    if not agent_results:
        return ""

    sorted_results: list[AgentResult] = sorted(
        agent_results.values(),
        key=lambda r: r.step,
    )

    sections: list[str] = []
    for result in sorted_results:
        if result.error:
            sections.append(
                f"[Step {result.step} — {result.agent}] ERROR: {result.error}"
            )
        elif result.output:
            sections.append(
                f"[Step {result.step} — {result.agent}]\n{result.output}"
            )

    return "\n\n".join(sections)


def estimated_tokens(messages: List[BaseMessage]) -> int:
    total = 0
    for msg in messages:
        if isinstance(msg.content, str):
            total += len(msg.content) // 4
        elif isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict):
                    total += len(block.get("text", "")) // 4
    return total

def should_suggest(state: PoolAgentState) -> bool:
    """
    Lógica pura, cero llamadas al LLM. Corre antes de cualquier gasto de
    cuota. El orden es intencional: lo más barato y más frecuente primero.
    """
    if state.get("archetype") in _SUPPRESSED_ARCHETYPES:
        return False
 
    if state.get("error"):
        return False
 
    # Sin agentes usables no hay contenido del cual predecir nada.
    if not agents_from_results(state.get("agent_results") or {}):
        return False
 
    # El usuario ya ignoró chips dos turnos seguidos: dejar de ofrecerlos.
    if state.get("ignored_chip_streak", 0) >= _IGNORED_CHIP_LIMIT:
        return False
 
    # El synthesizer ya cerró con una pregunta propia; un chip encima es ruido.
    if answer_ends_with_question(state):
        return False
 
    return True

def _build_answered_summary(state: PoolAgentState) -> str:
    """Solo el tier 1: es lo que el usuario efectivamente leyó."""
    response = state.get("response")
    if response is None:
        return "(sin respuesta disponible)"
    return response.tier1_markdown()
 
 
def _unconsumed_entities(state: PoolAgentState, thread_id: str) -> List:
    """
    Nodos que el retrieval tocó este turno pero que la respuesta no cubrió.
 
    Doble filtro:
      1. Anti-hub: los supernodos nunca son buen material de chip.
      2. Redundancia: si el nombre ya aparece en la respuesta, está cubierto.
 
    Es deliberadamente conservador — preferimos perder un candidato válido
    a alimentar el prompt con algo ya respondido.
    """
    touched = get_touched(thread_id)
    if not touched:
        return []
 
    answer = _build_answered_summary(state).lower()
 
    return [
        n for n in touched
        if n.id.lower() not in SUPERNODES
        and n.name.lower().replace("_", " ") not in answer
    ]
 
 
def _format_entities(nodes: List) -> str:
    if not nodes:
        return "(ninguna)"
    return "\n".join(f"- {n.id} | {n.name} | {n.label}" for n in nodes)

_SUPPRESSED_ARCHETYPES = frozenset({"critical", "conversational", "oos"})
 
_IGNORED_CHIP_LIMIT = 2
 
_suggester_llm = None
 
 
def _get_llm_suggester():
    """Lazy: no construir el cliente si el gate de supresión corta antes."""
    global _suggester_llm
    if _suggester_llm is None:
        _suggester_llm = create_suggester_llm()
    return _suggester_llm

# ================================================================
# CONTEXT NODE
# ================================================================
@observe(as_type='agent', name="Context Node")
def build_context_node(
    state: PoolAgentState,
) -> Command[Literal["summarize_memory_node", "planner"]]:

    next_node: Literal["summarize_memory_node", "planner"] = (
        "summarize_memory_node"
        if estimated_tokens(state["messages"]) > TOKEN_LIMIT
        else "planner"
    )

    return Command(
        update={"turn_started_at": time.time()},
        goto=next_node,
    )

# ================================================================
# SUMMARIZE MEMORY NODE
# ================================================================
@observe(as_type='agent', name="Sumarize Node")
def summarize_memory_node(state: PoolAgentState) -> Command[Literal["planner"]]:
    messages = state.get("messages", [])
    previous_summary = state.get("conversation_summary", "")

    if len(messages) <= MESSAGES_TO_KEEP:
        return Command(goto="planner")

    if previous_summary:
        prompt_text = (
            f"Previous conversation summary:\n{previous_summary}\n\n"
            "Extend this summary by incorporating the new messages. "
            "Be concise, but preserve key facts, decisions, and important context."
        )
    else:
        prompt_text = (
            "Summarize the following conversation concisely. "
            "Preserve key facts, decisions, and important context."
        )

    # ✅ Lazy — solo se crea cuando se invoca el nodo
    new_summary_msg = _get_llm().invoke(
        messages + [HumanMessage(content=prompt_text)]
    )

    messages_to_delete = messages[:-MESSAGES_TO_KEEP]
    removals = [RemoveMessage(id=m.id) for m in messages_to_delete]

    return Command(
        update={
            "conversation_summary": new_summary_msg.content,
            "messages": removals,
        },
        goto="planner",
    )

# ================================================================
# PLANNER NODE
# ================================================================

@observe(as_type='agent', name="Planner Node")
def planner(state: PoolAgentState, config: RunnableConfig):
    thread_id = config["configurable"]["thread_id"]
    reset_turn(thread_id)
    user_input = state["messages"][-1].content

    agent_messages = [
        m for m in state["messages"]
        if isinstance(m, AIMessage) and getattr(m, "name", None) == "Marlin"
    ]

    last_agent_msg = agent_messages[-1].content if agent_messages else ""
    if isinstance(last_agent_msg, list):
        last_agent_msg = " ".join(
            i.get("text", "") for i in last_agent_msg if isinstance(i, dict)
        ).strip()

    context_for_planner = (
        f"[Last agent message]: {last_agent_msg}\n"
        f"[User reply]: {user_input}"
        if last_agent_msg
        else user_input
    )

    fallback_language = state.get("detected_language") or "es"

    try:
        # ✅ Lazy — planner chain se inicializa solo aquí
        plan = _get_planner_chain().invoke([
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user",   "content": context_for_planner},
        ])
    except Exception as e:
        get_client().update_current_span(
            level="WARNING",
            status_message=f"planner_llm_failed: {e}",
        )
        fallback_step = ExecutionStep(
            step=1, task=user_input, assigned_agent="general", oos=False
        )
        return Command(
            update={
                "detected_language": fallback_language,
                "execution_plan": [fallback_step],
                "current_step": 0,
                "agent_results": None,
                "planner_error": str(e),
            },
            goto=_route_from_plan([fallback_step]),          
        )

    detected_language = plan.detected_language or fallback_language

    return Command(
        update={
            "detected_language": detected_language,
            "execution_plan": plan.execution_plan,
            "current_step": 0,
            "agent_results": None,
        },
        goto=_route_from_plan(plan.execution_plan),
    )

# ================================================================
# ORCHESTRATOR NODE
# ================================================================

# @observe(as_type="agent", name="Orchestrator Node")
# def orchestrator(state: PoolAgentState) -> Command:
#     execution_plan = state.get("execution_plan", [])
#     agent_results  = dict(state.get("agent_results") or {})
#     current_idx    = state.get("current_step", 0)

#     if not execution_plan:
#         return Command(
#             update={"error": "execution_plan is empty; cannot orchestrate."},
#             goto="synthesizer",
#         )

#     if current_idx >= len(execution_plan):
#         return Command(goto="synthesizer")

#     step     = execution_plan[current_idx]
#     step_key = f"step_{step.step}"

#     messages = state.get("messages", [])
#     user_message = ""
#     for msg in reversed(messages):
#         if hasattr(msg, "type") and msg.type == "human":
#             user_message = _extract_text(msg.content)
#             break

#     try:
#         agent_result = _run_step(step, user_message)
#     except Exception as exc:
#         agent_result = AgentResult(
#             agent=step.assigned_agent,
#             step=step.step,
#             output="",
#             error=str(exc),
#         )

#     agent_results[step_key] = agent_result

#     next_idx = current_idx + 1
#     goto = "orchestrator" if next_idx < len(execution_plan) else "synthesizer"

#     return Command(
#         update={
#             "agent_results": agent_results,
#             "current_step": next_idx,
#         },
#         goto=goto,
#     )
def _deps_ok(step, results) -> tuple[bool, str|None]:
    for dep in step.get("depends_on", []):
        r = results.get(f"step_{dep}")
        if r is None:
            return False, f"dependency step_{dep} did not run"
        if r.get("error") or not r.get("output"):
            return False, f"dependency step_{dep} failed: {r.get('error','empty output')}"
    return True, None



@observe(as_type="agent", name="Orchestrator Node")
def orchestrator(state: PoolAgentState) -> Command:
    execution_plan = state.get("execution_plan", [])
    agent_results = state.get("agent_results") or {}
 
    if not execution_plan:
        return Command(
            update={"error": "execution_plan is empty; cannot orchestrate."},
            goto="synthesizer",
        )
 
    # --- 1. Clasificar lo ya ejecutado: éxito != "presente en agent_results" ---
    ok_steps: set[int] = set()
    failed_steps: set[int] = set()
    for key, result in agent_results.items():
        num = _step_num(key)
        if num is None:
            continue
        if _status(result) == "ok":
            ok_steps.add(num)
        else:
            failed_steps.add(num)
 
    done = ok_steps | failed_steps
    pending = [s for s in execution_plan if s.step not in done]
 
    if not pending:
        return Command(goto="synthesizer")
 
    # --- 2. Circuit breaker: un 504/503/429 no se recupera dentro del turno ---
    infra_hit = next(
        (
            (num, _field(r, "error"))
            for key, r in agent_results.items()
            if (num := _step_num(key)) is not None
            and is_infra_error(_field(r, "error"))
        ),
        None,
    )
    if infra_hit:
        failed_num, failed_err = infra_hit
        reason = f"SKIPPED_UPSTREAM_INFRA_FAILURE: step_{failed_num} -> {failed_err}"
        return Command(
            update={
                "agent_results": {
                    f"step_{s.step}": _skipped_result(s, reason) for s in pending
                },
                "error": f"UPSTREAM_INFRA_FAILURE at step_{failed_num}",
            },
            goto="synthesizer",
        )
 
    # --- 3. Presupuesto de turno ---
    remaining = _remaining_budget(state)
    if remaining <= MIN_STEP_BUDGET_S:
        reason = f"SKIPPED_TURN_DEADLINE_EXCEEDED: {remaining:.1f}s left"
        return Command(
            update={
                "agent_results": {
                    f"step_{s.step}": _skipped_result(s, reason) for s in pending
                },
                "error": "TURN_DEADLINE_EXCEEDED",
            },
            goto="synthesizer",
        )
 
    # --- 4. Cascada de dependencias (punto fijo, resuelve cadenas 1->2->3) ---
    blocked: dict[str, object] = {}
    blocked_nums: set[int] = set()
    poisoned = set(failed_steps)
 
    changed = True
    while changed:
        changed = False
        for s in pending:
            if s.step in blocked_nums:
                continue
            bad = [d for d in (s.depends_on or []) if d in poisoned]
            if not bad:
                continue
            src = bad[0]
            src_err = _field(agent_results.get(f"step_{src}"), "error", "upstream skipped")
            blocked[f"step_{s.step}"] = _skipped_result(
                s, f"SKIPPED_DEPENDENCY_FAILED: step_{src} -> {src_err}"
            )
            blocked_nums.add(s.step)
            poisoned.add(s.step)
            changed = True
 
    runnable = [
        s
        for s in pending
        if s.step not in blocked_nums
        and all(d in ok_steps for d in (s.depends_on or []))
    ]
    waiting = [s for s in pending if s.step not in blocked_nums and s not in runnable]
 
    update: dict = {"agent_results": blocked} if blocked else {}
 
    if runnable:
        messages = state.get("messages", [])
        user_message = ""
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "human":
                user_message = _extract_text(msg.content)
                break
 
        # El paso nunca puede durar más que lo que queda del turno.
        step_budget = max(MIN_STEP_BUDGET_S, min(STEP_DEADLINE_S, remaining))
 
        return Command(
            update=update,
            goto=[
                Send(
                    "run_step",
                    {
                        "step": s,
                        "user_message": user_message,
                        "deadline_s": step_budget,
                    },
                )
                for s in runnable
            ],
        )
 
    if waiting:
        # Pendientes sin bloquear y sin poder correr -> ciclo o depends_on inválido.
        update["error"] = (
            f"Deadlock in execution_plan: {[s.step for s in waiting]} blocked."
        )
 
    return Command(update=update, goto="synthesizer")
 
 
# ---------------------------------------------------------------------------
# Run step
# ---------------------------------------------------------------------------

def _run_with_deadline(fn, deadline_s: float, *args):
    """Ejecuta fn con techo de wall-clock, propagando el contexto de Langfuse.
 
    copy_context() es obligatorio: sin él, los spans que _run_step abre dentro
    del thread pierden el parent OTel y aparecen sueltos en el trace.
    """
    ctx = contextvars.copy_context()
    future = _STEP_POOL.submit(ctx.run, fn, *args)
    try:
        return future.result(timeout=deadline_s)
    except FuturesTimeout:
        future.cancel()  # no mata el thread en curso; ver nota sobre timeout del cliente
        raise
 
 
@observe(as_type="agent", name="Run Step Node")
def run_step_node(payload: dict) -> Command:
    from .state import AgentResult  # ajustá el import
 
    step = payload["step"]
    user_message = payload["user_message"]
    deadline_s = float(payload.get("deadline_s", STEP_DEADLINE_S))
    step_key = f"step_{step.step}"
 
    if step.assigned_agent == MATH and not math_inputs_present(user_message):
        # Gate determinista: sin ningún dígito en el turno, ninguna fórmula
        # del catálogo tiene con qué calcular.
        return Command(
            update={"agent_results": {step_key: missing_inputs_result(step, user_message)}},
            goto="orchestrator",
        )
 
    started = time.monotonic()
    try:
        agent_result = _run_with_deadline(_run_step, deadline_s, step, user_message)
 
    except FuturesTimeout:
        agent_result = AgentResult(
            agent=step.assigned_agent,
            step=step.step,
            output="",
            sources=[],
            error=f"STEP_DEADLINE_EXCEEDED after {deadline_s:.0f}s",
            status="failed",
        )
 
    except Exception as exc:
        err = str(exc).strip() or exc.__class__.__name__
        if exc.__class__.__name__ in _INFRA_EXC_NAMES and not _INFRA_CODE_RE.match(err):
            err = f"{exc.__class__.__name__}: {err}"
        agent_result = AgentResult(
            agent=step.assigned_agent,
            step=step.step,
            output="",
            sources=[],
            error=err,
            status="failed",
        )
   
    return Command(
        update={"agent_results": {step_key: agent_result}},
        goto="orchestrator",
    )
# ================================================================
# SYNTHESIZER NODE
# ================================================================

@observe(as_type="generation", name="Synthesizer Node")
def synthesizer(state: PoolAgentState) -> dict:
    execution_plan: list[ExecutionStep] = state.get("execution_plan", [])
    agent_results:  dict[str, AgentResult] = state.get("agent_results") or {}
    language_code:  str = state.get("detected_language", "es")

    is_oos = _is_oos(execution_plan)
    oos_instruction = _OOS_INSTRUCTION_ACTIVE if is_oos else _OOS_INSTRUCTION_INACTIVE
    language_instruction = _LANGUAGE_MAP.get(language_code, _LANGUAGE_MAP["es"])

    usable    = usable_results(agent_results)
    agents    = [r.agent for r in usable]
    # archetype ya viene resuelto del orchestrator (punto de fan-out)
    archetype = state.get("archetype", "conversational")
    contract  = get_contract(archetype)

    raw_content = _build_raw_content(agent_results)
    if not raw_content:
        raw_content = "(no prior content — generate a warm greeting and offer help)"
        # fallback defensivo — no debería dispararse si el orchestrator hizo su parte,
        # pero cubre la rama de error / cualquier estado inconsistente
        archetype, contract = "conversational", get_contract("conversational")

    system_content = SYNTHESIZER_PROMPT.format(
        archetype=archetype,
        shape=contract["shape"],
        budget=contract["budget"],
        detail_labels=", ".join(contract["details"]) or "(ninguna)",
        oos_instruction=oos_instruction,
        language=language_instruction,
        raw_content=raw_content,
    )

    try:
        payload = _get_llm().with_structured_output(SynthesizerOutput).invoke([
            SystemMessage(content=system_content),
            HumanMessage(content="Generate the final refined response now."),
        ])
        payload, report = enforce_contract(payload, contract, agents,
                                           detail_cls=DetailSection)
        validation = report.to_dict()
    except Exception as exc:
        raw = _get_llm().invoke([
            SystemMessage(content=system_content),
            HumanMessage(content="Generate the final refined response now."),
        ])
        payload = fallback_payload(_flatten(raw.content), SynthesizerOutput)
        validation = {"fallback": True, "reason": str(exc)}

    _attach_sources(payload, usable)

    return {
        "archetype": archetype,
        "response": payload,
        "validation": validation,
        "messages": [AIMessage(content=payload.tier1_markdown(), name="Marlin")],
    }

# ================================================================
# SUGGESTER NODE
# ================================================================
def suggester(state: PoolAgentState, config: RunnableConfig) -> dict:
    """
    Rama paralela del fan-out del orchestrator. Lee agent_results y
    archetype; no depende del synthesizer ni lo bloquea.

    Devuelve siempre la clave "suggestions" — nunca la omite, para que
    el frontend pueda distinguir "no hubo chips" de "el nodo no corrió".
    """
    if not should_suggest(state):
        return {"suggestions": []}

    thread_id = config.get("configurable", {}).get("thread_id", "")

    unconsumed = _unconsumed_entities(state, thread_id)
    if not unconsumed:
        # Sin entidades libres el LLM solo puede inventar. Ahorramos la
        # llamada: es el caso más común en turnos sin retrieval.
        return {"suggestions": []}

    language_code = state.get("detected_language", "es")
    language = "español" if language_code == "es" else "English"
    answer_text = _build_answered_summary(state)

    system_content = SUGGESTER_PROMPT.format(
        language=language,
        roster=roster_text(),
        answered_summary=answer_text,
        unconsumed_entities=_format_entities(unconsumed),
    )

    messages = [
        SystemMessage(content=system_content),
        HumanMessage(content="Generá las sugerencias ahora, o ninguna."),
    ]

    try:
        chain = _get_llm_suggester().with_structured_output(SuggesterOutput)
        # El deadline lo impone el executor, no el cliente: la API exige
        # timeout >= 10s, y 10s bloqueando la superstep rompe el invariante
        # "NUNCA bloquea al synthesizer".
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            payload: SuggesterOutput = ex.submit(chain.invoke, messages).result(
                timeout=_SUGGESTER_DEADLINE_S
            )
    except Exception as exc:
        # Todo se degrada igual: 429, TimeoutError del executor, o structured
        # output inválido. Se loguea para poder ver la distribución real de
        # fallas en Langfuse, pero nunca se propaga: un chip opcional no rompe
        # el turno del usuario.
        logger.warning(
            "suggester degraded to []: %s: %s", type(exc).__name__, exc
        )
        return {"suggestions": []}

    raw: List[Suggestion] = payload.suggestions or []
    gated, report = apply_gates_with_report(raw, answer_text)

    # El reporte va al log, no al state: es telemetría de calidad del prompt
    # (paso 9: "cuál gate descarta más"), no algo que el frontend consuma.
    if report["input"] != report["output"]:
        logger.info("suggester gates: %s", report)

    return {"suggestions": gated}

@observe(as_type="agent", name="General Node")
def general(state: PoolAgentState) -> Command[Literal["synthesizer"]]:
    plan = state.get("execution_plan") or []
    step_num = plan[0].step if plan else 1

    text, err = _direct_answer(state, GENERAL_PROMPT)

    result = AgentResult(
        agent=GENERAL_AGENT,
        step=step_num,
        output=text,
        sources=[],
        error=err,
        status="ok" if text and not err else "failed",
    )

    return Command(
        update={
            "agent_results": {f"step_{step_num}": result},
            "archetype": "conversational",
        },
        goto="synthesizer",
    )

@observe(as_type="agent", name="OOS Node")
def oos(state: PoolAgentState) -> Command[Literal["synthesizer"]]:
    plan = state.get("execution_plan") or []
    step_num = plan[0].step if plan else 1

    text, err = _direct_answer(state, OOS_PROMPT)

    result = AgentResult(
        agent="oos",
        step=step_num,
        output=text,
        sources=[],
        error=err,
        status="ok" if text and not err else "failed",
    )

    return Command(
        update={
            "agent_results": {f"step_{step_num}": result},
            "archetype": "oos",
        },
        goto="synthesizer",
    )