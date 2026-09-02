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
from langgraph.errors import GraphRecursionError
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
from .agents import get_agent_by_name, SPECIALIST_SPECS
from .gates import (
    math_inputs_present,
    missing_inputs_result
)
from ..prompts.prompts_sub_agents import MATH, AGENT_REGISTRY
from ..prompts.prompts import GENERAL_PROMPT , OOS_PROMPT
# Graph context
from ..graph_context.response_contracts import (
    SynthesizerOutput, get_contract, resolve_archetype,
    usable_results, DetailSection, agents_from_results
)
from ..graph_context.response_validator import enforce_contract, fallback_payload
from ..graph_context.response_contracts import build_synthesizer_archetype_section
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
from .tools import begin_tool_scope
# ================================================================
# CONFIGURATION
# ================================================================

logger = logging.getLogger(__name__)

TOKEN_LIMIT = 25000
MESSAGES_TO_KEEP = 6
_SUGGESTER_DEADLINE_S = 6
_MAX_MISROUTE_RETRIES = 2
STEP_DEADLINE_S = 60.0 

# ================================================================
# ROUTING: planner → general | oos | orchestrator
# ================================================================

GENERAL_AGENT = "general"
OOS_AGENT = "oos"
# Roster válido para recuperar un MISROUTE. Sin whitelist, un nombre
# alucinado por el LLM explota adentro de get_agent_by_name en run_step.
_MISROUTE_AGENTS = frozenset({
    "contamination", "safety", "chemistry", "compliance",
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
        return "orchestrator"

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


def _direct_answer(state: PoolAgentState, system_prompt: str, deadline_s: float = STEP_DEADLINE_S) -> tuple[str, str | None]:
    """
    Una sola llamada al LLM con deadline.
    """
    plan = state.get("execution_plan") or []
    user_message = _last_human_text(state)
    task = plan[0].task if plan else user_message
    language = _LANGUAGE_MAP.get(state.get("detected_language", "es"), _LANGUAGE_MAP["es"])

    try:
        # Ejecutar con deadline
        def _invoke():
            return _get_llm().invoke([
                SystemMessage(content=f"{system_prompt}\n\nRespond in: {language}"),
                HumanMessage(content=f"Task: {task}\n\nUser context: {user_message}"),
            ])
        
        result = _run_with_deadline(_invoke, deadline_s)
        return _extract_text(result.content), None
    except FuturesTimeout:
        return "", "STEP_DEADLINE_EXCEEDED"
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
 
STEP_DEADLINE_S = 120.0    # techo por sub-agente
TURN_DEADLINE_S = 120.0    # techo por turno completo
MIN_STEP_BUDGET_S = 8.0   # si queda menos que esto, no arranques otro paso
 
# Pool dedicado: no compartir con el executor por defecto de LangGraph.
_STEP_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="run_step")
_SUGGESTER_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="suggester")



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
_SOFT_ERROR_PREFIXES = ("MISSING_INPUTS", "CANNOT_COMPUTE", "NO_GRAPH_COVERAGE", "TOOL_BUDGET_EXCEEDED")

# ================================================================
# HELPERS
# ================================================================
_CODE_FENCE_RE = re.compile(r"^\s*```(?:json|markdown)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """
    Red de seguridad: los sub-agentes emiten BASE_OUTPUT_CONTRACT envuelto en
```json y el modelo a veces arrastra ese envoltorio al `answer`. El usuario
    nunca debe ver un bloque de código -- si queda JSON dentro, al menos se
    muestra sin el fence.
    """
    if not text:
        return text
    m = _CODE_FENCE_RE.match(text.strip())
    return m.group(1).strip() if m else text

def _normalize_agent_results(raw) -> dict:
    """
    dict | list -> dict[str, AgentResult]. Defensivo: el reducer del state
    debería entregar siempre un dict, pero un fan-out mal formado o un
    checkpoint viejo pueden traer otra cosa.
    """
    if isinstance(raw, dict):
        out = {}
        for k, v in raw.items():
            if isinstance(v, AgentResult):
                out[k] = v
            elif isinstance(v, dict):
                try:
                    out[k] = AgentResult(**v)
                except Exception as e:
                    logger.error("no se pudo convertir %s a AgentResult: %s", k, e)
        return out

    if isinstance(raw, list):
        logger.warning("synthesizer: agent_results llegó como lista (%d items)", len(raw))
        out = {}
        for r in raw:
            step = getattr(r, "step", None)
            if step is None and isinstance(r, dict):
                step = r.get("step")
            if step is None:
                continue
            try:
                out[f"step_{step}"] = r if isinstance(r, AgentResult) else AgentResult(**r)
            except Exception as e:
                logger.error("no se pudo convertir item de lista: %s", e)
        return out

    return {}

# Agregar al inicio del archivo, después de los imports
def _resolve_and_update_archetype(
    execution_plan: list[ExecutionStep],
    agent_results: dict,
    extra_results: dict | None = None,
    error: str | None = None,
    force_archetype: str | None = None,
    agent_message: str | None = None,  # ✅ NUEVO PARÁMETRO
) -> dict:
    """
    Helper unificado para resolver el archetype y preparar el update.
    """
    from langchain_core.messages import AIMessage
    
    merged = {**agent_results, **(extra_results or {})}
    usable = usable_results(merged)
    agents = [r.agent for r in usable]

    archetype = force_archetype or resolve_archetype(
        agents=agents,
        is_oos=_is_oos(execution_plan),
    )

    update: dict = {"archetype": archetype}

    # Persistir TODO lo que se usó para resolver el archetype, no solo
    # extra_results. Antes `agent_results` se calculaba y se descartaba:
    # el synthesizer recibía {} y caía en la rama del saludo.
    if merged:
        update["agent_results"] = merged

    if error:
        update["error"] = error

    if agent_message:
        update["messages"] = [AIMessage(content=agent_message, name="Marlin")]
        # `agent_output` se elimina: no es un canal declarado en PoolAgentState
        # y LangGraph lo descartaba silenciosamente.

    return update

_EMPTY_RESULTS_FALLBACK = {
    "es": {
        "answer": (
            "No pude completar tu consulta en este intento. "
            "¿Podés volver a formularla?"
        ),
        "safety": (
            "Si se trata de una emergencia o de una exposición química, "
            "contactá a los servicios de emergencia o al centro de "
            "toxicología de inmediato."
        ),
    },
    "en": {
        "answer": (
            "I could not complete your request on this attempt. "
            "Could you rephrase it?"
        ),
        "safety": (
            "If this is an emergency or a chemical exposure, contact "
            "emergency services or poison control immediately."
        ),
    },
}


def _empty_results_fallback(state: PoolAgentState, reason: str) -> dict:
    """
    Backstop para cuando el turno ejecutó un plan pero no llegó ningún
    resultado al synthesizer.

    Nunca debería alcanzarse: significa que un nodo no escribió en
    `agent_results`. Existe para que ese bug salga como un fallo visible
    y no como un saludo de primer contacto — que es lo que pasaba antes.
    """
    language_code = state.get("detected_language", "es")
    strings = _EMPTY_RESULTS_FALLBACK.get(
        language_code, _EMPTY_RESULTS_FALLBACK["es"]
    )

    payload = SynthesizerOutput(
        answer=strings["answer"],
        actions=[],
        safety=strings["safety"],
        details=[],
    )

    return {
        "archetype": "conversational",
        "response": payload,
        "validation": {"fallback": "empty_results", "reason": reason},
        "messages": [AIMessage(content=payload.tier1_markdown(), name="Marlin")],
    }

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

_SLUG_TO_CONFIG = {
    slug: AGENT_REGISTRY[registry_key] for slug, registry_key in SPECIALIST_SPECS
}
_SLUG_TO_CONFIG["math"] = AGENT_REGISTRY[MATH]


def _recursion_limit_for(agent_name: str) -> int:
    """
    The 'tool budget' in each AgentConfig is currently just text in the prompt --
    confirmed by two traces where the model exceeded it (12/6 and 9/6) despite
    saying "Hard limit ... non-negotiable". recursion_limit is the only real
    cutoff: each turn of create_agent's internal graph = 1 model node +
    1 tool node, so we need double the tool budget, plus margin for the final
    response that doesn't call any tool.

    No entry in _SLUG_TO_CONFIG (general/OOS don't go through run_step, or a
    new agent not yet registered) -> conservative default of 6.
    """
    config = _SLUG_TO_CONFIG.get(_normalize_agent(agent_name))
    budget = getattr(config, "tool_budget", 6) if config else 6
    return budget * 2 + 2

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
    """
    Static fallback payload when everything fails.
    """
    _SERVICE_UNAVAILABLE_TEXT = {
        "es": "Lo siento, nuestro asistente está experimentando una interrupción temporal por alta demanda. Probá de nuevo en unos minutos.",
        "en": "Sorry, our assistant is experiencing a temporary service interruption due to high demand. Please try again in a few minutes.",
    }
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


def _build_raw_content(agent_results) -> str:
    """
    Build raw content from agent results.
    
    Defensive: handles both dict and list inputs.
    """
    if not agent_results:
        return ""

    # Normalizar a lista de resultados
    results_list = []
    if isinstance(agent_results, dict):
        results_list = list(agent_results.values())
    elif isinstance(agent_results, list):
        results_list = agent_results
    
    # Filtrar y ordenar
    valid_results = []
    for result in results_list:
        if isinstance(result, AgentResult):
            valid_results.append(result)
        elif isinstance(result, dict):
            try:
                valid_results.append(AgentResult(**result))
            except Exception:
                pass
    
    # Ordenar por step
    sorted_results = sorted(valid_results, key=lambda r: r.step)

    sections = []
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

def _suggest_block_reason(state: PoolAgentState) -> str | None:
    """
    El motivo por el que este turno no lleva chips, o None si sí los lleva.

    Separado de `should_suggest` para que el nodo pueda loguear cuál gate
    cortó. El orden es intencional: lo más barato y más frecuente primero.
    """
    archetype = state.get("archetype")
    if archetype in _SUPPRESSED_ARCHETYPES:
        return f"archetype_suppressed:{archetype}"

    if state.get("error"):
        return f"turn_error:{str(state.get('error'))[:80]}"

    # Sin agentes usables no hay contenido del cual predecir nada.
    if not agents_from_results(state.get("agent_results") or {}):
        return "no_usable_agents"

    # El usuario ya ignoró chips dos turnos seguidos: dejar de ofrecerlos.
    streak = state.get("ignored_chip_streak", 0)
    if streak >= _IGNORED_CHIP_LIMIT:
        return f"ignored_chip_streak:{streak}"

    # El synthesizer ya cerró con una pregunta propia; un chip encima es ruido.
    if answer_ends_with_question(state):
        return "answer_ends_with_question"

    return None


def should_suggest(state: PoolAgentState) -> bool:
    """Lógica pura, cero llamadas al LLM. Corre antes de cualquier gasto de cuota."""
    return _suggest_block_reason(state) is None

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
 
def _add_agent_message_to_update(update: dict, message: str) -> dict:
    """
    Agrega un mensaje de agente al update.

    No escribe `agent_output`: no es un canal declarado en PoolAgentState,
    así que LangGraph lo descartaba en silencio. El texto viaja por
    `agent_results` (que sí es un canal) y por `messages`.
    """
    if not message:
        return update

    from langchain_core.messages import AIMessage

    if "messages" not in update:
        update["messages"] = []
    update["messages"].append(AIMessage(content=message, name="Marlin"))

    return update

def _format_entities(nodes: List) -> str:
    if not nodes:
        return "(ninguna)"
    return "\n".join(f"- {n.id} | {n.name} | {n.label}" for n in nodes)

_SUPPRESSED_ARCHETYPES = frozenset({"critical", "conversational", "oos"})
 
_IGNORED_CHIP_LIMIT = 2
 
_suggester_llm = None
 
def _to_synthesizer(
    execution_plan: list[ExecutionStep],
    agent_results: dict,
    extra_results: dict | None = None,
    error: str | None = None,
    force_archetype: str | None = None,
) -> Command:
    """Única salida hacia el synthesizer usando el helper unificado."""
    update = _resolve_and_update_archetype(
        execution_plan, agent_results, extra_results, error, force_archetype
    )
    return Command(update=update, goto="synthesizer")

 
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
        update={
            "turn_started_at": time.time(),
            # Reset de turno. `agent_results` ya se limpia en el planner vía
            # el centinela None de merge_agent_results; estos canales no
            # tenían equivalente y sobrevivían en el checkpointer.
            #
            # `error` es el que más dolía: should_suggest corta con cualquier
            # error, así que un solo turno fallido dejaba el thread sin chips
            # de forma permanente.
            #
            # Va acá y no en el planner porque este nodo es la única puerta
            # de entrada garantizada del turno — el planner puede no llegar
            # a ejecutarse, o fallar antes de emitir su update.
            "error": None,
            "planner_error": None,
            "archetype": None,
            "misroute_retries": 0,
            "response": None,
            "validation": {},
            "suggestions": [],
        },
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

@observe(as_type="agent", name="Orchestrator Node")
def orchestrator(state: PoolAgentState) -> Command:
    execution_plan = state.get("execution_plan", [])
    agent_results = state.get("agent_results") or {}
 
    if not execution_plan:
        return _to_synthesizer(
            execution_plan, agent_results,
            error="EMPTY_EXECUTION_PLAN: planner produced no steps.",
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
        return _to_synthesizer(
            execution_plan, agent_results
        )
 
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
        return _to_synthesizer(
            execution_plan, agent_results,
            extra_results={f"step_{s.step}": _skipped_result(s, reason) for s in pending},
            error=f"UPSTREAM_INFRA_FAILURE at step_{failed_num}",
        )
 
    # --- 3. Presupuesto de turno ---
    remaining = _remaining_budget(state)
    if remaining <= MIN_STEP_BUDGET_S:
        reason = f"SKIPPED_TURN_DEADLINE_EXCEEDED: {remaining:.1f}s left"
        return _to_synthesizer(
            execution_plan, agent_results,
            extra_results={f"step_{s.step}": _skipped_result(s, reason) for s in pending},
            error="TURN_DEADLINE_EXCEEDED",
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
    
    runnable_nums = {s.step for s in runnable}
    waiting = [s for s in pending if s.step not in blocked_nums and s.step not in runnable_nums]
 
    err = (
        f"Deadlock in execution_plan: {[s.step for s in waiting]} blocked."
        if waiting else None
    )
    
    waiting_results = {
        f"step_{s.step}": _skipped_result(
            s, f"SKIPPED_DEADLOCK: step_{s.step} is in a dependency cycle"
        )
        for s in waiting
    }
    
    extra = {**blocked, **waiting_results} if (blocked or waiting) else None
 
    if runnable:
        messages = state.get("messages", [])
        user_message = ""
        for msg in reversed(messages):
            if getattr(msg, "type", None) == "human":
                user_message = _extract_text(msg.content)
                break
 
        step_budget = max(MIN_STEP_BUDGET_S, min(STEP_DEADLINE_S, remaining))
 
        return Command(
            update={"agent_results": blocked} if blocked else {},
            goto=[
                Send(
                    "run_step",
                    {
                        "step": s,
                        "user_message": user_message,
                        "deadline_s": step_budget,
                        "agent_results": agent_results,
                    },
                )
                for s in runnable
            ],
        )
    
    # Si no hay runnable, vamos a synthesizer
    # Primero obtenemos el update correcto usando _to_synthesizer
    if waiting:
        # Caso de deadlock: vamos directo a synthesizer con el error
        return _to_synthesizer(
            execution_plan, agent_results,
            extra_results=extra,
            error=err,
        )
    
    # Si no hay runnable ni waiting, es porque todo está bloqueado o completado
    # Usamos _to_synthesizer para obtener el update correcto
    synth_command = _to_synthesizer(
        execution_plan, agent_results,
        extra_results=extra,
        error=err or "No runnable steps available",
    )
    
    # Ahora decidimos si hacer fan-out o ir directo a synthesizer
    return synth_command
 
 
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
    from langchain_core.messages import HumanMessage
    import logging
    
    logger = logging.getLogger(__name__)
 
    step = payload["step"]
    user_message = payload["user_message"]
    deadline_s = float(payload.get("deadline_s", STEP_DEADLINE_S))
    step_key = f"step_{step.step}"
    
    # ✅ OBTENER EL ESTADO COMPLETO DEL PAYLOAD
    # Asumiendo que el Send desde orchestrator incluye el estado
    state = {"agent_results": payload.get("agent_results") or {}}
 
    # ✅ GATE PARA MATH
    if step.assigned_agent == MATH and not math_inputs_present(user_message):
        return Command(
            update={"agent_results": {step_key: missing_inputs_result(step, user_message)}},
            goto="orchestrator",
        )
 
    # ✅ CONSTRUIR CONTEXTO ENRIQUECIDO
    agent_context = _build_agent_context(state, step, user_message)
    
    # ✅ CREAR EL INPUT DEL AGENTE CON CONTEXTO COMPARTIDO
    agent_input = {
        "messages": [
            HumanMessage(
                content=agent_context
            )
        ]
    }
    
    # Log para debugging (opcional)
    logger.info(f"run_step: step_{step.step} ({step.assigned_agent}) - Contexto construido con {len(agent_context)} caracteres")
 
    started = time.monotonic()
    try:
        # ✅ PASAR EL INPUT ENRIQUECIDO EN LUGAR DE SOLO step Y user_message
        begin_tool_scope() 
        agent_result = _run_with_deadline(
            _run_step_enriched,  # Nueva función que acepta agent_input
            deadline_s, 
            step, 
            agent_input,            
        )
 
    except FuturesTimeout:
        agent_result = AgentResult(
            agent=step.assigned_agent,
            step=step.step,
            output="",
            sources=[],
            error=f"STEP_DEADLINE_EXCEEDED after {deadline_s:.0f}s",
            status="failed",
        )

    except GraphRecursionError:
        # El agente agotó su recursion_limit sin encontrar evidencia
        # suficiente para responder. No es un fallo del proveedor -- es un
        # gap de negocio, tratado como MISSING_INPUTS/CANNOT_COMPUTE en
        # is_infra_error para que no dispare el circuit breaker del turno.
        agent_result = AgentResult(
            agent=step.assigned_agent,
            step=step.step,
            output="",
            sources=[],
            error=f"TOOL_BUDGET_EXCEEDED: {step.assigned_agent} exceeded its recursion_limit",
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


# ✅ NUEVA FUNCIÓN HELPER PARA CONSTRUIR EL CONTEXTO
def _build_agent_context(state: dict, step: ExecutionStep, user_message: str) -> str:
    """
    Construye el contexto enriquecido para el agente basado en:
    1. El mensaje del usuario original
    2. La tarea específica del paso
    3. Los resultados de pasos anteriores (dependencias)
    4. (Opcional) El resultado del paso inmediatamente anterior
    """
    agent_results = state.get("agent_results", {})
    
    # 1. Tarea y mensaje del usuario
    context_parts = [
        f"TASK: {step.task}",
        "",
        f"USER MESSAGE: {user_message}",
        "",
    ]
    
    # 2. Resultados de pasos de los que depende
    if step.depends_on:
        context_parts.append("--- PREVIOUS STEP RESULTS (Dependencies) ---")
        for dep_step_num in step.depends_on:
            dep_key = f"step_{dep_step_num}"
            dep_result = agent_results.get(dep_key)
            if dep_result:
                # Extraer output y status
                output = _get_result_output(dep_result)
                status = _get_result_status(dep_result)
                
                if output:
                    context_parts.append(f"Step {dep_step_num} (status: {status}):")
                    context_parts.append(output)
                    context_parts.append("")  # Línea en blanco para separación
                elif status == "failed":
                    error = _get_result_error(dep_result)
                    context_parts.append(f"Step {dep_step_num} FAILED: {error}")
                    context_parts.append("")
            else:
                context_parts.append(f"Step {dep_step_num}: No result available")
                context_parts.append("")
    
    # 3. (Opcional) Resultado del paso inmediatamente anterior para más contexto
    previous_step_num = step.step - 1
    if previous_step_num >= 1:
        prev_key = f"step_{previous_step_num}"
        # Evitar duplicar si ya está en depends_on
        if prev_key not in [f"step_{d}" for d in (step.depends_on or [])]:
            prev_result = agent_results.get(prev_key)
            if prev_result:
                output = _get_result_output(prev_result)
                if output:
                    context_parts.append("--- ADDITIONAL CONTEXT (Previous Step) ---")
                    context_parts.append(f"Step {previous_step_num} result:")
                    context_parts.append(output)
                    context_parts.append("")
    
    # 4. Instrucción sobre cómo usar el contexto
    if step.depends_on or previous_step_num >= 1:
        context_parts.append("--- INSTRUCTIONS ---")
        context_parts.append(
            "Apply the Context Sharing rules from your system prompt to the "
            "material above. Check each step's status before treating it as "
            "established."
        )
    
    return "\n".join(context_parts)


# ✅ NUEVA FUNCIÓN HELPER PARA EXTRAER OUTPUT DE UN RESULTADO
def _get_result_output(result) -> str:
    """Extrae el output de un AgentResult o dict."""
    if hasattr(result, 'output'):
        return result.output or ""
    if isinstance(result, dict):
        return result.get('output', "")
    return ""


# ✅ NUEVA FUNCIÓN HELPER PARA EXTRAER STATUS
def _get_result_status(result) -> str:
    """Extrae el status de un AgentResult o dict."""
    if hasattr(result, 'status'):
        return result.status or "unknown"
    if isinstance(result, dict):
        return result.get('status', "unknown")
    return "unknown"


# ✅ NUEVA FUNCIÓN HELPER PARA EXTRAER ERROR
def _get_result_error(result) -> str:
    """Extrae el error de un AgentResult o dict."""
    if hasattr(result, 'error'):
        return result.error or ""
    if isinstance(result, dict):
        return result.get('error', "")
    return ""


# ✅ NUEVA FUNCIÓN _run_step_enriched (reemplaza a _run_step)
def _run_step_enriched(step: ExecutionStep, agent_input: dict) -> AgentResult:
    """
    Versión enriquecida de _run_step que acepta agent_input pre-construido.
    """
    agent = get_agent_by_name(step.assigned_agent)
    result = agent.invoke(
        agent_input,
        config={"recursion_limit": _recursion_limit_for(step.assigned_agent)},
    )

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
# ================================================================
# SYNTHESIZER NODE
# ================================================================


@observe(as_type="span", name="Synthesizer Node")
def synthesizer(state: PoolAgentState) -> dict:
    """
    Convierte los resultados de los sub-agentes en UNA respuesta en lenguaje
    natural para el usuario.

    Invariante: el `answer` que sale de aquí es prosa. Los sub-agentes emiten
    JSON estructurado (BASE_OUTPUT_CONTRACT); ese JSON es materia prima para el
    LLM de síntesis, nunca la respuesta. El único texto que puede pasar sin
    sintetizar es el de `general`, que ya produce prosa por diseño.
    """
    execution_plan: list[ExecutionStep] = state.get("execution_plan", [])
    language_code: str = state.get("detected_language", "es")

    agent_results = _normalize_agent_results(state.get("agent_results") or {})

    # ============================================================
    # ÚNICO ATAJO: clarificación de `general`
    # ------------------------------------------------------------
    # Se decide por el PLAN, no por keywords en el texto. Buscar
    # "provide"/"need"/"volume" en la salida de un especialista da falso
    # positivo casi siempre -- un informe de contaminación contiene esas
    # palabras sin ser una clarificación -- y devolvía el JSON crudo al
    # usuario, saltándose la síntesis por completo.
    # ============================================================
    if (
        len(execution_plan) == 1
        and _normalize_agent(execution_plan[0].assigned_agent) == "general"
    ):
        single = next(iter(agent_results.values()), None)
        if single and single.output and not single.error:
            logger.info("synthesizer: clarificación de `general`, se usa su prosa directa")
            text = _strip_code_fences(single.output)
            payload = SynthesizerOutput(
                answer=text,
                actions=[],
                safety=None,
                details=[],
            )
            return {
                "archetype": state.get("archetype", "conversational"),
                "response": payload,
                "validation": {"direct_agent_message": True, "is_clarification": True},
                "messages": [AIMessage(content=text, name="Marlin")],
            }

    # ============================================================
    # OOS / IDIOMA
    # ============================================================
    is_oos = _is_oos(execution_plan)
    oos_instruction = _OOS_INSTRUCTION_ACTIVE if is_oos else _OOS_INSTRUCTION_INACTIVE
    language_instruction = _LANGUAGE_MAP.get(language_code, _LANGUAGE_MAP["es"])

    # ============================================================
    # RESULTADOS
    # ------------------------------------------------------------
    # `usable` y `agents` van filtrados: alimentan enforce_contract (que
    # resuelve el safety condicional por HAZARD_AGENTS) y _attach_sources.
    # `raw_content` ve el turno COMPLETO, incluidos los steps con error o
    # skipped: sin eso el synthesizer no puede decir "no pude verificar X" y
    # degrada a un saludo genérico cuando el único step del turno falló.
    # ============================================================
    usable = usable_results(agent_results)
    agents = [r.agent for r in usable]

    archetype = state.get("archetype")
    if not archetype:
        logger.warning(
            "synthesizer: 'archetype' ausente del state — el orchestrator no lo "
            "resolvió. Degradando a 'conversational'."
        )
        archetype = "conversational"

    raw_content = _build_raw_content(agent_results)
    if not raw_content:
        if execution_plan:
            # Se ejecutó un plan y no llegó nada: es un bug de escritura de
            # estado, no un turno vacío. El saludo NUNCA es correcto acá.
            logger.error(
                "synthesizer: raw_content vacío con execution_plan no vacío "
                "(agents=%s) — algún nodo no escribió en agent_results",
                [s.assigned_agent for s in execution_plan],
            )
            return _empty_results_fallback(
                state, reason="empty_raw_content_with_plan"
            )
        raw_content = "(no prior content — generate a warm greeting and offer help)"
        archetype, agents, usable = "conversational", [], []

    contract = get_contract(archetype)

    # ============================================================
    # PROMPT
    # ============================================================
    system_content = SYNTHESIZER_PROMPT.format(
        archetype_section=build_synthesizer_archetype_section(archetype, agents),
        oos_instruction=oos_instruction,
        language=language_instruction,
        raw_content=raw_content,
    )
    llm_messages = [
        SystemMessage(content=system_content),
        HumanMessage(content="Generate the final refined response now."),
    ]

    # ============================================================
    # FASE 1 — generación, degradación en tres niveles
    # ============================================================
    validation: dict = {}
    try:
        payload = _get_llm().with_structured_output(SynthesizerOutput).invoke(llm_messages)
    except Exception as exc:
        logger.warning(
            "synthesizer: structured output falló (%s); reintentando sin estructura", exc
        )
        try:
            raw = _get_llm().invoke(llm_messages)
            payload = fallback_payload(_flatten(raw.content), SynthesizerOutput)
            validation = {"fallback": "unstructured", "reason": str(exc)}
        except Exception as exc2:
            # 429 / 503 / 504: el reintento falla por la misma causa que el
            # primero. Payload estático: cero red, cero enforcement (el
            # validador también podría lanzar y ya es el último nivel).
            logger.error(
                "synthesizer: ambos intentos de generación fallaron (%s | %s)", exc, exc2
            )
            payload = static_service_unavailable_payload(SynthesizerOutput, language_code)
            return {
                "archetype": archetype,
                "response": payload,
                "validation": {"fallback": "static", "reason": f"{exc} | {exc2}"},
                "messages": [AIMessage(content=payload.tier1_markdown(), name="Marlin")],
            }

    # ============================================================
    # FASE 2 — enforcement (siempre, venga el payload de donde venga)
    # ============================================================
    try:
        payload, report = enforce_contract(
            payload, contract, agents, detail_cls=DetailSection
        )
        validation = {**validation, **report.to_dict()}
    except Exception as exc:
        # Un bug del validador no debe costar otra llamada al modelo.
        logger.error("synthesizer: enforce_contract lanzó (%s)", exc, exc_info=True)
        validation = {**validation, "enforcement_error": str(exc)}

    # ============================================================
    # FASE 3 — garantía de prosa + fuentes
    # ============================================================
    payload.answer = _strip_code_fences(payload.answer or "")
    if payload.safety:
        payload.safety = _strip_code_fences(payload.safety)

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
@observe(as_type="agent", name="Suggester")
def suggester(state: PoolAgentState, config: RunnableConfig) -> dict:
    """
    Produce los chips de seguimiento del turno.

    Corre DESPUÉS del synthesizer (edge secuencial synthesizer → suggester),
    no en paralelo: necesita `state["response"]` para saber qué quedó sin
    cubrir y para no ofrecer un chip encima de una pregunta del synthesizer.
    Eso lo pone en el camino crítico del usuario, y por eso el deadline es
    corto y todo error degrada a [] en vez de propagarse.

    Devuelve SIEMPRE la clave "suggestions" — nunca la omite, para que el
    frontend pueda distinguir "no hubo chips" de "el nodo no corrió".
    """
    blocked = _suggest_block_reason(state)
    if blocked:
        logger.info("suggester skipped: %s", blocked)
        return {"suggestions": []}

    thread_id = config.get("configurable", {}).get("thread_id", "")

    unconsumed = _unconsumed_entities(state, thread_id)
    if not unconsumed:
        # Sin entidades libres el LLM solo puede inventar. Ahorramos la
        # llamada: es el caso más común en turnos sin retrieval.
        #
        # Se loguea `touched` crudo aparte del filtrado: distingue "el
        # retrieval no tocó nada" (turn_cache vacío) de "tocó y los dos
        # filtros de _unconsumed_entities se lo comieron todo". Son dos
        # bugs distintos.
        logger.info(
            "suggester skipped: no_unconsumed_entities (thread=%s, touched=%d)",
            thread_id or "(vacío)",
            len(get_touched(thread_id) or []),
        )
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
        # timeout >= 10s y no queremos esperar tanto.
        #
        # NO usar `with ThreadPoolExecutor(...)`: su __exit__ hace
        # shutdown(wait=True), así que aunque .result() lance el timeout, el
        # bloque se queda esperando la llamada completa igual. Se pagaba la
        # latencia entera y encima se descartaba el resultado.
        #
        # copy_context() propaga el parent OTel de Langfuse al thread; sin él
        # el span del suggester aparece suelto en el trace.
        ctx = contextvars.copy_context()
        future = _SUGGESTER_POOL.submit(ctx.run, chain.invoke, messages)
        try:
            payload: SuggesterOutput = future.result(timeout=_SUGGESTER_DEADLINE_S)
        except FuturesTimeout:
            future.cancel()  # no mata el thread en curso, solo libera el slot
            raise
    except Exception as exc:
        # Todo se degrada igual: 429, TimeoutError del executor, o structured
        # output inválido. Se loguea para poder ver la distribución real de
        # fallas en Langfuse, pero nunca se propaga: un chip opcional no rompe
        # el turno del usuario.
        logger.warning("suggester degraded to []: %s: %s", type(exc).__name__, exc)
        return {"suggestions": []}

    raw: List[Suggestion] = payload.suggestions or []
    gated, report = apply_gates_with_report(raw, answer_text)

    # El reporte va al log, no al state: es telemetría de calidad del prompt
    # (paso 9: "cuál gate descarta más"), no algo que el frontend consuma.
    if report["input"] != report["output"]:
        logger.info("suggester gates: %s", report)

    logger.info("suggester: %d chips generados", len(gated))
    return {"suggestions": gated}

@observe(as_type="agent", name="General Node")
def general(state: PoolAgentState) -> Command[Literal["synthesizer"]]:
    plan = state.get("execution_plan") or []
    step_num = plan[0].step if plan else 1

    remaining = _remaining_budget(state)
    step_budget = max(MIN_STEP_BUDGET_S, min(STEP_DEADLINE_S, remaining))
    
    text, err = _direct_answer(state, GENERAL_PROMPT, deadline_s=step_budget)

    result = AgentResult(
        agent=GENERAL_AGENT,
        step=step_num,
        output=text,
        sources=[],
        error=err,
        status="ok" if text and not err else "failed",
    )

    update = _resolve_and_update_archetype(
        execution_plan=plan,
        agent_results={f"step_{step_num}": result},
        force_archetype="conversational",
    )
    
    # ✅ AGREGAR EL MENSAJE
    update = _add_agent_message_to_update(update, text)
    
    return Command(update=update, goto="synthesizer")

@observe(as_type="agent", name="OOS Node")
def oos(state: PoolAgentState) -> Command[Literal["orchestrator", "synthesizer"]]:
    plan = state.get("execution_plan") or []
    step_num = plan[0].step if plan else 1

    remaining = _remaining_budget(state)
    step_budget = max(MIN_STEP_BUDGET_S, min(STEP_DEADLINE_S, remaining))

    text, err = _direct_answer(state, OOS_PROMPT, deadline_s=step_budget)

    # ── MISROUTE ─────────────────────────────────────────────────────
    misroute_match = _MISROUTE_RE.match(text) if text else None

    if misroute_match:
        target_agent = misroute_match.group(1).strip().lower()
        rest_text = misroute_match.group(2).strip()

        misroute_retries = state.get("misroute_retries", 0)

        if misroute_retries >= _MAX_MISROUTE_RETRIES:
            # Se agotaron los reintentos. `text` todavía empieza con
            # "MISROUTE: <agente>" — señal de control interna que nunca
            # debe llegar al usuario ni al historial. Se propaga solo el
            # AgentResult fallido; el synthesizer decide qué decir.
            logger.warning(
                "oos: MISROUTE a '%s' agotó %d reintentos",
                target_agent, _MAX_MISROUTE_RETRIES,
            )
            result = AgentResult(
                agent=OOS_AGENT,
                step=step_num,
                output=rest_text,
                sources=[],
                error=f"MAX_MISROUTE_RETRIES_EXCEEDED: {target_agent}",
                status="failed",
            )
            update = _resolve_and_update_archetype(
                execution_plan=plan,
                agent_results={f"step_{step_num}": result},
                force_archetype="oos",
            )
            return Command(update=update, goto="synthesizer")

        if target_agent in _MISROUTE_AGENTS:
            logger.info(
                "oos: MISROUTE a '%s' (intento %d)",
                target_agent, misroute_retries + 1,
            )
            new_step = ExecutionStep(
                step=step_num,
                task=rest_text or state["messages"][-1].content,
                assigned_agent=target_agent,
                oos=False,
            )
            return Command(
                update={
                    "execution_plan": [new_step],
                    "misroute_retries": misroute_retries + 1,
                    "archetype": None,
                },
                goto="orchestrator",
            )

        logger.warning("oos: MISROUTE a agente desconocido '%s'", target_agent)

    # ── Camino normal ────────────────────────────────────────────────
    result = AgentResult(
        agent=OOS_AGENT,
        step=step_num,
        output=text,
        sources=[],
        error=err,
        status="ok" if text and not err else "failed",
    )

    update = _resolve_and_update_archetype(
        execution_plan=plan,
        agent_results={f"step_{step_num}": result},
        force_archetype="oos",
    )
    update = _add_agent_message_to_update(update, text)

    return Command(update=update, goto="synthesizer")