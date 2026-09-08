from __future__ import annotations

import contextvars
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import List, Literal, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langfuse import get_client, observe
from langgraph.errors import GraphRecursionError
from langgraph.types import Command
from pydantic import BaseModel, Field

from .agents import build_cluster_agent, is_tooled
from .chains import create_planner_chain
from .state import AgentResult, PlanStep, PoolAgentState, RESULT_KEY
from .tools import begin_tool_scope
from ..config.llm import create_llm, create_suggester_llm
from ..graph_context.response_contracts import (
    DetailSection,
    SynthesizerOutput,
    agents_from_results,
    get_contract,
    resolve_archetype,
    usable_results,
)
from ..graph_context.response_validator import enforce_contract, fallback_payload
from ..graph_context.suggestions import (
    Suggestion,
    SuggesterOutput,
    apply_gates_with_report,
    roster_text,
)
from ..graph_context.turn_cache import reset_turn
from ..prompts.prompt_archetype import (
    build_cluster_prompt,
    build_synthesizer_archetype_section,
)
from ..prompts.prompts import (
    GENERAL_PROMPT,
    OOS_PROMPT,
    PLANNER_PROMPT,
    SUGGESTER_PROMPT,
    SYNTHESIZER_PROMPT,
)
from ..prompts.prompts_sub_agents import CLUSTER_REGISTRY, MODULE_TO_CLUSTER

# ================================================================
# CONFIGURATION
# ================================================================

logger = logging.getLogger(__name__)

TOKEN_LIMIT = 25000
MESSAGES_TO_KEEP = 6

# ── Presupuesto de turno ─────────────────────────────────────────────
# TURN_DEADLINE_S es el techo del turno completo. CLUSTER_DEADLINE_S es el
# techo de UNA llamada al cluster, y tiene que dejar margen para lo que
# viene después: judge + (posible revisión) + synthesizer.
#
# Antes STEP_DEADLINE_S estaba definido dos veces -- 60.0 arriba y 120.0
# más abajo, ganando el segundo por orden de asignación -- y quedaba igual
# a TURN_DEADLINE_S, así que un solo paso podía consumir el turno entero.
TURN_DEADLINE_S = 120.0
CLUSTER_DEADLINE_S = 45.0    # se observaron llamadas de compliance a 29.5s
JUDGE_DEADLINE_S = 15.0
MIN_STEP_BUDGET_S = 8.0

# Presupuesto mínimo que tiene que quedar para que valga la pena mandar una
# revisión: el cluster corre otra vez (con techo reducido) y después todavía
# falta el synthesizer. Si no alcanza, el judge auto-aprueba.
_REVISION_MIN_BUDGET_S = CLUSTER_DEADLINE_S / 2 + MIN_STEP_BUDGET_S

# Una sola revisión. El ciclo cuesta una llamada de judge más una de cluster
# completa; con dos revisiones el turno se pasa del deadline de forma
# sistemática.
_MAX_JUDGE_RETRIES = 1

_SUGGESTER_DEADLINE_S = 6
_SUGGESTER_KICKOFF = {
    "es": "Generá las sugerencias ahora, en español, o ninguna.",
    "en": "Generate the suggestions now, in English, or none.",
}
_MAX_MISROUTE_RETRIES = 2

# Pool dedicado: no compartir con el executor por defecto de LangGraph.
_STEP_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="run_step")
_SUGGESTER_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="suggester")


# ---------------------------------------------------------------------------
# Clasificación de errores
# ---------------------------------------------------------------------------

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
_SOFT_ERROR_PREFIXES = (
    "MISSING_INPUTS",
    "CANNOT_COMPUTE",
    "TOOL_BUDGET_EXCEEDED",
    "EMPTY_AGENT_OUTPUT",
)


def is_infra_error(err: str | None, exc: BaseException | None = None) -> bool:
    """True solo para fallos del proveedor / timeouts, no para contratos de negocio."""
    if exc is not None and exc.__class__.__name__ in _INFRA_EXC_NAMES:
        return True
    if not err:
        return False
    if err.startswith(_SOFT_ERROR_PREFIXES):
        return False
    return bool(_INFRA_CODE_RE.match(err)) or any(n in err for n in _INFRA_NAMES)


# ================================================================
# ROUTING: planner → cluster | general | oos
# ================================================================

GENERAL_AGENT = "general"
OOS_AGENT = "oos"

# El módulo del que se recupera un MISROUTE, mapeado por MODULE_TO_CLUSTER
# al nodo real. Sin whitelist, un nombre alucinado explota en el goto.
#
# 'general' estaba faltando: OOS_PROMPT lista saludos y preguntas de
# capacidad como misroute a general, así que un "hola" mal ruteado emitía
# MISROUTE: general, caía en el else de agente desconocido, y el usuario
# recibía un rechazo de scope por saludar.
_MISROUTE_AGENTS = frozenset({
    "contamination", "safety", "chemistry", "compliance", "general",
})

_MISROUTE_RE = re.compile(r"^\s*MISROUTE:\s*([A-Za-z_]+)\s*(.*)", re.DOTALL)


def _normalize_agent(agent) -> str:
    """AgentName puede ser str, Enum o None."""
    if agent is None:
        return ""
    value = getattr(agent, "value", agent)
    return str(value).strip().lower()


def _route_from_plan(plan) -> str:
    """
    Nodo destino para el único step del turno.

    El planner emite un module_id; MODULE_TO_CLUSTER resuelve el cluster.
    'general' y 'oos' no son módulos y tienen su propio nodo.

    Un módulo desconocido degrada a 'general' en vez de reventar en el
    goto: el planner es un LLM y puede alucinar un nombre pese al Literal.
    """
    if plan is None:
        logger.error("_route_from_plan: plan es None; degradando a general")
        return GENERAL_AGENT

    module = _normalize_agent(getattr(plan, "primary_module", None))

    if bool(getattr(plan, "oos", False)) or module == OOS_AGENT:
        return OOS_AGENT
    if module == GENERAL_AGENT:
        return GENERAL_AGENT

    cluster = MODULE_TO_CLUSTER.get(module)
    if cluster is None:
        logger.error(
            "_route_from_plan: módulo '%s' no está en MODULE_TO_CLUSTER; "
            "degradando a general", module,
        )
        return GENERAL_AGENT
    return cluster


def _last_human_text(state: PoolAgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "type", None) == "human":
            return _extract_text(msg.content)
    return ""


# ================================================================
# LAZY LLM + PLANNER CHAIN
# ================================================================

_llm = None
_planner_chain = None
_suggester_llm = None


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


def _get_llm_suggester():
    """Lazy: no construir el cliente si el gate de supresión corta antes."""
    global _suggester_llm
    if _suggester_llm is None:
        _suggester_llm = create_suggester_llm()
    return _suggester_llm


# ================================================================
# HELPERS
# ================================================================

_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json|markdown)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL
)


def _strip_code_fences(text: str) -> str:
    """
    Red de seguridad: el cluster emite BASE_OUTPUT_CONTRACT envuelto en
    ```json y el modelo a veces arrastra ese envoltorio al `answer`. El
    usuario nunca debe ver un bloque de código.
    """
    if not text:
        return text
    m = _CODE_FENCE_RE.match(text.strip())
    return m.group(1).strip() if m else text


def _extract_text(content) -> str:
    """Normaliza el content de un LLM a texto plano sea cual sea su forma."""
    if isinstance(content, list):
        return " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("text", "").strip()
        ).strip()
    return str(content).strip()


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
    return "failed"


def _remaining_budget(state) -> float:
    started = state.get("turn_started_at")
    if not started or started < time.time() - 3600:
        # stale de un turno viejo, o nunca se seteó -> no confiar en el budget
        return TURN_DEADLINE_S
    return TURN_DEADLINE_S - (time.time() - float(started))


def _recursion_limit_for(cluster_name: str) -> int:
    """
    Techo real del loop del agente. El 'tool budget' del prompt es solo
    texto -- dos trazas mostraron al modelo excediéndolo (12/6 y 9/6) pese
    a decir "non-negotiable".

    Cada turno del grafo interno de create_agent = 1 nodo de modelo +
    1 nodo de tool, así que hace falta el doble del budget, más margen para
    la respuesta final que no llama tools.
    """
    cluster = CLUSTER_REGISTRY.get(cluster_name)
    if cluster is None:
        return 14
    budget = getattr(cluster, "tool_budget_per_calc_step", 4) or 4
    return budget * 2 * 2 + 2


def _run_with_deadline(fn, deadline_s: float, *args):
    """
    Ejecuta fn con techo de wall-clock, propagando el contexto de Langfuse.

    copy_context() es obligatorio: sin él, los spans que fn abre dentro del
    thread pierden el parent OTel y aparecen sueltos en el trace.
    """
    ctx = contextvars.copy_context()
    future = _STEP_POOL.submit(ctx.run, fn, *args)
    try:
        return future.result(timeout=deadline_s)
    except FuturesTimeout:
        future.cancel()  # no mata el thread en curso, solo libera el slot
        raise


def _normalize_agent_results(raw) -> dict:
    """
    dict | list -> dict[str, AgentResult]. Defensivo: el reducer del state
    debería entregar siempre un dict, pero un checkpoint viejo puede traer
    otra cosa.
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
        logger.warning("agent_results llegó como lista (%d items)", len(raw))
        out = {}
        for r in raw:
            try:
                out[RESULT_KEY] = r if isinstance(r, AgentResult) else AgentResult(**r)
            except Exception as e:
                logger.error("no se pudo convertir item de lista: %s", e)
        return out

    return {}


def _archetype_for_plan(plan) -> str | None:
    """
    El archetype declarado por el módulo primario del plan.

    Devuelve None para general/oos (no son módulos y no tienen entrada en
    ningún cluster): esos nodos pasan force_archetype.
    """
    module = _normalize_agent(getattr(plan, "primary_module", None))
    if not module or module in (GENERAL_AGENT, OOS_AGENT):
        return None

    cluster_name = MODULE_TO_CLUSTER.get(module)
    cluster = CLUSTER_REGISTRY.get(cluster_name) if cluster_name else None
    if cluster is None:
        return None

    knowledge_module = cluster.modules.get(module)
    return getattr(knowledge_module, "archetype", None)


def _resolve_and_update_archetype(
    plan,
    agent_results: dict,
    error: str | None = None,
    force_archetype: str | None = None,
    agent_message: str | None = None,
) -> dict:
    """
    Resuelve el archetype y arma el update del state.

    El archetype sale de KnowledgeModule.archetype vía archetype_for(), no
    de los agentes que produjeron output: con un solo módulo primario por
    turno no hay empate que resolver, y el módulo es la única fuente de
    verdad desde que se eliminó AGENT_TO_ARCHETYPE.
    """
    archetype = force_archetype or resolve_archetype(
        _archetype_for_plan(plan),
        is_oos=bool(getattr(plan, "oos", False)),
    )

    update: dict = {"archetype": archetype}

    if agent_results:
        update["agent_results"] = agent_results

    if error:
        update["error"] = error

    if agent_message:
        update["messages"] = [AIMessage(content=agent_message, name="Marlin")]

    return update


def _is_oos(plan) -> bool:
    return bool(getattr(plan, "oos", False))


def _active_modules(plan) -> list[str]:
    """module_id activos del turno: primario + apoyo."""
    if plan is None:
        return []
    primary = _normalize_agent(getattr(plan, "primary_module", None))
    support = list(getattr(plan, "support_modules", None) or [])
    return [m for m in (primary, *support) if m]


def _build_raw_content(agent_results) -> str:
    """Material crudo para el synthesizer. Defensivo con dict y list."""
    if not agent_results:
        return ""

    results_list = []
    if isinstance(agent_results, dict):
        results_list = list(agent_results.values())
    elif isinstance(agent_results, list):
        results_list = agent_results

    valid_results = []
    for result in results_list:
        if isinstance(result, AgentResult):
            valid_results.append(result)
        elif isinstance(result, dict):
            try:
                valid_results.append(AgentResult(**result))
            except Exception:
                pass

    sections = []
    for result in valid_results:
        if result.error:
            sections.append(f"[{result.agent}] ERROR: {result.error}")
        elif result.output:
            sections.append(f"[{result.agent}]\n{result.output}")

    return "\n\n".join(sections)


def _cluster_output_text(state: PoolAgentState) -> str:
    """El output crudo del cluster de este turno, o ''."""
    results = _normalize_agent_results(state.get("agent_results") or {})
    result = results.get(RESULT_KEY)
    return (result.output or "") if result else ""


def _attach_sources(payload: SynthesizerOutput, results: list) -> None:
    seen, srcs = set(), []
    for r in results:
        for s in r.sources:
            if s not in seen:
                seen.add(s)
                srcs.append(s)
    if srcs:
        payload.details.append(
            DetailSection(label="Fuentes", body="\n".join(f"- {s}" for s in srcs))
        )


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

_SERVICE_UNAVAILABLE_TEXT = {
    "es": (
        "Lo siento, nuestro asistente está experimentando una interrupción "
        "temporal por alta demanda. Probá de nuevo en unos minutos."
    ),
    "en": (
        "Sorry, our assistant is experiencing a temporary service interruption "
        "due to high demand. Please try again in a few minutes."
    ),
}


def static_service_unavailable_payload(output_cls, language_code: str):
    """Payload estático de último recurso: cero red, cero enforcement."""
    text = _SERVICE_UNAVAILABLE_TEXT.get(
        language_code, _SERVICE_UNAVAILABLE_TEXT["es"]
    )
    return output_cls(answer=text, actions=[], safety=None, details=[])


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
    y no como un saludo de primer contacto.
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


# ================================================================
# CONTEXT NODE
# ================================================================

@observe(as_type="agent", name="Context Node")
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
            # Reset de turno. `agent_results` se limpia en el planner vía el
            # centinela None de merge_agent_results; estos canales no tenían
            # equivalente y sobrevivían en el checkpointer.
            #
            # `error` es el que más dolía: should_suggest corta con cualquier
            # error, así que un solo turno fallido dejaba el thread sin chips
            # de forma permanente.
            #
            # Va acá y no en el planner porque este nodo es la única puerta
            # de entrada garantizada del turno.
            "error": None,
            "planner_error": None,
            "archetype": None,
            "misroute_retries": 0,
            "judge_retries": 0,
            "judge_feedback": None,
            "response": None,
            "validation": {},
            "suggestions": [],
        },
        goto=next_node,
    )


# ================================================================
# SUMMARIZE MEMORY NODE
# ================================================================

@observe(as_type="agent", name="Summarize Node")
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

@observe(as_type="agent", name="Planner Node")
def planner(state: PoolAgentState, config: RunnableConfig):
    thread_id = config["configurable"]["thread_id"]
    reset_turn(thread_id)

    # _last_human_text en vez de messages[-1].content: normaliza bloques de
    # contenido y no asume que el último mensaje es el del usuario, que deja
    # de ser cierto después de summarize_memory_node.
    user_input = _last_human_text(state)

    agent_messages = [
        m for m in state.get("messages", [])
        if isinstance(m, AIMessage) and getattr(m, "name", None) == "Marlin"
    ]
    last_agent_msg = (
        _extract_text(agent_messages[-1].content) if agent_messages else ""
    )

    context_for_planner = (
        f"[Last agent message]: {last_agent_msg}\n"
        f"[User reply]: {user_input}"
        if last_agent_msg
        else user_input
    )

    fallback_language = state.get("detected_language") or "es"

    try:
        # El chain es dueño del system prompt y expone UNA variable, {input}.
        # Antes se invocaba con una lista de dicts de mensaje que incluía otra
        # copia de PLANNER_PROMPT: como el template tiene exactamente una
        # variable, _validate_input la envolvía en {"input": <lista>} en vez
        # de fallar, y el modelo recibía el prompt duplicado como repr() de
        # Python con el mensaje real del usuario sepultado al final. Nunca
        # lanzó, así que nunca cayó al fallback -- solo ruteaba mal.
        result = _get_planner_chain().invoke({"input": context_for_planner})
        plan = result.plan
        detected_language = result.detected_language or fallback_language
    except Exception as e:
        logger.exception("planner_llm_failed")
        get_client().update_current_span(
            level="WARNING",
            status_message=f"planner_llm_failed: {e}",
        )
        # El task arranca con el gatillo literal que GENERAL_PROMPT espera
        # para su rama de clarificación. Y el idioma sale del state, no
        # hardcodeado en español como antes.
        fallback_plan = PlanStep(
            primary_module=GENERAL_AGENT,
            task=(
                "Ask the user to provide more detail about their request. "
                f"Their message was: {user_input}"
            ),
            oos=False,
        )
        return Command(
            update={
                "detected_language": fallback_language,
                "plan": fallback_plan,
                "agent_results": None,
                "planner_error": str(e),
            },
            goto=GENERAL_AGENT,
        )

    logger.info(
        "planner: %s (+%s) → %s | deferred=%s",
        plan.primary_module,
        plan.support_modules or "—",
        _route_from_plan(plan),
        plan.deferred_intents or "—",
    )

    return Command(
        update={
            "detected_language": detected_language,
            "plan": plan,
            # Centinela de reset de merge_agent_results.
            "agent_results": None,
        },
        goto=_route_from_plan(plan),
    )


# ================================================================
# CLUSTER NODES
# ================================================================

def _cluster_node(
    state: PoolAgentState,
    cluster_name: str,
    config: RunnableConfig | None = None,
) -> Command:
    """
    Cuerpo compartido de los tres nodos de cluster.

    AQUATIC_CHEM y SYSTEMS corren con grafo de agente (declaran tools de
    cálculo). GOVERNANCE declara tools=() y corre con _direct_answer: un
    grafo cuyo nodo de tools es inalcanzable no aporta nada.
    """
    plan = state.get("plan")
    if plan is None:
        logger.error("%s: state sin plan", cluster_name)
        return Command(
            update=_resolve_and_update_archetype(
                plan=None,
                agent_results={},
                error="MISSING_PLAN",
                force_archetype="conversational",
            ),
            goto="synthesizer",
        )

    primary = _normalize_agent(plan.primary_module)
    support = list(plan.support_modules or [])
    revising = bool(state.get("judge_feedback"))

    remaining = _remaining_budget(state)
    # En una revisión hay que dejarle presupuesto al synthesizer, que
    # todavía no corrió. Sin este techo, una revisión lenta consume el turno
    # y el usuario no recibe nada.
    ceiling = CLUSTER_DEADLINE_S if not revising else CLUSTER_DEADLINE_S / 2
    deadline_s = max(MIN_STEP_BUDGET_S, min(ceiling, remaining))

    logger.info(
        "%s: primary=%s support=%s revising=%s deadline=%.1fs",
        cluster_name, primary, support or "—", revising, deadline_s,
    )

    if is_tooled(cluster_name):
        text, err = _invoke_cluster_agent(
            state, cluster_name, primary, support, revising, deadline_s, config
        )
    else:
        text, err = _direct_answer(
            state,
            build_cluster_prompt(
                cluster_name=cluster_name,
                primary_module=primary,
                support_modules=support,
                revising=revising,
            ),
            deadline_s=deadline_s,
        )

    result = AgentResult(
        agent=primary,
        output=text,
        sources=[],
        error=err,
        status="ok" if text and not err else "failed",
    )

    # Clave constante (RESULT_KEY): la revisión SOBREESCRIBE el intento
    # rechazado en vez de convivir con él en agent_results. Con claves
    # distintas, el synthesizer vería las dos versiones.
    update = _resolve_and_update_archetype(
        plan=plan,
        agent_results={RESULT_KEY: result},
    )

    return Command(update=update, goto="judge")


def _invoke_cluster_agent(
    state: PoolAgentState,
    cluster_name: str,
    primary: str,
    support: list[str],
    revising: bool,
    deadline_s: float,
    config: RunnableConfig | None = None,
) -> tuple[str, str | None]:
    """Construye e invoca el agente del cluster con techo de wall-clock."""
    plan = state.get("plan")
    user_message = _last_human_text(state)
    task = getattr(plan, "task", None) or user_message

    content = f"Task: {task}\n\nUser context: {user_message}"

    feedback = state.get("judge_feedback")
    if feedback:
        # El REVISION_PROTOCOL ya está en el system prompt vía revising=True;
        # acá va el objeto concreto que debe atender.
        content += f"\n\njudge_feedback: {feedback}"

    try:
        agent = build_cluster_agent(
            cluster_name=cluster_name,
            primary_module=primary,
            support_modules=support,
            revising=revising,
        )
    except Exception as exc:
        # ValueError de build_cluster_agent = bug de ruteo (módulo que no
        # pertenece al cluster). No se degrada en silencio.
        logger.exception("%s: no se pudo construir el agente", cluster_name)
        return "", f"AGENT_BUILD_FAILED: {exc}"

    if config is not None:
        begin_tool_scope(config.get("configurable", {}).get("thread_id", ""))

    def _invoke():
        return agent.invoke(
            {"messages": [HumanMessage(content=content)]},
            config={"recursion_limit": _recursion_limit_for(cluster_name)},
        )

    try:
        result = _run_with_deadline(_invoke, deadline_s)
    except FuturesTimeout:
        return "", f"STEP_DEADLINE_EXCEEDED after {deadline_s:.0f}s"
    except GraphRecursionError:
        return "", f"TOOL_BUDGET_EXCEEDED: {cluster_name} agotó su recursion_limit"
    except Exception as exc:
        err = str(exc).strip() or exc.__class__.__name__
        if exc.__class__.__name__ in _INFRA_EXC_NAMES and not _INFRA_CODE_RE.match(err):
            err = f"{exc.__class__.__name__}: {err}"
        return "", err

    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            return _extract_text(msg.content), None
    return "", "EMPTY_AGENT_OUTPUT"


@observe(as_type="agent", name="Aquatic Chem Node")
def aquatic_chem(
    state: PoolAgentState, config: RunnableConfig
) -> Command[Literal["judge", "synthesizer"]]:
    return _cluster_node(state, "AQUATIC_CHEM", config)


@observe(as_type="agent", name="Systems Node")
def systems(
    state: PoolAgentState, config: RunnableConfig
) -> Command[Literal["judge", "synthesizer"]]:
    return _cluster_node(state, "SYSTEMS", config)


@observe(as_type="agent", name="Governance Node")
def governance(
    state: PoolAgentState, config: RunnableConfig
) -> Command[Literal["judge", "synthesizer"]]:
    return _cluster_node(state, "GOVERNANCE", config)


# ================================================================
# DIRECT ANSWER (general, oos, GOVERNANCE)
# ================================================================

def _direct_answer(
    state: PoolAgentState,
    system_prompt: str,
    deadline_s: float = CLUSTER_DEADLINE_S,
) -> tuple[str, str | None]:
    """Una sola llamada al LLM con deadline, sin grafo de agente ni tools."""
    plan = state.get("plan")
    user_message = _last_human_text(state)
    task = getattr(plan, "task", None) or user_message
    language = _LANGUAGE_MAP.get(
        state.get("detected_language", "es"), _LANGUAGE_MAP["es"]
    )

    content = f"Task: {task}\n\nUser context: {user_message}"
    feedback = state.get("judge_feedback")
    if feedback:
        content += f"\n\njudge_feedback: {feedback}"

    try:
        def _invoke():
            return _get_llm().invoke([
                SystemMessage(content=f"{system_prompt}\n\nRespond in: {language}"),
                HumanMessage(content=content),
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


@observe(as_type="agent", name="General Node")
def general(state: PoolAgentState) -> Command[Literal["synthesizer"]]:
    plan = state.get("plan")

    remaining = _remaining_budget(state)
    deadline_s = max(MIN_STEP_BUDGET_S, min(CLUSTER_DEADLINE_S, remaining))

    text, err = _direct_answer(state, GENERAL_PROMPT, deadline_s=deadline_s)

    result = AgentResult(
        agent=GENERAL_AGENT,
        output=text,
        sources=[],
        error=err,
        status="ok" if text and not err else "failed",
    )

    update = _resolve_and_update_archetype(
        plan=plan,
        agent_results={RESULT_KEY: result},
        force_archetype="conversational",
        agent_message=text,
    )

    # general no pasa por el judge: emite prosa por diseño, no el JSON
    # estructurado que el judge evalúa.
    return Command(update=update, goto="synthesizer")


@observe(as_type="agent", name="OOS Node")
def oos(state: PoolAgentState) -> Command:
    plan = state.get("plan")

    remaining = _remaining_budget(state)
    deadline_s = max(MIN_STEP_BUDGET_S, min(CLUSTER_DEADLINE_S, remaining))

    text, err = _direct_answer(state, OOS_PROMPT, deadline_s=deadline_s)

    # ── MISROUTE ─────────────────────────────────────────────────────
    misroute_match = _MISROUTE_RE.match(text) if text else None

    if misroute_match:
        target_module = misroute_match.group(1).strip().lower()
        rest_text = misroute_match.group(2).strip()
        retries = state.get("misroute_retries", 0)

        if retries >= _MAX_MISROUTE_RETRIES:
            logger.warning(
                "oos: MISROUTE a '%s' agotó %d reintentos",
                target_module, _MAX_MISROUTE_RETRIES,
            )
            # `text` todavía empieza con "MISROUTE: <módulo>", señal de
            # control interna que nunca debe llegar al usuario. Se propaga
            # solo el resto; el synthesizer decide qué decir.
            result = AgentResult(
                agent=OOS_AGENT,
                output=rest_text,
                sources=[],
                error=f"MAX_MISROUTE_RETRIES_EXCEEDED: {target_module}",
                status="failed",
            )
            return Command(
                update=_resolve_and_update_archetype(
                    plan=plan,
                    agent_results={RESULT_KEY: result},
                    force_archetype="oos",
                ),
                goto="synthesizer",
            )

        if target_module in _MISROUTE_AGENTS:
            # Se reescribe el plan y se rutea al nodo real. 'general' está en
            # la whitelist y no tiene cluster: _route_from_plan lo manda a su
            # propio nodo.
            new_plan = PlanStep(
                primary_module=target_module,
                task=rest_text or _last_human_text(state),
                oos=False,
                deferred_intents=list(getattr(plan, "deferred_intents", []) or []),
            )
            destination = _route_from_plan(new_plan)
            logger.info(
                "oos: MISROUTE a '%s' → %s (intento %d)",
                target_module, destination, retries + 1,
            )
            return Command(
                update={
                    "plan": new_plan,
                    "misroute_retries": retries + 1,
                    "archetype": None,
                    "agent_results": None,
                },
                goto=destination,
            )

        logger.warning("oos: MISROUTE a módulo desconocido '%s'", target_module)

    # ── Camino normal ────────────────────────────────────────────────
    result = AgentResult(
        agent=OOS_AGENT,
        output=text,
        sources=[],
        error=err,
        status="ok" if text and not err else "failed",
    )

    update = _resolve_and_update_archetype(
        plan=plan,
        agent_results={RESULT_KEY: result},
        force_archetype="oos",
        agent_message=text,
    )

    return Command(update=update, goto="synthesizer")


# ================================================================
# JUDGE NODE
# ================================================================

JUDGE_REJECT_REASONS = (
    "incomplete_step",
    "dependency_broken",
    "unsourced_number",
    "contract_violation",
    "safety_gap",
)


class JudgeOutput(BaseModel):
    """
    Veredicto sobre la salida CRUDA del cluster, antes del synthesizer.

    Los reject_reason son exactamente los que REVISION_PROTOCOL (en
    prompts_sub_agents.py) le enseña al cluster a atender. Si se agrega uno
    acá, hay que agregarlo allá o el cluster no sabrá qué hacer con él.
    """

    verdict: Literal["accept", "revise"] = Field(
        description=(
            "'accept' if the output is usable as raw material for the "
            "synthesizer. 'revise' ONLY for a defect the cluster can actually "
            "fix by rewriting. A visible, correctly-declared gap is NOT a "
            "defect: an answer that says what it could not establish is "
            "acceptable and must be accepted."
        )
    )

    reject_reason: Optional[Literal[
        "incomplete_step",
        "dependency_broken",
        "unsourced_number",
        "contract_violation",
        "safety_gap",
    ]] = Field(
        default=None,
        description="Required when verdict is 'revise'. Omit when accepting.",
    )

    observations: str = Field(
        default="",
        description=(
            "When revising: the specific defect, in one or two sentences, "
            "actionable enough for the cluster to fix without guessing. Name "
            "what is wrong, not how to write it better. Empty when accepting."
        ),
    )


JUDGE_PROMPT = """You review the raw output of a specialist cluster in a pool and
spa assistant. The user never sees this output — a synthesizer turns it into
prose afterwards. You are judging whether it is sound raw material, not whether
it reads well.

Reject ONLY for a defect the cluster can fix by rewriting:
- incomplete_step: the assigned task was not actually answered.
- dependency_broken: a stated value was ignored or contradicted downstream.
- unsourced_number: a numeric result is asserted with no calculation behind it.
  A number the cluster explicitly declines to compute is NOT this defect.
- contract_violation: a required output field is missing or malformed.
- safety_gap: the content involves chemical handling, dosing, or a direct
  physical hazard, and the required warning is absent.

Accept everything else. In particular, ACCEPT when:
- The output declares what it could not establish. A precise gap is a complete
  answer in this system, and rejecting it produces an invented answer instead.
- The output asks the user for a missing input rather than assuming one.
- The output states that part of the request falls outside its scope.
- You would only be improving tone, length, structure, or wording. That is the
  synthesizer's job, not the cluster's.

Bias toward accepting. A revision costs the user a full extra round trip, so
the defect must be worth that. When you are unsure, accept.

ASSIGNED TASK:
{task}

CLUSTER OUTPUT:
{output}
"""


@observe(as_type="agent", name="Judge Node")
def judge(state: PoolAgentState) -> Command:
    """
    Evalúa la salida cruda del cluster y decide: al usuario, o de vuelta al
    cluster con observaciones.

    Tres reglas de diseño, todas para que el judge no pueda comerse el turno:

    1. FAIL-OPEN. Cualquier excepción -> accept. Si el judge bloquea al
       fallar, un 429 del proveedor le cuesta al usuario la respuesta entera,
       que ya estaba lista.
    2. COTA DURA. _MAX_JUDGE_RETRIES revisiones y se acepta lo que haya.
    3. PRESUPUESTO. Si no queda tiempo para una revisión más el synthesizer,
       se acepta sin siquiera llamar al modelo.

    Al aprobar hace fan-out a synthesizer y suggester en paralelo: el
    suggester ya no depende de state["response"] (sus candidatos salen del
    output del cluster y de plan.deferred_intents), así que sus ~6s no tienen
    por qué sumarse al camino crítico.
    """
    plan = state.get("plan")
    results = _normalize_agent_results(state.get("agent_results") or {})
    result = results.get(RESULT_KEY)
    retries = state.get("judge_retries", 0)

    approve = Command(
        update={"judge_feedback": None},
        goto=["synthesizer", "suggester"],
    )

    # Un cluster que falló no se revisa: el defecto es de infraestructura, no
    # de contenido. El synthesizer sabe decir qué quedó sin responder.
    if result is None or not result.output or result.error:
        logger.info(
            "judge: accept sin evaluar (result=%s)",
            "ausente" if result is None else (result.error or "output vacío"),
        )
        return approve

    if retries >= _MAX_JUDGE_RETRIES:
        logger.info("judge: accept por cota de reintentos (%d)", retries)
        return approve

    remaining = _remaining_budget(state)
    if remaining < _REVISION_MIN_BUDGET_S:
        logger.info(
            "judge: accept por presupuesto (%.1fs < %.1fs)",
            remaining, _REVISION_MIN_BUDGET_S,
        )
        return approve

    deadline_s = max(MIN_STEP_BUDGET_S, min(JUDGE_DEADLINE_S, remaining))

    try:
        chain = _get_llm().with_structured_output(JudgeOutput)
        messages = [
            SystemMessage(content=JUDGE_PROMPT.format(
                task=getattr(plan, "task", "") or "(no task recorded)",
                output=result.output,
            )),
            HumanMessage(content="Return your verdict now."),
        ]
        verdict: JudgeOutput = _run_with_deadline(
            lambda: chain.invoke(messages), deadline_s
        )
    except Exception as exc:
        # Fail-open. Se loguea para ver la distribución real de fallas en
        # Langfuse, pero nunca se propaga.
        logger.warning("judge: accept por fallo (%s: %s)", type(exc).__name__, exc)
        return approve

    if verdict.verdict == "accept":
        logger.info("judge: accept")
        return approve

    if not verdict.reject_reason:
        # 'revise' sin motivo es inaccionable: el cluster no sabría qué
        # arreglar y gastaríamos un round trip para nada.
        logger.warning("judge: revise sin reject_reason; se acepta")
        return approve

    feedback = {
        "reject_reason": verdict.reject_reason,
        "observations": verdict.observations,
    }
    destination = _route_from_plan(plan)
    logger.info(
        "judge: revise (%s) → %s (intento %d)",
        verdict.reject_reason, destination, retries + 1,
    )

    return Command(
        update={
            "judge_feedback": feedback,
            "judge_retries": retries + 1,
            "archetype": None,
        },
        goto=destination,
    )


# ================================================================
# SYNTHESIZER NODE
# ================================================================

@observe(as_type="span", name="Synthesizer Node")
def synthesizer(state: PoolAgentState) -> dict:
    """
    Convierte el resultado del cluster en UNA respuesta en lenguaje natural.

    Invariante: el `answer` que sale de aquí es prosa. El cluster emite JSON
    estructurado (BASE_OUTPUT_CONTRACT); ese JSON es materia prima para el
    LLM de síntesis, nunca la respuesta. El único texto que puede pasar sin
    sintetizar es el de `general`, que ya produce prosa por diseño.
    """
    plan = state.get("plan")
    language_code: str = state.get("detected_language", "es")

    agent_results = _normalize_agent_results(state.get("agent_results") or {})

    # ============================================================
    # ÚNICO ATAJO: clarificación de `general`
    # ------------------------------------------------------------
    # Se decide por el PLAN, no por keywords en el texto. Buscar
    # "provide"/"need"/"volume" en la salida de un especialista da falso
    # positivo casi siempre y devolvía el JSON crudo al usuario.
    # ============================================================
    if _normalize_agent(getattr(plan, "primary_module", None)) == GENERAL_AGENT:
        single = agent_results.get(RESULT_KEY)
        if single and single.output and not single.error:
            logger.info("synthesizer: prosa directa de `general`")
            text = _strip_code_fences(single.output)
            payload = SynthesizerOutput(
                answer=text, actions=[], safety=None, details=[]
            )
            return {
                "archetype": state.get("archetype", "conversational"),
                "response": payload,
                "validation": {"direct_agent_message": True},
                "messages": [AIMessage(content=text, name="Marlin")],
            }

    # ============================================================
    # OOS / IDIOMA
    # ============================================================
    is_oos = _is_oos(plan)
    oos_instruction = _OOS_INSTRUCTION_ACTIVE if is_oos else _OOS_INSTRUCTION_INACTIVE
    language_instruction = _LANGUAGE_MAP.get(language_code, _LANGUAGE_MAP["es"])

    # ============================================================
    # RESULTADOS
    # ------------------------------------------------------------
    # `usable` alimenta _attach_sources. `raw_content` ve el turno COMPLETO,
    # incluidos los errores: sin eso el synthesizer no puede decir "no pude
    # verificar X" y degrada a un saludo genérico.
    # ============================================================
    usable = usable_results(agent_results)
    modules = _active_modules(plan)

    archetype = state.get("archetype")
    if not archetype:
        logger.warning(
            "synthesizer: 'archetype' ausente del state. Degradando a "
            "'conversational'."
        )
        archetype = "conversational"

    raw_content = _build_raw_content(agent_results)
    if not raw_content:
        if plan is not None:
            # Se ejecutó un plan y no llegó nada: es un bug de escritura de
            # estado, no un turno vacío. El saludo NUNCA es correcto acá.
            logger.error(
                "synthesizer: raw_content vacío con plan presente "
                "(primary=%s) — algún nodo no escribió en agent_results",
                getattr(plan, "primary_module", "?"),
            )
            return _empty_results_fallback(state, reason="empty_raw_content_with_plan")
        raw_content = "(no prior content — generate a warm greeting and offer help)"
        archetype, modules, usable = "conversational", [], []

    contract = get_contract(archetype)

    # ============================================================
    # PROMPT
    # ============================================================
    system_content = SYNTHESIZER_PROMPT.format(
        archetype_section=build_synthesizer_archetype_section(archetype, modules),
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
            "synthesizer: structured output falló (%s); reintentando sin estructura",
            exc,
        )
        try:
            raw = _get_llm().invoke(llm_messages)
            payload = fallback_payload(_extract_text(raw.content), SynthesizerOutput)
            validation = {"fallback": "unstructured", "reason": str(exc)}
        except Exception as exc2:
            # 429 / 503 / 504: el reintento falla por la misma causa que el
            # primero. Payload estático: cero red, cero enforcement.
            logger.error(
                "synthesizer: ambos intentos de generación fallaron (%s | %s)",
                exc, exc2,
            )
            payload = static_service_unavailable_payload(
                SynthesizerOutput, language_code
            )
            return {
                "archetype": archetype,
                "response": payload,
                "validation": {"fallback": "static", "reason": f"{exc} | {exc2}"},
                "messages": [
                    AIMessage(content=payload.tier1_markdown(), name="Marlin")
                ],
            }

    # ============================================================
    # FASE 2 — enforcement (siempre, venga el payload de donde venga)
    # ============================================================
    try:
        payload, report = enforce_contract(
            payload, contract, modules, detail_cls=DetailSection
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

_SUPPRESSED_ARCHETYPES = frozenset({"critical", "conversational", "oos"})
_IGNORED_CHIP_LIMIT = 2


def _suggest_block_reason(state: PoolAgentState) -> str | None:
    """
    El motivo por el que este turno no lleva chips, o None si sí los lleva.

    Separado para que el nodo pueda loguear cuál gate cortó. El orden es
    intencional: lo más barato y más frecuente primero.
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

    # PROXY de answer_ends_with_question. Corriendo en paralelo al
    # synthesizer, state["response"] todavía no existe, así que no se puede
    # verificar si la respuesta final cierra con una pregunta. Se usa el
    # output del cluster: si declaró información faltante, el synthesizer
    # casi seguro va a cerrar preguntando, y un chip encima es ruido.
    #
    # Es heurística, no la verificación real: un falso negativo pone un chip
    # sobre una pregunta abierta. Ese es el costo aceptado del fan-out.
    output = _cluster_output_text(state).lower()
    if "missing_information" in output and '"missing_information": []' not in output:
        return "cluster_reported_missing_information"

    return None


def should_suggest(state: PoolAgentState) -> bool:
    """Lógica pura, cero llamadas al LLM. Corre antes de cualquier gasto de cuota."""
    return _suggest_block_reason(state) is None


def _leftover_material(state: PoolAgentState) -> list[str]:
    """
    Material del que pueden salir chips.

    Reemplaza a _unconsumed_entities, que leía los nodos que el retrieval
    tocó vía turn_cache. Sin retrieval ese cache está siempre vacío, así que
    el suggester habría devuelto [] en el 100% de los turnos sin un solo
    error en el log.

    Dos fuentes, en orden de calidad:
      1. plan.deferred_intents: sub-intents que el usuario planteó y que este
         turno no cubrió por ser de otro cluster. Son intenciones reales, no
         adivinanzas.
      2. Las líneas del output del cluster que declaran algo sin resolver.
    """
    plan = state.get("plan")
    material: list[str] = list(getattr(plan, "deferred_intents", None) or [])

    output = _cluster_output_text(state)
    if output:
        for line in output.splitlines():
            stripped = line.strip()
            if not stripped or len(stripped) > 200:
                continue
            lowered = stripped.lower()
            if any(
                marker in lowered
                for marker in ("missing_information", "unresolved", "escalation_target")
            ):
                material.append(stripped)

    # Dedup preservando orden.
    seen: set[str] = set()
    out: list[str] = []
    for item in material:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _build_answered_summary(state: PoolAgentState) -> str:
    """
    Lo que el turno respondió.

    En paralelo al synthesizer no existe state["response"], así que se usa el
    output crudo del cluster. Es un superconjunto de lo que el synthesizer va
    a conservar, o sea que el filtro de redundancia queda MÁS agresivo:
    puede descartar un candidato que el synthesizer habría dejado afuera.
    Alineado con la filosofía conservadora de los gates.
    """
    output = _cluster_output_text(state)
    return output or "(sin respuesta disponible)"


def _format_material(items: list[str]) -> str:
    if not items:
        return "(ninguna)"
    return "\n".join(f"- {i}" for i in items)


@observe(as_type="agent", name="Suggester")
def suggester(state: PoolAgentState, config: RunnableConfig) -> dict:
    """
    Produce los chips de seguimiento del turno.

    Corre EN PARALELO al synthesizer (fan-out desde el judge). Escribe solo
    `suggestions`; el synthesizer escribe archetype/response/validation/
    messages, así que no hay canales en conflicto.

    Devuelve SIEMPRE la clave "suggestions" — nunca la omite, para que el
    frontend pueda distinguir "no hubo chips" de "el nodo no corrió".
    """
    blocked = _suggest_block_reason(state)
    if blocked:
        logger.info("suggester skipped: %s", blocked)
        return {"suggestions": []}

    material = _leftover_material(state)
    if not material:
        # Sin material libre el LLM solo puede inventar. Ahorramos la llamada:
        # es el caso más común en turnos que cubrieron todo lo preguntado.
        logger.info("suggester skipped: no_leftover_material")
        return {"suggestions": []}

    language_code = "es" if state.get("detected_language") == "es" else "en"
    language = "español" if language_code == "es" else "English"
    answer_text = _build_answered_summary(state)

    system_content = SUGGESTER_PROMPT.format(
        language=language,
        roster=roster_text(),
        answered_summary=answer_text,
        unconsumed_entities=_format_material(material),
    )

    messages = [
        SystemMessage(content=system_content),
        # Última posición del prompt = mayor peso para la elección de idioma.
        # Con el kickoff hardcodeado en español, los turnos en inglés
        # devolvían chips en español pese al {language} del system prompt.
        HumanMessage(
            content=_SUGGESTER_KICKOFF.get(language_code, _SUGGESTER_KICKOFF["en"])
        ),
    ]

    try:
        chain = _get_llm_suggester().with_structured_output(SuggesterOutput)
        # El deadline lo impone el executor, no el cliente: la API exige
        # timeout >= 10s y no queremos esperar tanto.
        #
        # NO usar `with ThreadPoolExecutor(...)`: su __exit__ hace
        # shutdown(wait=True), así que aunque .result() lance el timeout, el
        # bloque espera la llamada completa igual.
        #
        # copy_context() propaga el parent OTel de Langfuse al thread.
        ctx = contextvars.copy_context()
        future = _SUGGESTER_POOL.submit(ctx.run, chain.invoke, messages)
        try:
            payload: SuggesterOutput = future.result(timeout=_SUGGESTER_DEADLINE_S)
        except FuturesTimeout:
            future.cancel()  # no mata el thread en curso, solo libera el slot
            raise
    except Exception as exc:
        # Todo se degrada igual: 429, timeout del executor, o structured
        # output inválido. Un chip opcional no rompe el turno del usuario.
        logger.warning("suggester degraded to []: %s: %s", type(exc).__name__, exc)
        return {"suggestions": []}

    raw: List[Suggestion] = payload.suggestions or []
    gated, report = apply_gates_with_report(raw, answer_text)

    # El reporte va al log, no al state: es telemetría de calidad del prompt,
    # no algo que el frontend consuma.
    if report["input"] != report["output"]:
        logger.info("suggester gates: %s", report)

    logger.info("suggester: %d chips generados", len(gated))
    return {"suggestions": gated}