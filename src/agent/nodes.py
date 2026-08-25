from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send
import logging
from langfuse import observe, get_client
from typing import List, Literal
import concurrent.futures
from .state import PoolAgentState, ExecutionStep, AgentResult
from ..prompts.prompts import PLANNER_PROMPT, SYNTHESIZER_PROMPT, SUGGESTER_PROMPT
from .chains import create_planner_chain
from ..config.llm import create_llm, create_suggester_llm
from .agents import get_agent_by_name

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

# ================================================================
# HELPERS
# ================================================================
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

def build_context_node(
    state: PoolAgentState,
) -> Command[Literal["summarize_memory_node", "planner"]]:
    next_node: Literal["summarize_memory_node", "planner"] = (
        "summarize_memory_node"
        if estimated_tokens(state["messages"]) > TOKEN_LIMIT
        else "planner"
    )
    return Command(goto=next_node)

# ================================================================
# SUMMARIZE MEMORY NODE
# ================================================================

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
                "current_step": 1,
                "agent_results": {
                    "step_1": AgentResult(
                        agent="planner", step=1, output="", error=str(e)
                    )
                },
            },
            goto="synthesizer",
        )

    detected_language = plan.detected_language or fallback_language

    return Command(
        update={
            "detected_language": detected_language,
            "execution_plan": plan.execution_plan,
            "current_step": 0,
            "agent_results": {},
        },
        goto="orchestrator",
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

@observe(as_type="agent", name="Orchestrator Node")
def orchestrator(state: PoolAgentState) -> Command:
    execution_plan = state.get("execution_plan", [])
    agent_results  = state.get("agent_results") or {}

    if not execution_plan:
        return Command(
            update={"error": "execution_plan is empty; cannot orchestrate."},
            goto="synthesizer",
        )

    done_keys = set(agent_results.keys())
    pending   = [s for s in execution_plan if f"step_{s.step}" not in done_keys]

    if not pending:
        return Command(goto="synthesizer")

    ready = [
        s for s in pending
        if all(f"step_{d}" in done_keys for d in s.depends_on)
    ]

    if not ready:
        # hay steps pendientes pero ninguno con dependencias resueltas
        # -> plan mal formado (ciclo o depends_on inválido)
        return Command(
            update={"error": f"Deadlock in execution_plan: {[s.step for s in pending]} blocked."},
            goto="synthesizer",
        )

    messages = state.get("messages", [])
    user_message = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            user_message = _extract_text(msg.content)
            break

    return Command(
        goto=[
            Send("run_step", {"step": s, "user_message": user_message})
            for s in ready
        ]
    )


@observe(as_type="agent", name="Run Step Node")
def run_step_node(payload: dict) -> Command:
    step         = payload["step"]
    user_message = payload["user_message"]
    step_key     = f"step_{step.step}"

    try:
        agent_result = _run_step(step, user_message)
    except Exception as exc:
        agent_result = AgentResult(
            agent=step.assigned_agent,
            step=step.step,
            output="",
            error=str(exc),
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