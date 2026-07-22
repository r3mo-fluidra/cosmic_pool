from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, BaseMessage, RemoveMessage
from langgraph.types import Command
from langfuse import observe
from typing import List, Literal
from .state import PoolAgentState, ExecutionStep, AgentResult
from .prompts import PLANNER_PROMPT, SYNTHESIZER_PROMPT
from .chains import create_planner_chain
from ..config.llm import create_llm
from .agents import get_agent_by_name

# ================================================================
# CONFIGURATION
# ================================================================

TOKEN_LIMIT = 4_000
MESSAGES_TO_KEEP = 6

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

@observe(as_type='trace', name="Planner Node")
def planner(state: PoolAgentState):
    user_input = state["messages"][-1].content

    agent_messages = [
        m for m in state["messages"]
        if isinstance(m, AIMessage) and getattr(m, "name", None) == "Izel"
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

    # ✅ Lazy — planner chain se inicializa solo aquí
    plan = _get_planner_chain().invoke([
        {"role": "system", "content": PLANNER_PROMPT},
        {"role": "user",   "content": context_for_planner},
    ])

    detected_language = plan.detected_language or state.get("detected_language") or "es"

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

@observe(as_type="trace", name="Orchestrator Node")
def orchestrator(state: PoolAgentState) -> Command:
    execution_plan = state.get("execution_plan", [])
    agent_results  = dict(state.get("agent_results") or {})
    current_idx    = state.get("current_step", 0)

    if not execution_plan:
        return Command(
            update={"error": "execution_plan is empty; cannot orchestrate."},
            goto="synthesizer",
        )

    if current_idx >= len(execution_plan):
        return Command(goto="synthesizer")

    step     = execution_plan[current_idx]
    step_key = f"step_{step.step}"

    messages = state.get("messages", [])
    user_message = ""
    for msg in reversed(messages):
        if hasattr(msg, "type") and msg.type == "human":
            user_message = _extract_text(msg.content)
            break

    try:
        agent_result = _run_step(step, user_message)
    except Exception as exc:
        agent_result = AgentResult(
            agent=step.assigned_agent,
            step=step.step,
            output="",
            error=str(exc),
        )

    agent_results[step_key] = agent_result

    next_idx = current_idx + 1
    goto = "orchestrator" if next_idx < len(execution_plan) else "synthesizer"

    return Command(
        update={
            "agent_results": agent_results,
            "current_step": next_idx,
        },
        goto=goto,
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

    raw_content = _build_raw_content(agent_results)
    if not raw_content:
        raw_content = "(no prior content — generate a warm greeting and offer help)"

    system_content = SYNTHESIZER_PROMPT.format(
        oos_instruction=oos_instruction,
        language=language_instruction,
        raw_content=raw_content,
    )

    # ✅ Lazy — LLM se inicializa solo aquí
    response = _get_llm().invoke([
        SystemMessage(content=system_content),
        HumanMessage(content="Generate the final refined response now."),
    ])

    content = response.content
    if isinstance(content, list):
        final_text = " ".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("text", "").strip()
        ).strip()
    else:
        final_text = str(content).strip()

    return {
        "messages": [AIMessage(content=final_text, name="Izel")]
    }