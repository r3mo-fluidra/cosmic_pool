import os
import uuid
import time
import streamlit as st
from neo4j import GraphDatabase
from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler
from langchain_core.messages import HumanMessage, AIMessage

# Import your compiled LangGraph workflow
from src.agent.graph import graph
from src.ui.theme import (
    SCREEN_HEIGHT_PX,
    assistant_label,
    inject_theme,
    page_footer,
    page_header,
    phone_header,
    role_marker,
    theme_slider,
)

from dotenv import load_dotenv

load_dotenv()

def get_config(key: str, default=None):
    """
    Get configuration from environment variables first, then Streamlit secrets.

    Local development:
        .env / environment variables

    Streamlit deployment:
        .streamlit/secrets.toml

    Never raises StreamlitSecretNotFoundError when secrets.toml is absent.
    """
    value = os.getenv(key)

    if value:
        return value

    try:
        return st.secrets[key]
    except (FileNotFoundError, KeyError):
        return default

# ==========================================
# CONFIGURATION & LANGFUSE SETUP
# ==========================================
st.set_page_config(
    page_title="IA Pool Assistant V1",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Deep Water (dark) / Sunlit Lagoon (light). Must run before any other widget:
# it reads the slider's session-state value and paints every surface below.
inject_theme()

# --- page copy -------------------------------------------------------------
PAGE_TITLE = "IA Pool Assistant V1"
PAGE_SUBTITLE = (
    "Conversational assistant for pool water chemistry, equipment and "
    "maintenance, grounded in the product documentation and a curated pool "
    "knowledge graph. Ask in English or Spanish — it answers in the language "
    "you write in."
)
PHONE_BRAND = "Pool Assistant"
FOOTER_TEAM = "AI/ML Team — Bogotá, Colombia"
FOOTER_PHASE = "UX/UI first test phase"

# --- opening turn ----------------------------------------------------------
GREETING = (
    "Good evening. I'm your pool assistant — tell me what's going on and "
    "we'll figure it out together."
)

OPENERS = [
    "My water looks cloudy",
    "I saw a warning in the app",
    "I have a maintenance question",
]


def initial_messages() -> list[dict]:
    """The conversation as it looks before the user has said anything."""
    return [{"role": "assistant", "content": GREETING, "opening": True}]

SHOW_PER_ANSWER_FEEDBACK = False

IGNORED_CHIP_LIMIT = 2

# --- Score naming ---
SCORE_FEEDBACK = "user_feedback"
SCORE_REASON = "feedback_reason"
TRACE_NAME = "pool-chemistry-turn"

FEEDBACK_REASONS = {
    "Incorrect information": "incorrect_information",
    "Incomplete answer": "incomplete_answer",
    "Misunderstood my question": "misunderstood_question",
    "Unsafe chemical advice": "unsafe_chemical_advice",
    "Other": "other",
}


@st.cache_resource
def get_langfuse():
    """
    Single cached Langfuse client. Instantiating Langfuse() on every Streamlit
    rerun would spawn a new background thread + queue each time.

    environment/release are set here so scores and traces coming from dev runs
    can be filtered out of the console without touching the score payload.
    """
    try:
        client = Langfuse(
            environment=get_config("APP_ENV", "development"),
            release=get_config("APP_RELEASE", "local"),
        )
        if not client.auth_check():
            return None
        return client
    except Exception:
        return None


lf = get_langfuse()
if lf is None:
    st.sidebar.warning("Langfuse credentials not found or invalid. Tracking disabled.")


# ==========================================
# IDEMPOTENT SCORING HELPERS
# ==========================================
def build_score_id(trace_id: str, score_name: str) -> str:
    """
    Deterministic score id. Langfuse upserts by score id, so re-sending the same
    id overwrites instead of appending. This is the actual idempotency guarantee:
    N clicks -> exactly 1 score row, and flipping the verdict mutates the value.
    """
    return f"{trace_id}-{score_name}"


def submit_feedback(
    trace_id: str,
    value: int,
    comment: str,
    reason: str | None = None,
    turn_index: int | None = None,
):
    """
    Sends (or overwrites) the feedback score for a given trace.
    value: 1 = thumbs up, 0 = thumbs down (data_type NUMERIC)

    score metadata carries the conversation coordinates, so a score row is
    traceable back to a specific turn even when queried through the scores API
    in isolation from its trace.
    """
    if lf is None:
        st.toast("Langfuse deshabilitado: el feedback no se registró.")
        return False

    score_metadata = {
        "thread_id": st.session_state.thread_id,
        "turn_index": turn_index,
        "source": "streamlit_ui",
        "submitted_at": time.time(),
    }

    try:
        lf.create_score(
            score_id=build_score_id(trace_id, SCORE_FEEDBACK),
            trace_id=trace_id,
            name=SCORE_FEEDBACK,
            value=value,
            data_type="NUMERIC",
            comment=comment,
            metadata=score_metadata,
        )

        if reason is not None:
            lf.create_score(
                score_id=build_score_id(trace_id, SCORE_REASON),
                trace_id=trace_id,
                name=SCORE_REASON,
                value=reason,
                data_type="CATEGORICAL",
                comment=comment,
                metadata=score_metadata,
            )

        lf.flush()
        return True
    except Exception as e:
        st.error(f"No se pudo registrar el feedback: {e}")
        return False


# ==========================================
# CHIP TELEMETRY
# ==========================================
def log_chip_event(trace_id: str, event: str, payload: dict) -> None:
    """
    Records chip impressions and taps as Langfuse scores, which is what feeds
    the metrics in step 9 of the plan: emission rate, tap rate, and which
    entity/agent pairs actually get tapped.

    Fire-and-forget: chip telemetry must never break a turn, so every failure
    is swallowed. Uses a deterministic score id per (trace, event) so a rerun
    upserts instead of duplicating.
    """
    if lf is None:
        return
    try:
        lf.create_score(
            score_id=build_score_id(trace_id, f"chip_{event}"),
            trace_id=trace_id,
            name=f"chip_{event}",
            value=payload.get("count", 1),
            data_type="NUMERIC",
            metadata={
                "thread_id": st.session_state.thread_id,
                **payload,
            },
        )
        lf.flush()
    except Exception:
        pass


# ==========================================
# NEO4J CONNECTION HANDLER
# ==========================================
@st.cache_resource
def get_neo4j_driver():
    uri = get_config("NEO4J_URI")
    user = get_config("NEO4J_USER")
    password = get_config("NEO4J_PASSWORD")

    # Safe diagnostics — NEVER print the password
    print("=== Neo4j configuration ===")
    print(f"NEO4J_URI: {uri}")
    print(f"NEO4J_USER: {user}")
    print(f"NEO4J_PASSWORD exists: {bool(password)}")

    if not uri:
        raise RuntimeError("NEO4J_URI is missing")

    if not user:
        raise RuntimeError("NEO4J_USER is missing")

    if not password:
        raise RuntimeError("NEO4J_PASSWORD is missing")

    return GraphDatabase.driver(
        uri,
        auth=(user, password),
    )


def check_and_handle_neo4j() -> tuple[bool, str]:
    """
    Checks Neo4j connectivity. Returns (online, message).

    The message is returned rather than rendered, so the caller can show it
    inside the phone screen. Rendering it here would paint a full-width alert
    above the page title and break the framed layout.

    Note: If using Neo4j Aura Free Tier, it pauses automatically.
    Standard Python drivers cannot "unpause" Aura. You must use the Aura API
    to programmatically resume it, or resume it via the Neo4j Console.
    """
    driver = get_neo4j_driver()
    if not driver:
        return False, "Neo4j environment variables are missing."

    try:
        driver.verify_connectivity()
        return True, ""
    except Exception as e:
        if "getaddrinfo failed" in str(e) or "Timeout" in str(e):
            return False, (
                "**Neo4j Database is currently unreachable or paused.**\n\n"
                "If you are using Neo4j Aura Free Tier, it may have been paused due to inactivity. "
                "Please log into the [Neo4j Aura Console](https://console.neo4j.io/) to resume your instance."
            )
        return False, f"Neo4j Connection Error: {e}"


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = initial_messages()
if "feedback" not in st.session_state:
    # trace_id -> {"value": int, "reason": str|None, "awaiting_detail": bool}
    st.session_state.feedback = {}
if "turn_counter" not in st.session_state:
    # Monotonic per-session counter. len(messages) is NOT safe as a trace seed:
    # a turn that dies before appending its assistant message leaves the length
    # unchanged, and the next prompt would reuse the same deterministic id.
    st.session_state.turn_counter = 0
if "db_online" not in st.session_state:
    st.session_state.db_online, st.session_state.db_message = check_and_handle_neo4j()

# --- chip bookkeeping ------------------------------------------------------
# The suggester reads ignored_chip_streak from the graph state, but the UI is
# the only side that can observe whether a chip was tapped, so the UI owns the
# counter and ships it into the graph on every turn.
if "ignored_chip_streak" not in st.session_state:
    st.session_state.ignored_chip_streak = 0
if "chips_shown_last_turn" not in st.session_state:
    st.session_state.chips_shown_last_turn = False
if "last_turn_was_tap" not in st.session_state:
    st.session_state.last_turn_was_tap = False


# ==========================================
# OPENING SUGGESTIONS
# ==========================================
def render_openers(options: list[str], disabled: bool = False):
    """
    The openers under the greeting. A click stashes the text and reruns; the
    composer picks it up below and the turn runs exactly as if it were typed —
    so they follow the composer and go dead while the graph is unreachable.

    Container key stays "pa-chips" so the existing theme rule keeps applying.
    """
    with st.container(key="pa-chips"):
        for i, text in enumerate(options):
            if st.button(text, key=f"opener_{i}", disabled=disabled):
                st.session_state.pending_prompt = text
                st.rerun()


# ==========================================
# SUGGESTION CHIPS (suggester node)
# ==========================================
def render_chips(
    suggestions: list,
    turn_index: int,
    trace_id: str | None = None,
    disabled: bool = False,
):
    """
    Predicted next-question chips, rendered under the tier-1 answer. Zero chips
    is the normal case: the suggester suppresses itself far more often than it
    fires, so this returns early without drawing anything.

    A tap sends the label as a plain user message and resets the ignored streak.

    NOTE: the `agent` the suggester already resolved is discarded here — the tap
    re-enters through the planner like any typed message, costing one extra LLM
    call per tap. Removing that cost needs a route_hint path in the planner,
    which is not implemented yet.
    """
    if not suggestions:
        return

    # Decode suggestions - handle both dict and Pydantic model
    decoded = []
    for sug in suggestions:
        if isinstance(sug, dict):
            decoded.append(sug)
        elif hasattr(sug, "model_dump"):
            # Pydantic v2
            decoded.append(sug.model_dump())
        elif hasattr(sug, "dict"):
            # Pydantic v1
            decoded.append(sug.dict())
        else:
            # Fallback: try to access attributes directly
            decoded.append({
                "label": getattr(sug, "label", ""),
                "agent": getattr(sug, "agent", ""),
                "entity": getattr(sug, "entity", ""),
            })

    with st.container(key=f"pa-suggest-{turn_index}"):
        # Show a small header for chips
        st.caption("Sugerencias:")
        cols = st.columns(min(len(decoded), 3))
        
        for i, sug in enumerate(decoded):
            label = sug.get("label", "")
            agent = sug.get("agent", "")
            entity = sug.get("entity", "")

            if not label:
                continue

            # Distribute buttons across columns
            col_idx = i % len(cols)
            with cols[col_idx]:
                # Add tooltip with agent info
                tooltip = f"Agent: {agent}" if agent else ""
                if st.button(
                    label, 
                    key=f"chip_{turn_index}_{i}",
                    disabled=disabled,
                    help=tooltip,
                    use_container_width=True,
                ):
                    st.session_state.pending_prompt = label
                    st.session_state.last_turn_was_tap = True
                    st.session_state.ignored_chip_streak = 0
                    if trace_id:
                        log_chip_event(
                            trace_id,
                            "tap",
                            {
                                "turn_index": turn_index,
                                "position": i,
                                "label": label,
                                "agent": agent,
                                "entity": entity,
                            },
                        )
                    st.rerun()


# ==========================================
# FEEDBACK WIDGET
# ==========================================
def render_feedback(trace_id: str, turn_index: int | None = None):
    """
    Rendered from the history loop (NOT from inside the chat_input block), so it
    survives the rerun that a button click triggers.
    """
    if not SHOW_PER_ANSWER_FEEDBACK:
        return

    state = st.session_state.feedback.get(trace_id)

    # --- Phase 3: thumbs-down detail form ---
    if state and state.get("awaiting_detail"):
        st.caption("Registrado. ¿Nos contás qué falló? (opcional)")
        label = st.selectbox(
            "Motivo principal",
            options=list(FEEDBACK_REASONS.keys()),
            key=f"reason_{trace_id}",
        )
        detail = st.text_area(
            "Detalle",
            key=f"detail_{trace_id}",
            placeholder="Ej: recomendó 3x la dosis de cloro para mi volumen de pileta.",
            height=80,
        )
        c1, c2, _ = st.columns([1, 1, 6])
        with c1:
            if st.button("Enviar", key=f"send_{trace_id}", type="primary"):
                reason = FEEDBACK_REASONS[label]
                comment = detail.strip() or f"({reason}) sin detalle"
                if submit_feedback(trace_id, 0, comment, reason=reason, turn_index=turn_index):
                    st.session_state.feedback[trace_id] = {
                        "value": 0,
                        "reason": reason,
                        "awaiting_detail": False,
                    }
                    st.toast("¡Gracias por el detalle!")
                    st.rerun()
        with c2:
            if st.button("Omitir", key=f"skip_{trace_id}"):
                st.session_state.feedback[trace_id] = {
                    "value": 0,
                    "reason": None,
                    "awaiting_detail": False,
                }
                st.rerun()
        return

    # --- Already scored: show status, allow changing it ---
    if state:
        verdict = "útil" if state["value"] == 1 else "no útil"
        extra = f" · `{state['reason']}`" if state.get("reason") else ""
        c1, c2 = st.columns([3, 7])
        with c1:
            st.caption(f"Feedback enviado · {verdict}{extra}")
        with c2:
            if st.button("Cambiar", key=f"change_{trace_id}"):
                del st.session_state.feedback[trace_id]
                st.rerun()
        return

    # --- Phase 1: no feedback yet ---
    c1, c2, _ = st.columns([1, 1, 8])
    with c1:
        if st.button("Me ayudó", key=f"good_{trace_id}"):
            if submit_feedback(trace_id, 1, "Respuesta útil", turn_index=turn_index):
                st.session_state.feedback[trace_id] = {
                    "value": 1,
                    "reason": None,
                    "awaiting_detail": False,
                }
                st.toast("¡Gracias por tu feedback!")
                st.rerun()
    with c2:
        if st.button("No me ayudó", key=f"bad_{trace_id}"):
            if submit_feedback(trace_id, 0, "Respuesta no útil (pendiente de detalle)", turn_index=turn_index):
                st.session_state.feedback[trace_id] = {
                    "value": 0,
                    "reason": None,
                    "awaiting_detail": True,
                }
                st.rerun()


# ==========================================
# UI: MAIN CHAT INTERFACE
# ==========================================
online = st.session_state.db_online

# Mode slider first: it sits in the top-right corner, above the title block.
theme_slider()

page_header(PAGE_TITLE, PAGE_SUBTITLE)

# ── The phone ─────────────────────────────────────────────────────────────
with st.container(key="pa-phone"):
    with st.container(key="pa-screen"):
        phone_header(
            PHONE_BRAND,
            "Online" if online else "Unavailable",
            online=online,
        )

        screen_scroll = st.container(height=SCREEN_HEIGHT_PX, key="pa-scroll")

        with screen_scroll:
            if not online:
                st.error(st.session_state.get("db_message", "Service unavailable."))

            # Chat history. Feedback rides on answers the assistant has
            # actually settled — see `is_definitive_answer`.
            for idx, msg in enumerate(st.session_state.messages):
                with st.chat_message(msg["role"]):
                    role_marker(msg["role"])
                    if msg["role"] == "assistant":
                        assistant_label()
                    st.markdown(msg["content"])
                    if msg["role"] == "assistant" and msg.get("can_rate"):
                        render_feedback(msg["trace_id"], turn_index=msg.get("turn_index"))

                    # Chips only on the latest assistant message. Older turns
                    # keep their suggestions in history for telemetry, but
                    # rendering them would offer stale next-steps for a
                    # conversation that already moved on.
                    is_last = idx == len(st.session_state.messages) - 1
                    if (
                        msg["role"] == "assistant"
                        and is_last
                        and msg.get("suggestions")
                    ):
                        render_chips(
                            msg["suggestions"],
                            turn_index=msg.get("turn_index", idx),
                            trace_id=msg.get("trace_id"),
                            disabled=not online,
                        )

                # Openers sit outside the greeting card, as in the prototypes,
                # and only while the greeting is still the whole conversation.
                if msg.get("opening") and len(st.session_state.messages) == 1:
                    render_openers(OPENERS, disabled=not online)

        # Nested inside a container, so it renders inline in the phone rather
        # than pinned to the bottom of the viewport.
        prompt = st.chat_input(
            "Ask anything…" if online else "Unavailable — the knowledge graph is offline",
            disabled=not online,
        )

# A clicked opener or chip behaves like a typed message. Popped unconditionally
# so a stale value can never fire a turn on a later rerun.
suggested = st.session_state.pop("pending_prompt", None)
if suggested and not prompt:
    prompt = suggested

page_footer(FOOTER_TEAM, FOOTER_PHASE)

# Reset stays available, kept quiet and off the phone.
with st.container(key="pa-reset", horizontal=True, horizontal_alignment="center"):
    if st.button("Reset conversation"):
        st.session_state.messages = initial_messages()
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.feedback = {}
        st.session_state.turn_counter = 0
        st.session_state.ignored_chip_streak = 0
        st.session_state.chips_shown_last_turn = False
        st.session_state.last_turn_was_tap = False
        st.rerun()


# Answers shorter than this that end in a question mark are read as the
# assistant still working the problem rather than closing it.
CLARIFYING_LENGTH = 320


def is_definitive_answer(plan, runs, text: str) -> bool:
    """
    Whether this turn settled the question, which is what decides if the
    feedback chips appear. Troubleshooting turns — the assistant asking for a
    reading, a symptom, a model number — get no chips: there is nothing to rate
    yet.

    The graph exposes no such flag, so it is derived from what the stream
    already carries:

      * a refusal (`oos`) is not an answer about the pool;
      * an answer nobody grounded — every step failed, or the only agents that
        ran were `general`/`oos` — is conversation, not a documented answer;
      * a short reply ending in a question mark is the assistant asking back.

    `plan` is the planner's ExecutionStep list, `runs` the (agent, error) pairs
    the orchestrator emitted. The durable version of this is a flag set by the
    synthesizer, which would remove the length heuristic — worth raising with
    the backend before this leaves the test phase.
    """
    text = (text or "").strip()
    if not text:
        return False
    if any(getattr(step, "oos", False) for step in plan):
        return False
    grounded = any(
        agent not in ("oos", "general") and not error for agent, error in runs
    )
    if not grounded:
        return False
    if len(text) < CLARIFYING_LENGTH and text.endswith("?"):
        return False
    return True


def run_turn(prompt: str, trace_id: str, turn_index: int) -> tuple[str, bool, list]:
    """
    Executes the graph and streams intermediate state into the UI.
    Separated from the trace plumbing so the span wrapper stays readable.

    Returns the final response, whether it is a definitive answer, and the
    suggester's chips (empty list is the common case).
    """
    config = {
        "configurable": {"thread_id": st.session_state.thread_id},
        "callbacks": [CallbackHandler()] if lf else [],
        "metadata": {
            "thread_id": st.session_state.thread_id,
            "turn_index": turn_index,
        },
    }

    final_response = ""
    plan_steps: list = []
    agent_runs: list[tuple[str, str | None]] = []
    suggestions: list = []

    for event in graph.stream(
        {
            "messages": [HumanMessage(content=prompt)],
            "ignored_chip_streak": st.session_state.ignored_chip_streak,
        },
        config=config,
        stream_mode="updates",
    ):
        # Debug: show what events we're receiving
        st.write(f"📨 Event received: {list(event.keys())}")
        
        if "planner" in event:
            plan = event["planner"].get("execution_plan", [])
            plan_steps = list(plan)
            st.markdown("**Planner Output:**")
            for step in plan:
                st.markdown(f"- Step {step.step} `[{step.assigned_agent}]`: {step.task}")

        elif "orchestrator" in event:
            orch_state = event.get("orchestrator")
            if isinstance(orch_state, dict):
                results = orch_state.get("agent_results", {})
                if results:
                    latest_key = sorted(results.keys())[-1]
                    latest_res = results[latest_key]

                    agent_name = getattr(latest_res, "agent", "unknown")
                    error_msg = getattr(latest_res, "error", None)
                    agent_runs.append((agent_name, error_msg))

                    if error_msg:
                        st.error(f"**{agent_name.upper()} Agent Error:** {error_msg}")
                    else:
                        st.info(f"**{agent_name.upper()} Agent executed successfully.**")
        
        elif "general" in event:
            st.info("**General Agent executed successfully.**")
            # Extract result if needed
            gen_state = event.get("general", {})
            if "messages" in gen_state:
                # The synthesizer will handle the final message
                pass
        
        elif "oos" in event:
            st.warning("**Out of Scope handler executed.**")
            oos_state = event.get("oos", {})
            if "messages" in oos_state:
                # The synthesizer will handle the final message
                pass

        elif "synthesizer" in event:
            messages = event["synthesizer"].get("messages", [])
            if messages:
                final_response = messages[-1].content
            
            # Also capture archetype and response for debugging
            archetype = event["synthesizer"].get("archetype")
            if archetype:
                st.write(f"📊 Archetype: {archetype}")

        elif "suggester" in event:
            # Parallel branch: can arrive before or after the synthesizer, so
            # nothing here may assume the answer already exists. An empty list
            # is the expected outcome most turns.
            suggester_state = event.get("suggester")
            if isinstance(suggester_state, dict):
                suggestions = suggester_state.get("suggestions") or []
                st.write(f"💡 Suggester output: {len(suggestions)} suggestions")
                if suggestions:
                    st.write("Suggestions:", suggestions)
                else:
                    st.info("No suggestions this turn (gate suppressed or no entities)")

    return (
        final_response,
        is_definitive_answer(plan_steps, agent_runs, final_response),
        suggestions,
    )


# User Input — the composer lives inside the phone, above.
if prompt:
    # Chip bookkeeping, before anything else touches the counters. If last turn
    # rendered chips and this prompt did not come from tapping one, they were
    # ignored; IGNORED_CHIP_LIMIT of those in a row and the suggester goes quiet.
    if st.session_state.chips_shown_last_turn and not st.session_state.last_turn_was_tap:
        st.session_state.ignored_chip_streak += 1
    st.session_state.last_turn_was_tap = False

    # Monotonic turn index -> the seed is unique even if a previous turn failed.
    st.session_state.turn_counter += 1
    turn_index = st.session_state.turn_counter

    # Deterministic, OTEL-compatible trace id (32 lowercase hex chars).
    current_trace_id = Langfuse.create_trace_id(
        seed=f"{st.session_state.thread_id}:{turn_index}"
    )

    st.session_state.messages.append({"role": "user", "content": prompt})
    with screen_scroll.chat_message("user"):
        role_marker("user")
        st.markdown(prompt)

    with screen_scroll.chat_message("assistant"):
        role_marker("assistant")
        assistant_label()
        with st.status("Agent Thinking Process...", expanded=True) as status:
            final_response = ""
            definitive = False
            suggestions: list = []
            turn_error: Exception | None = None

            if lf is not None:
                with lf.start_as_current_observation(
                    as_type="span",
                    name=TRACE_NAME,
                    trace_context={"trace_id": current_trace_id},
                    input={"prompt": prompt},
                ) as span:
                    propagate_attributes(
                        user_id=st.session_state.thread_id,
                        session_id=st.session_state.thread_id,
                        tags=["streamlit", "pool-chemistry"],
                        metadata={
                            "turn_index": turn_index,
                        },
                    )

                    try:
                        final_response, definitive, suggestions = run_turn(
                            prompt,
                            current_trace_id,
                            turn_index,
                        )

                        span.update(
                            output={
                                "response": final_response,
                                "suggestion_count": len(suggestions),
                            }
                        )

                        span.set_trace_io(
                            input={"prompt": prompt},
                            output={"response": final_response},
                        )

                    except Exception as e:
                        turn_error = e
                        span.update(
                            level="ERROR",
                            status_message=str(e),
                            output={"error": str(e)},
                        )
                        span.set_trace_io(
                            input={"prompt": prompt},
                            output={"error": str(e)},
                        )

                lf.flush()
            else:
                try:
                    final_response, definitive, suggestions = run_turn(
                        prompt, current_trace_id, turn_index
                    )
                except Exception as e:
                    turn_error = e

            if turn_error is not None:
                status.update(label="Turn failed", state="error", expanded=True)
                st.error(f"El agente falló: {turn_error}")
                st.stop()

            status.update(label="Response Generated", state="complete", expanded=False)

        if final_response:
            st.markdown(final_response)
            
            # Store the message with all metadata
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response,
                "trace_id": current_trace_id,
                "turn_index": turn_index,
                "can_rate": definitive,
                "suggestions": suggestions,
            })

            # Emission rate denominator: recorded whether or not chips appeared.
            st.session_state.chips_shown_last_turn = bool(suggestions)
            log_chip_event(
                current_trace_id,
                "impression",
                {
                    "turn_index": turn_index,
                    "count": len(suggestions),
                    "ignored_streak": st.session_state.ignored_chip_streak,
                },
            )

            # Rerun so the feedback widget renders from the history loop
            st.rerun()