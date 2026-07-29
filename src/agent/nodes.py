import os
import uuid
import streamlit as st
from neo4j import GraphDatabase
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langchain_core.messages import HumanMessage, AIMessage

# Import your compiled LangGraph workflow
from src.agent.graph import graph

# ==========================================
# CONFIGURATION & LANGFUSE SETUP
# ==========================================
st.set_page_config(
    page_title="Pool Chemistry Assistant",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Score naming (kept in constants so score_id stays stable across reruns) ---
SCORE_FEEDBACK = "user_feedback"
SCORE_REASON = "feedback_reason"

# Negative-feedback categories: label (EN) -> stored categorical value
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
    """
    try:
        client = Langfuse()
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
    N clicks -> exactly 1 score row, and flipping 👍 -> 👎 mutates the value.
    """
    return f"{trace_id}-{score_name}"


def submit_feedback(trace_id: str, value: int, comment: str, reason: str | None = None):
    """
    Sends (or overwrites) the feedback score for a given trace.
    value: 1 = thumbs up, 0 = thumbs down (data_type NUMERIC)
    """
    if lf is None:
        st.toast("Langfuse deshabilitado: el feedback no se registró.", icon="⚠️")
        return False

    try:
        lf.create_score(
            score_id=build_score_id(trace_id, SCORE_FEEDBACK),
            trace_id=trace_id,
            name=SCORE_FEEDBACK,
            value=value,
            data_type="NUMERIC",
            comment=comment,
        )

        if reason is not None:
            lf.create_score(
                score_id=build_score_id(trace_id, SCORE_REASON),
                trace_id=trace_id,
                name=SCORE_REASON,
                value=reason,
                data_type="CATEGORICAL",
                comment=comment,
            )

        # The v3 SDK batches in a background thread. A Streamlit rerun can end
        # the script before the batch ships, so flush explicitly.
        lf.flush()
        return True
    except Exception as e:
        st.error(f"No se pudo registrar el feedback: {e}")
        return False


# ==========================================
# NEO4J CONNECTION HANDLER
# ==========================================
@st.cache_resource
def get_neo4j_driver():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri:
        return None
    return GraphDatabase.driver(uri, auth=(user, password))


def check_and_handle_neo4j():
    """
    Checks Neo4j connectivity.
    Note: If using Neo4j Aura Free Tier, it pauses automatically.
    Standard Python drivers cannot "unpause" Aura. You must use the Aura API
    to programmatically resume it, or resume it via the Neo4j Console.
    """
    driver = get_neo4j_driver()
    if not driver:
        st.error("Neo4j environment variables are missing.")
        return False

    try:
        driver.verify_connectivity()
        return True
    except Exception as e:
        if "getaddrinfo failed" in str(e) or "Timeout" in str(e):
            st.error(
                "🚨 **Neo4j Database is currently unreachable or paused.**\n\n"
                "If you are using Neo4j Aura Free Tier, it may have been paused due to inactivity. "
                "Please log into the [Neo4j Aura Console](https://console.neo4j.io/) to resume your instance."
            )
        else:
            st.error(f"Neo4j Connection Error: {e}")
        return False


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    # trace_id -> {"value": int, "reason": str|None, "awaiting_detail": bool}
    st.session_state.feedback = {}
if "db_online" not in st.session_state:
    st.session_state.db_online = check_and_handle_neo4j()


# ==========================================
# UI: SIDEBAR (STATISTICS)
# ==========================================
with st.sidebar:
    st.title("📊 Agent Diagnostics")
    st.markdown("---")

    status_color = "🟢 Online" if st.session_state.db_online else "🔴 Offline"
    st.metric(label="Knowledge Graph Status", value=status_color)

    st.metric(label="Current Session ID", value=st.session_state.thread_id[:8].upper())
    st.metric(label="Interactions", value=len(st.session_state.messages) // 2)
    st.metric(label="Scored Responses", value=len(st.session_state.feedback))

    st.markdown("---")
    st.caption("Tracking powered by **Langfuse**")
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.feedback = {}
        st.rerun()


# ==========================================
# FEEDBACK WIDGET
# ==========================================
def render_feedback(trace_id: str):
    """
    Rendered from the history loop (NOT from inside the chat_input block), so it
    survives the rerun that a button click triggers.
    """
    state = st.session_state.feedback.get(trace_id)

    # --- Phase 3: thumbs-down detail form ---
    if state and state.get("awaiting_detail"):
        st.caption("👎 registrado. ¿Nos contás qué falló? (opcional)")
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
                # Same score_id as the initial 👎 -> upsert, not a duplicate.
                if submit_feedback(trace_id, 0, comment, reason=reason):
                    st.session_state.feedback[trace_id] = {
                        "value": 0,
                        "reason": reason,
                        "awaiting_detail": False,
                    }
                    st.toast("¡Gracias por el detalle!")
                    st.rerun()
        with c2:
            if st.button("Omitir", key=f"skip_{trace_id}"):
                # The value=0 score is already persisted; nothing to send.
                st.session_state.feedback[trace_id] = {
                    "value": 0,
                    "reason": None,
                    "awaiting_detail": False,
                }
                st.rerun()
        return

    # --- Already scored: show status, allow changing it ---
    if state:
        icon = "👍" if state["value"] == 1 else "👎"
        extra = f" · `{state['reason']}`" if state.get("reason") else ""
        c1, c2 = st.columns([3, 7])
        with c1:
            st.caption(f"{icon} Feedback enviado{extra}")
        with c2:
            if st.button("Cambiar", key=f"change_{trace_id}"):
                del st.session_state.feedback[trace_id]
                st.rerun()
        return

    # --- Phase 1: no feedback yet ---
    c1, c2, _ = st.columns([1, 1, 8])
    with c1:
        if st.button("👍 Me ayudó", key=f"good_{trace_id}"):
            if submit_feedback(trace_id, 1, "Respuesta útil"):
                st.session_state.feedback[trace_id] = {
                    "value": 1,
                    "reason": None,
                    "awaiting_detail": False,
                }
                st.toast("¡Gracias por tu feedback!")
                st.rerun()
    with c2:
        if st.button("👎 No me ayudó", key=f"bad_{trace_id}"):
            # Persist the negative signal immediately: if the user abandons the
            # detail form, we still keep the 0. Safe because score_id is fixed.
            if submit_feedback(trace_id, 0, "Respuesta no útil (pendiente de detalle)"):
                st.session_state.feedback[trace_id] = {
                    "value": 0,
                    "reason": None,
                    "awaiting_detail": True,
                }
                st.rerun()


# ==========================================
# UI: MAIN CHAT INTERFACE
# ==========================================
st.title("🌊 Pool Chemistry & Maintenance Assistant")
st.markdown("Describe your pool symptoms, maintenance needs, or equipment issues.")

if not st.session_state.db_online:
    st.stop()

# Display chat history (+ feedback widget per assistant turn)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("trace_id"):
            render_feedback(msg["trace_id"])

# User Input
if prompt := st.chat_input("E.g., My pool is cloudy and the pH is 8.2..."):
    # Deterministic, OTEL-compatible trace id (32 lowercase hex chars).
    # Seeded by session + turn index so a page reload resolves to the same trace.
    turn_index = len(st.session_state.messages)
    current_trace_id = Langfuse.create_trace_id(
        seed=f"{st.session_state.thread_id}:{turn_index}"
    )

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        lf_handler = CallbackHandler() if lf else None
        with st.status("🧠 Agent Thinking Process...", expanded=True) as status:
            config = {
                "configurable": {"thread_id": st.session_state.thread_id},
                "callbacks": [lf_handler] if lf_handler else [],
                # v3 propagates trace identity through metadata, not run_id.
                "metadata": {
                    "langfuse_trace_id": current_trace_id,
                    "langfuse_session_id": st.session_state.thread_id,
                },
            }

            final_response = ""

            for event in graph.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
                stream_mode="updates",
            ):
                if "planner" in event:
                    plan = event["planner"].get("execution_plan", [])
                    st.markdown("**📝 Planner Output:**")
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

                            if error_msg:
                                st.error(f"**🛠️ {agent_name.upper()} Agent Error:** {error_msg}")
                            else:
                                st.info(f"**✅ {agent_name.upper()} Agent executed successfully.**")

                elif "synthesizer" in event:
                    messages = event["synthesizer"].get("messages", [])
                    if messages:
                        final_response = messages[-1].content

            status.update(label="Response Generated", state="complete", expanded=False)

        if final_response:
            st.markdown(final_response)
            # trace_id travels with the message so any past turn stays scorable.
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_response,
                "trace_id": current_trace_id,
            })
            # Rerun so the feedback widget renders from the history loop, where
            # it survives the reruns that its own buttons trigger.
            st.rerun()