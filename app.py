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

# Initialize Langfuse client for tracking feedback
try:
    lf = Langfuse()
except Exception as e:
    st.sidebar.warning("Langfuse credentials not found. Tracking disabled.")
    lf = None

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
            # Placeholder for programmatic unpause if you have Aura API keys
            # requests.post("https://api.neo4j.io/v1/instances/{INSTANCE_ID}/resume", headers={"Authorization": f"Bearer {AURA_API_TOKEN}"})
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
if "trace_id" not in st.session_state:
    st.session_state.trace_id = None
if "db_online" not in st.session_state:
    st.session_state.db_online = check_and_handle_neo4j()

# ==========================================
# UI: SIDEBAR (STATISTICS)
# ==========================================
with st.sidebar:
    st.title("📊 Agent Diagnostics")
    st.markdown("---")
    
    # DB Status
    status_color = "🟢 Online" if st.session_state.db_online else "🔴 Offline"
    st.metric(label="Knowledge Graph Status", value=status_color)
    
    # Session Info
    st.metric(label="Current Session ID", value=st.session_state.thread_id[:8].upper())
    st.metric(label="Interactions", value=len(st.session_state.messages) // 2)
    
    st.markdown("---")
    st.caption("Tracking powered by **Langfuse**")
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.trace_id = None
        st.rerun()

# ==========================================
# UI: MAIN CHAT INTERFACE
# ==========================================
st.title("🌊 Pool Chemistry & Maintenance Assistant")
st.markdown("Describe your pool symptoms, maintenance needs, or equipment issues.")

# Stop execution if DB is offline
if not st.session_state.db_online:
    st.stop()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if prompt := st.chat_input("E.g., My pool is cloudy and the pH is 8.2..."):
    # 1. Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate a Trace ID for Langfuse
    current_trace_id = str(uuid.uuid4())
    st.session_state.trace_id = current_trace_id

    # 2. Agent Execution & Scratchpad
    with st.chat_message("assistant"):
        # UI Container for the Thinking Process
        # Initialize the v3 handler (Empty - no arguments)
        lf_handler = CallbackHandler() if lf else None
        with st.status("🧠 Agent Thinking Process...", expanded=True) as status:
            config = {
                "configurable": {"thread_id": st.session_state.thread_id},
                "run_id": current_trace_id, # Langfuse uses this as the Trace ID
                "callbacks": [lf_handler] if lf_handler else []
            }
            
            final_response = ""
            
            # Stream the LangGraph execution
            for event in graph.stream({"messages": [HumanMessage(content=prompt)]}, config=config, stream_mode="updates"):
                
                # Intercept Planner Output
                if "planner" in event:
                    plan = event["planner"].get("execution_plan", [])
                    st.markdown("**📝 Planner Output:**")
                    for step in plan:
                        st.markdown(f"- Step {step.step} `[{step.assigned_agent}]`: {step.task}")
                
                # Intercept Orchestrator/Sub-Agent Output
                elif "orchestrator" in event:
                    # Get the orchestrator state safely
                    orch_state = event.get("orchestrator")
                    
                    # Ensure orch_state is a dict and has 'agent_results'
                    if isinstance(orch_state, dict):
                        results = orch_state.get("agent_results", {})
                        
                        # Find the most recently added result
                        if results:
                            latest_key = sorted(results.keys())[-1]
                            latest_res = results[latest_key]
                            
                            agent_name = getattr(latest_res, "agent", "unknown")
                            error_msg = getattr(latest_res, "error", None)
                            
                            if error_msg:
                                st.error(f"**🛠️ {agent_name.upper()} Agent Error:** {error_msg}")
                            else:
                                st.info(f"**✅ {agent_name.upper()} Agent executed successfully.**")
                # Intercept Synthesizer Output
                elif "synthesizer" in event:
                    messages = event["synthesizer"].get("messages", [])
                    if messages:
                        final_response = messages[-1].content
                        
            status.update(label="Response Generated", state="complete", expanded=False)

        # 3. Display Final Response
        if final_response:
            st.markdown(final_response)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            
            # 4. Render Feedback Buttons
            col1, col2, _ = st.columns([1, 1, 8])
            with col1:
                if st.button("👍 Good", key=f"good_{current_trace_id}"):
                    if lf:
                        lf.score(trace_id=current_trace_id, name="user_feedback", value=1.0)
                        st.toast("Feedback recorded! Thank you.")
            with col2:
                if st.button("👎 Bad", key=f"bad_{current_trace_id}"):
                    if lf:
                        lf.score(trace_id=current_trace_id, name="user_feedback", value=0.0)
                        st.toast("Feedback recorded! Thank you.")