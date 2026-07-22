from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Optional, TypedDict
from typing_extensions import NotRequired

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.types import Command


# =====================================================================
# 0. AGENT NAMES (Centralized to avoid typos)
# =====================================================================

AgentName = Literal[
    "diagnosis",      # Symptom → causal parameter
    "dosage",         # P_* → C_* with RAISES/LOWERS edges
    "equipment",      # P_* → E_* with CORRODES/SCALES/DEGRADES
    "maintenance",    # Routines and seasonal maintenance
    "ooo",            # Out of Scope handler (triggers OOS response)
]


# =====================================================================
# 1. PLANNER OUTPUT MODELS
# =====================================================================

class ExecutionStep(BaseModel):
    """A single deterministic task step within the multi-agent execution pipeline."""

    step: int = Field(
        description="Sequential execution order, starting strictly at 1."
    )

    task: str = Field(
        description=(
            "Actionable, technical task description written exclusively in English. "
            "Must explicitly state the inputs required and expected edge operations. "
            "Example: 'Map green water symptom to causal chemical parameters' or "
            "'Calculate calcium chloride dosage using RAISES edge to correct low CH'."
        )
    )

    assigned_agent: AgentName = Field(
        description=(
            "The specific target sub-agent designated to run this step. "
            "Rules:\n"
            "- 'diagnosis': Mapping symptoms to P_* parameters.\n"
            "- 'dosage': Running metric balance calculations via RAISES/LOWERS.\n"
            "- 'equipment': Evaluating CORRODES/SCALES/DEGRADES structural/hardware effects.\n"
            "- 'maintenance': Routine, calendar, or seasonal tasks.\n"
            "- 'ooo': Mandatory choice if the query contains unsafe or irrelevant content."
        )
    )

    oos: bool = Field(
        default=False,
        description=(
            "CRITICAL: Set to True if this step handles out-of-scope topics "
            "(e.g., human health advice, treating chemical contact, industrial "
            "waste processing, dangerous chemical synthesis, or general chit-chat). "
            "If True, assigned_agent MUST be set to 'ooo'."
        ),
    )


class PlannerOutput(BaseModel):
    """
    Structured output from the Planner node.
    The LLM returns this object using structured output.
    """

    detected_language: Literal["es", "en"] = Field(
        description="Language detected in the user's message",
    )

    execution_plan: List[ExecutionStep] = Field(
        min_length=1,
        max_length=5,
        description=(
            "Ordered list of steps to fulfill the user's request. "
            "If the query is out of scope, return a single step with oos=True."
        ),
    )


# =====================================================================
# 2. SUB-AGENT RESULTS
# =====================================================================

class AgentResult(BaseModel):
    """Result written by each sub-agent into the shared state."""

    agent: AgentName
    step: int
    output: str                        # Processed text ready for Synthesizer
    sources: List[str] = Field(default_factory=list)
    error: Optional[str] = None


# =====================================================================
# 3. GLOBAL GRAPH STATE
# =====================================================================

class PoolAgentState(TypedDict):
    # ── Public conversation ──────────────────────────────────────────
    messages: Annotated[List[BaseMessage], add_messages]
    conversation_summary: str
    # Only user messages and final PoolAgent responses go here.

    # ── Planner output ───────────────────────────────────────────────
    detected_language: NotRequired[str]
    execution_plan: NotRequired[List[ExecutionStep]]

    # ── Orchestrator control ─────────────────────────────────────────
    current_step: NotRequired[int]          # 0-based index

    # ── Sub-agent results (dual-track pattern) ───────────────────────
    agent_results: NotRequired[Dict[str, AgentResult]]
    # Keys: "step_1", "step_2", ...

    # ── Error handling ───────────────────────────────────────────────
    error: NotRequired[Optional[str]]