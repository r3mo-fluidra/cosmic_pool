from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Optional, TypedDict
from typing_extensions import NotRequired
import operator

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.types import Command
from ..graph_context.response_contracts import SynthesizerOutput
from ..graph_context.suggestions import Suggestion
from ..agent.agent_names import AgentName


def merge_agent_results(left: dict | None, right: dict | None) -> dict:
    """
    Merge para el fan-out paralelo de run_step, con reset explícito.

    right is None  → RESET (inicio de turno, lo emite el planner)
    right is dict  → merge (escrituras concurrentes de run_step)

    Sin el sentinel None no hay forma de vaciar el dict: {..} | {} == {..},
    y los agent_results del turno anterior sobreviven al siguiente planner.
    """
    if right is None:
        return {}
    return {**(left or {}), **right}

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
        "Actionable and technically precise task description written exclusively "
        "in English. The task must be specific enough for the assigned sub-agent "
        "to execute independently without inferring the planner's intent. "
        "It must identify the relevant pool-system context, the information "
        "or inputs to analyze, the expected operation or reasoning, and the "
        "desired outcome. "
        "When knowledge-graph retrieval is relevant, explicitly describe the "
        "entities, relationships, properties, or graph traversal that should "
        "be investigated. When calculations are required, explicitly identify "
        "the required variables, units, formula or quantitative objective. "
        "When procedural guidance is required, explicitly state the condition "
        "or problem and the expected procedure or recommendation. "
        "Do not include the final answer; describe only the task that the "
        "assigned agent must perform.\n"
        "Examples:\n"
        "- 'Analyze the reported green-water symptom and identify the most "
        "relevant pool chemistry parameters and possible causal relationships.'\n"
        "- 'Evaluate the circulation system for a reported low-flow condition "
        "and identify the hydraulic components and relationships that should "
        "be investigated.'\n"
        "- 'Determine the required chemical treatment quantity from the provided "
        "pool volume and measured water parameters, including all variables, "
        "units, and assumptions required for the calculation.'\n"
        "- 'Identify the applicable safety risks associated with storing and "
        "handling the reported pool chemical and provide the relevant preventive "
        "controls to investigate.'\n"
        "- 'Retrieve the relevant maintenance records and identify recurring "
        "equipment failures associated with the reported pump issue.'"
    )
)

    assigned_agent: AgentName = Field(
    description=(
        "The specific target sub-agent designated to execute this step. "
        "Select the agent whose domain expertise best matches the task. "
        "Rules:\n"
        "- 'general': General pool knowledge, basic guidance, and questions "
        "that do not require a specialized domain agent.\n"
        "- 'chemistry': Pool water chemistry, chemical parameters, water balance, "
        "treatment relationships, and chemical corrective actions.\n"
        "- 'equipment': Pool equipment, components, operation, performance, "
        "degradation, scaling, corrosion, and equipment-related failures.\n"
        "- 'hydraulics': Hydraulic circuits, circulation, flow, pressure, "
        "piping, valves, pumps, and hydraulic performance.\n"
        "- 'operations': Pool operation, routine procedures, maintenance "
        "activities, cleaning, inspections, and operational workflows.\n"
        "- 'compliance': Regulations, standards, requirements, documentation, "
        "and compliance-related guidance.\n"
        "- 'contamination': Water contamination, biological contamination, "
        "foreign substances, water-quality incidents, and contamination response.\n"
        "- 'facility_design': Pool facility design, layout, infrastructure, "
        "equipment-room configuration, and system-level design considerations.\n"
        "- 'safety': Pool, equipment, chemical, electrical, and operational "
        "safety risks and preventive measures.\n"
        "- 'recovery': Recovery procedures following failures, incidents, "
        "abnormal conditions, or system disruptions.\n"
        "- 'records': Logs, maintenance records, inspection records, "
        "historical data, documentation, and record-management tasks.\n"
        "- 'math': Numerical calculations, formulas, measurements, conversions, "
        "dosage calculations, hydraulic calculations, and quantitative reasoning.\n"
        "- 'oos': Mandatory choice for unsafe, prohibited, irrelevant, "
        "or out-of-scope requests.\n"
        "The planner may assign multiple steps to the same agent when "
        "multiple independent tasks require the same specialization."
    )
)

    oos: bool = Field(
        default=False,
        description=(
            "CRITICAL: Set to True if this step handles out-of-scope topics "
            "(e.g., human health advice, treating chemical contact, industrial "
            "waste processing, dangerous chemical synthesis, or general chit-chat). "
            "If True, assigned_agent MUST be set to 'oos'."
        ),
    ),
    depends_on: List[int] = Field(
        default_factory=list,
        description=(
            "Step numbers this step must wait for. Leave empty if this step "
            "is independent and can run in parallel with other steps. Only "
            "populate when this step genuinely needs the OUTPUT of another "
            "step (not just related topic area)."
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
        )
    ),
    missing_inputs: list[str] = Field(
        default_factory=list,
        description="Required parameters absent from the user message. "
                    "If non-empty, execution_plan MUST be empty."
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
    status: Literal["ok", "failed", "skipped"] = "ok"

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
    turn_started_at: Annotated[float, lambda old, new: new]
    current_step: NotRequired[int]          # 0-based index

    # ── Sub-agent results (dual-track pattern) ───────────────────────
    agent_results: Annotated[
    Optional[dict[str, AgentResult]],
    merge_agent_results,
    ]
    # Keys: "step_1", "step_2", ...
    # ── Response contract (NUEVO) ────────────────────────────────────
    archetype: NotRequired[str]
    response: NotRequired[SynthesizerOutput]   # payload para el frontend
    validation: NotRequired[dict]   
    # ── Sugerencias (chips) ──────────────────────────────────────────
    suggestions: NotRequired[List[Suggestion]]
    ignored_chip_streak: NotRequired[int]

    # ── Error handling ───────────────────────────────────────────────
    error: NotRequired[Optional[str]]



# futuro: test para asegurar que todos los agentes declarados en AgentName estén mapeados en AGENT_TO_ARCHETYPE
#     from state import AgentName. 
# from graph_context.response_contracts import AGENT_TO_ARCHETYPE

# def test_todos_los_agentes_mapeados():
#     declarados = set(AgentName.__args__)
#     mapeados   = set(AGENT_TO_ARCHETYPE)
#     assert declarados == mapeados, f"sin mapear: {declarados - mapeados}"