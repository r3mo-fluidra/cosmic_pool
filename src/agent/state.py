from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Optional, TypedDict
from typing_extensions import NotRequired
import operator

from pydantic import BaseModel, Field, model_validator
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.types import Command
from ..graph_context.response_contracts import SynthesizerOutput
from ..graph_context.suggestions import Suggestion
from ..agent.agent_names import AgentName  # debe incluir "general" y "oos"


def merge_agent_results(left: dict | None, right: dict | None) -> dict:
    """
    Merge para el fan-out paralelo de run_step, con reset explícito.

    right is None  → RESET (inicio de turno, lo emite el planner)
    right is dict  → merge (escrituras concurrentes de run_step)
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
        description="Sequential execution order, starting strictly at 1.",
        ge=1,
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
            "equipment failures associated with the reported pump issue.'\n"
            "- 'Request the missing parameters required for acid dosage to lower pH: "
            "pool volume, current pH, target pH, and type/strength of acid.'"
        ),
    )

    assigned_agent: AgentName = Field(
        description=(
            "The specific target sub-agent designated to execute this step. "
            "Select the agent whose domain expertise best matches the task.\n"
            "Rules:\n"
            "- 'general': Greetings, capability questions, educational topics with no "
            "reference to the user's facility, AND clarification requests when required "
            "numeric inputs for dosing/sizing are missing.\n"
            "- 'chemistry': Water chemistry of a specific pool, symptoms, test results, "
            "corrective chemical actions (not the numeric dosage itself).\n"
            "- 'equipment': Faulty/worn/fouled components, service procedures, parts.\n"
            "- 'hydraulics': Flow rate, turnover, head loss, pump operating point.\n"
            "- 'operations': Schedules, preventive maintenance programs, routines.\n"
            "- 'compliance': Whether something is required/permitted/inspectable under "
            "US or Canadian codes only.\n"
            "- 'contamination': Fecal/vomit/blood incidents, RWI outbreaks.\n"
            "- 'facility_design': New builds or renovations (system does not exist yet).\n"
            "- 'safety': Prevention, supervision, PPE, emergency preparedness.\n"
            "- 'recovery': Flood, storm, sewage backup, prolonged abandonment.\n"
            "- 'records': How to structure logs, retention, inspection packages.\n"
            "- 'math': Pure numeric computation once inputs and formula are known.\n"
            "- 'oos': Unsafe, medical advice for a person, illegal activity, or any "
            "jurisdiction outside the US and Canada.\n"
            "For dosing questions with missing volume/current/target/chemical type → "
            "ALWAYS use 'general' (never chemistry, math, compliance, or operations)."
        ),
    )

    oos: bool = Field(
        default=False,
        description=(
            "Set to True only when this step handles a genuine out-of-scope topic "
            "(personal medical advice, illegal activity, jurisdiction outside US/Canada, "
            "or topics unrelated to pools). If True, assigned_agent MUST be 'oos'."
        ),
    )

    depends_on: List[int] = Field(
        default_factory=list,
        description=(
            "Step numbers this step must wait for. Leave empty if independent. "
            "Only populate when this step genuinely needs the OUTPUT of another step."
        ),
    )

    @model_validator(mode="after")
    def oos_requires_oos_agent(self) -> "ExecutionStep":
        if self.oos and self.assigned_agent != "oos":
            raise ValueError(
                "When oos=True, assigned_agent must be 'oos'."
            )
        if self.assigned_agent == "oos" and not self.oos:
            # Normalizar: si eligen oos, forzar el flag
            self.oos = True
        return self


class PlannerOutput(BaseModel):
    """
    Structured output from the Planner node.
    """

    detected_language: Literal["es", "en"] = Field(
        description="Language detected in the user's message from the raw input text.",
    )

    # Permitir plan vacío cuando hay missing_inputs
    execution_plan: List[ExecutionStep] = Field(
        default_factory=list,
        max_length=5,
        description=(
            "Ordered list of steps to fulfill the user's request. "
            "If missing_inputs is non-empty, this MUST be empty (or contain only a "
            "single 'general' clarification step — prefer empty + missing_inputs). "
            "If the query is fully out of scope, return a single step with "
            "assigned_agent='oos' and oos=True."
        ),
    )

    missing_inputs: List[str] = Field(
        default_factory=list,
        description=(
            "Required parameters absent from the user message for a dosing/sizing "
            "question (e.g. 'pool volume', 'current pH', 'target pH', 'acid type'). "
            "If non-empty, do NOT create chemistry/math steps; either leave "
            "execution_plan empty or put a single general clarification step."
        ),
    )

    @model_validator(mode="after")
    def consistency_rules(self) -> "PlannerOutput":
        # Si hay missing_inputs y el plan tiene steps de cálculo, es inconsistente.
        # No bloqueamos hard para no romper el LLM, pero puedes endurecer si quieres.
        if self.missing_inputs and not self.execution_plan:
            return self
        if not self.execution_plan and not self.missing_inputs:
            # Debe haber al menos algo (plan o missing)
            # Opcional: raise ValueError("execution_plan and missing_inputs cannot both be empty")
            pass
        return self


# =====================================================================
# 2. SUB-AGENT RESULTS
# =====================================================================

class AgentResult(BaseModel):
    """Result written by each sub-agent into the shared state."""

    agent: AgentName  # debe incluir general y oos
    step: int
    output: str
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

    # ── Planner output ───────────────────────────────────────────────
    detected_language: NotRequired[str]
    execution_plan: NotRequired[List[ExecutionStep]]
    missing_inputs: NotRequired[List[str]]  # ← añádelo al state también

    # ── Orchestrator control ─────────────────────────────────────────
    turn_started_at: Annotated[float, lambda old, new: new]
    current_step: NotRequired[int]

    # ── Sub-agent results ────────────────────────────────────────────
    agent_results: Annotated[
        Optional[dict[str, AgentResult]],
        merge_agent_results,
    ]

    # ── Response contract ────────────────────────────────────────────
    archetype: NotRequired[str]
    response: NotRequired[SynthesizerOutput]
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