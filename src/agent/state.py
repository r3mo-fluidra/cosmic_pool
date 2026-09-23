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
            "assigned agent must perform."
            "Examples:"
            "- 'Analyze the reported green-water symptom and identify the most "
            "relevant pool chemistry parameters and possible causal relationships.'"
            "- 'Evaluate the circulation system for a reported low-flow condition "
            "and identify the hydraulic components and relationships that should "
            "be investigated.'"
            "- 'Determine the required chemical treatment quantity from the provided "
            "pool volume and measured water parameters, including all variables, "
            "units, and assumptions required for the calculation.'"
            "- 'Identify the applicable safety risks associated with storing and "
            "handling the reported pool chemical and provide the relevant preventive "
            "controls to investigate.'"
            "- 'Retrieve the relevant maintenance records and identify recurring "
            "equipment failures associated with the reported pump issue.'"
            "- 'Request the parameters still missing for an acid dose: pool volume "
            "and current pH.' (Only what the user has NOT given, and never the "
            "target or the product type — the specialist assumes standard values "
            "for those.)"
        ),
    ),
    retrieval_query: str = Field(
        default="",
        description=(
            "Keyword query for the semantic search the assigned agent will run "
            "first. Six to twelve content words in English, space separated: "
            "the equipment, symptom, process, chemical or procedure named in "
            "the task. No sentence, no question, no articles, prepositions or "
            "verbs like 'analyze', 'determine' or 'provide' — those describe "
            "what the agent does, not what is in the manual. "
            "Example task: 'Analyze the pressure differential and the "
            "ineffective backwash to diagnose the filter issue.' "
            "Example query: 'sand filter pressure differential backwash "
            "channeling media calcification replacement'. "
            "Leave empty only for steps assigned to 'general', 'oos' or 'math', "
            "which do not search the manual."
        ),
    )
    retrieval_intent: Literal["normative", "procedural", "diagnostic", "descriptive", "any"] = Field(
        default="any",
        description=(
            "Which kind of graph node can answer this step, so the right node "
            "labels rank first in the knowledge-graph search. Pick exactly one:\n"
            "- 'normative': the step asks for a threshold, range, limit, "
            "required value, or code provision.\n"
            "- 'diagnostic': the step asks why something is failing, what is "
            "wrong, or what a symptom or reading means.\n"
            "- 'procedural': the step asks how to perform, service, clean, "
            "install or correct something.\n"
            "- 'descriptive': the step asks what something is or how it works, "
            "with nothing failing and nothing to do.\n"
            "- 'any': only when none of the four fits, or for steps assigned to "
            "'general', 'oos' or 'math'.\n"
            "When a step carries two of these, pick the one its FIRST clause "
            "asks for: a step that diagnoses a fault and then gives the repair "
            "procedure is 'diagnostic'."
        ),
    )


    assigned_agent: AgentName = Field(
        description=(
            "The specific target sub-agent designated to execute this step. "
            "Select the agent whose domain expertise best matches the task."
            "Rules:"
            "- 'general': Greetings, capability questions, educational topics with no "
            "reference to the user's facility, AND clarification requests asking only "
            "for the inputs still missing after the specialists have covered what is "
            "already answerable."
            "- 'chemistry': Water chemistry of a specific pool, symptoms, test results, "
            "corrective chemical actions (not the numeric dosage itself)."
            "- 'equipment': Faulty/worn/fouled components, service procedures, parts."
            "- 'hydraulics': Flow rate, turnover, head loss, pump operating point."
            "- 'operations': Schedules, preventive maintenance programs, routines."
            "- 'compliance': Whether something is required/permitted/inspectable under "
            "US or Canadian codes only."
            "- 'contamination': Fecal/vomit/blood incidents, RWI outbreaks."
            "- 'facility_design': New builds or renovations (system does not exist yet)."
            "- 'safety': Prevention, supervision, PPE, emergency preparedness."
            "- 'recovery': Flood, storm, sewage backup, prolonged abandonment."
            "- 'records': How to structure logs, retention, inspection packages."
            "- 'math': Pure numeric computation once inputs and formula are known."
            "- 'oos': Unsafe, medical advice for a person, illegal activity, or any "
            "jurisdiction outside the US and Canada."
            "A missing input blocks the NUMBER, never the DIAGNOSIS. If the user "
            "reports a symptom, a reading or an observation, the owning specialist "
            "ALWAYS gets a step — the mechanism and the order of correction do not "
            "depend on the missing value. Add a 'general' step afterwards asking only "
            "for what is still missing. Route to 'general' alone only when nothing at "
            "all is answerable yet."
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
            "Step numbers whose OUTPUT this step consumes. Default is empty, and "
            "empty is the common case."
            "Steps with no dependency between them run AT THE SAME TIME; a step "
            "with depends_on waits for the other to finish first. So a "
            "dependency declared out of caution, or just because one step reads "
            "as coming 'after' another, adds its full duration to the turn for "
            "nothing. Measured: two steps of roughly 16s each took 32.6s "
            "serialized where they would have taken ~17s in parallel."
            "The test is data, not narrative order: would this step's task be "
            "impossible to write without knowing the other step's ANSWER? "
            "Needing the same background, covering a related topic, or reading "
            "as the natural next thing to say are NOT dependencies."
            "Genuine: a dose calculation that needs the target value another "
            "step establishes. Not a dependency: explaining a mechanism and "
            "quantifying it — both come from the same retrieved material."
        ),
    )

    explanatory: bool = Field(
        default=False,
        description=(
            "True when the user asked to UNDERSTAND something rather than to fix "
            "it: a mechanism, an equilibrium, why one parameter affects another, "
            "what a reading means, or a specific quantity or fraction. The tell is "
            "that a correct answer is information, not a task — nothing needs to "
            "be done to the pool once it is read."
            "Decide on the state of the pool, never on the grammar of the "
            "question. If the user reports that something at their facility is "
            "not working as it should — cloudy water, lost pressure, a reading "
            "out of range, a noise, a system underperforming — this is NOT "
            "explanatory, however the question is phrased. 'Could this be "
            "circulation?', 'is our turnover adequate?' and 'what's failing?' "
            "read as requests for information and are not: each one names a "
            "facility with a fault, and the operator will have to act on the "
            "answer. A yes/no or diagnostic framing does not turn a broken pool "
            "into a knowledge question."
            "Examples that ARE explanatory: 'why does chlorine lose effectiveness "
            "as pH rises', 'what share of free chlorine is hypochlorous acid at "
            "pH 7.2 versus 7.8', 'what does cyanuric acid actually do', 'my ORP "
            "and DPD disagree, which one do I trust'. None of these reports a "
            "fault: the first three ask about chemistry in general, and the "
            "fourth asks which instrument to believe, which is answered by "
            "reading it."
            "Examples that are NOT: 'what is out of range in these readings', "
            "'how do I fix cloudy water', 'how much acid do I need', 'the deep "
            "end stays cloudy while the shallow end clears — could this be "
            "circulation', 'our flow meter reads 610 GPM, is that adequate', "
            "'the pump is whining and suction vacuum is up, what's failing'."
            "A question can report readings and still be explanatory if what it "
            "asks for is the reason behind them and nothing is malfunctioning. "
            "Setting this wrong is costly in both directions: an explanatory "
            "question forced into the operational format loses the answer to "
            "make room for advice nobody asked for, and a fault marked "
            "explanatory withholds the corrective actions the specialist "
            "produced, because the response contract suppresses them."
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
            "A non-empty missing_inputs does NOT empty this list: it only blocks the "
            "'math' step. Whatever the user already reported — a symptom, a reading, "
            "an observation — still gets its specialist step, because explaining the "
            "mechanism and the order of correction needs no further input. Leave this "
            "empty only when nothing at all is answerable yet. "
            "If the query is fully out of scope, return a single step with "
            "assigned_agent='oos' and oos=True."
        ),
    )

    missing_inputs: List[str] = Field(
        default_factory=list,
        description=(
            "Parameters the user has NOT provided that are required to compute a "
            "number: pool volume and the current reading of the target parameter. "
            "Do NOT list the target reading or the product type — those have standard "
            "values the specialist states as assumptions, so they are never blockers. "
            "A non-empty list blocks only the 'math' step; the specialist step for the "
            "reported symptom is still planned, followed by a 'general' step asking "
            "for exactly these parameters and nothing the user already gave."
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
    vessel: Annotated[dict, lambda a, b: b]
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
    archetype: Annotated[str | None, lambda a, b: b]
    response: NotRequired[SynthesizerOutput]
    validation: NotRequired[dict]

    # ── Sugerencias (chips) ──────────────────────────────────────────
    suggestions: NotRequired[List[Suggestion]]
    misroute_retries: Annotated[int, lambda a, b: b] 
    ignored_chip_streak: Annotated[int, lambda a, b: b]

    # ── Error handling ───────────────────────────────────────────────
    # Reducer explícito last-wins: el reset de turno lo hace
    # build_context_node escribiendo None. Sin reducer declarado, un None
    # entrante se trata igual, pero dejarlo explícito documenta que este
    # canal se limpia por turno y no acumula.
    error: Annotated[Optional[str], lambda a, b: b]

    #: Lo escribe el fallback del planner cuando la cadena LLM falla.
    #: Estaba sin declarar: LangGraph lo descartaba en silencio.
    planner_error: Annotated[Optional[str], lambda a, b: b]


# futuro: test para asegurar que todos los agentes declarados en AgentName estén mapeados en AGENT_TO_ARCHETYPE
#     from state import AgentName. 
# from graph_context.response_contracts import AGENT_TO_ARCHETYPE

# def test_todos_los_agentes_mapeados():
#     declarados = set(AgentName.__args__)
#     mapeados   = set(AGENT_TO_ARCHETYPE)
#     assert declarados == mapeados, f"sin mapear: {declarados - mapeados}"