"""
response_contracts.py

Contrato de respuesta mobile-first para el nodo Synthesizer.

Dos responsabilidades:
  1. Definir el esquema de salida estructurada (`SynthesizerOutput`).
  2. Derivar determinísticamente el ARQUETIPO de respuesta a partir de los
     agentes que efectivamente produjeron contenido.

Un ARQUETIPO no es un agente: es la FORMA que la respuesta toma en pantalla.
Varios agentes colapsan en el mismo arquetipo si se renderizan igual en móvil.

Este módulo no importa nada del proyecto ni de LangGraph: es data + una
función pura, testeable sin correr el grafo.
"""

from __future__ import annotations

from typing import List, Literal, Optional, TYPE_CHECKING, Dict, Any
from pydantic import BaseModel, Field

# =====================================================================
# IMPORT DIFERIDO PARA EVITAR CIRCULAR
# =====================================================================
# AgentResult se importa solo cuando se necesita en runtime
# TYPE_CHECKING permite que los type hints funcionen sin import circular
if TYPE_CHECKING:
    from ..agent.state import AgentResult


# =====================================================================
# 1. ESQUEMA DE SALIDA
# =====================================================================

class ReadingLine(BaseModel):
    """
    Una lectura de análisis en el tier 1, renderizada como línea de lista.

    Existe porque la cobertura no cabía en la prosa. El contrato pide un
    `answer` de una a tres oraciones, y un panel de agua trae siete
    parámetros: forzarlos dentro producía un volcado con puntos y comas que
    en un móvil se lee como JSON traducido. Separando el campo, la prosa
    vuelve a ser prosa y las cifras se leen como lo que son, una tabla.
    """

    parameter: str = Field(
        description="Parameter name as the operator knows it: pH, Free Chlorine, Cyanuric Acid."
    )

    measured: str = Field(
        description="The measured value with its unit, as reported: '0.8 ppm', '7.9'."
    )

    note: str = Field(
        description=(
            "What this value means, in a handful of words and in the user's "
            "language: 'below the 2.0 ppm minimum', 'at its ceiling, no "
            "headroom'. Never intensify past the status the specialist "
            "assigned — a value sitting exactly on a published bound complies "
            "with it and is never described as a breach."
        )
    )


class DetailSection(BaseModel):
    """Collapsible section (Tier 2). The user opens it with a tap."""

    label: str = Field(
        description=(
            "Short section title, ≤ 5 words, in the user's language. "
            "This is the accordion text the user taps."
        )
    )

    body: str = Field(
        description=(
            "Expanded content: reasoning, calculations, assumptions, "
            "and alternatives. Markdown is allowed."
        )
    )


class SynthesizerOutput(BaseModel):
    """
    Structured output from the Synthesizer.

    Partition: answer + actions + safety = TIER 1, immediately visible,
    subject to the word budget. details = TIER 2, collapsible.
    """

    answer: str = Field(
        description=(
            "The direct answer in 1-3 sentences. What the user should do "
            "or know. Never include the reasoning or calculation."
        )
    )

    actions: List[str] = Field(
        default_factory=list,
        description=(
            "0-4 concrete actions, each ≤ 12 words, ordered by execution. "
            "Empty is valid."
        ),
    )

    safety: Optional[str] = Field(
        default=None,
        description=(
            "One line with a critical warning. Required when handling "
            "chemicals or when there is a direct safety risk. NEVER collapsible."
        ),
    )

    readings: List["ReadingLine"] = Field(
        default_factory=list,
        description=(
            "TIER 1. One entry per reported measurement that is NOT comfortably "
            "in range — the panel the operator scans before doing anything. "
            "Renders as a list, so it does not have to be forced into prose: "
            "`answer` stays one to three sentences and carries the verdict and "
            "the reasoning, while the per-parameter figures live here. "
            "Empty for any turn that did not interpret test results."
        ),
    )

    details: List[DetailSection] = Field(
        default_factory=list,
        description=(
            "Collapsible sections containing the justification and technical "
            "details. An empty list is valid and common."
        ),
    )

    # ---------------------------------------------------------------
    # Helpers de renderizado
    # ---------------------------------------------------------------

    def tier1_markdown(self) -> str:
        """
        Only what is visible. This is what goes into messages as an AIMessage:
        the history must not carry Tier 2 content that the user never read.
        """
        parts = [self.answer.strip()]
        # Las lecturas van entre el veredicto y las acciones: el operador lee
        # qué pasa, comprueba los números, y entonces actúa.
        if self.readings:
            parts.append("\n".join(
                f"- **{r.parameter}** {r.measured}" + (f" — {r.note}" if r.note else "")
                for r in self.readings
            ))
        if self.actions:
            parts.append("\n".join(f"- {a}" for a in self.actions))
        if self.safety:
            parts.append(f"!!! {self.safety.strip()}")
        return "\n\n".join(p for p in parts if p)

    def to_markdown(self) -> str:
        """
        Flat output, Tier 1 + Tier 2. Compatibility bridge: allows merging 
        the backend without modifying the frontend yet.
        """
        parts = [self.tier1_markdown()]
        for d in self.details:
            parts.append(f"**{d.label}**\n{d.body.strip()}")
        return "\n\n".join(p for p in parts if p)


# =====================================================================
# 2. MAPEO AGENTE -> ARQUETIPO
# =====================================================================

AGENT_TO_ARCHETYPE: dict[str, str] = {
    "math":            "calculation",
    "chemistry":       "assessment",
    "hydraulics":      "assessment",
    "facility_design": "assessment",
    "equipment":       "procedure",
    "operations":      "procedure",
    "recovery":        "procedure",
    "records":         "reference",
    "compliance":      "compliance",
    "contamination":   "critical",
    "safety":          "critical",
    "general":         "conversational",
    "oos":             "oos",
}

# Cuando un turno produce varios arquetipos, gana el de índice más bajo.
#   critical  -> una advertencia nunca queda plegada ni desplazada.
#   calculation -> si hay un número, el número va arriba.
PRECEDENCE: list[str] = [
    "critical",
    "calculation",
    "explanation",
    "assessment",
    "procedure",
    "reference",
    "compliance",
    "conversational",
]

DEFAULT_ARCHETYPE = "conversational"


# =====================================================================
# 3. CONTRATOS POR ARQUETIPO
# =====================================================================

NO_CAP = 9999  # centinela: sin techo de palabras

ARCHETYPE_CONTRACTS = {
    "critical": {
        "shape": "Immediate action first. Then the remediation sequence.",
        "budget": 9999,          # unlimited
        "details": [],           # nothing folded
        "safety_required": True,
    },

    "calculation": {
        "shape": "Result with its unit in one sentence. Then 2-4 actions of ≤12 words.",
        "budget": 800,
        "details": [
            "How it was calculated",
            "Formula and assumptions",
            "What happens if it is not corrected"
        ],
        "safety_required": "conditional",   # only if the calculation involves chemical handling
    },
    "assessment": {
        "shape": (
            "Lead with the verdict in one sentence: what is wrong and, when a "
            "closure or a stop is called for, WHICH single reading triggers it. "
            "That sentence is the opening, not the whole answer. Keep `answer` "
            "to the verdict and the reasoning — one to three sentences — and "
            "let the panel be a panel.\n"
            "What `answer` is for is what the panel cannot say: which single "
            "reading forces the closure, what mechanism connects them, and "
            "what produced the state. When the raw content carries a likely "
            "cause, name it — correcting the values without naming what "
            "produced them means the same state returns.\n"
            "Then the corrective actions, then the first verification."
        ),
        "budget": 900,
        "details": [
            "Full reading breakdown",
            "Other possible causes",
            "How to confirm",
        ],
        "safety_required": "conditional",
    },

    # Para preguntas que piden entender algo, no arreglarlo: mecanismos,
    # equilibrios, por qué un parámetro afecta a otro, qué significa una
    # lectura. La diferencia con `assessment` no es el agente que responde
    # sino lo que se preguntó — `chemistry` produce los dos.
    "explanation": {
        "shape": (
            "Answer exactly what was asked, in the form in which it was asked. "
            "A question about a quantity is answered with the quantity, stated "
            "before any advice. Then the mechanism that produces it."
        ),
        "budget": 900,
        "details": [
            "Why this happens",
            "What changes it",
            "Where this comes from",
        ],
        "safety_required": False,
        # Único arquetipo donde la lista vacía es el resultado NORMAL, no una
        # degradación. Una pregunta conceptual no genera tareas, y forzar el
        # andamio de acciones es lo que desplazó la respuesta: se produjo
        # "Test pool pH daily" en lugar de los dos porcentajes pedidos.
        "actions_optional": True,
    },

    "procedure": {
        "shape": "3-5 numbered steps in execution order.",
        "budget": 1000,
        "details": [
            "Required tools",
            "Common mistakes",
            "Recommended frequency"
        ],
        "safety_required": False,
    },
    "reference": {
        "shape": "List of fields or elements. No narrative between items.",
        "budget": 900,
        "details": [
            "Retention and format",
            "Requirement that originates it"
        ],
        "safety_required": False,
    },

    "compliance": {
        "shape": "Verdict (required / permitted / not permitted) and the standard that establishes it.",
        "budget": 700,
        "details": [
            "Standard text",
            "What the inspector verifies"
        ],
        "safety_required": False,
    },

    "conversational": {
        "shape": "One brief, warm paragraph.",
        "budget": 350,
        "details": [],
        "safety_required": False,
    },
    "oos": {
        "shape": "Brief redirection to the pool domain.",
        "budget": 300,
        "details": [],
        "safety_required": False,
    },

}

# `enforce_contract` lee contract["_name"] para el reporte de validación.
for _name, _contract in ARCHETYPE_CONTRACTS.items():
    _contract["_name"] = _name


# =====================================================================
# 4. RESOLUCIÓN
# =====================================================================

def resolve_archetype(
    agents: list[str],
    is_oos: bool = False,
    explanatory: bool = False,
) -> str:
    """
    Pure function: agents that produced content -> response archetype.

    Args:
        agents: agents that produced usable output (no error, non-empty output).
                This is NOT the planner's plan: it represents what was actually obtained.
        is_oos: True if any step in the plan was marked as oos.
        explanatory: True if the planner marked the request as a question about
                a mechanism or a quantity rather than about what to do. The map
                below is keyed by AGENT, and the same agent answers both kinds:
                `chemistry` interprets a test panel (assessment) and explains the
                HOCl/OCl- equilibrium (explanation). Without this flag the agent
                decides the shape, and a conceptual question inherits a contract
                whose shape is "verdict on what is out of range" — leaving the
                answer nowhere to go.

    Returns:
        Key of ARCHETYPE_CONTRACTS.
    """
    if is_oos:
        return "oos"

    if not agents:
        # Todos los agentes fallaron: degradación suave a un párrafo breve
        # en lugar de exigir un contrato imposible de cumplir.
        return DEFAULT_ARCHETYPE

    candidates = {
        AGENT_TO_ARCHETYPE[a] for a in agents if a in AGENT_TO_ARCHETYPE
    }
    if not candidates:
        return DEFAULT_ARCHETYPE

    if explanatory:
        # Se añade como candidato en lugar de imponerse: la precedencia sigue
        # decidiendo. Un turno explicativo que además destapó un peligro sale
        # como `critical`, y uno que además calculó un número, como
        # `calculation`. Explicar nunca debe enterrar una advertencia.
        candidates.add("explanation")

    return min(candidates, key=lambda c: PRECEDENCE.index(c)
               if c in PRECEDENCE else len(PRECEDENCE))


def get_contract(archetype: str) -> dict:
    """Acceso defensivo: un arquetipo desconocido no debe romper el nodo."""
    return ARCHETYPE_CONTRACTS.get(archetype,
                                   ARCHETYPE_CONTRACTS[DEFAULT_ARCHETYPE])


# build_synthesizer_archetype_section vivía acá y devolvía el NOMBRE del
# arquetipo — la cadena "assessment" — y nada más.
#
# nodes.py la importaba desde este módulo, así que el synthesizer recibía como
# `{archetype_section}` de su prompt una única palabra: sin la forma exigida,
# sin el presupuesto de palabras, sin las categorías de `details`, sin la
# instrucción de `safety`. Todo el sistema de contratos de este fichero estaba
# escrito y no llegaba al modelo.
#
# La implementación real, que renderiza el contrato entero, está en
# prompts/prompt_archetype.py y nadie la llamaba. Dos funciones con el mismo
# nombre en módulos distintos, y el import apuntando a la que no era.
#
# No se deja un alias: reexportarla desde acá es lo que permitió la confusión.
# Importar de prompt_archetype.py, que es de donde sale el texto.


# =====================================================================
# 5. HELPERS DE STATE (CON IMPORT DIFERIDO)
# =====================================================================

def usable_results(agent_results: Any) -> list:
    """
    Extract usable results from agent_results dict or list.
    
    Defensive: handles both dict and list inputs.
    Importa AgentResult solo cuando se llama a la función.
    """
    # Import diferido - evita el circular en tiempo de importación
    from ..agent.state import AgentResult
    
    results = []
    
    # Caso: es dict
    if isinstance(agent_results, dict):
        for value in agent_results.values():
            if isinstance(value, AgentResult) and value.output and not value.error:
                results.append(value)
            elif isinstance(value, dict):
                # Intentar convertir dict a AgentResult
                try:
                    result = AgentResult(**value)
                    if result.output and not result.error:
                        results.append(result)
                except Exception:
                    pass
    # Caso: es list
    elif isinstance(agent_results, list):
        for item in agent_results:
            if isinstance(item, AgentResult) and item.output and not item.error:
                results.append(item)
            elif isinstance(item, dict):
                try:
                    result = AgentResult(**item)
                    if result.output and not result.error:
                        results.append(result)
                except Exception:
                    pass
    
    return results


def agents_from_results(agent_results: Any) -> list:
    """
    Extract agent names from results.
    
    Defensive: handles both dict and list inputs.
    Importa AgentResult solo cuando se llama a la función.
    """
    # Import diferido - evita el circular en tiempo de importación
    from ..agent.state import AgentResult
    
    agents = []
    
    if isinstance(agent_results, dict):
        for value in agent_results.values():
            if isinstance(value, AgentResult):
                agents.append(value.agent)
            elif isinstance(value, dict):
                agents.append(value.get('agent', 'unknown'))
    elif isinstance(agent_results, list):
        for item in agent_results:
            if isinstance(item, AgentResult):
                agents.append(item.agent)
            elif isinstance(item, dict):
                agents.append(item.get('agent', 'unknown'))
    
    return agents



def fallback_payload(content: str, output_cls: type = SynthesizerOutput) -> SynthesizerOutput:
    """
    Create fallback payload from unstructured content.
    """
    return output_cls(
        archetype="conversational",
        answer=content,
        actions=[],
        safety=None,
        details=[],
    )


# =====================================================================
# 6. FUNCIONES ADICIONALES PARA EL SYNTHESIZER
# =====================================================================

def resolve_archetype_from_plan(
    execution_plan: list,
    agent_results: Any,
    extra_results: dict | None = None,
    error: str | None = None,
    force_archetype: str | None = None,
) -> str:
    """
    Resolve archetype from execution plan and results.
    
    Args:
        execution_plan: List of ExecutionStep
        agent_results: Dict or List of AgentResult
        extra_results: Additional results
        error: Error message
        force_archetype: Force a specific archetype
    
    Returns:
        Archetype string
    """
    if force_archetype:
        return force_archetype
    
    # Obtener agentes usables
    agents = agents_from_results(agent_results)
    
    # Verificar si hay OOS
    is_oos = False
    if execution_plan:
        for step in execution_plan:
            if hasattr(step, 'oos') and step.oos:
                is_oos = True
                break
    
    # Resolver archetype
    return resolve_archetype(agents, is_oos)

class VesselDeclaration(BaseModel):
    """
    The vessel the readings came from, as the specialist understood it.

    This is not cosmetic metadata: it selects the free chlorine floor and
    whether cyanuric acid is permitted at all. Leave an axis null when the
    user did not say — a guess here silently moves a closure threshold.
    """
    kind: Literal["pool", "spa"] | None = Field(
        default=None,
        description="pool or spa. Null if the user did not specify.",
    )
    indoor: bool | None = Field(
        default=None,
        description="true if indoor/covered/enclosed, false if outdoor. "
                    "Null if the user did not specify.",
    )