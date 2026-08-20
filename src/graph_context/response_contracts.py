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

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# =====================================================================
# 1. ESQUEMA DE SALIDA
# =====================================================================

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
        if self.actions:
            parts.append("\n".join(f"- {a}" for a in self.actions))
        if self.safety:
            parts.append(f"⚠️ {self.safety.strip()}")
        return "\n\n".join(p for p in parts if p)

    def to_markdown(self) -> str:
        """
        Flat output, Tier 1 + Tier 2. Compatibility bridge: allows merging the backend without modifying the frontend yet.
        """
        parts = [self.tier1_markdown()]
        for d in self.details:
            parts.append(f"**{d.label}**\n{d.body.strip()}")
        return "\n\n".join(parts)


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
        "shape": "One-sentence verdict: what is out of range or what fails in the design. Then the first verification.",
        "budget": 900,
        "details": [
            "Other possible causes",
            "How to confirm"
        ],
        "safety_required": False,
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

def resolve_archetype(agents: list[str], is_oos: bool = False) -> str:
    """
    Pure function: agents that produced content -> response archetype.

    Args:
        agents: agents that produced usable output (no error, non-empty output).
                This is NOT the planner's plan: it represents what was actually obtained.
        is_oos: True if any step in the plan was marked as oos.

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

    return min(candidates, key=lambda c: PRECEDENCE.index(c)
               if c in PRECEDENCE else len(PRECEDENCE))


def get_contract(archetype: str) -> dict:
    """Acceso defensivo: un arquetipo desconocido no debe romper el nodo."""
    return ARCHETYPE_CONTRACTS.get(archetype,
                                   ARCHETYPE_CONTRACTS[DEFAULT_ARCHETYPE])


# =====================================================================
# 5. HELPERS DE STATE
# =====================================================================

def usable_results(agent_results: dict) -> list:
    """
    agent_results es Dict["step_N", AgentResult]. El orden del dict no es
    confiable: se ordena por .step y se filtran los que fallaron.
    """
    results = sorted((agent_results or {}).values(), key=lambda r: r.step)
    return [r for r in results if not r.error and r.output and r.output.strip()]


def agents_from_results(agent_results: dict) -> list[str]:
    return [r.agent for r in usable_results(agent_results)]