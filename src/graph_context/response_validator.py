"""
response_validator.py

Enforcement determinística del contrato de respuesta emitido por el synthesizer.

El prompt le PIDE al LLM que respete el presupuesto; este módulo lo GARANTIZA.
Principio rector: nunca borrar información, solo reubicarla a `details`.
La única excepción es el recorte de acciones malformadas (>12 palabras),
que se reubican también en lugar de descartarse.

Uso:
    payload, report = enforce_contract(payload, contract, state["assigned_agents"])
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Union

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------

MAX_ACTIONS = 4
MAX_ACTION_WORDS = 12
NO_CAP = 9999  # presupuesto centinela: arquetipo `critical`, sin techo

# Agentes cuyo contenido implica manejo de producto químico o riesgo directo.
# Resuelven `safety_required = "conditional"` a True.
HAZARD_AGENTS = {"chemistry", "contamination", "safety"}

# Fallback léxico por si el arquetipo es `calculation` sin agente peligroso
# pero el contenido igual describe manipulación de producto.
HAZARD_PATTERN = re.compile(
    r"\b(ácido|acido|acid|cloro|chlorine|hipoclorito|hypochlorite|muriático|muriatic|"
    r"tricloro|dicloro|bromo|bromine|ozono|ozone|soda\s+cáustica|caustic|"
    r"peróxido|peroxide|alguicida|algaecide|EPP|PPE)\b",
    re.IGNORECASE,
)

# Etiquetas usadas al reubicar contenido excedente.
OVERFLOW_LABEL = "Next actions (overflow)"
SAFETY_LABEL_PATTERN = re.compile(
    r"safety|warning|hazard",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Reporte de validación (alimenta las métricas del paso 8)
# --------------------------------------------------------------------------

@dataclass
class ValidationReport:
    archetype: str = ""
    visible_words_before: int = 0
    visible_words_after: int = 0
    budget: int = 0
    overflowed: bool = False
    actions_relocated: int = 0
    safety_promoted: bool = False
    safety_missing: bool = False       # -> gatilla retry
    answer_exceeds_budget: bool = False  # -> gatilla retry
    notes: list[str] = field(default_factory=list)

    @property
    def needs_retry(self) -> bool:
        """Un solo retry. Si vuelve a fallar, se acepta la degradación."""
        return self.safety_missing or self.answer_exceeds_budget

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def _has_hazard_agent(agents: Optional[list[str]]) -> bool:
    """
    Tolerante al namespace. Acepta slug ("chemistry"), display name
    ("Pool Chemistry Agent") y cualquier casing/separador, porque el valor
    que llega en state["assigned_agents"] lo produce el planner y no está
    normalizado. Un fallo aquí es silencioso: intersection() con un display
    name da vacío y la advertencia simplemente no se exige.
    """
    for raw in agents or []:
        norm = re.sub(r"[^a-z0-9]+", "_", (raw or "").lower()).strip("_")
        if not norm:
            continue
        if norm in HAZARD_AGENTS:                      # slug exacto
            return True
        if set(norm.split("_")) & HAZARD_AGENTS:       # display name
            return True
        if any(h in norm for h in HAZARD_AGENTS if "_" in h):  # slug compuesto
            return True
    return False


def _safety_trigger(contract: dict, agents: Optional[list[str]], payload) -> str:
    """
    Devuelve QUÉ activó la exigencia de safety: "contract" | "agent" |
    "lexicon" | "". Se separa del booleano para que enforce_contract pueda
    registrar el motivo sin cambiar la firma pública.
    """
    required = contract.get("safety_required", False)
    if required is True:
        return "contract"
    if required != "conditional":
        return ""

    if _has_hazard_agent(agents):
        return "agent"

    surface = " ".join([
        getattr(payload, "answer", "") or "",
        *(getattr(payload, "actions", None) or []),
    ])
    return "lexicon" if HAZARD_PATTERN.search(surface) else ""


def _words(text: str | None) -> int:
    if not text:
        return 0
    return len(text.split())


def _visible_words(payload) -> int:
    """Cuenta el tier 1: lo que el usuario ve sin tocar nada."""
    return (
        _words(payload.answer)
        + sum(_words(a) for a in payload.actions)
        + _words(payload.safety)
    )


def _get_detail(payload, label: str):
    for d in payload.details:
        if d.label == label:
            return d
    return None


def _append_detail(payload, label: str, body: str, detail_cls) -> None:
    """Agrega o extiende una sección plegable, sin duplicar labels."""
    existing = _get_detail(payload, label)
    if existing:
        existing.body = f"{existing.body}\n{body}".strip()
    else:
        payload.details.append(detail_cls(label=label, body=body))


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return parts[0] if parts else text.strip()


# --------------------------------------------------------------------------
# Resolución de safety condicional
# --------------------------------------------------------------------------

def resolve_safety_required(contract: dict, agents: Optional[list[str]], payload) -> bool:
    """
    `safety_required` puede ser True, False o "conditional".

    "conditional" (arquetipo `calculation`): un cálculo de turnover no necesita
    advertencia; una dosis de ácido sí. Se decide por agente de origen y,
    como red, por léxico del contenido visible.
    """
    required = contract.get("safety_required", False)
    if required is not True and required != "conditional":
        return False
    if required is True:
        return True

    if _has_hazard_agent(agents):
        return True

    surface = " ".join([payload.answer or "", *(payload.actions or [])])
    return bool(_safety_trigger(contract, agents, payload))


# --------------------------------------------------------------------------
# Normalización de acciones
# --------------------------------------------------------------------------

def normalize_actions(payload, detail_cls, report: ValidationReport) -> None:
    """
    Reglas: máximo MAX_ACTIONS bullets, cada uno ≤ MAX_ACTION_WORDS palabras.
    Lo que no cumple NO se borra: se reubica a `details`.
    """
    kept: list[str] = []
    relocated: list[str] = []

    for action in payload.actions or []:
        action = action.strip()
        if not action:
            continue
        if _words(action) > MAX_ACTION_WORDS or len(kept) >= MAX_ACTIONS:
            relocated.append(action)
        else:
            kept.append(action)

    payload.actions = kept

    if relocated:
        body = "\n".join(f"- {a}" for a in relocated)
        _append_detail(payload, OVERFLOW_LABEL, body, detail_cls)
        report.actions_relocated += len(relocated)
        report.notes.append(f"{len(relocated)} acción(es) reubicadas a details")


# --------------------------------------------------------------------------
# Overflow de presupuesto
# --------------------------------------------------------------------------

def overflow_to_details(payload, budget: int, detail_cls,
                        report: ValidationReport) -> None:
    """
    Baja acciones desde el final hasta entrar en presupuesto.

    `answer` y `safety` son intocables: si por sí solos exceden el presupuesto,
    se marca `answer_exceeds_budget` y decide el caller (retry o aceptar).
    Truncar una frase a mitad es peor que pasarse de largo.
    """
    if budget >= NO_CAP:
        return

    floor = _words(payload.answer) + _words(payload.safety)
    if floor > budget:
        report.answer_exceeds_budget = True
        report.notes.append(
            f"answer+safety={floor}p supera el presupuesto de {budget}p por sí solos"
        )
        # No se puede corregir reubicando; se deja pasar y el caller decide.
        return

    relocated: list[str] = []
    while payload.actions and _visible_words(payload) > budget:
        relocated.insert(0, payload.actions.pop())

    if relocated:
        body = "\n".join(f"- {a}" for a in relocated)
        _append_detail(payload, OVERFLOW_LABEL, body, detail_cls)
        report.overflowed = True
        report.actions_relocated += len(relocated)
        report.notes.append(f"{len(relocated)} acción(es) reubicadas por presupuesto")


# --------------------------------------------------------------------------
# Promoción de seguridad
# --------------------------------------------------------------------------

def promote_safety_from_details(payload, report: ValidationReport) -> None:
    """
    Regla dura: nada de seguridad queda detrás del pliegue.

    Si el contrato exige `safety` y el LLM lo omitió, se busca una sección
    plegada con pinta de advertencia y se sube su primera frase a tier 1.
    Si no hay nada que promover, se marca `safety_missing` -> retry.
    """
    if payload.safety and payload.safety.strip():
        return

    for detail in list(payload.details):
        if SAFETY_LABEL_PATTERN.search(detail.label) or \
           SAFETY_LABEL_PATTERN.search(detail.body):
            payload.safety = _first_sentence(detail.body)
            payload.details.remove(detail)
            report.safety_promoted = True
            report.notes.append(f"safety promovida desde details: '{detail.label}'")
            return

    report.safety_missing = True
    report.notes.append("contrato exige safety y no hay contenido para promover")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def enforce_contract(payload, contract: dict, agents: list[str] | None = None,
                     detail_cls=None) -> tuple[Any, ValidationReport]:
    """
    Aplica el contrato al payload del synthesizer.

    Args:
        payload:    instancia de SynthesizerOutput (mutada in place).
        contract:   entrada de ARCHETYPE_CONTRACTS.
        agents:     state["assigned_agents"], para resolver safety condicional.
        detail_cls: clase del item de details. Si es None se infiere del payload
                    o se cae a un dict-like compatible.

    Returns:
        (payload, report). Si `report.needs_retry` es True, el caller puede
        reintentar UNA vez con instrucción correctiva; si no, acepta la
        degradación determinística que ya se aplicó.
    """
    report = ValidationReport(
        archetype=contract.get("_name", ""),
        budget=contract.get("budget", NO_CAP),
    )

    if detail_cls is None:
        detail_cls = _infer_detail_cls(payload)

    payload.actions = payload.actions or []
    payload.details = payload.details or []

    report.visible_words_before = _visible_words(payload)

    # 1. Normalizar bullets antes de medir presupuesto.
    normalize_actions(payload, detail_cls, report)

    # 2. Seguridad: promover ANTES del overflow, porque suma al conteo visible.
    trigger = _safety_trigger(contract, agents or [], payload)
    if trigger:
        report.notes.append(f"safety exigida por: {trigger}")
        promote_safety_from_details(payload, report)

    # 3. Presupuesto.
    overflow_to_details(payload, report.budget, detail_cls, report)

    # 4. Podar secciones vacías.
    payload.details = [d for d in payload.details if d.body and d.body.strip()]

    report.visible_words_after = _visible_words(payload)
    return payload, report


def _infer_detail_cls(payload):
    """Recupera la clase del item de details desde el modelo Pydantic."""
    if payload.details:
        return type(payload.details[0])
    try:  # pydantic v2
        return payload.model_fields["details"].annotation.__args__[0]
    except Exception:  # fallback laxo
        from types import SimpleNamespace
        return lambda label, body: SimpleNamespace(label=label, body=body)


# --------------------------------------------------------------------------
# Fallback (paso 7)
# --------------------------------------------------------------------------

def fallback_payload(raw_text: str, output_cls):
    """
    Si el structured output falla, el front nunca ve un formato distinto.
    Todo el texto crudo va a `answer`, sin plegado y sin validación.
    """
    return output_cls(answer=raw_text, actions=[], safety=None, details=[])