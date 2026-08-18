"""
graph_context/suggestions.py
============================
Modelo de salida para los chips de sugerencia + gates de calidad
determinísticos.

Principio: el prompt PIDE que se cumplan las reglas; los gates las
IMPONEN. Nunca confiar en que el LLM respete el contrato.

Principio 2: descartar es barato, mostrar basura es caro. Todos los
gates son conservadores — ante la duda, se descarta el chip. Devolver
0 chips es el resultado correcto y esperado la mayor parte del tiempo.

Sin dependencias del proyecto salvo AgentName (que vive en su propio
módulo justamente para que este import no sea circular).
"""

from __future__ import annotations

import re
from typing import List, Sequence, Set, Tuple

from pydantic import BaseModel, Field
from typing_extensions import get_args

from ..agent.agent_names import AgentName   # AJUSTAR la ruta al ubicar el módulo


# =====================================================================
# 1. CONSTANTES
# =====================================================================

_VALID_AGENTS: Set[str] = set(get_args(AgentName))

# Agentes que nunca deben ser destino de un chip, aunque sean válidos
# como valor de AgentName. Un chip que rutea a "oos" es una invitación
# a salirse del dominio; uno que rutea a "general" no aporta nada
# específico que justifique ocupar espacio en pantalla.
_NON_SUGGESTABLE_AGENTS: Set[str] = {"oos", "general"}

# Supernodos del grafo (grado alto) — validado en retrieval que el
# bloqueo anti-hub importa; acá aplica igual. Un chip sobre "pH" es
# tan genérico que no predice nada.
SUPERNODES: Set[str] = {
    "free_chlorine",
    "cyanuric_acid",
    "ph",
}

_MAX_LABEL_WORDS = 5
_MAX_LABEL_CHARS = 28      # 2 chips por fila a 380px
_MAX_SUGGESTIONS = 3

# Umbral de solapamiento de tokens para considerar dos labels
# reformulaciones el uno del otro (gate 6).
_SIMILARITY_THRESHOLD = 0.6

# Stopwords mínimas, es/en. No se usa NLTK ni nada externo: el objetivo
# es solo evitar que "de/la/the/of" inflen la similitud entre labels.
_STOPWORDS: Set[str] = {
    "de", "la", "el", "los", "las", "un", "una", "y", "o", "en", "que",
    "del", "al", "por", "para", "con", "como", "es", "se", "su", "lo",
    "the", "a", "an", "and", "or", "in", "of", "to", "for", "is", "it",
    "how", "what", "when", "why", "which", "cual", "cuales", "como",
    "que", "cuando", "donde", "por", "puedo", "debo", "can", "should",
}


# =====================================================================
# 2. MODELO DE SALIDA
# =====================================================================

class Suggestion(BaseModel):
    """Un chip: botón corto y tocable debajo de la respuesta."""

    label: str = Field(
        description=(
            f"Texto del chip, en el idioma del usuario. "
            f"Máximo {_MAX_LABEL_WORDS} palabras y {_MAX_LABEL_CHARS} "
            f"caracteres. Debe leerse como una pregunta o acción corta, "
            f"no como una oración completa."
        )
    )

    agent: AgentName = Field(
        description=(
            "Agente que respondería si el usuario toca este chip. "
            "Debe ser un agente especializado: nunca 'general' ni 'oos'."
        )
    )

    entity: str = Field(
        description=(
            "Slug del nodo de Neo4j al que apunta el chip. Se usa para "
            "el bloqueo anti-supernodo y para telemetría de taps."
        )
    )


class SuggesterOutput(BaseModel):
    """Salida estructurada del nodo suggester. Lista vacía es válida."""

    suggestions: List[Suggestion] = Field(
        default_factory=list,
        max_length=_MAX_SUGGESTIONS,
        description=(
            "0 a 3 sugerencias. Devolver una lista vacía es el resultado "
            "correcto y esperado la mayoría de las veces."
        ),
    )


# =====================================================================
# 3. UTILIDADES DE TEXTO
# =====================================================================

def _normalize(text: str) -> str:
    """Minúsculas, sin puntuación, espacios colapsados."""
    text = text.lower().replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _content_tokens(text: str) -> Set[str]:
    """Tokens significativos: sin stopwords, sin palabras de 1-2 letras."""
    return {
        t for t in _normalize(text).split()
        if t not in _STOPWORDS and len(t) > 2
    }


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def answer_ends_with_question(state) -> bool:
    """
    True si la respuesta ya termina con una pregunta propia del
    synthesizer (ej. "¿Querés que revise el filtro?").

    En ese caso los chips son ruido: ya hay una pregunta abierta
    esperando respuesta. Heurística deliberadamente simple — solo mira
    el final del texto visible, no detecta preguntas retóricas a mitad
    de párrafo. Si aparecen falsos positivos en Langfuse, se afina.
    """
    response = state.get("response")
    if response is None:
        return False

    text = response.tier1_markdown().strip()
    return text.endswith("?") or text.endswith("？")


# =====================================================================
# 4. GATES DE CALIDAD (post-generación, determinísticos)
# =====================================================================
# Cada gate recibe y devuelve una lista, para poder componerlos y para
# poder medir individualmente cuál descarta más (paso 9 del plan).

def gate_routability(candidates: Sequence[Suggestion]) -> List[Suggestion]:
    """
    Gate 1 — Enrutabilidad.

    El agente sugerido debe existir en AgentName y debe ser un agente
    especializado. Un chip que el sistema no puede responder daña más
    la confianza que no mostrar nada.
    """
    return [
        c for c in candidates
        if c.agent in _VALID_AGENTS and c.agent not in _NON_SUGGESTABLE_AGENTS
    ]


def gate_length(candidates: Sequence[Suggestion]) -> List[Suggestion]:
    """
    Gate 2 — Longitud.

    <= 5 palabras y <= 28 caracteres. No se trunca: un label truncado
    pierde sentido y se ve peor que su ausencia. Se descarta.
    """
    out: List[Suggestion] = []
    for c in candidates:
        label = c.label.strip()
        if not label:
            continue
        if len(label.split()) > _MAX_LABEL_WORDS:
            continue
        if len(label) > _MAX_LABEL_CHARS:
            continue
        out.append(c)
    return out


def gate_anti_hub(candidates: Sequence[Suggestion]) -> List[Suggestion]:
    """
    Gate 4 — Anti-hub.

    Descarta chips que apuntan a supernodos genéricos. Chequea tanto
    el slug de `entity` como el texto del label, porque el LLM puede
    poner el supernodo en el label y un slug vecino en entity.
    """
    out: List[Suggestion] = []
    for c in candidates:
        entity_slug = _normalize(c.entity).replace(" ", "_")
        if entity_slug in SUPERNODES:
            continue

        label_tokens = _content_tokens(c.label)
        hub_tokens = {t for s in SUPERNODES for t in _content_tokens(s)}
        # "ph" queda fuera de _content_tokens por longitud; se chequea aparte.
        if label_tokens & hub_tokens:
            continue
        if "ph" in _normalize(c.label).split():
            continue

        out.append(c)
    return out


def gate_no_redundancy(
    candidates: Sequence[Suggestion],
    answer_text: str,
) -> List[Suggestion]:
    """
    Gate 5 — No redundancia.

    Descarta chips cuyo contenido ya fue respondido en tier 1. Dos
    señales: la entidad aparece literal en la respuesta, o los tokens
    de contenido del label ya están todos cubiertos por la respuesta.
    """
    answer_norm = _normalize(answer_text)
    answer_tokens = _content_tokens(answer_text)

    out: List[Suggestion] = []
    for c in candidates:
        entity_norm = _normalize(c.entity)
        if entity_norm and entity_norm in answer_norm:
            continue

        label_tokens = _content_tokens(c.label)
        # Si todo lo que dice el label ya está en la respuesta, es redundante.
        if label_tokens and label_tokens.issubset(answer_tokens):
            continue

        out.append(c)
    return out


def gate_mutual_distinction(candidates: Sequence[Suggestion]) -> List[Suggestion]:
    """
    Gate 6 — Distinción mutua.

    Dos chips no pueden ser reformulaciones entre sí. Se aplican dos
    criterios, en orden:
      1. Mismo destino (agent, entity) -> duplicado exacto.
      2. Labels con solapamiento de tokens >= _SIMILARITY_THRESHOLD.

    Gana el primero de la lista (se asume orden de relevancia del LLM).
    """
    kept: List[Suggestion] = []
    seen_targets: Set[Tuple[str, str]] = set()

    for c in candidates:
        target = (c.agent, _normalize(c.entity))
        if target in seen_targets:
            continue

        tokens = _content_tokens(c.label)
        if any(
            _jaccard(tokens, _content_tokens(k.label)) >= _SIMILARITY_THRESHOLD
            for k in kept
        ):
            continue

        seen_targets.add(target)
        kept.append(c)

    return kept


def gate_cardinality(candidates: Sequence[Suggestion]) -> List[Suggestion]:
    """
    Gate 3 — Cardinalidad.

    Máximo 3, preservando el orden del LLM. Va último en el pipeline:
    recortar antes desperdiciaría candidatos válidos si los primeros
    caen por otros gates.
    """
    return list(candidates[:_MAX_SUGGESTIONS])


# =====================================================================
# 5. PIPELINE
# =====================================================================

def apply_gates(
    candidates: Sequence[Suggestion],
    answer_text: str,
) -> List[Suggestion]:
    """
    Corre los 6 gates en orden. Los baratos y más selectivos primero;
    cardinalidad al final para no desperdiciar candidatos.
    """
    result = gate_routability(candidates)
    result = gate_length(result)
    result = gate_anti_hub(result)
    result = gate_no_redundancy(result, answer_text)
    result = gate_mutual_distinction(result)
    result = gate_cardinality(result)
    return result


def apply_gates_with_report(
    candidates: Sequence[Suggestion],
    answer_text: str,
) -> Tuple[List[Suggestion], dict]:
    """
    Igual que apply_gates pero devuelve cuántos descartó cada gate.

    Para el paso 9 del plan: "tasa de rechazo por gate (cuál gate
    descarta más)". Emitir este dict a Langfuse permite saber si el
    prompt hay que ajustarlo o si un gate está de más.
    """
    report: dict = {"input": len(candidates)}
    result = list(candidates)

    for name, fn in (
        ("routability", gate_routability),
        ("length", gate_length),
        ("anti_hub", gate_anti_hub),
    ):
        before = len(result)
        result = fn(result)
        report[name] = before - len(result)

    before = len(result)
    result = gate_no_redundancy(result, answer_text)
    report["no_redundancy"] = before - len(result)

    before = len(result)
    result = gate_mutual_distinction(result)
    report["mutual_distinction"] = before - len(result)

    before = len(result)
    result = gate_cardinality(result)
    report["cardinality"] = before - len(result)

    report["output"] = len(result)
    return result, report


# =====================================================================
# 6. ROSTER PARA EL PROMPT
# =====================================================================
# Segunda fuente de verdad respecto a AGENT_REGISTRY (que está keyed por
# nombres largos: "Pool Chemistry Agent"). Se mantiene a mano porque el
# suggester necesita una línea por agente, no el prompt completo.
#
# Ver el test de paridad sugerido al final de state.py: conviene uno
# análogo cruzando estas claves contra AgentName.

ROSTER_DESCRIPTIONS: dict[str, str] = {
    "chemistry":       "Química del agua: balance, desinfección, interpretación de tests.",
    "equipment":       "Equipos instalados: bombas, filtros, calentadores, fallas y mantenimiento.",
    "hydraulics":      "Caudal, turnover, pérdida de carga, circulación de sistemas instalados.",
    "operations":      "Rutinas diarias, mantenimiento preventivo, apertura y cierre.",
    "compliance":      "Códigos, permisos, inspecciones, requisitos normativos.",
    "contamination":   "Respuesta a incidentes biológicos activos: fecal, vómito, sangre.",
    "facility_design": "Diseño de instalaciones nuevas o renovaciones, dimensionamiento.",
    "safety":          "Supervisión, prevención de ahogamiento, barreras, planes de emergencia.",
    "recovery":        "Recuperación post-desastre: inundación, tormenta, corte prolongado.",
    "records":         "Diseño de logs, retención, documentación para inspección.",
    "math":            "Cálculo numérico: dosis, volumen, caudal, conversiones.",
}
# "general" y "oos" quedan fuera a propósito: _NON_SUGGESTABLE_AGENTS.


def roster_text() -> str:
    """Roster formateado para interpolar en el SUGGESTER_PROMPT."""
    return "\n".join(f"- {a}: {d}" for a, d in ROSTER_DESCRIPTIONS.items())