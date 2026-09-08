"""
prompt_archetype.py

Ensamblado de prompts, por nodo:

  build_cluster_prompt()                -> prompt de un nodo de cluster
  build_subagent_archetype_section()    -> sección 10 de BASE_POOL_AGENT_PROMPT
  build_synthesizer_archetype_section() -> sección 9 del prompt del Synthesizer

Rationale: el cluster emite BASE_OUTPUT_CONTRACT en JSON que el usuario nunca
ve. Solo el Synthesizer emite SynthesizerOutput, así que solo el Synthesizer
está sujeto a presupuesto de palabras, partición en tiers o `details`. Darle
presupuesto al cluster le quita evidencia al Synthesizer.

Todo literal que el validador impone se importa, nunca se retipea.
"""

from __future__ import annotations

from ..graph_context.response_contracts import get_contract, NO_CAP
from ..graph_context.response_validator import (
    MAX_ACTIONS,
    MAX_ACTION_WORDS,
    HAZARD_AGENTS,
    OVERFLOW_LABEL,
)
from .prompts import BASE_POOL_AGENT_PROMPT
from .prompts_sub_agents import (
    CLUSTER_REGISTRY,
    CALC_PROTOCOL,
    REVISION_PROTOCOL,
    contract_for,
    archetype_for,
)


# =====================================================================
# Bloques condicionales
# =====================================================================

def _details_lines(details: list[str]) -> str:
    return "\n".join(f"- {d}" for d in details) if details else "- (none)"


def _resolve_safety(safety_required, modules: list[str] | None):
    """
    `modules` son los module_id activos del turno (primario + apoyo).
    Antes era un solo `agent_key`; ahora un turno puede tener dos módulos
    y basta que uno sea de riesgo.
    """
    if safety_required is True:
        return True
    if safety_required == "conditional":
        if HAZARD_AGENTS.intersection(modules or []):
            return True
        return "conditional"
    return False


def _budget_block(budget: int) -> str:
    if budget >= NO_CAP:
        return (
            "### Length\n"
            "No word cap applies. Use only the length the task requires — "
            "length is not a proxy for thoroughness. Lead with the action that "
            "must happen first."
        )
    return f"""### Word budget
**Visible budget: {budget} words.**
This counts `answer` + `actions` + `safety` only — the text the user sees
before tapping anything. Words inside `details` are NOT counted.

* Never delete content to fit. Move it into `details` instead.
* `actions`: at most {MAX_ACTIONS} items, each ≤ {MAX_ACTION_WORDS} words.
  Order them so the most important is first — anything over budget is dropped
  from the end into a `details` section, not discarded.
* Never shorten `answer` or `safety` to fit. They are exempt from relocation,
  and an `answer` + `safety` that exceeds {budget} words on its own forces a
  full regeneration. A complete sentence over budget beats a truncated one."""


def _safety_block(resolved: bool | str) -> str:
    if resolved is True:
        return """### Safety (required)
`safety` must be populated: one imperative line naming the single most
consequential hazard and the action that avoids it. It is rendered in tier 1
and is never collapsed. Leaving it empty forces a regeneration.
Immediate protective action comes before any explanation."""
    if resolved == "conditional":
        return """### Safety (conditional)
Populate `safety` only when the turn involves chemical handling or dosing,
hazardous or energized equipment, electrical hazard, or pressurized systems.
A turnover or volume calculation needs no warning; an acid dose does.
When it applies the rules above are binding: one imperative line, tier 1, never
collapsed. Otherwise leave it null — no generic boilerplate."""
    return """### Safety
Leave `safety` null. Do not add generic precautions the task does not require."""


def _details_block(details: list[str]) -> str:
    if not details:
        return """### Details
This archetype folds nothing. Leave `details` empty — do not manufacture
sections to fill it."""
    return f"""### Details (tier 2, collapsible)
Populate a section for each category below that the evidence supports. Omit the
rest; do not invent one. If a category is genuinely unresolved, say so in one
line rather than fabricating content.

{_details_lines(details)}

* `label`: ≤ 5 words, in the user's language.
* Do not use a label containing "safety", "warning", "hazard", or "risk" for
  anything that is not a safety warning — such a section is auto-promoted into
  tier 1 and its first sentence becomes the visible warning.
* "{OVERFLOW_LABEL}" is reserved for the enforcement layer. Do not emit it."""


# =====================================================================
# Cluster (sección 10 de BASE_POOL_AGENT_PROMPT)
# =====================================================================

_SUBAGENT_TEMPLATE = """## 10. DOWNSTREAM RESPONSE SHAPE

Your JSON output is not shown to the user. A Synthesizer node consumes it and
renders the final answer. Your job is to supply the material that shape needs,
already sorted, so the Synthesizer never has to guess or infer.

**Target shape of the final answer:** {shape}

**Material the Synthesizer will need:**
{details}
Cover each item you have evidence for, in the fields of your output contract.
List anything you could not establish under `missing_information`.

**Do not self-truncate.** The Synthesizer, not you, is under a word budget.
Omitting evidence to look concise here removes it from the final answer
permanently. Be complete and non-redundant, not short.

{safety}"""


def build_subagent_archetype_section(
    archetype: str,
    modules: list[str] | None = None,
) -> str:
    c = get_contract(archetype)
    resolved = _resolve_safety(c.get("safety_required", False), modules)

    if resolved is True:
        safety = ("**Safety:** the final answer will carry a mandatory warning. "
                  "Surface every hazard and its mitigation explicitly — the "
                  "Synthesizer can only promote what you provide.")
    elif resolved == "conditional":
        safety = ("**Safety:** if your findings involve chemical handling or "
                  "dosing, hazardous equipment, electrical hazard, or pressure, "
                  "state the hazard and mitigation explicitly rather than "
                  "implying it.")
    else:
        safety = ("**Safety:** report hazards you actually find. Do not pad the "
                  "output with generic precautions.")

    return _SUBAGENT_TEMPLATE.format(
        shape=c["shape"],
        details=_details_lines(c["details"]),
        safety=safety,
    )


# =====================================================================
# Synthesizer (sección 9)
# =====================================================================

_SYNTH_TEMPLATE = """## 9. OUTPUT ARCHETYPE

**Archetype:** `{archetype}`
**Required shape:** {shape}

Follow that shape exactly; do not substitute a preferred format. Numbered steps
means numbered steps. A list means no narrative between items. A one-sentence
verdict means the verdict comes first, before any qualification.

{budget}

{details}

{safety}"""


def build_synthesizer_archetype_section(
    archetype: str,
    agents: list[str] | None = None,
) -> str:
    """
    `archetype` viene resuelto del módulo primario del plan.
    `agents` son los module_id que produjeron output, y colapsan un safety
    "conditional" del mismo modo que lo hará el validador.
    """
    c = get_contract(archetype)
    resolved = _resolve_safety(c.get("safety_required", False), agents)

    return _SYNTH_TEMPLATE.format(
        archetype=archetype,
        shape=c["shape"],
        budget=_budget_block(c.get("budget", NO_CAP)),
        details=_details_block(c["details"]),
        safety=_safety_block(resolved),
    )


# =====================================================================
# Builder de cluster
# =====================================================================

def build_cluster_prompt(
    cluster_name: str,
    primary_module: str,
    support_modules: list[str] | None = None,
    revising: bool = False,
) -> str:
    """
    Ensambla el prompt de un nodo de cluster para este turno.

    Reemplaza a build_agent_prompt(config), que recibía un AgentConfig por
    agente. Ahora el prompt se arma en tiempo de turno: el cluster es fijo,
    pero los módulos activos los elige el planner, así que el role framing,
    las responsabilidades y el contrato de salida varían por turno.

    El role_framing solo sale del módulo PRIMARIO -- un prompt con tres
    "You are a specialist in..." no tiene rol.
    """
    cluster = CLUSTER_REGISTRY[cluster_name]
    support = [m for m in (support_modules or []) if m != primary_module]

    primary = cluster.modules[primary_module]
    active = (primary, *(cluster.modules[m] for m in support if m in cluster.modules))

    responsibilities: list[str] = []
    for module in active:
        for item in module.responsibilities:
            if item not in responsibilities:
                responsibilities.append(item)

    # El scope del cluster va primero: contiene las precondiciones que
    # aplican siempre (frontera temporal en SYSTEMS, jurisdicción y
    # obligación-vs-artefacto en GOVERNANCE, CYA en AQUATIC_CHEM).
    specialization = cluster.scope
    if support:
        specialization += (
            f"\n\nThis turn also draws on: {', '.join(support)}. "
            "Apply that expertise in service of the primary task, not as a "
            "second answer."
        )

    excluded = list(cluster.hard_exclusions)
    # Sin orquestador no hay a dónde rutear mid-turn: las exclusiones dejan
    # de ser instrucciones de ruteo y pasan a ser fronteras. Lo no cubierto
    # se declara, no se deriva.
    excluded.append(
        "For anything above: do NOT answer it and do NOT route it. State "
        "plainly that it falls outside this turn's scope and name it in "
        "`missing_information` so it can be offered to the user as a "
        "follow-up."
    )

    protocols = [*cluster.safety_rules]
    if cluster.tools:
        protocols.append(CALC_PROTOCOL)
    if revising:
        protocols.append(REVISION_PROTOCOL)

    tool_instructions = cluster.tool_instructions or (
        "No tools are authorized for this cluster. Never state a computed "
        "numeric value from arithmetic you performed yourself."
    )
    if protocols:
        tool_instructions = tool_instructions + "\n\n" + "\n\n".join(protocols)

    return BASE_POOL_AGENT_PROMPT.format(
        agent_name=cluster.cluster_name,
        specialization=specialization,
        responsibilities="\n".join(f"- {r}" for r in responsibilities),
        excluded_tasks="\n".join(f"- {e}" for e in excluded),
        tools=", ".join(cluster.tools) or "(none)",
        tool_instructions=tool_instructions,
        output_contract=contract_for(active),
        archetype_section=build_subagent_archetype_section(
            archetype_for(primary),
            [primary_module, *support],
        ),
        tool_budget=cluster.tool_budget_per_calc_step * (1 + len(support)),
    )