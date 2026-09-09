"""
prompt_archetype.py

Section 9 of the prompt, split by node:

  build_subagent_archetype_section()   -> goes into BASE_POOL_AGENT_PROMPT
  build_synthesizer_archetype_section() -> goes into the Synthesizer prompt

Rationale: sub-agents emit BASE_OUTPUT_CONTRACT JSON that the user never sees.
Only the Synthesizer emits SynthesizerOutput, so only the Synthesizer is
subject to a word budget, tier partitioning, or `details`. Giving sub-agents a
budget starves the Synthesizer of evidence and the budgets do not compose
(3 agents x 90w != a better 80w answer).

Every literal that the validator enforces is imported, never retyped.
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

# =====================================================================
# Bloques condicionales
# =====================================================================

def _details_lines(details: list[str]) -> str:
    return "\n".join(f"- {d}" for d in details) if details else "- (none)"


def _resolve_safety(safety_required, agent_key):
    if safety_required is True:
        return True
    if safety_required == "conditional":
        return True if agent_key in HAZARD_AGENTS else "conditional"
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
# Sub-agente
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


def build_subagent_archetype_section(archetype: str,
                                     agent_key: str | None = None) -> str:
    c = get_contract(archetype)
    resolved = _resolve_safety(c.get("safety_required", False), agent_key)

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
# Synthesizer
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


def build_synthesizer_archetype_section(archetype: str,
                                        agents: list[str] | None = None) -> str:
    """
    `archetype` comes from resolve_archetype(...) at turn time, not from a
    static agent config. `agents` is state["assigned_agents"], used to collapse
    a "conditional" safety requirement the same way the validator will.
    """
    c = get_contract(archetype)
    required = c.get("safety_required", False)
    resolved = required
    if required == "conditional" and HAZARD_AGENTS.intersection(agents or []):
        resolved = True

    return _SYNTH_TEMPLATE.format(
        archetype=archetype,
        shape=c["shape"],
        budget=_budget_block(c.get("budget", NO_CAP)),
        details=_details_block(c["details"]),
        safety=_safety_block(resolved),
    )


# =====================================================================
# Builder corregido
# =====================================================================

def build_agent_prompt(config, agent_key: str | None = None) -> str:
    """
    Fixes vs. the previous version:
      - config.agent_name (the dataclass has no `name`).
      - get_contract() instead of ARCHETYPE_CONTRACTS[...] -> no KeyError.
      - archetype section pre-rendered, so no unused `budget`/`safety_required`
        kwargs silently doing nothing.
    """
    return BASE_POOL_AGENT_PROMPT.format(
        agent_name=config.agent_name,
        specialization=config.specialization,
        responsibilities="\n".join(f"- {i}" for i in config.responsibilities),
        excluded_tasks="\n".join(f"- {i}" for i in config.excluded_tasks),
        tools=", ".join(config.tools),
        tool_instructions=config.tool_instructions,
        output_contract=config.output_contract,
        archetype_section=build_subagent_archetype_section(
            config.archetype, agent_key
        ),
        tool_budget=config.tool_budget
    )