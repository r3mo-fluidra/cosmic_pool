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


from ..tool_budgets import RETRIEVAL_TOOL_BUDGETS
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


def _resolve_safety(safety_required, agents=None):
    """
    `agents` llega como str desde build_agent_prompt (un especialista) o como
    list[str] desde build_synthesizer_archetype_section (los agentes que
    produjeron output). La firma anterior esperaba solo str, así que la
    llamada del synthesizer evaluaba `["chemistry"] in HAZARD_AGENTS` ->
    TypeError: unhashable type: 'list'. No explotó porque ningún arquetipo
    ejercitado hasta ahora resuelve a "conditional".

    Y desde el especialista llegaba None, porque agents.py nunca pasaba el
    segundo argumento: la rama True era inalcanzable y todo agente de riesgo
    recibía la instrucción blanda.
    """
    if safety_required is True:
        return True
    if safety_required == "conditional":
        if isinstance(agents, str):
            agents = [agents]
        if HAZARD_AGENTS.intersection(agents or ()):
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
before tapping anything. Words inside `details` and `readings` are NOT counted.

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
consequential hazard and the action that avoids it. It names what the operator
cannot work out from the readings alone — a product incompatibility, a handling
risk, a contamination exposure. It is NOT a restatement of `actions[0]`, and
NOT a generic precaution. It is rendered in tier 1 and is never collapsed.
Immediate protective action comes before any explanation."""
    if resolved == "conditional":
        return """### Safety (conditional)
Populate `safety` only when the turn involves chemical handling or dosing,
hazardous or energized equipment, electrical hazard, or pressurized systems.
A turnover or volume calculation needs no warning; an acid dose does.
When it applies the rules above are binding: one imperative line, tier 1, never
collapsed, naming what the operator cannot work out from the readings alone —
never a restatement of `actions[0]` and never a generic precaution. Otherwise
leave it null — no generic boilerplate."""
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

def build_test_readings_section(has_test_interpretation: bool) -> str:
    """
    Return the reading-panel rules, or an empty string when the turn has none.

    The panel rules are ~55 lines of parameter-grading guidance that only make
    sense when a specialist emitted `test_interpretation`. Sending them on a
    hydraulics or equipment turn spends tokens on vocabulary from a domain the
    turn does not touch, and that vocabulary leaks: trace b986219a produced
    "calculate your actual turnover and dosing" on a payload that never
    mentioned dosing.
    """
    if not has_test_interpretation:
        return ""

    return """## Test readings (`test_interpretation`)
When the raw content carries per-parameter entries, they are the answer, not
background. Three rules, and the first one is not negotiable.

**1. You never grade a reading. The panel is generated, not written.**

Each parameter's line is built by template from its own `status`,
`measured` and `regulatory_limit`, after you finish. `at_floor` and
`at_ceiling` come out as compliant-with-no-margin, because a value sitting on
a published bound complies with it.

So do not re-state that grading in prose, and above all do not intensify it:
no "extremely high" over a value the specialist called `at_ceiling`, no
"critical" over an "elevated". Implying a breach that did not happen
misstates the facility's regulatory position, and an operator who repeats it
to an inspector reports a violation that does not exist.

The sharpest form of that error is calling a number a code violation. The
knowledge base publishes educational bands, not code bounds — it says so on
every page that carries a figure — so "above the published range" is a claim
you can support and "above the legal maximum" is not. Where a panel line reads
"educational range, not a code limit", your prose may not upgrade it. The
finding is still real and still worth stating plainly; what you do not have is
the authority to attach to it. The limit that governs is the local code, and
this system does not know it.

What `answer` is for is what the panel cannot say: which single reading forces
the closure, what mechanism connects them, and what produced the state.

**2. Attribute a closure to the reading that causes it.** When several
parameters are off but only one triggers the stop, say which one. Listing a
compliant parameter among the reasons for a closure is the same error as
calling it a violation.

**3. `operating_target` is distinct from `regulatory_limit`.** The target is
the number the operator dials to; the limit is the floor they must not cross.
If the two differ, the target is what they act on and the limit is context. A
missing input that blocks a dose does not block stating the target.

"""


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

**Write notes, not prose.** Your reader is a machine that rewrites everything
you produce. Facts cost it nothing; sentences it has to re-parse cost it work.

* One fact per line. Fragments beat sentences.
* No opening, no transitions, no summary, no closing remark.
* Do not restate the task or read the user's own numbers back to them.
* Do not narrate your reasoning or announce what you are about to say.
* Numbers, thresholds, chapter references and node ids go verbatim — those
  are the parts the Synthesizer cannot reconstruct.

**Completeness is not length.** Never drop a fact to be shorter: an omission
here is permanent, because the Synthesizer cannot recover what you did not
send. Cut wording, never evidence. If everything you hold fits in a dozen
lines, that is a complete answer.

**Two things are never compressed.** Brevity does not apply to them:

* **Handling hazards for anything you tell the user to add.** If your output
  recommends dosing a product, the hazard of handling that product is
  evidence, not wording — incompatibility between products, gas release on
  contact, order of addition, required PPE. A recommendation to add acid and
  chlorine that omits the mixing hazard is incomplete, not concise.
* **`missing_information`.** If you recommend a dose, a volume, or any value
  that depends on inputs you were not given, name every missing input.
  Leaving it empty asserts that you established everything.

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

`readings` is not yours to write: leave it as an empty array. The system
renders the panel from the specialist payload, and it does not repeat what you
put there. Do not restate measured figures in `answer` beyond the one that
carries the verdict.

Where `actions` apply, keep the order the specialist gave them, most important
first. When the raw content carries an `order_rationale`, the sequence it
describes IS the order — do not resequence it.

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
    static agent config. `agents` is the list of agent names that produced
    usable results this turn, used to collapse a "conditional" safety
    requirement the same way the validator will.
    """
    c = get_contract(archetype)
    required = c.get("safety_required", False)
    resolved = required
    if required == "conditional" and HAZARD_AGENTS.intersection(agents or []):
        resolved = True

    shape = c["shape"]
    if c.get("actions_optional"):
        shape += (
            "\n\n**`actions` may be empty, and usually should be.** This "
            "question asked for understanding, not for work. Emit an action "
            "ONLY if the raw content states something the user must actually "
            "do; never add one to fill the field. An explanation followed by "
            "unrequested chores reads as evasion, and the advice competes for "
            "the word budget with the answer itself.\n"
            "If the question asked for a quantity, a fraction or a ratio, that "
            "number belongs in `answer`. An `answer` that discusses the "
            "quantity without stating it has not answered the question."
        )

    return _SYNTH_TEMPLATE.format(
        archetype=archetype,
        shape=shape,
        budget=_budget_block(c.get("budget", NO_CAP)),
        details=_details_block(c["details"]),
        safety=_safety_block(resolved),
    )


# =====================================================================
# Builder corregido
# =====================================================================
def _render_tool_budget(config) -> str:
    """
    Declara al agente el presupuesto REAL que _gate va a aplicar.

    Para los agentes de retrieval el límite no es un pozo común: son caps por
    tool. Anunciar un total intercambiable hace que el agente reintente la
    misma tool creyendo que le queda saldo, y cada rechazo cuesta un round
    trip de LLM sin devolver evidencia.
    """
    caps = {t: RETRIEVAL_TOOL_BUDGETS[t] for t in config.tools
            if t in RETRIEVAL_TOOL_BUDGETS}

    if not caps:
        return (
            f"Hard limit: **{config.tool_budget} tool calls** this turn. "
            "Count every call to any authorized tool."
        )

    lineas = "\n".join(f"  {t:<20} {n}" for t, n in caps.items())
    return (
        f"Hard limit: **{sum(caps.values())} tool calls** this turn, allocated "
        f"per tool. They are NOT interchangeable:\n\n{lineas}\n\n"
        "Spending the vector_search call does not free a second "
        "search_seed_nodes. A call beyond a tool's own cap is refused by the "
        "system and returns nothing — it does not fail over to another tool."
    )

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
        tool_budget_block=_render_tool_budget(config),
    )