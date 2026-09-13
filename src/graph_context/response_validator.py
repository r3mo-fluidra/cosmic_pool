"""
response_validator.py

Deterministic enforcement of the response contract emitted by the synthesizer.

The prompt ASKS the LLM to respect the budget; this module GUARANTEES it.
Guiding principle: never delete information, only relocate it to `details`.

Usage:
    payload, report = enforce_contract(payload, contract, state["assigned_agents"])
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from .water_targets import (
    closure_required, is_published_figure, resolve_panel, target_band,
)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MAX_ACTIONS = 4

#: Word cap per bullet.
#:
#: Its ONLY job is to catch a MALFORMED bullet: a whole paragraph where a line
#: belonged. It is NOT a selection criterion.
#:
#: It was 12, then 22, and both times it discarded the turn's central action —
#: the last one by a single word. The cause is that length is orthogonal to
#: importance: the specialist orders by priority, and its most important
#: actions are the long ones because they carry product, quantity and target.
#: With MAX_ACTIONS=4 the worst case is ~160 visible words, comfortable inside
#: any archetype budget, so the real trimming is done by MAX_ACTIONS and
#: `overflow_to_details`, not by this.
MAX_ACTION_WORDS = 40

NO_CAP = 9999  # sentinel budget: `critical` archetype, no ceiling

# Agents whose content implies chemical handling or direct risk.
# They resolve `safety_required = "conditional"` to True.
HAZARD_AGENTS = {"chemistry", "contamination", "safety", "math", "recovery"}

# Lexical fallback in case the archetype is `calculation` with no hazardous
# agent but the content still describes product handling.
HAZARD_PATTERN = re.compile(
    r"\b(ácido|acido|acid|cloro|chlorine|hipoclorito|hypochlorite|muriático|muriatic|"
    r"tricloro|dicloro|bromo|bromine|ozono|ozone|soda\s+cáustica|caustic|"
    r"peróxido|peroxide|alguicida|algaecide|EPP|PPE)\b",
    re.IGNORECASE,
)

# Labels used when relocating overflow content.
OVERFLOW_LABEL = "Next actions (overflow)"
SAFETY_LABEL_PATTERN = re.compile(r"safety|warning|hazard", re.IGNORECASE)

#: Provenance notice, ONCE at the foot of the panel — not glued to every line.
#:
#: An earlier version hung "educational range, not a code limit" off every
#: verdict, which left the response contradicting itself: the prose said "you
#: must close the pool because free chlorine is below the required 2.0 ppm
#: minimum" and the panel, two lines below, said that minimum was not a code
#: limit. An operator who needs to close just received the perfect excuse not
#: to.
PANEL_FOOTER = {
    "en": ("Limits follow the CDC Model Aquatic Health Code where published; "
           "ranges marked *typical* are operating practice. Your state or "
           "county code governs — verify against it."),
    "es": ("Los límites siguen el CDC Model Aquatic Health Code donde está "
           "publicado; los rangos marcados *habitual* son práctica operativa. "
           "Rige el código de tu estado o condado: verificalo."),
}


# --------------------------------------------------------------------------
# Validation report (feeds the observability metrics)
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
    #: The contract required `safety` and there was nothing to put there. This
    #: NO LONGER triggers a retry: now that the line is derived from the
    #: specialist payload (`render_safety`), the real case that fired it — a
    #: model omitting it while holding the data — is solved without calling
    #: the model again. Still measured: if this comes back full, the payload
    #: did not support a warning either.
    safety_missing: bool = False
    answer_exceeds_budget: bool = False  # -> triggers retry
    #: `actions` / `safety` built by code from the specialist payload rather
    #: than accepted as the model wrote them.
    actions_rendered: bool = False
    safety_rendered: bool = False
    #: Parameters whose `regulatory_limit` turned out to be a relabelled
    #: practice band. While this stays full, the specialist keeps presenting
    #: targets as code, and all that changed is that we now detect it.
    limits_demoted: list[str] = field(default_factory=list)
    #: Parameters that DO have a regulatory bound and kept the code verdict.
    #: Counterpart to `limits_demoted`: if this empties out, something broke
    #: `_CODE_BACKED` and the panel stopped asserting limits that exist.
    limits_kept: list[str] = field(default_factory=list)
    #: Parameters whose `operating_target` was filled in from the band because
    #: the specialist left it empty.
    targets_filled: list[str] = field(default_factory=list)
    #: Parameters whose `operating_target` had to be REPLACED because what the
    #: specialist supplied was implausible: zero, an echo of the measured
    #: value, or a figure outside the band. Distinct from `targets_filled`.
    targets_replaced: list[str] = field(default_factory=list)
    #: Readings whose `status` disagreed with the band-derived one, formatted
    #: as "pH: above_maximum→in_range". This is the direct measure of the
    #: oscillation: the specialist judging the same numbers two ways across
    #: two runs. The panel no longer reflects it, but the turn produced it.
    status_corrected: list[str] = field(default_factory=list)
    #: The panel forces a closure and the action was inserted by code.
    closure_inserted: bool = False
    #: Out-of-range readings the visible tier omitted.
    readings_missing: list[str] = field(default_factory=list)
    readings_appended: bool = False
    #: Quantities in the visible tier that do not appear in the source
    #: material. The worst failure mode per the synthesizer's own prompt.
    unsupported_numbers: list[str] = field(default_factory=list)
    #: `safety` repeats an action instead of adding something new. Measured,
    #: not corrected: dropping a safety line is worse than repeating one.
    safety_duplicates_action: bool = False
    #: Actions asking to correct a parameter the panel reported as in range.
    #: Measured only: the fix is for the specialist to draft against the
    #: reconciled readings, and that is a graph change, not a validator one.
    actions_vs_panel: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def needs_retry(self) -> bool:
        """
        One retry only. If it fails again, the degradation is accepted.

        `safety_missing` was removed from here. It was the trigger that fired
        most — seven rounds in a row — and every firing costs a full model
        call, 2–3.5 s of pure latency per turn, to ask for a line that is now
        built by template from the payload. When `render_safety` returns None
        it is because the payload supports no specific warning, and asking
        again does not change the data: it produces a generic one.
        """
        return self.answer_exceeds_budget

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Phrasing tables — the model writes none of this
# --------------------------------------------------------------------------

#: The wording of each status, by template. The model does NOT write it.
#:
#: Four evaluation rounds on the same query showed it does not hold as a
#: prompt instruction. The observed failures, all on data the specialist had
#: classified correctly:
#:   - "extremely high" over an at_ceiling (intensification)
#:   - an in_range turned into "above the maximum" (re-grading)
#:   - "above the 120 ppm operating ceiling" over an entry with no limit
#:     (an operating target presented as a regulatory ceiling)
#: All three travel just as far: an operator repeating them to an inspector
#: reports a violation that does not exist.
#:
#: STRUCTURE: nested by language, like `_BAND_PHRASING`. An earlier version
#: defined it flat and under a different name, so `reading_note` raised
#: NameError as soon as a reading did NOT derive from a band.
#:
#: `{limit}` is filled with `regulatory_limit` ALREADY FORMATTED WITH ITS UNIT
#: by `_with_unit`, so the template carries no `{u}`: "{limit}{u}" produced
#: "2 ppm ppm" and, worse, a KeyError because `.format()` only receives
#: `limit`.
_STATUS_PHRASING = {
    "en": {
        "below_minimum": "in violation, below the {limit} minimum",
        "at_floor":      "compliant, no margin at the floor",
        "in_range":      "in range",
        "at_ceiling":    "compliant, no margin at the ceiling",
        "above_maximum": "in violation, above the {limit} cap",
    },
    "es": {
        "below_minimum": "en infracción, por debajo del mínimo de {limit}",
        "at_floor":      "cumple, sin margen en el piso",
        "in_range":      "en rango",
        "at_ceiling":    "cumple, sin margen en el techo",
        "above_maximum": "en infracción, por encima del tope de {limit}",
    },
}

#: The same verdict when the figure backing it is an OPERATING PRACTICE band
#: rather than a code limit. The value is still outside the published range —
#: that is actionable and is said — but without attributing regulatory
#: authority it does not have.
#:
#: The disclaimer does NOT live here. It used to be glued to each line and
#: left the response contradicting its own prose; it now lives once in
#: `PANEL_FOOTER`. The word "typical"/"habitual" already marks the difference
#: inside the line, which is all the line needs.
_BAND_PHRASING = {
    "en": {
        "below_minimum": "below the typical {bound} floor",
        "at_floor":      "at the floor of the typical range",
        "in_range":      "in range",
        "at_ceiling":    "at the top of the typical range",
        "above_maximum": "above the typical {bound} ceiling",
    },
    "es": {
        "below_minimum": "por debajo del piso habitual de {bound}",
        "at_floor":      "en el piso del rango habitual",
        "in_range":      "en rango",
        "at_ceiling":    "en el techo del rango habitual",
        "above_maximum": "por encima del techo habitual de {bound}",
    },
}

#: For a reading whose status does not hold: report the value and, if there is
#: one, the target — with no compliance language in either direction.
_NO_VERDICT = {
    "en": "reported; no code bound available",
    "es": "reportado; sin límite normativo disponible",
}
_REPORTED_WITH_TARGET = {
    "en": "reported; operating target {target}",
    "es": "reportado; objetivo operativo {target}",
}

#: The operating target, when there IS a verdict. The limit says where the
#: violation begins; the target says where to leave the water. A panel
#: carrying only the limit leaves the operator correcting to the edge of the
#: violation.
#:
#: It goes as a suffix and with its own word ("target"/"objetivo") precisely
#: so it is not confused with the regulatory figure: one observed failure was
#: an operating_target presented as a code ceiling.
_TARGET_SUFFIX = {
    "en": "; target {target}",
    "es": "; objetivo {target}",
}

#: The target when `constraint_conflict` says it CANNOT be reached.
#:
#: The panel was printing "target 6.75 ppm" while the prose of the same turn
#: said that level exceeds the permitted maximum. An unqualified target reads
#: as achievable, and the synthesizer prompt already warned exactly that:
#: "never present the in-range target on its own when this field is set —
#: alone it reads as achievable and sufficient, and it is neither".
_TARGET_BLOCKED_SUFFIX = {
    "en": "; target {target} — blocked until {fix} comes down",
    "es": "; objetivo {target} — bloqueado hasta bajar {fix}",
}

#: Parameters with a REAL regulatory bound, not a practice band. These are not
#: demoted: they keep "in violation, below the 2 ppm minimum".
#:
#: That the corpus lacks the ingested citation is a retrieval problem, not a
#: reason to tell the operator the limit does not exist. And the cost
#: asymmetry runs the other way than for pH: failing to assert the pH code
#: leaves the operator with the number and no label; failing to assert the
#: free chlorine minimum puts "not a code limit" under the figure that
#: triggers a mandatory closure.
#:
#: NOTE — the free chlorine minimum depends on stabilizer (2.0 ppm with CYA
#: present, 1.0 without). The band in `water_targets` must encode the
#: stabilized case for this citation to be correct; verify it there.
_CODE_BACKED = frozenset({
    "free chlorine", "fc",
    "combined chlorine", "cc",
    "cyanuric acid", "cya",
})

#: Unit per parameter. The specialist emits `measured` as a bare number —0.8,
#: 7.9, 90.0— so the unit is lost between its payload and the screen: in trace
#: 148acb15 all seven values arrived without ppm. A figure without a unit is
#: not a reading, it is a number, and 0.8 without ppm tells nobody the pool is
#: in violation.
#:
#: Temperature is NOT in the table, on purpose. The specialist returns 82 with
#: no scale, and choosing one here is invention: 82 °F and 82 °C describe two
#: unrelated pools. No unit is worse than the right one, but far better than
#: the wrong one.
_UNITS = {
    "free chlorine":     "ppm",
    "combined chlorine": "ppm",
    "total chlorine":    "ppm",
    "cyanuric acid":     "ppm",
    "total alkalinity":  "ppm",
    "calcium hardness":  "ppm",
    "salt":              "ppm",
    "tds":               "ppm",
    "orp":               "mV",
    "ph":                "",
    # Acronyms: the specialist alternates between long name and acronym.
    "fc": "ppm", "cc": "ppm", "tc": "ppm",
    "cya": "ppm", "ta": "ppm", "ch": "ppm",
}

#: Parameter spellings that do not survive a .title(). "ph" -> "Ph" is a name
#: no operator writes.
_SPELLING = {
    "ph": "pH", "orp": "ORP", "tds": "TDS", "cya": "CYA", "lsi": "LSI",
    "fc": "FC", "cc": "CC", "ta": "TA", "ch": "CH", "ppm": "ppm",
}


def _key(parameter: str) -> str:
    """Normalized parameter name: 'Free_Chlorine' -> 'free chlorine'."""
    return re.sub(r"[^a-z0-9]+", " ", (parameter or "").lower()).strip()


def unit_for(parameter: str) -> str:
    """The unit of a parameter, or "" if unknown or not applicable (pH)."""
    return _UNITS.get(_key(parameter), "")


def is_code_backed(parameter: str) -> bool:
    """Does this parameter have a published regulatory bound, not a band?"""
    return _key(parameter) in _CODE_BACKED


def _lang_table(mapping: dict, language: str) -> dict:
    """
    The language sub-table, falling back to English.

    `or` rather than `.get(language, default)`: if someone later adds
    `"fr": {}` the default would not kick in and the empty dict would break
    downstream. The fallback is English because the specialist writes in
    English and that is the language the panel always has material in.
    """
    return mapping.get(language) or mapping["en"]


def _format_number(value) -> str:
    """
    No padding zeros: 2.0 -> '2', 7.8 -> '7.8'.

    A string is returned untouched: if the specialist wrote "0.8 ppm", that
    unit is theirs and beats the bare number.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value}"


def _with_unit(value, unit: str) -> str:
    """The number with its unit, without duplicating one already present."""
    text = _format_number(value)
    if not unit or not text:
        return text
    if any(c.isalpha() for c in text):
        # "0.8 ppm" already arrives complete: appending gives "0.8 ppm ppm".
        return text
    return f"{text} {unit}"


def _format_parameter(name: str) -> str:
    clean = name.replace("_", " ").strip()
    if not clean:
        return clean
    if clean.lower() in _SPELLING:
        return _SPELLING[clean.lower()]
    if not clean.islower():
        # Already capitalized: the specialist chose its spelling.
        return clean
    return " ".join(_SPELLING.get(w, w.capitalize()) for w in clean.split())


# --------------------------------------------------------------------------
# Text utilities
# --------------------------------------------------------------------------

#: Numbers that need no backing: they are language, not quantities.
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_TRIVIAL_NUMBERS = frozenset({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                              "10", "12", "24", "48", "72", "100"})

#: Minimal stopwords, es/en. No external dependency needed: all that matters
#: is that "the/de/la/to" do not inflate the similarity between short phrases.
_STOPWORDS = frozenset({
    "el", "la", "los", "las", "un", "una", "de", "del", "a", "al", "en", "y",
    "o", "que", "no", "se", "su", "hasta", "para", "con", "por",
    "the", "an", "of", "to", "in", "and", "or", "not", "your", "until",
    "is", "are", "be", "all",
})
_WORD_RE = re.compile(r"[a-záéíóúñü]+", re.IGNORECASE)
_DUPLICATE_THRESHOLD = 0.55
#: Stem truncation, not lemmatization. "cerrada"/"cerrar" and "closed"/"close"
#: describe the same action and without this they look nothing alike: literal
#: comparison scored 0.5 and 0.4 on what is obviously a repetition. Five chars
#: is short enough to absorb inflection in both languages and long enough not
#: to conflate distinct words.
_STEM = 5


def _content_tokens(text: str) -> set[str]:
    return {w.lower()[:_STEM] for w in _WORD_RE.findall(text or "")
            if w.lower() not in _STOPWORDS and len(w) > 2}


def _words(text: str | None) -> int:
    if not text:
        return 0
    return len(text.split())


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return parts[0] if parts else text.strip()


def unsupported_numbers(payload, raw_content: str) -> list[str]:
    """
    Quantities that appear in the visible tier and not in the source material.

    The synthesizer prompt calls this the system's worst failure mode ("never
    invent a dosage... filling a gap to satisfy a shape"), and it happened
    anyway: with `calculation_request` unexecuted, the synthesizer wrote
    "drain and refill thirty to forty percent". The number was nowhere in the
    payload, and it was also wrong — neither fraction reached the stabilizer
    target the agent itself had asked for.

    An invented number is worse than a missing one: it arrives with the same
    confidence as the real ones and the operator cannot tell them apart.

    Detects, does not correct. Rewriting the sentence containing it would
    require understanding the sentence; what can be asserted unambiguously is
    that the number has no backing, and that is enough to decide a retry and
    to measure frequency in Langfuse.
    """
    if not raw_content:
        return []

    def _canon(x: str) -> str:
        """3.0 and 3 are the same number; 90 and 9 are not.

        A bare `rstrip("0")` turned "90" into "9", so a 90 in the panel did
        not match the 90.0 in the source and was reported as invented. Zeros
        are only stripped when there is a decimal part to strip.
        """
        x = x.replace(",", ".")
        return x.rstrip("0").rstrip(".") if "." in x else x

    source_numbers = {_canon(o) for o in _NUMBER_RE.findall(raw_content)}

    # Only `answer`, `actions` and `safety`: what the MODEL writes. The
    # `readings` field is built by this module from the specialist payload, so
    # its figures come from the source by definition and counting them would
    # only produce noise.
    model_written = " ".join([
        getattr(payload, "answer", "") or "",
        getattr(payload, "safety", "") or "",
        *(getattr(payload, "actions", None) or []),
    ])

    # "thirty to forty percent" carries no digits: numbers spelled as words
    # escape this check, and that is a conscious limitation. It covers the
    # frequent case (figures) without risking false positives from a numeral
    # parser in two languages.
    suspects = []
    for n in _NUMBER_RE.findall(model_written):
        if n in _TRIVIAL_NUMBERS or _canon(n) in source_numbers:
            continue
        suspects.append(n)

    return suspects


# --------------------------------------------------------------------------
# Status coherence and reconciliation against published bands
# --------------------------------------------------------------------------

#: Statuses that assert a violation. They only hold if there is a regulatory
#: bound to measure them against.
_VIOLATION_STATUSES = ("below_minimum", "above_maximum")


def coherent_status(reading: dict) -> str | None:
    """
    A reading's status, demoted if it contradicts itself.

    An `above_maximum` with `regulatory_limit` null asserts that a maximum the
    entry itself says it does not know was exceeded. Measured: the specialist
    returned total alkalinity as above_maximum with no limit, judging it
    against its operating_target — and the parameter was not only compliant,
    it was practically on target. Declaring a healthy parameter in violation
    travels just as far as calling a ceiling a violation: an operator
    repeating it to an inspector reports a violation that does not exist.

    Demoted to None rather than to "in_range": we do not know it is in range,
    we know we cannot assert the opposite. None keeps the reading out of the
    required set and out of the text this module generates, which is the safe
    behaviour when the data contradicts itself.

    A reading already reconciled against a band (`band_derived`) is exempt:
    its verdict rests on `educational_bound`, not on `regulatory_limit`, and
    that field is populated.
    """
    status = reading.get("status")
    if reading.get("band_derived"):
        return status
    if status in _VIOLATION_STATUSES and reading.get("regulatory_limit") is None:
        return None
    return status


def _is_published_bound(name: str, limit) -> bool:
    """
    Is the `regulatory_limit` the specialist supplied a relabelled published
    band rather than a real citation?

    Delegates to `water_targets.is_published_figure`. If that function has a
    different signature, we assume yes (the module's previous behaviour, which
    demoted everything) — but only parameters that are NOT `_CODE_BACKED` ever
    reach here.
    """
    if limit is None:
        return False
    try:
        return bool(is_published_figure(name, limit))
    except TypeError:
        return True


def _target_is_plausible(name: str, target, measured) -> bool:
    """
    Does the specialist's `operating_target` work as a target?

    Three rejections, all three observed in real traces:

      - ZERO. No pool parameter has a target of zero. Combined chlorine came
        back with 0.0 in six consecutive runs and the panel printed
        "target 0 ppm" — zero chloramines is not an operable state.
      - ECHO OF THE MEASURED VALUE. Temperature came back with
        `operating_target: 82.0` over `measured: 82.0`: that is the reading
        repeated, not a target.
      - OUTSIDE THE BAND. If the target, treated as a reading, would not
        resolve in range, it is a target aimed at the violation.

    The third is evaluated by running `resolve_panel` on a synthetic item
    rather than reading the band's endpoints, because the internal shape of
    `target_band` is not part of this module's contract.
    """
    if target is None:
        return False
    try:
        value = float(target)
    except (TypeError, ValueError):
        return True  # text like "7.4-7.6": written deliberately
    if value == 0:
        return False
    if measured is not None:
        try:
            if abs(value - float(measured)) < 1e-6:
                return False
        except (TypeError, ValueError):
            pass
    try:
        status, _ = resolve_panel([{"parameter": name, "measured": value}])[0]
    except Exception:
        return True
    return status in ("in_range", "at_floor", "at_ceiling")


def reconcile_with_published_bands(test_interpretation, report=None) -> list[dict]:
    """
    Replace `status` and `regulatory_limit` with what the table supports.

    THREE THINGS, AND THE SECOND IS THE ONE THAT STOPS THE OSCILLATION.

    1. The limit. A `regulatory_limit` matching an endpoint of the published
       band is that band relabelled, and it is withdrawn.

       WITH ONE EXCEPTION: `_CODE_BACKED` parameters keep the regulatory
       verdict. Free chlorine, combined chlorine and cyanuric acid DO have
       bounds in the MAHC; that the corpus lacks the citation is a retrieval
       failure and does not license telling the operator the limit does not
       exist. An earlier version demoted everything unconditionally —
       `is_published_figure` was imported and never called — and the result
       was a panel saying "not a code limit" under the figure that triggers a
       mandatory closure, contradicting the prose of the same turn.

    2. The status. It is DERIVED from the measured value against the band, and
       whatever the specialist said is discarded. An earlier version replaced
       the limit and left the status to the model, so a total alkalinity of
       130 tagged `above_maximum` rendered as "above the 180 ceiling" — with
       130 < 180. The same panel gave different verdicts on the same numbers
       across runs, and that is the oscillation.

    3. The operating target. Filled from the band when the specialist left it
       empty, and REPLACED when what it supplied is implausible (see
       `_target_is_plausible`). Checking only `is None` was not enough: a
       target of 0.0 is not None and printed as "target 0 ppm" for six runs.

    DELIBERATE ASYMMETRY on non-regulatory parameters: a real jurisdiction may
    set its pH maximum at exactly 7.8, and in that case this withdraws a true
    limit. Accepted. Failing to assert a code that exists leaves the operator
    with the number and no label; asserting one that does not exist makes them
    report an invented violation to an inspector.

    A parameter the table does NOT cover is left untouched: there is no basis
    to withdraw its limit or to judge it, and declaring it in range out of
    ignorance would be a clearance nobody issued.

    Returns a new list; does not mutate the caller's dicts.
    """
    if not isinstance(test_interpretation, list):
        return []

    resolved = resolve_panel(test_interpretation)
    output = []
    for r, (status, bound) in zip(test_interpretation, resolved):
        if not isinstance(r, dict):
            continue
        r = dict(r)
        name = str(r.get("parameter", ""))
        band = target_band(name)

        if band is not None:
            if status is not None:
                if report is not None and status != r.get("status"):
                    report.status_corrected.append(
                        f"{_format_parameter(name)}: {r.get('status')}→{status}"
                    )
                r["status"] = status

            if is_code_backed(name):
                # Regulatory bound: the code verdict is kept. If the
                # specialist did not supply the limit, the band's is used —
                # for these parameters that IS the MAHC figure.
                if r.get("regulatory_limit") is None and bound is not None:
                    r["regulatory_limit"] = bound
                r["band_derived"] = False
                if report is not None:
                    report.limits_kept.append(_format_parameter(name))
            elif (_is_published_bound(name, r.get("regulatory_limit"))
                  or r.get("regulatory_limit") is None):
                # Practice band: regulatory authority is withdrawn and the
                # verdict moves to the "typical"/"habitual" phrasing.
                if r.get("regulatory_limit") is not None and report is not None:
                    report.limits_demoted.append(_format_parameter(name))
                r["regulatory_limit"] = None
                if status is not None:
                    r["band_derived"] = True
                    r["educational_bound"] = bound
            else:
                # Supplied a limit that does NOT match the band: it may be a
                # real jurisdictional citation. Respected as it came.
                if report is not None:
                    report.limits_kept.append(_format_parameter(name))
                r["band_derived"] = False

            current_target = r.get("operating_target")
            if not _target_is_plausible(name, current_target, r.get("measured")):
                replacement = band.target_text()
                if replacement:
                    r["operating_target"] = replacement
                    if report is not None:
                        label = _format_parameter(name)
                        (report.targets_filled if current_target is None
                         else report.targets_replaced).append(label)
                elif current_target is not None:
                    # No band target to substitute: withdraw it rather than
                    # print one that does not serve.
                    r["operating_target"] = None
                    if report is not None:
                        report.targets_replaced.append(_format_parameter(name))

        output.append(r)
    return output


def required_readings(test_interpretation) -> list[dict]:
    """
    The readings the visible tier cannot omit.

    No longer used by `enforce_contract` — `enforce_visible_readings` rebuilds
    the whole panel, in-range readings included — but kept because it is part
    of the module's public surface and may be imported by tests or by the
    synthesizer node.
    """
    if not isinstance(test_interpretation, list):
        return []
    return [
        r for r in test_interpretation
        if isinstance(r, dict) and coherent_status(r) not in (None, "in_range")
    ]


# --------------------------------------------------------------------------
# Reading panel
# --------------------------------------------------------------------------

def _target_blocked_by(reading: dict, conflict: dict | None) -> str | None:
    """
    The parameter to correct, if THIS reading's target is the one the conflict
    declares unreachable. None otherwise.

    Matching is by VALUE (`operating_target` == `needed_level`), not by name:
    `parameter_to_correct` names the parameter to MOVE — cyanuric acid — not
    the one that ended up blocked — free chlorine — and the payload has no
    field for the latter.
    """
    if not isinstance(conflict, dict):
        return None
    needed = conflict.get("needed_level")
    target = reading.get("operating_target")
    if needed is None or target is None:
        return None
    try:
        if abs(float(target) - float(needed)) > 1e-6:
            return None
    except (TypeError, ValueError):
        return None
    fix = conflict.get("parameter_to_correct")
    return _format_parameter(str(fix)) if fix else None


def reading_note(reading: dict, language: str, unit: str | None = None,
                 conflict: dict | None = None) -> str:
    """
    A reading's note, assembled by template from its own fields.

    The model plays no part: `status` picks the phrase, and the number
    accompanying it comes from `regulatory_limit`, never from
    `operating_target`. Mixing the two was one of the observed failures —
    "above the 120 ppm operating ceiling" presents an industry target as if it
    were a code ceiling.

    The target does appear, but behind and with its own label: it is the
    number the operator acts on, and without it the note says where the
    violation begins without saying where to leave the water. And if
    `conflict` declares it unreachable, it is marked as such: a blocked target
    printed bare invites exactly what the prose just forbade.
    """
    # `unit=None` means "derive it", not "no unit": a caller that forgets the
    # argument produces a half-finished line — the value with ppm and the
    # limit without — and that is precisely the defect this closes.
    if unit is None:
        unit = unit_for(str(reading.get("parameter", "")))

    status = coherent_status(reading)
    limit = reading.get("regulatory_limit")
    target = reading.get("operating_target")
    blocked_by = _target_blocked_by(reading, conflict)

    def _append_target(note: str, state: str) -> str:
        # Not given on `in_range`: there is nothing to correct, and a target
        # hanging off a healthy reading reads as a pending task that does not
        # exist.
        if target is None or state == "in_range":
            return note
        target_text = _with_unit(target, unit)
        if blocked_by:
            return note + _lang_table(_TARGET_BLOCKED_SUFFIX, language).format(
                target=target_text, fix=blocked_by)
        return note + _lang_table(_TARGET_SUFFIX, language).format(
            target=target_text)

    # Practice band: verdict without regulatory authority. `status` is read
    # from the original because `coherent_status` lets it through when
    # band_derived is set.
    if reading.get("band_derived"):
        band = _lang_table(_BAND_PHRASING, language)
        raw_status = reading.get("status")
        if raw_status in band:
            bound = reading.get("educational_bound")
            return _append_target(
                band[raw_status].format(
                    bound=_with_unit(bound, unit) if bound is not None else ""),
                raw_status,
            )

    phrasing = _lang_table(_STATUS_PHRASING, language)
    if status is None or status not in phrasing:
        if target is not None:
            return _lang_table(_REPORTED_WITH_TARGET, language).format(
                target=_with_unit(target, unit))
        return _lang_table(_NO_VERDICT, language)

    template = phrasing[status]
    if "{limit}" in template:
        if limit is None:
            # Should not happen — coherent_status already demotes those cases
            # — but a template with an unfilled hole is worse than a phrase
            # with no figure.
            return _lang_table(_NO_VERDICT, language)
        note = template.format(limit=_with_unit(limit, unit))
    else:
        note = template

    return _append_target(note, status)


def build_readings(test_interpretation: list[dict], language: str, line_cls,
                   conflict: dict | None = None):
    """
    Every panel line, built from the specialist's data.

    REPLACES whatever the synthesizer wrote in `readings`; does not complete
    it. An earlier version only added what was missing, so a badly worded line
    from the model passed through intact while the absent ones were corrected
    — half the problem fixed and half not.

    ALL reported parameters are included, in-range ones too: the operator
    handed over seven readings and seeing all seven is the confirmation that
    all seven were read. Omitting the correct ones forces inference by
    absence.

    `conflict` is the specialist payload's `constraint_conflict`. It is passed
    whole to `reading_note`, which decides which line — if any — carries its
    target marked unreachable.
    """
    lines = []
    for r in test_interpretation or []:
        if not isinstance(r, dict):
            continue
        name = _format_parameter(str(r.get("parameter", "")))
        measured = r.get("measured")
        if not name or measured is None:
            continue
        # The unit is resolved from the RAW name ("free_chlorine"), not the
        # formatted one, and applied to all three figures in the line: the
        # value, the limit and the target. Half a line with units is worse
        # than none.
        unit = unit_for(str(r.get("parameter", "")))
        lines.append(line_cls(
            parameter=name,
            measured=_with_unit(measured, unit),
            note=reading_note(r, language, unit, conflict),
        ))
    return lines


def panel_needs_footer(test_interpretation) -> bool:
    """
    Does any panel line rest on a practice band?

    If every reading has a regulatory bound, the provenance notice is
    redundant and only costs screen space.
    """
    return any(
        isinstance(r, dict) and r.get("band_derived")
        for r in (test_interpretation or [])
    )


def _infer_reading_cls(payload):
    """The ReadingLine class, from the model itself. None if absent."""
    existing = getattr(payload, "readings", None)
    if existing:
        return type(existing[0])
    try:  # pydantic v2
        return type(payload).model_fields["readings"].annotation.__args__[0]
    except Exception:
        return None


def enforce_visible_readings(payload, readings: list[dict], language: str,
                             report: ValidationReport,
                             conflict: dict | None = None) -> None:
    """
    Replace `readings` with the lines built from the data.

    It does not check what the model wrote nor complete what is missing: it
    replaces it. Four evaluation rounds on the same query showed that
    status-based phrasing does not hold as a prompt instruction, and the
    version that only completed let a badly worded line through intact while
    correcting the absent ones — half a solution.

    What the model still writes is the prose above: the verdict, the
    reasoning, the cause. The figure panel is assembled here.
    """
    line_cls = _infer_reading_cls(payload)
    if line_cls is None or not readings:
        return

    previous = {
        (getattr(r, "parameter", ""), getattr(r, "note", ""))
        for r in (getattr(payload, "readings", None) or [])
    }
    payload.readings = build_readings(readings, language, line_cls, conflict)

    current = {(r.parameter, r.note) for r in payload.readings}
    report.readings_appended = True
    # Telemetry: which lines did not match what the model had put there. If
    # this comes back full turn after turn, the prompt still is not achieving
    # it and the enforcement is covering for it.
    report.readings_missing = sorted(
        {p for p, _ in current - previous}
    ) if previous else [r.parameter for r in payload.readings]


# --------------------------------------------------------------------------
# Actions and safety from the specialist payload
# --------------------------------------------------------------------------
#
# Same principle as `readings`: the model writes the prose — the verdict, the
# reasoning, the cause — and the code assembles the structured fields. The
# difference from the reading panel is that these two CANNOT be built by
# template: `actions` is the specialist's text, and that text comes in
# English. So they are only substituted when the turn is answered in English;
# in Spanish the translation remains the model's.

#: Products that, if they CAUSED the problem, cannot be part of the fix.
#: Searched in `likely_cause`, not in the doses: dosing hypochlorite is
#: normal, having got here on trichlor is the finding.
_STABILIZED = ("trichlor", "dichlor", "stabilized chlorine",
               "chlorinated isocyanurate", "isocyanurate",
               "tricloro", "dicloro", "cloro estabilizado")

#: Commonly handled acids. They appear in `chemical_actions`, not in the
#: cause: what matters is that the operator will handle one.
_ACIDS = ("muriatic", "hydrochloric", "sodium bisulfate", "dry acid",
          "muriático", "muriatico", "clorhídrico", "clorhidrico",
          "bisulfato", "ácido seco", "acido seco")

#: The safety line, by template and by language. Like `_STATUS_PHRASING`: the
#: model does not write it.
#:
#: The `stabilized_at_ceiling` variant is only chosen when a reading backs it.
#: Asserting a limit the data does not support is exactly the failure
#: `coherent_status` exists to prevent, and it does not stop being one by
#: appearing in a warning.
#:
#: Cyanuric acid IS `_CODE_BACKED`, so "regulatory ceiling" can be used here
#: with propriety: the reading keeps its regulatory verdict and the warning
#: reintroduces nothing the panel withdrew.
_SAFETY_PHRASING = {
    "en": {
        "stabilized_at_ceiling": (
            "Do not use trichlor or dichlor — they add cyanuric acid, "
            "already at its regulatory ceiling."
        ),
        "stabilized": (
            "Do not use trichlor or dichlor — they add the cyanuric acid "
            "that produced this state."
        ),
        "acid": (
            "Never mix acid and chlorine products — add each separately "
            "with the pump running."
        ),
    },
    "es": {
        "stabilized_at_ceiling": (
            "No uses tricloro ni dicloro: aportan ácido cianúrico, que ya "
            "está en su techo normativo."
        ),
        "stabilized": (
            "No uses tricloro ni dicloro: aportan el ácido cianúrico que "
            "produjo este estado."
        ),
        "acid": (
            "Nunca mezcles ácido con productos clorados: agregá cada uno "
            "por separado y con la bomba en marcha."
        ),
    },
}

#: Names under which the specialist reports the stabilizer.
_CYA_NAMES = ("cyanuric", "cianúrico", "cianurico", "stabilizer")

#: The closure action, by template. The model does not write it and it does
#: not depend on the specialist remembering to include it.
_CLOSURE = {
    "en": "Close the pool to bathers immediately",
    "es": "Cerrá la pileta a los bañistas de inmediato",
}

#: Leading time clause: "After dilution,", "Once refilled,". Trimmed BEFORE
#: classifying, because it names a step that is NOT this item's. Without this,
#: "After dilution, retest all parameters" contains "dilution" and
#: `_CORRECTIVE_VERB_RE` treated it as corrective — the filter did nothing.
_LEADING_TIME_CLAUSE_RE = re.compile(
    r"^\s*(?:after|before|once|following|when|"
    r"luego\s+de|después\s+de|despues\s+de|antes\s+de|una\s+vez)\b"
    r"[^,]{0,40},\s*",
    re.IGNORECASE,
)

#: Same at the tail: "Retest pH 4 hours after adding acid" — the "adding"
#: belongs to the previous step, not to this one.
_TRAILING_TIME_CLAUSE_RE = re.compile(
    r"[,\s]+(?:after|before|once|following|"
    r"luego\s+de|después\s+de|despues\s+de|antes\s+de)\s+.*$",
    re.IGNORECASE,
)

#: Main verification verb.
_RETEST_RE = re.compile(
    r"^\s*(?:re-?test|verify|confirm|check|monitor|recheck|"
    r"retesteá|retestear|verificá|verificar|confirmá|confirmar|comprobá)\b",
    re.IGNORECASE,
)

#: Corrective verb. If it appears in the item's core, it is not verification
#: only.
_CORRECTIVE_VERB_RE = re.compile(
    r"\b(?:add|dose|dilut|drain|refill|lower|raise|chlorinat|shock|close|shut|"
    r"backwash|brush|vacuum|clean|replace|adjust|balance|"
    r"agreg|dosif|diluí|diluir|drená|drenar|bajá|bajar|subí|subir|"
    r"clorá|clorar|cerrá|cerrar|reemplaz|ajust)\w*\b",
    re.IGNORECASE,
)


#: Frontera de oración: punto o punto y coma, espacio, y mayúscula. No parte
#: "7.4-7.6" ni "aprox. 50%" porque exige el espacio y la mayúscula después.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+(?=[A-ZÁÉÍÓÚÑ])")


def _as_list(value) -> list[str]:
    """
    `recommendations` arrives as a list, as a numbered string, or as running
    prose. Normalize all three.

    The numbered split requires a period PLUS a space (`\\d+\\.\\s+`), so "7.4"
    or "3.4 ppm" do not break the sentence in half.

    THE PROSE FALLBACK is not cosmetic. In trace 57cabbd9 the specialist sent
    four actions as one unnumbered paragraph; with only the numbered split it
    came back as a single 43-word item, which exceeded MAX_ACTION_WORDS and
    was relocated whole. The operator saw a closed pool and no corrective
    step. The specialist has now alternated between list, numbered string and
    running prose across runs, so all three have to work.
    """
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if not isinstance(value, str) or not value.strip():
        return []

    text = value.strip()
    parts = [p.strip() for p in re.split(r"(?:^|\s)\d+\.\s+", text) if p.strip()]
    if len(parts) > 1:
        return parts

    # No numbering: fall back to sentence boundaries. A single sentence comes
    # back as a single item, which is the correct outcome.
    return [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]


def _is_verification_only(item: str) -> bool:
    """
    Is this item purely a verification instruction?

    Evaluated on the CORE — the item stripped of its time clauses — because
    those clauses name other steps and contaminate verb detection.

    Fails toward keeping: an ambiguous item stays in the list. Losing a real
    correction to a false positive is far worse than spending a slot on one
    retest too many.
    """
    core = _LEADING_TIME_CLAUSE_RE.sub("", item)
    core = _TRAILING_TIME_CLAUSE_RE.sub("", core)
    return bool(_RETEST_RE.match(core)) and not _CORRECTIVE_VERB_RE.search(core)


def render_actions(specialist: dict) -> list[str]:
    """
    Corrective actions from the specialist payload.

    Prefers `recommendations` (already imperative); falls back to the
    `chemical_actions` list, which wraps the action in an object.

    RETURNS EVERYTHING, WITH NO LENGTH FILTER. An earlier version discarded
    anything over MAX_ACTION_WORDS here, so those bullets never reached
    `payload.actions` and `normalize_actions` could not relocate them: they
    were lost silently, breaking the module's guiding principle in the first
    function that runs. That is how the dilution and the breakpoint
    chlorination — the turn's two central corrections — evaporated while the
    prose kept saying the pool had to be diluted. Trimming and relocation are
    `normalize_actions`' responsibility.

    THE ONLY THING DISCARDED HERE is pure verification items, and not for
    their shape but because they are DUPLICATE content: `retest_guidance` is
    its own payload field and has its own `details` section, so a "retest all
    parameters" inside `recommendations` is the third copy of the same text.
    The difference from the length filter is that that one lost information
    and this one does not: the data is still there, twice, where it belongs.

    It matters because there are four slots and the specialist does not
    prioritize them. In trace 2ad488c6, "After dilution, retest all
    parameters" took third place and pushed "Chlorinate to restore the
    disinfectant barrier" out of the visible tier — in a pool closed precisely
    for chlorine below minimum.
    """
    items = _as_list(specialist.get("recommendations"))
    if not items:
        items = [
            str(a.get("action", "")).strip()
            for a in specialist.get("chemical_actions") or []
            if isinstance(a, dict) and str(a.get("action", "")).strip()
        ]

    corrective = [i for i in items if not _is_verification_only(i)]
    # If EVERYTHING was verification, the list is returned intact: the turn
    # may legitimately be a follow-up — "I already dosed, what do I watch?" —
    # and leaving it with no actions would be worse than repeating the retest.
    return corrective or items


def _cya_at_ceiling(specialist: dict) -> bool:
    """Is there a stabilizer reading that backs the word 'ceiling'?"""
    for r in specialist.get("test_interpretation") or []:
        if not isinstance(r, dict):
            continue
        name = str(r.get("parameter", "")).lower()
        tokens = set(re.split(r"[^a-záéíóúñü]+", name))
        if any(n in name for n in _CYA_NAMES) or "cya" in tokens:
            if coherent_status(r) in ("at_ceiling", "above_maximum"):
                return True
    return False


def render_safety(specialist: dict, language: str = "es") -> str | None:
    """
    One imperative line, derived from the payload. Never duplicates an action.

    Priority: the product that caused the problem over the handling
    incompatibility. If the operator got here on trichlor, repeating trichlor
    is what puts them back in the same place; the other is handling hygiene,
    always true and therefore less informative.

    Neither can come out of an action in the list: one bans a product that
    will not be used and the other is about mixing order. By construction, not
    by check.

    Returns None when the payload supports neither. The caller then keeps
    whatever the model wrote: no generic warning is invented here, which would
    be noise on the most-read line of the visible tier.
    """
    phrases = _lang_table(_SAFETY_PHRASING, language)

    cause = str(specialist.get("likely_cause") or "").lower()
    if any(k in cause for k in _STABILIZED):
        key = "stabilized_at_ceiling" if _cya_at_ceiling(specialist) else "stabilized"
        return phrases[key]

    chems = " ".join(
        f"{a.get('chemical', '')} {a.get('action', '')}"
        for a in specialist.get("chemical_actions") or []
        if isinstance(a, dict)
    ).lower()
    if any(k in chems for k in _ACIDS):
        return phrases["acid"]

    return None



def enforce_closure_action(payload, readings, language: str,
                           report: ValidationReport) -> None:
    """
    If the panel forces a closure, the closure is `actions[0]`. Put by code.

    One round this was exactly what was missing: the closure action came from
    `recommendations`, so it depended on the specialist writing it — and the
    turn that needs it most is the one where the specialist gets it wrong. The
    trigger is the measured disinfectant value against the published minimum,
    via `closure_required`, not anyone's `status`.

    If the specialist ALREADY wrote it, it is not duplicated: it is reordered
    to the front. Two bullets saying the same thing spend the slot the next
    correction needs.
    """
    if not closure_required(readings):
        return

    line = _lang_table(_CLOSURE, language)
    tokens = _content_tokens(line)

    def _is_just_the_closure(action: str) -> bool:
        """
        Both directions, not one.

        The old check measured only how much of the closure appears in the
        action, so any long action that mentioned closing in passing scored
        0.75 and was dropped. In trace 57cabbd9 that removed a paragraph
        carrying the dilution, the acid and the chlorination because its first
        four words were "Close the pool immediately".

        Requiring the overlap in both directions means an item is only
        discarded when it is SUBSTANTIALLY the closure and nothing else.
        """
        other = _content_tokens(action)
        if not tokens or not other:
            return False
        overlap = len(tokens & other)
        return (overlap / len(tokens) >= _DUPLICATE_THRESHOLD
                and overlap / len(other) >= _DUPLICATE_THRESHOLD)

    rest = [a for a in (payload.actions or []) if not _is_just_the_closure(a)]
    payload.actions = [line] + rest
    report.closure_inserted = True
    report.notes.append("closure inserted by code: disinfectant below minimum")


def enforce_visible_tier(payload, specialist: dict, language: str,
                         report: ValidationReport) -> None:
    """
    Replace `actions` and `safety` with what can be derived from the payload.

    Each is replaced only if there is something to replace it with. What the
    model wrote is not relocated to `details`: that would be the same content
    twice, once above and once folded, and the source material already keeps
    it whole.

    `actions` is skipped when the turn is not answered in English. The
    specialist writes in English and its bullets would go in untranslated in a
    Spanish response — swapping a well-written action for the same action in
    another language is a regression, not enforcement. `safety` always
    applies: it comes from a template and the template exists in both.
    """
    if language == "en":
        actions = render_actions(specialist)
        if actions:
            report.actions_rendered = True
            dropped = [a for a in (payload.actions or []) if a not in actions]
            if dropped:
                report.notes.append(
                    f"{len(dropped)} model action(s) replaced by the specialist's"
                )
            payload.actions = actions

    line = render_safety(specialist, language)
    if line:
        report.safety_rendered = True
        payload.safety = line


# --------------------------------------------------------------------------
# Visible-tier quality telemetry (measures, does not correct)
# --------------------------------------------------------------------------

def safety_repeats_an_action(payload) -> bool:
    """
    Does `safety` say no more than an action already in the list?

    Measured: under an action "close the pool to bathers", the safety field
    said "keep the pool closed until chlorine is restored". The most-read line
    of the visible tier spent repeating the first action, and the one that did
    carry unique information — the ban on the product that caused the problem
    — gone.

    Measured, not corrected. Suppressing a safety line for resembling
    something else is a far worse failure than leaving it repeated, and
    lexical overlap cannot distinguish a repetition from a deliberate
    reinforcement. The prompt asks for the line with the longest reach; this
    says how often it does not deliver one.
    """
    safety = getattr(payload, "safety", None)
    actions = getattr(payload, "actions", None) or []
    if not safety or not actions:
        return False

    tokens = _content_tokens(safety)
    if not tokens:
        return False

    return any(
        len(tokens & _content_tokens(a)) / len(tokens) >= _DUPLICATE_THRESHOLD
        for a in actions
    )


#: Aliases an action might use to name a panel parameter. The key is the form
#: normalized by `_key()`.
_PARAM_ALIASES = {
    "free chlorine":     ("free chlorine", "fc", "chlorine residual",
                          "cloro libre"),
    "combined chlorine": ("combined chlorine", "chloramine", "chloramines",
                          "cloro combinado", "cloramina"),
    "ph":                ("ph",),
    "cyanuric acid":     ("cyanuric acid", "cya", "stabilizer", "stabiliser",
                          "ácido cianúrico", "acido cianurico", "estabilizante"),
    "total alkalinity":  ("total alkalinity", "alkalinity", "ta",
                          "alcalinidad total", "alcalinidad"),
    "calcium hardness":  ("calcium hardness", "hardness", "ch",
                          "dureza cálcica", "dureza"),
}

#: Verbs that ask to MOVE a parameter. Naming it without one of these is not a
#: contradiction: "retest alkalinity" does not argue with "in range".
_ADJUST_VERB_RE = re.compile(
    r"\b(?:lower|raise|reduce|increase|drop|bring\s+(?:up|down)|correct|adjust|"
    r"bajá|bajar|subí|subir|reducí|reducir|aumentá|aumentar|corregí|ajustá)\w*\b",
    re.IGNORECASE,
)


def _mentions_parameter(text: str, param_key: str) -> bool:
    """
    Does the action name this parameter? With word boundaries: "ph" cannot
    match inside "phosphate".
    """
    lowered = (text or "").lower()
    for alias in _PARAM_ALIASES.get(param_key, ()):
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", lowered):
            return True
    return False


def actions_contradict_panel(payload, readings) -> list[str]:
    """
    Actions asking to correct a parameter the panel declared in range.

    MEASURES, DOES NOT CORRECT. The recommendations are drafted by the
    specialist against ITS own `status`, and `reconcile_with_published_bands`
    corrects the reading afterwards: if the specialist sent total alkalinity
    as `above_maximum` with an invented limit of 120, the panel shows it "in
    range" and the action still says "lower total alkalinity". The operator
    receives both.

    The specialist's text is not edited. Rewriting a recommendation would
    require understanding which part of the sentence belongs to which
    parameter — "add acid to lower total alkalinity AND pH" is half correct —
    and a blind trim would break the action that is actually needed. What can
    be asserted unambiguously is that there is a discrepancy, and that is
    enough to measure frequency in Langfuse.

    The real fix does not live here: it is for the specialist to receive the
    reconciled readings BEFORE drafting. This counter says how urgent that
    graph change is.

    Requires the ALREADY reconciled readings: on the raw ones there would be
    no discrepancy to detect, because they are the source of the text.
    """
    findings = []
    in_range = {
        _key(str(r.get("parameter", "")))
        for r in (readings or [])
        if isinstance(r, dict) and coherent_status(r) == "in_range"
    }
    if not in_range:
        return findings

    for action in (getattr(payload, "actions", None) or []):
        if not _ADJUST_VERB_RE.search(action):
            continue
        for key in in_range:
            if _mentions_parameter(action, key):
                findings.append(
                    f"{_format_parameter(key)}: in range, but an action asks "
                    f"to correct it"
                )
    return sorted(set(findings))


# --------------------------------------------------------------------------
# Contract utilities
# --------------------------------------------------------------------------

def _has_hazard_agent(agents: Optional[list[str]]) -> bool:
    """
    Namespace tolerant. Accepts slug ("chemistry"), display name ("Pool
    Chemistry Agent") and any casing/separator, because the value arriving in
    state["assigned_agents"] is produced by the planner and is not normalized.
    A failure here is silent: intersection() with a display name is empty and
    the warning is simply not required.
    """
    for raw in agents or []:
        norm = re.sub(r"[^a-z0-9]+", "_", (raw or "").lower()).strip("_")
        if not norm:
            continue
        if norm in HAZARD_AGENTS:                              # exact slug
            return True
        if set(norm.split("_")) & HAZARD_AGENTS:               # display name
            return True
        if any(h in norm for h in HAZARD_AGENTS if "_" in h):  # compound slug
            return True
    return False


def _safety_trigger(contract: dict, agents: Optional[list[str]], payload) -> str:
    """
    Returns WHAT required safety: "contract" | "agent" | "lexicon" | "".
    Kept separate from the boolean so enforce_contract can record the reason
    without changing the public signature.
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


def resolve_safety_required(contract: dict, agents: Optional[list[str]],
                            payload) -> bool:
    """
    Boolean version of `_safety_trigger`, for external callers.

    `enforce_contract` uses the trigger directly because it needs the reason
    for telemetry; this is kept in case the synthesizer node or the tests
    import it.
    """
    return bool(_safety_trigger(contract, agents, payload))


def _visible_words(payload) -> int:
    """Counts tier 1: what the user sees without tapping anything."""
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
    """Add or extend a collapsible section, without duplicating labels."""
    existing = _get_detail(payload, label)
    if existing:
        existing.body = f"{existing.body}\n{body}".strip()
    else:
        payload.details.append(detail_cls(label=label, body=body))


def _infer_detail_cls(payload):
    """Recover the details item class from the Pydantic model."""
    if payload.details:
        return type(payload.details[0])
    try:  # pydantic v2
        return type(payload).model_fields["details"].annotation.__args__[0]
    except Exception:  # loose fallback
        from types import SimpleNamespace
        return lambda label, body: SimpleNamespace(label=label, body=body)


# --------------------------------------------------------------------------
# Action normalization
# --------------------------------------------------------------------------

def normalize_actions(payload, detail_cls, report: ValidationReport) -> None:
    """
    At most MAX_ACTIONS bullets. Overflow is NOT deleted: it goes to `details`.

    The specialist's ORDER is preserved: the first four stay visible, the rest
    move down. A bullet exceeding MAX_ACTION_WORDS moves down too, but that is
    a malformation and is noted separately — if it starts firing often, the
    specialist prompt is returning paragraphs where it asks for imperatives.

    This is the ONLY point in the pipeline where the action list is trimmed.
    `render_actions` no longer filters by length precisely so everything in
    excess passes through here and ends up in `details` instead of vanishing.
    """
    kept: list[str] = []
    relocated: list[str] = []
    malformed = 0

    for action in payload.actions or []:
        action = action.strip()
        if not action:
            continue
        if _words(action) > MAX_ACTION_WORDS:
            relocated.append(action)
            malformed += 1
        elif len(kept) >= MAX_ACTIONS:
            relocated.append(action)
        else:
            kept.append(action)

    payload.actions = kept

    if relocated:
        body = "\n".join(f"- {a}" for a in relocated)
        _append_detail(payload, OVERFLOW_LABEL, body, detail_cls)
        report.actions_relocated += len(relocated)
        report.notes.append(f"{len(relocated)} action(s) relocated to details")
    if malformed:
        report.notes.append(
            f"{malformed} action(s) exceeded {MAX_ACTION_WORDS} words"
        )


# --------------------------------------------------------------------------
# Budget overflow
# --------------------------------------------------------------------------

def overflow_to_details(payload, budget: int, detail_cls,
                        report: ValidationReport) -> None:
    """
    Move actions down from the end until the budget is met.

    `answer` and `safety` are untouchable: if they exceed the budget on their
    own, `answer_exceeds_budget` is set and the caller decides (retry or
    accept). Truncating a sentence mid-way is worse than running long.
    """
    if budget >= NO_CAP:
        return

    floor = _words(payload.answer) + _words(payload.safety)
    if floor > budget:
        report.answer_exceeds_budget = True
        report.notes.append(
            f"answer+safety={floor}w exceeds the {budget}w budget on their own"
        )
        # Cannot be fixed by relocating; let it through and the caller decides.
        return

    relocated: list[str] = []
    while payload.actions and _visible_words(payload) > budget:
        relocated.insert(0, payload.actions.pop())

    if relocated:
        body = "\n".join(f"- {a}" for a in relocated)
        _append_detail(payload, OVERFLOW_LABEL, body, detail_cls)
        report.overflowed = True
        report.actions_relocated += len(relocated)
        report.notes.append(f"{len(relocated)} action(s) relocated for budget")


# --------------------------------------------------------------------------
# Safety promotion
# --------------------------------------------------------------------------

def promote_safety_from_details(payload, report: ValidationReport) -> None:
    """
    Hard rule: nothing safety-related stays behind the fold.

    If the contract requires `safety` and the LLM omitted it, a folded section
    that looks like a warning is located and its first sentence is promoted to
    tier 1. If there is nothing to promote, `safety_missing` is set — which no
    longer triggers a retry (see ValidationReport.needs_retry), only measures.
    """
    if payload.safety and payload.safety.strip():
        return

    for detail in list(payload.details):
        if SAFETY_LABEL_PATTERN.search(detail.label) or \
           SAFETY_LABEL_PATTERN.search(detail.body):
            payload.safety = _first_sentence(detail.body)
            payload.details.remove(detail)
            report.safety_promoted = True
            report.notes.append(f"safety promoted from details: '{detail.label}'")
            return

    report.safety_missing = True
    report.notes.append("contract requires safety and there is nothing to promote")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def enforce_contract(payload, contract: dict, agents: list[str] | None = None,
                     detail_cls=None, readings: list[dict] | None = None,
                     language: str = "es",
                     raw_content: str = "",
                     specialist: dict | None = None) -> tuple[Any, ValidationReport]:
    """
    Apply the contract to the synthesizer payload.

    Args:
        payload:    SynthesizerOutput instance (mutated in place).
        contract:   ARCHETYPE_CONTRACTS entry.
        agents:     state["assigned_agents"], to resolve conditional safety.
        detail_cls: details item class. If None it is inferred from the
                    payload or falls back to a compatible dict-like.
        readings:   the specialist's `test_interpretation`, raw.
        specialist: the sub-agents' merged JSON payload. `actions`, `safety`
                    and `constraint_conflict` come from here.

    Returns:
        (payload, report). If `report.needs_retry` is True the caller may
        retry ONCE with a corrective instruction; otherwise it accepts the
        deterministic degradation already applied.

        There is no `report.panel_footer`: the panel foot is decided with
        `panel_needs_footer(reconciled_readings)` and rendered in the node,
        not here — this module does not compose the final surface.
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

    # 0. Deterministic visible tier: `actions` and `safety` from the
    #    specialist payload. Runs BEFORE normalization so the caps and the
    #    budget apply to the final text, not to what gets discarded.
    if specialist:
        enforce_visible_tier(payload, specialist, language, report)

    # 0b. The closure, if the disinfectant forces one. Runs over the RAW
    #     readings — `closure_required` derives from the measured value, not
    #     from anyone's status — and before normalization, so it takes a
    #     bullet slot like any other action instead of sneaking past the cap.
    if readings:
        enforce_closure_action(payload, readings, language, report)

    # 1. Normalize bullets before measuring the budget. Only trimming point:
    #    what exceeds is relocated to `details`, never lost.
    normalize_actions(payload, detail_cls, report)

    # 2. Safety: promote BEFORE the overflow, because it counts as visible.
    trigger = _safety_trigger(contract, agents or [], payload)
    if trigger:
        report.notes.append(f"safety required by: {trigger}")
        promote_safety_from_details(payload, report)

    # 3. Readings BEFORE the budget: anything added has to compete for space
    #    like the rest of the visible tier, not sneak past the cap.
    #
    #    Reconciliation against the published bands goes first: the panel is
    #    built on the already-degraded readings, never on the raw ones. The
    #    other way round, the line would assert a nonexistent code and the
    #    degradation would arrive too late.
    if readings:
        readings = reconcile_with_published_bands(readings, report)
        enforce_visible_readings(
            payload, readings, language, report,
            conflict=(specialist or {}).get("constraint_conflict"),
        )

    # 4. Budget.
    overflow_to_details(payload, report.budget, detail_cls, report)

    # 5. Prune empty sections.
    payload.details = [d for d in payload.details if d.body and d.body.strip()]

    # 6. Visible-tier quality telemetry. Corrects nothing: measures how often
    #    the prompt does not achieve what it asks for.
    report.safety_duplicates_action = safety_repeats_an_action(payload)
    if report.safety_duplicates_action:
        report.notes.append("safety repeats an action instead of adding something new")

    if readings:
        report.actions_vs_panel = actions_contradict_panel(payload, readings)
        if report.actions_vs_panel:
            report.notes.append(
                f"actions arguing with the panel: {report.actions_vs_panel}"
            )

    # 7. Unbacked quantities. Last, over the final text, so nothing the
    #    validator itself added escapes the check.
    if raw_content:
        report.unsupported_numbers = unsupported_numbers(payload, raw_content)
        if report.unsupported_numbers:
            report.notes.append(
                f"unbacked quantities in the visible tier: {report.unsupported_numbers}"
            )

    report.visible_words_after = _visible_words(payload)
    return payload, report


# --------------------------------------------------------------------------
# Fallback
# --------------------------------------------------------------------------

def fallback_payload(raw_text: str, output_cls):
    """
    If structured output fails, the front end never sees a different format.
    All the raw text goes into `answer`, unfolded and unvalidated.
    """
    return output_cls(answer=raw_text, actions=[], safety=None, details=[])