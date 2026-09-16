"""
response_validator.py

Deterministic enforcement of the response contract emitted by the synthesizer.

The prompt ASKS the LLM to respect the budget; this module GUARANTEES it.
Guiding principle: never delete information, only relocate it to `details`.

    payload, report = enforce_contract(payload, contract, agents, vessel=vessel)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional
import logging


from .water_targets import (
    closure_required,
    is_published_figure,
    resolve_panel,
    target_band,
    VesselContext,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MAX_ACTIONS = 4

MAX_ACTION_WORDS = 40

NO_CAP = 9999  # sentinel budget: `critical` archetype, no ceiling

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
#: Hung off every verdict, it contradicted the prose: "close the pool, free
#: chlorine is below the required minimum" over a line saying that minimum was
#: not a code limit. That is a perfect excuse not to close.
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
    remediation_target_inserted: bool = False
    budget: int = 0
    overflowed: bool = False
    actions_relocated: int = 0
    safety_promoted: bool = False
    #: Contract required `safety` and there was nothing to put there. No longer
    #: triggers a retry: the line is now built by template from the specialist
    #: payload. Measured only.
    safety_missing: bool = False
    answer_exceeds_budget: bool = False  # -> triggers retry
    #: `actions` / `safety` built by code from the specialist payload rather
    #: than accepted as the model wrote them.
    actions_rendered: bool = False
    safety_rendered: bool = False
    #: `regulatory_limit` values that turned out to be relabelled practice
    #: bands. While this stays full, the specialist keeps presenting targets as
    #: code and all that changed is that we detect it.
    limits_demoted: list[str] = field(default_factory=list)
    #: Parameters that DO have a regulatory bound and kept the code verdict.
    #: If this empties out, `_CODE_BACKED` broke.
    limits_kept: list[str] = field(default_factory=list)
    #: `operating_target` filled in from the band because the specialist left
    #: it empty.
    targets_filled: list[str] = field(default_factory=list)
    #: `operating_target` REPLACED or withdrawn: zero, an echo of the measured
    #: value, outside the band, or a parameter with no publishable target.
    targets_replaced: list[str] = field(default_factory=list)
    #: Readings whose `status` disagreed with the band-derived one, as
    #: "pH: above_maximum→in_range". Direct measure of the oscillation.
    status_corrected: list[str] = field(default_factory=list)
    #: The panel forces a closure and the action was inserted by code.
    closure_inserted: bool = False
    #: Out-of-range readings the visible tier omitted.
    readings_missing: list[str] = field(default_factory=list)
    readings_appended: bool = False
    #: Quantities in the visible tier absent from the source material. The
    #: worst failure mode per the synthesizer's own prompt.
    unsupported_numbers: list[str] = field(default_factory=list)
    #: `safety` repeats an action. Measured, not corrected: dropping a safety
    #: line is worse than repeating one.
    safety_duplicates_action: bool = False
    #: Actions asking to correct a parameter the panel reported as in range.
    #: Measured only: the fix belongs in the specialist, not here.
    actions_vs_panel: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def needs_retry(self) -> bool:
        """
        One retry only. `safety_missing` was removed from here: it fired seven
        rounds in a row at 2-3.5 s each, for a line now built by template.
        """
        return self.answer_exceeds_budget

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Phrasing tables — the model writes none of this
# --------------------------------------------------------------------------


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
#: authority it does not have. The disclaimer lives once in `PANEL_FOOTER`.
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

#: For a reading whose status does not hold: the value and, if there is one,
#: the target — with no compliance language in either direction.
_NO_VERDICT = {
    "en": "reported; no code bound available",
    "es": "reportado; sin límite normativo disponible",
}

_REMEDIATION = {
    "en": {
        "with_time": "Raise {param} to {conc} and hold for {time}.",
        "no_time":   "Raise {param} to {conc}.",
    },
    "es": {
        "with_time": "Elevar {param} a {conc} y mantener durante {time}.",
        "no_time":   "Elevar {param} a {conc}.",
    },
}

_REPORTED_WITH_TARGET = {
    "en": "reported; operating target {target}",
    "es": "reportado; objetivo operativo {target}",
}

#: The limit says where the violation begins; the target says where to leave
#: the water. Goes as a suffix with its own word so it is not read as the
#: regulatory figure — one observed failure was exactly that.
_TARGET_SUFFIX = {
    "en": "; target {target}",
    "es": "; objetivo {target}",
}

#: The target when `constraint_conflict` says it CANNOT be reached. The panel
#: was printing "target 6.75 ppm" while the same turn said that level exceeds
#: the permitted maximum: unqualified, a target reads as achievable.
_TARGET_BLOCKED_SUFFIX = {
    "en": "; target {target} — blocked until {fix} comes down",
    "es": "; objetivo {target} — bloqueado hasta bajar {fix}",
}

#: Parameters with a REAL regulatory bound, not a practice band. Not demoted:
#: they keep "in violation, below the 2 ppm minimum".
#:
#: That the corpus lacks the ingested citation is a retrieval problem, not a
#: reason to tell the operator the limit does not exist. And the cost
#: asymmetry runs the other way than for pH: failing to assert the free
#: chlorine minimum puts "not a code limit" under the figure that triggers a
#: mandatory closure.
#:
#: NOTE — the free chlorine minimum depends on stabilizer AND on vessel type
#: (3 ppm in spas). `water_targets` encodes both; this set only decides
#: whether the verdict keeps regulatory language.
_CODE_BACKED = frozenset({
    "free chlorine", "fc",
    "combined chlorine", "cc",
    "cyanuric acid", "cya",
})

#: Unit per parameter. The specialist emits `measured` as a bare number, so the
#: unit is lost between its payload and the screen: in trace 148acb15 all seven
#: values arrived without ppm. 0.8 without ppm tells nobody the pool is in
#: violation.
#:
#: Temperature is NOT here, on purpose: the specialist returns 82 with no
#: scale, and 82 °F and 82 °C describe two unrelated pools.
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

#: Spellings that do not survive a .title(). "ph" -> "Ph" is a name no operator
#: writes.
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
    The language sub-table, falling back to English. `or` rather than `.get`:
    a later `"fr": {}` would otherwise return the empty dict.
    """
    return mapping.get(language) or mapping["en"]


def _format_number(value) -> str:
    """No padding zeros: 2.0 -> '2'. A string is returned untouched — if the
    specialist wrote "0.8 ppm", that unit is theirs."""
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

#: Minimal stopwords, es/en. All that matters is that "the/de/la/to" do not
#: inflate the similarity between short phrases.
_STOPWORDS = frozenset({
    "el", "la", "los", "las", "un", "una", "de", "del", "a", "al", "en", "y",
    "o", "que", "no", "se", "su", "hasta", "para", "con", "por",
    "the", "an", "of", "to", "in", "and", "or", "not", "your", "until",
    "is", "are", "be", "all",
})
_WORD_RE = re.compile(r"[a-záéíóúñü]+", re.IGNORECASE)
_DUPLICATE_THRESHOLD = 0.55
#: Stem truncation, not lemmatization. "cerrada"/"cerrar" scored 0.5 and 0.4
#: literally on what is obviously a repetition. Five chars absorbs inflection
#: in both languages without conflating distinct words.
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
    Quantities in the visible tier absent from the source material.

    With `calculation_request` unexecuted, the synthesizer wrote "drain and
    refill thirty to forty percent" — nowhere in the payload, and wrong. An
    invented number arrives with the same confidence as the real ones.

    Detects, does not correct: rewriting the sentence would require
    understanding it.
    """
    if not raw_content:
        return []

    def _canon(x: str) -> str:
        """3.0 and 3 are the same number; 90 and 9 are not. A bare rstrip("0")
        turned "90" into "9" and reported a real figure as invented."""
        x = x.replace(",", ".")
        return x.rstrip("0").rstrip(".") if "." in x else x

    source_numbers = {_canon(o) for o in _NUMBER_RE.findall(raw_content)}

    # Only what the MODEL writes. `readings` is built by this module from the
    # specialist payload, so its figures come from the source by definition.
    model_written = " ".join([
        getattr(payload, "answer", "") or "",
        getattr(payload, "safety", "") or "",
        *(getattr(payload, "actions", None) or []),
    ])

    # Numbers spelled as words escape this check: a conscious limitation, to
    # avoid false positives from a numeral parser in two languages.
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

    `above_maximum` with `regulatory_limit` null asserts a maximum the entry
    says it does not know. Observed: total alkalinity judged above_maximum
    against its operating_target while practically on target — an operator
    repeating that to an inspector reports a violation that does not exist.

    Demoted to None, not "in_range": we do not know it is in range, we know we
    cannot assert the opposite. Readings reconciled against a band
    (`band_derived`) are exempt — their verdict rests on `educational_bound`.
    """
    status = reading.get("status")
    if reading.get("band_derived"):
        return status
    if status in _VIOLATION_STATUSES and reading.get("regulatory_limit") is None:
        return None
    return status


def _is_published_bound(name: str, limit,
                        vessel: "VesselContext | None" = None) -> bool:
    """
    Is the specialist's `regulatory_limit` a relabelled published band rather
    than a real citation? Only non-`_CODE_BACKED` parameters reach here.

    The `except TypeError` assumes yes (the module's previous behaviour). It
    also masks a signature mismatch, so verify `is_published_figure` accepts
    `vessel` before trusting a run where everything came back demoted.
    """
    if limit is None:
        return False
    try:
        return bool(is_published_figure(name, limit, vessel))
    except TypeError:
        return True


def _target_is_plausible(name: str, target, measured,
                         vessel: "VesselContext | None" = None) -> bool:
    """
    Does the specialist's `operating_target` work as a target?

    Three rejections, all observed: ZERO (combined chlorine came back 0.0 in
    six runs — zero chloramines is not an operable state), ECHO of the measured
    value (temperature 82.0 over 82.0), and OUTSIDE THE BAND (a target aimed at
    the violation).

    The third runs `resolve_panel` on a synthetic item rather than reading the
    band's endpoints: the internal shape of `target_band` is not this module's
    contract. That item carries no cyanuric acid, so the stabilized floor never
    applies here.
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
        status, _ = resolve_panel(
            [{"parameter": name, "measured": value}], vessel)[0]
    except Exception:
        return True
    return status in ("in_range", "at_floor", "at_ceiling")


def reconcile_with_published_bands(
        test_interpretation, report=None,
        vessel: "VesselContext | None" = None) -> list[dict]:
    """
    Rewrite each reading's verdict from the published band for THIS vessel.

    `vessel` is load-bearing, not decoration. In trace 35b8a774 an indoor spa
    resolved against pool bands: the specialist had correctly called free
    chlorine 2.2 below_minimum and cyanuric acid 35 above_maximum, this
    function demoted both to in_range, and the panel printed a clearance under
    an order to close.
    """
    if not isinstance(test_interpretation, list):
        return []

    resolved = resolve_panel(test_interpretation, vessel)
    output = []
    for r, (status, bound) in zip(test_interpretation, resolved):
        if not isinstance(r, dict):
            continue
        r = dict(r)
        name = str(r.get("parameter", ""))
        band = target_band(name, vessel)

        if band is not None:
            if status is not None:
                if report is not None and status != r.get("status"):
                    report.status_corrected.append(
                        f"{_format_parameter(name)}: {r.get('status')}→{status}"
                    )
                r["status"] = status

            if is_code_backed(name):
                # Regulatory bound: the code verdict is kept. If the specialist
                # did not supply the limit, the band's is used.
                if r.get("regulatory_limit") is None and bound is not None:
                    r["regulatory_limit"] = bound
                r["band_derived"] = False
                if report is not None:
                    report.limits_kept.append(_format_parameter(name))
            elif (_is_published_bound(name, r.get("regulatory_limit"), vessel)
                  or r.get("regulatory_limit") is None):
                # Practice band: regulatory authority withdrawn, verdict moves
                # to the "typical"/"habitual" phrasing.
                if r.get("regulatory_limit") is not None and report is not None:
                    report.limits_demoted.append(_format_parameter(name))
                r["regulatory_limit"] = None
                if status is not None:
                    r["band_derived"] = True
                    r["educational_bound"] = bound
            else:
                # A limit that does NOT match the band: may be a real
                # jurisdictional citation. Respected as it came.
                if report is not None:
                    report.limits_kept.append(_format_parameter(name))
                r["band_derived"] = False

            current_target = r.get("operating_target")
            if not band.fill_target:
                # The corpus says this parameter has no publishable target.
                # The flag was only consulted in `target_text()` — the
                # REPLACEMENT path — so a plausible target from the specialist
                # printed anyway: free chlorine with CYA 90 showed "target
                # 2 ppm" while the same turn explained 6.75 were needed.
                if current_target is not None:
                    r["operating_target"] = None
                    if report is not None:
                        report.targets_replaced.append(_format_parameter(name))
            elif not _target_is_plausible(
                    name, current_target, r.get("measured"), vessel):
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
    The readings the visible tier cannot omit. No longer used by
    `enforce_contract` — `enforce_visible_readings` rebuilds the whole panel —
    but kept as public surface.
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
    declares unreachable.

    Matched by VALUE, not by name: `parameter_to_correct` names the parameter
    to MOVE (cyanuric acid), not the one left blocked (free chlorine), and the
    payload has no field for the latter.
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
    A reading's note, assembled by template. The model plays no part.

    `status` picks the phrase; the number comes from `regulatory_limit`, never
    from `operating_target` — mixing them produced "above the 120 ppm operating
    ceiling", an industry target dressed as a code ceiling. The target does
    appear, behind and labelled, because without it the note says where the
    violation begins but not where to leave the water.
    """
    # `unit=None` means "derive it", not "no unit": a caller that forgets the
    # argument produces a half-finished line.
    if unit is None:
        unit = unit_for(str(reading.get("parameter", "")))

    status = coherent_status(reading)
    limit = reading.get("regulatory_limit")
    target = reading.get("operating_target")
    blocked_by = _target_blocked_by(reading, conflict)

    def _append_target(note: str, state: str) -> str:
        # Not on `in_range`: a target hanging off a healthy reading reads as a
        # pending task that does not exist.
        if target is None or state == "in_range":
            return note
        target_text = _with_unit(target, unit)
        if blocked_by:
            return note + _lang_table(_TARGET_BLOCKED_SUFFIX, language).format(
                target=target_text, fix=blocked_by)
        return note + _lang_table(_TARGET_SUFFIX, language).format(
            target=target_text)

    # Practice band: verdict without regulatory authority.
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
            # Should not happen — `coherent_status` demotes those — but an
            # unfilled hole is worse than a phrase with no figure.
            return _lang_table(_NO_VERDICT, language)
        note = template.format(limit=_with_unit(limit, unit))
    else:
        note = template

    return _append_target(note, status)


def build_readings(test_interpretation: list[dict], language: str, line_cls,
                   conflict: dict | None = None):
    """
    Every panel line, built from the specialist's data. REPLACES `readings`,
    does not complete it: an earlier version let a badly worded model line
    through intact while correcting the absent ones.

    ALL reported parameters are included, in-range ones too: the operator
    handed over seven readings and seeing all seven confirms all seven were
    read.
    """
    lines = []
    for r in test_interpretation or []:
        if not isinstance(r, dict):
            continue
        name = _format_parameter(str(r.get("parameter", "")))
        measured = r.get("measured")
        if not name or measured is None:
            continue
        # Unit resolved from the RAW name and applied to all three figures:
        # half a line with units is worse than none.
        unit = unit_for(str(r.get("parameter", "")))
        lines.append(line_cls(
            parameter=name,
            measured=_with_unit(measured, unit),
            note=reading_note(r, language, unit, conflict),
        ))
    return lines


def panel_needs_footer(test_interpretation) -> bool:
    """Does any panel line rest on a practice band? If none does, the
    provenance notice only costs screen space."""
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

    Four evaluation rounds showed status-based phrasing does not hold as a
    prompt instruction. The model still writes the prose above — the verdict,
    the reasoning, the cause; the figure panel is assembled here.
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
    # Telemetry: which lines did not match what the model put there. Full turn
    # after turn means the prompt still is not achieving it.
    report.readings_missing = sorted(
        {p for p, _ in current - previous}
    ) if previous else [r.parameter for r in payload.readings]


# --------------------------------------------------------------------------
# Actions and safety from the specialist payload
# --------------------------------------------------------------------------
#
# Same principle as `readings`: the model writes the prose, the code assembles
# the structured fields. The difference is that these two cannot be built by
# template — `actions` is the specialist's text, and that text comes in
# English. So they are only substituted when the turn is answered in English.

#: Products that, if they CAUSED the problem, cannot be part of the fix.
#: Searched in `likely_cause`, not in the doses.
_STABILIZED = ("trichlor", "dichlor", "stabilized chlorine",
               "chlorinated isocyanurate", "isocyanurate",
               "tricloro", "dicloro", "cloro estabilizado")

#: Commonly handled acids. Searched in `chemical_actions`: what matters is that
#: the operator will handle one.
_ACIDS = ("muriatic", "hydrochloric", "sodium bisulfate", "dry acid",
          "muriático", "muriatico", "clorhídrico", "clorhidrico",
          "bisulfato", "ácido seco", "acido seco")

#: The safety line, by template. The `stabilized_at_ceiling` variant is only
#: chosen when a reading backs it: asserting an unsupported limit does not stop
#: being a failure by appearing in a warning.
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

#: The closure action, by template. Does not depend on the specialist
#: remembering to include it.
_CLOSURE = {
    "en": {
        "pool": "Close the pool to bathers immediately",
        "spa":  "Close the spa to bathers immediately",
    }
}

_VESSEL_NOUNS = frozenset({"pool", "spa", "pilet", "vesse", "tub"})


_LEADING_TIME_CLAUSE_RE = re.compile(
    r"^\s*(?:after|before|once|following|when|"
    r"luego\s+de|después\s+de|despues\s+de|antes\s+de|una\s+vez)\b"
    r"[^,]{0,40},\s*",
    re.IGNORECASE,
)

#: Same at the tail: "Retest pH 4 hours after adding acid".
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

#: The same set WITHOUT the closure verbs. An item carrying one of these
#: orders something besides closing, so it is not a duplicate of the closure
#: line no matter how much it overlaps.
_NON_CLOSURE_VERB_RE = re.compile(
    r"\b(?:add|dose|dilut|drain|refill|lower|raise|chlorinat|shock|"
    r"backwash|brush|vacuum|clean|replace|adjust|balance|"
    r"agreg|dosif|diluí|diluir|drená|drenar|bajá|bajar|subí|subir|"
    r"clorá|clorar|reemplaz|ajust)\w*\b",
    re.IGNORECASE,
)

#: Sentence boundary: period or semicolon, space, uppercase. Does not split
#: "7.4-7.6" because it requires the space and the capital.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;])\s+(?=[A-ZÁÉÍÓÚÑ])")


def _as_list(value) -> list[str]:
    """
    `recommendations` arrives as a list, as a numbered string, or as running
    prose. All three have been observed across runs, so all three must work.

    The numbered split requires a period PLUS a space, so "7.4" does not break
    the sentence in half. The PROSE fallback is not cosmetic: in trace 57cabbd9
    four actions came as one unnumbered paragraph, exceeded MAX_ACTION_WORDS
    and were relocated whole — the operator saw a closed pool and no corrective
    step.
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
    # back as a single item, which is correct.
    return [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]


def _is_verification_only(item: str) -> bool:
    """
    Is this item purely a verification instruction? Evaluated on the CORE — the
    item stripped of its time clauses, which name other steps.

    Fails toward keeping: losing a real correction to a false positive is far
    worse than one retest too many.
    """
    core = _LEADING_TIME_CLAUSE_RE.sub("", item)
    core = _TRAILING_TIME_CLAUSE_RE.sub("", core)
    return bool(_RETEST_RE.match(core)) and not _CORRECTIVE_VERB_RE.search(core)


def render_actions(specialist: dict) -> list[str]:
    """The specialist's actions, verification-only items dropped. If
    EVERYTHING was verification the list is returned intact: the turn may be a
    follow-up, and no actions would be worse."""
    items = _as_list(specialist.get("recommendations"))
    if not items:
        items = [
            str(a.get("action", "")).strip()
            for a in specialist.get("chemical_actions") or []
            if isinstance(a, dict) and str(a.get("action", "")).strip()
        ]

    corrective = [i for i in items if not _is_verification_only(i)]
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
    """The safety line by template, or None when the payload supports no
    specific warning."""
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
                           report: ValidationReport,
                           vessel: "VesselContext | None" = None) -> None:
    
    if not closure_required(readings, vessel):
        return

    kind = (vessel.kind if vessel and vessel.kind else "pool")
    line = _lang_table(_CLOSURE, language).get(kind, _lang_table(_CLOSURE, language)["pool"])
    tokens = _content_tokens(line) - _VESSEL_NOUNS

    def _is_just_the_closure(action: str) -> bool:
        """
        An item is the closure repeated if it CONTAINS the closure and orders
        no other correction.

        Bidirectional overlap could not separate the two cases: the 57cabbd9
        paragraph was long because it carried dilution, acid and chlorination;
        the 10b378b9 line was long because it carried a justification. The
        second passed, duplicated the closure and pushed dilution out of the
        visible tier. What distinguishes them is a corrective verb other than
        closing, not word count.
        """
        other = _content_tokens(action) - _VESSEL_NOUNS
        if not tokens or not other:
            return False
        has_closure = len(tokens & other) / len(tokens) >= _DUPLICATE_THRESHOLD
        return has_closure and not _NON_CLOSURE_VERB_RE.search(action)

    rest = [a for a in (payload.actions or []) if not _is_just_the_closure(a)]
    payload.actions = [line] + rest
    report.closure_inserted = True
    report.notes.append("closure inserted by code: disinfectant below minimum")


def _fmt_target_value(value, unit_hint: str = "ppm") -> str | None:
    """
    `concentration` llega como 20.0 en una corrida y como
    "20 ppm (after lowering CYA to <= 15 ppm)" en otra. Deriva del esquema,
    no de un caso raro: el contrato declara el campo sin tipo ni unidad.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return f"{_format_number(value)} {unit_hint}"
    text = str(value).strip()
    return text or None


def _fmt_contact_time(value) -> str | None:
    """
    Numérico se interpreta en MINUTOS: es lo que emitió el especialista
    (1680.0 = 28 h) y lo que usa el catálogo de CT en ppm-min. Un valor
    menor a 60 se lee como horas, porque ningún hold de desinfección dura
    menos de una hora y "45 minutos" casi siempre sería "45 horas" mal
    tipado.

    Heurística, no contrato. El arreglo de fondo es declarar la unidad en
    el campo; mientras no esté, esto evita mostrar "28 minutos".
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        minutes = float(value)
        if minutes < 60:
            return f"{_format_number(minutes)} hours"
        hours = minutes / 60
        if abs(hours - round(hours)) < 0.01:
            return f"{int(round(hours))} hours"
        return f"{_format_number(round(hours, 1))} hours"
    text = str(value).strip()
    return text or None


def render_remediation_target(specialist: dict, language: str) -> str | None:
    """La línea del objetivo de remediación, o None si no hay uno utilizable."""
    target = specialist.get("remediation_target")
    if not isinstance(target, dict):
        return None

    conc = _fmt_target_value(target.get("concentration"))
    if not conc:
        return None

    param = _format_parameter(str(target.get("parameter") or "free chlorine"))
    time = _fmt_contact_time(target.get("contact_time"))
    phrases = _lang_table(_REMEDIATION, language)

    if time:
        return phrases["with_time"].format(param=param, conc=conc, time=time)
    return phrases["no_time"].format(param=param, conc=conc)


def enforce_remediation_target(payload, specialist: dict, language: str,
                               report: ValidationReport) -> None:
    """
    Garantiza que el objetivo de remediación esté en el tier visible.

    Va en la posición 1, detrás del cierre: `normalize_actions` recorta por
    el final, así que una acción al principio no se puede relocalizar a
    `details`. Un turno crítico no puede decirle al operador que
    hipercloriné sin decirle a cuánto.
    """
    logger.warning(
        "REMED: llamada | tipo=%s | valor=%r | acciones=%d",
        type(specialist.get("remediation_target")).__name__,
        specialist.get("remediation_target"),
        len(payload.actions or []),
    )
    line = render_remediation_target(specialist, language)
    if not line:
        return

    target = specialist.get("remediation_target") or {}
    conc = _fmt_target_value(target.get("concentration"))
    numero = re.search(r"\d+(?:[.,]\d+)?", conc or "")
    patron = (
        re.compile(rf"\b{re.escape(numero.group(0))}\s*ppm\b", re.I)
        if numero else None
    )

    acciones = list(payload.actions or [])

    # El dedupe corre contra lo que va a SOBREVIVIR, no contra la lista
    # entera. En el trace 5608d16d la recomendación del especialista traía
    # el target, el dedupe la dio por cubierta, y normalize_actions la
    # relocalizó a `details` por venir séptima de trece: el turno salió sin
    # target visible justo cuando esta función existía para evitarlo.
    supervivientes = acciones[:MAX_ACTIONS]

    if patron:
        for i, accion in enumerate(supervivientes):
            if patron.search(accion):
                # Ya está y va a sobrevivir: nada que insertar.
                return
        # Está, pero fuera del corte: la subimos en vez de duplicarla.
        for i, accion in enumerate(acciones[MAX_ACTIONS:], start=MAX_ACTIONS):
            if patron.search(accion):
                acciones.pop(i)
                line = accion          # el texto del especialista gana
                break

    posicion = min(len(acciones), MAX_ACTIONS - 1)
    payload.actions = acciones[:posicion] + [line] + acciones[posicion:]
    report.remediation_target_inserted = True
    report.notes.append("remediation target inserted by code")

def enforce_visible_tier(payload, specialist: dict, language: str,
                         report: ValidationReport) -> None:
    """`actions` and `safety` from the specialist payload. Actions only in
    English — the specialist's text is English and translating it is the
    model's job."""
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
    """Does `safety` restate an action instead of adding something new?"""
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


#: Aliases an action might use to name a panel parameter. Keys are `_key()`
#: normalized.
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

#: Verbs that ask to MOVE a parameter. Naming it without one is not a
#: contradiction: "retest alkalinity" does not argue with "in range".
_ADJUST_VERB_RE = re.compile(
    r"\b(?:lower|raise|reduce|increase|drop|bring\s+(?:up|down)|correct|adjust|"
    r"bajá|bajar|subí|subir|reducí|reducir|aumentá|aumentar|corregí|ajustá)\w*\b",
    re.IGNORECASE,
)


def _mentions_parameter(text: str, param_key: str) -> bool:
    """Does the action name this parameter? With word boundaries: "ph" cannot
    match inside "phosphate"."""
    lowered = (text or "").lower()
    for alias in _PARAM_ALIASES.get(param_key, ()):
        if re.search(rf"(?<![a-z]){re.escape(alias)}(?![a-z])", lowered):
            return True
    return False


def actions_contradict_panel(payload, readings) -> list[str]:
    """Actions asking to correct a parameter the panel reported as in range.
    Measured only."""
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
    """Did a hazardous-domain agent produce content this turn?"""
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
    """Why safety is required this turn: "contract", "agent", "lexicon", or ""
    for not required."""
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
    """Public boolean form of `_safety_trigger`."""
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
    """Cap the bullet list at MAX_ACTIONS and MAX_ACTION_WORDS. Overflow moves
    to `details`, never gets deleted."""
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
    """Relocate trailing actions until the visible tier fits the budget."""
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
    Hard rule: nothing safety-related stays behind the fold. If the contract
    requires `safety` and it is empty, promote the first sentence of a folded
    section that looks like a warning.
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
                     specialist: dict | None = None,
                     vessel: "VesselContext | None" = None
                     ) -> tuple[Any, ValidationReport]:
    """
    Apply the contract to the synthesizer payload.

    Args:
        payload:    SynthesizerOutput instance (mutated in place).
        contract:   ARCHETYPE_CONTRACTS entry.
        agents:     state["assigned_agents"], to resolve conditional safety.
        detail_cls: details item class, inferred from the payload if None.
        readings:   the specialist's `test_interpretation`, raw.
        specialist: the sub-agents' merged JSON payload. `actions`, `safety`
                    and `constraint_conflict` come from here.
        vessel:     pool/spa and indoor/outdoor. Selects the free chlorine
                    floor and whether cyanuric acid is permitted, so passing
                    None where a spa was meant produces a false clearance.

    Returns (payload, report). `report.needs_retry` lets the caller retry ONCE.
    The panel footer is decided with `panel_needs_footer` in the node, not
    here: this module does not compose the final surface.
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
    #    specialist payload. BEFORE normalization so the caps apply to the
    #    final text, not to what gets discarded.
    if specialist:
        enforce_visible_tier(payload, specialist, language, report)

    # 0a. El objetivo de remediación, sobre las acciones que 0 acaba de
    #     poner. El orden importa: con la lista vacía la inserción es lo
    #     único que sobrevive, y el trace 907cfabe salió con una sola
    #     acción -- clorar a 20 ppm, sin el cierre ni el drenaje delante.
    if specialist:
        enforce_remediation_target(payload, specialist, language, report)

    # 1. Normalize bullets before measuring the budget. Only trimming point:
    #    what exceeds is relocated to `details`, never lost.
    normalize_actions(payload, detail_cls, report)

    # 2. Safety: promote BEFORE the overflow, because it counts as visible.
    trigger = _safety_trigger(contract, agents or [], payload)
    if trigger:
        report.notes.append(f"safety required by: {trigger}")
        promote_safety_from_details(payload, report)

    # 3. Readings BEFORE the budget: anything added competes for space like the
    #    rest of the visible tier. Reconciliation goes first — the panel is
    #    built on degraded readings, never on raw ones, or the line would
    #    assert a nonexistent code and the degradation would arrive too late.
    if readings:
        readings = reconcile_with_published_bands(readings, report,
                                                  vessel=vessel)
        enforce_visible_readings(
            payload, readings, language, report,
            conflict=(specialist or {}).get("constraint_conflict"),
        )

    # 4. Budget.
    overflow_to_details(payload, report.budget, detail_cls, report)

    # 5. Prune empty sections.
    payload.details = [d for d in payload.details if d.body and d.body.strip()]

    # 6. Visible-tier quality telemetry. Corrects nothing.
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
    """If structured output fails, the front end never sees a different
    format. All the raw text goes into `answer`, unfolded and unvalidated."""
    return output_cls(answer=raw_text, actions=[], safety=None, details=[])