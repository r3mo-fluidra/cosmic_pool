# izel_math/tools.py
"""
The six authorized tools for the MATH agent.

Every tool returns a formatted string for LLM consumption and never raises to
the agent: failures come back as explicit, actionable messages so the model
stops rather than inventing a number.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.tools import tool

from .catalog import (
    DISAMBIGUATION,
    CalcResult,
    CatalogError,
    ConversionError,
    GuardResult,
    MissingVariable,
    UnsafeExpression,
    catalog,
    convert,
    evaluate_guards,
    get_constant_spec,
    get_formula,
    index,
    is_executable,
    match_formulas,
    resolve_product,
    round_sig,
    safe_eval,
    sig_figs,
    table_lookup,
)

MATH_TOOL_NAMES = (
    "resolve_formula",
    "get_constant",
    "convert_units",
    "lookup_product",
    "calculate",
    "check_plausibility",
)

_CATEGORY_LABEL = {
    "a": "universal math",
    "b": "industry convention",
    "c": "CODE/AHJ-DEPENDENT",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_guards(guards: list[GuardResult]) -> str:
    if not guards:
        return "  (none)\n"
    lines = []
    for g in guards:
        marker = {
            "failed": "FAILED", "passed": "ok",
            "advisory": "note", "not_evaluated": "not evaluated",
        }[g.status]
        lines.append(f"  [{g.severity}/{marker}] {g.message}")
        if g.check and g.status in ("failed", "not_evaluated"):
            lines.append(f"      condition: {g.check}")
    return "\n".join(lines) + "\n"


def _fmt_formula(f: dict, verbose: bool = True) -> str:
    cat = str(f.get("category", "?"))
    parts = [
        f"formula_id: {f['formula_id']}",
        f"name: {f['name']}",
        f"domain: {f.get('domain', '-')} | tier: {f.get('tier', '-')} | "
        f"category: {cat} ({_CATEGORY_LABEL.get(cat, 'uncategorized')})",
    ]

    if not is_executable(f):
        parts.append("EXECUTABLE: NO -- this entry is a procedure, not a computation.")
        parts.append(f"procedure: {f.get('procedure', '(none recorded)')}")
        parts.append(f"result_unit: {f.get('result_unit', '-')}")
        parts.append(f"source_id: {f.get('source_id', '-')}")
        parts.append("Do NOT call calculate() on this formula_id.")
        return "\n".join(parts) + "\n"

    parts.append(f"expression: {f['expression']}")
    parts.append(f"result_unit: {f.get('result_unit', '-')}")

    if verbose:
        parts.append("required_inputs:")
        for i in f.get("inputs", []):
            bits = [f"  - {i['name']} [{i.get('unit', '-')}]"]
            if i.get("min") is not None or i.get("max") is not None:
                bits.append(f"valid {i.get('min')}..{i.get('max')}")
            if i.get("from_table"):
                bits.append(
                    f"from table '{i['from_table']}' keyed on '{i.get('table_input')}' "
                    f"(supply '{i.get('table_input')}' and it is looked up automatically)"
                )
            if i.get("default_constant"):
                bits.append(f"defaults to constant '{i['default_constant']}'")
            if i.get("note"):
                bits.append(f"note: {i['note']}")
            parts.append(" | ".join(bits))

        if f.get("constants"):
            parts.append(f"constants_used: {', '.join(f['constants'])}")
        if f.get("product_spec"):
            parts.append(f"product_spec: {f['product_spec']}")
        if f.get("conservative_direction"):
            parts.append(f"conservative_direction: {f['conservative_direction']}")
        if f.get("target_range"):
            parts.append(f"target_range: {f['target_range']}")
        if f.get("accuracy"):
            parts.append(f"accuracy: {f['accuracy']}")

        parts.append("guards:")
        parts.append(_fmt_guards(evaluate_guards(f.get("guards", []), {})).rstrip())

    parts.append(f"source_id: {f.get('source_id', '-')}")

    we = f.get("worked_example")
    if we:
        parts.append(
            f"validated_against: inputs={we.get('inputs')} -> expected "
            f"{we.get('expected')} (tolerance {we.get('tolerance')})"
        )
    elif f.get("tier") == "extended":
        parts.append(
            "validated_against: NONE. Tier is 'extended' -- no published worked "
            "example validates this expression. Report that caveat to the user."
        )

    if f.get("verify"):
        parts.append(
            "WARNING verify=true: one or more values in this entry are provisional "
            "and unconfirmed against a primary source. State this to the user."
        )
    if f.get("ahj_override"):
        parts.append(
            "WARNING ahj_override=true: the governing value varies by local health "
            "authority. This is NOT a compliance answer -- route that to the compliance agent."
        )
    if f.get("preferred_alternative"):
        parts.append(
            f"PREFERRED ALTERNATIVE: '{f['preferred_alternative']}' is the better "
            "formula for this calculation. Use it unless the user specifically "
            "requires this legacy form."
        )

    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# 1. resolve_formula
# ---------------------------------------------------------------------------

@tool
def resolve_formula(
    intent: str,
    geometry: Optional[str] = None,
    venue_type: Optional[str] = None,
) -> str:
    """
    Retrieve the governing formula for a requested quantity from the catalog.
    Call this FIRST for every calculation request.

    Resolution is an exact lookup against formula_id and declared aliases. There
    is no semantic search: an intent that does not match is reported as not
    found, never approximated to the nearest formula.

    BUDGET: at most two calls per calculation. One to resolve, and at most one
    more to pick a formula_id from a CANDIDATES list. Rephrasing the intent
    hits the same lookup table and returns the same result -- it never finds a
    formula that the first call missed.

    The first line of the result is a machine-readable STATUS:
      RESOLVED    one formula matched. Your next action is either `calculate`
                  or reporting the missing required_inputs -- never another
                  resolve_formula call.
      CANDIDATES  several matched. Call again with intent=<exact formula_id>
                  from the list. Do not rephrase.
      NOT_FOUND   nothing matched. Stop. Do not retry, do not reconstruct the
                  formula from general knowledge.

    When STATUS is RESOLVED, compare required_inputs against what the user
    actually provided. If any is missing, stop and ask for it. Do not call
    get_constant or lookup_product to work around a missing user input -- a
    constant is not a substitute for the pool's volume.

    Args:
        intent: The quantity to compute, in canonical English terms, OR the
            exact formula_id returned by a previous CANDIDATES response.
            Examples: "pool volume", "turnover time", "required flow rate",
            "liquid chlorine dose", "breakpoint chlorination target",
            "dilution volume", "spa water replacement interval",
            "saturation index", "filtration rate", "pipe velocity".
        geometry: Basin shape when the formula depends on it: "rectangular",
            "circular", "oval", "stadium", "kidney", "irregular",
            "constant_depth". Required for volume and surface-area intents.
        venue_type: "pool", "spa", "wading_pool", "therapy_pool" when the
            formula or its guards differ by venue.

    Returns:
        A formatted FormulaSpec: formula_id, tier, category, executable
        expression, required inputs with units and valid ranges, constants,
        guards, source_id, and the worked example it was validated against.
        Flags verify, ahj_override, and preferred_alternative where present.
        Returns candidates when several match, and an explicit "no formula
        found" message when none do.
    """
    try:
        # Salida directa del bucle: si `intent` es un formula_id literal --
        # típicamente porque el modelo está respondiendo a un CANDIDATES
        # anterior -- resuélvelo sin pasar por el fuzzy matcher. Sin esto,
        # "acid demand" y "lower pH" siguen cayendo en el mismo matcher
        # difuso que ya los desambiguó distinto la primera vez.
        try:
            direct = get_formula(intent)
        except (CatalogError, KeyError, LookupError):
            direct = None
        if direct is not None:
            return "STATUS: RESOLVED\n" + _fmt_formula(direct)

        matches = match_formulas(intent, limit=None)

        if not matches:
            domains = ", ".join(sorted(index()["by_domain"]))
            return (
                f"STATUS: NOT_FOUND\n"
                f"No formula in the catalog computes '{intent}'.\n"
                f"Do NOT retry with a synonym or a related term. Do not "
                f"reconstruct a formula from general knowledge. Report to the "
                f"user that the catalog has no entry for this calculation.\n"
                f"Available domains: {domains}"
            )

        hints = [t for t in " ".join(filter(None, [geometry, venue_type])).lower().split()
                 if len(t) > 3]
        if hints and len(matches) > 1:
            boosted = [m for m in matches
                       if any(t in m[0].lower() for t in hints)]
            if boosted:
                # A geometry hint is decisive: "pool volume" + circular means
                # volume_circular, not a six-way menu.
                matches = ([(fid, 1.0) for fid, _ in boosted]
                           if len(boosted) == 1
                           else boosted + [m for m in matches if m not in boosted])

        matches = matches[:6]

        if len(matches) == 1 or matches[0][1] >= 1.0:
            exact = [m for m in matches if m[1] >= 1.0] or matches[:1]
            if len(exact) == 1:
                return "STATUS: RESOLVED\n" + _fmt_formula(get_formula(exact[0][0]))
            matches = exact

        out = [
            f"STATUS: CANDIDATES ({len(matches)} matches for '{intent}')\n"
            f"Pick ONE formula_id from the list below and call resolve_formula "
            f"again with intent=<that exact formula_id>. Do NOT rephrase the "
            f"intent -- re-querying with a synonym returns this same list.\n"
            f"If none of these computes what was asked, stop and report "
            f"insufficient_evidence.\n"
        ]
        for fid, score in matches:
            f = get_formula(fid)
            out.append(
                f"--- {fid} (match {score:.2f}) ---\n"
                f"{f['name']} | domain {f.get('domain')} | tier {f.get('tier')}\n"
                f"expression: {f.get('expression') or '(procedure, not executable)'}\n"
                f"inputs: {', '.join(i['name'] for i in f.get('inputs', [])) or '-'}\n"
            )
        return "\n".join(out)

    except CatalogError as e:
        return f"STATUS: NOT_FOUND\nCatalog error in resolve_formula: {e}"
    except Exception as e:  # pragma: no cover
        return f"STATUS: NOT_FOUND\nError in resolve_formula: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 2. get_constant
# ---------------------------------------------------------------------------

@tool
def get_constant(key: str) -> str:
    """
    Retrieve a domain constant from the registry together with its safety
    factor. Never recall a constant from memory -- always call this.

    Args:
        key: Registry key in lowercase_snake_case. Examples:
            "gal_per_cuft_industry", "gal_per_cuft_exact", "lb_per_gal_water",
            "dose_const_liquid", "ft_head_per_psi", "pi", "naocl_fraction",
            "lsi_tds_constant", "spa_replacement_const".

    Returns:
        Value, unit, category (a=universal math, b=industry convention,
        c=code/AHJ-dependent), the conservative-error direction and why, the
        nominal range when the value is a range, a disambiguation note when a
        near-identical constant exists, and the verify flag.
        Returns "constant not found" rather than a substituted value.
    """
    try:
        c = get_constant_spec(key)
        lines = [
            f"constant: {c.key}",
            f"value: {c.value}" + (f" {c.unit}" if c.unit else ""),
            f"category: {c.category} -- {c.category_label}",
        ]
        if c.value_range:
            lines.append(
                f"nominal_range: {c.value_range[0]} to {c.value_range[1]} "
                f"(the single value above is the catalog default within that range)"
            )
        if c.conservative_direction:
            lines.append(f"conservative_direction: {c.conservative_direction}")
        if c.notes:
            lines.append(f"notes: {c.notes}")
        if c.key in DISAMBIGUATION:
            lines.append(f"DISAMBIGUATION: {DISAMBIGUATION[c.key]}")
        if c.verify:
            lines.append(
                "WARNING verify=true: provisional value, not confirmed against a "
                "primary source. State this when you use it."
            )
        if c.category == "c":
            lines.append(
                "WARNING category c: this is a non-normative default. The local "
                "health authority overrides it. Not a compliance answer."
            )
        return "\n".join(lines)

    except CatalogError as e:
        return (
            f"CONSTANT NOT FOUND: {e}\n"
            "Do not substitute a remembered value. Stop the calculation and report this."
        )
    except Exception as e:  # pragma: no cover
        return f"Error in get_constant: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 3. convert_units
# ---------------------------------------------------------------------------

@tool
def convert_units(
    value: float,
    from_unit: str,
    to_unit: str,
    quantity: Optional[str] = None,
) -> str:
    """
    Convert a value between units using the conversion registry. Use for every
    unit change, including trivial ones.

    Args:
        value: Numeric value to convert.
        from_unit: Source unit. Supported: gal, ft3, L, mL, fl_oz, acre_ft |
            lb, oz, kg, g | gpm, gph, gpd, cfs | psi, ft_head, kPa |
            ft, in, m, cm | ft2, m2, in2 | ppm, mg_L, percent, fraction |
            min, hr, day | F, C.
        to_unit: Target unit, same vocabulary.
        quantity: Optional dimension hint ("volume", "mass", "flow", "pressure",
            "length", "area", "concentration", "time", "temperature") used to
            reject dimensionally invalid conversions.

    Returns:
        Converted value and the factor applied. Rejects cross-dimension
        conversions with an explicit error rather than producing a number.
    """
    try:
        result, detail = convert(float(value), from_unit, to_unit, quantity)
        return (
            f"result: {result} {to_unit}\n"
            f"operation: {detail}\n"
            f"source_id: catalog constants (unit registry)"
        )
    except ConversionError as e:
        return f"CONVERSION REJECTED: {e}"
    except Exception as e:  # pragma: no cover
        return f"Error in convert_units: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 4. lookup_product
# ---------------------------------------------------------------------------

def lookup_product(
    product_name: str,
    label_percent: Optional[float] = None,
) -> str:
    """
    Retrieve dosing-relevant properties for a pool chemical: available chlorine
    and CYA contribution for sanitizers, acid strength and dose-rate scaling for
    acids, plus the handling hazards that must be surfaced with any dose.
    Call before any dosing calculation that names a product.

    Args:
        product_name: Product as the user described it. Examples:
            "sodium hypochlorite", "liquid chlorine", "cal hypo", "dichlor",
            "trichlor", "muriatic acid", "dry acid", "sodium bisulfate".
        label_percent: Strength from the product label (as a percent, e.g. 12.5
            for hypochlorite, 31.45 for muriatic acid), if the user supplied it.
            When present it overrides the catalog range entirely and no safety
            factor is applied.

    Returns:
        For sanitizers: available chlorine as a RANGE when label_percent is
        absent, the fraction selected, CYA contribution per ppm FC, and the
        safety-factor direction. For acids: the strength assumed by the catalog
        dose rate, the scaling multiplier if the user's product differs, and the
        formula_id to use. For every product: mandatory handling hazards and an
        explicit statement that the product label controls.
    """
    try:
        canonical, spec = resolve_product(product_name)
        strength = get_constant_spec(spec["strength"])
        product_class = spec.get("class", "sanitizer")

        lines = [f"product: {canonical}", f"class: {product_class}"]

        if label_percent is not None:
            fraction = float(label_percent) / 100.0
            lines += [
                f"fraction_available: {fraction} (FROM LABEL, {label_percent}%)",
                "safety_factor: none applied -- label value supersedes the catalog range.",
            ]
        else:
            fraction = strength.value
            if strength.value_range:
                lo, hi = strength.value_range
                lines.append(f"nominal_range: {lo * 100:.1f}% to {hi * 100:.1f}%")
            lines += [
                f"fraction_available: {fraction} (CATALOG DEFAULT -- an assumption)",
                f"safety_factor: {strength.conservative_direction or 'not declared'}",
                "ASSUMPTION TO REPORT: no label strength was supplied. State the "
                "value used and that the result changes if the label differs.",
            ]

        if product_class == "acid":
            lines += _acid_lines(spec, fraction, label_percent)
        else:
            lines += _sanitizer_lines(canonical, spec)

        for hazard in spec.get("hazards", []):
            lines.append(f"HAZARD (surface this with any dose): {hazard}")

        if strength.notes:
            lines.append(f"notes: {strength.notes}")

        lines.append("THE PRODUCT LABEL CONTROLS. Strengths vary by manufacturer and lot.")
        return "\n".join(lines)

    except CatalogError as e:
        return (
            f"PRODUCT NOT FOUND: {e}\n"
            "Do not assume a strength or a dose rate. Ask the user for the "
            "product's full name and its label strength."
        )
    except Exception as e:  # pragma: no cover
        return f"Error in lookup_product: {type(e).__name__}: {e}"


def _sanitizer_lines(canonical: str, spec: dict) -> list[str]:
    lines = []
    if spec.get("cya"):
        cya = get_constant_spec(spec["cya"])
        lines.append(
            f"cya_contribution: {cya.value} ppm CYA per 1.0 ppm FC delivered "
            f"-- FLAG THIS to the user even if they only asked about chlorine. "
            f"CYA accumulates and is removable only by dilution."
        )
    else:
        lines.append("cya_contribution: none")

    if canonical == "sodium hypochlorite":
        lines.append(
            "DEGRADATION: liquid chlorine loses strength in storage. Stored "
            "product is weaker than its label. Prefer a measured strength."
        )
    return lines


def _acid_lines(spec: dict, fraction: float, label_percent: Optional[float]) -> list[str]:
    rate = get_constant_spec(spec["dose_rate"])
    reference = spec["reference_strength"]
    lines = [
        "cya_contribution: not applicable (this is an acid, not a sanitizer).",
        f"dose_formula: {spec['dose_formula']}",
        f"dose_rate: {rate.value} {spec['dose_unit']} per 10,000 gal per 10 ppm TA drop, "
        f"AT A REFERENCE STRENGTH OF {reference * 100:.1f}%",
    ]

    scaling = reference / fraction if fraction else None
    if scaling is not None and abs(scaling - 1.0) > 0.02:
        lines.append(
            f"STRENGTH SCALING REQUIRED: the user's product is {fraction * 100:.1f}%, "
            f"not the {reference * 100:.1f}% the dose rate assumes. Multiply the "
            f"computed dose by {scaling:.2f}. Do not report the unscaled figure."
        )
    else:
        lines.append("strength_scaling: 1.00 (product matches the reference strength)")

    if label_percent is None:
        lines.append(
            "Assumed the HIGH end of the strength range, which UNDERSTATES the "
            "dose. Underdosing is the recoverable direction: low-pH overshoot is "
            "corrosive and slow to reverse. Add incrementally, retest, repeat."
        )
    if rate.verify:
        lines.append(f"VERIFY: {rate.notes}")
    return lines


# ---------------------------------------------------------------------------
# 5. calculate
# ---------------------------------------------------------------------------

def _coerce(entry: Any) -> tuple[float | None, str | None]:
    if isinstance(entry, dict):
        v = entry.get("value")
        return (float(v) if v is not None else None), entry.get("unit")
    if isinstance(entry, (int, float)):
        return float(entry), None
    try:
        return float(entry), None
    except (TypeError, ValueError):
        return None, None


@tool
def calculate(
    formula_id: str,
    inputs: dict[str, Any],
    precision: Optional[int] = None,
) -> str:
    """
    Execute a catalog formula against named inputs. Call only after
    resolve_formula returned a FormulaSpec and all required inputs are present.

    Args:
        formula_id: The formula_id returned by resolve_formula. NOT a free-text
            expression -- the expression comes from the catalog, never from you.
        inputs: Mapping of input name to {"value": <number>, "unit": "<unit>"},
            using the names and units declared in the FormulaSpec. A bare number
            is accepted and assumed to already be in the declared unit. For
            table-driven inputs supply the table key instead (e.g. "temp_F" and
            the temperature factor is looked up automatically).
        precision: Optional significant figures. Defaults to the least precise
            input rather than to the float representation.

    Returns:
        Substituted expression, intermediate steps, result with unit, guards
        triggered, and every safety factor applied. Reports an error rather than
        computing when an input is missing, a unit mismatches, or a value falls
        outside its declared valid range.
    """
    try:
        f = get_formula(formula_id)
    except CatalogError as e:
        return f"UNKNOWN FORMULA: {e}\nCall resolve_formula first."

    if not is_executable(f):
        return (
            f"NOT EXECUTABLE: '{formula_id}' is a procedure, not a computation.\n"
            f"procedure: {f.get('procedure', '(none recorded)')}\n"
            "Report the procedure to the user. Do not produce a number."
        )

    inputs = inputs or {}
    steps: list[str] = []
    safety: list[str] = []
    env: dict[str, Any] = {}
    errors: list[str] = []

    # --- constants -------------------------------------------------------
    for key in f.get("constants", []):
        try:
            c = get_constant_spec(key)
        except CatalogError as e:
            return f"MISSING CONSTANT: {e}\nCalculation stopped."
        env[key] = c.value
        steps.append(f"constant {key} = {c.value}" + (f" {c.unit}" if c.unit else ""))
        if c.conservative_direction:
            safety.append(f"{key}: {c.conservative_direction}")
        if c.verify:
            safety.append(f"{key}: PROVISIONAL (verify=true)")

    # --- declared inputs -------------------------------------------------
    for spec in f.get("inputs", []):
        name = spec["name"]
        want_unit = spec.get("unit")

        # table-driven input
        if spec.get("from_table") and name not in inputs:
            tkey = spec.get("table_input")
            if tkey in inputs:
                raw, _ = _coerce(inputs[tkey])
                if raw is None:
                    errors.append(f"'{tkey}' is not numeric")
                    continue
                val, detail = table_lookup(spec["from_table"], raw)
                env[name] = val
                steps.append(f"{name} <- {detail}")
                continue

        if name not in inputs:
            if spec.get("default_constant"):
                c = get_constant_spec(spec["default_constant"])
                env[name] = c.value
                steps.append(
                    f"{name} = {c.value} (DEFAULTED from constant "
                    f"'{spec['default_constant']}' -- report as an assumption)"
                )
                safety.append(
                    f"{name}: defaulted to {spec['default_constant']}"
                    + (f" ({c.conservative_direction})" if c.conservative_direction else "")
                )
                continue
            if spec.get("from_table"):
                errors.append(
                    f"missing '{name}' (or supply '{spec.get('table_input')}' for lookup)"
                )
            else:
                errors.append(f"missing '{name}' [{want_unit}]")
            continue

        value, unit = _coerce(inputs[name])
        if value is None:
            errors.append(f"'{name}' is not numeric")
            continue

        if unit and want_unit and unit.lower() != str(want_unit).lower():
            try:
                value, detail = convert(value, unit, want_unit)
                steps.append(f"converted {name}: {detail}")
            except ConversionError as e:
                errors.append(f"'{name}' unit mismatch: {e}")
                continue

        lo, hi = spec.get("min"), spec.get("max")
        if lo is not None and value < lo:
            errors.append(f"'{name}' = {value} is below the valid minimum {lo} {want_unit}")
            continue
        if hi is not None and value > hi:
            errors.append(f"'{name}' = {value} is above the valid maximum {hi} {want_unit}")
            continue

        env[name] = value
        steps.append(f"{name} = {value} {want_unit or ''}".rstrip())

    if errors:
        return (
            f"CANNOT COMPUTE '{formula_id}'. Refusing rather than estimating.\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\nList exactly these to the user and request them."
        )

    # extra context for guard evaluation only
    guard_env = dict(env)
    for k, v in inputs.items():
        if k not in guard_env:
            val, _ = _coerce(v)
            guard_env[k] = val if val is not None else v

    # --- evaluate --------------------------------------------------------
    expr = f["expression"]
    try:
        raw_value = safe_eval(expr, env)
    except MissingVariable as e:
        return (
            f"CANNOT COMPUTE '{formula_id}': expression needs undeclared variable(s): {e}\n"
            "This is a catalog defect -- report it rather than substituting a value."
        )
    except (UnsafeExpression, ZeroDivisionError, ValueError) as e:
        return f"CANNOT COMPUTE '{formula_id}': {type(e).__name__}: {e}"

    # Precision is governed by the user's inputs, NOT by catalog constants.
    # A constant written as 7.5 in YAML is exact-as-defined; letting it cap the
    # result at two significant figures would destroy real precision.
    const_keys = set(f.get("constants", []))
    measured = [
        v for k, v in env.items()
        if isinstance(v, (int, float)) and k not in const_keys
    ]
    figs = precision or (min((sig_figs(v) for v in measured), default=6))
    figs = max(3, min(figs, 8))
    shown = round_sig(float(raw_value), figs) if isinstance(raw_value, float) else raw_value

    substituted = expr
    for name in sorted(env, key=len, reverse=True):
        substituted = substituted.replace(name, str(env[name]))

    guards = evaluate_guards(f.get("guards", []), guard_env)
    if f.get("conservative_direction"):
        safety.append(f"formula: {f['conservative_direction']}")

    out = [
        f"formula_id: {formula_id}",
        f"formula_name: {f['name']}",
        f"expression: {expr}",
        f"substituted: {substituted}",
        "steps:",
        *[f"  {i + 1}. {s}" for i, s in enumerate(steps)],
        f"result: {shown} {f.get('result_unit', '')}".rstrip(),
        f"exact_value: {raw_value}",
        f"precision: rounded to {figs} significant figures (least precise input governs)",
        f"source_id: {f.get('source_id', '-')}",
        "guards:",
        _fmt_guards(guards).rstrip(),
    ]

    if safety:
        out += ["safety_factors_applied:", *[f"  - {s}" for s in safety]]
    else:
        out.append("safety_factors_applied: none")

    if f.get("verify"):
        out.append(
            "WARNING verify=true: this entry contains provisional values. Say so."
        )
    if f.get("ahj_override"):
        out.append(
            "WARNING ahj_override=true: governing value varies by local authority. "
            "Not a compliance answer."
        )
    if f.get("tier") == "extended":
        out.append(
            "NOTE tier=extended: no published worked example validates this "
            "expression. Report that caveat."
        )

    failed = [g for g in guards if g.status == "failed" and g.severity == "error"]
    if failed:
        out.append(
            "ERROR-SEVERITY GUARD FAILED. The result above is not trustworthy -- "
            "report the failure, do not present the number as an answer."
        )

    return "\n".join(out)


# ---------------------------------------------------------------------------
# 6. check_plausibility
# ---------------------------------------------------------------------------

_QUANTITY_ALIASES = {
    "volume": ("residential_pool_volume_gal", "commercial_pool_volume_gal", "spa_volume_gal"),
    "turnover_time": ("turnover_pool_hr", "turnover_spa_hr"),
    "flow_rate": ("recirculation_flow_gpm",),
    "filtration_rate": ("filtration_rate_gpm_sqft",),
    "pipe_velocity": ("pipe_velocity_suction_fps", "pipe_velocity_return_fps"),
    "free_chlorine": ("fc_pool_ppm", "fc_spa_ppm"),
    "dose": ("liquid_chlorine_dose_floz",),
    "temperature": ("spa_water_temp_F",),
    "calcium_hardness": ("calcium_hardness_pool_ppm", "calcium_hardness_spa_ppm"),
    "avg_depth": ("avg_depth_pool_ft", "avg_depth_spa_ft"),
    "saturation_index": ("lsi",),
}


def _candidate_ranges(quantity: str, venue_type: str | None) -> list[str]:
    ranges = catalog()["plausibility_ranges"]
    q = (quantity or "").lower().strip()

    if q in ranges:
        return [q]

    keys = list(_QUANTITY_ALIASES.get(q, ()))
    if not keys:
        keys = [k for k in ranges if q and (q in k.lower())]

    if venue_type:
        v = venue_type.lower()
        venue_hits = [k for k in keys if v.split("_")[0] in k.lower()]
        if venue_hits:
            return venue_hits
    return keys


@tool
def check_plausibility(
    quantity: str,
    value: float,
    unit: str,
    venue_type: Optional[str] = None,
) -> str:
    """
    Compare a computed result against the catalog's operating ranges. Call on
    every final result before reporting it.

    Args:
        quantity: What was computed. Examples: "volume", "turnover_time",
            "flow_rate", "filtration_rate", "pipe_velocity", "dose",
            "free_chlorine", "ph", "cya", "saturation_index", "temperature".
            An exact plausibility_ranges key is also accepted.
        value: The computed value.
        unit: Its unit.
        venue_type: "pool", "spa", "wading_pool", "therapy_pool" -- ranges
            differ by venue (a 0.5 h spa turnover is normal; for a lap pool it
            is not).

    Returns:
        Pass or fail, the range compared against, and its normative status.
        Ranges here are ENGINEERING SANITY BOUNDS, not code limits. A failed
        check means "outside typical operating range", never "non-compliant".
    """
    try:
        ranges = catalog()["plausibility_ranges"]
        keys = _candidate_ranges(quantity, venue_type)

        if not keys:
            return (
                f"NO RANGE for quantity '{quantity}'. Cannot validate.\n"
                f"Report the result without a plausibility claim.\n"
                f"Known quantities: {', '.join(sorted(ranges))}"
            )

        lines = [f"value: {value} {unit}"]
        any_fail = False

        for k in keys:
            r = ranges[k]
            lo, hi = r.get("min"), r.get("max")
            ok = (lo is None or value >= lo) and (hi is None or value <= hi)
            any_fail = any_fail or not ok
            lines.append(
                f"  [{'PASS' if ok else 'OUTSIDE RANGE'}] {k}: {lo} to {hi}"
                + (f" -- {r['flag']}" if r.get("flag") and not ok else "")
            )

        lines.append(
            "normative: FALSE -- these are engineering sanity bounds from the "
            "catalog, not code limits."
        )
        if any_fail:
            lines.append(
                "ACTION: report this as 'outside the typical operating range' and "
                "recheck the inputs. Do NOT report it as non-compliant. Any "
                "compliance question goes to the compliance agent."
            )
        if len(keys) > 1:
            lines.append(
                "NOTE: several ranges matched. Pass venue_type to narrow, and say "
                "which range you compared against."
            )
        return "\n".join(lines)

    except Exception as e:  # pragma: no cover
        return f"Error in check_plausibility: {type(e).__name__}: {e}"


MATH_TOOLS = [
    resolve_formula,
    get_constant,
    convert_units,
    lookup_product,
    calculate,
    check_plausibility,
]
