# izel_math/catalog.py
"""
Deterministic engine behind the math agent's tools.

Single source of truth is pool_math_catalog_v2.yaml. Nothing in this module
hard-codes a domain constant: every factor is read from the catalog so the YAML
stays the only place a value can be changed.
"""

from __future__ import annotations

import ast
import math
import os
import re
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

import yaml

# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------

DEFAULT_CATALOG_PATH = os.getenv(
    "POOL_MATH_CATALOG",
    os.path.join(os.path.dirname(__file__), "data", "pool_math_catalog_v2.yaml"),
)

_catalog: Optional[dict] = None
_index: Optional[dict] = None
_lock = threading.Lock()


class CatalogError(RuntimeError):
    """Raised when the catalog is missing, malformed, or an entry is absent."""


def load_catalog(path: str | None = None, force: bool = False) -> dict:
    """Load and cache the YAML catalog. Thread-safe."""
    global _catalog, _index
    with _lock:
        if _catalog is not None and not force:
            return _catalog

        target = path or DEFAULT_CATALOG_PATH
        if not os.path.exists(target):
            raise CatalogError(f"Catalog not found at {target}")

        with open(target, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        for block in ("catalog", "constants", "formulas", "plausibility_ranges"):
            if block not in data:
                raise CatalogError(f"Catalog is missing required block '{block}'")

        _catalog = data
        _index = _build_index(data)
        return _catalog


def _build_index(data: dict) -> dict:
    by_id: dict[str, dict] = {}
    by_alias: dict[str, list[str]] = {}

    for f in data["formulas"]:
        fid = f["formula_id"]
        by_id[fid] = f
        for key in _alias_keys(fid) + [_norm(a) for a in f.get("aliases", [])]:
            by_alias.setdefault(key, [])
            if fid not in by_alias[key]:
                by_alias[key].append(fid)

    return {
        "by_id": by_id,
        "by_alias": by_alias,
        "by_domain": _group(data["formulas"], "domain"),
    }


def _group(items: list[dict], key: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for it in items:
        out.setdefault(it.get(key, "unknown"), []).append(it["formula_id"])
    return out


def index() -> dict:
    load_catalog()
    assert _index is not None
    return _index


def catalog() -> dict:
    return load_catalog()


# ---------------------------------------------------------------------------
# Normalization / matching
# ---------------------------------------------------------------------------

_STOP = {"the", "a", "an", "of", "for", "in", "my", "is", "to", "how", "what", "much", "many"}


def _norm(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s_]", " ", text)
    text = re.sub(r"[\s_]+", " ", text)
    return text.strip()


def _tokens(text: str) -> set[str]:
    return {t for t in _norm(text).split() if t and t not in _STOP}


def _alias_keys(formula_id: str) -> list[str]:
    return [formula_id, _norm(formula_id.replace("_", " "))]


def match_formulas(intent: str, limit: int = 5) -> list[tuple[str, float]]:
    """
    Exact-then-token match against formula_id and declared aliases.
    Deliberately NOT semantic: a non-match is reported, never approximated.
    Returns [(formula_id, score)] with score 1.0 for an exact alias hit.
    """
    idx = index()
    key = _norm(intent)

    if key in idx["by_alias"]:
        return [(fid, 1.0) for fid in idx["by_alias"][key]][:limit]

    want = _tokens(intent)
    if not want:
        return []

    scored: dict[str, float] = {}
    for alias_key, fids in idx["by_alias"].items():
        have = _tokens(alias_key)
        if not have:
            continue
        overlap = len(want & have)
        if not overlap:
            continue
        # Jaccard-style, biased toward covering the alias
        score = overlap / max(len(have), 1) * 0.6 + overlap / max(len(want), 1) * 0.4
        for fid in fids:
            scored[fid] = max(scored.get(fid, 0.0), score)

    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    hits = [(fid, s) for fid, s in ranked if s >= 0.5]
    # Caller trims. Returning the full band lets geometry/venue boosting reach
    # a correct-but-lower-scored entry before truncation.
    return hits if limit is None else hits[: max(limit, 12)]


# ---------------------------------------------------------------------------
# Safe expression evaluation
# ---------------------------------------------------------------------------

_ALLOWED_FUNCS = {
    "floor": math.floor,
    "ceil": math.ceil,
    "sqrt": math.sqrt,
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Load, ast.Call,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
    ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.Compare, ast.Eq, ast.NotEq,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.BoolOp, ast.And, ast.Or, ast.Not,
    ast.IfExp, ast.Tuple,
)


class UnsafeExpression(ValueError):
    pass


class MissingVariable(KeyError):
    pass


def safe_eval(expression: str, variables: dict[str, Any], strict: bool = True) -> Any:
    """
    Evaluate a whitelisted arithmetic expression. No attribute access, no
    subscripting, no comprehensions, no imports, no builtins beyond _ALLOWED_FUNCS.

    strict=True  -- every referenced name must exist; raises MissingVariable.
                    This is the mode used for formulas: a missing input must
                    stop the calculation, never be silently defaulted.
    strict=False -- names may be absent; Python's own short-circuit rules
                    decide whether they are reached. Used for guards, where
                    `a == 0 or b == true` can be satisfied by the left side
                    alone. An unreached name is never bound to a value, so a
                    comparison like `x <= 1000` cannot pass spuriously.
    """
    src = (expression or "").strip()
    if not src:
        raise UnsafeExpression("Empty expression")

    # Guard strings in the catalog use YAML-ish literals
    src = re.sub(r"\btrue\b", "True", src)
    src = re.sub(r"\bfalse\b", "False", src)

    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpression(f"Cannot parse expression: {exc}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeExpression(
                f"Disallowed syntax {type(node).__name__} in expression"
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise UnsafeExpression("Only whitelisted math functions may be called")

    env: dict[str, Any] = dict(_ALLOWED_FUNCS)
    env.update(variables)

    if strict:
        missing = sorted(
            {
                n.id
                for n in ast.walk(tree)
                if isinstance(n, ast.Name) and n.id not in env
            }
        )
        if missing:
            raise MissingVariable(", ".join(missing))

    try:
        return eval(compile(tree, "<catalog>", "eval"), {"__builtins__": {}}, env)
    except NameError as exc:
        raise MissingVariable(str(exc)) from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

@dataclass
class ConstantSpec:
    key: str
    value: float
    unit: str | None
    category: str            # a=universal math, b=industry convention, c=code/AHJ
    verify: bool
    conservative_direction: str | None
    value_range: list[float] | None
    notes: str

    @property
    def category_label(self) -> str:
        return {
            "a": "universal math (does not vary)",
            "b": "industry convention (widely used, not binding)",
            "c": "CODE/AHJ-DEPENDENT (non-normative default, local authority overrides)",
        }.get(self.category, "uncategorized")


def get_constant_spec(key: str) -> ConstantSpec:
    data = catalog()["constants"]
    if key not in data:
        near = [k for k in data if key.lower() in k.lower() or k.lower() in key.lower()]
        hint = f" Did you mean: {', '.join(sorted(near)[:5])}?" if near else ""
        raise CatalogError(f"Constant '{key}' is not in the registry.{hint}")

    raw = data[key]
    return ConstantSpec(
        key=key,
        value=float(raw["value"]),
        unit=raw.get("unit"),
        category=str(raw.get("category", "?")),
        verify=bool(raw.get("verify", False)),
        conservative_direction=raw.get("conservative_direction"),
        value_range=raw.get("range"),
        notes=raw.get("notes", ""),
    )


def constants_env(keys: list[str]) -> dict[str, float]:
    return {k: get_constant_spec(k).value for k in keys}


# Constants that have a near-identical sibling and must not be confused.
DISAMBIGUATION = {
    "gal_per_cuft_exact": (
        "gal_per_cuft_industry (7.5) is the pool-volume convention. Use the exact "
        "7.4805 only when an explicit ft3-to-gallon conversion IS the operation "
        "(filter media, surge tanks)."
    ),
    "gal_per_cuft_industry": (
        "gal_per_cuft_exact (7.4805) is the true conversion. Use 7.5 only for "
        "basin volume computed from dimensions."
    ),
    "pi": "pi_legacy (3.14) exists only to reproduce older published examples. Never mix.",
    "pi_legacy": "Use pi (3.14159) unless explicitly reproducing a legacy worked example.",
    "lsi_tds_constant": "Use lsi_tds_constant_high (12.2) when TDS exceeds ~1000 ppm.",
    "lsi_tds_constant_high": "Use lsi_tds_constant (12.1) at or below ~1000 ppm TDS.",
    "dose_const_liquid": (
        "Legacy CPO constant; ignores specific gravity. dose_liquid_chlorine_sg is "
        "the preferred formula for liquid product."
    ),
}


# ---------------------------------------------------------------------------
# Unit conversion
#
# Factors are pulled from the catalog constants, never hard-coded, so the YAML
# remains the single source of truth. Each unit maps to (dimension, factor to
# that dimension's base unit).
# ---------------------------------------------------------------------------

def _unit_table() -> dict[str, tuple[str, float]]:
    c = lambda k: get_constant_spec(k).value  # noqa: E731

    gal_per_cuft = c("gal_per_cuft_exact")
    return {
        # volume -> base: gal
        "gal": ("volume", 1.0),
        "gallon": ("volume", 1.0),
        "gallons": ("volume", 1.0),
        "ft3": ("volume", gal_per_cuft),
        "cuft": ("volume", gal_per_cuft),
        "l": ("volume", 1.0 / c("liters_per_gal")),
        "liter": ("volume", 1.0 / c("liters_per_gal")),
        "ml": ("volume", 0.001 / c("liters_per_gal")),
        "fl_oz": ("volume", 1.0 / c("floz_per_gal")),
        "floz": ("volume", 1.0 / c("floz_per_gal")),
        "acre_ft": ("volume", c("gal_per_acre_ft")),
        # mass -> base: lb
        "lb": ("mass", 1.0),
        "lbs": ("mass", 1.0),
        "pound": ("mass", 1.0),
        "oz": ("mass", 1.0 / c("oz_per_lb")),
        "ounce": ("mass", 1.0 / c("oz_per_lb")),
        "kg": ("mass", c("lb_per_kg")),
        "g": ("mass", 1.0 / c("g_per_lb")),
        "gram": ("mass", 1.0 / c("g_per_lb")),
        # flow -> base: gpm
        "gpm": ("flow", 1.0),
        "gph": ("flow", 1.0 / c("min_per_hr")),
        "gpd": ("flow", 1.0 / c("min_per_day")),
        "cfs": ("flow", c("gpm_per_cfs")),
        # pressure -> base: psi
        "psi": ("pressure", 1.0),
        "ft_head": ("pressure", c("psi_per_ft_head")),
        "kpa": ("pressure", 1.0 / c("kpa_per_psi")),
        # length -> base: ft
        "ft": ("length", 1.0),
        "feet": ("length", 1.0),
        "in": ("length", 1.0 / 12.0),
        "inch": ("length", 1.0 / 12.0),
        "m": ("length", 3.28084),
        "cm": ("length", 0.0328084),
        # area -> base: ft2
        "ft2": ("area", 1.0),
        "sqft": ("area", 1.0),
        "m2": ("area", 10.7639),
        "in2": ("area", 1.0 / 144.0),
        # concentration -> base: ppm
        "ppm": ("concentration", 1.0),
        "mg_l": ("concentration", c("mg_per_L_per_ppm")),
        "percent": ("concentration", c("ppm_per_percent")),
        "fraction": ("concentration", c("ppm_per_percent") * 100.0),
        # time -> base: min
        "min": ("time", 1.0),
        "minute": ("time", 1.0),
        "hr": ("time", 60.0),
        "hour": ("time", 60.0),
        "sec": ("time", 1.0 / 60.0),
        "day": ("time", 1440.0),
    }


_TEMPERATURE_UNITS = {"f", "degf", "c", "degc"}


def _norm_unit(u: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", (u or "").lower().replace("°", "").replace("/", "_"))


class ConversionError(ValueError):
    pass


def convert(value: float, from_unit: str, to_unit: str,
            quantity: str | None = None) -> tuple[float, str]:
    """Returns (converted_value, human-readable description of the operation)."""
    fu, tu = _norm_unit(from_unit), _norm_unit(to_unit)

    if fu in _TEMPERATURE_UNITS or tu in _TEMPERATURE_UNITS:
        return _convert_temperature(value, fu, tu)

    table = _unit_table()
    if fu not in table:
        raise ConversionError(f"Unknown source unit '{from_unit}'")
    if tu not in table:
        raise ConversionError(f"Unknown target unit '{to_unit}'")

    dim_from, f_from = table[fu]
    dim_to, f_to = table[tu]

    if dim_from != dim_to:
        raise ConversionError(
            f"Cannot convert {dim_from} to {dim_to} ('{from_unit}' -> '{to_unit}'). "
            "Dimensionally invalid; no number produced."
        )
    if quantity and _norm_unit(quantity) not in (dim_from, ""):
        raise ConversionError(
            f"Declared quantity '{quantity}' does not match unit dimension '{dim_from}'."
        )

    result = value * f_from / f_to
    return result, f"{value} {from_unit} x ({f_from} / {f_to}) = {result} {to_unit}"


def _convert_temperature(value: float, fu: str, tu: str) -> tuple[float, str]:
    fu = "f" if fu in ("f", "degf") else "c" if fu in ("c", "degc") else fu
    tu = "f" if tu in ("f", "degf") else "c" if tu in ("c", "degc") else tu
    if fu not in ("f", "c") or tu not in ("f", "c"):
        raise ConversionError("Temperature converts only between F and C.")
    if fu == tu:
        return value, f"{value} = {value} (no change)"
    if fu == "c":
        r = value * 9.0 / 5.0 + 32.0
        return r, f"({value} x 9/5) + 32 = {r} F"
    r = (value - 32.0) * 5.0 / 9.0
    return r, f"({value} - 32) x 5/9 = {r} C"


# ---------------------------------------------------------------------------
# Lookup tables (LSI factors)
# ---------------------------------------------------------------------------

def table_lookup(table_name: str, x: float) -> tuple[float, str]:
    tables = catalog().get("lookup_tables", {})
    if table_name not in tables:
        raise CatalogError(f"Lookup table '{table_name}' not found.")

    t = tables[table_name]
    pts = sorted((float(a), float(b)) for a, b in t["points"])
    lo, hi = pts[0], pts[-1]

    if x <= lo[0]:
        return lo[1], f"{table_name}: clamped at low end ({lo[0]} -> {lo[1]})"
    if x >= hi[0]:
        return hi[1], f"{table_name}: clamped at high end ({hi[0]} -> {hi[1]})"

    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            if t.get("interpolate") != "linear" or x1 == x0:
                return y0, f"{table_name}: step lookup {x0} -> {y0}"
            y = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
            return y, f"{table_name}: linear interp between ({x0},{y0}) and ({x1},{y1}) -> {round(y, 4)}"

    raise CatalogError(f"Lookup failed for {table_name} at {x}")


# ---------------------------------------------------------------------------
# Product registry (derived from catalog constants)
# ---------------------------------------------------------------------------

PRODUCT_MAP: dict[str, dict[str, Any]] = {
    "sodium hypochlorite": {"strength": "naocl_fraction", "cya": None,
                            "synonyms": ["liquid chlorine", "bleach", "naocl", "liquid chlorinating"]},
    "calcium hypochlorite": {"strength": "calhypo_fraction", "cya": None,
                             "synonyms": ["cal hypo", "cal-hypo", "calhypo", "granular chlorine"]},
    "lithium hypochlorite": {"strength": "lihypo_fraction", "cya": None,
                             "synonyms": ["lithium", "lihypo"]},
    "dichlor": {"strength": "dichlor_fraction", "cya": "dichlor_cya_ratio",
                "synonyms": ["sodium dichloroisocyanurate", "di-chlor"]},
    "trichlor": {"strength": "trichlor_fraction", "cya": "trichlor_cya_ratio",
                 "synonyms": ["trichloroisocyanuric acid", "tri-chlor", "tabs", "pucks"]},
}


def resolve_product(name: str) -> tuple[str, dict[str, Any]]:
    key = _norm(name)
    for canonical, spec in PRODUCT_MAP.items():
        candidates = [canonical] + spec["synonyms"]
        if any(_norm(c) == key for c in candidates):
            return canonical, spec
    for canonical, spec in PRODUCT_MAP.items():
        candidates = [canonical] + spec["synonyms"]
        if any(_norm(c) in key or key in _norm(c) for c in candidates):
            return canonical, spec
    raise CatalogError(
        f"Product '{name}' is not in the registry. Known: {', '.join(PRODUCT_MAP)}"
    )


# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------

def sig_figs(value: Any) -> int:
    try:
        d = Decimal(str(value)).normalize()
    except Exception:
        return 6
    digits = d.as_tuple().digits
    s = "".join(str(x) for x in digits).rstrip("0") or "0"
    return max(len(s), 1)


def round_sig(value: float, figs: int) -> float:
    if value == 0 or not math.isfinite(value):
        return value
    figs = max(1, min(figs, 12))
    return round(value, -int(math.floor(math.log10(abs(value)))) + (figs - 1))


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

@dataclass
class GuardResult:
    severity: str
    message: str
    check: str | None = None
    status: str = "advisory"     # passed | failed | advisory | not_evaluated


def evaluate_guards(guards: list[dict], env: dict[str, Any]) -> list[GuardResult]:
    """
    Guards evaluate in non-strict mode so Python's short-circuit rules apply:
    `cya_ppm == 0 or carbonate_alkalinity_corrected == true` can be satisfied by
    the left side alone. A name that IS reached but absent yields
    'not_evaluated', never a spurious pass -- notably, a missing name is never
    bound to a placeholder, so `tds_ppm <= 1000` cannot silently succeed.
    """
    out: list[GuardResult] = []
    for g in guards or []:
        check = g.get("check")
        sev = g.get("severity", "info")
        msg = g.get("message", "")

        if not check:
            out.append(GuardResult(sev, msg, None, "advisory"))
            continue

        try:
            ok = bool(safe_eval(check, env, strict=False))
        except (MissingVariable, UnsafeExpression, TypeError):
            out.append(GuardResult(sev, msg, check, "not_evaluated"))
            continue

        out.append(GuardResult(sev, msg, check, "passed" if ok else "failed"))
    return out


# ---------------------------------------------------------------------------
# Formula access
# ---------------------------------------------------------------------------

def get_formula(formula_id: str) -> dict:
    idx = index()
    if formula_id not in idx["by_id"]:
        raise CatalogError(f"Formula '{formula_id}' is not in the catalog.")
    return idx["by_id"][formula_id]


def is_executable(formula: dict) -> bool:
    return bool(formula.get("expression")) and formula.get("executable", True) is not False


@dataclass
class CalcResult:
    formula_id: str
    substituted: str
    steps: list[str]
    value: float
    unit: str
    guards: list[GuardResult] = field(default_factory=list)
    safety_notes: list[str] = field(default_factory=list)
    precision_note: str = ""
