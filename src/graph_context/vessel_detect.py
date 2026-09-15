"""
Vessel detection from free-form user text.

The vessel type decides two bands in WATER_TARGETS: the free chlorine floor
(3 ppm in spas vs 1/2 ppm in pools) and whether cyanuric acid is permitted at
all. Both are closure-relevant, so getting this wrong is not cosmetic — trace
35b8a774 declared an indoor spa's readings "in range" underneath an order to
close it, because the panel resolved against pool bands.

This module is the FALLBACK. The specialist declares the vessel in its own
payload when it knows; `merge_vessel` below prefers that and uses these
patterns only to fill the axes the specialist left blank.
"""

from __future__ import annotations

import re

from .water_targets import VesselContext

#: Spa-family vessels. Checked BEFORE pool terms: "whirlpool" and "spa pool"
#: both contain a pool word and must not resolve to a pool.
_SPA_RE = re.compile(
    r"\b(?:spas?|hot\s+tubs?|whirlpools?|jacuzzis?|"
    r"therapy\s+pools?|hydrotherapy)\b",
    re.IGNORECASE,
)

_POOL_RE = re.compile(
    r"\b(?:pools?|lap\s+pools?|swimming\s+pools?|wading\s+pools?|"
    r"splash\s+pads?|natatoriums?)\b",
    re.IGNORECASE,
)

_INDOOR_RE = re.compile(
    r"\b(?:indoors?|covered|enclosed|natatoriums?)\b",
    re.IGNORECASE,
)

_OUTDOOR_RE = re.compile(
    r"\b(?:outdoors?|open[-\s]air|rooftop|uncovered)\b",
    re.IGNORECASE,
)


def detect_vessel(text: str | None) -> VesselContext:
    """
    Best-effort vessel context from user text. Unknown axes stay None.

    Each axis resolves independently: "indoor spa" gives both, "spa" gives
    only the kind, "indoor" only the location. A text that names both a spa
    and a pool is ambiguous and yields no kind — a facility question about
    "the pool and the spa" must not silently pick one.
    """
    if not text:
        return VesselContext()

    es_spa = bool(_SPA_RE.search(text))
    es_pool = bool(_POOL_RE.search(text))
    kind = None
    if es_spa != es_pool:
        kind = "spa" if es_spa else "pool"

    adentro = bool(_INDOOR_RE.search(text))
    afuera = bool(_OUTDOOR_RE.search(text))
    indoor = None
    if adentro != afuera:
        indoor = adentro

    return VesselContext(kind=kind, indoor=indoor)


def merge_vessel(declared: VesselContext | None,
                 detected: VesselContext | None) -> VesselContext:
    """
    Combine the specialist's declaration with the text fallback, per axis.

    The specialist wins where it spoke, because it read the whole turn and the
    retrieved corpus; the fallback fills only what it left as None. Neither
    source may overwrite the other's answer with None — that would turn a
    known vessel back into an unknown one.
    """
    d = declared or VesselContext()
    f = detected or VesselContext()
    return VesselContext(
        kind=d.kind if d.kind is not None else f.kind,
        indoor=d.indoor if d.indoor is not None else f.indoor,
    )