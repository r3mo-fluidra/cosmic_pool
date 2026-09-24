#!/usr/bin/env python3
"""
measure_prompts.py — mide el tamaño de los prompts renderizados.

  python scripts/measure_prompts.py --save   # guarda la línea base
  python scripts/measure_prompts.py          # compara contra la línea base
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

CHARS_PER_TOKEN = 4.45  # calibrado con el trace 3ae21882 (llamada 1 de compliance)
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
BASELINE = Path(__file__).resolve().parent / "prompt_baseline.json"


class _Blank(dict):
    """Placeholders sin valor se renderizan vacíos (raw_content, roster, etc.)."""
    def __missing__(self, key):
        return ""


def _find_pkg():
    hits = [p for p in SRC.rglob("prompt_archetype.py")
            if "site-packages" not in p.parts]
    if len(hits) != 1:
        sys.exit(f"Esperaba 1 prompt_archetype.py bajo {SRC}, encontré {len(hits)}: {hits}")
    rel = hits[0].parent.relative_to(ROOT)
    return ".".join(rel.parts), hits[0].parent


def measure():
    pkg, path = _find_pkg()
    sys.path.insert(0, str(ROOT))
    pa = importlib.import_module(f"{pkg}.prompt_archetype")
    pr = importlib.import_module(f"{pkg}.prompts")
    sa = importlib.import_module(f"{pkg}.prompts_sub_agents")

    slug_by_name = {v: k for k, v in pr.AGENT_SLUGS.items()}
    out = {
        "planner": pr.PLANNER_PROMPT,
        "general": pr.GENERAL_PROMPT,
        "oos": pr.OOS_PROMPT,
    }
    for name, cfg in sa.AGENT_REGISTRY.items():
        slug = slug_by_name.get(name, name)
        out[f"agent:{slug}"] = pa.build_agent_prompt(cfg, slug)

    out["synthesizer"] = pr.SYNTHESIZER_PROMPT.format_map(_Blank(
        archetype_section=pa.build_synthesizer_archetype_section(
            "explanation", ["compliance"]),
        language="English",
    ))
    out["suggester"] = pr.SUGGESTER_PROMPT.format_map(_Blank(language="English"))
    return path, {k: len(v) for k, v in out.items()}


def _tok(chars):
    return round(chars / CHARS_PER_TOKEN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="guardar como línea base")
    args = ap.parse_args()

    path, sizes = measure()
    base = {}
    if BASELINE.exists() and not args.save:
        base = json.loads(BASELINE.read_text())

    print(f"Prompts leídos desde: {path}\n")
    print(f"{'prompt':<28}{'chars':>8}{'~tokens':>9}{'base':>9}{'Δ%':>8}")
    print("-" * 62)
    for k, c in sizes.items():
        b = base.get(k)
        base_tok = _tok(b) if b else "—"
        delta = f"{(c - b) / b * 100:+.1f}" if b else "—"
        print(f"{k:<28}{c:>8}{_tok(c):>9}{base_tok:>9}{delta:>8}")

    total = sum(sizes.values())
    print("-" * 62)
    if base:
        btotal = sum(base.values())
        print(f"{'TOTAL':<28}{total:>8}{_tok(total):>9}{_tok(btotal):>9}"
              f"{(total - btotal) / btotal * 100:>+8.1f}")
    else:
        print(f"{'TOTAL':<28}{total:>8}{_tok(total):>9}")

    if args.save:
        BASELINE.write_text(json.dumps(sizes, indent=2))
        print(f"\nLínea base guardada en {BASELINE}")


if __name__ == "__main__":
    main()