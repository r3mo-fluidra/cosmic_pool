"""
Auditoría de cobertura del knowledge base.

POR QUÉ ESTE FICHERO Y NO UN SCRIPT DE CARGA
--------------------------------------------
Una evaluación experta de cinco respuestas del agente de química señaló que
faltaban cinco piezas de conocimiento: el objetivo operativo de cloro libre en
función del estabilizador, la corrección de alcalinidad por cianurato, las
interferencias de medición, las firmas de patrón y la potencia relativa del
ácido hipocloroso.

Ninguna faltaba. Las cinco estaban en el grafo:

    fc_to_cya_ratio_heuristic          (:Concept)
    higher_fc_minimum_with_cyanurates  (:Requirement)
    cya_alkalinity_correction_formula  (:Formula)
    hocl_fraction_formula              (:Formula)
    mps_test_interference              (:Hazard)
    cya_accumulation_projection        (:Procedure)

Lo que fallaba era el RETRIEVAL: `Concept`, `Formula`, `Condition` y
`DecisionRule` no figuraban en INTENT_LABELS (agent/tools.py), así que el
scoring los marcaba "[off-intent: contexto, no respuesta]" — le decía al
modelo que el nodo que respondía la pregunta era material de fondo. Y con
`intent="any"`, no encontrar ninguna etiqueta prioritaria ponía el turno en
STATUS: WEAK, cuyo aviso instruye al agente a parar y reportar evidencia
insuficiente.

Escribir conocimiento nuevo encima habría duplicado el corpus sin arreglar la
causa, y habría dejado dos versiones de cada regla divergiendo en silencio.

La lección es la que motiva este fichero: **antes de decidir que al KB le
falta algo, comprobar que lo que tiene es recuperable.** Eso es lo que hace
este script.

USO
---
    .venv/bin/python src/data/kb_coverage_check.py

Para cada pregunta de referencia ejecuta el retrieval real e informa de si el
nodo que debería responderla aparece entre los seeds y con qué marca. Un
`ESPERADO NO RECUPERADO` señala un hueco real; un `off-intent` señala un
problema de etiquetado, no de contenido.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from src.agent.tools import begin_tool_scope, search_seed_nodes  # noqa: E402


# (pregunta del operador, consulta de retrieval, id que DEBE aparecer)
#
# Los ids salen de una inspección del grafo real, no de lo que sería bonito
# que existiera. Añadir una fila acá al detectar una respuesta pobre es la
# forma barata de distinguir "el KB no lo tiene" de "el KB no lo entrega".
CASOS: list[tuple[str, str, str]] = [
    (
        "Con CYA 90, ¿a qué cloro libre apunto?",
        "free chlorine target when cyanuric acid is high",
        "fc_to_cya_ratio_heuristic",
    ),
    (
        "¿Qué parte de la alcalinidad medida es cianurato?",
        "cyanurate contribution to total alkalinity correction",
        "cya_alkalinity_correction_formula",
    ),
    (
        "¿Qué fracción del cloro libre es HOCl a pH 7.2 y 7.8?",
        "fraction of free chlorine as hypochlorous acid at pH",
        "hocl_fraction_formula",
    ),
    (
        "El ORP y el DPD no coinciden, ¿a cuál hago caso?",
        "monopersulfate interference with DPD chlorine test",
        "mps_test_interference",
    ),
    (
        "¿Por qué se me ha disparado el estabilizador?",
        "cyanuric acid accumulation from stabilized chlorine feeder",
        "cya_accumulation_projection",
    ),
]

_SEED_RE = re.compile(
    r"--- Seed \d+ \(score: (?P<score>[\d.]+)\)(?P<off>\s*\[off-intent[^\]]*\])?[^\n]*\n"
    r"ID: (?P<id>\S+)"
)


def _seeds(texto: str) -> list[tuple[str, float, bool]]:
    return [
        (m.group("id"), float(m.group("score")), bool(m.group("off")))
        for m in _SEED_RE.finditer(texto)
    ]


def main() -> int:
    fallos = 0

    for pregunta, consulta, esperado in CASOS:
        begin_tool_scope(f"cov-{esperado}")
        salida = search_seed_nodes.invoke({"query": consulta})
        status = salida.split("\n", 1)[0].replace("STATUS: ", "")
        seeds = _seeds(salida)

        print(f"\n{pregunta}")
        print(f"  status: {status}")

        encontrado = next((s for s in seeds if s[0] == esperado), None)
        if encontrado is None:
            print(f"  ESPERADO NO RECUPERADO: {esperado}")
            print(f"  devueltos: {', '.join(s[0] for s in seeds[:5]) or '(ninguno)'}")
            fallos += 1
            continue

        _, score, off_intent = encontrado
        marca = "OFF-INTENT" if off_intent else "on-intent"
        print(f"  {esperado}  score={score:.3f}  {marca}")
        if off_intent:
            print("  -> su label no está en INTENT_LABELS: el modelo lo recibe "
                  "como contexto, no como respuesta")
            fallos += 1
        if status == "WEAK":
            print("  -> STATUS WEAK: al agente se le dice que pare y reporte "
                  "evidencia insuficiente")
            fallos += 1

    print(f"\n{'=' * 60}")
    if fallos:
        print(f"{fallos} problema(s) de cobertura. Revisar INTENT_LABELS en "
              f"src/agent/tools.py antes de añadir contenido nuevo al grafo.")
    else:
        print(f"Los {len(CASOS)} casos se recuperan correctamente.")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
