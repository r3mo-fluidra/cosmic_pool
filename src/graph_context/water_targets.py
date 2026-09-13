"""
La tabla canónica de bandas publicadas, y la regla de que una banda no es un
límite.

POR QUÉ ESTE MÓDULO
-------------------
`regulatory_limit` venía siendo una tirada. La causa no es que el retrieval a
veces falle: es que **el límite que el modelo dice estar citando no existe en
el corpus**. Comprobado contra el grafo real:

    MATCH (n) WHERE any(k IN keys(n) WHERE k IN
      ['min_value','max_value','minimum','maximum','limit','threshold'])
    -> 0 filas

Cero nodos con cotas estructuradas. Los números viven en prosa, y cada uno
llega envuelto en su propia negativa a ser normativo:

    free_chlorine      "typical educational targets… approximately 1 to 4 ppm"
    ph                 "Typical educational targets run from about 7.2 to 7.8"
    cyanuric_acid      "about 20 to 50 ppm outdoors against a commonly cited
                        90 ppm maximum"
    ch11-f28           "Typical educational targets only. Your local code
                        controls and is often more demanding."

Y el trace 6660e14f encaja pieza por pieza: pH con `regulatory_limit` 7.8,
cianúrico con 90, cloro libre con 2. Los tres son extremos de la banda
educativa de arriba, reetiquetados como techo de código. El fallo no era que
el modelo inventara cifras — las cifras son del corpus. Era que les cambiaba
la etiqueta.

QUÉ HACE ESTE MÓDULO, Y QUÉ NO
------------------------------
Canoniza lo que el corpus SÍ publica —la banda educativa, con su procedencia—
y degrada lo que no puede sostener. NO trae una tabla de límites normativos:
escribirla sería convertir la tirada en una constante, citada con seguridad en
todos los turnos, que `coherent_status` ya no podría degradar. Un operador que
repite "7.8 es el máximo de código" ante un inspector de una jurisdicción que
pone 8.0 reporta una infracción que no existe, y se la habría dictado este
sistema.

El límite real es el código local, y esa sigue siendo la pieza que falta. La
diferencia es que ahora el sistema lo dice en vez de rellenarlo.

MANTENIMIENTO
-------------
Cada fila es transcripción literal de un nodo del grafo. `verbatim` guarda la
frase completa, con su hedging intacto, para que una revisión pueda comparar
sin volver a consultar. Al cambiar el corpus se regenera con:

    .venv/bin/python -m src.graph_context.water_targets --check
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TargetBand:
    """
    Lo que el corpus publica para un parámetro. Nunca un límite normativo.

    `low`/`high` son la banda educativa completa; `preferred_*` la banda
    estrecha cuando el corpus la distingue. `figures` es el conjunto de cifras
    publicadas para este parámetro: si un `regulatory_limit` coincide con una
    de ellas, es esta banda reetiquetada.
    """
    parameter: str
    unit: str
    low: float | None
    high: float | None
    preferred_low: float | None = None
    preferred_high: float | None = None
    #: Cifras adicionales que el corpus publica y que no son extremos de la
    #: banda (mínimos condicionales, niveles de acción, techos citados).
    extra_figures: tuple[float, ...] = ()
    #: False cuando rellenar `operating_target` desde esta banda sería unsafe.
    fill_target: bool = True
    fill_target_reason: str = ""
    node_id: str = ""
    source: str = ""
    verbatim: str = ""

    @property
    def figures(self) -> frozenset[float]:
        vals = [self.low, self.high, self.preferred_low, self.preferred_high]
        return frozenset(
            [v for v in vals if v is not None] + list(self.extra_figures)
        )

    def target_text(self) -> str | None:
        """La banda como texto, para `operating_target`. None si no se rellena."""
        if not self.fill_target:
            return None
        lo, hi = (self.preferred_low, self.preferred_high)
        if lo is None or hi is None:
            lo, hi = self.low, self.high
        if lo is None or hi is None:
            return None
        return f"{_num(lo)}–{_num(hi)}"


def _num(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v}"


# ---------------------------------------------------------------------------
# La tabla. Una fila por parámetro, transcrita del nodo que la publica.
# ---------------------------------------------------------------------------

WATER_TARGETS: dict[str, TargetBand] = {
    "free chlorine": TargetBand(
        parameter="free chlorine", unit="ppm",
        low=1.0, high=4.0, extra_figures=(2.0, 3.0, 5.0, 10.0),
        # El objetivo de cloro libre NO se rellena desde la banda. Con
        # estabilizador presente el número que sirve escala con el CYA
        # (fc_cya_proportional_target: ~7.5% del cianúrico medido), y el propio
        # nodo del grafo advierte que dar la cifra en rango por su cuenta es
        # unsafe: "it reads as sufficient and it is not". Con CYA 90 la banda
        # diría 1–4 ppm y el operador reabriría con el agua sin desinfectar.
        fill_target=False,
        fill_target_reason="el objetivo escala con el cianúrico; ver fc_cya_proportional_target",
        node_id="free_chlorine",
        source='CH19-19.1";CH02-2.2;CH11-11.6-free-chlorine-fc;CH31-31.3.1',
        verbatim=(
            "Typical educational targets are approximately 1 to 4 ppm in pools "
            "with a commonly cited minimum of 1 ppm, a higher minimum often 2 "
            "ppm where chlorinated isocyanurates are in use, and approximately "
            "3 to 5 ppm in spas with a commonly cited 3 ppm minimum; maxima are "
            "commonly capped near 10 ppm."
        ),
    ),
    "combined chlorine": TargetBand(
        parameter="combined chlorine", unit="ppm",
        low=None, high=0.4, extra_figures=(0.2,),
        fill_target=False,
        fill_target_reason="el corpus da un nivel de acción, no una banda a la que apuntar",
        node_id="combined_chlorine",
        source='CH19-19.2";CH02-2.2;CH11-11.7-combined-chlorine-cc',
        verbatim=(
            "The target is as low as achievable, with a commonly cited "
            "corrective-action level near 0.4 ppm."
        ),
    ),
    "ph": TargetBand(
        parameter="ph", unit="",
        low=7.2, high=7.8, preferred_low=7.4, preferred_high=7.6,
        node_id="ph",
        source='CH19-19.1";CH11-11.8-ph;CH24-24.7.4.4;CH26-26.2;CH31-31.3.1',
        verbatim=(
            "Typical educational targets run from about 7.2 to 7.8 with a "
            "preferred operating band of about 7.4 to 7.6; operating at the top "
            "of the band sacrifices roughly a third of disinfection capacity "
            "while still logging in range."
        ),
    ),
    "cyanuric acid": TargetBand(
        parameter="cyanuric acid", unit="ppm",
        low=20.0, high=50.0, extra_figures=(0.0, 90.0),
        node_id="cyanuric_acid",
        source='CH19-19.1";CH13-13.2-what-cyanuric-acid-is-and-what-it-does;CH31-31.4',
        verbatim=(
            "Typical educational targets are about 20 to 50 ppm outdoors "
            "against a commonly cited 90 ppm maximum, zero indoors, and it is "
            "commonly prohibited in spas and increased-risk venues."
        ),
    ),
    "total alkalinity": TargetBand(
        parameter="total alkalinity", unit="ppm",
        low=60.0, high=180.0, preferred_low=80.0, preferred_high=120.0,
        node_id="total_alkalinity",
        source='CH19-19.1";CH02-2.2;CH11-11.9-total-alkalinity-ta;CH26-26.2',
        verbatim=(
            "Typical educational targets run from about 60 to 180 ppm with a "
            "preferred band of about 80 to 120 ppm, run higher where an acidic "
            "chlorine source is used and lower with basic sources."
        ),
    ),
    "calcium hardness": TargetBand(
        parameter="calcium hardness", unit="ppm",
        low=150.0, high=400.0,
        node_id="calcium_hardness",
        source='CH19-19.1";CH11-11.10-calcium-hardness-ch;CH26-26.2',
        verbatim=(
            "Typical educational targets run from about 150 to 400 ppm, with "
            "the lower end acceptable for vinyl and fiberglass vessels that "
            "have no cementitious material to protect and some codes permitting "
            "higher maxima."
        ),
    ),
    "orp": TargetBand(
        parameter="orp", unit="mV",
        low=650.0, high=750.0,
        node_id="oxidation_reduction_potential",
        source="CH02-2.4;CH11-11.5-a-note-on-automated-controllers-and-orp;CH20-20.3.1",
        verbatim=(
            "Oxidation-Reduction Potential measures the oxidizing activity of "
            "the water in millivolts and is the quantity most automated "
            "controllers actually sense, with a typical educational target "
            "range of roughly 650 to 750 mV."
        ),
    ),
    "salt": TargetBand(
        parameter="salt", unit="ppm",
        low=2700.0, high=3400.0,
        fill_target=False,
        fill_target_reason="depende de la especificación del fabricante del generador",
        node_id="salt_concentration",
        source="CH14-14.3.2-operator-parameters-to-monitor",
        verbatim=(
            "Conventional generators operate at typical ranges of 2,700–3,400 "
            "ppm; low-salt designs operate substantially lower, per "
            "manufacturer specification."
        ),
    ),
    # TDS entra sin banda A PROPÓSITO: el corpus dice explícitamente que la
    # acción se juzga contra una línea base registrada y "not at any absolute
    # figure". Una fila con low/high aquí inventaría el absoluto que la fuente
    # niega. Está presente para que el parámetro se reconozca y no se rellene.
    "tds": TargetBand(
        parameter="tds", unit="ppm",
        low=None, high=None,
        fill_target=False,
        fill_target_reason="el corpus juzga contra una línea base, no contra una cifra absoluta",
        node_id="total_dissolved_solids",
        source='CH19-19.6";CH02-2.5;CH11-11.12-total-dissolved-solids-tds',
        verbatim=(
            "Action is commonly considered when the value rises roughly 1,000 "
            "to 1,500 ppm above a recorded startup baseline rather than at any "
            "absolute figure, because a vessel served by a salt chlorine "
            "generator operates intentionally at several thousand ppm."
        ),
    ),
}

#: Siglas y grafías con las que el especialista alterna. Mismo criterio que
#: `_UNIDAD` en response_validator: el nombre llega sin normalizar.
_ALIAS = {
    "fc": "free chlorine", "free cl": "free chlorine",
    "cc": "combined chlorine", "combined cl": "combined chlorine",
    "chloramines": "combined chlorine",
    "ta": "total alkalinity", "alkalinity": "total alkalinity",
    "ch": "calcium hardness", "hardness": "calcium hardness",
    "cya": "cyanuric acid", "stabilizer": "cyanuric acid",
    "conditioner": "cyanuric acid", "isocyanuric acid": "cyanuric acid",
    "oxidation reduction potential": "orp", "redox": "orp",
    "salt concentration": "salt", "salinity": "salt",
    "total dissolved solids": "tds",
}


def _canon(parameter: str) -> str:
    clave = re.sub(r"[^a-z0-9]+", " ", (parameter or "").lower()).strip()
    return _ALIAS.get(clave, clave)


def target_band(parameter: str) -> TargetBand | None:
    """La banda publicada para un parámetro, o None si el corpus no la trae."""
    return WATER_TARGETS.get(_canon(parameter))


#: Tolerancia de comparación. El especialista devuelve 7.8 y 90.0 tal cual,
#: pero un float que pasó por JSON puede llegar como 7.800000000000001.
_EPS = 1e-6


def is_published_figure(parameter: str, value) -> bool:
    """
    ¿Este número es una cifra que el corpus publica como educativa?

    Si lo es, un `regulatory_limit` que lo contenga no es una cita de código:
    es esta banda con otra etiqueta. Es la comprobación que separa "el
    especialista recuperó un límite" de "el especialista reetiquetó un target",
    y hoy —con cero cotas normativas en el grafo— siempre es lo segundo.
    """
    banda = target_band(parameter)
    if banda is None or value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return any(abs(v - f) < _EPS for f in banda.figures)


# ---------------------------------------------------------------------------
# Verificación de procedencia
# ---------------------------------------------------------------------------

def _check() -> int:
    """
    Contrasta cada `verbatim` con el nodo del grafo del que se transcribió.

    Existe porque la tabla es una copia: sin esto, el corpus cambia y la copia
    se queda diciendo lo de antes en silencio, que es peor que no tenerla —
    una cifra con procedencia falsa se defiende sola en una revisión.

    No falla si Neo4j no está configurado: informa y sale. Es una herramienta
    de mantenimiento, no un test de arranque.
    """
    import os
    try:
        from dotenv import load_dotenv
        from neo4j import GraphDatabase
    except ImportError:
        print("neo4j/dotenv no instalados; nada que comprobar")
        return 0

    load_dotenv()
    uri = os.getenv("NEO4J_URI")
    if not uri:
        print("NEO4J_URI no configurado; nada que comprobar")
        return 0

    driver = GraphDatabase.driver(
        uri, auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD"))
    )
    desviaciones = 0
    with driver.session() as s:
        for clave, banda in WATER_TARGETS.items():
            rec = s.run(
                "MATCH (n {id:$i}) RETURN n.description AS d, n.source_reference AS s",
                i=banda.node_id,
            ).single()
            if rec is None:
                print(f"✗ {clave}: el nodo '{banda.node_id}' ya no existe")
                desviaciones += 1
                continue
            desc = " ".join((rec["d"] or "").split())
            if banda.verbatim not in desc:
                print(f"✗ {clave}: `verbatim` ya no aparece en {banda.node_id}")
                print(f"    tabla:  {banda.verbatim[:120]}…")
                print(f"    grafo:  {desc[:120]}…")
                desviaciones += 1
            else:
                print(f"✓ {clave}  ({banda.node_id})")
    driver.close()

    print(f"\n{len(WATER_TARGETS) - desviaciones}/{len(WATER_TARGETS)} filas verificadas")
    return 1 if desviaciones else 0


if __name__ == "__main__":
    import sys
    sys.exit(_check() if "--check" in sys.argv else
             print(__doc__) or 0)
