"""
response_validator.py

Enforcement determinística del contrato de respuesta emitido por el synthesizer.

El prompt le PIDE al LLM que respete el presupuesto; este módulo lo GARANTIZA.
Principio rector: nunca borrar información, solo reubicarla a `details`.
La única excepción es el recorte de acciones malformadas (>12 palabras),
que se reubican también en lugar de descartarse.

Uso:
    payload, report = enforce_contract(payload, contract, state["assigned_agents"])
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Union

from .water_targets import (
    closure_required, is_published_figure, resolve_panel, target_band,
)

# --------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------

MAX_ACTIONS = 4
MAX_ACTION_WORDS = 12
NO_CAP = 9999  # presupuesto centinela: arquetipo `critical`, sin techo

# Agentes cuyo contenido implica manejo de producto químico o riesgo directo.
# Resuelven `safety_required = "conditional"` a True.
HAZARD_AGENTS = {"chemistry", "contamination", "safety", "math", "recovery"}

# Fallback léxico por si el arquetipo es `calculation` sin agente peligroso
# pero el contenido igual describe manipulación de producto.
HAZARD_PATTERN = re.compile(
    r"\b(ácido|acido|acid|cloro|chlorine|hipoclorito|hypochlorite|muriático|muriatic|"
    r"tricloro|dicloro|bromo|bromine|ozono|ozone|soda\s+cáustica|caustic|"
    r"peróxido|peroxide|alguicida|algaecide|EPP|PPE)\b",
    re.IGNORECASE,
)

# Etiquetas usadas al reubicar contenido excedente.
OVERFLOW_LABEL = "Next actions (overflow)"
SAFETY_LABEL_PATTERN = re.compile(
    r"safety|warning|hazard",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Reporte de validación (alimenta las métricas del paso 8)
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
    #: El contrato exigía `safety` y no hubo nada que poner. YA NO gatilla
    #: retry: desde que la línea se deriva del payload del especialista
    #: (`render_safety`), el caso real que lo disparaba —modelo que la omite
    #: teniendo el dato— se resuelve sin volver a llamar al modelo. Se sigue
    #: midiendo: si esto viene lleno, es que el payload tampoco la sostenía.
    safety_missing: bool = False
    answer_exceeds_budget: bool = False  # -> gatilla retry
    #: `actions` / `safety` construidas por código desde el payload del
    #: especialista en vez de aceptadas tal como las escribió el modelo.
    actions_rendered: bool = False
    safety_rendered: bool = False
    #: Parámetros cuyo `regulatory_limit` resultó ser una banda educativa
    #: reetiquetada. Es LA métrica del paso 7: mientras venga llena, el
    #: especialista sigue presentando targets como código, y lo único que
    #: cambia es que ahora se detecta. Si alguna vez se vacía sin que caiga
    #: `targets_filled`, es que el retrieval empezó a traer citas de verdad.
    limits_demoted: list[str] = field(default_factory=list)
    #: Parámetros a los que se les rellenó `operating_target` desde la tabla.
    targets_filled: list[str] = field(default_factory=list)
    #: Lecturas cuyo `status` no coincidía con el que sale de la banda, en
    #: formato "pH: above_maximum→in_range". Es la medida directa de la
    #: oscilación: el especialista juzgando los mismos números de dos maneras
    #: en dos corridas. El panel ya no la refleja, pero el turno la produjo.
    status_corrected: list[str] = field(default_factory=list)
    #: El panel obliga a cerrar y la acción se insertó por código.
    closure_inserted: bool = False
    #: Lecturas fuera de rango que el tier visible omitió. El modelo lleva
    #: tres iteraciones incumpliendo esto con el presupuesto casi vacío, así
    #: que deja de ser una instrucción y pasa a ser una comprobación.
    readings_missing: list[str] = field(default_factory=list)
    readings_appended: bool = False
    #: Cantidades del tier visible que no aparecen en el material de origen.
    #: El peor modo de fallo según el propio prompt del synthesizer.
    unsupported_numbers: list[str] = field(default_factory=list)
    #: `safety` repite una acción en lugar de aportar algo nuevo. Se mide, no
    #: se corrige: borrar una línea de seguridad es peor que repetirla.
    safety_duplicates_action: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def needs_retry(self) -> bool:
        """
        Un solo retry. Si vuelve a fallar, se acepta la degradación.

        `safety_missing` salió de acá. Era el gatillo que más disparaba —siete
        rondas seguidas— y cada disparo cuesta una llamada completa al modelo,
        2–3.5 s de latencia pura por turno, para pedir una línea que ahora se
        arma por plantilla desde el payload. Cuando `render_safety` devuelve
        None es porque el payload no sostiene ninguna advertencia específica, y
        volver a preguntar no cambia el dato: produce una genérica.
        """
        return self.answer_exceeds_budget

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Lecturas obligatorias en el tier visible
# --------------------------------------------------------------------------

#: El fraseo de cada estado, por plantilla. NO lo redacta el modelo.
#:
#: Cuatro rondas de evaluación sobre la misma consulta mostraron que como
#: instrucción de prompt no se sostiene. Los fallos observados, todos sobre
#: datos que el especialista había clasificado bien:
#:   - "extremely high" sobre un at_ceiling (intensificación)
#:   - un in_range convertido en "above the maximum" (recalificación)
#:   - "above the 120 ppm operating ceiling" sobre una entrada sin límite
#:     (un objetivo operativo presentado como techo normativo)
#: Los tres llegan igual de lejos: un operador que los repite ante un
#: inspector reporta un incumplimiento que no existe.
#:
#: `{limit}` se rellena con `regulatory_limit`. Las plantillas que lo usan
#: solo se eligen cuando ese campo tiene valor: un estado de violación sin
#: límite que lo respalde se degrada antes de llegar aquí (coherent_status).
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
        "at_floor":      "cumple, sin margen sobre el mínimo",
        "in_range":      "en rango",
        "at_ceiling":    "cumple, sin margen bajo el máximo",
        "above_maximum": "en infracción, por encima del máximo de {limit}",
    },
}

#: El mismo veredicto cuando la cifra que lo respalda es una banda EDUCATIVA,
#: no un límite de código. Ver water_targets: el corpus no publica cotas
#: normativas, así que un `regulatory_limit` que coincide con un extremo de la
#: banda publicada es esa banda reetiquetada, y se degrada a esto.
#:
#: La degradación no borra el hallazgo — el valor sigue estando fuera del rango
#: publicado y eso es accionable — le quita la autoridad que no tiene. La
#: diferencia entre "infracción" y "fuera del rango habitual" es la diferencia
#: entre lo que el sistema sabe y lo que se estaba atribuyendo.
_BANDA_PHRASING = {
    "en": {
        "below_minimum": "below the typical {bound} floor — educational range, not a code limit",
        "at_floor":      "at the floor of the typical range",
        "in_range":      "in range",
        "at_ceiling":    "at the top of the typical range",
        "above_maximum": "above the typical {bound} ceiling — educational range, not a code limit",
    },
    "es": {
        "below_minimum": "por debajo del piso habitual de {bound} — rango educativo, no límite de código",
        "at_floor":      "en el piso del rango habitual",
        "in_range":      "en rango",
        "at_ceiling":    "en el techo del rango habitual",
        "above_maximum": "por encima del techo habitual de {bound} — rango educativo, no límite de código",
    },
}

#: Para una lectura cuyo estado no se sostiene: se informa el valor y, si lo
#: hay, el objetivo, sin lenguaje de cumplimiento en ninguna dirección.
_SIN_VEREDICTO = {
    "en": "reported; no code bound available",
    "es": "reportado; sin límite normativo disponible",
}
_CON_OBJETIVO = {
    "en": "reported; operating target {target}",
    "es": "reportado; objetivo operativo {target}",
}

#: El objetivo operativo, cuando SÍ hay veredicto. El límite dice dónde empieza
#: la infracción; el objetivo, dónde hay que dejar el vaso. Un panel que solo
#: trae el límite deja al operador corrigiendo hasta el borde de la infracción.
#:
#: Va como sufijo y con su propia palabra ("target"/"objetivo") justamente para
#: que no se confunda con el número normativo: el último fallo observado fue un
#: operating_target presentado como techo de código.
_OBJETIVO = {
    "en": "; target {target}",
    "es": "; objetivo {target}",
}

#: Unidad por parámetro. El especialista emite `measured` como número pelado
#: —0.8, 7.9, 90.0— así que la unidad se pierde entre su payload y la pantalla:
#: en el trace 148acb15 los siete valores llegaron sin ppm. Una cifra sin
#: unidad no es una lectura, es un número, y 0.8 sin ppm no le dice a nadie
#: que la pileta está en infracción.
#:
#: La temperatura NO está en la tabla, a propósito. El especialista devuelve 82
#: sin escala, y elegirla acá es inventar: 82 °F y 82 °C describen dos piletas
#: que no tienen nada que ver. Sin unidad es peor que con la correcta, pero
#: mucho mejor que con la equivocada.
_UNIDAD = {
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
    # Siglas: el especialista alterna entre el nombre largo y la sigla.
    "fc": "ppm", "cc": "ppm", "tc": "ppm",
    "cya": "ppm", "ta": "ppm", "ch": "ppm",
}


def unit_for(parameter: str) -> str:
    """La unidad de un parámetro, o "" si no la conocemos o no la lleva (pH)."""
    clave = re.sub(r"[^a-z0-9]+", " ", (parameter or "").lower()).strip()
    return _UNIDAD.get(clave, "")


#: Números que no hace falta respaldar: son lenguaje, no cantidades.
#: Rangos horarios, ordinales y unidades sueltas aparecen en prosa normal.
_NUMERO_RE = re.compile(r"\d+(?:[.,]\d+)?")
_TRIVIALES = frozenset({"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                        "12", "24", "48", "72", "100"})


def unsupported_numbers(payload, raw_content: str) -> list[str]:
    """
    Cantidades que aparecen en el tier visible y no en el material de origen.

    El SYNTHESIZER_PROMPT llama a esto el peor modo de fallo del sistema
    ("never invent a dosage... filling a gap to satisfy a shape"), y aun así
    ocurrió: con el `calculation_request` sin ejecutar, el synthesizer escribió
    "drena y rellena entre un treinta y un cuarenta por ciento". El número no
    estaba en ningún sitio del payload, y además era erróneo — ninguna de las
    dos fracciones alcanzaba el objetivo de estabilizador que el propio agente
    había pedido.

    Un número inventado es peor que un dato ausente: llega con la misma
    confianza que los verdaderos y el operador no tiene forma de distinguirlos.

    Detecta, no corrige. Reescribir la frase que lo contiene exigiría entender
    la frase; lo que se puede afirmar sin ambigüedad es que ese número no tiene
    respaldo, y eso basta para decidir un reintento y para medir la frecuencia
    en Langfuse.
    """
    if not raw_content:
        return []

    def _canon(x: str) -> str:
        """3.0 y 3 son el mismo número; 90 y 9 no.

        `rstrip("0")` a secas convertía "90" en "9", así que un 90 del panel
        no casaba con el 90.0 del origen y se denunciaba como inventado. Solo
        se recortan ceros cuando hay parte decimal que recortar.
        """
        x = x.replace(",", ".")
        return x.rstrip("0").rstrip(".") if "." in x else x

    numeros_origen = {_canon(o) for o in _NUMERO_RE.findall(raw_content)}

    # Solo `answer`, `actions` y `safety`: lo que el MODELO escribe. El campo
    # `readings` lo construye este módulo desde el propio payload del
    # especialista, así que sus cifras vienen del origen por definición y
    # contarlas solo produciría ruido.
    escrito_por_el_modelo = " ".join([
        getattr(payload, "answer", "") or "",
        getattr(payload, "safety", "") or "",
        *(getattr(payload, "actions", None) or []),
    ])

    # "thirty to forty percent" no lleva dígitos: los números en palabras se
    # escapan de este chequeo, y es una limitación consciente. Cubre el caso
    # frecuente (cifras) sin arriesgar falsos positivos con un parser de
    # numerales en dos idiomas.
    sospechosos = []
    for n in _NUMERO_RE.findall(escrito_por_el_modelo):
        if n in _TRIVIALES or _canon(n) in numeros_origen:
            continue
        sospechosos.append(n)

    return sospechosos


#: Palabras vacías mínimas, es/en. Sin dependencias externas: solo hace falta
#: que "the/de/la/to" no inflen el parecido entre dos frases cortas.
_VACIAS = frozenset({
    "el", "la", "los", "las", "un", "una", "de", "del", "a", "al", "en", "y",
    "o", "que", "no", "se", "su", "hasta", "para", "con", "por",
    "the", "a", "an", "of", "to", "in", "and", "or", "not", "your", "until",
    "is", "are", "be", "all",
})
_PALABRA_RE = re.compile(r"[a-záéíóúñü]+", re.IGNORECASE)
_DUPLICADO_UMBRAL = 0.55
#: Truncado a raíz, no lematización. "cerrada"/"cerrar" y "closed"/"close"
#: describen la misma acción y sin esto no se parecen en nada: la comparación
#: literal daba 0.5 y 0.4 sobre un caso que es una repetición evidente.
#: Cinco caracteres es lo bastante corto para absorber la flexión de los dos
#: idiomas y lo bastante largo para no confundir palabras distintas.
_RAIZ = 5


def _contenido(texto: str) -> set[str]:
    return {w.lower()[:_RAIZ] for w in _PALABRA_RE.findall(texto or "")
            if w.lower() not in _VACIAS and len(w) > 2}


def safety_repeats_an_action(payload) -> bool:
    """
    ¿`safety` no dice más que una acción que ya está en la lista?

    Medido: bajo una acción "cierra la pileta a los bañistas", el campo de
    seguridad decía "mantén la pileta cerrada hasta restaurar el cloro". La
    línea que más se lee del tier visible gastada en repetir la primera
    acción, y la que sí llevaba información única — la prohibición del
    producto que causó el problema — desaparecida.

    Se mide y no se corrige. Suprimir una línea de seguridad por parecerse a
    otra cosa es un fallo mucho peor que dejarla repetida, y no hay forma de
    distinguir por solapamiento léxico una repetición de un refuerzo
    deliberado. El prompt pide la línea con más alcance; esto dice cuántas
    veces no la da.
    """
    safety = getattr(payload, "safety", None)
    acciones = getattr(payload, "actions", None) or []
    if not safety or not acciones:
        return False

    tokens = _contenido(safety)
    if not tokens:
        return False

    return any(
        len(tokens & _contenido(a)) / len(tokens) >= _DUPLICADO_UMBRAL
        for a in acciones
    )


def _visible_text(payload) -> str:
    partes = [getattr(payload, "answer", "") or "", getattr(payload, "safety", "") or ""]
    partes += list(getattr(payload, "actions", None) or [])
    # `readings` también es tier 1: una lectura que el synthesizer ya puso ahí
    # está reportada, y volver a añadirla la duplicaría.
    for r in (getattr(payload, "readings", None) or []):
        partes += [getattr(r, "parameter", "") or "", getattr(r, "measured", "") or "",
                   getattr(r, "note", "") or ""]
    return " ".join(partes).lower()


def _reading_is_visible(reading: dict, visible: str) -> bool:
    """
    ¿Está esta lectura REPORTADA en el tier visible, no solo nombrada?

    La señal es el VALOR MEDIDO, no el nombre del parámetro. Nombrarlo dentro
    de una acción ("baja el pH") no cuenta: el operador no se entera de cuánto
    marcó ni de que incumple. El valor es lo que convierte una tarea en un dato.

    Y el nombre no sirve para decidirlo: el especialista emite sus parámetros
    en inglés ("Free Chlorine") y el turno puede estar en español, así que
    buscar el nombre daba ausente en TODAS las lecturas de cualquier turno en
    español — y el validador las duplicaba todas. Las cifras no se traducen.

    El coste es un falso positivo posible: si el mismo número aparece en el
    texto por otro motivo, la lectura se da por reportada. Prefiero eso a
    duplicar, porque el synthesizer sí tiene instrucciones de poblar
    `readings` y este chequeo es la red, no la vía principal.
    """
    medido = reading.get("measured")
    if medido is None:
        # Sin valor no hay nada que comprobar ni que añadir.
        return True

    crudo = f"{medido}"
    normalizado = crudo.rstrip("0").rstrip(".") if "." in crudo else crudo
    return crudo in visible or normalizado in visible


#: Estados que afirman un incumplimiento. Solo son sostenibles si hay un
#: límite normativo contra el que medirlos.
_VIOLATION_STATUSES = ("below_minimum", "above_maximum")


def coherent_status(reading: dict) -> str | None:
    """
    El estado de una lectura, degradado si se contradice a sí mismo.

    Un `above_maximum` con `regulatory_limit` en null afirma que se superó un
    máximo que la propia entrada dice no conocer. Medido: el especialista
    devolvió la alcalinidad total como above_maximum sin límite, juzgándola
    contra su operating_target — y el parámetro no solo cumplía, estaba
    prácticamente en objetivo. Declarar infracción a un parámetro sano es el
    mismo error que llamar violación a un techo, y llega igual de lejos: un
    operador que lo repite ante un inspector reporta un incumplimiento que no
    existe.

    Se degrada a None en vez de a "in_range": no sabemos que esté en rango,
    sabemos que no podemos afirmar lo contrario. None deja la lectura fuera de
    las obligatorias y fuera del texto que este módulo genera, que es la
    conducta segura cuando el dato se contradice.
    """
    status = reading.get("status")
    if status in _VIOLATION_STATUSES and reading.get("regulatory_limit") is None:
        return None
    return status


def reconcile_with_published_bands(test_interpretation, report=None) -> list[dict]:
    """
    Reemplaza `status` y `regulatory_limit` por lo que la tabla sostiene.

    DOS COSAS, Y LA SEGUNDA ES LA QUE CORTA LA OSCILACIÓN.

    1. El límite. El corpus no publica ninguno —cero nodos con cotas, ver
       water_targets— así que todo `regulatory_limit` sobre un parámetro
       cubierto es una banda educativa reetiquetada y se retira. En el trace
       6660e14f las tres cifras (pH 7.8, cianúrico 90, cloro 2) son extremos
       publicados; ninguna era una cita.

    2. El estado. Se DERIVA del valor medido contra la banda, y lo que dijera
       el especialista se descarta. La versión anterior de este paso sustituía
       el límite y dejaba el status al modelo, con lo que una alcalinidad de
       130 etiquetada `above_maximum` se renderizaba "por encima del techo de
       180" — con 130 < 180. El mismo panel daba veredictos distintos en
       corridas distintas sobre los mismos números, y esa es la oscilación.

    ASIMETRÍA DELIBERADA: una jurisdicción real puede poner su máximo de pH
    justo en 7.8, y en ese caso esto retira un límite verdadero. Se acepta. El
    sistema no puede distinguir una coincidencia de un reetiquetado, y las dos
    equivocaciones no cuestan lo mismo: no afirmar un código que existe deja al
    operador con el número y sin la etiqueta; afirmar uno que no existe le hace
    reportar una infracción inventada ante un inspector.

    Un parámetro que la tabla NO cubre se deja intacto: no hay base para
    retirarle el límite ni para juzgarlo, y declararlo en rango por
    desconocimiento sería un alta que nadie emitió.

    También rellena `operating_target` cuando el especialista no lo dio y la
    banda sirve de objetivo. Cloro libre queda fuera a propósito: su objetivo
    escala con el cianúrico y dar la banda suelta es unsafe.

    Devuelve una lista nueva; no muta los dicts del caller.
    """
    if not isinstance(test_interpretation, list):
        return []

    resueltos = resolve_panel(test_interpretation)
    salida = []
    for r, (status, cota) in zip(test_interpretation, resueltos):
        if not isinstance(r, dict):
            continue
        r = dict(r)
        nombre = str(r.get("parameter", ""))
        banda = target_band(nombre)

        if banda is not None:
            if r.get("regulatory_limit") is not None and report is not None:
                report.limits_demoted.append(_format_parameter(nombre))
            r["regulatory_limit"] = None

            if status is not None:
                if report is not None and status != r.get("status"):
                    report.status_corrected.append(
                        f"{_format_parameter(nombre)}: {r.get('status')}→{status}"
                    )
                r["status"] = status
                r["band_derived"] = True
                r["educational_bound"] = cota

            if r.get("operating_target") is None:
                objetivo = banda.target_text()
                if objetivo:
                    r["operating_target"] = objetivo
                    if report is not None:
                        report.targets_filled.append(_format_parameter(nombre))

        salida.append(r)
    return salida


def required_readings(test_interpretation) -> list[dict]:
    """
    Las lecturas que el tier visible NO puede omitir.

    Filtra también las incoherentes: una violación sin límite que la respalde
    no se reporta como violación.
    """
    if not isinstance(test_interpretation, list):
        return []
    return [
        r for r in test_interpretation
        if isinstance(r, dict) and coherent_status(r) not in (None, "in_range")
    ]


def _format_number(valor) -> str:
    """
    Sin ceros de relleno: 2.0 -> '2', 7.8 -> '7.8'.

    Una cadena se devuelve intacta: si el especialista escribió "0.8 ppm", esa
    unidad es suya y vale más que el número pelado.
    """
    if isinstance(valor, str):
        return valor.strip()
    if isinstance(valor, float) and valor.is_integer():
        return str(int(valor))
    return f"{valor}"


def _con_unidad(valor, unidad: str) -> str:
    """El número con su unidad, sin duplicarla si el especialista ya la puso."""
    texto = _format_number(valor)
    if not unidad or not texto:
        return texto
    if any(c.isalpha() for c in texto):
        # "0.8 ppm" ya viene completo: añadir la unidad daría "0.8 ppm ppm".
        return texto
    return f"{texto} {unidad}"


#: Parámetros cuya grafía no sobrevive a un .title(). "ph" -> "Ph" es un
#: nombre que ningún operador escribe.
_GRAFIA = {
    "ph": "pH", "orp": "ORP", "tds": "TDS", "cya": "CYA", "lsi": "LSI",
    "fc": "FC", "cc": "CC", "ta": "TA", "ch": "CH", "ppm": "ppm",
}


def _format_parameter(nombre: str) -> str:
    limpio = nombre.replace("_", " ").strip()
    if not limpio:
        return limpio
    if limpio.lower() in _GRAFIA:
        return _GRAFIA[limpio.lower()]
    if not limpio.islower():
        # Ya viene con mayúsculas: el especialista eligió su grafía.
        return limpio
    return " ".join(_GRAFIA.get(w, w.capitalize()) for w in limpio.split())


def reading_note(reading: dict, language: str, unit: str | None = None) -> str:
    """
    La nota de una lectura, armada por plantilla desde sus propios campos.

    El modelo no interviene: elige la frase el `status`, y el número que la
    acompaña sale de `regulatory_limit`, nunca de `operating_target`. Mezclar
    los dos fue el último fallo observado — "above the 120 ppm operating
    ceiling" presenta un objetivo de industria como si fuera un techo de
    código.

    El objetivo sí aparece, pero detrás y con su propia etiqueta: es el número
    sobre el que el operador actúa, y sin él la nota dice dónde empieza la
    infracción sin decir dónde hay que dejar el vaso.
    """
    # `unit=None` significa "deducila", no "sin unidad": un caller que se
    # olvide del argumento produce una línea a medias —el valor con ppm y el
    # límite sin— y ese es justo el defecto que este paso viene a cerrar.
    if unit is None:
        unit = unit_for(str(reading.get("parameter", "")))

    idioma = _STATUS_PHRASING.get(language, _STATUS_PHRASING["es"])
    status = coherent_status(reading)
    limite = reading.get("regulatory_limit")
    objetivo = reading.get("operating_target")

    # Un límite degradado por `reconcile_with_published_bands` deja aquí su
    # cifra. El veredicto se emite igual —el valor está fuera del rango
    # publicado— pero con el fraseo que no le atribuye código, y `status` se
    # lee del original: sin `regulatory_limit`, coherent_status lo habría
    # anulado y la lectura saldría sin veredicto teniendo con qué darlo.
    cota = reading.get("educational_bound")
    if reading.get("band_derived"):
        banda = _BANDA_PHRASING.get(language, _BANDA_PHRASING["es"])
        crudo = reading.get("status")
        if crudo in banda:
            nota = banda[crudo].format(
                bound=_con_unidad(cota, unit) if cota is not None else "")
            if objetivo is not None and crudo != "in_range":
                nota += _OBJETIVO.get(language, _OBJETIVO["es"]).format(
                    target=_con_unidad(objetivo, unit))
            return nota

    if status is None or status not in idioma:
        if objetivo is not None:
            return _CON_OBJETIVO.get(language, _CON_OBJETIVO["es"]).format(
                target=_con_unidad(objetivo, unit))
        return _SIN_VEREDICTO.get(language, _SIN_VEREDICTO["es"])

    plantilla = idioma[status]
    if "{limit}" in plantilla:
        if limite is None:
            # No debería ocurrir — coherent_status ya degrada esos casos —
            # pero una plantilla con un hueco sin rellenar es peor que una
            # frase sin cifra.
            return _SIN_VEREDICTO.get(language, _SIN_VEREDICTO["es"])
        nota = plantilla.format(limit=_con_unidad(limite, unit))
    else:
        nota = plantilla

    # En `in_range` no se da: no hay nada que corregir, y un objetivo colgado
    # de una lectura sana se lee como una tarea pendiente que no existe.
    if objetivo is not None and status != "in_range":
        nota += _OBJETIVO.get(language, _OBJETIVO["es"]).format(
            target=_con_unidad(objetivo, unit))
    return nota


def build_readings(test_interpretation: list[dict], language: str, linea_cls):
    """
    Todas las líneas del panel, construidas desde los datos del especialista.

    SUSTITUYE lo que el synthesizer haya escrito en `readings`; no lo
    completa. La versión anterior solo añadía lo que faltaba, y por eso una
    línea mal redactada por el modelo pasaba intacta mientras las ausentes se
    corregían — la mitad del problema arreglada y la otra mitad no.

    Se incluyen TODOS los parámetros reportados, también los que están en
    rango: el operador entregó siete lecturas y ver las siete es la
    confirmación de que se leyeron todas. Omitir las correctas obliga a
    deducir por ausencia.
    """
    lineas = []
    for r in test_interpretation or []:
        if not isinstance(r, dict):
            continue
        nombre = _format_parameter(str(r.get("parameter", "")))
        medido = r.get("measured")
        if not nombre or medido is None:
            continue
        # La unidad se resuelve del nombre CRUDO ("free_chlorine"), no del ya
        # formateado, y se aplica a las tres cifras de la línea: el valor, el
        # límite y el objetivo. Media línea con unidades es peor que ninguna.
        unidad = unit_for(str(r.get("parameter", "")))
        lineas.append(linea_cls(
            parameter=nombre,
            measured=_con_unidad(medido, unidad),
            note=reading_note(r, language, unidad),
        ))
    return lineas


def enforce_visible_readings(payload, readings: list[dict], language: str,
                             report: ValidationReport) -> None:
    """
    Sustituye `readings` por las líneas construidas desde los datos.

    No comprueba lo que escribió el modelo ni completa lo que falta: lo
    reemplaza. Cuatro rondas de evaluación sobre la misma consulta mostraron
    que el fraseo por estado no se sostiene como instrucción de prompt, y la
    versión que solo completaba dejaba pasar intacta una línea mal redactada
    mientras corregía las ausentes — media solución.

    Lo que el modelo sigue escribiendo es la prosa de arriba: el veredicto, el
    razonamiento, la causa. El panel de cifras se arma acá.
    """
    linea_cls = _infer_reading_cls(payload)
    if linea_cls is None or not readings:
        return

    previas = {
        (getattr(r, "parameter", ""), getattr(r, "note", ""))
        for r in (getattr(payload, "readings", None) or [])
    }
    payload.readings = build_readings(readings, language, linea_cls)

    ahora = {(r.parameter, r.note) for r in payload.readings}
    report.readings_appended = True
    # Telemetría: qué líneas no coincidían con lo que el modelo había puesto.
    # Si esto viene lleno turno tras turno, el prompt sigue sin conseguirlo y
    # el enforcement lo está tapando.
    report.readings_missing = sorted(
        {p for p, _ in ahora - previas}
    ) if previas else [r.parameter for r in payload.readings]


def _infer_reading_cls(payload):
    """La clase de ReadingLine, desde el propio modelo. None si no la tiene."""
    existentes = getattr(payload, "readings", None)
    if existentes:
        return type(existentes[0])
    try:  # pydantic v2
        return type(payload).model_fields["readings"].annotation.__args__[0]
    except Exception:
        return None


# --------------------------------------------------------------------------
# Acciones y seguridad desde el payload del especialista
# --------------------------------------------------------------------------
#
# Mismo principio que `readings`: el modelo escribe la prosa —el veredicto, el
# razonamiento, la causa— y el código arma los campos estructurados. La
# diferencia con el panel de lecturas es que estas dos NO se pueden armar por
# plantilla: `actions` es texto del especialista, y ese texto viene en inglés
# (ver _reading_is_visible). Por eso solo se sustituyen cuando el turno se
# responde en inglés; en español la traducción sigue siendo del modelo.

#: Productos que, si CAUSARON el problema, no pueden formar parte de la
#: corrección. Se busca en `likely_cause`, no en las dosis: dosificar
#: hipoclorito es normal, haber llegado acá con tricloro es el hallazgo.
_STABILIZED = ("trichlor", "dichlor", "stabilized chlorine",
               "chlorinated isocyanurate", "isocyanurate",
               "tricloro", "dicloro", "cloro estabilizado")

#: Ácidos de manejo habitual. Aparecen en `chemical_actions`, no en la causa:
#: lo que importa es que el operador va a manipular uno.
_ACIDS = ("muriatic", "hydrochloric", "sodium bisulfate", "dry acid",
          "muriático", "muriatico", "clorhídrico", "clorhidrico",
          "bisulfato", "ácido seco", "acido seco")

#: La línea de seguridad, por plantilla y por idioma. Igual que
#: `_STATUS_PHRASING`: no la redacta el modelo.
#:
#: La variante `stabilized_at_ceiling` solo se elige cuando una lectura la
#: respalda. Afirmar un límite que el dato no sostiene es exactamente el fallo
#: que `coherent_status` existe para evitar, y no deja de serlo por aparecer en
#: una advertencia.
#:
#: Y dice "el rango publicado", no "su techo": el techo sería una cota de
#: código, y el corpus no publica ninguna (ver water_targets). Una línea de
#: seguridad que reintroduce la afirmación que el panel acaba de quitar deja al
#: turno diciendo las dos cosas.
_SAFETY_PHRASING = {
    "en": {
        "stabilized_at_ceiling": (
            "Do not use trichlor or dichlor — they add cyanuric acid, "
            "already at the top of the published range."
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
            "está en el techo del rango publicado."
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

#: Nombres con los que el especialista reporta el estabilizante.
_CYA_NAMES = ("cyanuric", "cianúrico", "cianurico", "stabilizer", "cya")


def _as_list(value) -> list[str]:
    """
    `recommendations` llega como lista o como string numerado. Normaliza.

    El split exige punto MÁS espacio (`\\d+\\.\\s+`), así que "7.4" o "3.4 ppm"
    no parten la frase por la mitad: el separador es "1. ", no cualquier dígito
    seguido de punto.
    """
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        parts = re.split(r"(?:^|\s)\d+\.\s+", value.strip())
        return [p.strip() for p in parts if p.strip()]
    return []


def render_actions(specialist: dict, max_items: int = MAX_ACTIONS) -> list[str]:
    """
    Acciones correctivas desde el payload del especialista.

    Prioriza `recommendations` (ya vienen en imperativo); cae a la lista
    `chemical_actions`, que trae la acción dentro de un objeto.

    Se descarta lo que no entra en MAX_ACTION_WORDS en vez de recortarlo. Un
    bullet largo no se pierde —sigue en el material de origen y el modelo lo
    tiene para la prosa— y truncar una frase a mitad es peor que no ponerla,
    que es la misma regla que aplica `overflow_to_details`. Si NADA entra, se
    devuelve vacío y el caller conserva lo que escribió el modelo: el
    especialista fue prolijo en párrafos y el modelo ya los condensó.
    """
    items = _as_list(specialist.get("recommendations"))
    if not items:
        items = [
            str(a.get("action", "")).strip()
            for a in specialist.get("chemical_actions") or []
            if isinstance(a, dict) and str(a.get("action", "")).strip()
        ]
    return [i for i in items if _words(i) <= MAX_ACTION_WORDS][:max_items]


def _cya_at_ceiling(specialist: dict) -> bool:
    """¿Hay una lectura de estabilizante que respalde la palabra 'techo'?"""
    for r in specialist.get("test_interpretation") or []:
        if not isinstance(r, dict):
            continue
        nombre = str(r.get("parameter", "")).lower()
        tokens = set(re.split(r"[^a-záéíóúñü]+", nombre))
        if any(n in nombre for n in _CYA_NAMES[:-1]) or "cya" in tokens:
            if coherent_status(r) in ("at_ceiling", "above_maximum"):
                return True
    return False


def render_safety(specialist: dict, language: str = "es") -> str | None:
    """
    Una línea imperativa, derivada del payload. Nunca duplica una acción.

    Prioridad: el producto que causó el problema por encima de la
    incompatibilidad de manejo. Si el operador llegó acá con tricloro, repetir
    tricloro es lo que lo devuelve al mismo sitio; lo otro es higiene de
    manipulación, cierta siempre y por eso menos informativa.

    Ninguna de las dos puede salir de una acción de la lista: una prohíbe un
    producto que no se va a usar y la otra habla del orden de mezcla. Por
    construcción, no por comprobación.

    Devuelve None cuando el payload no sostiene ninguna de las dos. El caller
    conserva entonces lo que haya escrito el modelo: acá no se inventa una
    advertencia genérica, que es ruido en la línea más leída del tier visible.
    """
    frases = _SAFETY_PHRASING.get(language, _SAFETY_PHRASING["es"])

    cause = str(specialist.get("likely_cause") or "").lower()
    if any(k in cause for k in _STABILIZED):
        clave = "stabilized_at_ceiling" if _cya_at_ceiling(specialist) else "stabilized"
        return frases[clave]

    chems = " ".join(
        f"{a.get('chemical', '')} {a.get('action', '')}"
        for a in specialist.get("chemical_actions") or []
        if isinstance(a, dict)
    ).lower()
    if any(k in chems for k in _ACIDS):
        return frases["acid"]

    return None


#: La acción de cierre, por plantilla. No la redacta el modelo y no depende de
#: que el especialista se acuerde de ponerla.
_CIERRE = {
    "en": "Close the pool to bathers immediately",
    "es": "Cerrá la pileta a los bañistas de inmediato",
}


def enforce_closure_action(payload, readings, language: str,
                           report: ValidationReport) -> None:
    """
    Si el panel obliga a cerrar, el cierre es `actions[0]`. Puesto por código.

    La ronda pasada faltó justamente esto: la acción de cierre venía de
    `recommendations`, así que dependía de que el especialista la escribiera —
    y el turno que más la necesita es aquel en el que se equivoca. El
    disparador es el valor medido de desinfectante contra el mínimo publicado,
    vía `closure_required`, no un `status` de nadie.

    Si el especialista YA la escribió, no se duplica: se reordena al frente.
    Dos bullets diciendo lo mismo gastan la plaza que necesita la corrección
    que viene después.
    """
    if not closure_required(readings):
        return

    linea = _CIERRE.get(language, _CIERRE["es"])
    tokens = _contenido(linea)
    resto = [
        a for a in (payload.actions or [])
        if not (tokens and len(tokens & _contenido(a)) / len(tokens) >= _DUPLICADO_UMBRAL)
    ]
    payload.actions = [linea] + resto
    report.closure_inserted = True
    report.notes.append("cierre insertado por código: desinfectante bajo mínimo")


def enforce_visible_tier(payload, specialist: dict, language: str,
                         report: ValidationReport) -> None:
    """
    Sustituye `actions` y `safety` por lo que se puede derivar del payload.

    Cada una se sustituye solo si hay con qué. Lo que el modelo escribió no se
    reubica a `details`: sería el mismo contenido dos veces, una arriba y otra
    plegada, y el material de origen ya lo conserva íntegro.

    `actions` queda fuera cuando el turno no se responde en inglés. El
    especialista escribe en inglés y sus bullets entrarían sin traducir en una
    respuesta en español — cambiar una acción bien redactada por la misma
    acción en otro idioma es una regresión, no un enforcement. `safety` sí se
    aplica siempre: sale de plantilla y la plantilla está en los dos idiomas.
    """
    if language == "en":
        acciones = render_actions(specialist)
        if acciones:
            report.actions_rendered = True
            descartadas = [a for a in (payload.actions or []) if a not in acciones]
            if descartadas:
                report.notes.append(
                    f"{len(descartadas)} acción(es) del modelo sustituidas por las del especialista"
                )
            payload.actions = acciones

    linea = render_safety(specialist, language)
    if linea:
        report.safety_rendered = True
        payload.safety = linea


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def _has_hazard_agent(agents: Optional[list[str]]) -> bool:
    """
    Tolerante al namespace. Acepta slug ("chemistry"), display name
    ("Pool Chemistry Agent") y cualquier casing/separador, porque el valor
    que llega en state["assigned_agents"] lo produce el planner y no está
    normalizado. Un fallo aquí es silencioso: intersection() con un display
    name da vacío y la advertencia simplemente no se exige.
    """
    for raw in agents or []:
        norm = re.sub(r"[^a-z0-9]+", "_", (raw or "").lower()).strip("_")
        if not norm:
            continue
        if norm in HAZARD_AGENTS:                      # slug exacto
            return True
        if set(norm.split("_")) & HAZARD_AGENTS:       # display name
            return True
        if any(h in norm for h in HAZARD_AGENTS if "_" in h):  # slug compuesto
            return True
    return False


def _safety_trigger(contract: dict, agents: Optional[list[str]], payload) -> str:
    """
    Devuelve QUÉ activó la exigencia de safety: "contract" | "agent" |
    "lexicon" | "". Se separa del booleano para que enforce_contract pueda
    registrar el motivo sin cambiar la firma pública.
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


def _words(text: str | None) -> int:
    if not text:
        return 0
    return len(text.split())


def _visible_words(payload) -> int:
    """Cuenta el tier 1: lo que el usuario ve sin tocar nada."""
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
    """Agrega o extiende una sección plegable, sin duplicar labels."""
    existing = _get_detail(payload, label)
    if existing:
        existing.body = f"{existing.body}\n{body}".strip()
    else:
        payload.details.append(detail_cls(label=label, body=body))


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return parts[0] if parts else text.strip()


# --------------------------------------------------------------------------
# Resolución de safety condicional
# --------------------------------------------------------------------------

def resolve_safety_required(contract: dict, agents: Optional[list[str]], payload) -> bool:
    """
    `safety_required` puede ser True, False o "conditional".

    "conditional" (arquetipo `calculation`): un cálculo de turnover no necesita
    advertencia; una dosis de ácido sí. Se decide por agente de origen y,
    como red, por léxico del contenido visible.
    """
    required = contract.get("safety_required", False)
    if required is not True and required != "conditional":
        return False
    if required is True:
        return True

    if _has_hazard_agent(agents):
        return True

    surface = " ".join([payload.answer or "", *(payload.actions or [])])
    return bool(_safety_trigger(contract, agents, payload))


# --------------------------------------------------------------------------
# Normalización de acciones
# --------------------------------------------------------------------------

def normalize_actions(payload, detail_cls, report: ValidationReport) -> None:
    """
    Reglas: máximo MAX_ACTIONS bullets, cada uno ≤ MAX_ACTION_WORDS palabras.
    Lo que no cumple NO se borra: se reubica a `details`.
    """
    kept: list[str] = []
    relocated: list[str] = []

    for action in payload.actions or []:
        action = action.strip()
        if not action:
            continue
        if _words(action) > MAX_ACTION_WORDS or len(kept) >= MAX_ACTIONS:
            relocated.append(action)
        else:
            kept.append(action)

    payload.actions = kept

    if relocated:
        body = "\n".join(f"- {a}" for a in relocated)
        _append_detail(payload, OVERFLOW_LABEL, body, detail_cls)
        report.actions_relocated += len(relocated)
        report.notes.append(f"{len(relocated)} acción(es) reubicadas a details")


# --------------------------------------------------------------------------
# Overflow de presupuesto
# --------------------------------------------------------------------------

def overflow_to_details(payload, budget: int, detail_cls,
                        report: ValidationReport) -> None:
    """
    Baja acciones desde el final hasta entrar en presupuesto.

    `answer` y `safety` son intocables: si por sí solos exceden el presupuesto,
    se marca `answer_exceeds_budget` y decide el caller (retry o aceptar).
    Truncar una frase a mitad es peor que pasarse de largo.
    """
    if budget >= NO_CAP:
        return

    floor = _words(payload.answer) + _words(payload.safety)
    if floor > budget:
        report.answer_exceeds_budget = True
        report.notes.append(
            f"answer+safety={floor}p supera el presupuesto de {budget}p por sí solos"
        )
        # No se puede corregir reubicando; se deja pasar y el caller decide.
        return

    relocated: list[str] = []
    while payload.actions and _visible_words(payload) > budget:
        relocated.insert(0, payload.actions.pop())

    if relocated:
        body = "\n".join(f"- {a}" for a in relocated)
        _append_detail(payload, OVERFLOW_LABEL, body, detail_cls)
        report.overflowed = True
        report.actions_relocated += len(relocated)
        report.notes.append(f"{len(relocated)} acción(es) reubicadas por presupuesto")


# --------------------------------------------------------------------------
# Promoción de seguridad
# --------------------------------------------------------------------------

def promote_safety_from_details(payload, report: ValidationReport) -> None:
    """
    Regla dura: nada de seguridad queda detrás del pliegue.

    Si el contrato exige `safety` y el LLM lo omitió, se busca una sección
    plegada con pinta de advertencia y se sube su primera frase a tier 1.
    Si no hay nada que promover, se marca `safety_missing` -> retry.
    """
    if payload.safety and payload.safety.strip():
        return

    for detail in list(payload.details):
        if SAFETY_LABEL_PATTERN.search(detail.label) or \
           SAFETY_LABEL_PATTERN.search(detail.body):
            payload.safety = _first_sentence(detail.body)
            payload.details.remove(detail)
            report.safety_promoted = True
            report.notes.append(f"safety promovida desde details: '{detail.label}'")
            return

    report.safety_missing = True
    report.notes.append("contrato exige safety y no hay contenido para promover")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def enforce_contract(payload, contract: dict, agents: list[str] | None = None,
                     detail_cls=None, readings: list[dict] | None = None,
                     language: str = "es",
                     raw_content: str = "",
                     specialist: dict | None = None) -> tuple[Any, ValidationReport]:
    """
    Aplica el contrato al payload del synthesizer.

    Args:
        payload:    instancia de SynthesizerOutput (mutada in place).
        contract:   entrada de ARCHETYPE_CONTRACTS.
        agents:     state["assigned_agents"], para resolver safety condicional.
        detail_cls: clase del item de details. Si es None se infiere del payload
                    o se cae a un dict-like compatible.
        specialist: payload JSON de los sub-agentes, ya fusionado. De ahí salen
                    `actions` y `safety` cuando el dato los sostiene.

    Returns:
        (payload, report). Si `report.needs_retry` es True, el caller puede
        reintentar UNA vez con instrucción correctiva; si no, acepta la
        degradación determinística que ya se aplicó.
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

    # 0. Tier visible determinístico: `actions` y `safety` desde el payload del
    #    especialista. Va ANTES de normalizar para que los caps y el
    #    presupuesto se apliquen al texto definitivo, no al que se descarta.
    if specialist:
        enforce_visible_tier(payload, specialist, language, report)

    # 0b. El cierre, si el desinfectante lo obliga. Va sobre las lecturas
    #     CRUDAS —`closure_required` deriva del valor medido, no de ningún
    #     status— y antes de normalizar, para que ocupe plaza de bullet como
    #     cualquier otra acción en vez de colarse por encima del cap.
    if readings:
        enforce_closure_action(payload, readings, language, report)

    # 1. Normalizar bullets antes de medir presupuesto.
    normalize_actions(payload, detail_cls, report)

    # 2. Seguridad: promover ANTES del overflow, porque suma al conteo visible.
    trigger = _safety_trigger(contract, agents or [], payload)
    if trigger:
        report.notes.append(f"safety exigida por: {trigger}")
        promote_safety_from_details(payload, report)

    # 3. Lecturas obligatorias ANTES del presupuesto: si algo hay que añadir,
    #    tiene que competir por el espacio como el resto del tier visible, no
    #    colarse por encima del cap.
    #
    #    La reconciliación con las bandas publicadas va primero: el panel se
    #    arma sobre las lecturas ya degradadas, nunca sobre el crudo. Si se
    #    hiciera al revés, la línea saldría afirmando un código inexistente y
    #    la degradación llegaría tarde.
    if readings:
        readings = reconcile_with_published_bands(readings, report)
        enforce_visible_readings(payload, readings, language, report)

    # 4. Presupuesto.
    overflow_to_details(payload, report.budget, detail_cls, report)

    # 4. Podar secciones vacías.
    payload.details = [d for d in payload.details if d.body and d.body.strip()]

    # 5. Telemetría de calidad del tier visible. No corrige nada: mide cuántas
    #    veces el prompt no consigue lo que pide.
    report.safety_duplicates_action = safety_repeats_an_action(payload)
    if report.safety_duplicates_action:
        report.notes.append("safety repite una acción en vez de aportar algo nuevo")

    # 6. Cantidades sin respaldo. Al final, sobre el texto definitivo, para
    #    que no se le escape lo que el propio validador haya añadido.
    if raw_content:
        report.unsupported_numbers = unsupported_numbers(payload, raw_content)
        if report.unsupported_numbers:
            report.notes.append(
                f"cantidades sin respaldo en el tier visible: {report.unsupported_numbers}"
            )

    report.visible_words_after = _visible_words(payload)
    return payload, report


def _infer_detail_cls(payload):
    """Recupera la clase del item de details desde el modelo Pydantic."""
    if payload.details:
        return type(payload.details[0])
    try:  # pydantic v2
        return type(payload).model_fields["details"].annotation.__args__[0]
    except Exception:  # fallback laxo
        from types import SimpleNamespace
        return lambda label, body: SimpleNamespace(label=label, body=body)


# --------------------------------------------------------------------------
# Fallback (paso 7)
# --------------------------------------------------------------------------

def fallback_payload(raw_text: str, output_cls):
    """
    Si el structured output falla, el front nunca ve un formato distinto.
    Todo el texto crudo va a `answer`, sin plegado y sin validación.
    """
    return output_cls(answer=raw_text, actions=[], safety=None, details=[])