"""
ui/streaming.py
===============
Extraer el campo `answer` de un JSON que todavía se está escribiendo.

Por qué hace falta
──────────────────
El synthesizer no emite prosa: emite un SynthesizerOutput por
`with_structured_output`, o sea JSON. Los chunks que llegan por
stream_mode="messages" son trozos de ese JSON:

    {"answer": "Bajá el pH agregando ácido mur

Pintar eso tal cual le mostraría llaves y comillas al usuario. Este módulo
recupera el texto del campo `answer` de un buffer incompleto, para poder
mostrar la respuesta mientras se genera en lugar de esperar al nodo entero.

Funciona porque `answer` es el PRIMER campo de SynthesizerOutput y Gemini
respeta el orden del schema: es lo primero que llega, antes de actions,
safety y details. Si algún día se reordena el modelo, esto deja de dar texto
temprano — no se rompe, simplemente no encuentra nada hasta más tarde.

Contrato: ante cualquier duda, devolver None. El llamador no pinta nada
parcial y el turno se ve exactamente como antes del streaming. Mostrar JSON
a medias es peor que no mostrar nada.
"""

from __future__ import annotations

import json
import re

# Captura el contenido de "answer" SIN exigir la comilla de cierre: la gracia
# es leer una cadena que aún no terminó. `(?:[^"\\]|\\.)*` avanza por
# caracteres normales o por pares de escape, así que un \" interno no se
# confunde con el final de la cadena.
_ANSWER_RE = re.compile(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)', re.DOTALL)


def partial_answer(buffer: str) -> str | None:
    """
    El valor de `answer` tal como va hasta ahora, o None si aún no se puede.

    Devuelve None cuando:
      - el campo todavía no apareció en el buffer
      - lo que hay no se puede des-escapar a texto válido

    El texto vuelve des-escapado (\\n como salto real, \\" como comilla),
    porque va directo a st.markdown.
    """
    if not buffer:
        return None

    match = _ANSWER_RE.search(buffer)
    if not match:
        return None

    raw = match.group(1)

    # Un backslash suelto al final es una secuencia de escape cortada por la
    # mitad ("...ácido \" justo cuando el chunk terminó). json.loads la
    # rechaza; soltarla cuesta un carácter que llega en el chunk siguiente.
    if raw.endswith("\\") and not raw.endswith("\\\\"):
        raw = raw[:-1]

    try:
        return json.loads(f'"{raw}"')
    except (ValueError, TypeError):
        # Escape incompleto de los largos: á cortado en \u00. Se resuelve
        # solo en el chunk siguiente; hasta entonces, no pintar.
        return None


def chunk_text(chunk) -> str:
    """
    El texto de un AIMessageChunk, sea cual sea la forma de `.content`.

    Gemini devuelve str en unos casos y lista de bloques en otros (cuando hay
    thinking o partes múltiples). Mismo criterio que `_extract_text` en
    nodes.py, repetido acá para que este módulo no dependa del paquete agent:
    es UI, y se testea sin levantar el grafo.
    """
    content = getattr(chunk, "content", chunk)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )

    return ""
