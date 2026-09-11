"""
Tests del parser incremental del synthesizer (src/ui/streaming.py).

La regla que protegen: el usuario NUNCA ve JSON. El synthesizer emite un
SynthesizerOutput estructurado, así que los chunks que llegan por
stream_mode="messages" son trozos de JSON a medio escribir. Si el parser se
equivoca, lo que aparece en pantalla son llaves y comillas.

Por eso casi todos los casos de abajo comprueban lo MISMO: ante la duda,
None. Un turno sin texto parcial se ve exactamente como antes del streaming;
un turno con JSON crudo en pantalla es un bug visible para el usuario.
"""

import pytest

from src.ui.streaming import chunk_text, partial_answer


# Lo que emite el synthesizer, con los escapes que de verdad aparecen:
# comillas internas y saltos de línea.
COMPLETE = (
    '{"answer": "Bajá el pH agregando \\"ácido\\" muriático.\\nSon 250 ml.", '
    '"actions": ["Medir el pH a las 4 h"], "safety": null, "details": []}'
)
EXPECTED = 'Bajá el pH agregando "ácido" muriático.\nSon 250 ml.'


class TestPartialAnswerDevuelveTexto:
    def test_extrae_el_answer_completo_y_desescapado(self):
        assert partial_answer(COMPLETE) == EXPECTED

    def test_extrae_el_answer_aunque_el_json_no_haya_cerrado(self):
        cortado = '{"answer": "Bajá el pH agregando ácido'
        assert partial_answer(cortado) == "Bajá el pH agregando ácido"

    def test_ignora_los_campos_posteriores(self):
        # `answer` es tier 1; actions y details no se pintan en el streaming.
        assert "Medir el pH" not in partial_answer(COMPLETE)


class TestPartialAnswerDevuelveNone:
    """Todos estos casos deben pintar NADA, nunca algo a medias."""

    @pytest.mark.parametrize(
        "buffer, motivo",
        [
            ("", "buffer vacío"),
            ("{", "apenas empezó el objeto"),
            ('{"answer"', "la clave sin los dos puntos"),
            ('{"answer":', "sin comilla de apertura del valor"),
            ('{"actions": ["a"], "safety": null}', "no hay campo answer"),
            ('{"answer": "caf\\u00', "escape unicode cortado a la mitad"),
        ],
    )
    def test_sin_texto_utilizable_devuelve_none(self, buffer, motivo):
        assert partial_answer(buffer) is None, motivo

    def test_backslash_colgando_no_rompe_el_parseo(self):
        # El chunk terminó justo en el backslash de un \\" . json.loads lo
        # rechazaría; el parser suelta ese carácter y sigue.
        assert partial_answer('{"answer": "dijo \\') == "dijo "


class TestNuncaFiltraJsonCrudo:
    def test_ningun_prefijo_del_stream_produce_json_visible(self):
        """
        El invariante de verdad: se reconstruye el stream carácter a carácter
        y en NINGÚN punto puede asomar sintaxis JSON en lo que se pinta.
        """
        for i in range(1, len(COMPLETE) + 1):
            pintado = partial_answer(COMPLETE[:i])
            if pintado is None:
                continue
            assert '"answer"' not in pintado
            assert not pintado.startswith("{")
            # El valor legítimo trae comillas internas (\"ácido\"), así que no
            # se puede prohibir la comilla: lo que no puede aparecer es la
            # sintaxis de campo.
            assert '":' not in pintado

    def test_el_ultimo_parcial_coincide_con_el_texto_final(self):
        parciales = [
            p for p in (partial_answer(COMPLETE[:i]) for i in range(1, len(COMPLETE) + 1))
            if p is not None
        ]
        assert parciales[-1] == EXPECTED

    def test_el_texto_solo_crece(self):
        """Un parcial nunca puede encoger: en pantalla se vería un borrado."""
        anterior = ""
        for i in range(1, len(COMPLETE) + 1):
            actual = partial_answer(COMPLETE[:i])
            if actual is None:
                continue
            assert len(actual) >= len(anterior)
            anterior = actual


class _Chunk:
    """Stand-in de AIMessageChunk: solo importa `.content`."""

    def __init__(self, content):
        self.content = content


class TestChunkText:
    def test_content_como_string(self):
        assert chunk_text(_Chunk("hola")) == "hola"

    def test_content_como_lista_de_bloques(self):
        # Gemini parte el contenido en bloques cuando hay varias partes.
        assert chunk_text(_Chunk([{"text": "ho"}, {"text": "la"}])) == "hola"

    def test_bloques_sin_texto_no_aportan(self):
        # Los bloques de thinking no traen 'text' y no deben pintarse.
        assert chunk_text(_Chunk([{"thought": True}, {"text": "ok"}])) == "ok"

    def test_content_inesperado_no_revienta(self):
        assert chunk_text(_Chunk(None)) == ""
        assert chunk_text(_Chunk(123)) == ""
