"""
`actions` y `safety` las arma el código, no el modelo.

Misma tesis que el panel de lecturas (test_visible_readings): lo que el
especialista ya entregó como dato estructurado no se le vuelve a pedir al
modelo en prosa. El modelo escribe el veredicto, el razonamiento y la causa;
estos dos campos se derivan del payload.

El cambio tiene además un efecto de latencia: `safety_missing` era el gatillo
de reintento que más disparaba —siete rondas seguidas sobre la misma
consulta— y cada disparo cuesta una llamada completa al modelo.
"""

import pytest

from src.graph_context.response_contracts import (
    DetailSection,
    SynthesizerOutput,
    get_contract,
)
from src.graph_context.response_validator import (
    enforce_contract,
    render_actions,
    render_safety,
    _as_list,
)


# El payload del trace: pileta con CYA en el techo por programa de tricloro.
PAYLOAD = {
    "likely_cause": (
        "Sustained use of trichlor tablets as the primary sanitizer, which "
        "adds cyanuric acid with every dose."
    ),
    "recommendations": [
        "Close the pool immediately due to insufficient disinfectant residual",
        "Perform a partial drain and refill of approximately 50%",
        "Adjust pH down to 7.4 using muriatic acid",
        "Chlorinate to the new proportional target of 3.4 ppm",
    ],
    "chemical_actions": [
        {"action": "Lower pH", "chemical": "muriatic acid",
         "rationale": "pH above the code maximum"},
    ],
    "test_interpretation": [
        {"parameter": "Cyanuric Acid", "measured": 90, "status": "at_ceiling",
         "regulatory_limit": 90.0, "operating_target": 40.0},
    ],
}


def _payload(answer="Cerrá la pileta.", actions=None, safety=None):
    return SynthesizerOutput(answer=answer, actions=actions or [],
                             safety=safety, details=[])


class TestNormalizacionDeRecommendations:
    def test_una_lista_pasa_tal_cual(self):
        assert _as_list(["a", "b"]) == ["a", "b"]

    def test_un_string_numerado_se_parte(self):
        texto = "1. Cerrar la pileta 2. Bajar el pH 3. Clorar"
        assert _as_list(texto) == ["Cerrar la pileta", "Bajar el pH", "Clorar"]

    def test_un_decimal_no_parte_la_frase(self):
        """
        El separador es "1. " —punto MÁS espacio—, no cualquier dígito seguido
        de punto. Sin eso, "ajustá el pH a 7.4 usando ácido" se rompía en dos
        y la mitad del bullet desaparecía.
        """
        assert _as_list("Adjust pH to 7.4 using acid") == ["Adjust pH to 7.4 using acid"]

    def test_vacio_no_inventa_nada(self):
        assert _as_list(None) == [] and _as_list("") == [] and _as_list([]) == []


class TestRenderActions:
    def test_salen_las_recomendaciones_del_especialista(self):
        assert render_actions(PAYLOAD) == PAYLOAD["recommendations"]

    def test_cae_a_chemical_actions_si_no_hay_recomendaciones(self):
        sin_recs = {**PAYLOAD, "recommendations": []}
        assert render_actions(sin_recs) == ["Lower pH"]

    def test_respeta_el_tope_de_bullets(self):
        muchas = {"recommendations": [f"Acción {i}" for i in range(9)]}
        assert len(render_actions(muchas)) == 4

    def test_lo_que_no_entra_en_el_cap_se_descarta_entero(self):
        """
        No se recorta a mitad de frase: esa es la misma regla que aplica
        `overflow_to_details` a `answer`. El contenido sigue en el material de
        origen y el modelo lo tiene para la prosa.
        """
        largo = " ".join(["palabra"] * 20)
        r = render_actions({"recommendations": ["Cerrar la pileta", largo]})
        assert r == ["Cerrar la pileta"]

    def test_si_nada_entra_devuelve_vacio(self):
        """
        Un especialista que escribió párrafos en vez de imperativos. Vacío es
        la señal para que el caller conserve lo que el modelo ya condensó.
        """
        largo = " ".join(["palabra"] * 20)
        assert render_actions({"recommendations": [largo]}) == []


class TestRenderSafety:
    def test_el_producto_que_causo_el_problema_manda(self):
        linea = render_safety(PAYLOAD, "en")
        assert "trichlor" in linea and "dichlor" in linea

    def test_el_techo_solo_se_afirma_con_una_lectura_que_lo_sostenga(self):
        """
        Afirmar un límite que el dato no respalda es el fallo que
        `coherent_status` existe para evitar, y no deja de serlo por aparecer
        en una advertencia.
        """
        assert "already at its ceiling" in render_safety(PAYLOAD, "en")

        sin_lectura = {**PAYLOAD, "test_interpretation": []}
        linea = render_safety(sin_lectura, "en")
        assert "ceiling" not in linea and "cyanuric acid" in linea

    def test_la_causa_gana_sobre_el_manejo(self):
        """
        El payload tiene ácido muriático en `chemical_actions` Y tricloro en la
        causa. Repetir tricloro devuelve al operador al mismo sitio; el orden
        de mezcla es higiene cierta siempre y por eso menos informativa.
        """
        assert "trichlor" in render_safety(PAYLOAD, "en")

    def test_sin_causa_de_producto_cae_al_manejo_de_acido(self):
        solo_acido = {**PAYLOAD, "likely_cause": "Heavy bather load over the weekend"}
        assert "Never mix acid and chlorine" in render_safety(solo_acido, "en")

    def test_sin_nada_que_sostenerla_devuelve_none(self):
        """
        Acá no se inventa una advertencia genérica: es ruido en la línea más
        leída del tier visible.
        """
        assert render_safety({"likely_cause": "Heavy bather load"}, "en") is None
        assert render_safety({}, "en") is None

    def test_en_español_sale_en_español(self):
        """
        La plantilla está en los dos idiomas, así que `safety` se aplica
        siempre — a diferencia de `actions`, que es texto del especialista.
        """
        linea = render_safety(PAYLOAD, "es")
        assert "tricloro" in linea and "cianúrico" in linea

    def test_no_repite_una_accion(self):
        from src.graph_context.response_validator import safety_repeats_an_action
        p = _payload(actions=render_actions(PAYLOAD), safety=render_safety(PAYLOAD, "en"))
        assert safety_repeats_an_action(p) is False


class TestCableadoEnElContrato:
    def _aplicar(self, p, language="en", specialist=PAYLOAD):
        return enforce_contract(
            p, get_contract("assessment"), ["chemistry"],
            detail_cls=DetailSection, language=language, specialist=specialist,
        )

    def test_sustituye_lo_que_escribio_el_modelo(self):
        p = _payload(actions=["Bajá el pH cuando puedas"], safety="Usá guantes.")
        p, rep = self._aplicar(p)
        assert p.actions == PAYLOAD["recommendations"]
        assert "trichlor" in p.safety
        assert rep.actions_rendered and rep.safety_rendered

    def test_sin_payload_no_toca_nada(self):
        p = _payload(actions=["Bajá el pH"], safety="Usá guantes.")
        p, rep = self._aplicar(p, specialist=None)
        assert p.actions == ["Bajá el pH"] and p.safety == "Usá guantes."
        assert not rep.actions_rendered and not rep.safety_rendered

    def test_en_español_las_acciones_siguen_siendo_del_modelo(self):
        """
        El especialista escribe en inglés (ver _reading_is_visible). Cambiar
        una acción bien redactada por la misma acción sin traducir es una
        regresión, no un enforcement.
        """
        p = _payload(actions=["Cerrá la pileta a los bañistas"])
        p, rep = self._aplicar(p, language="es")
        assert p.actions == ["Cerrá la pileta a los bañistas"]
        assert not rep.actions_rendered
        assert rep.safety_rendered and "tricloro" in p.safety

    def test_una_safety_derivada_no_gatilla_reintento(self):
        p = _payload()
        _, rep = self._aplicar(p)
        assert rep.safety_missing is False
        assert rep.needs_retry is False

    def test_sin_derivada_tampoco_gatilla_reintento(self):
        """El ahorro de latencia no depende de que la plantilla aplique."""
        p = _payload()
        _, rep = self._aplicar(p, specialist={"likely_cause": "Heavy bather load"})
        assert rep.safety_missing is True
        assert rep.needs_retry is False

    def test_el_cap_de_bullets_se_aplica_al_texto_definitivo(self):
        """
        La sustitución va ANTES de `normalize_actions`: lo que se descarta no
        puede gastar presupuesto ni plaza de bullet.
        """
        muchas = {"recommendations": [f"Acción corta {i}" for i in range(9)]}
        p, rep = self._aplicar(_payload(), specialist=muchas)
        assert len(p.actions) == 4
        assert rep.actions_relocated == 0
