"""
Toda lectura fuera de rango va donde el usuario lee. Comprobado, no pedido.

Origen: trace 6660e14f, cuarta evaluación de B1. El contrato de `assessment`
ya lo exige literalmente — "una lectura que el usuario reportó y la respuesta
no menciona se lee como una lectura que te pareció aceptable" — y el modelo
lo incumplió igual:

    tier visible: 80 palabras de un presupuesto de 900
    pH 7.9 (above_maximum, violación de código): ausente
    cloro combinado 0.4 (at_ceiling): ausente
    cyanuric acid 90 (at_ceiling): solo nombrado dentro de una acción

No fue falta de sitio: relocalizó a `details` teniendo 820 palabras libres.
Tres rondas de evaluación señalaron el mismo síntoma y las instrucciones no
lo corrigieron, así que deja de ser una instrucción.

Nombrar el parámetro dentro de una acción ("baja el pH") no cuenta como
reportarlo: el operador no se entera de cuánto marcó ni de que incumple. Por
eso el chequeo exige nombre Y valor.
"""

import pytest

from src.graph_context.response_contracts import DetailSection, SynthesizerOutput
from src.graph_context.response_validator import (
    enforce_contract,
    enforce_visible_readings,
    required_readings,
)
from src.graph_context.response_contracts import get_contract


def _lectura(parametro, medido, status, target=None):
    return {"parameter": parametro, "measured": medido, "status": status,
            "operating_target": target}


# El panel exacto del trace.
PANEL = [
    _lectura("Free Chlorine", 0.8, "below_minimum", 3.0),
    _lectura("Combined Chlorine", 0.4, "at_ceiling", 0.2),
    _lectura("pH", 7.9, "above_maximum", 7.5),
    _lectura("Cyanuric Acid", 90, "at_ceiling", 40.0),
    _lectura("Total Alkalinity", 130, "in_range", 90.0),
    _lectura("Calcium Hardness", 380, "in_range", 300.0),
    _lectura("Temperature", 82, "in_range", None),
]


def _payload(answer, actions=None):
    return SynthesizerOutput(answer=answer, actions=actions or [], safety=None, details=[])


class TestQueLecturasSonObligatorias:
    def test_solo_las_que_no_estan_en_rango(self):
        req = required_readings(PANEL)
        assert {r["parameter"] for r in req} == {
            "Free Chlorine", "Combined Chlorine", "pH", "Cyanuric Acid"}

    def test_in_range_no_obliga_a_nada(self):
        # Mencionar los siete siempre convertiría la respuesta en un listado.
        assert all(r["status"] != "in_range" for r in required_readings(PANEL))

    def test_una_entrada_sin_status_no_obliga(self):
        assert required_readings([{"parameter": "pH", "measured": 7.9}]) == []

    def test_tolera_una_entrada_que_no_sea_lista(self):
        assert required_readings(None) == []
        assert required_readings("{}") == []


class TestDeteccionDeOmisiones:
    def test_nombrar_en_una_accion_no_cuenta_como_reportar(self):
        """El caso del trace: 'baja el pH' sin decir que marcó 7.9."""
        from src.graph_context.response_validator import ValidationReport
        p = _payload("Tu cloro libre 0.8 está bajo el mínimo.",
                     ["Añade ácido para bajar el pH"])
        rep = ValidationReport()
        enforce_visible_readings(p, [_lectura("pH", 7.9, "above_maximum")], "es", rep)
        assert rep.readings_missing == ["pH"]

    def test_con_nombre_y_valor_si_cuenta(self):
        from src.graph_context.response_validator import ValidationReport
        p = _payload("El pH 7.9 supera el máximo permitido.")
        rep = ValidationReport()
        enforce_visible_readings(p, [_lectura("pH", 7.9, "above_maximum")], "es", rep)
        assert rep.readings_missing == []
        assert rep.readings_appended is False


class TestPromocionAlTierVisible:
    def test_añade_las_omitidas_con_valor_y_estado(self):
        from src.graph_context.response_validator import ValidationReport
        p = _payload("El cloro libre 0.8 está por debajo del mínimo de 2.0.")
        rep = ValidationReport()
        enforce_visible_readings(p, required_readings(PANEL), "es", rep)

        assert rep.readings_appended is True
        for omitido in ["pH", "Cyanuric Acid", "Combined Chlorine"]:
            assert omitido.lower() in p.answer.lower()
        assert "7.9" in p.answer and "90" in p.answer

    def test_el_texto_añadido_dice_el_estado(self):
        from src.graph_context.response_validator import ValidationReport
        p = _payload("Cloro libre 0.8 bajo mínimo.")
        rep = ValidationReport()
        enforce_visible_readings(p, [_lectura("pH", 7.9, "above_maximum")], "es", rep)
        assert "por encima del máximo" in p.answer

    def test_at_ceiling_se_describe_como_sin_margen_no_como_violacion(self):
        """
        El error de las cuatro evaluaciones: un techo exacto descrito como
        infracción. El texto que añade el validador nunca puede cometerlo.
        """
        from src.graph_context.response_validator import ValidationReport
        p = _payload("Cloro libre bajo.")
        rep = ValidationReport()
        enforce_visible_readings(p, [_lectura("Cyanuric Acid", 90, "at_ceiling")], "es", rep)
        assert "sin margen" in p.answer
        for prohibido in ["excede", "viola", "infracción", "por encima del máximo"]:
            assert prohibido not in p.answer.lower()

    def test_respeta_el_idioma(self):
        from src.graph_context.response_validator import ValidationReport
        p = _payload("Free chlorine is low.")
        rep = ValidationReport()
        enforce_visible_readings(p, [_lectura("pH", 7.9, "above_maximum")], "en", rep)
        assert "above the maximum" in p.answer
        assert "Also:" in p.answer

    def test_no_toca_nada_si_estan_todas(self):
        from src.graph_context.response_validator import ValidationReport
        p = _payload("Free chlorine 0.8, combined chlorine 0.4, pH 7.9, cyanuric acid 90.")
        antes = p.answer
        rep = ValidationReport()
        enforce_visible_readings(p, required_readings(PANEL), "en", rep)
        assert p.answer == antes
        assert rep.readings_appended is False


class TestIntegracionConElContrato:
    def test_enforce_contract_aplica_el_chequeo(self):
        p = _payload("El cloro libre 0.8 está bajo el mínimo.")
        p, rep = enforce_contract(
            p, get_contract("assessment"), ["chemistry"],
            detail_cls=DetailSection,
            readings=required_readings(PANEL), language="es",
        )
        assert rep.readings_appended is True
        assert "7.9" in p.answer

    def test_sin_readings_el_comportamiento_no_cambia(self):
        p = _payload("Una respuesta cualquiera.")
        antes = p.answer
        p, rep = enforce_contract(p, get_contract("assessment"), ["chemistry"],
                                  detail_cls=DetailSection)
        assert p.answer == antes
        assert rep.readings_missing == []

    def test_el_reporte_nombra_lo_que_faltaba(self):
        p = _payload("Cloro libre 0.8 bajo el mínimo.")
        _, rep = enforce_contract(
            p, get_contract("assessment"), ["chemistry"],
            detail_cls=DetailSection,
            readings=required_readings(PANEL), language="es",
        )
        # Telemetría: si esto aparece seguido en Langfuse, el prompt del
        # synthesizer sigue sin surtir efecto y el enforcement lo está tapando.
        assert set(rep.readings_missing) >= {"pH", "Cyanuric Acid"}


class TestExtraccionDesdeLosSubAgentes:
    def test_lee_el_test_interpretation_del_output(self):
        import json as _json
        from src.agent.nodes import _readings_from_results
        from src.agent.state import AgentResult

        salida = _json.dumps({"status": "closed", "test_interpretation": PANEL})
        res = {"step_1": AgentResult(agent="chemistry", step=1, output=salida)}
        assert len(_readings_from_results(res)) == 7

    def test_atraviesa_un_code_fence(self):
        import json as _json
        from src.agent.nodes import _readings_from_results
        from src.agent.state import AgentResult

        salida = "```json\n" + _json.dumps({"test_interpretation": PANEL}) + "\n```"
        res = {"step_1": AgentResult(agent="chemistry", step=1, output=salida)}
        assert len(_readings_from_results(res)) == 7

    def test_un_json_roto_no_tumba_el_turno(self):
        from src.agent.nodes import _readings_from_results
        from src.agent.state import AgentResult

        res = {"step_1": AgentResult(agent="chemistry", step=1,
                                     output='{"test_interpretation": [ roto')}
        assert _readings_from_results(res) == []

    def test_una_salida_en_prosa_no_aporta_lecturas(self):
        from src.agent.nodes import _readings_from_results
        from src.agent.state import AgentResult

        res = {"step_1": AgentResult(agent="general", step=1, output="Hola, ¿en qué ayudo?")}
        assert _readings_from_results(res) == []
