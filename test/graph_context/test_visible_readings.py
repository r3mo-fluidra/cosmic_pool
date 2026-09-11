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


def _lectura(parametro, medido, status, target=None, limite=None):
    return {"parameter": parametro, "measured": medido, "status": status,
            "operating_target": target, "regulatory_limit": limite}


# El panel del trace 6660e14f, con los límites que trae el especialista.
# Un status de violación SIN límite es incoherente y se degrada: ver
# TestCoherenciaDelStatus.
PANEL = [
    _lectura("Free Chlorine", 0.8, "below_minimum", 3.0, 2.0),
    _lectura("Combined Chlorine", 0.4, "at_ceiling", 0.2, 0.4),
    _lectura("pH", 7.9, "above_maximum", 7.5, 7.8),
    _lectura("Cyanuric Acid", 90, "at_ceiling", 40.0, 90.0),
    _lectura("Total Alkalinity", 130, "in_range", 90.0, 180.0),
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


class TestFraseoDeterminista:
    """
    El fraseo de cada lectura sale de una plantilla elegida por su `status`.
    El modelo no lo redacta.

    Cuatro rondas de evaluación sobre la misma consulta mostraron que como
    instrucción de prompt no se sostiene. Los tres fallos, todos sobre datos
    que el especialista había clasificado bien:
      - "extremely high" sobre un at_ceiling
      - un in_range convertido en "above the maximum"
      - "above the 120 ppm operating ceiling" sobre una entrada sin límite
    """

    def _nota(self, lectura, idioma="en"):
        from src.graph_context.response_validator import reading_note
        return reading_note(lectura, idioma)

    def test_below_minimum_cita_el_limite(self):
        assert self._nota(_lectura("FC", 0.8, "below_minimum", 3.0, 2.0)) == \
            "in violation, below the 2 ppm minimum; target 3 ppm"

    def test_above_maximum_cita_el_limite(self):
        assert self._nota(_lectura("pH", 7.9, "above_maximum", 7.5, 7.8)) == \
            "in violation, above the 7.8 cap; target 7.5"

    def test_at_ceiling_es_cumplimiento(self):
        nota = self._nota(_lectura("CYA", 90, "at_ceiling", 40.0, 90.0))
        assert nota == "compliant, no margin at the ceiling; target 40 ppm"
        assert "violation" not in nota

    def test_at_floor_es_cumplimiento(self):
        assert "compliant" in self._nota(_lectura("FC", 2.0, "at_floor", 3.0, 2.0))

    def test_in_range_no_dramatiza(self):
        assert self._nota(_lectura("CH", 380, "in_range", 300.0, 1000.0)) == "in range"

    def test_el_limite_nunca_sale_del_objetivo_operativo(self):
        """
        El último fallo: "above the 120 ppm operating ceiling" presentaba un
        objetivo de industria como techo de código. La plantilla solo se
        rellena con regulatory_limit.

        El objetivo sí aparece, pero detrás y etiquetado: el hueco del
        veredicto es del límite y de nadie más.
        """
        nota = self._nota(_lectura("TA", 130, "above_maximum", 100.0, 180.0))
        assert nota.startswith("in violation, above the 180 ppm cap")
        assert "100" not in nota.split(";")[0]

    def test_el_objetivo_operativo_acompana_al_veredicto(self):
        """
        Cuatro lecturas del trace 148acb15 salieron sin su objetivo: el panel
        decía dónde empieza la infracción y no dónde hay que dejar el vaso.
        """
        nota = self._nota(_lectura("FC", 0.8, "below_minimum", 4.0, 2.0))
        assert nota.endswith("; target 4 ppm")

    def test_una_lectura_en_rango_no_arrastra_un_objetivo(self):
        # No hay nada que corregir: el sufijo se leería como tarea pendiente.
        assert self._nota(_lectura("CH", 380, "in_range", 300.0, 1000.0)) == "in range"

    def test_un_objetivo_en_cero_se_reporta(self):
        # El objetivo de cloro combinado es 0.0, y 0.0 es un dato, no un hueco.
        nota = self._nota(_lectura("CC", 0.4, "above_maximum", 0.0, 0.2))
        assert nota == "in violation, above the 0.2 ppm cap; target 0 ppm"

    def test_un_status_sin_limite_no_afirma_infraccion(self):
        # coherent_status lo degrada; la nota informa sin veredicto.
        nota = self._nota(_lectura("TA", 130, "above_maximum", 100.0))
        assert "violation" not in nota
        assert "100" in nota

    def test_sin_limite_ni_objetivo_lo_dice(self):
        nota = self._nota(_lectura("Temp", 82, "above_maximum"))
        assert nota == "reported; no code bound available"

    def test_los_enteros_no_arrastran_ceros(self):
        assert "2 ppm minimum" in self._nota(_lectura("FC", 0.8, "below_minimum", 3.0, 2.0))

    def test_en_espanol(self):
        nota = self._nota(_lectura("pH", 7.9, "above_maximum", 7.5, 7.8), "es")
        assert nota == "en infracción, por encima del máximo de 7.8; objetivo 7.5"


class TestElPanelSeConstruyeEntero:
    def _construir(self, panel, idioma="en"):
        from src.graph_context.response_contracts import ReadingLine
        from src.graph_context.response_validator import build_readings
        return build_readings(panel, idioma, ReadingLine)

    def test_incluye_todos_los_parametros_tambien_los_en_rango(self):
        """
        El operador entregó siete lecturas: ver las siete confirma que se
        leyeron todas. Omitir las correctas obliga a deducir por ausencia.
        """
        assert len(self._construir(PANEL)) == len(PANEL)

    def test_reemplaza_lo_que_escribio_el_modelo(self):
        from src.graph_context.response_contracts import ReadingLine
        from src.graph_context.response_validator import ValidationReport, enforce_visible_readings
        p = _payload("Cierra la pileta.")
        p.readings = [ReadingLine(parameter="pH", measured="7.9",
                                  note="extremely high, way over the limit")]
        rep = ValidationReport()
        enforce_visible_readings(p, PANEL, "en", rep)
        notas = {r.note for r in p.readings}
        assert "extremely high, way over the limit" not in notas
        assert "in violation, above the 7.8 cap; target 7.5" in notas

    def test_ignora_entradas_sin_valor(self):
        assert self._construir([{"parameter": "pH", "status": "in_range"}]) == []

    def test_normaliza_el_nombre(self):
        lineas = self._construir([_lectura("free_chlorine", 0.8, "below_minimum", 3.0, 2.0)])
        assert lineas[0].parameter == "Free Chlorine"


class TestIntegracionConElContrato:
    def test_enforce_contract_aplica_el_chequeo(self):
        p = _payload("El cloro libre 0.8 está bajo el mínimo.")
        p, rep = enforce_contract(
            p, get_contract("assessment"), ["chemistry"],
            detail_cls=DetailSection,
            readings=required_readings(PANEL), language="es",
        )
        assert rep.readings_appended is True
        assert "7.9" in {r.measured for r in p.readings}

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


class TestCoherenciaDelStatus:
    """
    Trace 4e3153ed: el especialista devolvió la alcalinidad total como
    above_maximum con regulatory_limit en null — la juzgó contra su
    operating_target. El parámetro no solo cumplía: con la corrección de
    cianurato estaba prácticamente en objetivo.

    Declarar infracción a un parámetro sano llega igual de lejos que negar
    una real: el operador lo repite ante un inspector y reporta un
    incumplimiento que no existe.
    """

    def test_una_violacion_sin_limite_no_se_sostiene(self):
        from src.graph_context.response_validator import coherent_status
        assert coherent_status(
            _lectura("Total Alkalinity", 130, "above_maximum", 100.0)) is None

    def test_una_violacion_con_limite_se_respeta(self):
        from src.graph_context.response_validator import coherent_status
        assert coherent_status(
            _lectura("pH", 7.9, "above_maximum", 7.5, 7.8)) == "above_maximum"

    def test_at_ceiling_no_necesita_degradarse(self):
        # Un techo se afirma contra el valor publicado, que la entrada trae.
        from src.graph_context.response_validator import coherent_status
        assert coherent_status(
            _lectura("Cyanuric Acid", 90, "at_ceiling", 40.0, 90.0)) == "at_ceiling"

    def test_la_incoherente_no_llega_al_tier_visible(self):
        incoherente = _lectura("Total Alkalinity", 130, "above_maximum", 100.0)
        assert required_readings([incoherente]) == []

    def test_el_caso_completo_del_trace(self):
        panel = PANEL[:4] + [_lectura("Total Alkalinity", 130, "above_maximum", 100.0)]
        nombres = {r["parameter"] for r in required_readings(panel)}
        assert "Total Alkalinity" not in nombres
        assert nombres == {"Free Chlorine", "Combined Chlorine", "pH", "Cyanuric Acid"}


class TestCantidadesSinRespaldo:
    """
    El synthesizer escribió "drena y rellena entre un treinta y un cuarenta
    por ciento" con el calculation_request sin ejecutar. El número no estaba
    en el payload del agente, y encima era erróneo: ninguna de las dos
    fracciones alcanzaba el objetivo de estabilizador que él mismo pidió.

    Un número inventado es peor que un dato ausente: llega con la misma
    confianza que los verdaderos.
    """

    def test_detecta_una_cantidad_que_no_esta_en_el_origen(self):
        from src.graph_context.response_validator import unsupported_numbers
        p = _payload("Drena el 37.5% del vaso.")
        assert "37.5" in unsupported_numbers(p, "El cianúrico está en 90 ppm.")

    def test_un_numero_del_origen_pasa(self):
        from src.graph_context.response_validator import unsupported_numbers
        p = _payload("El cianúrico está en 90 ppm.")
        assert unsupported_numbers(p, "cyanuric acid 90 ppm at ceiling") == []

    def test_tolera_diferencias_de_formato(self):
        from src.graph_context.response_validator import unsupported_numbers
        p = _payload("Cloro libre 3 ppm.")
        assert unsupported_numbers(p, '"operating_target": 3.0') == []

    def test_los_numeros_de_lenguaje_no_cuentan(self):
        # "las 24 horas", "los 3 pasos": no son cantidades que respaldar.
        from src.graph_context.response_validator import unsupported_numbers
        p = _payload("Repite el test en 24 horas, en 3 puntos del vaso.")
        assert unsupported_numbers(p, "sin cifras") == []

    def test_sin_material_de_origen_no_acusa(self):
        from src.graph_context.response_validator import unsupported_numbers
        assert unsupported_numbers(_payload("47 ppm"), "") == []

    def test_enforce_contract_lo_registra(self):
        p = _payload("Drena el 37.5% del vaso.")
        _, rep = enforce_contract(
            p, get_contract("assessment"), ["chemistry"],
            detail_cls=DetailSection, raw_content="cyanuric acid 90 ppm",
        )
        assert "37.5" in rep.unsupported_numbers


class TestElReporteSigueSiendoSerializable:
    def test_to_dict_es_un_metodo_de_la_clase(self):
        """
        Regresión de un bug propio: al insertar funciones de módulo dentro del
        bloque de la clase, to_dict quedó huérfano a nivel de módulo. El
        synthesizer llama report.to_dict() y el AttributeError se tragaba
        en su except, anulando el enforcement ENTERO sin que se notara salvo
        por una línea en `validation`.
        """
        from src.graph_context.response_validator import ValidationReport
        rep = ValidationReport()
        assert callable(getattr(rep, "to_dict", None))
        d = rep.to_dict()
        assert "readings_missing" in d and "unsupported_numbers" in d


class TestSafetyConInformacionUnica:
    """
    Bajo una acción "cierra la pileta a los bañistas", el campo de seguridad
    decía "mantén la pileta cerrada hasta restaurar el cloro". La línea que
    más se lee del tier visible, gastada en repetir la primera acción — y la
    que sí llevaba información única, la prohibición del producto que causó
    el problema, desaparecida del tier visible.

    Se mide, no se corrige: suprimir una línea de seguridad por parecerse a
    otra cosa es peor fallo que dejarla repetida.
    """

    def test_detecta_la_repeticion_del_trace(self):
        from src.graph_context.response_validator import safety_repeats_an_action
        p = _payload("Cierra la pileta.", ["Cerrar la pileta a los bañistas"])
        p.safety = "Mantén la pileta cerrada a los bañistas hasta restaurarlo."
        assert safety_repeats_an_action(p) is True

    def test_una_linea_con_informacion_propia_pasa(self):
        from src.graph_context.response_validator import safety_repeats_an_action
        p = _payload("Cierra la pileta.", ["Cerrar la pileta a los bañistas"])
        p.safety = "No uses tricloro ni dicloro: suben el estabilizador."
        assert safety_repeats_an_action(p) is False

    def test_sin_safety_no_hay_duplicado(self):
        from src.graph_context.response_validator import safety_repeats_an_action
        assert safety_repeats_an_action(_payload("x", ["Cerrar la pileta"])) is False

    def test_sin_acciones_tampoco(self):
        from src.graph_context.response_validator import safety_repeats_an_action
        p = _payload("x")
        p.safety = "No mezcles ácido con hipoclorito."
        assert safety_repeats_an_action(p) is False

    def test_funciona_en_ingles(self):
        from src.graph_context.response_validator import safety_repeats_an_action
        p = _payload("Close it.", ["Close the pool to bathers immediately"])
        p.safety = "Keep the pool closed to all bathers until restored."
        assert safety_repeats_an_action(p) is True

    def test_enforce_contract_lo_registra(self):
        p = _payload("Cierra.", ["Cerrar la pileta a los bañistas"])
        p.safety = "Mantén la pileta cerrada a los bañistas."
        _, rep = enforce_contract(p, get_contract("assessment"), ["chemistry"],
                                  detail_cls=DetailSection)
        assert rep.safety_duplicates_action is True

    def test_no_borra_la_linea(self):
        """El campo sigue ahí: se mide, no se suprime."""
        p = _payload("Cierra.", ["Cerrar la pileta a los bañistas"])
        p.safety = "Mantén la pileta cerrada a los bañistas."
        p, _ = enforce_contract(p, get_contract("assessment"), ["chemistry"],
                                detail_cls=DetailSection)
        assert p.safety


class TestVozEnSegundaPersona:
    @pytest.fixture
    def P(self):
        from src.prompts.prompts import SYNTHESIZER_PROMPT
        return SYNTHESIZER_PROMPT

    def test_prohibe_el_nosotros_corporativo(self, P):
        # "We cannot calculate your doses yet" — voz de empresa detrás de un
        # formulario, no del técnico al lado de la pileta.
        assert 'Never "we", "us"' in P
        assert "we cannot calculate your doses yet" in P

    def test_da_la_alternativa(self, P):
        assert "I need your" in P

    def test_el_safety_debe_aportar_algo_nuevo(self, P):
        assert "information that is NOT already in `actions`" in P

    def test_prefiere_la_linea_de_mayor_alcance(self, P):
        assert "the one with\n  the longest reach" in P or "the one with the longest reach" in P

    def test_vacio_es_mejor_que_duplicado(self, P):
        assert "An empty\n  field is better than a duplicate one" in P or \
               "An empty field is better than a duplicate one" in P


class TestGrafiaYUnidades:
    def _linea(self, lectura):
        from src.graph_context.response_contracts import ReadingLine
        from src.graph_context.response_validator import build_readings
        return build_readings([lectura], "en", ReadingLine)[0]

    def test_ph_no_se_convierte_en_ph_capitalizado(self):
        # .title() producía "Ph", que ningún operador escribe.
        assert self._linea(_lectura("ph", 7.9, "in_range")).parameter == "pH"

    def test_las_siglas_se_respetan(self):
        for sigla, esperado in [("cya", "CYA"), ("orp", "ORP"), ("tds", "TDS")]:
            assert self._linea(_lectura(sigla, 1, "in_range")).parameter == esperado

    def test_los_guiones_bajos_se_deshacen(self):
        assert self._linea(
            _lectura("free_chlorine", 0.8, "in_range")).parameter == "Free Chlorine"

    def test_una_grafia_ya_elegida_se_respeta(self):
        assert self._linea(
            _lectura("Combined Chlorine", 0.4, "in_range")).parameter == "Combined Chlorine"

    def test_conserva_la_unidad_si_el_especialista_la_dio(self):
        # "0.8 ppm" vale más que el número pelado, y no se duplica en
        # "0.8 ppm ppm".
        assert self._linea(_lectura("fc", "0.8 ppm", "in_range")).measured == "0.8 ppm"

    def test_los_enteros_no_arrastran_el_cero(self):
        assert self._linea(_lectura("cya", 90.0, "in_range")).measured == "90 ppm"

    def test_el_valor_medido_llega_con_su_unidad(self):
        """
        El especialista emite `measured` como número pelado, así que los siete
        valores del trace 148acb15 llegaron a pantalla sin ppm. 0.8 sin unidad
        no le dice a nadie que la pileta está en infracción.
        """
        assert self._linea(
            _lectura("free_chlorine", 0.8, "below_minimum", 4.0, 2.0)).measured == "0.8 ppm"

    def test_el_ph_no_lleva_unidad(self):
        assert self._linea(_lectura("ph", 7.9, "in_range")).measured == "7.9"

    def test_la_temperatura_no_inventa_escala(self):
        """
        82 °F y 82 °C son dos piletas distintas y el especialista no dice
        cuál. Sin unidad es peor que con la correcta y mucho mejor que con la
        equivocada.
        """
        assert self._linea(_lectura("temperature", 82.0, "in_range")).measured == "82"

    def test_un_parametro_desconocido_no_inventa_unidad(self):
        assert self._linea(_lectura("turbidity", 0.3, "in_range")).measured == "0.3"


class TestReintentoPorContrato:
    """
    `needs_retry` existía desde el principio y NADIE lo consultaba: el
    docstring de enforce_contract prometía que "el caller puede reintentar UNA
    vez con instrucción correctiva" y el synthesizer nunca lo miraba.

    El caso real es `safety`: el contrato la exige cuando hay un agente de
    riesgo, el modelo la omitió, y el validador no puede inventarla — una
    línea genérica es ruido y una específica sería contenido que los
    especialistas no dieron.
    """

    def test_safety_ausente_pide_reintento(self):
        p = _payload("Cierra la pileta.", ["Cerrar a los bañistas"])
        _, rep = enforce_contract(p, get_contract("assessment"), ["chemistry"],
                                  detail_cls=DetailSection)
        assert rep.safety_missing is True
        assert rep.needs_retry is True

    def test_con_safety_no_hay_reintento(self):
        p = _payload("Cierra la pileta.", ["Cerrar a los bañistas"])
        p.safety = "No mezcles ácido con hipoclorito."
        _, rep = enforce_contract(p, get_contract("assessment"), ["chemistry"],
                                  detail_cls=DetailSection)
        assert rep.needs_retry is False

    def test_un_agente_sin_riesgo_no_exige_safety(self):
        p = _payload("Los registros se guardan cinco años.")
        _, rep = enforce_contract(p, get_contract("assessment"), ["records"],
                                  detail_cls=DetailSection)
        assert rep.needs_retry is False

    def test_el_synthesizer_consulta_needs_retry(self):
        import inspect
        from src.agent import nodes
        src = inspect.getsource(nodes.synthesizer)
        assert "report.needs_retry" in src, "el reintento seguía sin conectarse"
        assert "safety_missing" in src


class TestCanonicalizacionDeNumeros:
    def _sin_respaldo(self, visible, origen):
        from src.graph_context.response_validator import unsupported_numbers
        return unsupported_numbers(_payload(visible), origen)

    def test_un_entero_no_pierde_su_cero_final(self):
        """`"90".rstrip("0")` daba "9", así que un 90 no casaba con 90.0."""
        assert self._sin_respaldo("El cianúrico está en 90.", "cyanuric 90.0 ppm") == []

    def test_decimal_contra_entero(self):
        assert self._sin_respaldo("Apunta a 3 ppm.", '"operating_target": 3.0') == []

    def test_entero_contra_decimal(self):
        assert self._sin_respaldo("Mínimo 2.0 ppm.", "minimum is 2 ppm") == []

    def test_sigue_detectando_lo_inventado(self):
        assert "37.5" in self._sin_respaldo("Drena el 37.5%.", "cyanuric 90 ppm")

    def test_las_readings_no_cuentan_como_inventadas(self):
        """Las construye este módulo desde el payload: su origen es el dato."""
        from src.graph_context.response_contracts import ReadingLine
        from src.graph_context.response_validator import unsupported_numbers
        p = _payload("Cierra la pileta.")
        p.readings = [ReadingLine(parameter="CYA", measured="417", note="en rango")]
        assert unsupported_numbers(p, "sin cifras relevantes") == []
