"""
El contrato de arquetipo: que exista, que llegue al modelo, y que la forma
no desplace la respuesta.

Origen: evaluación experta de cinco respuestas del agente de química. La peor
(2/10) fue una pregunta conceptual — "qué fracción del cloro libre es ácido
hipocloroso a pH 7.2 frente a 7.8" — que recuperó la fórmula y el pKa
correctos y se respondió con un rango operativo y tres tareas de
mantenimiento, sin un solo porcentaje.

Dos causas, ambas cubiertas acá:

1. `chemistry` tenía un único arquetipo, `assessment`, cuya forma es "veredicto
   de una frase sobre qué está fuera de rango, luego la primera verificación".
   Una pregunta conceptual no tiene nada fuera de rango: el molde no admitía la
   respuesta.

2. La sección de arquetipo NUNCA llegaba al synthesizer. Había dos funciones
   homónimas y nodes.py importaba la de response_contracts.py, que devolvía el
   NOMBRE del arquetipo y nada más. El modelo recibía la palabra "assessment"
   en lugar del contrato entero.
"""

import pytest

from src.agent.nodes import build_synthesizer_archetype_section
from src.agent.state import ExecutionStep
from src.graph_context.response_contracts import (
    ARCHETYPE_CONTRACTS,
    PRECEDENCE,
    get_contract,
    resolve_archetype,
)


class TestElContratoLlegaAlModelo:
    """Regresión del bug de las dos funciones homónimas."""

    def test_la_seccion_no_es_solo_el_nombre_del_arquetipo(self):
        s = build_synthesizer_archetype_section("assessment", ["chemistry"])
        assert s.strip() != "assessment"
        assert len(s) > 500, "el contrato renderizado no puede caber en una palabra"

    @pytest.mark.parametrize("arq", ["assessment", "calculation", "critical", "explanation"])
    def test_la_seccion_incluye_la_forma_exigida(self, arq):
        s = build_synthesizer_archetype_section(arq, ["chemistry"])
        assert get_contract(arq)["shape"][:40] in s

    def test_la_seccion_incluye_presupuesto_y_details(self):
        s = build_synthesizer_archetype_section("assessment", ["chemistry"])
        assert "budget" in s.lower()
        assert "Other possible causes" in s  # una categoría de details del contrato

    def test_el_safety_condicional_se_resuelve_por_agente(self):
        # `calculation` marca safety como "conditional"; chemistry es HAZARD.
        con_riesgo = build_synthesizer_archetype_section("calculation", ["chemistry"])
        sin_riesgo = build_synthesizer_archetype_section("calculation", ["records"])
        assert con_riesgo != sin_riesgo


class TestArquetipoExplicativo:
    def test_existe(self):
        assert "explanation" in ARCHETYPE_CONTRACTS

    def test_permite_lista_de_acciones_vacia(self):
        assert get_contract("explanation").get("actions_optional") is True

    def test_es_el_unico_que_lo_permite(self):
        """
        Si otro arquetipo se marcara opcional, el andamio de acciones dejaría
        de aplicarse donde sí hace falta.
        """
        opcionales = [k for k, v in ARCHETYPE_CONTRACTS.items() if v.get("actions_optional")]
        assert opcionales == ["explanation"]

    def test_su_forma_exige_responder_lo_que_se_preguntó(self):
        shape = get_contract("explanation")["shape"]
        assert "quantity is answered with the quantity" in shape

    def test_la_seccion_avisa_de_que_las_acciones_sobran(self):
        s = build_synthesizer_archetype_section("explanation", ["chemistry"])
        assert "may be empty" in s
        assert "belongs in `answer`" in s


class TestPrecedencia:
    """Explicar no puede enterrar una advertencia ni desplazar un número."""

    def test_una_advertencia_gana_a_la_explicacion(self):
        assert resolve_archetype(["chemistry", "safety"], explanatory=True) == "critical"

    def test_un_calculo_gana_a_la_explicacion(self):
        assert resolve_archetype(["chemistry", "math"], explanatory=True) == "calculation"

    def test_la_explicacion_gana_a_la_evaluacion(self):
        assert resolve_archetype(["chemistry"], explanatory=True) == "explanation"
        assert PRECEDENCE.index("explanation") < PRECEDENCE.index("assessment")

    def test_sin_el_flag_el_comportamiento_no_cambia(self):
        assert resolve_archetype(["chemistry"]) == "assessment"
        assert resolve_archetype(["equipment"]) == "procedure"

    def test_oos_sigue_mandando_sobre_todo(self):
        assert resolve_archetype(["chemistry"], is_oos=True, explanatory=True) == "oos"


class TestElPlannerPuedeMarcarlo:
    def test_el_step_lleva_el_flag_y_por_defecto_es_operativo(self):
        assert ExecutionStep(step=1, task="t", assigned_agent="chemistry").explanatory is False

    def test_se_puede_marcar(self):
        s = ExecutionStep(step=1, task="t", assigned_agent="chemistry", explanatory=True)
        assert s.explanatory is True

    def test_el_prompt_del_planner_explica_cuando_marcarlo(self):
        from src.prompts.prompts import PLANNER_PROMPT
        assert "explanatory=true" in PLANNER_PROMPT
        # Y que no se confunda con enrutar a `general`: ese era el fallo previo.
        assert "still `chemistry`" in PLANNER_PROMPT


class TestDisciplinaDeLecturaDeQuimica:
    """
    Los principios que la evaluación experta echó en falta. Sin cifras: los
    valores viven en el grafo, el prompt solo obliga a buscarlos.
    """

    @pytest.fixture
    def prompt(self):
        from src.prompts.prompt_archetype import build_agent_prompt
        from src.prompts.prompts_sub_agents import AGENT_REGISTRY, CHEMISTRY
        return build_agent_prompt(AGENT_REGISTRY[CHEMISTRY], "chemistry")

    def test_taxonomia_de_estados_cerrada(self, prompt):
        # "en el límite" dejó de colapsar con "fuera de rango": informar de un
        # techo exacto como violación es una mala lectura con efectos legales.
        for estado in ["below_minimum", "at_floor", "in_range", "at_ceiling", "above_maximum"]:
            assert estado in prompt

    def test_separa_limite_legal_de_objetivo_operativo(self, prompt):
        # El error más peligroso del set: dar el mínimo de código como objetivo
        # sanitario deja al operador reabriendo una piscina sin desinfectar.
        assert "REGULATORY limit from an OPERATING target" in prompt
        assert "never state either from memory" in prompt

    def test_exige_clasificar_todos_los_parametros(self, prompt):
        assert "A reading you do not mention" in prompt

    def test_exige_leer_las_interacciones(self, prompt):
        assert "as a system, not as a column of independent values" in prompt
        assert "Retrieve the interaction rules; do not infer them." in prompt

    def test_exige_nombrar_la_causa_del_patron(self, prompt):
        assert "likely_cause" in prompt

    def test_desconfia_de_los_instrumentos(self, prompt):
        assert "claim that can be wrong" in prompt

    def test_exige_cuantificar_las_instrucciones(self, prompt):
        assert "are not instructions" in prompt

    def test_no_hardcodea_valores_de_dominio(self, prompt):
        """
        Decisión de diseño: los números viven en el grafo, no acá. Un umbral
        escrito en el prompt es una segunda fuente de verdad que nadie
        actualiza cuando cambia el código aplicable.
        """
        import re
        for patron in [r"\b7\.5\s*%", r"\bpKa\s*7\.5", r"\b90\s*ppm", r"\b7\.5\s*ppm"]:
            assert not re.search(patron, prompt), f"valor de dominio hardcodeado: {patron}"


# =====================================================================
# Interpretación de panel: el synthesizer recibía los datos y los tiraba
# =====================================================================
# Origen: trace 75ba3706, tercera evaluación de B1 (siete lecturas de agua).
#
# `chemistry` produjo un test_interpretation completo: status por parámetro,
# operating_target, source_id, interactions y un likely_cause que nombraba el
# tricloro. El synthesizer emitió 285 caracteres mencionando DOS parámetros,
# sin target y sin causa.
#
# No fue el presupuesto — son 900 palabras y se usaron ~40. Fue la forma:
# "One-sentence verdict: what is out of range". Con siete lecturas, un
# veredicto de una frase obliga a descartar cinco.
#
# Y un fallo cruzado: chemistry escribió "extremely high" sobre un valor que
# su propio status clasificaba como at_ceiling — cumplimiento sin margen. La
# prosa contradecía al dato estructurado, y el synthesizer creyó a la prosa.


class TestInterpretacionDePanel:
    @pytest.fixture
    def shape(self):
        return get_contract("assessment")["shape"]

    def test_el_veredicto_es_la_apertura_no_la_respuesta_entera(self, shape):
        assert "not the whole answer" in shape

    def test_exige_nombrar_que_lectura_dispara_el_cierre(self, shape):
        # Atribuir un cierre a un parámetro que cumple es informar de una
        # infracción inexistente.
        assert "WHICH single reading triggers it" in shape

    def test_exige_cubrir_toda_lectura_fuera_de_rango(self, shape):
        assert "reads as a reading you found acceptable" in shape

    def test_exige_el_objetivo_operativo_y_la_causa(self, shape):
        assert "operating target" in shape
        assert "likely cause" in shape

    def test_hay_un_details_para_el_desglose_completo(self):
        assert "Full reading breakdown" in get_contract("assessment")["details"]

    def test_la_seccion_renderizada_llega_con_todo(self):
        s = build_synthesizer_archetype_section("assessment", ["chemistry"])
        assert "WHICH single reading triggers it" in s
        assert "Full reading breakdown" in s


class TestElSynthesizerHonraLosStatus:
    @pytest.fixture
    def P(self):
        from src.prompts.prompts import SYNTHESIZER_PROMPT
        return SYNTHESIZER_PROMPT

    def test_declara_los_cinco_estados(self, P):
        for e in ["below_minimum", "at_floor", "in_range", "at_ceiling", "above_maximum"]:
            assert e in P

    def test_at_ceiling_y_at_floor_son_cumplimiento(self, P):
        assert "`at_floor` and `at_ceiling` are PASSES" in P

    def test_prohibe_intensificar_por_encima_del_status(self, P):
        # El caso literal del trace: "high" -> "extremely high".
        assert "Never intensify past the status" in P
        assert 'you do not\nwrite "extremely high"' in P

    def test_exige_atribuir_el_cierre_a_su_causa(self, P):
        assert "Attribute a closure to the reading that causes it" in P

    def test_arrastra_target_causa_y_orden(self, P):
        for campo in ["`operating_target`", "`likely_cause`", "`order_rationale`"]:
            assert campo in P

    def test_el_input_que_falta_no_bloquea_el_target(self, P):
        # El error de razonamiento del trace: el volumen hace falta para la
        # DOSIS, no para la concentración objetivo.
        assert "does not block stating the target concentration" in P

    def test_prohibe_dictaminar_inspecciones(self, P):
        assert "fails inspection" in P
        assert "only an\nauthority can" in P


class TestChemistryNoSeContradice:
    @pytest.fixture
    def prompt(self):
        from src.prompts.prompt_archetype import build_agent_prompt
        from src.prompts.prompts_sub_agents import AGENT_REGISTRY, CHEMISTRY
        return build_agent_prompt(AGENT_REGISTRY[CHEMISTRY], "chemistry")

    def test_la_prosa_debe_coincidir_con_los_status(self, prompt):
        assert "Keep your prose consistent with your own statuses" in prompt

    def test_exige_aplicar_las_reglas_dependientes_recuperadas(self, prompt):
        # Recuperó higher_fc_minimum_with_cyanurates y aun así dio el rango
        # genérico: el retrieval acertó y el uso no.
        assert "APPLY it and show the result" in prompt

    def test_exige_ordenar_por_efecto_fisico(self, prompt):
        # Clorar y luego diluir el 50% tira la mitad del cloro recién añadido.
        assert "removes\n  whatever was added before it" in prompt or \
               "removes whatever was added before it" in prompt
