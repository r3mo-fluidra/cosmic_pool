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

    def test_el_panel_no_lo_escribe_el_modelo(self, P):
        """
        Cuatro rondas mostraron que el fraseo por estado no se sostiene como
        instrucción. Ahora se genera por plantilla y al modelo se le dice que
        no lo escriba, en vez de pedirle que lo escriba bien.
        """
        assert "The panel is generated, not written" in P
        assert "leave it empty. Always" in P

    def test_at_ceiling_sigue_siendo_cumplimiento(self, P):
        assert "compliant-with-no-margin" in P

    def test_prohibe_intensificar_en_la_prosa(self, P):
        # El caso literal del trace: "high" -> "extremely high".
        assert 'no "extremely high"' in P
        assert "misstates the facility's regulatory position" in P

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


# =====================================================================
# Conflicto de restricción: por qué diluir es una conclusión, no una tarea
# =====================================================================
# El corpus (CH13-13.4) reconoce el ratio FC/CYA — "captura correctamente la
# dirección del efecto real" — y se niega a darle un número: "circulan varios
# porcentajes", "expresamente no es un estándar regulatorio". Y resuelve el
# conflicto: "donde un cálculo por ratio implique un cloro libre por encima
# del máximo permitido, la respuesta correcta es reducir el cianúrico".
#
# Con el rango del corpus en 1-4 ppm (nodo free_chlorine), un CYA de 90 pide
# por ratio ~6.75 ppm: fuera de rango. De ahí que la vía sea diluir.
#
# El sistema recomendaba diluir y acertaba, pero sin decir por qué. Un
# operador que recibe "drena la mitad" sin la razón lo pospone o lo deshace;
# uno que entiende que no puede desinfectar a ese nivel de estabilizador, no.
#
# El peligro simétrico, y el motivo de que el target solo no baste: 3 ppm
# entregados sin el conflicto parecen alcanzables y suficientes. El operador
# los dosifica, reabre, y el agua sigue sin desinfectar. Fue el error de la
# primera versión de B1 con su "por encima de 2.0 ppm".


class TestConflictoDeRestriccion:
    @pytest.fixture
    def chem(self):
        from src.prompts.prompt_archetype import build_agent_prompt
        from src.prompts.prompts_sub_agents import AGENT_REGISTRY, CHEMISTRY
        return build_agent_prompt(AGENT_REGISTRY[CHEMISTRY], "chemistry")

    @pytest.fixture
    def synth(self):
        from src.prompts.prompts import SYNTHESIZER_PROMPT
        return SYNTHESIZER_PROMPT

    def test_un_target_fuera_de_rango_es_el_hallazgo(self, chem):
        assert "that is the finding" in chem
        assert "do not quietly clamp it back" in chem

    def test_la_correccion_pasa_al_otro_parametro(self, chem):
        assert "belongs to the OTHER parameter" in chem

    def test_exige_las_tres_partes_del_conflicto(self, chem):
        assert "the level that would be needed" in chem
        assert "the bound that\nforbids it" in chem or "the bound that forbids it" in chem
        assert "what has to change instead" in chem

    def test_prohibe_publicar_un_target_fuera_de_limites(self, chem):
        # Dar 6.75 ppm a un operador cuyo código tope en 4 lo pone en
        # infracción: la química no autoriza a saltarse el código.
        assert "Never publish a target above a permitted maximum" in chem

    def test_prohibe_el_target_a_secas_que_es_la_mitad_peligrosa(self, chem):
        # El error de la primera versión: "sube por encima de 2.0 ppm".
        assert "The number without the conflict is the more dangerous half" in chem

    def test_el_campo_esta_en_el_contrato_de_salida(self):
        from src.prompts.prompts_sub_agents import AGENT_REGISTRY, CHEMISTRY
        oc = AGENT_REGISTRY[CHEMISTRY].output_contract
        assert "constraint_conflict" in oc
        for parte in ["needed_level", "blocking_bound", "parameter_to_correct"]:
            assert parte in oc

    def test_el_synthesizer_lo_arrastra_al_tier_visible(self, synth):
        assert "`constraint_conflict`" in synth
        assert "what level would be needed" in synth

    def test_el_synthesizer_no_puede_dar_el_target_solo(self, synth):
        assert "Never present the in-range target on its own" in synth

    def test_sigue_sin_hardcodear_el_ratio(self, chem):
        """
        La decisión se mantiene: el corpus no da un porcentaje, y el prompt
        tampoco lo inventa. El razonamiento es genérico; los números salen del
        grafo o no salen.
        """
        import re
        for patron in [r"\b7\.5\s*%", r"\b6\.75", r"\b1\s*to\s*4\s*ppm"]:
            assert not re.search(patron, chem), f"valor de dominio hardcodeado: {patron}"


class TestLimiteVsObjetivo:
    """
    El agente etiquetó regulatory_limit=400 para dureza de calcio. El corpus
    dice "typical educational targets run from about 150 to 400 ppm ... some
    codes permitting higher maxima": es un objetivo de industria, y el propio
    texto avisa de que hay códigos más laxos.

    Inventar un límite MÁS ESTRICTO que el código es el mismo error que
    llamar violación a un techo, con el signo cambiado: hace que una
    instalación en regla parezca estarlo incumpliendo.
    """

    @pytest.fixture
    def oc(self):
        from src.prompts.prompts_sub_agents import AGENT_REGISTRY, CHEMISTRY
        return AGENT_REGISTRY[CHEMISTRY].output_contract

    def test_limite_solo_si_la_fuente_lo_presenta_como_codigo(self, oc):
        assert "ONLY a value the source presents as a code" in oc

    def test_reconoce_el_lenguaje_de_los_objetivos_educativos(self, oc):
        assert "typical target" in oc and "educational range" in oc

    def test_sin_evidencia_de_codigo_el_limite_es_null(self, oc):
        assert "regulatory_limit is null" in oc

    def test_nombra_el_riesgo_de_inventar_un_limite_estricto(self, oc):
        assert "stricter than the code makes a compliant" in oc

    def test_el_target_no_puede_ser_eco_del_medido(self, oc):
        # Temperatura devolvió operating_target=82.0 con 82°F medidos.
        assert "never echo the measured value back into it" in oc

    def test_el_status_se_juzga_contra_el_limite(self, oc):
        assert "never against a target" in oc


class TestPracticaDeIndustriaVsCodigo:
    """
    La relación proporcional FC/CYA se añadió al grafo como práctica de
    industria (fc_cya_proportional_target), no como norma: el corpus se niega
    expresamente a darle un número y advierte de que no es estándar
    regulatorio. El nodo arrastra esa advertencia por QUALIFIED_BY.

    El prompt tiene que mantener la distinción, porque el riesgo es simétrico:
    citar una práctica como si fuera código expone al operador ante un
    inspector, y descartarla lo deja sin la razón por la que hay que diluir.
    """

    @pytest.fixture
    def prompt(self):
        from src.prompts.prompt_archetype import build_agent_prompt
        from src.prompts.prompts_sub_agents import AGENT_REGISTRY, CHEMISTRY
        return build_agent_prompt(AGENT_REGISTRY[CHEMISTRY], "chemistry")

    def test_separa_practica_de_requisito(self, prompt):
        assert "Keep industry practice and code requirement apart" in prompt

    def test_prohibe_atribuirla_a_una_seccion_de_codigo(self, prompt):
        assert "never\nattribute it to a code section" in prompt or \
               "never attribute it to a code section" in prompt

    def test_el_status_se_juzga_contra_la_norma_no_contra_la_practica(self, prompt):
        assert "A `status` is judged against the\nregulatory bound" in prompt or \
               "A `status` is judged against the regulatory bound" in prompt

    def test_puede_informar_el_objetivo_operativo(self, prompt):
        # Descartarla dejaría al operador sin el porqué de la dilución.
        assert "can inform the" in prompt and "operating_target" in prompt
