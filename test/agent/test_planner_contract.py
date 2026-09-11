"""
El contrato que el planner debe cumplir, expresado sobre el schema.

Estos tests NO llaman al modelo: comprueban que las instrucciones que el
modelo recibe dicen lo que queremos que diga. Un prompt es código — se
despliega, cambia el comportamiento y se puede romper en una edición — pero
no compila, así que esta es la única red que tiene.

Origen: el trace b0496bf9. El usuario escribió "tengo 50.000 litros, el pH
está en 8.2 y el cloro no actúa, ¿qué hago?" — volumen, lectura y síntoma,
todo dado — y el planner produjo UN solo paso de `general` pidiendo cuatro
datos más. `chemistry` no llegó a ejecutarse nunca.

La causa era el Precondition Check, que ordenaba "create exactly one
clarification step, and only that step" en cuanto faltara cualquier input, y
contaba como imprescindibles el pH objetivo y el tipo de ácido — dos valores
que el dominio ya define. La regla estaba además duplicada en el prompt y
replicada en este schema, así que llegaba al modelo por dos vías.

Regla nueva: un input que falta bloquea el NÚMERO, nunca el DIAGNÓSTICO.
"""

import re

from src.agent.state import ExecutionStep, PlannerOutput
from src.prompts.prompts import PLANNER_PROMPT


def _field_doc(model, name: str) -> str:
    return model.model_fields[name].description or ""


class TestElPromptNoBloqueaElDiagnostico:
    def test_no_queda_la_orden_de_un_unico_paso_de_clarificacion(self):
        """La frase exacta que produjo el fallo del trace b0496bf9."""
        assert "only that step" not in PLANNER_PROMPT
        assert "exactly one clarification step" not in PLANNER_PROMPT

    def test_el_precondition_check_se_declara_acotado_al_numero(self):
        assert "never the DIAGNOSIS" in PLANNER_PROMPT

    def test_el_target_y_el_producto_ya_no_son_bloqueantes(self):
        assert "The target reading and the product identity are NOT required" in PLANNER_PROMPT

    def test_el_prompt_trae_el_caso_del_trace_como_ejemplo(self):
        # Un ejemplo pesa más que una regla: que el caso que falló esté
        # resuelto explícitamente en el prompt es la mejor defensa.
        assert "pH is 8.2 and the chlorine isn't working" in PLANNER_PROMPT
        assert "assigned_agent=\"chemistry\"" in PLANNER_PROMPT

    def test_la_regla_no_esta_duplicada(self):
        """
        Estaba dos veces, casi literal, y eso la reforzaba frente a la regla 8
        ("route to general only when no retrieved fact is needed"), con la que
        entra en conflicto directo.
        """
        ocurrencias = len(re.findall(r"Minimum inputs,? by request family", PLANNER_PROMPT))
        assert ocurrencias == 1, f"la regla aparece {ocurrencias} veces"

    def test_sigue_prohibido_inventar_mediciones(self):
        # Relajar el gate no puede convertirse en alucinar un volumen.
        assert "Never invent or assume a pool volume" in PLANNER_PROMPT


class TestElSchemaDiceLoMismoQueElPrompt:
    """
    El schema viaja al modelo como json_schema, así que sus `description` son
    prompt. Cuando contradecían al PLANNER_PROMPT, ganaba el schema.
    """

    def test_el_schema_ya_no_manda_las_dosis_incompletas_a_general(self):
        doc = _field_doc(ExecutionStep, "assigned_agent")
        assert "ALWAYS use 'general'" not in doc

    def test_el_schema_afirma_que_el_especialista_siempre_corre(self):
        doc = _field_doc(ExecutionStep, "assigned_agent")
        assert "blocks the NUMBER, never the DIAGNOSIS" in doc

    def test_missing_inputs_no_vacia_el_plan(self):
        doc = _field_doc(PlannerOutput, "execution_plan")
        assert "does NOT empty this list" in doc

    def test_missing_inputs_solo_lista_lo_no_inferible(self):
        doc = _field_doc(PlannerOutput, "missing_inputs")
        assert "Do NOT list the target reading or the product type" in doc


class TestElPlanDelTraceSigueSiendoConstruible:
    """
    El plan que el prompt ahora pide para el caso que falló tiene que ser
    válido contra el schema. Si el validador lo rechazara, la instrucción
    sería inalcanzable por muy bien redactada que esté.
    """

    def test_diagnostico_mas_clarificacion_es_un_plan_valido(self):
        plan = PlannerOutput(
            detected_language="es",
            execution_plan=[
                ExecutionStep(
                    step=1,
                    task="Explain why free chlorine loses sanitizing power at pH 8.2.",
                    assigned_agent="chemistry",
                ),
                ExecutionStep(
                    step=2,
                    task="Ask which acid the user has available and its strength.",
                    assigned_agent="general",
                    depends_on=[1],
                ),
            ],
            missing_inputs=["acid type"],
        )
        assert [s.assigned_agent for s in plan.execution_plan] == ["chemistry", "general"]
        # Lo que el bug impedía: que ambas cosas convivan en un mismo turno.
        assert plan.missing_inputs and plan.execution_plan

    def test_el_orchestrator_no_despacha_el_dependiente_antes_de_tiempo(self):
        """depends_on=[1] debe hacer que el paso 2 espere al diagnóstico."""
        import time

        from src.agent.nodes import orchestrator

        plan = [
            ExecutionStep(step=1, task="diagnose", assigned_agent="chemistry"),
            ExecutionStep(step=2, task="ask", assigned_agent="general", depends_on=[1]),
        ]
        cmd = orchestrator(
            {
                "execution_plan": plan,
                "agent_results": {},
                "messages": [],
                "turn_started_at": time.time(),
                "conversation_summary": "",
            }
        )
        despachados = [s.arg["step"].step for s in cmd.goto]
        assert despachados == [1], "el paso 2 no puede correr antes que su dependencia"
