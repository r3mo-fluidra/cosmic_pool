"""
Tests del gate determinista de MATH.

Regresión de un bug que estuvo vivo sin dar la cara porque eran DOS bugs que
se anulaban:

  1. nodes.py comparaba `step.assigned_agent == MATH`, con
     MATH = "Pool Math Agent" (un nombre de display) contra un slug "math".
     False siempre: el gate no se ejecutó nunca.
  2. gates.py construía `AgentResult(agent=MATH)`, y "Pool Math Agent" no es
     un AgentName válido. Habría sido un ValidationError... si el gate
     hubiera llegado a dispararse alguna vez.

O sea: el primer bug tapaba al segundo. El coste era que una consulta de
dosificación sin números se comía un ReAct loop entero para concluir lo que
un `if` resuelve en microsegundos.
"""

import pytest
from pydantic import ValidationError

from src.agent.agent_names import MATH_SLUG
from src.agent.gates import math_inputs_present, missing_inputs_result
from src.agent.nodes import _normalize_agent
from src.agent.state import AgentResult, ExecutionStep
from src.prompts.prompts_sub_agents import MATH


def _step(agent="math"):
    return ExecutionStep(step=1, task="calcular dosis de ácido", assigned_agent=agent)


class TestSlugVsDisplayName:
    """La distinción que costó el gate. Si esto se vuelve a mezclar, falla."""

    def test_math_es_un_nombre_de_display_no_un_identificador(self):
        assert MATH == "Pool Math Agent"
        assert MATH != MATH_SLUG

    def test_el_slug_es_lo_que_produce_el_planner(self):
        assert _normalize_agent(_step().assigned_agent) == MATH_SLUG

    def test_comparar_contra_el_display_name_nunca_acierta(self):
        # Exactamente la condición que tenía nodes.py.
        assert _step().assigned_agent != MATH

    def test_el_display_name_no_es_un_agentname_valido(self):
        # El ValidationError que el otro bug mantenía escondido.
        with pytest.raises(ValidationError):
            AgentResult(agent=MATH, step=1, output="")


class TestMathInputsPresent:
    @pytest.mark.parametrize(
        "mensaje",
        [
            "tengo 50000 litros y el pH en 7.8",
            "subir cloro a 3 ppm",
            "1",
        ],
    )
    def test_con_digitos_deja_pasar(self, mensaje):
        assert math_inputs_present(mensaje) is True

    @pytest.mark.parametrize(
        "mensaje",
        [
            "¿cuánto ácido le pongo a la pileta?",
            "necesito bajar el pH",
            "",
        ],
    )
    def test_sin_digitos_corta(self, mensaje):
        assert math_inputs_present(mensaje) is False

    def test_none_no_revienta(self):
        assert math_inputs_present(None) is False


class TestMissingInputsResult:
    def test_produce_un_agentresult_valido(self):
        """El test de regresión directo: antes esto lanzaba ValidationError."""
        result = missing_inputs_result(_step(), "cuánto ácido?")
        assert result.agent == MATH_SLUG
        assert result.step == 1

    def test_el_output_es_un_marcador_no_prosa(self):
        # El synthesizer sigue siendo dueño de la redacción y del idioma.
        # Devolver prosa acá le haría parafrasear trabajo ya hecho.
        result = missing_inputs_result(_step(), "cuánto ácido?")
        assert result.output.startswith("STATUS: MISSING_INPUTS")

    def test_pide_los_parametros_que_faltan(self):
        result = missing_inputs_result(_step(), "cuánto ácido?")
        assert "volume" in result.output.lower()
