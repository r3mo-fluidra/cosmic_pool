"""
Tests del suggester: de dónde saca el texto y qué gates lo filtran.

Regresión del bug de fan-out. El suggester corre en el MISMO superstep que
el synthesizer (`_to_synthesizer` emite goto=["synthesizer", "suggester"]),
así que `state["response"]` todavía es None cuando lee. La versión anterior
devolvía entonces la cadena "(sin respuesta disponible)", que no contiene
ninguna entidad ni token de dominio — y con eso los dos gates de contenido
quedaban inertes: no descartaban nada y los chips salían a ciegas.

El invariante que protegen estos tests: el suggester SIEMPRE tiene material
real contra el que medir "esto ya está respondido".
"""

from src.agent.nodes import _suggester_material
from src.agent.state import AgentResult
from src.graph_context.suggestions import (
    Suggestion,
    gate_anti_hub,
    gate_cardinality,
    gate_no_redundancy,
)


def _result(output, step=1, agent="chemistry", error=None, status="ok"):
    return AgentResult(
        agent=agent, step=step, output=output, error=error, status=status
    )


def _chip(label="¿Y el calentador?", agent="equipment", entity="heater"):
    return Suggestion(label=label, agent=agent, entity=entity)


class TestSuggesterMaterial:
    def test_usa_la_respuesta_final_cuando_existe(self):
        """Camino secuencial: si el synthesizer ya escribió, esa es la señal."""

        class _Response:
            def tier1_markdown(self):
                return "Bajá el pH a 7.4"

        material = _suggester_material({"response": _Response(), "agent_results": {}})
        assert material == "Bajá el pH a 7.4"

    def test_en_fanout_cae_a_los_outputs_de_los_agentes(self):
        """El caso real: response es None y agent_results es lo único que hay."""
        state = {
            "response": None,
            "agent_results": {"step_1": _result("El pH bajo corroe el calentador")},
        }
        assert _suggester_material(state) == "El pH bajo corroe el calentador"

    def test_nunca_devuelve_el_placeholder_del_bug(self):
        """
        El corazón de la regresión. Con response=None la versión anterior
        devolvía este literal y los gates se quedaban sin nada que medir.
        """
        state = {
            "response": None,
            "agent_results": {"step_1": _result("contenido real")},
        }
        assert "sin respuesta disponible" not in _suggester_material(state)

    def test_junta_los_outputs_de_varios_steps(self):
        state = {
            "response": None,
            "agent_results": {
                "step_1": _result("cloro combinado alto", step=1),
                "step_2": _result("revisar el filtro", step=2, agent="equipment"),
            },
        }
        material = _suggester_material(state)
        assert "cloro combinado alto" in material
        assert "revisar el filtro" in material

    def test_ignora_los_steps_fallidos(self):
        """
        Un traceback no responde nada. Si entrara al material, el gate de
        redundancia descartaría chips por culpa de un mensaje de error.
        """
        state = {
            "response": None,
            "agent_results": {
                "step_1": _result("", error="503 UNAVAILABLE", status="failed"),
            },
        }
        assert _suggester_material(state) == ""

    def test_sin_resultados_devuelve_cadena_vacia(self):
        # Vacío es correcto y manejable; el nodo corta antes de llamar al LLM.
        assert _suggester_material({"response": None, "agent_results": {}}) == ""


class TestGateNoRedundancia:
    """Este gate es el que estaba inerte: recibía el placeholder y no filtraba."""

    def test_descarta_el_chip_cuya_entidad_ya_se_respondio(self):
        chips = [_chip(entity="heater", label="¿Y el calentador?")]
        assert gate_no_redundancy(chips, "Revisá el heater por corrosión") == []

    def test_conserva_el_chip_sobre_algo_no_cubierto(self):
        chips = [_chip(entity="salt_cell", label="¿Y la celda de sal?")]
        sobreviven = gate_no_redundancy(chips, "Bajá el pH agregando ácido")
        assert len(sobreviven) == 1

    def test_con_material_vacio_no_descarta_nada(self):
        """
        Turno sin retrieval: nada que comparar, nada que filtrar. El gate
        debe ser permisivo, no bloquear por falta de señal.
        """
        chips = [_chip()]
        assert len(gate_no_redundancy(chips, "")) == 1


class TestGateAntiHub:
    def test_descarta_los_supernodos(self):
        # Un chip sobre "pH" es tan genérico que no predice nada.
        assert gate_anti_hub([_chip(entity="ph")]) == []

    def test_conserva_las_entidades_especificas(self):
        assert len(gate_anti_hub([_chip(entity="salt_cell")])) == 1


class TestGateCardinalidad:
    def test_corta_en_tres(self):
        chips = [_chip(entity=f"e{i}", label=f"Chip {i}") for i in range(6)]
        assert len(gate_cardinality(chips)) == 3

    def test_no_inventa_cuando_hay_menos(self):
        assert len(gate_cardinality([_chip()])) == 1

    def test_cero_chips_es_una_salida_valida(self):
        assert gate_cardinality([]) == []
