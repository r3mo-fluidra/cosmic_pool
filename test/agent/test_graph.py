"""
Tests de src.agent.graph.

Verifican el cableado, que es lo único que este módulo hace: qué nodos se
registran, con qué destinos declarados, y cómo se cierra el grafo.

Los `destinations` no son documentación: LangGraph los usa para validar los
`Command(goto=...)` en runtime. Un destino que falta convierte un salto
legítimo en un error de ejecución, así que se afirman uno por uno.
"""

from unittest.mock import MagicMock, patch

import pytest

import src.agent.graph as graph_module


EXPECTED_NODES = {
    "build_context_node": ["summarize_memory_node", "planner"],
    "summarize_memory_node": ["planner"],
    "planner": ["orchestrator", "general", "oos"],
    "general": ["synthesizer"],
    "oos": ["synthesizer"],
    "orchestrator": ["run_step", "synthesizer", "suggester"],
    "run_step": ["orchestrator"],
}


@pytest.fixture
def built_graph():
    """build_graph() con StateGraph e InMemorySaver mockeados."""
    with patch("src.agent.graph.InMemorySaver") as mock_saver, \
         patch("src.agent.graph.StateGraph") as mock_state_graph:
        builder = MagicMock()
        app = MagicMock()
        mock_state_graph.return_value = builder
        builder.compile.return_value = app

        result = graph_module.build_graph()

        yield {
            "builder": builder,
            "app": app,
            "result": result,
            "state_graph": mock_state_graph,
            "saver": mock_saver,
        }


def test_graph_is_typed_with_pool_agent_state(built_graph):
    built_graph["state_graph"].assert_called_once_with(graph_module.PoolAgentState)


@pytest.mark.parametrize("name,destinations", sorted(EXPECTED_NODES.items()))
def test_routing_node_declares_its_destinations(built_graph, name, destinations):
    built_graph["builder"].add_node.assert_any_call(
        name,
        getattr(graph_module, "run_step_node" if name == "run_step" else name),
        destinations=destinations,
    )


def test_terminal_nodes_are_registered_without_destinations(built_graph):
    builder = built_graph["builder"]

    builder.add_node.assert_any_call("synthesizer", graph_module.synthesizer)
    builder.add_node.assert_any_call("suggester", graph_module.suggester)


def test_registers_exactly_the_expected_nodes(built_graph):
    registered = {
        call.args[0] for call in built_graph["builder"].add_node.call_args_list
    }

    assert registered == set(EXPECTED_NODES) | {"synthesizer", "suggester"}


def test_entry_point_is_build_context_node(built_graph):
    built_graph["builder"].add_edge.assert_any_call(
        graph_module.START, "build_context_node"
    )


def test_both_fan_out_branches_close_the_turn(built_graph):
    """synthesizer y suggester corren en paralelo; el turno cierra con ambos."""
    builder = built_graph["builder"]

    builder.add_edge.assert_any_call("synthesizer", graph_module.END)
    builder.add_edge.assert_any_call("suggester", graph_module.END)


def test_compiles_with_an_in_memory_saver_by_default(built_graph):
    built_graph["saver"].assert_called_once()
    built_graph["builder"].compile.assert_called_once_with(
        checkpointer=built_graph["saver"].return_value
    )
    assert built_graph["result"] is built_graph["app"]


@patch("src.agent.graph.StateGraph")
def test_a_provided_checkpointer_replaces_the_default(mock_state_graph):
    builder = MagicMock()
    app = MagicMock()
    mock_state_graph.return_value = builder
    builder.compile.return_value = app

    custom_checkpointer = MagicMock()

    with patch("src.agent.graph.InMemorySaver") as mock_inmemory:
        result = graph_module.build_graph(custom_checkpointer)

    mock_inmemory.assert_not_called()
    builder.compile.assert_called_once_with(checkpointer=custom_checkpointer)
    assert result is app


def test_module_exports_a_compiled_graph():
    assert graph_module.graph is not None
