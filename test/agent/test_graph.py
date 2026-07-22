"""
Tests for src.agent.graph

✅ StateGraph(PoolAgentState)
✅ Registro de los 5 nodos
✅ Destinations de cada nodo
✅ START -> build_context_node
✅ synthesizer -> END
✅ Creación automática de InMemorySaver
✅ Uso de un checkpointer personalizado
✅ compile(checkpointer=...)
✅ Valor retornado (app)

"""

from unittest.mock import MagicMock, patch

import src.agent.graph as graph_module


@patch("src.agent.graph.InMemorySaver")
@patch("src.agent.graph.StateGraph")
def test_build_graph_creates_expected_graph(
    mock_state_graph,
    mock_inmemory_saver,
):
    """
    Verify build_graph registers the expected nodes,
    edges and compiles the graph with the default checkpointer.
    """

    builder = MagicMock()
    app = MagicMock()

    mock_state_graph.return_value = builder
    builder.compile.return_value = app

    checkpointer = MagicMock()
    mock_inmemory_saver.return_value = checkpointer

    result = graph_module.build_graph()

    mock_state_graph.assert_called_once_with(graph_module.PoolAgentState)

    assert builder.add_node.call_count == 5

    builder.add_node.assert_any_call(
        "build_context_node",
        graph_module.build_context_node,
        destinations=["summarize_memory_node", "planner"],
    )

    builder.add_node.assert_any_call(
        "summarize_memory_node",
        graph_module.summarize_memory_node,
        destinations=["planner"],
    )

    builder.add_node.assert_any_call(
        "planner",
        graph_module.planner,
        destinations=["orchestrator"],
    )

    builder.add_node.assert_any_call(
        "orchestrator",
        graph_module.orchestrator,
        destinations=["orchestrator", "synthesizer"],
    )

    builder.add_node.assert_any_call(
        "synthesizer",
        graph_module.synthesizer,
    )

    builder.add_edge.assert_any_call(graph_module.START, "build_context_node")
    builder.add_edge.assert_any_call("synthesizer", graph_module.END)

    mock_inmemory_saver.assert_called_once()

    builder.compile.assert_called_once_with(checkpointer=checkpointer)

    assert result is app


@patch("src.agent.graph.StateGraph")
def test_build_graph_uses_custom_checkpointer(mock_state_graph):
    """
    Verify that a provided checkpointer is used instead
    of creating an InMemorySaver.
    """

    builder = MagicMock()
    app = MagicMock()

    mock_state_graph.return_value = builder
    builder.compile.return_value = app

    custom_checkpointer = MagicMock()

    with patch("src.agent.graph.InMemorySaver") as mock_inmemory:
        result = graph_module.build_graph(custom_checkpointer)

    mock_inmemory.assert_not_called()

    builder.compile.assert_called_once_with(
        checkpointer=custom_checkpointer
    )

    assert result is app


@patch("src.agent.graph.build_graph")
def test_default_graph_is_created_from_build_graph(mock_build_graph):
    """
    Verify the module-level graph export is created from build_graph().
    """

    app = MagicMock()
    mock_build_graph.return_value = app

    import importlib

    importlib.reload(graph_module)

    assert graph_module.graph is not None