"""
Unit tests for Equipment and Maintenance tools.

External dependencies are mocked through:

- _execute_cypher()
- get_vector_store()

These tests validate only the business logic.
"""

from unittest.mock import patch

import src.agent.tools as tools


# ============================================================
# EQUIPMENT TOOLS
# ============================================================


class TestEquipmentTools:

    @patch("src.agent.tools._execute_cypher")
    def test_query_hardware_impact_by_parameter(
        self,
        mock_execute,
        mock_neo4j_results,
    ):
        """
        Should search equipment affected by a parameter.
        """

        mock_execute.return_value = mock_neo4j_results

        result = tools.query_hardware_impact(
            parameter_id="PH"
        )

        assert result == mock_neo4j_results

        query, params = mock_execute.call_args.args

        assert "Equipment" in query

        assert params == {
            "parameter_id": "PH"
        }


    @patch("src.agent.tools._execute_cypher")
    def test_query_hardware_impact_by_equipment(
        self,
        mock_execute,
        mock_neo4j_results,
    ):
        """
        Should search parameters affecting one equipment.
        """

        mock_execute.return_value = mock_neo4j_results

        result = tools.query_hardware_impact(
            equipment_id="PUMP001"
        )

        assert result == mock_neo4j_results

        _, params = mock_execute.call_args.args

        assert params == {
            "equipment_id": "PUMP001"
        }


    @patch("src.agent.tools._execute_cypher")
    def test_query_hardware_impact_by_parameter_and_equipment(
        self,
        mock_execute,
        mock_neo4j_results,
    ):
        """
        Should filter by parameter and equipment.
        """

        mock_execute.return_value = mock_neo4j_results

        result = tools.query_hardware_impact(
            parameter_id="PH",
            equipment_id="PUMP001",
        )

        assert result == mock_neo4j_results

        _, params = mock_execute.call_args.args

        assert params == {
            "parameter_id": "PH",
            "equipment_id": "PUMP001",
        }


    def test_query_hardware_impact_requires_arguments(self):
        """
        Should return an error if no filters are provided.
        """

        result = tools.query_hardware_impact()

        assert result == [
            {
                "error": "Must provide either parameter_id or equipment_id"
            }
        ]


    @patch("src.agent.tools.get_vector_store")
    def test_search_equipment_manuals(
        self,
        mock_get_store,
        mock_vector_store,
    ):
        """
        Should return formatted manual excerpts.
        """

        mock_get_store.return_value = mock_vector_store

        result = tools.search_equipment_manuals(
            "pump leaking"
        )

        assert len(result) == 1

        assert result[0]["manual_excerpt"] == "Example page content"

        assert result[0]["tags"] == [
            "pump",
            "filter",
        ]

        mock_vector_store.similarity_search.assert_called_once()


    @patch("src.agent.tools.get_vector_store")
    def test_search_equipment_manuals_store_exception(
        self,
        mock_get_store,
    ):
        """
        Should gracefully handle Vector Store failures.
        """

        mock_get_store.side_effect = RuntimeError(
            "Qdrant unavailable"
        )

        result = tools.search_equipment_manuals(
            "pump"
        )

        assert result == [
            {
                "error":
                "Vector store initialization failed: Qdrant unavailable"
            }
        ]


# ============================================================
# MAINTENANCE TOOLS
# ============================================================


class TestMaintenanceTools:

    @patch("src.agent.tools.get_vector_store")
    def test_search_maintenance_procedures(
        self,
        mock_get_store,
        mock_vector_store,
    ):
        """
        Should search maintenance procedures.
        """

        mock_get_store.return_value = mock_vector_store

        result = tools.search_maintenance_procedures(
            "backwash"
        )

        assert len(result) == 1

        assert result[0]["checklist_and_safety"] == "Example page content"

        assert result[0]["source"] == "manual.pdf"

        mock_vector_store.similarity_search.assert_called_once()


    @patch("src.agent.tools.get_vector_store")
    def test_search_maintenance_procedures_with_environment(
        self,
        mock_get_store,
        mock_vector_store,
    ):
        """
        Should enrich the semantic query when an
        environmental factor is provided.
        """

        mock_get_store.return_value = mock_vector_store

        tools.search_maintenance_procedures(
            procedure_type="winterization",
            environmental_factor="heavy rain",
        )

        args = mock_vector_store.similarity_search.call_args.args

        assert (
            "winterization dealing with heavy rain"
            in args[0]
        )


    @patch("src.agent.tools.get_vector_store")
    def test_search_maintenance_procedures_store_exception(
        self,
        mock_get_store,
    ):
        """
        Should return a formatted error if the
        vector store cannot be initialized.
        """

        mock_get_store.side_effect = RuntimeError(
            "Vector DB offline"
        )

        result = tools.search_maintenance_procedures(
            "backwash"
        )

        assert result == [
            {
                "error":
                "Vector store initialization failed: Vector DB offline"
            }
        ]


    @patch("src.agent.tools._execute_cypher")
    def test_query_maintenance_dependencies(
        self,
        mock_execute,
        mock_neo4j_results,
    ):
        """
        Should query maintenance dependencies.
        """

        mock_execute.return_value = mock_neo4j_results

        result = tools.query_maintenance_dependencies(
            "TASK001"
        )

        assert result == mock_neo4j_results

        query, params = mock_execute.call_args.args

        assert "BALANCE_BEFORE" in query

        assert params == {
            "node_id": "TASK001"
        }