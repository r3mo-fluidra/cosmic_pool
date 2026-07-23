"""
Unit tests for Diagnosis and Dosage tools.

External services are mocked through:

- _execute_cypher()
- get_vector_store()

These tests validate only the business logic of each tool.
"""

from unittest.mock import patch

import src.agent.tools as tools


# ============================================================
# DIAGNOSIS TOOLS
# ============================================================


class TestDiagnosisTools:

    @patch("src.agent.tools._execute_cypher")
    def test_query_symptom_graph(self, mock_execute, mock_neo4j_results):
        """
        Should delegate the Cypher execution and return its results.
        """

        mock_execute.return_value = mock_neo4j_results

        result = tools.query_symptom_graph("green water")

        assert result == mock_neo4j_results

        mock_execute.assert_called_once()

        query, params = mock_execute.call_args.args

        assert "MATCH" in query
        assert params == {
            "symptom_keyword": "green water"
        }


    @patch("src.agent.tools.get_vector_store")
    def test_search_troubleshooting_kb(
        self,
        mock_get_store,
        mock_vector_store,
    ):
        """
        Should perform a similarity search and format the response.
        """

        mock_get_store.return_value = mock_vector_store

        result = tools.search_troubleshooting_kb(
            "cloudy water",
            limit=5,
        )

        assert len(result) == 1

        assert result[0]["content"] == "Example page content"
        assert result[0]["source"] == "manual.pdf"
        assert result[0]["category"] == "Maintenance"

        mock_vector_store.similarity_search.assert_called_once()


    @patch("src.agent.tools.get_vector_store")
    def test_search_troubleshooting_kb_store_exception(
        self,
        mock_get_store,
    ):
        """
        Should gracefully return an error if the vector store
        cannot be initialized.
        """

        mock_get_store.side_effect = RuntimeError(
            "Vector DB unavailable"
        )

        result = tools.search_troubleshooting_kb(
            "green pool"
        )

        assert result == [
            {
                "error":
                "Vector store initialization failed: Vector DB unavailable"
            }
        ]


# ============================================================
# DOSAGE TOOLS
# ============================================================


class TestDosageTools:

    @patch("src.agent.tools._execute_cypher")
    def test_query_chemical_actions_without_filter(
        self,
        mock_execute,
        mock_neo4j_results,
    ):
        """
        Should retrieve every action for a parameter.
        """

        mock_execute.return_value = mock_neo4j_results

        result = tools.query_chemical_actions("PH")

        assert result == mock_neo4j_results

        query, params = mock_execute.call_args.args

        assert "RAISES" in query
        assert "LOWERS" in query

        assert params == {
            "parameter_id": "PH"
        }


    @patch("src.agent.tools._execute_cypher")
    def test_query_chemical_actions_raise_only(
        self,
        mock_execute,
        mock_neo4j_results,
    ):
        """
        Should add the RAISES filter.
        """

        mock_execute.return_value = mock_neo4j_results

        tools.query_chemical_actions(
            "PH",
            desired_action="RAISES",
        )

        query, _ = mock_execute.call_args.args

        assert "STARTS WITH 'RAISES'" in query


    @patch("src.agent.tools._execute_cypher")
    def test_query_chemical_actions_lower_only(
        self,
        mock_execute,
        mock_neo4j_results,
    ):
        """
        Should add the LOWERS filter.
        """

        mock_execute.return_value = mock_neo4j_results

        tools.query_chemical_actions(
            "PH",
            desired_action="LOWERS",
        )

        query, _ = mock_execute.call_args.args

        assert "STARTS WITH 'LOWERS'" in query


    @patch("src.agent.tools.get_vector_store")
    def test_get_dosing_formulas(
        self,
        mock_get_store,
        mock_vector_store,
    ):
        """
        Should return formatted dosing instructions.
        """

        mock_get_store.return_value = mock_vector_store

        result = tools.get_dosing_formulas(
            "C_MuriaticAcid"
        )

        assert len(result) == 1

        assert result[0]["instructions"] == "Example page content"
        assert result[0]["category"] == "Maintenance"

        assert "context" not in result[0]

        mock_vector_store.similarity_search.assert_called_once()


    @patch("src.agent.tools.get_vector_store")
    def test_get_dosing_formulas_with_pool_volume(
        self,
        mock_get_store,
        mock_vector_store,
    ):
        """
        Should include dynamic pool-volume context.
        """

        mock_get_store.return_value = mock_vector_store

        result = tools.get_dosing_formulas(
            "C_MuriaticAcid",
            pool_volume_kL=45,
        )

        assert result[0]["context"] == (
            "Apply these formulas using the user's pool volume: 45 kL."
        )


    @patch("src.agent.tools.get_vector_store")
    def test_get_dosing_formulas_vector_store_exception(
        self,
        mock_get_store,
    ):
        """
        Should return an error if the vector store fails.
        """

        mock_get_store.side_effect = RuntimeError(
            "Vector store unavailable"
        )

        result = tools.get_dosing_formulas(
            "C_MuriaticAcid"
        )

        assert result == [
            {
                "error":
                "Vector store initialization failed: Vector store unavailable"
            }
        ]