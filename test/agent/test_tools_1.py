"""
Tests for the infrastructure layer in src.agent.tools.

This module validates:

- Secret resolution
- Neo4j driver initialization
- Cypher execution helper
- Vector store initialization

All higher-level tests rely on these utilities.
"""

from unittest.mock import MagicMock, patch

import pytest

import src.agent.tools as tools


# ============================================================
# _get_secret
# ============================================================


class TestGetSecret:

    @patch("src.agent.tools.os.getenv")
    @patch("src.agent.tools.secrets")
    def test_returns_environment_variable_when_present(
        self,
        mock_secrets,
        mock_getenv,
    ):
        mock_getenv.return_value = "value_from_env"

        result = tools._get_secret("TEST_KEY")

        assert result == "value_from_env"

        mock_secrets.get.assert_not_called()


    @patch("src.agent.tools.os.getenv")
    @patch("src.agent.tools.secrets")
    def test_returns_streamlit_secret_when_env_missing(
        self,
        mock_secrets,
        mock_getenv,
    ):
        mock_getenv.return_value = None
        mock_secrets.get.return_value = "value_from_secret"

        result = tools._get_secret("TEST_KEY")

        assert result == "value_from_secret"


    @patch("src.agent.tools.os.getenv")
    @patch("src.agent.tools.secrets")
    def test_returns_default_when_missing_everywhere(
        self,
        mock_secrets,
        mock_getenv,
    ):
        mock_getenv.return_value = None
        mock_secrets.get.return_value = None

        result = tools._get_secret("TEST_KEY", "default")

        assert result == "default"


# ============================================================
# get_vector_store
# ============================================================


class TestVectorStore:

    def setup_method(self):
        tools.GLOBAL_VECTOR_STORE = None


    @patch("src.agent.tools.cargar_vector_store")
    def test_initializes_vector_store_once(
        self,
        mock_loader,
        mock_vector_store,
    ):
        mock_loader.return_value = mock_vector_store

        store = tools.get_vector_store()

        assert store is mock_vector_store

        mock_loader.assert_called_once()


    @patch("src.agent.tools.cargar_vector_store")
    def test_returns_cached_vector_store(
        self,
        mock_loader,
        mock_vector_store,
    ):
        mock_loader.return_value = mock_vector_store

        first = tools.get_vector_store()
        second = tools.get_vector_store()

        assert first is second

        mock_loader.assert_called_once()


# ============================================================
# get_neo4j_driver
# ============================================================


class TestNeo4jDriver:

    def setup_method(self):
        tools._neo4j_driver = None


    @patch("src.agent.tools.GraphDatabase.driver")
    @patch("src.agent.tools._get_secret")
    def test_creates_driver_only_once(
        self,
        mock_secret,
        mock_driver_factory,
        mock_driver,
    ):
        mock_secret.side_effect = [
            "neo4j+s://localhost",
            "neo4j",
            "password",
        ]

        mock_driver_factory.return_value = mock_driver

        driver = tools.get_neo4j_driver()

        assert driver is mock_driver

        mock_driver_factory.assert_called_once_with(
            "neo4j+s://localhost",
            auth=("neo4j", "password"),
        )


    @patch("src.agent.tools.GraphDatabase.driver")
    @patch("src.agent.tools._get_secret")
    def test_returns_cached_driver(
        self,
        mock_secret,
        mock_driver_factory,
        mock_driver,
    ):
        mock_secret.side_effect = [
            "neo4j+s://localhost",
            "neo4j",
            "password",
        ]

        mock_driver_factory.return_value = mock_driver

        first = tools.get_neo4j_driver()
        second = tools.get_neo4j_driver()

        assert first is second

        mock_driver_factory.assert_called_once()


    @patch("src.agent.tools._get_secret")
    def test_raises_when_uri_missing(
        self,
        mock_secret,
    ):
        mock_secret.side_effect = [
            None,
            "neo4j",
            "password",
        ]

        with pytest.raises(ValueError):
            tools.get_neo4j_driver()


# ============================================================
# _execute_cypher
# ============================================================


class TestExecuteCypher:

    @patch("src.agent.tools._get_secret")
    @patch("src.agent.tools.get_neo4j_driver")
    def test_executes_query_successfully(
        self,
        mock_get_driver,
        mock_secret,
        mock_driver,
    ):
        mock_secret.return_value = "neo4j"

        mock_get_driver.return_value = mock_driver

        result = tools._execute_cypher(
            "MATCH (n) RETURN n",
            {"id": 1},
        )

        assert result == [
            {
                "parameter_id": "PH",
                "parameter_name": "pH",
            }
        ]

        mock_driver.session.assert_called_once_with(database="neo4j")


    @patch("src.agent.tools._get_secret")
    @patch("src.agent.tools.get_neo4j_driver")
    def test_returns_error_when_query_fails(
        self,
        mock_get_driver,
        mock_secret,
        mock_driver,
    ):
        mock_secret.return_value = "neo4j"

        session = mock_driver.session.return_value.__enter__.return_value

        session.run.side_effect = Exception("Database error")

        mock_get_driver.return_value = mock_driver

        result = tools._execute_cypher("MATCH (n)", {})

        assert result == [
            {
                "error": "Database error",
            }
        ]