"""
Tests de la capa de infraestructura de src.agent.tools.

Cubre lo que sigue existiendo tras la v2:
  - resolución de secrets (env primero, Streamlit después)
  - inicialización perezosa y cacheada del driver de Neo4j
  - inicialización perezosa y cacheada del vector store

Lo que se borró: los tests de `_execute_cypher`, una función que ya no
existe. El grafo se consulta hoy desde las tools de retrieval
(search_seed_nodes / expand_subgraph), no desde un helper genérico.
"""

from unittest.mock import MagicMock, patch

import pytest

import src.agent.tools as tools


# ============================================================
# _get_secret
# ============================================================


class TestGetSecret:
    """El nombre del atributo es `_st_secrets`: `from streamlit import
    secrets as _st_secrets`, envuelto en try/except porque streamlit puede
    no estar instalado."""

    @patch("src.agent.tools.os.getenv")
    def test_env_var_wins_over_streamlit_secrets(self, mock_getenv):
        mock_secrets = MagicMock()
        mock_getenv.return_value = "value_from_env"

        with patch.object(tools, "_st_secrets", mock_secrets):
            assert tools._get_secret("TEST_KEY") == "value_from_env"

        mock_secrets.get.assert_not_called()

    @patch("src.agent.tools.os.getenv")
    def test_falls_back_to_streamlit_secrets_when_env_missing(self, mock_getenv):
        mock_secrets = MagicMock()
        mock_getenv.return_value = None
        mock_secrets.get.return_value = "value_from_secret"

        with patch.object(tools, "_st_secrets", mock_secrets):
            assert tools._get_secret("TEST_KEY") == "value_from_secret"

    @patch("src.agent.tools.os.getenv")
    def test_returns_default_when_missing_everywhere(self, mock_getenv):
        mock_secrets = MagicMock()
        mock_getenv.return_value = None
        mock_secrets.get.return_value = None

        with patch.object(tools, "_st_secrets", mock_secrets):
            assert tools._get_secret("TEST_KEY", "default") == "default"

    @patch("src.agent.tools.os.getenv")
    def test_returns_default_when_streamlit_is_not_installed(self, mock_getenv):
        mock_getenv.return_value = None

        with patch.object(tools, "_st_secrets", None):
            assert tools._get_secret("TEST_KEY", "default") == "default"


# ============================================================
# get_vector_store
# ============================================================


class TestVectorStore:

    def setup_method(self):
        tools._vector_store = None
        tools._store_error = None
        tools._store_error_at = 0.0

    teardown_method = setup_method

    @patch("src.agent.tools.cargar_vector_store")
    def test_initializes_vector_store_once(self, mock_loader, mock_vector_store):
        mock_loader.return_value = mock_vector_store

        assert tools.get_vector_store() is mock_vector_store
        mock_loader.assert_called_once()

    @patch("src.agent.tools.cargar_vector_store")
    def test_returns_cached_vector_store(self, mock_loader, mock_vector_store):
        mock_loader.return_value = mock_vector_store

        first = tools.get_vector_store()
        second = tools.get_vector_store()

        assert first is second
        mock_loader.assert_called_once()

    @patch("src.agent.tools.cargar_vector_store")
    def test_failure_is_not_retried_during_the_cooldown(self, mock_loader):
        mock_loader.side_effect = RuntimeError("qdrant down")

        with pytest.raises(RuntimeError):
            tools.get_vector_store()

        # Segundo intento: corta con el error cacheado, sin volver a llamar.
        with pytest.raises(tools.VectorStoreConfigError):
            tools.get_vector_store()

        mock_loader.assert_called_once()


# ============================================================
# get_neo4j_driver
# ============================================================


class TestNeo4jDriver:

    def setup_method(self):
        tools._neo4j_driver = None

    teardown_method = setup_method

    @patch("src.agent.tools.GraphDatabase.driver")
    @patch("src.agent.tools._get_secret")
    def test_creates_driver_only_once(
        self, mock_secret, mock_driver_factory, mock_driver
    ):
        mock_secret.side_effect = ["neo4j+s://localhost", "neo4j", "password"]
        mock_driver_factory.return_value = mock_driver

        assert tools.get_neo4j_driver() is mock_driver
        mock_driver_factory.assert_called_once_with(
            "neo4j+s://localhost",
            auth=("neo4j", "password"),
        )

    @patch("src.agent.tools.GraphDatabase.driver")
    @patch("src.agent.tools._get_secret")
    def test_returns_cached_driver(
        self, mock_secret, mock_driver_factory, mock_driver
    ):
        mock_secret.side_effect = ["neo4j+s://localhost", "neo4j", "password"]
        mock_driver_factory.return_value = mock_driver

        first = tools.get_neo4j_driver()
        second = tools.get_neo4j_driver()

        assert first is second
        mock_driver_factory.assert_called_once()

    @patch("src.agent.tools._get_secret")
    def test_raises_when_uri_missing(self, mock_secret):
        mock_secret.side_effect = [None, "neo4j", "password"]

        with pytest.raises(ValueError):
            tools.get_neo4j_driver()

    @patch("src.agent.tools._get_secret")
    def test_raises_when_password_missing(self, mock_secret):
        mock_secret.side_effect = ["neo4j+s://localhost", "neo4j", None]

        with pytest.raises(ValueError):
            tools.get_neo4j_driver()
