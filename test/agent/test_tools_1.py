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


# =====================================================================
# Visibilidad de etiquetas en el scoring de seeds
# =====================================================================
# Un label que existe en el grafo pero no en INTENT_LABELS es invisible para
# el scoring: sus nodos se entregan marcados "[off-intent: contexto, no
# respuesta]" por bien que respondan, y con intent="any" además ponen el turno
# en STATUS: WEAK, cuyo aviso le dice al agente que pare.
#
# Cómo se detectó: una evaluación experta concluyó que al KB le faltaban cinco
# reglas de química. Las cinco estaban en el grafo, en nodos :Concept y
# :Formula — dos labels ausentes de INTENT_LABELS. No faltaba conocimiento:
# no se entregaba.

class TestVisibilidadDeEtiquetas:
    def _visibles(self):
        from src.agent.tools import INTENT_LABELS, TOP_PRIORITY_LABELS
        vis = set(TOP_PRIORITY_LABELS)
        for v in INTENT_LABELS.values():
            vis |= set(v)
        return vis

    def test_los_portadores_de_reglas_son_visibles(self):
        """
        Donde vive lo que distingue una respuesta experta de una recitada: la
        regla que fija un objetivo, el estado que describe un desequilibrio,
        la fórmula que lo cuantifica.
        """
        for label in ["DecisionRule", "Condition", "Formula", "Concept", "Requirement"]:
            assert label in self._visibles(), f"{label} invisible para el scoring"

    def test_intent_any_incluye_los_portadores_de_reglas(self):
        # "any" no significa "solo lo prioritario": significa que no hay una
        # intención concreta contra la que medir pertinencia.
        from src.agent.tools import INTENT_LABELS
        for label in ["DecisionRule", "Condition", "Formula"]:
            assert label in INTENT_LABELS["any"]

    def test_intent_any_no_degrada_a_weak_por_falta_de_etiqueta(self):
        """
        Regresión del segundo bug: `adj` eximía a intent="any" del castigo de
        score, pero el chequeo de estado seguía exigiendo una etiqueta
        prioritaria. Un seed perfecto salía WEAK.
        """
        import inspect
        from src.agent import tools
        src = inspect.getsource(tools)
        assert 'sin_etiqueta_util = intent != "any"' in src
