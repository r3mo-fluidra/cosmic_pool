"""
Shared fixtures for src.agent.tools tests.

The goal of this module is to provide reusable fake objects for all
tests without patching any production code.

Every test decides which dependency it wants to patch while reusing
the same fake objects.
"""

from unittest.mock import MagicMock

import pytest


# ============================================================
# VECTOR STORE FIXTURES
# ============================================================


@pytest.fixture
def mock_document():
    """
    Single LangChain Document.
    """
    doc = MagicMock()

    doc.page_content = "Example page content"

    doc.metadata = {
        "source": "manual.pdf",
        "category": "Maintenance",
        "tags": ["pump", "filter"],
    }

    return doc


@pytest.fixture
def mock_documents(mock_document):
    """
    Default list returned by similarity_search().
    """
    return [mock_document]


@pytest.fixture
def mock_vector_store(mock_documents):
    """
    Fake Vector Store.
    """
    store = MagicMock()

    store.similarity_search.return_value = mock_documents

    return store


# ============================================================
# NEO4J FIXTURES
# ============================================================


@pytest.fixture
def mock_neo4j_results():
    """
    Generic Cypher response.
    """
    return [
        {
            "parameter_id": "PH",
            "parameter_name": "pH",
            "relationship": "LOW_CAUSES",
            "detail": "Low pH causes corrosion.",
        }
    ]


@pytest.fixture
def mock_record():
    """
    Fake Neo4j Record.
    """
    record = MagicMock()

    record.data.return_value = {
        "parameter_id": "PH",
        "parameter_name": "pH",
    }

    return record


@pytest.fixture
def mock_session(mock_record):
    """
    Fake Neo4j Session.
    """
    session = MagicMock()

    session.run.return_value = [mock_record]

    return session


@pytest.fixture
def mock_driver(mock_session):
    """
    Fake Neo4j Driver.

    Supports

        with driver.session() as session:
            ...
    """

    driver = MagicMock()

    driver.session.return_value.__enter__.return_value = mock_session

    return driver


# ============================================================
# ENVIRONMENT FIXTURES
# ============================================================


@pytest.fixture
def fake_secrets():
    """
    Default secrets used across tests.
    """
    return {
        "NEO4J_URI": "neo4j+s://localhost",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "password",
        "NEO4J_DATABASE": "neo4j",
    }


# ============================================================
# COMMON DATA
# ============================================================


@pytest.fixture
def sample_pool_values():
    """
    Common chemistry values used by multiple tests.
    """
    return {
        "ph": 7.5,
        "ta": 90,
        "ch": 300,
        "temp_c": 25,
        "cya": 30,
        "tds": 1000,
    }