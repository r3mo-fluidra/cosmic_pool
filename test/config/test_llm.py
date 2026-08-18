from unittest.mock import patch

from src.config.llm import (
    _get_secret,
    create_llm,
    create_routing_llm,
    create_synthesizer_llm,
)





# ==========================================================
# create_llm()
# ==========================================================

@patch("src.config.llm.ChatGoogleGenerativeAI")
@patch("src.config.llm._get_secret")
def test_create_llm(mock_get_secret, mock_chat):
    mock_get_secret.return_value = "fake-key"

    create_llm()

    mock_chat.assert_called_once_with(
        model="gemini-3.1-flash-lite",
        google_api_key="fake-key",
        temperature=0.2,
        )


# ==========================================================
# create_routing_llm()
# ==========================================================

@patch("src.config.llm.ChatGoogleGenerativeAI")
@patch("src.config.llm._get_secret")
def test_create_routing_llm(mock_get_secret, mock_chat):
    mock_get_secret.return_value = "fake-key"

    create_routing_llm()

    mock_chat.assert_called_once_with(
        model="gemini-3.1-flash-lite",
        google_api_key="fake-key",
        temperature=0.0,
    )


# ==========================================================
# create_synthesizer_llm()
# ==========================================================

@patch("src.config.llm.ChatGoogleGenerativeAI")
@patch("src.config.llm._get_secret")
def test_create_synthesizer_llm(mock_get_secret, mock_chat):
    mock_get_secret.return_value = "fake-key"

    create_synthesizer_llm()

    mock_chat.assert_called_once_with(
        model="gemini-3.1-flash-lite",
        google_api_key="fake-key",
        temperature=0.4,
    )