from unittest.mock import MagicMock, patch, ANY

from src.agent.chains import create_planner_chain
from src.agent.state import PlannerOutput


@patch("src.agent.chains.ChatPromptTemplate")
def test_create_planner_chain(mock_prompt_template):
    """
    Should build the planner chain with structured output.
    """

    # Arrange
    mock_llm = MagicMock()

    mock_prompt = MagicMock()
    mock_prompt_template.from_messages.return_value = mock_prompt

    mock_structured_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    expected_chain = MagicMock()
    mock_prompt.__or__.return_value = expected_chain

    # Act
    chain = create_planner_chain(mock_llm)

    # Assert

    mock_prompt_template.from_messages.assert_called_once_with(
        [
            ("system", ANY),
            ("user", "{input}")
        ]
    )

    mock_llm.with_structured_output.assert_called_once_with(
        schema=PlannerOutput,
        method="function_calling",
    )

    mock_prompt.__or__.assert_called_once_with(mock_structured_llm)

    assert chain == expected_chain