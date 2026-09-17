"""The tool descriptions the model reads must match the enforced budgets."""

import re

from src.agent.tools import vector_search, search_seed_nodes, expand_subgraph
from src.tool_budgets import RETRIEVAL_TOOL_BUDGETS

_WORD = {1: "once", 2: "two", 3: "three"}


def test_descriptions_match_budgets():
    for tool in (vector_search, search_seed_nodes, expand_subgraph):
        budget = RETRIEVAL_TOOL_BUDGETS[tool.name]
        text = tool.description.lower()
        assert re.search(rf"at most {_WORD[budget]}\b", text), (
            f"{tool.name}: budget is {budget} but its description does not "
            f"say 'at most {_WORD[budget]}'"
        )