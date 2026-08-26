# agents/gates.py
"""
Deterministic pre-flight gates for agent nodes.

A gate runs before an agent's ReAct loop, decided by graph code rather than
by the model. Use one when a class of request is knowably unanswerable from
static properties of the turn -- no LLM call needed to discover that. Keep
gates conservative: a false negative just means the normal loop runs; a
false positive silently denies a request the agent could have answered.
"""

import re

from .agents import MATH          # ajusta al import real donde vive MATH
from .state import AgentResult           # ajusta al import real

_HAS_DIGIT = re.compile(r"\d")


def math_inputs_present(user_message: str) -> bool:
    """..."""
    return bool(_HAS_DIGIT.search(user_message or ""))


def missing_inputs_result(step, user_message: str) -> AgentResult:
    """..."""
    return AgentResult(
        agent=MATH,
        step=step.step,
        output=(
            "STATUS: MISSING_INPUTS\n"
            "The requested calculation needs numeric inputs that were not "
            "provided in this turn. Ask the user for: pool volume (gallons), "
            "current value of the parameter, and target value."
        ),
        sources=[],
    )