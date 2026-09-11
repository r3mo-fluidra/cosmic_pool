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

# MATH_SLUG y no MATH: `MATH` vale "Pool Math Agent", un nombre de display que
# AgentName no acepta. AgentResult(agent=MATH) era un ValidationError esperando
# a que el gate llegara a dispararse alguna vez.
#
# Importa de agent_names y no de .agents: este módulo solo necesita el
# identificador, no arrastrar el registro de agentes entero.
from .agent_names import MATH_SLUG
from .state import AgentResult

_HAS_DIGIT = re.compile(r"\d")


def math_inputs_present(user_message: str) -> bool:
    """
    ¿Hay alguna cantidad numérica con la que calcular en este turno?

    Conservador por diseño: un dígito en cualquier parte basta para dejar
    pasar el turno al ReAct loop. Un falso negativo le niega al usuario un
    cálculo que el agente sí podía hacer; un falso positivo solo gasta una
    iteración que habría gastado igual.
    """
    return bool(_HAS_DIGIT.search(user_message or ""))


def missing_inputs_result(step, user_message: str) -> AgentResult:
    """
    El AgentResult que devuelve el gate de MATH cuando corta.

    El output es un marcador machine-readable, no prosa: el synthesizer sigue
    siendo dueño de la redacción, el idioma y el arquetipo. Devolver prosa ya
    escrita le haría parafrasear trabajo hecho y gastar una generación.
    """
    return AgentResult(
        agent=MATH_SLUG,
        step=step.step,
        output=(
            "STATUS: MISSING_INPUTS\n"
            "The requested calculation needs numeric inputs that were not "
            "provided in this turn. Ask the user for: pool volume (gallons), "
            "current value of the parameter, and target value."
        ),
        sources=[],
    )