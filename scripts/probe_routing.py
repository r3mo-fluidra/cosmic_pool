"""Measures how prior-turn context shifts planner routing."""

from collections import Counter

from src.agent.nodes import _get_planner_chain

FILTER_HISTORY = (
    "Previous turn:\n"
    "User: Our sand filter's pressure gauge reads 22 psi against a 12 psi "
    "clean baseline and backwashing only brings it to 19 psi. Media is four "
    "years old.\n"
    "Assistant: Your filter media has failed due to severe calcification or "
    "channeling and must be replaced.\n\n"
    "Current turn:\n"
)

QUESTION = "My water looks cloudy"
RUNS = 5

chain = _get_planner_chain()

for label, prompt in (("sin historial", QUESTION),
                      ("con historial de filtro", FILTER_HISTORY + QUESTION)):
    agents = Counter()
    for _ in range(RUNS):
        plan = chain.invoke({"input": prompt})
        agents["+".join(s.assigned_agent for s in plan.execution_plan)] += 1
    print(f"{label:28s} {dict(agents)}")