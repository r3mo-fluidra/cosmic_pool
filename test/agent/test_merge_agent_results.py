"""
Tests del reducer `merge_agent_results` (src/agent/state.py).

Es el canal por el que vuelven todos los `run_step` de un fan-out. Dos
propiedades y nada más:

  - `right is None` es el CENTINELA de reset. Sin él no hay forma de limpiar
    el canal: un update con {} se mergea con lo del turno anterior en vez de
    borrarlo, y el synthesizer del turno 2 redacta con material del turno 1.
  - dos escrituras concurrentes se combinan, no se pisan. Es literalmente lo
    que hace que el fan-out paralelo sea posible.
"""

import pytest

from src.agent.state import AgentResult, merge_agent_results


def result(step: int, agent: str = "chemistry") -> AgentResult:
    return AgentResult(agent=agent, step=step, output=f"output {step}")


class TestResetSentinel:
    def test_none_resets_the_channel_to_empty(self):
        previous = {"step_1": result(1), "step_2": result(2)}

        assert merge_agent_results(previous, None) == {}

    def test_none_on_an_empty_channel_is_still_empty(self):
        assert merge_agent_results(None, None) == {}

    def test_an_empty_dict_is_not_a_reset(self):
        """{} mergea (no borra). Por eso el planner escribe None y no {}."""
        previous = {"step_1": result(1)}

        assert merge_agent_results(previous, {}) == previous


class TestMerge:
    def test_two_concurrent_writes_do_not_overwrite_each_other(self):
        """El fan-out: dos run_step del mismo superstep, cada uno con su key."""
        from_step_1 = {"step_1": result(1)}
        from_step_2 = {"step_2": result(2)}

        merged = merge_agent_results(from_step_1, from_step_2)

        assert set(merged) == {"step_1", "step_2"}
        assert merged["step_1"].output == "output 1"
        assert merged["step_2"].output == "output 2"

    def test_a_three_way_fan_out_keeps_every_step(self):
        merged = {}
        for n in (1, 2, 3):
            merged = merge_agent_results(merged, {f"step_{n}": result(n)})

        assert set(merged) == {"step_1", "step_2", "step_3"}

    def test_the_incoming_write_wins_on_the_same_key(self):
        """Un reintento del mismo step debe reemplazar, no duplicar."""
        left = {"step_1": AgentResult(agent="math", step=1, output="stale")}
        right = {"step_1": AgentResult(agent="math", step=1, output="fresh")}

        assert merge_agent_results(left, right)["step_1"].output == "fresh"

    def test_a_missing_left_side_is_treated_as_empty(self):
        incoming = {"step_1": result(1)}

        assert merge_agent_results(None, incoming) == incoming

    def test_does_not_mutate_the_previous_state(self):
        previous = {"step_1": result(1)}

        merge_agent_results(previous, {"step_2": result(2)})

        assert set(previous) == {"step_1"}
