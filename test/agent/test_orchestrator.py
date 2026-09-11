"""
Tests del nodo `orchestrator` (src/agent/nodes.py).

El orchestrator no llama a ningún LLM: decide. Por eso se puede testear
entero sin mocks de red — solo hace falta un state.

Cuatro reglas, una por bloque:
  1. Fan-out: los steps sin dependencias pendientes salen TODOS en el mismo
     Command, en Sends separados. Ese paralelismo es el motivo del rediseño.
  2. depends_on: un step espera hasta que su fuente esté en `ok`.
  3. Una dependencia fallida marca al dependiente como
     SKIPPED_DEPENDENCY_FAILED, en cascada, sin ejecutarlo.
  4. Un error de infraestructura (429/503/504) corta el turno: no se
     reintenta contra el mismo proveedor dentro del mismo turno.
"""

import time

import pytest
from langgraph.types import Command, Send

from src.agent import nodes
from src.agent.state import AgentResult, ExecutionStep


def step(n=1, agent="chemistry", depends_on=None) -> ExecutionStep:
    return ExecutionStep(
        step=n,
        task=f"task {n}",
        assigned_agent=agent,
        depends_on=depends_on or [],
    )


def ok(n=1, agent="chemistry", output="done") -> AgentResult:
    return AgentResult(agent=agent, step=n, output=output, status="ok")


def failed(n=1, agent="chemistry", error="boom") -> AgentResult:
    return AgentResult(agent=agent, step=n, output="", error=error, status="failed")


def state_for(plan, results=None, **extra) -> dict:
    return {
        "messages": [nodes.HumanMessage(content="my water is green")],
        "execution_plan": plan,
        "agent_results": results or {},
        "turn_started_at": time.time(),
        **extra,
    }


def sends_of(command: Command) -> list[Send]:
    goto = command.goto
    return [g for g in goto if isinstance(g, Send)] if isinstance(goto, list) else []


def dispatched_steps(command: Command) -> set[int]:
    return {send.arg["step"].step for send in sends_of(command)}


# ================================================================
# 1. Fan-out
# ================================================================

class TestFanOut:
    def test_two_independent_steps_are_dispatched_in_a_single_command(self):
        command = nodes.orchestrator(state_for([step(1), step(2, agent="math")]))

        assert len(sends_of(command)) == 2
        assert dispatched_steps(command) == {1, 2}

    def test_every_send_targets_the_run_step_node(self):
        command = nodes.orchestrator(state_for([step(1), step(2, agent="math")]))

        assert {send.node for send in sends_of(command)} == {"run_step"}

    def test_each_send_carries_the_user_message_and_the_summary(self):
        """El payload del Send ES el state del nodo: sin esto los especialistas
        solo ven su task y son amnésicos entre turnos."""
        command = nodes.orchestrator(
            state_for([step(1)], conversation_summary="pool is 50 m3")
        )

        payload = sends_of(command)[0].arg
        assert payload["user_message"] == "my water is green"
        assert payload["conversation_summary"] == "pool is 50 m3"
        assert payload["deadline_s"] > 0

    def test_already_finished_steps_are_not_dispatched_again(self):
        command = nodes.orchestrator(
            state_for([step(1), step(2, agent="math")], {"step_1": ok(1)})
        )

        assert dispatched_steps(command) == {2}


# ================================================================
# 2. depends_on
# ================================================================

class TestDependencies:
    def test_only_the_independent_step_runs_while_its_dependent_waits(self):
        plan = [step(1), step(2, agent="math", depends_on=[1])]

        command = nodes.orchestrator(state_for(plan))

        assert dispatched_steps(command) == {1}

    def test_the_dependent_runs_once_its_source_succeeded(self):
        plan = [step(1), step(2, agent="math", depends_on=[1])]

        command = nodes.orchestrator(state_for(plan, {"step_1": ok(1)}))

        assert dispatched_steps(command) == {2}

    def test_a_step_waits_for_all_of_its_dependencies(self):
        plan = [step(1), step(2, agent="math"), step(3, agent="safety", depends_on=[1, 2])]

        command = nodes.orchestrator(state_for(plan, {"step_1": ok(1)}))

        assert dispatched_steps(command) == {2}


# ================================================================
# 3. Dependencia fallida -> SKIPPED_DEPENDENCY_FAILED
# ================================================================

class TestFailedDependency:
    def test_a_dependent_of_a_failed_step_is_skipped_not_executed(self):
        plan = [step(1), step(2, agent="math", depends_on=[1])]

        command = nodes.orchestrator(
            state_for(plan, {"step_1": failed(1, error="CANNOT_COMPUTE: no data")})
        )

        skipped = command.update["agent_results"]["step_2"]
        assert skipped.status == "skipped"
        assert skipped.error.startswith("SKIPPED_DEPENDENCY_FAILED: step_1")
        assert sends_of(command) == []

    def test_the_skip_reason_names_the_upstream_error(self):
        plan = [step(1), step(2, agent="math", depends_on=[1])]

        command = nodes.orchestrator(
            state_for(plan, {"step_1": failed(1, error="CANNOT_COMPUTE: no data")})
        )

        assert "CANNOT_COMPUTE: no data" in command.update["agent_results"]["step_2"].error

    def test_the_skip_cascades_down_the_whole_chain(self):
        """1 -> 2 -> 3: el fallo de 1 tiene que envenenar también a 3."""
        plan = [
            step(1),
            step(2, agent="math", depends_on=[1]),
            step(3, agent="safety", depends_on=[2]),
        ]

        command = nodes.orchestrator(
            state_for(plan, {"step_1": failed(1, error="CANNOT_COMPUTE: no data")})
        )

        results = command.update["agent_results"]
        assert results["step_2"].status == "skipped"
        assert results["step_3"].status == "skipped"

    def test_an_independent_step_still_runs_when_another_branch_failed(self):
        plan = [step(1), step(2, agent="math", depends_on=[1]), step(3, agent="safety")]

        command = nodes.orchestrator(
            state_for(plan, {"step_1": failed(1, error="CANNOT_COMPUTE: no data")})
        )

        assert dispatched_steps(command) == {3}
        assert command.update["agent_results"]["step_2"].status == "skipped"


# ================================================================
# 4. Circuit breaker de infraestructura
# ================================================================

class TestInfraCircuitBreaker:
    def test_an_infra_error_ends_the_turn_without_running_the_rest(self):
        """Un 504 no se recupera dentro del turno: insistir solo gasta el budget."""
        plan = [step(1), step(2, agent="math"), step(3, agent="safety")]

        command = nodes.orchestrator(
            state_for(plan, {"step_1": failed(1, error="504 Deadline Exceeded")})
        )

        assert command.goto == ["synthesizer", "suggester"]
        assert sends_of(command) == []

    def test_the_pending_steps_are_marked_as_skipped_upstream_infra_failure(self):
        plan = [step(1), step(2, agent="math"), step(3, agent="safety")]

        command = nodes.orchestrator(
            state_for(plan, {"step_1": failed(1, error="504 Deadline Exceeded")})
        )

        results = command.update["agent_results"]
        for key in ("step_2", "step_3"):
            assert results[key].status == "skipped"
            assert results[key].error.startswith("SKIPPED_UPSTREAM_INFRA_FAILURE")

    def test_the_turn_error_points_at_the_step_that_broke(self):
        plan = [step(1), step(2, agent="math")]

        command = nodes.orchestrator(
            state_for(plan, {"step_1": failed(1, error="429 Too Many Requests")})
        )

        assert command.update["error"] == "UPSTREAM_INFRA_FAILURE at step_1"

    def test_a_business_failure_does_not_trip_the_breaker(self):
        """MISSING_INPUTS es un gap de negocio: el resto del plan sigue."""
        plan = [step(1), step(2, agent="math")]

        command = nodes.orchestrator(
            state_for(plan, {"step_1": failed(1, error="MISSING_INPUTS: need volume")})
        )

        assert dispatched_steps(command) == {2}


# ================================================================
# Salidas al synthesizer
# ================================================================

class TestExitToSynthesizer:
    def test_an_empty_plan_reports_the_error_and_ends_the_turn(self):
        command = nodes.orchestrator(state_for([]))

        assert command.goto == ["synthesizer", "suggester"]
        assert "EMPTY_EXECUTION_PLAN" in command.update["error"]

    def test_a_finished_plan_fans_out_to_synthesizer_and_suggester(self):
        """Los dos corren en el mismo superstep: el suggester sale del camino
        crítico del usuario."""
        command = nodes.orchestrator(state_for([step(1)], {"step_1": ok(1)}))

        assert command.goto == ["synthesizer", "suggester"]

    def test_the_completed_results_reach_the_synthesizer(self):
        """Antes se calculaban y se descartaban: el synthesizer recibía {} y
        caía en la rama del saludo."""
        command = nodes.orchestrator(state_for([step(1)], {"step_1": ok(1)}))

        assert command.update["agent_results"]["step_1"].output == "done"

    def test_an_exhausted_turn_budget_skips_everything_pending(self):
        state = state_for([step(1), step(2, agent="math")])
        state["turn_started_at"] = time.time() - (nodes.TURN_DEADLINE_S - 1)

        command = nodes.orchestrator(state)

        assert command.update["error"] == "TURN_DEADLINE_EXCEEDED"
        assert command.update["agent_results"]["step_1"].status == "skipped"
        assert sends_of(command) == []

    def test_a_dependency_cycle_is_reported_as_a_deadlock(self):
        plan = [
            step(1, depends_on=[2]),
            step(2, agent="math", depends_on=[1]),
        ]

        command = nodes.orchestrator(state_for(plan))

        assert "Deadlock" in command.update["error"]
        assert sends_of(command) == []
