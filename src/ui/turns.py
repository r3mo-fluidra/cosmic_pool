"""
turns.py
========
The two pieces of per-turn presentation logic that are not rendering: which
status rows the wait shows, and whether a finished turn is rateable.

Both used to live inside `app.py` — one as a closure over `run_turn`'s locals,
the other as a module function halfway down an executing script — which is why
neither had a test. Nothing here imports Streamlit, the graph, or anything under
`src/agent`: `TurnProgress` is fed plain event payloads by its caller and the
planner's steps are read with `getattr`, so the backend's types stay entirely
its own business.

What this module is *not*
-------------------------
It is not the graph call, the Langfuse span, or the trace plumbing. Those stay
in `app.py` where they are, untouched — this only decides what the reader sees
while they run.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

from src.ui.copy import RETRIEVE_ROTATION, agent_line, status_line

#: A status row: the label, and whether it is the stage currently in flight.
StatusRow = tuple[str, bool]

#: Answers shorter than this that end in a question mark are read as the
#: assistant still working the problem rather than closing it.
CLARIFYING_LENGTH = 320

#: Agents whose output does not count as grounding an answer. `general` and
#: `ooo` answer from the model, not from the manual or the graph.
UNGROUNDED_AGENTS = frozenset({"ooo", "general"})


def is_definitive_answer(
    plan: Iterable[Any],
    runs: Iterable[tuple[str, str | None]],
    text: str,
) -> bool:
    """
    Whether this turn settled the question, which is what decides if the
    feedback row appears. Troubleshooting turns — the assistant asking for a
    reading, a symptom, a model number — get none: there is nothing to rate yet.

    The graph exposes no such flag, so it is derived from what the stream
    already carries:

      * a refusal (`oos`) is not an answer about the pool;
      * an answer nobody grounded — every step failed, or the only agents that
        ran were `general`/`ooo` — is conversation, not a documented answer;
      * a short reply ending in a question mark is the assistant asking back.

    `plan` is the planner's ExecutionStep list, `runs` the `(agent, error)`
    pairs the orchestrator emitted. Both are read structurally rather than by
    type, so this stays importable without the backend.

    The durable version of this is a flag set by the synthesizer, which would
    remove the length heuristic — worth raising with the backend before this
    leaves the test phase, because this function decides whether feedback can be
    collected at all.
    """
    body = (text or "").strip()
    if not body:
        return False
    if any(getattr(step, "oos", False) for step in plan):
        return False
    grounded = any(
        agent not in UNGROUNDED_AGENTS and not error for agent, error in runs
    )
    if not grounded:
        return False
    if len(body) < CLARIFYING_LENGTH and body.endswith("?"):
        return False
    return True


class TurnProgress:
    """
    The growing status stack, advanced by the graph events as they arrive.

    Rows accumulate downward: a finished row stays on screen dimmed, and the new
    row opens beneath it — so the reader can see both what is happening now and
    what it has already been through. Only rows the pipeline has *actually*
    reached are ever emitted; nothing is pre-rendered as pending, because until
    the planner answers we do not know what the stages will be, and inventing
    them is exactly the status theatre FRONTEND.md §8.2 rules out.

    `on_change` is called with the whole row list every time a row actually
    opens — and only then. A no-op advance paints nothing, which is what keeps a
    repeated stage from costing a repaint.

    Usage against the live graph (`app.py`):

        progress = TurnProgress(detect_language(prompt), on_change=paint)
        progress.begin()
        for event in graph.stream(...):
            if "planner" in event:
                progress.plan(steps, event["planner"].get("detected_language"))
            elif "orchestrator" in event:
                progress.step_finished()
            elif "synthesizer" in event:
                progress.generating()

    Usage on scripted timings (`preview_ui.py`) goes through `advance()`
    directly, since there are no events to read.
    """

    def __init__(
        self,
        language: str,
        *,
        on_change: Callable[[list[StatusRow]], None] | None = None,
    ) -> None:
        self._language = language
        self._on_change = on_change
        self._rows: list[StatusRow] = []
        #: One entry per row the plan will need, holding the agent that row
        #: names. Consecutive steps on the same agent collapse into one entry:
        #: two identical lines stacked on top of each other read as a bug
        #: rather than as two units of work.
        self._agent_rows: list[str] = []
        #: Plan step index -> index into `_agent_rows`.
        self._step_row: list[int] = []
        #: Orchestrator invocations seen. The orchestrator runs exactly one
        #: planned step per invocation, so this is also the step cursor.
        self._steps_done = 0

    # -- reading -----------------------------------------------------------

    @property
    def language(self) -> str:
        """
        The language the status lines are written in: the caller's guess from
        the prompt until the planner reports one, and the planner's from then on.
        """
        return self._language

    @property
    def rows(self) -> list[StatusRow]:
        """The stack as it stands. A copy — callers render it, not mutate it."""
        return list(self._rows)

    # -- advancing ---------------------------------------------------------

    def advance(self, label: str) -> None:
        """
        Close every open row and start a new live one beneath them.

        Idempotent: the same label arriving twice — a synthesizer event after
        the last orchestrator step already announced generation — must not stack
        a duplicate row, and must not repaint.
        """
        if self._rows and self._rows[-1] == (label, True):
            return
        self._rows = [(text, False) for text, _ in self._rows]
        self._rows.append((label, True))
        self._paint()

    def begin(self) -> None:
        """
        Put something on screen before the graph is even called. The wait used
        to start with nothing at all, which was the whole complaint.
        """
        self.advance(status_line(self._language, "read"))

    def plan(
        self,
        steps: Sequence[Any],
        detected_language: str | None = None,
    ) -> None:
        """
        Adopt the planner's answer: its language, and one row per unit of work.

        The opening row is repainted in the planner's language, which may differ
        from the guess made before the graph was called — the first line has to
        render before there is anything to read, so a guess is unavoidable, but
        it does not have to persist.
        """
        self._language = detected_language or self._language

        for step in steps:
            name = getattr(step, "assigned_agent", "") or ""
            if not self._agent_rows or self._agent_rows[-1] != name:
                self._agent_rows.append(name)
            self._step_row.append(len(self._agent_rows) - 1)

        self._rows = [(status_line(self._language, "read"), False)]
        self.advance(
            agent_line(self._language, self._agent_rows[0])
            if self._agent_rows
            else status_line(self._language, RETRIEVE_ROTATION[0])
        )

    def step_finished(self) -> None:
        """
        One orchestrator invocation completed, i.e. one planned step of real
        retrieval — which earns the next row.

        The synthesizer's own update only arrives once generation has *finished*,
        so waiting for it left the longest silence of the turn labelled as
        retrieval — the exact dishonesty §8.2 forbids. The planner already said
        how many steps there are and the orchestrator runs exactly one per
        invocation, so the last step completing means generation is what happens
        next. Say so.
        """
        self._steps_done += 1
        done_row = self._row_of_step(self._steps_done - 1)
        next_row = self._row_of_step(self._steps_done)

        if next_row is None:
            self.advance(status_line(self._language, "write"))
        elif next_row != done_row:
            self.advance(agent_line(self._language, self._agent_rows[next_row]))
        # else: the same agent has another step — its row stays live.

    def generating(self) -> None:
        """
        The synthesizer is composing the reply. Covers the plan-less case; a
        no-op when `step_finished` already opened the generation row.
        """
        self.advance(status_line(self._language, "write"))

    # -- internals ---------------------------------------------------------

    def _row_of_step(self, index: int) -> int | None:
        """The row a plan step belongs to, or None if the plan has no such step."""
        if 0 <= index < len(self._step_row):
            return self._step_row[index]
        return None

    def _paint(self) -> None:
        if self._on_change is not None:
            self._on_change(self.rows)