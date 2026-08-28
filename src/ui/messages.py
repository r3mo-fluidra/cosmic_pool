"""
messages.py
===========
The shape of one entry in `st.session_state.messages`.

Split out for the same reason `copy.py` was: a plain TypedDict and a factory,
no Streamlit import, no state, no I/O — so both apps and the tests can agree on
the schema without importing a page.

Why a TypedDict and not a dataclass
-----------------------------------
The transcript is a list of dicts written straight into session state and read
back by `st.markdown` and the action row. A dataclass would mean converting on
the way in and out of a structure Streamlit already stores happily. A TypedDict
changes nothing at runtime and turns a misspelled key into a type error, which
is the failure this is here to catch: every optional field is read through
`.get()`, so `msg.get("can_rte")` renders nothing at all — and in a UI, absence
looks like a design decision rather than a bug.

`total=False` on the optional half is deliberate: a user turn carries only
`role` and `content`, and the opening greeting carries neither `trace_id` nor
`turn_index` because it never went to the graph.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from src.ui.copy import GREETING

Role = Literal["user", "assistant"]


class ChatMessage(TypedDict, total=False):
    """
    One turn in the transcript.

    `role` and `content` are always present; everything else depends on what
    kind of turn it is. The comments record *who* writes each field, because
    that is what decides whether it can be trusted on a later rerun.
    """

    # -- always present ---------------------------------------------------
    role: Role
    content: str

    # -- the opening greeting only ----------------------------------------
    #: Marks the locally rendered greeting, which is what gates the openers
    #: (they show only while the greeting is the whole conversation).
    opening: bool

    # -- assistant turns that reached the graph ---------------------------
    #: Deterministic Langfuse trace id for the turn. Also the key the feedback
    #: verdict and the sources panel are stored under, so it must be stable.
    trace_id: str
    #: Monotonic per-session counter — never `len(messages)`, which repeats
    #: itself when a turn dies before appending its reply.
    turn_index: int
    #: Whether the answer settled the question, and so whether the action row
    #: appears. Decided once, when the turn lands: a turn cannot become
    #: rateable later, and re-deriving it per rerun would be guesswork.
    can_rate: bool
    #: The language the *reader* wrote in, settled once here so the chrome
    #: under the answer does not get re-detected from the answer text.
    language: str
    #: Documents behind the answer. Empty today — nothing populates
    #: `AgentResult.sources` yet — and the panel says so plainly. This is the
    #: slot the backend fills when it starts returning them.
    sources: list[str]
    #: Pipeline-debug lines, kept only while `SHOW_PIPELINE_DEBUG` is on;
    #: `None` otherwise, so an off flag is not dead weight in session state.
    debug: list[tuple[str, str]] | None


def initial_messages() -> list[ChatMessage]:
    """The conversation as it looks before the user has said anything."""
    return [{"role": "assistant", "content": GREETING, "opening": True}]


def is_opening_only(messages: list[ChatMessage]) -> bool:
    """
    Whether the greeting is still the whole conversation, which is what gates
    the opening suggestion chips.

    Reads the transcript rather than a counter so it stays correct after a
    reset, which rebuilds the list rather than decrementing anything.
    """
    return len(messages) == 1 and bool(messages[0].get("opening"))


def sources_of(message: ChatMessage) -> list[str]:
    """
    The message's sources as a list of strings, whatever the field holds.

    `None`, a missing key and an empty list all mean the same thing to the
    panel — that the answer was not linked to a document — so they are
    flattened here instead of at each of the two call sites.
    """
    raw: Any = message.get("sources") or []
    return [str(item) for item in raw]