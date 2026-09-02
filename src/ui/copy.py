"""
copy.py
=======
User-facing chrome copy and the language pick that selects it.

Split out of `theme.py` on purpose: that module promises presentation only and
imports Streamlit, while everything here is a plain dict and a pure function.
No Streamlit import, no state, no I/O — so it is unit-testable without a
browser or a session.

Source of the strings
---------------------
`Design/demo-deep-water.html`'s `T` object is the approved artifact and the
authority for exact copy strings (`Documentation/FRONTEND.md` §0, rank 2).
Two deliberate deviations from it are marked inline below; both are noted in
FRONTEND.md §12.

EN/ES parity is *structural*, not best-effort (FRONTEND.md §8.3): both branches
must carry the same keys. `test/ui/test_copy.py` asserts it, so a key added to
one language and forgotten in the other fails the suite rather than falling
back to English at runtime.

Page chrome
-----------
The title, subtitle, greeting and openers live here too, and are imported by
both `app.py` and `preview_ui.py`. They used to be declared once in each,
byte-identical — which quietly undermined the harness, whose whole value is
rendering *the same* surface through the same `theme.py`. One declaration means
the two cannot drift.
"""

from __future__ import annotations


# =====================================================================
# PAGE CHROME
# =====================================================================
# The framing around the phone, and the opening turn inside it. Single
# declaration, imported by `app.py` and `preview_ui.py` alike.

PAGE_TITLE = "IA Pool Assistant V1"
PAGE_SUBTITLE = (
    "Conversational assistant for pool water chemistry, equipment and "
    "maintenance, grounded in the product documentation and a curated pool "
    "knowledge graph. Ask in English or Spanish — it answers in the language "
    "you write in."
)
PHONE_BRAND = "Pool Assistant"
FOOTER_TEAM = "AI/ML Team — Bogotá, Colombia"
FOOTER_PHASE = "UX/UI first test phase"

#: The assistant speaks first. Copy is the prototypes' greeting verbatim; it is
#: rendered locally and never sent to the graph, so it costs no tokens and adds
#: no turn to the checkpointer. English opens the conversation because nothing
#: has been detected yet — the assistant then answers in whatever language the
#: user writes in, and never announces the switch.
GREETING = (
    "Good evening. I'm your pool assistant — tell me what's going on and "
    "we'll figure it out together."
)

#: Openers, from the prototypes' first screen. Clicking one sends it as if it
#: had been typed. A tuple, not a list: nothing may mutate the deck at runtime.
SUGGESTIONS: tuple[str, ...] = (
    "My water looks cloudy",
    "I saw a warning in the app",
    "I have a maintenance question",
)

#: The same deck in Spanish, verbatim from the demo's `es.chips`.
#:
#: Kept beside `SUGGESTIONS` rather than folded into it because a test pins that
#: tuple to the demo's `en.chips` array, and because the greeting that opens the
#: conversation is English — there is no reader language to follow yet.
#:
#: It exists at all because these chips now also appear *after* an answer, where
#: the reader's language is known. §8.3 makes EN/ES parity structural: chrome
#: follows the language the reader wrote in, and a Spanish conversation offering
#: English openers would be the one place that breaks the rule.
SUGGESTIONS_ES: tuple[str, ...] = (
    "El agua se ve turbia",
    "Me apareció un aviso en la app",
    "Tengo una duda de mantenimiento",
)

SUGGESTION_DECKS: dict[str, tuple[str, ...]] = {
    "en": SUGGESTIONS,
    "es": SUGGESTIONS_ES,
}


# =====================================================================
# STATUS LINES
# =====================================================================
# Every line maps to a real stage of the graph — no decorative states
# (FRONTEND.md §8.2). The mapping the UI uses:
#
#   read      -> the turn has been handed to the graph, planner routing
#   retrieve  -> orchestrator is running a sub-agent (RAG / graph lookup)
#   write     -> synthesizer is composing the single user-facing reply
#
# `retrieve_alt*` are the rotation FRONTEND.md §8.2 specifies for
# retrieval-only turns, used when the orchestrator loops more than once so a
# multi-step plan does not sit on one frozen line. Never used for safety.

STATUS_COPY: dict[str, dict[str, str]] = {
    "en": {
        # DEVIATION 1 from the demo, which says "Read your message". That reads
        # as a completed action, and this line is shown *live* with the drops
        # animating. FRONTEND.md §8.2's vocabulary table gives the progressive
        # form; it is used here. The ES branch already uses it ("Leyendo…"), so
        # this also makes the two languages agree on tense.
        "read": "Reading your message…",
        # DEVIATION 2: the demo says "Skimming the manual — §4.2…". The section
        # number is demo fiction — nothing populates `AgentResult.sources` yet
        # (FRONTEND.md gap D6), so naming a section would be a claim the app
        # cannot back. §8.2 forbids status text that obscures what the system is
        # doing, so the number is dropped until sources are real.
        "retrieve": "Skimming the manual…",
        "retrieve_alt1": "Testing the waters…",
        "retrieve_alt2": "Filtering the details…",
        "write": "Writing your answer…",
    },
    "es": {
        "read": "Leyendo tu mensaje…",
        "retrieve": "Repasando el manual…",
        # New ES copy — FRONTEND.md §8.2 gives the alternates in EN only.
        # Needs a copy review before this leaves the test phase.
        "retrieve_alt1": "Probando las aguas…",
        "retrieve_alt2": "Filtrando los detalles…",
        "write": "Escribiendo tu respuesta…",
    },
}

# The rotation the orchestrator walks on repeat invocations. First loop gets
# the plain line, later loops the alternates, then it holds on the last one
# rather than cycling — a spinner that keeps changing text reads as churn.
#: One line per sub-agent the planner actually assigns, so the wait names the
#: work being done rather than a generic "thinking". Keys are the values of
#: `AgentName` in src/agent/state.py — read from there, never written to.
#:
#: Phrased in the reader's terms, not the graph's: nobody outside the team
#: knows what a "hydraulics agent" is, and §8.2 forbids status text that
#: obscures what the system is doing. `_fallback` covers an agent added to the
#: backend before it is added here — a new name must degrade to a truthful
#: line, not to a blank row or a KeyError.
AGENT_COPY: dict[str, dict[str, str]] = {
    "en": {
        "general": "Looking it up…",
        "chemistry": "Checking the water chemistry…",
        "equipment": "Checking your equipment…",
        "hydraulics": "Checking flow and circulation…",
        "operations": "Checking the routine…",
        "compliance": "Checking the standards…",
        "contamination": "Checking water quality…",
        "facility_design": "Checking the installation…",
        "safety": "Checking safety…",
        "recovery": "Working out the next steps…",
        "records": "Checking the records…",
        "math": "Running the numbers…",
        "ooo": "Seeing what I can help with…",
        "_fallback": "Looking into it…",
    },
    "es": {
        "general": "Buscando la respuesta…",
        "chemistry": "Revisando la química del agua…",
        "equipment": "Revisando tu equipo…",
        "hydraulics": "Revisando el caudal y la circulación…",
        "operations": "Revisando la rutina…",
        "compliance": "Revisando la normativa…",
        "contamination": "Revisando la calidad del agua…",
        "facility_design": "Revisando la instalación…",
        "safety": "Revisando la seguridad…",
        "recovery": "Definiendo los próximos pasos…",
        "records": "Revisando los registros…",
        "math": "Haciendo los cálculos…",
        "ooo": "Viendo en qué puedo ayudarte…",
        "_fallback": "Revisando…",
    },
}


#: Labels for the action row under an assistant answer: the two thumbs, the
#: overflow menu, and the sources panel it opens. Same structural rule as
#: STATUS_COPY — both branches carry identical keys, asserted by a test.
#:
#: These are chrome, not answer text, so they follow the language the *turn*
#: was written in, which is the language the reader just used.
ACTION_COPY: dict[str, dict[str, str]] = {
    "en": {
        "helpful": "This helped",
        "not_helpful": "This didn't help",
        "more": "More",
        "view_sources": "View sources",
        "sources_title": "Sources",
        # Deliberately not "no sources found": nothing populates
        # AgentResult.sources yet, so the honest statement is that the answer
        # did not come with any, not that a search came back empty.
        "sources_empty": "This answer wasn't linked to a document.",
        "hide": "Hide",
        "sent": "Feedback sent",
        "change": "Change",
    },
    "es": {
        "helpful": "Me ayudó",
        "not_helpful": "No me ayudó",
        "more": "Más",
        "view_sources": "Ver fuentes",
        "sources_title": "Fuentes",
        "sources_empty": "Esta respuesta no se vinculó a ningún documento.",
        "hide": "Ocultar",
        "sent": "Feedback enviado",
        "change": "Cambiar",
    },
}


#: The follow-up a thumbs-down opens: the prompt, the two field labels, the
#: placeholder, the two controls, and the two acknowledgements.
#:
#: This form used to hold its strings inline in `app.py`, Spanish-only, inside
#: otherwise-English chrome and with English category labels beside them. Same
#: structural rule as the tables above now applies: both branches, identical
#: keys, asserted by a test.
#:
#: The ES branch is the copy that was already shipping, word for word — this
#: adds the EN side that was missing rather than rewriting what testers saw.
FORM_COPY: dict[str, dict[str, str]] = {
    "en": {
        "detail_prompt": "Noted. Can you tell us what went wrong? (optional)",
        "reason_label": "Main reason",
        "detail_label": "Details",
        "detail_placeholder": (
            "E.g. it recommended 3x the chlorine dose for my pool volume."
        ),
        "send": "Send",
        "skip": "Skip",
        "thanks_verdict": "Thanks for your feedback!",
        "thanks_detail": "Thanks for the details!",
    },
    "es": {
        "detail_prompt": "Registrado. ¿Nos contás qué falló? (opcional)",
        "reason_label": "Motivo principal",
        "detail_label": "Detalle",
        "detail_placeholder": (
            "Ej: recomendó 3x la dosis de cloro para mi volumen de pileta."
        ),
        "send": "Enviar",
        "skip": "Omitir",
        "thanks_verdict": "¡Gracias por tu feedback!",
        "thanks_detail": "¡Gracias por el detalle!",
    },
}


#: The negative-feedback taxonomy, in the order the picker lists it.
#:
#: These are the values *stored* on the Langfuse score, so they are frozen —
#: renaming one splits a category's history in two. Display labels live in
#: `REASON_COPY` and are free to change; these are not.
REASON_ORDER: tuple[str, ...] = (
    "incorrect_information",
    "incomplete_answer",
    "misunderstood_question",
    "unsafe_chemical_advice",
    "other",
)

#: Display labels for `REASON_ORDER`. The EN branch is the label set that was
#: already in `app.py`; the ES branch is new. Keys are the stored values, so a
#: category added to the taxonomy without a label fails the parity test.
REASON_COPY: dict[str, dict[str, str]] = {
    "en": {
        "incorrect_information": "Incorrect information",
        "incomplete_answer": "Incomplete answer",
        "misunderstood_question": "Misunderstood my question",
        "unsafe_chemical_advice": "Unsafe chemical advice",
        "other": "Other",
    },
    "es": {
        "incorrect_information": "Información incorrecta",
        "incomplete_answer": "Respuesta incompleta",
        "misunderstood_question": "No entendió mi pregunta",
        "unsafe_chemical_advice": "Consejo químico peligroso",
        "other": "Otro",
    },
}


RETRIEVE_ROTATION: tuple[str, ...] = ("retrieve", "retrieve_alt1", "retrieve_alt2")


# =====================================================================
# LANGUAGE
# =====================================================================
# Ported verbatim from `detectLang` in Design/demo-deep-water.html.
#
# Why a local heuristic at all, when the graph's planner returns
# `detected_language`: the first status line renders *before* the graph is
# called, so there is nothing to read yet. The planner's answer is adopted for
# the remaining lines as soon as it arrives — see `app.py::run_turn`.

_ES_CHARS = frozenset("¿¡áéíóúñ")

# Byte-identical to the demo's `es` array, order included, so the two
# implementations cannot drift. Note the trailing space on "sal " — it is in
# the demo to stop "sal" matching inside "salt"/"salida", and matters because
# the text is padded with spaces on both sides before testing.
_ES_WORDS: tuple[str, ...] = (
    "agua", "cloro", "piscina", "hola", "gracias", "turbia", "sal ", "aviso",
    "ayuda", "mantenimiento", "filtro", "persona", "tragué", "trague",
    "bebí", "bebi", "qué", "como", "cómo", "por qué", "limpiar", "verde",
)


def detect_language(text: str) -> str:
    """
    "es" or "en", from the user's own words.

    Spanish punctuation or an accent decides it outright; otherwise a single
    hit in the keyword list is enough. Biased towards Spanish on purpose — the
    test cohort writes Spanish, and a Spanish status line in front of an
    English speaker is a smaller failure than the reverse.
    """
    if not text:
        return "en"
    padded = f" {text.lower()} "
    if any(ch in _ES_CHARS for ch in padded):
        return "es"
    return "es" if any(word in padded for word in _ES_WORDS) else "en"


def action_label(language: str, key: str) -> str:
    """
    A label from ACTION_COPY, falling back to English on an unknown language
    and on a key the requested branch happens to be missing.
    """
    table = ACTION_COPY.get(language, ACTION_COPY["en"])
    return table.get(key, ACTION_COPY["en"].get(key, ""))


def agent_line(language: str, agent: str) -> str:
    """
    The status line for a sub-agent, by the name the planner assigned.

    Unknown names fall back rather than raising: the backend's agent list can
    grow without this file, and a turn must never break because of a label.
    """
    table = AGENT_COPY.get(language, AGENT_COPY["en"])
    return table.get(agent) or table["_fallback"]


def status_line(language: str, stage: str) -> str:
    """
    One status string, falling back to English for an unknown language rather
    than raising — a status line is never worth failing a turn over.
    """
    table = STATUS_COPY.get(language, STATUS_COPY["en"])
    return table.get(stage, STATUS_COPY["en"].get(stage, ""))


def form_label(language: str, key: str) -> str:
    """
    A string from FORM_COPY, with the same two fallbacks as `action_label`:
    unknown language, then a key the requested branch happens to be missing.
    """
    table = FORM_COPY.get(language, FORM_COPY["en"])
    return table.get(key, FORM_COPY["en"].get(key, ""))


def reason_label(language: str, reason: str) -> str:
    """
    The display label for a stored feedback category.

    Falls back to the English label, then to the stored value itself — an
    unlabelled category still has to be pickable, or the reader cannot finish
    the form at all.
    """
    table = REASON_COPY.get(language, REASON_COPY["en"])
    return table.get(reason) or REASON_COPY["en"].get(reason) or reason


def suggestions(language: str) -> tuple[str, ...]:
    """
    The opener deck in the reader's language, falling back to English on an
    unknown one — the same contract as every other accessor here.
    """
    return SUGGESTION_DECKS.get(language, SUGGESTIONS)


def reason_choices(language: str) -> dict[str, str]:
    """
    `{display label: stored value}` in REASON_ORDER, ready for a picker.

    Returned as a mapping rather than two parallel lists so the caller cannot
    pair a label with the wrong value, and insertion order carries the order
    the picker should list.
    """
    return {reason_label(language, reason): reason for reason in REASON_ORDER}