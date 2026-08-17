"""
theme.py
========
Look-and-feel layer for the Streamlit UI. Presentation only — this module never
touches graph state, Langfuse, Neo4j or Qdrant, and importing it has no side
effects beyond `import streamlit`.

Layout it produces (the demo page, rebuilt with real Streamlit widgets):

    ┌──────────────────────────────────── [ Light ●──── ] ┐   theme slider, top right
    │                 IA Pool Assistant V1                │   page title
    │        short description + accepted languages       │   page subtitle
    │              ┌──────────────────────┐               │
    │              │  Pool Assistant      │               │   phone bezel + screen
    │              │  ● Online            │               │
    │              │  … chat …            │               │
    │              │  [ Ask anything… ↑ ] │               │
    │              └──────────────────────┘               │
    │            AI/ML Team — Bogotá, Colombia            │   footer
    │                UX/UI first test phase               │
    └─────────────────────────────────────────────────────┘

Palettes are the token blocks lifted from the approved prototypes at the repo
root:

    demo-deep-water.html     ->  DEEP_WATER     (dark mode)
    demo-sunlit-lagoon.html  ->  SUNLIT_LAGOON  (light mode)

Those two files share byte-identical markup and CSS and differ only in their
`:root` token block, so this module keeps the same split: one dict per palette,
one stylesheet that reads every colour through `var(--dw-*)`. Adding a third
palette means adding a third dict, never forking the stylesheet.

Mode selection
--------------
The slider drives *this* stylesheet, not Streamlit's native theme — Streamlit
exposes no API for an app to switch its own theme at runtime. That works here
because the CSS below paints every surface the viewer actually sees: page,
bezel, screen, bubbles, composer, buttons.

It only works, though, if the CSS is *self-sufficient*. Streamlit publishes no
CSS custom properties for its own theme — the emotion classes it generates
bake `theme.colors.bodyText` into each element — so anything this stylesheet
does not explicitly colour is painted by whatever the viewer picked in the ⋮
menu, on top of whatever palette the slider picked. Those two controls are
independent, and the mismatched combinations are unreadable (see the ink
enforcement block in section 5). Every text node inside the phone must
therefore be given a colour here; none may be left to inherit from Streamlit.

`.streamlit/config.toml` covers the remainder — the surfaces that render in
portals at the document root and no rule scoped to the phone can reach (menu
popover, toasts, tooltips, selectbox dropdowns). It pins the *page frame*
colours, which are identical in both palettes, into `[theme]`, `[theme.light]`
and `[theme.dark]` alike, so the ⋮ menu cannot change anything and the slider
stays the single source of truth. That file explains why at length.

Selector note: rules target Streamlit's `data-testid` attributes and the
`.st-key-*` classes produced by `st.container(key=...)`. Both are stable but
neither is a public API — a Streamlit upgrade can move them. Every
role-dependent rule matches two selectors: the marker span this module emits
itself (authoritative, ours) and Streamlit's avatar testid (fallback).
"""

from __future__ import annotations

import streamlit as st


# =====================================================================
# PALETTES
# =====================================================================
# Keys are CSS custom property names so the prototypes' vocabulary carries
# over unchanged and a token can be diffed against the HTML it came from.
# `--dw-*` are the prototypes' own tokens; `--pa-*` are additions Streamlit
# needs and a phone mock never had (page frame, bezel, the mode slider).

DEEP_WATER: dict[str, str] = {
    # -- phone screen surface & structure -----------------------------------
    "--dw-surface": "linear-gradient(168deg,#0d1b3d 0%,#0e2a4f 55%,#0d3a54 100%)",
    "--dw-glass-fill": "rgba(255,255,255,.055)",
    "--dw-glass-border": "rgba(255,255,255,.12)",
    "--dw-radius": "16px",
    "--dw-card-shadow": "none",
    # -- accents -----------------------------------------------------------
    "--dw-teal": "#59d0dd",
    "--dw-mint": "#3ee6a0",
    "--dw-on-teal": "#06222b",
    # -- text --------------------------------------------------------------
    "--dw-text": "#eaf3f7",
    "--dw-text-ast": "#dcebf2",
    "--dw-text-ctx": "#8fb6c9",
    "--dw-brand-ink": "#ffffff",
    # -- chips -------------------------------------------------------------
    "--dw-chip-text": "#a7e6ee",
    "--dw-chip-border": "rgba(89,208,221,.55)",
    "--dw-chip-fill": "rgba(13,42,79,.4)",
    # -- user bubble -------------------------------------------------------
    "--dw-bubble-fill": "rgba(89,208,221,.14)",
    "--dw-bubble-border": "rgba(89,208,221,.35)",
    "--dw-bubble-text": "#e9fbfd",
    # -- composer ----------------------------------------------------------
    "--dw-input-bg": "rgba(255,255,255,.08)",
    "--dw-input-border": "rgba(255,255,255,.16)",
    "--dw-input-shadow": "none",
    # -- citation ------------------------------------------------------------
    # Verbatim from option-c-deep-water.html's `.src` rule.
    "--dw-src-border": "rgba(89,208,221,.5)",
    # -- type --------------------------------------------------------------
    "--dw-serif": 'Georgia,"Times New Roman",serif',
    "--dw-sans": '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif',
    # -- page frame around the phone (Streamlit-only) ----------------------
    # Verbatim from the prototypes' `--page` / `--page-ink` / `--page-mut`.
    # Both palettes keep the page light: the mode slider repaints the phone's
    # interior, not the desk it sits on. An earlier revision darkened the page
    # in dark mode; that lost the pale frame the approved renders show around
    # the bezel, so it is reverted here.
    "--pa-page": "#f3f4f8",
    "--pa-page-ink": "#141a33",
    "--pa-page-mut": "#5a6078",
    # Near-black bezel, as in the prototypes: 17.33:1 on the light page, so it
    # needs no rim to draw its edge.
    "--pa-bezel": "#0c0f1e",
    "--pa-bezel-rim": "transparent",
    "--pa-bezel-shadow": "0 24px 60px rgba(20,26,51,.22)",
    # -- mode slider (Streamlit-only) --------------------------------------
    # The pill the label and the switch sit in, so the control reads as one
    # object on the page rather than a word next to a widget.
    "--pa-chrome-fill": "rgba(255,255,255,.72)",
    "--pa-chrome-border": "rgba(20,26,51,.10)",
    "--pa-chrome-shadow": "0 2px 10px rgba(20,26,51,.06)",
    "--pa-track": "rgba(13,58,68,.08)",
    # .54 rather than .20, to clear the 3:1 UI-component threshold.
    "--pa-track-border": "rgba(13,58,68,.54)",
    # On = dark mode, so the track fills with the bezel's ink and the teal knob
    # reads 6.4:1 against it — the fill *is* the state, not just the position.
    "--pa-track-on": "#0c0f1e",
    "--pa-track-on-border": "#0c0f1e",
    "--pa-knob-shadow": "0 1px 4px rgba(12,15,30,.38)",
    "--pa-focus-ring": "rgba(89,208,221,.45)",
}

SUNLIT_LAGOON: dict[str, str] = {
    # -- phone screen surface & structure -----------------------------------
    "--dw-surface": "linear-gradient(168deg,#fbfdfd 0%,#eef7f8 55%,#dff0f3 100%)",
    "--dw-glass-fill": "rgba(255,255,255,.72)",
    "--dw-glass-border": "rgba(13,58,68,.12)",
    "--dw-radius": "16px",
    "--dw-card-shadow": "0 4px 16px rgba(13,58,68,.07)",
    # -- accents -----------------------------------------------------------
    "--dw-teal": "#0b7480",
    "--dw-mint": "#0c8a56",
    "--dw-on-teal": "#ffffff",
    # -- text --------------------------------------------------------------
    "--dw-text": "#123a46",
    "--dw-text-ast": "#16414e",
    "--dw-text-ctx": "#3e6470",
    "--dw-brand-ink": "#0d2b33",
    # -- chips -------------------------------------------------------------
    "--dw-chip-text": "#0b6570",
    "--dw-chip-border": "rgba(11,116,128,.8)",
    "--dw-chip-fill": "rgba(14,154,167,.08)",
    # -- user bubble -------------------------------------------------------
    "--dw-bubble-fill": "rgba(14,154,167,.12)",
    "--dw-bubble-border": "rgba(14,154,167,.35)",
    "--dw-bubble-text": "#123a46",
    # -- composer ----------------------------------------------------------
    "--dw-input-bg": "#ffffff",
    "--dw-input-border": "rgba(13,58,68,.16)",
    "--dw-input-shadow": "0 2px 10px rgba(13,58,68,.05)",
    # -- citation ------------------------------------------------------------
    # Verbatim from option-c-sunlit-lagoon.html's `.src` rule.
    "--dw-src-border": "rgba(11,116,128,.5)",
    # -- type --------------------------------------------------------------
    "--dw-serif": 'Georgia,"Times New Roman",serif',
    "--dw-sans": '-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif',
    # -- page frame around the phone (Streamlit-only) ----------------------
    # --page / --page-ink / --page-mut, verbatim from both prototypes.
    "--pa-page": "#f3f4f8",
    "--pa-page-ink": "#141a33",
    "--pa-page-mut": "#5a6078",
    # Verbatim from the prototypes: near-black bezel reads 17.33:1 on the light
    # page, so it needs no rim.
    "--pa-bezel": "#0c0f1e",
    "--pa-bezel-rim": "transparent",
    "--pa-bezel-shadow": "0 24px 60px rgba(20,26,51,.22)",
    # -- mode slider (Streamlit-only) --------------------------------------
    "--pa-chrome-fill": "rgba(255,255,255,.82)",
    "--pa-chrome-border": "rgba(13,58,68,.12)",
    "--pa-chrome-shadow": "0 2px 10px rgba(13,58,68,.06)",
    "--pa-track": "rgba(13,58,68,.08)",
    # .54 rather than .20, to clear the 3:1 UI-component threshold.
    "--pa-track-border": "rgba(13,58,68,.54)",
    # Only ever painted while the switch is on, which means the dark palette is
    # live — kept here so both palettes declare the same keys.
    "--pa-track-on": "#0c0f1e",
    "--pa-track-on-border": "#0c0f1e",
    "--pa-knob-shadow": "0 1px 4px rgba(13,58,68,.28)",
    "--pa-focus-ring": "rgba(11,116,128,.35)",
}

PALETTES = {"dark": DEEP_WATER, "light": SUNLIT_LAGOON}

#: Serif italic label above each assistant turn, per the prototypes.
ASSISTANT_LABEL = "Your assistant"

#: Session-state key holding the slider position.
MODE_KEY = "pa_dark_mode"

#: Phone geometry, from the prototypes' `.phone` / `.screen` rules.
PHONE_WIDTH_PX = 392
#: Height of the scrolling conversation area. Chosen so the bezel totals the
#: prototypes' 718px `.phone` height once the header, composer and the screen's
#: own padding are added — measured against demo-deep-water.html, so it moves
#: whenever those insets do. It moved from 574 to 556 with the bubble fixes:
#: the header block used to lean on the -1rem Streamlit puts under every
#: markdown container, which made its box 16px shorter than the brand and
#: status line it holds. Zeroing that (see the rule near .pa-src) gave the
#: header its real height back, and the screen's padding-top went 16 → 18 at
#: the same time to match `.hdr`.
SCREEN_HEIGHT_PX = 556


# =====================================================================
# STYLESHEET
# =====================================================================
# Deliberately free of Python interpolation: every colour is a var() lookup,
# so the rules below are identical for both palettes and there are no braces
# to escape.

_STATIC_CSS = r"""
/* ==================================================================
   1. Page frame — the surface the phone sits on
   ================================================================== */
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
.stApp {
  background: var(--pa-page);
}
header[data-testid="stHeader"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] {
  background: transparent;
}

html, body,
[data-testid="stAppViewContainer"] {
  font-family: var(--dw-sans);
  -webkit-font-smoothing: antialiased;
}

/* ==================================================================
   2. Page title / subtitle / footer  (sans, centred — as in the demo)
   ================================================================== */
.pa-title {
  font-family: var(--dw-sans);
  font-size: 19px;
  font-weight: 800;
  letter-spacing: -.01em;
  color: var(--pa-page-ink);
  text-align: center;
  margin: 2px 0 0;
}
.pa-sub {
  font-size: 12.5px;
  line-height: 1.5;
  color: var(--pa-page-mut);
  text-align: center;
  max-width: 480px;
  /* !important because this is a <p>: Streamlit's own
     `[data-testid="stMarkdownContainer"] p { margin: ... }` outranks a single
     class, so the `auto` was dropped and the 480px block sat against the
     column's left edge — centred text, but 112px left of the phone it is
     meant to sit under. The margin is what aligns it with the bezel. */
  margin: 6px auto 20px !important;
}
.pa-footer {
  text-align: center;
  margin: 18px auto 0;
  max-width: 420px;
}
.pa-footer .pa-team {
  font-family: var(--dw-serif);
  font-size: 13.5px;
  color: var(--pa-page-ink);
}
.pa-footer .pa-phase {
  font-size: 11.5px;
  color: var(--pa-page-mut);
  margin-top: 3px;
  letter-spacing: .04em;
  text-transform: uppercase;
}

/* ==================================================================
   3. Mode slider — top right, drives this stylesheet
   ================================================================== */
/* The label and the switch share one pill: sized to its contents and pushed
   right, which is the alignment the container's horizontal_alignment gave it
   before it had a surface of its own. Streamlit's horizontal container also
   lets its children flex, so they are pinned to their natural width — a
   stretching child would drag the pill's right edge past the switch. */
.st-key-pa-themebar {
  width: fit-content !important;
  margin-left: auto !important;
  align-items: center !important;
  gap: 11px !important;
  padding: 5px 7px 5px 14px !important;
  border-radius: 999px;
  background: var(--pa-chrome-fill);
  border: 1px solid var(--pa-chrome-border);
  box-shadow: var(--pa-chrome-shadow);
}
.st-key-pa-themebar > div,
.st-key-pa-themebar [data-testid="stElementContainer"] {
  flex: 0 0 auto !important;
  width: auto !important;
  margin: 0 !important;
}
.pa-mode-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: .09em;
  text-transform: uppercase;
  color: var(--pa-page-mut);
  white-space: nowrap;
  line-height: 26px;   /* the track's height, so the two share a midline */
}

/* The native toggle's own visuals are hidden and the track + knob are drawn
   as pseudo-elements on the label, which stays the click target. Two
   selectors because the testid may sit on the wrapper or on the label. */
.st-key-pa-themebar [data-testid="stCheckbox"] label,
.st-key-pa-themebar label[data-testid="stCheckbox"] {
  position: relative;
  display: inline-block;
  width: 46px;
  height: 26px;
  margin: 0;
  cursor: pointer;
}
.st-key-pa-themebar [data-testid="stCheckbox"] label > *,
.st-key-pa-themebar label[data-testid="stCheckbox"] > * {
  opacity: 0 !important;   /* input stays clickable: opacity, not display */
}
.st-key-pa-themebar [data-testid="stCheckbox"] label::before,
.st-key-pa-themebar label[data-testid="stCheckbox"]::before {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: 999px;
  background: var(--pa-track);
  border: 1px solid var(--pa-track-border);
  transition: background .25s, border-color .25s;
}
/* On = dark mode, and the track fills with the bezel's ink to say so: the
   position alone was a 1.4:1 read against a near-white track, while the teal
   knob on the filled track is 6.4:1. The word beside it still names the mode,
   so the state never rests on colour alone. */
.st-key-pa-themebar [data-testid="stCheckbox"] label:has(input:checked)::before,
.st-key-pa-themebar label[data-testid="stCheckbox"]:has(input:checked)::before,
.st-key-pa-themebar [data-testid="stCheckbox"] label[data-selected="true"]::before,
.st-key-pa-themebar label[data-testid="stCheckbox"][data-selected="true"]::before {
  background: var(--pa-track-on);
  border-color: var(--pa-track-on-border);
}
.st-key-pa-themebar [data-testid="stCheckbox"] label::after,
.st-key-pa-themebar label[data-testid="stCheckbox"]::after {
  content: "";
  position: absolute;
  top: 4px;
  left: 4px;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--dw-teal);
  box-shadow: var(--pa-knob-shadow);
  transition: transform .22s cubic-bezier(.34,1.4,.64,1), background .25s;
}
/* Knob right = dark mode. 46 − 4 − 18 − 4 = 20, so it lands on the far inset. */
.st-key-pa-themebar [data-testid="stCheckbox"] label:has(input:checked)::after,
.st-key-pa-themebar label[data-testid="stCheckbox"]:has(input:checked)::after,
.st-key-pa-themebar [data-testid="stCheckbox"] label[data-selected="true"]::after,
.st-key-pa-themebar label[data-testid="stCheckbox"][data-selected="true"]::after {
  transform: translateX(20px);
}
/* Hover deepens the rim, press shrinks the knob — the only feedback the
   control gets, since Streamlit's own focus and hover styling is hidden with
   the native widget. */
.st-key-pa-themebar label:hover::before { border-color: var(--dw-teal); }
.st-key-pa-themebar label:active::after { transform: scale(.92); }
.st-key-pa-themebar label:has(input:checked):active::after {
  transform: translateX(20px) scale(.92);
}
.st-key-pa-themebar label:has(input:focus-visible)::before {
  box-shadow: 0 0 0 3px var(--pa-focus-ring);
}

/* ==================================================================
   4. Phone bezel + screen
   ================================================================== */
.st-key-pa-phone {
  width: 392px;
  max-width: 100%;
  margin: 0 auto;
  padding: 10px;
  border-radius: 38px;
  background: var(--pa-bezel);
  /* The rim is drawn as an inset outline, not a border: `width` is border-box
     here, so a 1px border would spend 2px of the 392px on the bezel edge and
     leave a 370px screen where the prototypes have 372 — every inset inside
     the phone would then be 1px shy. An outline costs no layout. */
  outline: 1px solid var(--pa-bezel-rim);
  outline-offset: -1px;
  box-shadow: var(--pa-bezel-shadow);
}
.st-key-pa-screen {
  border-radius: 30px;
  overflow: hidden;
  background: var(--dw-surface);
  color: var(--dw-text);
  /* Top inset is `.hdr`'s 18px; the bottom is `.composer`'s 18px
     padding-bottom, since the composer is a Streamlit widget with no padding
     of its own. Both are in the SCREEN_HEIGHT_PX arithmetic. */
  padding: 18px 14px 18px;
  gap: 0;
}

/* Scroll area: the prototypes hide the scrollbar inside the phone.
   A height-constrained st.container() also draws its own hairline box —
   that rectangle is not in the design, so it is removed here (and on the
   border wrapper Streamlit puts around it, whichever node carries the key). */
.st-key-pa-scroll,
.st-key-pa-scroll > div,
.st-key-pa-scroll [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stVerticalBlockBorderWrapper"]:has(> .st-key-pa-scroll),
.st-key-pa-screen,
.st-key-pa-screen [data-testid="stVerticalBlockBorderWrapper"] {
  border: 0 !important;
  outline: 0 !important;
  box-shadow: none !important;
}
.st-key-pa-scroll,
.st-key-pa-scroll > div { background: transparent !important; }
/* Padding-top is the only thing separating the header from the first turn, so
   it carries the whole gap: `.hdr`'s 10px bottom plus `.body`'s 14px top.
   The sides and bottom restate `.body` in the
   prototypes (`padding:14px 18px 8px; gap:14px`): a height-constrained
   st.container() carries 15px of Streamlit's own padding, which stacked on the
   screen's 14px and set the turns 29px in — 11px past the design, so every
   card was ~22px narrower than the prototypes' and its text wrapped early.
   The 4px tops the screen's 14px inset up to 18px; the gap is the space
   between turns, which Streamlit otherwise sets to 8.8px. */
.st-key-pa-scroll { padding: 24px 4px 8px; gap: 14px !important; }
.st-key-pa-scroll::-webkit-scrollbar { width: 0; height: 0; }
.st-key-pa-scroll { scrollbar-width: none; }

/* Phone header — serif brand + heartbeat dot. Nothing draws the boundary
   between it and the conversation: the gap below does the work, as in the
   prototypes. A hairline was tried here and read as a toolbar edge. */
.pa-brand {
  font-family: var(--dw-serif);
  font-size: 19px;
  color: var(--dw-brand-ink);
  line-height: 1.2;
  /* `.hdr` in the prototypes is inset 20px; the screen already gives 14px, so
     the brand and the status line top that up by 6. Horizontal only — the
     header's vertical rhythm is spent in SCREEN_HEIGHT_PX. */
  padding-left: 6px;
}
.pa-ctx {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11.5px;
  color: var(--dw-text-ctx);
  margin: 3px 0 0;
  padding-left: 6px;   /* see .pa-brand */
}
.pa-ctx .pa-dot {
  flex: 0 0 7px;
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--dw-mint);
  box-shadow: 0 0 8px var(--dw-mint);
}
.pa-ctx.pa-off .pa-dot {
  background: var(--dw-text-ctx);
  box-shadow: none;
}

/* Streamlit dims any element still mounted from the previous run to
   opacity:theme.stale (~0.3) for up to 1.5s while a rerun is in flight
   (`data-stale="true"` on stElementContainer). A rerun landing mid-turn — a
   suggestion chip, a fresh prompt — catches the whole phone in that dip,
   which reads as unreadable, washed-out text. Nothing in this UI depends on
   that affordance, so it is neutralised everywhere inside the phone. */
.st-key-pa-phone [data-stale="true"] {
  opacity: 1 !important;
  transition: none !important;
}

/* ==================================================================
   5. Chat turns
   ================================================================== */
/* The prototypes carry no avatars — the card itself signals who spoke. */
[data-testid^="stChatMessageAvatar"] { display: none !important; }

/* Role markers emitted by role_marker(); invisible and layout-free. */
.pa-role { display: none !important; }

[data-testid="stChatMessage"] {
  background: transparent;
  border: 0;
  padding: 0;
  gap: 0;
}

/* --- assistant: glass card, bottom-left tail (16/16/16/4) --------- */
[data-testid="stChatMessage"]:has([data-pa-role="assistant"]),
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
  background: var(--dw-glass-fill);
  border: 1px solid var(--dw-glass-border);
  border-radius: var(--dw-radius) var(--dw-radius) var(--dw-radius) 4px;
  box-shadow: var(--dw-card-shadow);
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
  /* `.glass` in the prototypes: padding 13px 14px, max-width 95%. Both were
     short here (12/13 and 80%), which cost the card ~50px of width and pulled
     the text in towards the border twice over — narrower lines *and* a
     thinner inset. Read together with the 18px column inset on
     .st-key-pa-scroll: 95% only lands on the prototypes' 319px card if the
     column underneath it is the prototypes' 336px. */
  padding: 13px 14px;
  max-width: 95%;
  color: var(--dw-text-ast);
}

/* --- user: teal bubble, right aligned, bottom-right tail (16/16/4/16) */
[data-testid="stChatMessage"]:has([data-pa-role="user"]),
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: var(--dw-bubble-fill);
  border: 1px solid var(--dw-bubble-border);
  border-radius: var(--dw-radius) var(--dw-radius) 4px var(--dw-radius);
  color: var(--dw-bubble-text);
  padding: 10px 14px;      /* `.usr` in the prototypes */
  width: fit-content;
  max-width: 82%;          /* `.usr` — narrower than the assistant's 95% */
  margin-left: auto;
}

/* `.glass .tx` is 14px/1.55; `.usr` is 14px/1.45 — the tighter leading is what
   keeps a one-line user bubble compact. */
[data-testid="stChatMessage"] p {
  font-size: 14px;
  line-height: 1.55;
}
[data-testid="stChatMessage"]:has([data-pa-role="user"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
  line-height: 1.45;
}
[data-testid="stChatMessage"] p:last-child { margin-bottom: 0; }

/* Streamlit pairs `p { margin-bottom: 1rem }` with `margin-bottom: -1rem` on
   the markdown container, so a block of markdown measures exactly as tall as
   its text. The `p:last-child` reset above removes the paragraph's 1rem but
   not the container's -1rem, which left every markdown block inside a bubble
   16px shorter than the text it wraps: the closing line of an answer, and the
   whole of a one-line user bubble, rendered *below* the card's border instead
   of inside its padding. Zeroing the compensation makes the box honest again;
   it must stay paired with the reset above (drop that and the 1rem comes
   back as a gap under every bubble). */
.st-key-pa-screen [data-testid="stMarkdownContainer"] {
  margin-bottom: 0 !important;
}

/* Inside a card the spacing is carried by the elements themselves — .pa-who's
   6px below the label, .pa-src's 9px above the citation — exactly as in the
   prototypes. Streamlit's 8.8px block gap would add to both, and (because the
   invisible .pa-role marker is still a flex item) would also spend one gap
   above the label, so the card's top inset read 21.8px instead of 13. */
[data-testid="stChatMessage"] [data-testid="stVerticalBlock"] { gap: 0 !important; }

/* --- ink enforcement: hand every Streamlit-painted text node back to us ---
   Streamlit exposes no CSS custom properties for its own theme. The emotion
   classes it generates bake `theme.colors.bodyText` straight into each
   element, so the `color` set on the bubble above is never inherited by the
   markdown inside it — Streamlit's explicit colour wins over inheritance
   regardless of specificity. Message text was therefore painted in the
   *native* theme's ink, and turned unreadable whenever the ⋮ menu disagreed
   with our slider: near-black on the Deep Water navy, and (symmetrically,
   though it was never the reported symptom) near-white on Sunlit Lagoon.
   Sunlit Lagoon only ever looked correct because Streamlit's light bodyText
   happens to read on a pale surface — coincidence, not theming.

   `inherit` rather than a token, deliberately: it hands the decision back to
   the container rules above, so an assistant card resolves to --dw-text-ast,
   a user bubble to --dw-bubble-text, and anything else inside the phone to
   the --dw-text set on .st-key-pa-screen. One rule covers both roles and the
   palette stays declared in exactly one place.

   An element list rather than `*`: the .pa-* nodes this module emits are
   themselves markdown children (assistant_label, citation, phone_header) and
   carry their own accent colours, which must survive. Every one of them is a
   <div>, which is why <div> is absent below. */
.st-key-pa-screen [data-testid="stChatMessageContent"],
.st-key-pa-screen [data-testid="stMarkdownContainer"],
.st-key-pa-screen .stMarkdown,
.st-key-pa-screen [data-testid="stMarkdownContainer"]
  :is(p, li, ul, ol, span, strong, em, h1, h2, h3, h4, h5, h6, blockquote, td, th) {
  color: inherit !important;
}

/* Links and inline code keep an accent instead of inheriting the body ink.
   Both are palette tokens, so they follow the slider like everything else.
   Inline code matters here: both app.py and preview_ui.py render the planner
   steps with `[agent_name]` in backticks. */
.st-key-pa-screen [data-testid="stMarkdownContainer"] a {
  color: var(--dw-teal) !important;
}
.st-key-pa-screen [data-testid="stMarkdownContainer"] code {
  color: var(--dw-chip-text) !important;
  background: var(--dw-chip-fill) !important;
  border: 1px solid var(--dw-chip-border);
  border-radius: 5px;
  padding: 0 4px;
  font-size: 12px;
}

.pa-who {
  font-family: var(--dw-serif);
  font-style: italic;
  font-size: 12.5px;
  color: var(--dw-teal);
  /* `.glass .who` — 6px, not the 5px of `.ast .who`: the label always sits
     inside a card here. */
  margin-bottom: 6px;
  /* The prototypes set no leading on `.who`, so its line box is 14px; under
     Streamlit's body 1.6 it grew to 20 and pushed the answer 6px down inside
     the card. Same for `.pa-src` below. */
  line-height: normal;
}

/* Citation — "Manual · §5 Maintenance", a dashed-underline chip closing an
   answer that is grounded in a document. Verbatim from the prototypes' `.src`
   rule (option-c-deep-water.html / option-c-sunlit-lagoon.html). */
.pa-src {
  display: inline-flex;
  gap: 6px;
  font-size: 11.5px;
  font-weight: 700;
  color: var(--dw-teal);
  margin-top: 9px;
  line-height: normal;   /* see .pa-who */
  border-bottom: 1px dashed var(--dw-src-border);
  padding-bottom: 2px;
}

/* ==================================================================
   6. Composer — inline inside the phone
   ==================================================================
   The prototypes' composer is one floating pill: translucent fill, 22px
   radius, a hairline border, and a teal disc holding an up arrow.

   Streamlit builds the widget from nested divs and paints its own opaque
   surface on an *inner* one, so styling the outer node alone left a white
   square sitting inside a rounded shell. The pill is therefore drawn once on
   the outermost node and every surface inside it is flattened. */

.st-key-pa-screen [data-testid="stChatInput"],
.st-key-pa-screen .stChatInput {
  background: var(--dw-input-bg) !important;
  border: 1px solid var(--dw-input-border) !important;
  border-radius: 22px !important;
  box-shadow: var(--dw-input-shadow) !important;
  /* `.input` in the prototypes: padding 8px 8px 8px 16px around a 34px send
     disc — 52px overall, with the text inset 16px from the pill's left edge.
     Both numbers are measured off demo-deep-water.html, so keep them paired
     with the 34px disc below; changing one alone breaks the pill's height. */
  padding: 8px 8px 8px 16px !important;
  /* `.composer` padding-top. Its 18px bottom is the screen's padding-bottom;
     the 2px sides top up the screen's 14px inset to the composer's 16px, so
     the pill is 340px wide on a 372px screen as in the prototypes. */
  margin: 10px 2px 0;
  overflow: hidden;
}
/* Centre every row, so the placeholder and the send disc share a midline, and
   strip the padding Streamlit puts on its inner wrapper (12px 16px) — that
   surround is what made the pill 68px tall and pushed the text to 30px in
   from the edge. The pill's own padding is the only inset. */
.st-key-pa-screen [data-testid="stChatInput"] > div,
.st-key-pa-screen [data-testid="stChatInput"] div,
.st-key-pa-screen [data-testid="stChatInput"] form {
  align-items: center !important;
  padding: 0 !important;
  /* No flex `gap` here: Streamlit's row carries a zero-width hidden item ahead
     of the textarea, so a gap lands *before* the text and pushes it 10px past
     the 16px inset. `.input`'s 10px gap is drawn as the disc's margin instead
     (below), which is the one place it can only fall between the two. */
  gap: 0 !important;
  /* Text and disc stay on one line. Streamlit re-flows the widget once the
     input holds more than a line's worth of text, dropping the disc onto a row
     of its own and taking the pill to 72px mid-sentence — and it stays there
     after the turn is sent. The prototypes' `.input` is a single flex row at
     all times, so the reflow is pinned out here. */
  flex-direction: row !important;
  flex-wrap: nowrap !important;
}
/* Holding that single row together needs the two flex items sized as well:
   the reflow Streamlit intended hands the text wrapper the full width (it was
   about to own a row of its own), which pushes the disc past the pill's edge
   and `overflow: hidden` then eats it. The text takes what is left, the disc
   keeps its 34px. */
.st-key-pa-screen [data-testid="stChatInput"] div:has(> textarea) {
  flex: 1 1 0% !important;
  min-width: 0 !important;
  width: auto !important;
}
.st-key-pa-screen
[data-testid="stChatInput"] div:has(> [data-testid="stChatInputSubmitButton"]) {
  flex: 0 0 auto !important;
}
/* Inner wrappers, the form, and the focus ring: no surface of their own. */
.st-key-pa-screen [data-testid="stChatInput"] div,
.st-key-pa-screen [data-testid="stChatInput"] form,
.st-key-pa-screen [data-testid="stChatInput"] [data-baseweb],
.st-key-pa-screen [data-testid="stChatInput"]:focus-within {
  background: transparent !important;
  border-color: var(--dw-input-border) !important;
  box-shadow: none !important;
}
.st-key-pa-screen [data-testid="stChatInput"] div {
  border: 0 !important;
}
/* One text line, vertically centred against the disc beside it — the disc,
   not the textarea, sets the row height, exactly as in the prototypes where
   `.input input` has no height of its own. Streamlit sizes the textarea from
   JS with an inline style and grows it as you type, which is what made the
   composer tall — hence !important on all three of min/max/height. Long input
   scrolls inside the pill, as it does in the prototypes, where the composer is
   a single-line <input>. */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"] {
  background: transparent !important;
  color: var(--dw-text) !important;
  font-size: 13.5px !important;
  -webkit-text-fill-color: var(--dw-text);
  box-sizing: border-box !important;
  height: 20px !important;
  min-height: 20px !important;
  max-height: 20px !important;
  line-height: 20px !important;
  padding: 0 !important;
  /* One line that scrolls sideways, like the prototypes' `<input>`: a wrapping
     textarea clamped to 20px would hide everything but the line the caret is
     on. `pre` stops the wrap; the scrollbar is hidden as it is everywhere else
     inside the phone. */
  white-space: pre !important;
  overflow-x: auto;
  overflow-y: hidden;
  scrollbar-width: none;
}
[data-testid="stChatInput"] textarea::-webkit-scrollbar,
[data-testid="stChatInputTextArea"]::-webkit-scrollbar {
  width: 0;
  height: 0;
}
[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInputTextArea"]::placeholder {
  color: var(--dw-text-ctx) !important;
  -webkit-text-fill-color: var(--dw-text-ctx);
  opacity: 1;
}

/* Send control: teal disc, 34px, arrow — never a square.
   34px is `.btn-send` in the prototypes; it is also what sets the pill's
   height, so it moves together with the composer padding above.
   Streamlit's own glyph (an SVG icon, or a ligature-font fallback depending on
   version) is hidden outright — font-size/color collapsed on the button
   itself so no native glyph or stray text node can show through — and the
   arrow is drawn as our own ::after instead. */
[data-testid="stChatInputSubmitButton"] {
  background: var(--dw-teal) !important;
  border: 0 !important;
  border-radius: 50% !important;
  flex: 0 0 34px;
  width: 34px !important;
  height: 34px !important;
  min-width: 34px;
  min-height: 34px;
  padding: 0 !important;
  margin-left: 10px !important;      /* `.input` gap, see the composer above */
  display: flex !important;
  align-items: center;
  justify-content: center;
  color: transparent !important;
  font-size: 0 !important;
  /* Full-strength teal at all times. Streamlit disables the button while the
     field is empty and fades it to .45, which over the navy surface reads as
     #3c7d90 instead of #59d0dd — the muted disc. The prototypes' `.btn-send`
     has no disabled state at all, so the fade is cancelled here rather than
     merely lightened. */
  opacity: 1 !important;
}
[data-testid="stChatInputSubmitButton"] > * { display: none !important; }
[data-testid="stChatInputSubmitButton"]::after {
  content: "\2191";                      /* U+2191 UPWARDS ARROW (↑) — this
                                             string MUST stay in a raw Python
                                             string: in a normal string, Python
                                             reads \21 as a 2-digit octal
                                             escape (an invisible control
                                             char) and leaves the literal "91"
                                             behind, which is the "91" bug. */
  font-family: var(--dw-sans);
  font-size: 16px !important;        /* `.btn-send` font-size */
  font-weight: 800;
  line-height: 1;
  color: var(--dw-on-teal) !important;
}
[data-testid="stChatInputSubmitButton"]:hover {
  background: var(--dw-teal) !important;
  opacity: .88 !important;
}
[data-testid="stChatInputSubmitButton"]:disabled { opacity: 1 !important; }

/* ==================================================================
   7. Buttons -> the prototypes' chip language
   ================================================================== */
.stButton > button,
[data-testid="stBaseButton-secondary"] {
  border-radius: 999px;
  border: 1px solid var(--dw-chip-border);
  background: var(--dw-chip-fill);
  color: var(--dw-chip-text);
  font-weight: 600;
  font-size: 12px;
  padding: .35rem .8rem;
}
.stButton > button:hover,
[data-testid="stBaseButton-secondary"]:hover {
  border-color: var(--dw-teal);
  color: var(--dw-teal);
  background: var(--dw-chip-fill);
}
[data-testid="stBaseButton-primary"] {
  border-radius: 999px;
  border: 0;
  background: var(--dw-teal);
  color: var(--dw-on-teal);
  font-weight: 700;
  font-size: 12px;
}
[data-testid="stBaseButton-primary"]:hover {
  background: var(--dw-teal);
  color: var(--dw-on-teal);
  opacity: .9;
}

/* Opening suggestions: stacked under the greeting, each pill hugging its own
   text, as in the prototypes' first screen. They are the only chips that sit
   outside a card. */
.st-key-pa-chips { gap: .4rem; margin: 2px 0 4px; }
.st-key-pa-chips .stButton > button,
.st-key-pa-chips [data-testid="stBaseButton-secondary"] {
  width: auto;
  font-size: 12.5px;
  padding: 7px 13px;
}

/* Reset control sits on the page, not the phone — page ink, not screen ink. */
.st-key-pa-reset .stButton > button,
.st-key-pa-reset [data-testid="stBaseButton-secondary"] {
  background: transparent;
  border-color: var(--pa-track-border);
  color: var(--pa-page-mut);
}
.st-key-pa-reset .stButton > button:hover,
.st-key-pa-reset [data-testid="stBaseButton-secondary"]:hover {
  border-color: var(--pa-page-mut);
  color: var(--pa-page-ink);
}

/* ==================================================================
   8. Expanders and alerts inside the phone
   ================================================================== */
[data-testid="stExpander"] {
  background: var(--dw-glass-fill);
  border: 1px solid var(--dw-glass-border);
  border-radius: var(--dw-radius);
  box-shadow: var(--dw-card-shadow);
}
/* `st.status` renders as an expander, so its label and chevron are the
   summary row. Streamlit paints both from the native theme, hence the same
   enforcement the chat bubbles need — !important and the child nodes named
   explicitly, since the colour sits on the inner span, not the <summary>. */
[data-testid="stExpander"] summary {
  font-size: 12px;
  color: var(--dw-text-ctx);
}
.st-key-pa-screen [data-testid="stExpander"] summary,
.st-key-pa-screen [data-testid="stExpander"] summary :is(span, p, div) {
  color: var(--dw-text-ctx) !important;
}
.st-key-pa-screen [data-testid="stExpander"] summary svg { fill: currentColor; }
/* The summary row is painted from the native theme's secondaryBackgroundColor
   (it resolves to a near-white #f9fafc). That value is correct for the page,
   which is where config.toml sets it, but the status header lives *inside*
   the phone — on Deep Water it came through as a white band. The expander
   already draws the card, so the summary needs no surface of its own. */
.st-key-pa-screen [data-testid="stExpander"] summary,
.st-key-pa-screen [data-testid="stExpander"] details {
  background-color: transparent !important;
}
[data-testid="stExpander"] summary:hover { color: var(--dw-teal); }

/* !important on the alert colour so the markdown nested inside it — which
   now resolves `inherit` against this node — lands on our ink and not on the
   colour Streamlit sets on stAlertContainer itself. Without it the inherit
   rule above would faithfully inherit the wrong value. */
[data-testid="stAlert"],
[data-testid="stAlertContainer"],
.stAlert {
  border-radius: var(--dw-radius);
  border: 1px solid var(--dw-glass-border);
  background: var(--dw-glass-fill);
  color: var(--dw-text-ast) !important;
  font-size: 12.5px;
}
[data-testid="stAlert"] a,
[data-testid="stAlertContainer"] a { color: var(--dw-teal); }

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
  color: var(--dw-text-ctx) !important;
  font-size: 11.5px;
}

/* st.selectbox — app.py's feedback flow puts one inside the phone. BaseWeb
   paints the closed control from the native theme. Its dropdown opens in a
   portal at the document root, outside .st-key-pa-screen and beyond the
   reach of any rule here; that surface is covered by .streamlit/config.toml
   instead. Same for st.toast, which also portals out. */
.st-key-pa-screen [data-testid="stSelectbox"] :is(div, span, input, button, svg) {
  color: var(--dw-text) !important;
}
.st-key-pa-screen [data-testid="stSelectbox"] svg { fill: currentColor; }
.st-key-pa-screen [data-testid="stSelectbox"] input::placeholder {
  color: var(--dw-text-ctx) !important;
  opacity: 1;
}
/* Same secondaryBackgroundColor leak as the status summary above: the control
   is painted on an inner emotion div, so every surface inside is flattened
   and the pill is drawn once on the node that owns the control.

   Two anchors, because Streamlit changed engines: 1.61 builds the selectbox
   with react-aria (`.react-aria-ComboBox` wrapping a [role="group"]), older
   builds used BaseWeb (`[data-baseweb="select"]`). Matching both keeps this
   working across the >=1.57,<2.0 range pyproject allows. Either selector
   outranks the blanket `div` rule above, so the flattening does not undo it. */
.st-key-pa-screen [data-testid="stSelectbox"] div {
  background-color: transparent !important;
}
.st-key-pa-screen [data-testid="stSelectbox"] .react-aria-ComboBox [role="group"],
.st-key-pa-screen [data-testid="stSelectbox"] [data-baseweb="select"] {
  background-color: var(--dw-input-bg) !important;
  border-color: var(--dw-input-border) !important;
  border-radius: 999px;
}

/* Tighten Streamlit's default vertical rhythm inside the phone. */
.st-key-pa-screen [data-testid="stVerticalBlock"] { gap: .55rem; }
"""


# =====================================================================
# INTERNALS
# =====================================================================

def _token_block(tokens: dict[str, str], selector: str = ":root") -> str:
    """Render a palette dict as a CSS custom-property block."""
    body = "\n".join(f"  {name}: {value};" for name, value in tokens.items())
    return f"{selector} {{\n{body}\n}}"


def current_mode() -> str:
    """
    The palette the slider is set to: "dark" or "light".

    Defaults to dark (Deep Water is the flagship). Reading this before any
    widget renders is safe — it only touches session state.
    """
    return "dark" if st.session_state.get(MODE_KEY, True) else "light"


def _stylesheet(mode: str) -> str:
    return "\n".join([
        f"/* palette: {mode} */",
        _token_block(PALETTES[mode]),
        _STATIC_CSS,
    ])


# =====================================================================
# PUBLIC API
# =====================================================================

def inject_theme() -> str:
    """
    Apply the palette the slider is set to. Call once, immediately after
    `st.set_page_config()`, before any other widget renders. Returns the
    active mode.
    """
    mode = current_mode()
    st.markdown(f"<style>\n{_stylesheet(mode)}\n</style>", unsafe_allow_html=True)
    return mode


def theme_slider() -> None:
    """
    The mode slider, right-aligned at the top of the page: knob left for
    Sunlit Lagoon (light), knob right for Deep Water (dark).

    The label beside it names the *current* mode. Because the widget's key is
    MODE_KEY, clicking it writes straight to session state and the rerun
    repaints with the other palette — `inject_theme()` reads the same key.
    """
    if MODE_KEY not in st.session_state:
        st.session_state[MODE_KEY] = True   # dark by default

    with st.container(
        key="pa-themebar",
        horizontal=True,
        horizontal_alignment="right",
        vertical_alignment="center",
    ):
        label = "Dark" if st.session_state[MODE_KEY] else "Light"
        st.markdown(f'<span class="pa-mode-label">{label}</span>',
                    unsafe_allow_html=True)
        st.toggle(
            "Dark mode",
            key=MODE_KEY,
            label_visibility="collapsed",
            help="Slide right for Deep Water (dark), left for Sunlit Lagoon (light)",
        )


def page_header(title: str, subtitle: str) -> None:
    """Centred page title and the short description above the phone."""
    st.markdown(
        f'<div class="pa-title">{title}</div>'
        f'<p class="pa-sub">{subtitle}</p>',
        unsafe_allow_html=True,
    )


def phone_header(brand: str, status: str, online: bool) -> None:
    """Serif brand line plus the service heartbeat dot, inside the phone."""
    state = "" if online else " pa-off"
    st.markdown(
        f'<div class="pa-brand">{brand}</div>'
        f'<div class="pa-ctx{state}"><i class="pa-dot"></i><span>{status}</span></div>',
        unsafe_allow_html=True,
    )


def page_footer(team: str, phase: str) -> None:
    """Who built it and which test phase this is, below the phone."""
    st.markdown(
        f'<div class="pa-footer">'
        f'<div class="pa-team">{team}</div>'
        f'<div class="pa-phase">{phase}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def role_marker(role: str) -> None:
    """
    Emit the invisible marker that tells the stylesheet which side a turn
    belongs to. Call as the first thing inside a `st.chat_message(...)` block.

    This is what makes the bubble styling independent of Streamlit's internal
    avatar test ids.
    """
    st.markdown(f'<span class="pa-role" data-pa-role="{role}"></span>',
                unsafe_allow_html=True)


def assistant_label(text: str = ASSISTANT_LABEL) -> None:
    """Serif italic label that opens an assistant turn."""
    st.markdown(f'<div class="pa-who">{text}</div>', unsafe_allow_html=True)


def citation(text: str) -> None:
    """
    Dashed-underline source chip closing an answer grounded in a document,
    e.g. "Manual · §5 Maintenance". Call last, inside the same
    `st.chat_message("assistant")` block, only when the answer actually cites
    a source.
    """
    st.markdown(f'<div class="pa-src">{text}</div>', unsafe_allow_html=True)
