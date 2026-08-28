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

import html
import json
import time

import streamlit as st
import streamlit.components.v1 as components  # fallback; see scroll_to_question


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
    # -- keys added with the Fluidra look; defined here so switching
    #    ACTIVE_LOOK back to this palette leaves nothing unresolved.
    "--dw-label": "#59d0dd",
    "--dw-accent-soft": "rgba(89,208,221,.14)",
    "--dw-tail": "4px",
    "--dw-input-radius": "22px",
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
    # The "…" menu's drop shadow. Held at the value the stylesheet used to
    # hardcode, identically in all three palettes, so promoting it to a token
    # changed nothing on screen — it only made the last colour literal in the
    # sheet addressable. Tune it per palette from here if it ever needs to be.
    "--pa-popover-shadow": "0 12px 32px rgba(0,0,0,.45)",
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
    # -- keys added with the Fluidra look; defined here so switching
    #    ACTIVE_LOOK back to this palette leaves nothing unresolved.
    "--dw-label": "#0b7480",
    "--dw-accent-soft": "rgba(11,116,128,.10)",
    "--dw-tail": "4px",
    "--dw-input-radius": "22px",
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
    # Same value in every palette — see the note in DEEP_WATER.
    "--pa-popover-shadow": "0 12px 32px rgba(0,0,0,.45)",
}

# ---------------------------------------------------------------------------
# Fluidra Night — the brand look, taken from the Figma frames (iPhone 16,
# 393pt). Values marked (figma) are read off an inspector field; the rest are
# sampled from the frames, which carry a wide-gamut profile and so land within
# ~1-2% of the true token. Swapping in the exact `Primary Colors` hexes is a
# find-and-replace inside this one dict.
#
# The accent keys keep the names `--dw-teal` / `--dw-mint` even though they now
# hold violet and green. Renaming them would mean touching every rule in the
# stylesheet; leaving them means the whole surface re-skins from this dict
# alone. Read them as "accent" and "signal", not as colours.
# ---------------------------------------------------------------------------

_BARLOW = '"Barlow",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif'

FLUIDRA_NIGHT: dict[str, str] = {
    # -- phone screen surface & structure -----------------------------------
    # Flat, not a gradient: the frames show one even field behind everything.
    "--dw-surface": "#0d091f",
    "--dw-glass-fill": "#211f33",       # assistant card — solid, not glass
    "--dw-glass-border": "#3f3868",
    "--dw-radius": "16px",              # measured off the card and bubble
    "--dw-tail": "0px",                 # the tail corner is square, not 4px
    "--dw-card-shadow": "none",
    # -- accents -----------------------------------------------------------
    "--dw-teal": "#7a3fe8",             # accent violet: send disc, status dots
    "--dw-mint": "#2cbf58",             # online dot
    "--dw-on-teal": "#ffffff",
    # -- text --------------------------------------------------------------
    "--dw-text": "#ffffff",
    "--dw-text-ast": "#ffffff",         # Light/White
    "--dw-text-ctx": "#9c96b8",
    "--dw-brand-ink": "#ffffff",        # text/field-value-light
    "--dw-label": "#d3bff7",            # "Your assistant" — Primary/Purple
    "--dw-accent-soft": "rgba(122,63,232,.14)",
    # -- chips -------------------------------------------------------------
    "--dw-chip-text": "#d3bff7",
    "--dw-chip-border": "#bfa2f4",
    "--dw-chip-fill": "transparent",
    # -- user bubble -------------------------------------------------------
    "--dw-bubble-fill": "#402376",
    "--dw-bubble-border": "#7a3fe8",
    "--dw-bubble-text": "#ffffff",
    # -- composer ----------------------------------------------------------
    "--dw-input-bg": "#24233b",         # (figma)
    "--dw-input-border": "#3f3868",
    "--dw-input-radius": "24px",        # (figma)
    "--dw-input-shadow": "none",
    # -- citation ------------------------------------------------------------
    "--dw-src-border": "rgba(122,63,232,.55)",
    # -- type --------------------------------------------------------------
    # Barlow across the board. `--dw-serif` is kept as a key so the rules that
    # reference it stay valid; in this look it resolves to Barlow too, which is
    # what retires the Georgia italic the earlier design used for labels.
    "--dw-serif": _BARLOW,
    "--dw-sans": _BARLOW,
    # -- page frame around the phone (Streamlit-only) ----------------------
    "--pa-page": "#eaeaea",
    "--pa-page-ink": "#141a33",
    "--pa-page-mut": "#5a6078",
    # The frames draw no bezel, so it takes the screen's own ink and reads as
    # one object; only the outer radius and the shadow remain.
    "--pa-bezel": "#0d091f",
    "--pa-bezel-rim": "transparent",
    "--pa-bezel-shadow": "0 24px 60px rgba(20,26,51,.22)",
    # -- mode slider (Streamlit-only) --------------------------------------
    # Not rendered while this look is active; defined so the keys resolve.
    "--pa-chrome-fill": "rgba(255,255,255,.72)",
    "--pa-chrome-border": "rgba(20,26,51,.10)",
    "--pa-chrome-shadow": "0 2px 10px rgba(20,26,51,.06)",
    "--pa-track": "rgba(13,9,31,.08)",
    "--pa-track-border": "rgba(13,9,31,.54)",
    "--pa-track-on": "#0d091f",
    "--pa-track-on-border": "#0d091f",
    "--pa-knob-shadow": "0 1px 4px rgba(13,9,31,.38)",
    "--pa-focus-ring": "rgba(122,63,232,.45)",
    # Same value in every palette — see the note in DEEP_WATER.
    "--pa-popover-shadow": "0 12px 32px rgba(0,0,0,.45)",
}

PALETTES = {"dark": DEEP_WATER, "light": SUNLIT_LAGOON, "fluidra": FLUIDRA_NIGHT}

#: The look the app renders. "fluidra" is the brand design; "dark" and
#: "light" are the earlier Deep Water / Sunlit Lagoon palettes, kept intact
#: above. "toggle" restores the on-page slider that chose between those two.
#: Changing this one string is the whole revert path.
ACTIVE_LOOK = "fluidra"

#: Serif italic label above each assistant turn, per the prototypes.
ASSISTANT_LABEL = "Your assistant"

#: Session-state key holding the slider position.
MODE_KEY = "pa_dark_mode"

#: Phone geometry, from the prototypes' `.phone` / `.screen` rules.
#:
#: Reaches the stylesheet as the `--pa-phone-width` token (see `_GEOMETRY`), so
#: this constant is the width rather than merely describing it. It used to be
#: neither read nor referenced while the CSS carried its own `392px` literal —
#: two numbers claiming to be the same one, which is exactly the drift
#: `SCREEN_HEIGHT_PX` is passed as an argument to avoid.
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
  font-weight: 700;
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
  width: var(--pa-phone-width);
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
/* margin-top, not more padding-top. The padding is part of the scrollable
   content box, so it scrolls away with the first turn and only spaces things
   at rest; measured with the conversation scrolled, a bubble passed straight
   under the header and sat right against "Online". A margin moves the scroll
   viewport's top edge instead, so the gap is there whatever the reader has
   scrolled to. Raising this grows the phone by the same amount, since the
   scroll area's own height is fixed by SCREEN_HEIGHT_PX. */
.st-key-pa-scroll { margin-top: 10px !important; }
.st-key-pa-scroll { padding: 24px 4px 8px; gap: 14px !important; }
.st-key-pa-scroll::-webkit-scrollbar { width: 0; height: 0; }
.st-key-pa-scroll { scrollbar-width: none; }

/* Phone header — serif brand + heartbeat dot. Nothing draws the boundary
   between it and the conversation: the gap below does the work, as in the
   prototypes. A hairline was tried here and read as a toolbar edge. */
.pa-brand {
  font-family: var(--dw-sans);
  /* NGA/Title Medium/xlarge — 22/28, weight 500. */
  font-size: 22px;
  font-weight: 500;
  color: var(--dw-brand-ink);
  line-height: 28px;
  /* `.hdr` in the prototypes is inset 20px; the screen already gives 14px, so
     the brand and the status line top that up by 6. Horizontal only — the
     header's vertical rhythm is spent in SCREEN_HEIGHT_PX. */
  padding-left: 4px;
}
.pa-ctx {
  display: flex;
  align-items: center;
  gap: 5px;
  /* NGA/Label Regular/medium — 12/16, on the same ink token as the brand
     (`text/field-value-light` in the Figma file, which both share). */
  font-size: 12px;
  line-height: 16px;
  color: var(--dw-brand-ink);
  margin: 4px 0 0;
  padding-left: 4px;   /* see .pa-brand */
}
.pa-ctx .pa-dot {
  flex: 0 0 6px;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--dw-mint);
  /* The frames draw a flat dot; the earlier design's halo is retired. */
  box-shadow: none;
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
  border-radius: var(--dw-radius) var(--dw-radius) var(--dw-radius) var(--dw-tail);
  box-shadow: var(--dw-card-shadow);
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
  /* `.glass` in the prototypes: padding 13px 14px, max-width 95%. Both were
     short here (12/13 and 80%), which cost the card ~50px of width and pulled
     the text in towards the border twice over — narrower lines *and* a
     thinner inset. Read together with the 18px column inset on
     .st-key-pa-scroll: 95% only lands on the prototypes' 319px card if the
     column underneath it is the prototypes' 336px. */
  padding: 12px;
  /* The Figma card is 337px inside a 393px screen with 28px margins, i.e. the
     full content column, and the column here is already 336px wide (372 screen
     − 2×14 screen inset − 2×4 scroll inset). Capping it at 80% is therefore a
     deliberate departure from the frames: it buys a visible margin on both
     sides of the conversation, and costs height, since a narrower column wraps
     the same answer over more lines. */
  max-width: 90%;
  color: var(--dw-text-ast);
}

/* --- user: teal bubble, right aligned, bottom-right tail (16/16/4/16) */
[data-testid="stChatMessage"]:has([data-pa-role="user"]),
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: var(--dw-bubble-fill);
  border: 1px solid var(--dw-bubble-border);
  border-radius: var(--dw-radius) var(--dw-radius) var(--dw-tail) var(--dw-radius);
  color: var(--dw-bubble-text);
  padding: 12px 16px;      /* Figma: 44px tall around a 20px line */
  width: fit-content;
  max-width: 90%;          /* `.usr` — same cap as the card above */
  margin-left: auto;
}

/* `.glass .tx` is 14px/1.55; `.usr` is 14px/1.45 — the tighter leading is what
   keeps a one-line user bubble compact. */
[data-testid="stChatMessage"] p {
  /* Barlow Regular 14/20, letter-spacing 0.1px — read straight off the
     Figma text panel for the assistant body copy. */
  font-size: 14px;
  line-height: 20px;
  letter-spacing: .1px;
}
[data-testid="stChatMessage"]:has([data-pa-role="user"]) p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
  line-height: 20px;
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
  font-family: var(--dw-sans);
  /* NGA/Label Regular/small — 11/16, upright. The Georgia italic the earlier
     design used for this label is retired by the brand type scale. */
  font-style: normal;
  font-size: 11px;
  color: var(--dw-label);
  /* `.glass .who` — 6px, not the 5px of `.ast .who`: the label always sits
     inside a card here. */
  margin-bottom: 8px;   /* Figma: label ends at y120, body starts at y144 */
  /* Spelled out rather than left to the browser: the Figma style is 11/16,
     and Streamlit's body 1.6 would otherwise make this line box 18px and push
     the answer down inside the card. Same reason `.pa-src` sets its own. */
  line-height: 16px;
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

/* Status steps — the honest loading state (option-c-deep-water.html §"Responding",
   FRONTEND.md §8.2). Ported from `.status` / `.step` / `.drops` in
   demo-deep-water.html, renamed to the `pa-` prefix this module uses and to
   `.is-done` / `.is-live` so the state classes cannot collide with Streamlit's
   own `.done` / `.live` should it ever grow them.

   Every line maps to a real graph stage; there are no decorative states, and a
   safety turn shows none of this at all — the template renders complete.

   These rules are NOT scoped to `.st-key-pa-screen`. The markup only ever
   renders inside the phone, and scoping cost nothing but made the selector
   fragile against Streamlit moving the `.st-key-*` class up or down a level. */
.pa-status {
  display: flex;
  flex-direction: column;
  gap: 7px;
  transition: opacity .25s;
}
.pa-step {
  display: flex;
  align-items: center;
  gap: 9px;
  font-size: 12.5px;
  color: var(--dw-text-ctx);
  /* Streamlit's body leading would make each row 20px tall and break the 7px
     rhythm the prototypes set between them — same reason .pa-who resets it. */
  line-height: normal;
}
/* A finished stage keeps the muted ink and gains a mint tick; the stage in
   flight goes teal and semibold. That contrast *is* the progress indicator —
   nothing here is a percentage or a bar, because the graph cannot honestly
   report either. */
.pa-step.is-done .pa-tick {
  color: var(--dw-mint);
  font-size: 11px;
}
.pa-step.is-live {
  color: var(--dw-teal);
  font-weight: 600;
}
/* A stage that has not started yet is present but recessed, so the list does
   not reflow as it fills in — the rows are laid out once and only change ink. */
.pa-step.is-pending {
  opacity: .45;
}

/* Three rising water drops. The one animation this UI runs during a normal
   turn; at 1.2s with a 4px travel it reads as breathing, which is the motion
   budget §7 allows. */
.pa-drops {
  display: inline-flex;
  gap: 3px;
  align-items: flex-end;
  height: 10px;
}
.pa-drops i {
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: var(--dw-teal);
  animation: paRise 1.2s infinite ease-in-out;
}
.pa-drops i:nth-child(2) { animation-delay: .18s; }
.pa-drops i:nth-child(3) { animation-delay: .36s; }
@keyframes paRise {
  0%, 100% { transform: translateY(0);    opacity: .45; }
  50%      { transform: translateY(-4px); opacity: 1;   }
}
/* The drops hold their own column width whether they animate or not, so the
   done/live swap never shifts the label sideways. */
.pa-step.is-done .pa-drops,
.pa-step.is-pending .pa-drops { visibility: hidden; }
/* A div (see status_steps for why), so it needs the flex box spelled out to
   sit on the row's midline like the drops it replaces. 18px is the drops' own
   measured width — three 4px dots and two 3px gaps — so a row's label does not
   shift sideways by a pixel at the moment its stage completes. */
.pa-tick {
  display: inline-flex;
  justify-content: center;
  flex: 0 0 18px;
}

/* One-line status: three violet dots, an 8px gap, then the sentence. This is
   the shape the Fluidra frames draw (`250 × 16`, spacing 8) and it replaces the
   three stacked steps on this look. The stacked component above is left in
   place — nothing references it while ACTIVE_LOOK is "fluidra", and it is what
   the earlier design used.

   The staging survives the change of shape: the sentence itself advances
   through the pipeline (reading → retrieving → writing) rather than sitting on
   one string for the whole wait, which is the point the loading state exists
   to make. */
.pa-pulse {
  display: flex;
  align-items: center;
  gap: 8px;
  height: 16px;
  font-size: 12px;
  line-height: 16px;
  color: var(--dw-text-ctx);
}
.pa-pulse .pa-drops {
  height: 8px;
  gap: 3px;
}
.pa-pulse .pa-drops i {
  width: 4px;
  height: 4px;
}
/* Several rows at once, one per stage the pipeline has actually reached. They
   accumulate downward: a finished row stays on screen, dimmed with its drops
   frozen, and the new row opens beneath it — so the reader can see both what
   is happening now and what it has already been through. 7px matches the
   spacing the earlier stacked status used. */
.pa-stack {
  display: flex;
  flex-direction: column;
  gap: 7px;
}
.pa-pulse.is-done {
  opacity: .42;
}
.pa-pulse.is-done .pa-drops i {
  animation: none;
  opacity: .8;
}

@media (prefers-reduced-motion: reduce) {
  /* §7: drops go static at .8. Nothing else in a normal turn animates. */
  .pa-drops i { animation: none; opacity: .8; }
}

/* The scroll-anchor helper renders through `st.components.v1.html`, which
   mounts a real iframe in document flow. It carries no visuals — collapsing it
   here keeps it from opening a gap under the phone. `height: 0 !important`
   rather than `display: none`: a display-none iframe is not guaranteed to
   execute, and the script inside is the entire point. */
/* The three selectors cover the versions of the helper's own node across the
   >=1.57,<2.0 range: `st.iframe` renders as stIFrame, the deprecated
   `components.html` as stCustomComponentV1, and older builds title the frame
   `streamlit_component`. A `.st-key-pa-scrollhook` block used to sit here as
   well; nothing has ever set that container key, so it matched nothing. */
[data-testid="stCustomComponentV1"],
[data-testid="stIFrame"],
iframe[title="streamlit_component"] {
  height: 0 !important;
  min-height: 0 !important;
  border: 0 !important;
  display: block !important;
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
  border-radius: var(--dw-input-radius) !important;
  box-shadow: var(--dw-input-shadow) !important;
  /* 9px top/bottom, 16px sides. The sides are the Figma text inset; the
     vertical pair is what sets the pill's height, since the 32px send disc is
     the tallest thing in the row: 9 + 32 + 9 = 50px, the same pill height the
     previous look had (8px around a 34px disc). A flat 16px here took it to
     64px, which read as a heavier composer than the one it replaced. Move
     this with the disc size below, never alone. */
  padding: 9px 16px !important;
  /* `.composer` padding-top. Its 18px bottom is the screen's padding-bottom;
     the 2px sides top up the screen's 14px inset to the composer's 16px, so
     the pill is 340px wide on a 372px screen as in the prototypes. */
  margin: 10px 4px 0;
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
  /* NGA/Label Regular/X Large — 16/24. */
  font-size: 16px !important;
  -webkit-text-fill-color: var(--dw-text);
  box-sizing: border-box !important;
  height: 24px !important;
  min-height: 24px !important;
  max-height: 24px !important;
  line-height: 24px !important;
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
  flex: 0 0 32px;
  width: 32px !important;
  height: 32px !important;
  min-width: 32px;
  min-height: 32px;
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
  font-weight: 700;
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
/* The `.stButton > button` selector above is (0,1,1) — one class, one element —
   and this one was `[data-testid="stBaseButton-primary"]` alone, which is
   (0,1,0). The secondary chip rule therefore *outranked* the primary rule, and
   every plain primary button in the app rendered as a ghost chip with
   `type="primary"` silently discarded. The action row's thumbs escaped it only
   because their own rules carry !important for other reasons.
   Fixed by matching the button through its wrapper as well, which takes this to
   (0,2,1) and wins on specificity — no !important needed, so the scoped rules
   further down can still override it where they mean to. */
.stButton > button[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-primary"] {
  border-radius: 999px;
  border: 0;
  background: var(--dw-teal);
  color: var(--dw-on-teal);
  font-weight: 700;
  font-size: 12px;
}
.stButton > button[data-testid="stBaseButton-primary"]:hover,
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
  /* NGA/Label Medium/large — 14/20 — in a 16/6 box, so the pill is 32px tall.
     Streamlit ships `min-height: 40px` on every button, which outranks the
     padding and left the pill 8px over; height is pinned here instead. */
  font-size: 14px;
  line-height: 20px;
  font-weight: 500;
  padding: 6px 16px;
  box-sizing: border-box;
  height: 32px;
  min-height: 32px;
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

/* --- widget labels and the free-text field -------------------------------
   The only widgets inside the phone that carry a visible label are the two in
   the thumbs-down form (`render_feedback_detail`): the reason picker and the
   note field. Both were missing from this sheet, and this module's whole
   premise is that anything it does not colour is painted from the *native*
   theme — which config.toml pins to the page frame, correctly, so the labels
   arrived as #141a33 on the screen's #0d091f (about 1.1:1) and the textarea as
   a white box inside the navy phone.

   Same treatment as every other surface in here: the ink comes from the screen
   tokens, and the field reuses the composer's own fill and border rather than
   introducing a shape of its own. `-webkit-text-fill-color` alongside `color`
   for the same reason the composer needs it — WebKit paints form-control text
   from the fill colour and ignores `color` on its own. */
.st-key-pa-screen [data-testid="stWidgetLabel"],
.st-key-pa-screen [data-testid="stWidgetLabel"] :is(p, span, div, label) {
  color: var(--dw-text-ctx) !important;
  font-size: 11.5px;
}
.st-key-pa-screen [data-testid="stTextArea"] textarea {
  background: var(--dw-input-bg) !important;
  border: 1px solid var(--dw-input-border) !important;
  border-radius: 12px !important;
  color: var(--dw-text) !important;
  -webkit-text-fill-color: var(--dw-text);
  font-family: var(--dw-sans) !important;
  font-size: 13px !important;
  box-shadow: none !important;
}
.st-key-pa-screen [data-testid="stTextArea"] textarea::placeholder {
  color: var(--dw-text-ctx) !important;
  -webkit-text-fill-color: var(--dw-text-ctx);
  opacity: 1;
}
/* Streamlit wraps the field in its own surfaced div, the same leak the
   selectbox and the status summary have; the border is drawn on the textarea
   above, so the wrapper needs no surface of its own. */
.st-key-pa-screen [data-testid="stTextArea"] div {
  background-color: transparent !important;
  border-color: var(--dw-input-border) !important;
}

/* The form's two controls, side by side. Same treatment as the action row and
   the opening chips: Streamlit lets a horizontal container's children flex, so
   they are pinned to their natural width and the pills hug their own text.
   `nowrap` is the load-bearing one — these labels are short enough to look
   safe, and the previous layout (two of eight columns inside a 344px area)
   still broke "Enviar" into six lines of one letter. */
[class*="st-key-pa-detail-actions"] {
  gap: 8px !important;
}
[class*="st-key-pa-detail-actions"] > div,
[class*="st-key-pa-detail-actions"] [data-testid="stElementContainer"] {
  flex: 0 0 auto !important;
  width: auto !important;
  margin: 0 !important;
}
[class*="st-key-pa-detail-actions"] button {
  width: auto !important;
  min-width: 0 !important;
  white-space: nowrap !important;
}

/* ==================================================================
   9. Answer actions — the row under an assistant turn
   ==================================================================
   Two thumbs, then a "…" menu holding "View sources". Ghost icon buttons: no
   pill, no fill, muted ink, accent on hover. Section 7's chip language is for
   things the reader is meant to *choose*; these sit quietly under an answer
   and should not compete with it.

   Matched on a class *prefix*, not an exact key: the container key carries the
   turn's trace id, so every answer's row is its own widget. */
[class*="st-key-pa-actions"] {
  gap: 2px !important;
  /* The row has to read as belonging to the answer above it, and proximity is
     the whole signal — there is no line, label or shared surface tying them
     together. `.st-key-pa-scroll` puts a 14px gap between every element in the
     conversation, so a plain margin could only ever push them further apart;
     the negative margin claws that back to 5px. What sells it is the contrast
     with the space *below* the row (14px gap + the composer's own 10px), so
     the icons sit clearly nearer the card they act on than anything else. */
  margin-top: -9px !important;
  align-items: center !important;
}
[class*="st-key-pa-actions"] > div,
[class*="st-key-pa-actions"] [data-testid="stElementContainer"] {
  flex: 0 0 auto !important;
  width: auto !important;
  margin: 0 !important;
}
[class*="st-key-pa-actions"] button {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  color: var(--dw-text-ctx) !important;
  padding: 0 !important;
  width: 28px !important;
  min-width: 28px !important;
  height: 28px !important;
  min-height: 28px !important;
  border-radius: 8px !important;
  display: inline-flex !important;
  align-items: center;
  justify-content: center;
}
/* The ink-enforcement block hands every span inside a markdown container back
   to `color: inherit !important`, which is exactly what is wanted here — the
   icon follows the button's own colour through hover and the chosen state. */
[class*="st-key-pa-actions"] button :is(span, p, div) {
  color: inherit !important;
  line-height: 1 !important;
}
[class*="st-key-pa-actions"] button span[data-testid="stIconMaterial"] {
  font-size: 17px !important;
}
[class*="st-key-pa-actions"] button:hover {
  color: var(--dw-label) !important;
  background: var(--dw-accent-soft) !important;
}
/* Keyboard focus, given back.
   Streamlit draws its focus ring as a box-shadow, and the ghost-button rules
   above strip box-shadow with !important to remove the pill — which took the
   focus indicator with it, on the one row in the app that is nothing but icon
   buttons. Redrawn as an `outline` instead: nothing else in this sheet sets
   outline on a button, so it needs no !important and cannot be stripped by the
   same accident twice.
   `:focus-visible`, not `:focus` — the browser shows it for keyboard
   navigation and withholds it from a mouse click, which is what keeps this
   invisible in normal use and leaves the resting appearance untouched. It also
   sits correctly alongside the mousedown/preventDefault guard in
   scroll_to_question: that suppresses *mouse* focus only, so tab focus, and
   now its indicator, both survive. */
[class*="st-key-pa-actions"] button:focus-visible,
[class*="st-key-pa-sources"] button:focus-visible,
[data-testid="stPopoverBody"] button:focus-visible {
  outline: 2px solid var(--dw-teal);
  outline-offset: 2px;
}
/* The verdict once given: the chosen thumb holds the accent, so the row
   remembers the answer without spending a sentence saying so. */
[class*="st-key-pa-actions"] [data-testid="stBaseButton-primary"] {
  color: var(--dw-teal) !important;
  background: var(--dw-accent-soft) !important;
}

/* The "…" menu opens in a portal at the document root — outside
   .st-key-pa-screen and beyond the reach of any scoped rule — so these are
   global by necessity. Same reason the selectbox dropdown is handled in
   .streamlit/config.toml. */
/* Placement. `st.popover` takes no placement argument, and BaseWeb anchors the
   panel 3.6px *below* the trigger's bottom edge, left-aligned with it — which
   put the menu on top of the composer. It is repositioned here instead:

     - the body itself is stripped to a zero-height, transparent anchor point;
     - the panel is drawn on its child and pulled above the trigger.

   Collapsing the body to zero height is what makes the offset deterministic:
   BaseWeb measures the panel when it opens and flips it above the trigger when
   it would not fit below, and a zero-height box always fits — so it always
   anchors below, and the one offset here is always right. 36px = the 28px
   trigger + BaseWeb's 3.6px + a 4px gap. Keep it paired with the button height
   in section 9. */
[data-testid="stPopoverBody"] {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
  height: 0 !important;
  min-height: 0 !important;
  min-width: 0 !important;
  overflow: visible !important;
}
[data-testid="stPopoverBody"] > [data-testid="stVerticalBlock"] {
  position: absolute;
  bottom: 36px;
  left: 0;
  width: max-content;
  min-width: 152px;
  /* Streamlit gives this block `flex: 1 1 0%`, which resolves to a 14px height
     against the zero-height anchor above and clipped the menu to a sliver. */
  height: auto !important;
  background: var(--dw-glass-fill);
  border: 1px solid var(--dw-glass-border);
  border-radius: 12px;
  padding: 6px;
  gap: 0 !important;
  box-shadow: var(--pa-popover-shadow);
}
[data-testid="stPopoverBody"] button {
  background: transparent !important;
  border: 0 !important;
  color: var(--dw-text-ast) !important;
  font-family: var(--dw-sans) !important;
  font-size: 13px !important;
  font-weight: 400 !important;
  justify-content: flex-start !important;
  width: 100% !important;
  height: auto !important;
  min-height: 0 !important;
  padding: 7px 10px !important;
  border-radius: 8px !important;
  white-space: nowrap;
}
[data-testid="stPopoverBody"] button:hover {
  background: var(--dw-accent-soft) !important;
  color: var(--dw-label) !important;
}
[data-testid="stPopoverBody"] button :is(span, p, div) { color: inherit !important; }

/* Sources panel — opened from the menu, closed from its own control. It sits
   under the answer rather than inside the card: it is about the answer, not
   part of it. */
[class*="st-key-pa-sources"] {
  margin-top: 8px !important;
  padding: 10px 12px !important;
  border-radius: 12px;
  background: var(--dw-accent-soft);
  border: 1px solid var(--dw-glass-border);
  gap: .3rem !important;
}
.pa-src-title {
  font-size: 10.5px;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--dw-label);
  line-height: normal;
  margin-bottom: 7px;
}
.pa-src-item {
  display: flex;
  gap: 7px;
  font-size: 12.5px;
  line-height: 18px;
  color: var(--dw-text-ast);
}
.pa-src-item + .pa-src-item { margin-top: 5px; }
.pa-src-empty {
  font-size: 12.5px;
  line-height: 18px;
  color: var(--dw-text-ctx);
}
/* The panel's own close control: a quiet text action, not one of section 7's
   chips — it dismisses a panel rather than offering the reader a choice. */
[class*="st-key-pa-sources"] button {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
  color: var(--dw-label) !important;
  font-size: 11.5px !important;
  font-weight: 500 !important;
  width: auto !important;
  min-width: 0 !important;
  height: auto !important;
  min-height: 0 !important;
  padding: 6px 0 0 !important;
  border-radius: 6px !important;
}
[class*="st-key-pa-sources"] button:hover { color: var(--dw-teal) !important; }
[class*="st-key-pa-sources"] button :is(span, p, div) { color: inherit !important; }

/* Streamlit's `help=` tooltip portals to the document root and is painted from
   the native theme, which is set for the page — so inside the phone it arrived
   as white-on-white. Same class of leak as the selectbox and the popover. */
[data-testid="stTooltipContent"] {
  background: var(--dw-glass-fill) !important;
  border: 1px solid var(--dw-glass-border) !important;
  border-radius: 8px !important;
  color: var(--dw-text-ast) !important;
  font-family: var(--dw-sans) !important;
  font-size: 11.5px !important;
  padding: 5px 9px !important;
}
[data-testid="stTooltipContent"] :is(p, span, div) {
  color: var(--dw-text-ast) !important;
  font-size: 11.5px !important;
  margin: 0 !important;
}

/* Hover-only tooltips, enforced.

   A thumb click reruns the script, which replaces the button node under the
   cursor. The replacement mounts already-hovered and opens its tooltip, but the
   `mouseleave` that would close it was owed to the node that no longer exists —
   so the bubble outlived the pointer and sat on top of the answer card.

   `pa-tip-off` is the off switch; the listener in scroll_to_question() adds it
   whenever the pointer is not over a tooltip anchor and removes it when it is.
   Kept as a class on <body> rather than JS touching the portal, so React keeps
   ownership of its own node and nothing has to be restored. */
body.pa-tip-off [data-testid="stTooltipContent"] { display: none !important; }

/* Tighten Streamlit's default vertical rhythm inside the phone. */
.st-key-pa-screen [data-testid="stVerticalBlock"] { gap: .55rem; }
"""


# =====================================================================
# INTERNALS
# =====================================================================

#: Geometry the stylesheet needs as a value, not as a rule.
#:
#: Kept apart from the palettes because it is not a palette: it is the same in
#: every look, and a new palette must not have to restate it. Rendered as its
#: own token block so `_STATIC_CSS` stays free of Python interpolation — the
#: property that makes the sheet byte-identical for every palette and keeps the
#: braces in it unescaped.
_GEOMETRY: dict[str, str] = {
    "--pa-phone-width": f"{PHONE_WIDTH_PX}px",
}


def _token_block(tokens: dict[str, str], selector: str = ":root") -> str:
    """Render a palette dict as a CSS custom-property block."""
    body = "\n".join(f"  {name}: {value};" for name, value in tokens.items())
    return f"{selector} {{\n{body}\n}}"


def current_mode() -> str:
    """
    The palette in force. Normally whatever ACTIVE_LOOK names.

    With ACTIVE_LOOK = "toggle" it falls back to the old two-palette slider and
    reads its session key instead. Reading this before any widget renders is
    safe — it only touches session state.
    """
    if ACTIVE_LOOK != "toggle":
        return ACTIVE_LOOK
    return "dark" if st.session_state.get(MODE_KEY, True) else "light"


#: Barlow is the brand face; 400/500/600 are the three weights the Figma text
#: styles use (Label Regular, Label Medium / Title Medium, Title SemiBold).
#: An @import has to be the first statement in the sheet, ahead of the token
#: block — a rule before it makes the browser drop the import silently.
_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2"
    "?family=Barlow:wght@400;500;600;700&display=swap');"
)


def _stylesheet(mode: str) -> str:
    return "\n".join([
        _FONT_IMPORT,
        f"/* palette: {mode} */",
        _token_block(PALETTES[mode]),
        _token_block(_GEOMETRY),
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
    # A fixed ACTIVE_LOOK makes this control a lie — it would repaint nothing.
    # Returning early leaves the call sites untouched, so setting ACTIVE_LOOK
    # back to "toggle" brings the slider back with no other edit.
    if ACTIVE_LOOK != "toggle":
        return

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


# The gap the scroll helper leaves above the anchored question.
#
# 14px, matching `.st-key-pa-scroll`'s own `gap` — the rhythm between messages.
# That makes the boundary exact: the previous turn's bottom edge lands flush
# with the top of the visible area instead of a few pixels inside it, where a
# card's rounded border reads as a rendering artifact rather than as history
# continuing above. The scroll area's 24px `padding-top` was the other
# candidate; it left a 10px sliver of the previous card showing.
SCROLL_TOP_PAD_PX = 14


def status_steps(steps: list[tuple[str, str]]) -> None:
    """
    The honest loading state: one row per pipeline stage, the stage in flight
    carrying the rising water drops and the finished ones a mint tick.

    `steps` is `[(label, state)]` with state in {"done", "live", "pending"}.
    Render into an `st.empty()` placeholder and call again to advance — see
    `app.py::run_turn`. Call `placeholder.empty()` when the answer is ready;
    the status is never part of the transcript.

    Labels are escaped: this is the first helper here to take text that is not
    a module constant (FRONTEND.md bug B9).
    """
    rows = []
    for label, state in steps:
        if state == "done":
            # A div, not a span, and for the same reason `.pa-who` and `.pa-src`
            # are divs: the ink-enforcement block hands every span inside a
            # markdown container back to `color: inherit !important`, which
            # silently repainted the mint tick in the row's muted ink. `div` is
            # deliberately absent from that rule's `:is()` list — see section 5.
            glyph = '<div class="pa-tick">\u2713</div>'
        elif state == "live":
            glyph = '<span class="pa-drops"><i></i><i></i><i></i></span>'
        else:
            # Present but hidden, so the row keeps its width and the list does
            # not shift sideways when this stage becomes live.
            glyph = '<span class="pa-drops"><i></i><i></i><i></i></span>'
        rows.append(
            f'<div class="pa-step is-{state}">{glyph}'
            f'<span>{html.escape(label)}</span></div>'
        )
    st.markdown(f'<div class="pa-status">{"".join(rows)}</div>',
                unsafe_allow_html=True)


def status_pulse(label: str) -> None:
    """
    The loading state for the Fluidra look: one line, three pulsing dots, and
    the sentence for the stage the graph is actually in.

    `label` changes as the turn progresses, so the row is honest about which
    stage is running without spending three lines on it.
    """
    st.markdown(
        '<div class="pa-pulse">'
        '<span class="pa-drops"><i></i><i></i><i></i></span>'
        f'<span>{html.escape(label)}</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def status_stack(rows: list[tuple[str, bool]]) -> None:
    """
    The loading state as a growing stack: `rows` = [(label, is_live)].

    Only rows the pipeline has actually reached are passed in — nothing is
    pre-rendered as "pending", because until the planner answers we do not know
    what the stages will be, and inventing them is exactly the status theatre
    §8.2 rules out. The last row is the live one; the rest are dimmed with
    their drops frozen.
    """
    out = []
    for label, live in rows:
        cls = "pa-pulse" if live else "pa-pulse is-done"
        out.append(
            f'<div class="{cls}">'
            '<span class="pa-drops"><i></i><i></i><i></i></span>'
            f"<span>{html.escape(label)}</span>"
            "</div>"
        )
    st.markdown(f'<div class="pa-stack">{"".join(out)}</div>', unsafe_allow_html=True)


# ==========================================
# TYPEWRITER REVEAL
# ==========================================
# The answer arrives written, not pasted.
#
# demo-deep-water.html::typeOut is the reference: it appends one word every
# 34ms, which works out at roughly 165 characters a second. This reveals by
# character instead of by word — the same throughput, a finer grain — because
# a word-at-a-time reveal on the longer answers the live graph returns reads
# as text jumping rather than as text being written.
#
# Two rules from the spec that this must not break:
#
#   * Safety turns are never streamed. §DES-009: the template renders complete,
#     instantly, with no status theatre at all. Call st.markdown for those.
#   * Nothing here is a pipeline claim. The status stack above reports real
#     graph stages; this runs *after* the answer is already in hand and is
#     pure presentation, so it never implies the tokens are arriving live.
#
# No caret: the demo has none, and the demo is the visual oracle.
STREAM_ANSWERS = True

# 165 chars/s = the demo's 34ms-per-word, measured against its own answer copy.
_STREAM_CPS = 165.0
# One repaint per 20ms. Finer than the eye needs and coarse enough that a long
# answer costs a few hundred websocket deltas rather than a few thousand.
_STREAM_FRAME_S = 0.02
# Past ~6s the reveal stops being a cue that the assistant is answering and
# becomes a wait imposed on someone who can already read. The demo needs no cap
# because its canned answers are short; live ones are not.
_STREAM_MAX_S = 6.0

# Characters that may be the first half of an inline markdown marker.
_PARTIAL_MARKERS = "*_`~"


def _typing_safe(prefix: str) -> str:
    """
    Make a mid-word slice of markdown safe to render on its own.

    Cutting `**8 h/day**` at an arbitrary character leaves `**8 h/d`, and
    Streamlit renders that literally — so the reader watches asterisks appear
    and then vanish once the closer lands. Instead: drop a dangling marker
    character, then close whatever span the cut left open, so every frame is
    balanced markdown and the bold is already bold as it is typed.

    Only `**` and backticks are re-balanced, the two that appear in answer copy.
    A slice through a link's `](url` renders as literal text for one frame
    (~20ms) — visible in principle, not in practice, and worth less than the
    parser it would take to handle properly.
    """
    body = prefix
    while body and body[-1] in _PARTIAL_MARKERS:
        body = body[:-1]
    if body.count("**") % 2:
        body += "**"
    if body.count("`") % 2:
        body += "`"
    return body


def type_out(slot, text: str, *, cps: float = _STREAM_CPS) -> None:
    """
    Reveal `text` into `slot` a few characters at a time, then leave the real
    text in place.

    `slot` is an `st.empty()` created where the answer belongs — inside the
    assistant's `chat_message`, after `assistant_label()`. The function blocks
    for as long as the reveal lasts, exactly as the status stack above does.

    The final repaint writes `text` itself rather than the last balanced frame,
    so the transcript never keeps a synthesised closer.
    """
    if not text:
        return
    if not STREAM_ANSWERS:
        slot.markdown(text)
        return

    n = len(text)
    duration = min(n / cps, _STREAM_MAX_S)

    # The cursor is read off the clock, not counted in fixed steps.
    #
    # A fixed n-chars-per-frame loop assumes the only cost per frame is its own
    # sleep. It is not: each repaint is a websocket delta and a React re-render
    # of the whole markdown block, which measured at ~30ms against a 20ms frame
    # — so a 1041-character answer took 8.2s to reveal under a 6s cap. Deriving
    # the cursor from elapsed time makes the repaint cost self-correcting: slow
    # frames simply advance further, and `duration` holds either way.
    start = time.monotonic()
    shown = 0
    while shown < n:
        elapsed = time.monotonic() - start
        if elapsed >= duration:
            break
        # `shown + 1` floor: the reveal must move every frame, or a very long
        # answer early in its duration would repaint the same slice repeatedly.
        shown = max(shown + 1, int(n * elapsed / duration))
        slot.markdown(_typing_safe(text[:shown]))
        drift = _STREAM_FRAME_S - (time.monotonic() - start - elapsed)
        if drift > 0:
            time.sleep(drift)
    slot.markdown(text)


def scroll_to_question(token: str) -> None:
    """
    Put the last question near the top of the phone's scroll area, with the
    opening lines of its answer visible below it.

    Why this exists
    ---------------
    Streamlit leaves a height-constrained container scrolled to the bottom
    after a rerun, so a long answer landed with its *end* on screen and the
    reader had to scroll back up to find what they had asked. That was the
    most-cited complaint in the user-testing sessions. The prototypes are no
    better here — `scrollDown()` in demo-deep-water.html scrolls to
    `scrollHeight` — so this rule supersedes the demo (see the changelog entry
    in option-c-deep-water.html).

    Why JavaScript
    --------------
    Streamlit exposes no scroll API. `components.html` mounts a same-origin
    iframe, so `window.parent.document` is reachable; the CSS above collapses
    the iframe to zero height. This is the only JS in the front-end.

    Two jobs, one script
    --------------------
    `token` changes exactly once per turn. On a **new turn** the view is placed
    against the question. On **every other rerun** — a thumbs click, the sources
    panel, the reset button — the token is unchanged and the reader's own
    position is *restored* instead.

    That restore is not belt-and-braces. Clicking a button inside the scroll
    area gives it focus, and the browser scrolls a focused element into view;
    the action row sits at the very bottom of the conversation, so rating an
    answer threw the reader to the end of it. Doing nothing on a non-turn rerun
    was enough only while nothing inside the conversation was clickable.
    """
    # A per-run nonce, and it is load-bearing. Streamlit reuses a component
    # whose markup is byte-identical, so with only the token in here the script
    # never re-executed on a rerun that was not a new turn — which is precisely
    # the case the restore branch below exists to handle. The nonce forces a
    # remount every rerun; `token` still decides anchor-vs-restore.
    st.session_state["_pa_scroll_run"] = st.session_state.get("_pa_scroll_run", 0) + 1
    payload = json.dumps({
        "token": token,
        "pad": SCROLL_TOP_PAD_PX,
        "run": st.session_state["_pa_scroll_run"],
    })
    script = f"""<script>
(function () {{
  const CFG = {payload};
  const W = window.parent;
  const doc = W.document;

  // Streamlit puts overflow on the keyed container, on its border wrapper, or
  // on a child, depending on version — the same instability that makes
  // theme.py reset four selectors for the scroll area. So find the node that
  // actually scrolls instead of naming one.
  function findScroller() {{
    const keyed = doc.querySelector('.st-key-pa-scroll');
    if (!keyed) return null;
    const candidates = [keyed, ...keyed.querySelectorAll('*')];
    let node = keyed.parentElement;
    for (let i = 0; i < 4 && node; i++) {{ candidates.push(node); node = node.parentElement; }}
    return candidates.find(el => el.scrollHeight > el.clientHeight + 4) || null;
  }}

  function anchor() {{
    // role_marker() emits these; the last one is the question just asked.
    const marks = doc.querySelectorAll('[data-pa-role="user"]');
    const last = marks[marks.length - 1];
    return last ? (last.closest('[data-testid="stChatMessage"]') || last) : null;
  }}

  // Only *user gestures* update the remembered position. A plain scroll
  // listener would faithfully record the auto-scroll this is here to undo, and
  // then restore the reader to the bottom it just rescued them from.
  //
  // The "already watching" mark is a WeakSet on the parent window, NOT a data
  // attribute on the element. Writing `sc.dataset.*` mutates the scroll
  // container, and Streamlit re-pins a height-constrained container to the
  // bottom whenever its subtree mutates — so the bookkeeping was triggering
  // the exact auto-scroll this function exists to undo.
  W.__paWatched = W.__paWatched || new WeakSet();
  let userMoved = false;
  function watch(sc) {{
    if (!sc || W.__paWatched.has(sc)) return;
    W.__paWatched.add(sc);
    const rec = () => {{
      userMoved = true;
      requestAnimationFrame(() => {{ W.__paScrollTop = sc.scrollTop; }});
    }};
    ['wheel', 'touchmove'].forEach(ev => sc.addEventListener(ev, rec, {{passive: true}}));
    // Keyboard scrolling never reaches the two above, and the scrollbar is
    // hidden inside the phone, so this is the only other way to move.
    doc.addEventListener('keydown', rec, {{passive: true}});
  }}

  // Controls inside the conversation must not take focus from a mouse click.
  // A focused element gets pulled back into view by the browser, and the action
  // row sits at the very bottom of the conversation — so rating an answer threw
  // the reader to the end of it. Undoing that scroll afterwards cannot win:
  // each correction moves the focused button out of view again and Streamlit
  // relaunches its own smooth scroll to the bottom, which measured as a visible
  // oscillation lasting as long as the guard did.
  //
  // preventDefault on mousedown suppresses *mouse* focus only. Tab focus and
  // the click itself are untouched, so keyboard users lose nothing.
  W.__paNoFocus = W.__paNoFocus || new WeakSet();
  const noFocus = e => e.preventDefault();
  doc.querySelectorAll(
    '[class*="st-key-pa-actions"] button, [class*="st-key-pa-sources"] button'
  ).forEach(btn => {{
    if (W.__paNoFocus.has(btn)) return;
    W.__paNoFocus.add(btn);
    btn.addEventListener('mousedown', noFocus);
  }});

  // The guard lives on the parent window, which survives reruns; the iframe
  // itself is rebuilt every time.
  const newTurn = W.__paScrollToken !== CFG.token;
  W.__paScrollToken = CFG.token;
  W.__paScrollRun = CFG.run;   // diagnostic: which render this script came from

  function place() {{
    // The reader took over — stop re-asserting rather than fight them.
    if (userMoved) return;
    const sc = findScroller();
    if (!sc) return;
    watch(sc);
    const max = sc.scrollHeight - sc.clientHeight;

    if (newTurn) {{
      const bubble = anchor();
      if (!bubble) return;
      // Rects, not offsetTop: the offsetParent chain inside a Streamlit
      // container is not something to rely on.
      const delta = bubble.getBoundingClientRect().top
                  - sc.getBoundingClientRect().top
                  - CFG.pad;
      // Clamped, deliberately. A short answer simply stays where it lands
      // rather than scrolling into manufactured empty space — no spacer trick.
      sc.scrollTop = Math.max(0, Math.min(sc.scrollTop + delta, max));
    }} else if (typeof W.__paScrollTop === 'number') {{
      // Drop focus first, or this fights a loop it cannot win. The button that
      // caused the rerun still holds focus; the browser keeps pulling a focused
      // element back into view, and the action row is at the very bottom of the
      // conversation. Restoring the scroll moves the button out of view, which
      // re-triggers the pull, which we undo again — measured as a visible
      // oscillation for as long as the guard stayed alive. Blurring removes the
      // cause instead of contesting it.
      const active = doc.activeElement;
      if (active && active !== doc.body && sc.contains(active)) active.blur();
      sc.scrollTop = Math.max(0, Math.min(W.__paScrollTop, max));
    }} else {{
      return;
    }}
    W.__paScrollTop = sc.scrollTop;
  }}

  // Streamlit finishes painting the markdown after this iframe mounts and
  // re-asserts its own scroll-to-bottom, so a single write gets overwritten.
  // Re-place across ~450ms. No smooth behaviour: it loses that race.
  // Streamlit keeps re-pinning the container to the bottom as the answer, the
  // action row and the cleared status all land, which runs well past the first
  // paint — measured settling at ~600ms after the anchor is first written. The
  // window is held open to 1.2s and is safe to hold that long because the first
  // user gesture cancels it.
  // Holding the position has to *react* to Streamlit rather than out-wait it.
  // Streamlit re-pins a height-constrained container to the bottom as the
  // answer, the action row and the cleared status land, and the timing moves
  // from run to run — a fixed window won some reruns and lost others.
  //
  // So: correct every frame for the window, AND listen for scrolls during it.
  // Any scroll we did not cause is undone on the spot, which converges because
  // `place()` is idempotent — writing a scrollTop that is already correct fires
  // no further scroll event. The first user gesture cancels the whole thing.
  // The guard is held until the reader moves or asks something new — NOT for a
  // fixed window. Measured: after a feedback click the container is re-pinned
  // to the bottom about 1.6s later, past any window worth polling for, and the
  // delay moves between runs. So the rule is stated as intent rather than as a
  // duration: hold this position until the reader takes over.
  //
  // `userMoved` ends it, the next render replaces it (each run tears the
  // previous one down through W.__paGuardOff), and it is a scroll listener
  // rather than a loop, so an untouched page costs nothing.
  if (W.__paGuardOff) {{ try {{ W.__paGuardOff(); }} catch (e) {{}} }}
  const sc0 = findScroller();
  const until = performance.now() + 2000;
  const guard = () => {{ if (!userMoved && performance.now() < until) place(); }};
  if (sc0) sc0.addEventListener('scroll', guard, {{passive: true}});
  W.__paGuardOff = () => {{ if (sc0) sc0.removeEventListener('scroll', guard); }};

  // A short frame loop as well, to settle the position while the answer, the
  // action row and the cleared status are still landing — correcting on every
  // frame keeps the initial overshoot to one frame, so it reads as no movement.
  const deadline = performance.now() + 1200;
  (function tick() {{
    place();
    if (!userMoved && performance.now() < deadline) requestAnimationFrame(tick);
  }})();

  // ---------------------------------------------------------------------
  // Hover-only tooltips (see the `pa-tip-off` rule in the stylesheet).
  //
  // Registered once per session, not once per rerun: every rerun mounts a
  // fresh iframe and would otherwise stack another listener on the parent
  // document. The tooltip anchors themselves are replaced on every rerun, so
  // the listener is delegated to the document and reads the anchor from the
  // event — never from a captured node.
  //
  // The pointer position is the whole state. If it is not over an anchor the
  // class is on and no bubble can show, whatever React thinks its hover state
  // is; move onto an anchor and the class comes off in the same event.
  // `mouseleave` on the document covers the pointer leaving the window
  // entirely, which fires no mousemove.
  // ---------------------------------------------------------------------
  if (!W.__paTipGuard) {{
    W.__paTipGuard = true;
    const ANCHOR = '[data-testid="stTooltipHoverTarget"]';
    const sync = (e) => {{
      const over = e && e.target && e.target.closest && e.target.closest(ANCHOR);
      doc.body.classList.toggle('pa-tip-off', !over);
    }};
    doc.addEventListener('mousemove', sync, {{passive: true}});
    doc.addEventListener('mouseleave', () => doc.body.classList.add('pa-tip-off'));
    // Scrolling moves the surface under a stationary pointer, which detaches
    // the bubble from what it was describing without emitting a mousemove.
    doc.addEventListener('scroll', () => doc.body.classList.add('pa-tip-off'),
                         {{passive: true, capture: true}});
  }}
}})();
</script>"""

    # `st.iframe` is the supported API and carries the same contract we depend on
    # — "embedded as-is in an iframe that allows JavaScript execution and
    # same-origin access to the Streamlit app". `st.components.v1.html` does the
    # same thing but is deprecated, and its stated removal date has already
    # passed, so it will disappear inside the `<2.0` range pyproject allows.
    #
    # Feature-detected rather than switched outright: `st.iframe` is newer than
    # the `>=1.57` floor, and the lockfile still resolves 1.60.0. Drop the
    # fallback once the floor moves past whichever release added `st.iframe`.
    #
    # height=1, not 0: `st.iframe` rejects 0 outright ("Height must be either a
    # positive integer, 'stretch', or 'content'"), and 'content' would measure
    # the script's own box. The CSS above takes it the rest of the way to zero.
    if hasattr(st, "iframe"):
        st.iframe(script, height=1)
    else:
        components.html(script, height=1)


def citation(text: str) -> None:
    """
    Dashed-underline source chip closing an answer grounded in a document,
    e.g. "Manual · §5 Maintenance". Call last, inside the same
    `st.chat_message("assistant")` block, only when the answer actually cites
    a source.
    """
    st.markdown(f'<div class="pa-src">{text}</div>', unsafe_allow_html=True)