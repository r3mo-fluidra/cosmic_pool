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
bezel, screen, bubbles, composer, buttons. `.streamlit/config.toml` still
defines both palettes as native themes so Streamlit-owned chrome that we cannot
reach (menu popovers, toasts, tooltips) stays in the right family.

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
    "--pa-track": "rgba(13,58,68,.08)",
    # .54 rather than .20, to clear the 3:1 UI-component threshold.
    "--pa-track-border": "rgba(13,58,68,.54)",
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
    "--pa-track": "rgba(13,58,68,.08)",
    # .54 rather than .20, to clear the 3:1 UI-component threshold.
    "--pa-track-border": "rgba(13,58,68,.54)",
}

PALETTES = {"dark": DEEP_WATER, "light": SUNLIT_LAGOON}

#: Serif italic label above each assistant turn, per the prototypes.
ASSISTANT_LABEL = "Your assistant"

#: Session-state key holding the slider position.
MODE_KEY = "pa_dark_mode"

#: Phone geometry, from the prototypes' `.phone` / `.screen` rules.
PHONE_WIDTH_PX = 392
SCREEN_HEIGHT_PX = 520


# =====================================================================
# STYLESHEET
# =====================================================================
# Deliberately free of Python interpolation: every colour is a var() lookup,
# so the rules below are identical for both palettes and there are no braces
# to escape.

_STATIC_CSS = """
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
  margin: 6px auto 20px;
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
.st-key-pa-themebar { padding-bottom: 2px; }
.pa-mode-label {
  font-size: 11.5px;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--pa-page-mut);
  white-space: nowrap;
}

/* The native toggle's own visuals are hidden and the track + knob are drawn
   as pseudo-elements on the label, which stays the click target. Two
   selectors because the testid may sit on the wrapper or on the label. */
.st-key-pa-themebar [data-testid="stCheckbox"] label,
.st-key-pa-themebar label[data-testid="stCheckbox"] {
  position: relative;
  display: inline-block;
  width: 54px;
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
  box-shadow: 0 1px 5px rgba(0,0,0,.3);
  transition: transform .25s ease;
}
/* Knob right = dark mode. */
.st-key-pa-themebar [data-testid="stCheckbox"] label:has(input:checked)::after,
.st-key-pa-themebar label[data-testid="stCheckbox"]:has(input:checked)::after {
  transform: translateX(28px);
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
  border: 1px solid var(--pa-bezel-rim);
  box-shadow: var(--pa-bezel-shadow);
}
.st-key-pa-screen {
  border-radius: 30px;
  overflow: hidden;
  background: var(--dw-surface);
  color: var(--dw-text);
  padding: 16px 14px 12px;
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
/* The only thing separating the header from the first turn, so it carries
   the whole gap. */
.st-key-pa-scroll { padding-top: 22px; }
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
}
.pa-ctx {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11.5px;
  color: var(--dw-text-ctx);
  margin: 3px 0 0;
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
  padding: 12px 13px;
  max-width: 80%;
  color: var(--dw-text-ast);
}

/* --- user: teal bubble, right aligned, bottom-right tail (16/16/4/16) */
[data-testid="stChatMessage"]:has([data-pa-role="user"]),
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: var(--dw-bubble-fill);
  border: 1px solid var(--dw-bubble-border);
  border-radius: var(--dw-radius) var(--dw-radius) 4px var(--dw-radius);
  color: var(--dw-bubble-text);
  padding: 9px 13px;
  width: fit-content;
  max-width: 80%;
  margin-left: auto;
}

[data-testid="stChatMessage"] p {
  font-size: 14px;
  line-height: 1.55;
}
[data-testid="stChatMessage"] p:last-child { margin-bottom: 0; }

.pa-who {
  font-family: var(--dw-serif);
  font-style: italic;
  font-size: 12.5px;
  color: var(--dw-teal);
  margin-bottom: 5px;
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
  /* The prototypes' well: an 8px surround on a 35px row, 53px overall.
     Streamlit's own padding is taller and left the pill looking slack. */
  padding: 8px !important;
  margin-top: 8px;
  overflow: hidden;
}
/* Centre the row, so the placeholder and the send disc share a midline. */
.st-key-pa-screen [data-testid="stChatInput"] > div,
.st-key-pa-screen [data-testid="stChatInput"] form {
  align-items: center !important;
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
/* One line, the same height as the send disc beside it. Streamlit sizes the
   textarea from JS with an inline style and grows it as you type, which is
   what made the composer tall — hence !important on all three of min/max/
   height. Long input scrolls inside the pill, as it does in the prototypes,
   where the composer is a single-line <input>. */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"] {
  background: transparent !important;
  color: var(--dw-text) !important;
  font-size: 13.5px !important;
  -webkit-text-fill-color: var(--dw-text);
  box-sizing: border-box !important;
  height: 35px !important;
  min-height: 35px !important;
  max-height: 35px !important;
  line-height: 19px !important;
  padding: 8px !important;   /* text sits 16px in from the pill's edge */
  overflow-y: auto;
}
[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInputTextArea"]::placeholder {
  color: var(--dw-text-ctx) !important;
  -webkit-text-fill-color: var(--dw-text-ctx);
  opacity: 1;
}

/* Send control: teal disc, 34px, arrow — never a square.
   Streamlit's own glyph is an icon-font ligature that renders as a filled box
   when the Material Symbols face is unavailable, so the button's children are
   hidden and the arrow is drawn as text in the page font instead. */
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
  display: flex !important;
  align-items: center;
  justify-content: center;
}
[data-testid="stChatInputSubmitButton"] > * { display: none !important; }
[data-testid="stChatInputSubmitButton"]::after {
  content: "\2191";                      /* ↑ */
  font-family: var(--dw-sans);
  font-size: 16px;
  font-weight: 800;
  line-height: 1;
  color: var(--dw-on-teal);
}
[data-testid="stChatInputSubmitButton"]:hover {
  background: var(--dw-teal) !important;
  opacity: .88;
}
[data-testid="stChatInputSubmitButton"]:disabled { opacity: .45; }

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
[data-testid="stExpander"] summary {
  font-size: 12px;
  color: var(--dw-text-ctx);
}
[data-testid="stExpander"] summary:hover { color: var(--dw-teal); }

[data-testid="stAlert"],
[data-testid="stAlertContainer"],
.stAlert {
  border-radius: var(--dw-radius);
  border: 1px solid var(--dw-glass-border);
  background: var(--dw-glass-fill);
  color: var(--dw-text-ast);
  font-size: 12.5px;
}
[data-testid="stAlert"] a,
[data-testid="stAlertContainer"] a { color: var(--dw-teal); }

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
  color: var(--dw-text-ctx);
  font-size: 11.5px;
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