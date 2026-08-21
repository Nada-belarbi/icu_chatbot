"""Central stylesheet for the Streamlit presentation layer.

Every visual constant lives here so the interface identity (colors, spacing,
radii) can be tuned without touching any state or business logic in ``ui.py``.
Palette and layout follow the product design specification exactly.
"""

from __future__ import annotations

import streamlit as st

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --navy: #123B5D;
  --blue: #2878A5;
  --teal: #29B6A6;
  --teal-hover: #24A696;
  --teal-light: #E8F7F5;
  --blue-light: #EAF4FA;
  --canvas: #F8FAFC;
  --white: #FFFFFF;
  --ink: #18364A;
  --muted: #6B7E8C;
  --border: #E2EBF0;
  --warn-bg: #FFF5E8;
  --warn-border: #F7DEC0;
  --warn-accent: #E89B49;
  --danger-bg: #FFF1F0;
  --danger-border: #D9534F;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
  background: var(--canvas);
  color: var(--ink);
  font-family: 'Inter', 'Manrope', -apple-system, sans-serif;
}
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; height: 0; }
[data-testid="stToolbar"] { display: none; }
.main .block-container { max-width: 900px; padding: 18px 32px 190px; }

/* ---------------------------------------------------------------- */
/* Sidebar                                                           */
/* ---------------------------------------------------------------- */
[data-testid="stSidebar"] { min-width: 280px; max-width: 280px; }
[data-testid="stSidebar"] > div:first-child {
  /* The dark brand zone has a fixed content height.  A percentage-based
     transition cut through the first conversation on shorter screens. */
  background: linear-gradient(180deg, #123B5D 0, #17546D 184px, #F5FAFC 184px, #F5FAFC 100%);
  padding-top: 0;
}
.app-header + div { min-width: 0; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.35rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin-bottom: 0; }

.brand-block { display:flex; align-items:center; gap:10px; padding:20px 18px 16px; }
.brand-text { line-height:1.25; }
.brand-title { color:#FFFFFF; font-size:14px; font-weight:600; }
.brand-sub { color:#BFEAE4; font-size:11px; letter-spacing:1.5px; font-weight:600; }

[data-testid="stSidebar"] .stButton > button {
  background: transparent; color: var(--navy); border: 1px solid transparent;
  border-radius: 10px; font-size: 13.5px; font-weight: 500; text-align: left;
  min-height: 38px; padding: 8px 12px; transition: background 150ms ease;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: var(--blue-light); border-color: transparent; color: var(--navy);
}
[data-testid="stSidebar"] .stButton > button p { color: inherit; font-size: 13.5px; }
[data-testid="stSidebar"] [data-testid^="stButton-conversation_"] > button {
  display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  width: 100%; margin: 0 10px; width: calc(100% - 20px);
}
[data-testid="stSidebar"] [data-testid^="stButton-conversation_"] > button p {
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; text-align: left;
}

/* New conversation CTA */
[data-testid="stSidebar"] [data-testid="stButton-new_conversation"] > button {
  background: var(--teal) !important; color: #FFFFFF !important; border: 0 !important;
  border-radius: 12px !important; font-weight: 600 !important; min-height: 48px !important;
  justify-content: center !important; margin: 4px 0 14px !important;
}
[data-testid="stSidebar"] [data-testid="stButton-new_conversation"] > button:hover {
  background: var(--teal-hover) !important;
}
[data-testid="stSidebar"] [data-testid="stButton-new_conversation"] > button p {
  color: #FFFFFF !important; font-size: 14px !important;
}

.sidebar-section-label {
  display: block; box-sizing: border-box; min-height: 32px;
  font-size: 11px; letter-spacing: 1.1px; font-weight: 700; color: #7C93A6;
  margin: 0; padding: 13px 16px 8px; text-transform: uppercase;
}
.conversation-date { font-size: 11px; color: #8798A5; margin: -5px 0 8px 23px; }

[data-testid="stSidebar"] details {
  margin: 8px 10px 4px; border-top: 1px solid #D9E6EC; padding-top: 8px;
}
[data-testid="stSidebar"] summary { color: #6B7E8C; cursor: pointer; font-size: 12px; }

[data-testid="stSidebar"] [data-testid="stPopover"] > button {
  background: transparent !important; border: 0 !important; color: #8798A5 !important;
  min-height: 30px !important; padding: 2px 6px !important; font-size: 13px !important;
}

.sidebar-footer {
  margin-top: 10px; padding: 12px 16px; border-top: 1px solid #D9E6EC;
  font-size: 11.5px; color: var(--muted); line-height: 1.5;
}
.sidebar-footer-item { display:flex; align-items:center; gap:7px; padding: 3px 0; }

/* ---------------------------------------------------------------- */
/* Header                                                             */
/* ---------------------------------------------------------------- */
.app-header {
  display:flex; align-items:center; gap:14px; min-height:72px;
  border-bottom: 1px solid #EEF2F5; margin: 0 0 4px; padding: 10px 0;
  background: var(--white); position: relative; overflow: hidden;
}
.app-header::after {
  content:""; position:absolute; right:-40px; top:-60px; width:220px; height:220px;
  border-radius:50%; background: radial-gradient(circle, rgba(41,182,166,0.10), transparent 70%);
  pointer-events:none;
}
.app-header-mark { width:40px; height:40px; display:grid; place-items:center;
  background: var(--teal-light); border-radius: 12px; flex-shrink:0; }
.app-header-mark svg circle { fill: transparent; }
.app-header-title { margin:0; color: var(--navy); font-size:19px; font-weight:700; }
.app-header-subtitle { margin:1px 0 0; color: var(--muted); font-size:12.5px; }

/* ---------------------------------------------------------------- */
/* Disclaimer                                                        */
/* ---------------------------------------------------------------- */
.disclaimer-card {
  display:flex; gap:10px; align-items:flex-start; background: var(--warn-bg);
  border: 1px solid var(--warn-border); border-radius: 12px; padding: 12px 16px;
  margin: 16px 0 6px; font-size: 13px; line-height:1.55; color:#6B4A1E;
}
.disclaimer-card svg { flex-shrink:0; margin-top:2px; }

/* ---------------------------------------------------------------- */
/* Empty state                                                       */
/* ---------------------------------------------------------------- */
.empty-state { text-align:center; padding: 46px 10px 8px; }
.empty-state h2 { color: var(--navy); font-size:22px; font-weight:700; margin:0 0 8px; }
.empty-state p { color: var(--muted); font-size:14px; margin:0; }
.suggestion-title { color: var(--navy); font-weight:600; font-size:13.5px; margin-bottom:3px; }
.suggestion-sub { color: var(--muted); font-size:12.5px; }
[data-testid^="stButton-suggestion_"] > button {
  background: var(--white); border: 1px solid var(--border); border-radius:14px;
  padding: 14px 16px; text-align:left; min-height: 74px; white-space: normal;
}
[data-testid^="stButton-suggestion_"] > button:hover {
  border-color: var(--teal); background: #F7FCFB;
}
[data-testid^="stButton-suggestion_"] > button p { white-space: pre-line; color: var(--ink); font-size:13.5px; line-height:1.5; }

/* ---------------------------------------------------------------- */
/* Messages                                                           */
/* ---------------------------------------------------------------- */
.msg-row { display:flex; margin: 14px 0; align-items:flex-start; gap:10px; }
.msg-row-user { justify-content:flex-end; }
.msg-row-assistant { justify-content:flex-start; }
.bubble-col { display:flex; flex-direction:column; max-width:80%; }
.bubble { font-size:15px; line-height:1.65; padding: 12px 16px; }
.bubble p { margin: 0 0 8px; }
.bubble p:last-child { margin-bottom:0; }
.bubble ul { margin: 4px 0 8px 18px; padding:0; }
.bubble-user {
  background: var(--blue-light); color: var(--ink); border-radius: 16px 16px 4px 16px;
  max-width:70%; margin-left:auto;
}
.bubble-assistant {
  background: var(--white); border: 1px solid #E5EEF2; color: var(--ink);
  border-radius: 4px 16px 16px 16px;
}
.msg-avatar { flex-shrink:0; margin-top:2px; }
.msg-meta { font-size:11px; color:#98A9B5; margin-top:4px; padding: 0 4px; }
.msg-meta-user { text-align:right; }

.card-neutral {
  display:flex; gap:10px; align-items:flex-start; background: #F3F7FA; border: 1px solid #DCE7EC;
  border-radius: 4px 16px 16px 16px; padding: 13px 16px; font-size:14.5px; line-height:1.6; color: var(--ink);
}
.card-emergency {
  display:flex; gap:10px; align-items:flex-start; background: var(--danger-bg);
  border-left: 4px solid var(--danger-border); border-radius: 4px 12px 12px 4px;
  padding: 13px 16px; font-size:14.5px; line-height:1.6; color:#7A2E2A;
}

/* Sources */
.sources-block { margin-top:10px; }
.sources-label { font-size:12px; font-weight:600; color:#7C93A6; margin-bottom:6px; }
.source-card {
  display:inline-flex; align-items:center; gap:7px; background:#F8FBFC; border:1px solid #E1EAEE;
  border-radius:10px; padding:7px 11px; margin:0 6px 6px 0; font-size:13px; color:#39515F;
}
.source-tag {
  background: var(--blue); color:#FFFFFF; border-radius:5px; font-weight:700;
  font-size:10.5px; padding:2px 6px; letter-spacing:.3px;
}
.sources-more { margin-top:2px; }
.sources-more summary {
  cursor:pointer; font-size:12px; color: var(--blue); font-weight:600; list-style:none;
}
.sources-more summary::-webkit-details-marker { display:none; }
.sources-more-list { margin-top:6px; }

/* ---------------------------------------------------------------- */
/* Chat input                                                         */
/* ---------------------------------------------------------------- */
[data-testid="stChatInput"] {
  max-width: 900px; margin: 0 auto; background: linear-gradient(transparent, var(--canvas) 45%);
  padding-top: 18px;
}
[data-testid="stChatInput"] textarea {
  background: var(--white) !important; border: 1px solid #DCE7EC !important;
  border-radius: 16px !important; color: var(--ink) !important;
  box-shadow: 0 5px 25px rgba(30,70,90,0.08);
}
.privacy-note {
  position: fixed; bottom: 116px; left: 50%; transform: translateX(-50%);
  display:flex; align-items:center; gap:6px; font-size:11.5px; color: var(--muted);
  background: rgba(248,250,252,0.96); padding: 5px 10px; border-radius:8px;
  border: 1px solid rgba(226,235,240,0.75); z-index:999; white-space: nowrap;
}

/* ---------------------------------------------------------------- */
/* Authentication                                                     */
/* ---------------------------------------------------------------- */
.auth-brand { text-align:center; margin-bottom:6px; }
.auth-brand h1 { color: var(--navy); font-size:22px; font-weight:700; margin:12px 0 2px; }
.auth-brand p { color: var(--muted); margin:0; font-size:13.5px; }
[data-testid="stForm"] { border: none; padding: 0; }
.stButton > button[kind="primary"] { background: var(--teal); border-radius: 12px; font-weight:600; }
.stButton > button[kind="primary"]:hover { background: var(--teal-hover); }
.stButton > button[kind="secondary"] { border-radius: 12px; border-color: var(--blue); color: var(--blue); }
</style>
"""


def apply_styles() -> None:
    """Inject the single, centralised stylesheet once per rerun."""
    st.markdown(APP_CSS, unsafe_allow_html=True)

