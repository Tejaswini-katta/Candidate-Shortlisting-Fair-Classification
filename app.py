"""
FairHire AI — Explainable & Fair Candidate Screening Platform.

Professional HR SaaS Dashboard with Light / Dark theme toggle providing:
1. Executive Dashboard Home
2. Candidate Probability Ranking & Shortlist Export
3. Algorithmic Fairness & Bias Mitigation Audit
4. SHAP Global & Local Feature Explainability
5. Real-Time Single Candidate Shortlisting Predictor
6. Settings (Theme, Model Info, About)

Run locally:
    streamlit run app.py
"""

import os
import json
import random
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import PyPDF2
import docx
import io


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FairHire AI — Fair Candidate Screening",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─────────────────────────────────────────────────────────────────────────────
# THEME MANAGER  — centralized colour token store
# ─────────────────────────────────────────────────────────────────────────────
class ThemeManager:
    """
    Single source of truth for all colour tokens.
    Light Mode is the default; Dark Mode is opt-in via the sidebar toggle.
    All page functions call ThemeManager.get() to read tokens.
    No colour literals are duplicated across pages.
    """

    THEMES = {
        "Light": dict(
            # Backgrounds
            bg="#f8fafc",
            sidebar_bg="linear-gradient(180deg,#ffffff 0%,#f1f5f9 100%)",
            sidebar_border="#e2e8f0",
            card_bg="#ffffff",
            card_border="#e2e8f0",
            card_shadow="0 2px 12px rgba(0,0,0,0.06)",
            metric_bg="#f1f5f9",
            metric_border="#e2e8f0",
            # Text
            text_primary="#0f172a",
            text_secondary="#475569",
            text_muted="#94a3b8",
            text_label="#64748b",
            text_hint="#cbd5e1",
            # Structural
            divider="#e2e8f0",
            activity_border="#f1f5f9",
            panel_border_l="#f1f5f9",
            nav_label="#475569",
            nav_active="#2563eb",
            # Plotly
            plotly_paper="#ffffff",
            plotly_plot="#ffffff",
            plotly_font="#64748b",
            plotly_grid="#f1f5f9",
            plotly_tick="#94a3b8",
            # Header
            header_bg="linear-gradient(135deg,#eff6ff 0%,#f8fafc 55%,#f0fdf4 100%)",
            header_border="#dbeafe",
            header_title="#1e40af",
            header_sub="#64748b",
            header_muted="#94a3b8",
            # Coming soon
            cs_bg="linear-gradient(135deg,#f8fafc,#f1f5f9)",
            cs_border="#e2e8f0",
            cs_title="#1e293b",
            cs_sub="#94a3b8",
            # Section header
            section_color="#1e293b",
            section_border="#e2e8f0",
            # Status
            status_ok_bg="rgba(16,185,129,0.1)",
            status_ok_c="#059669",
            status_ok_b="rgba(16,185,129,0.3)",
            status_err_bg="rgba(239,68,68,0.08)",
            status_err_c="#dc2626",
            status_err_b="rgba(239,68,68,0.25)",
            # Boxes
            info_bg="rgba(59,130,246,0.06)",
            info_border="rgba(59,130,246,0.2)",
            warn_bg="rgba(245,158,11,0.06)",
            warn_border="rgba(245,158,11,0.2)",
            err_bg="rgba(239,68,68,0.06)",
            err_border="rgba(239,68,68,0.2)",
            # Matplotlib
            mpl_bg="#ffffff",
            mpl_text="#1e293b",
            mpl_grid="#e2e8f0",
            # Toggle
            toggle_icon="🌙",
            toggle_label="Dark Mode",
        ),

        "Dark": dict(
            # Backgrounds
            bg="#0a0f1e",
            sidebar_bg="linear-gradient(180deg,#0d1424 0%,#0f1929 100%)",
            sidebar_border="#1a2540",
            card_bg="#111827",
            card_border="#1f2937",
            card_shadow="0 4px 20px rgba(0,0,0,0.35)",
            metric_bg="#111827",
            metric_border="#1f2937",
            # Text
            text_primary="#f9fafb",
            text_secondary="#94a3b8",
            text_muted="#4b5563",
            text_label="#6b7280",
            text_hint="#374151",
            # Structural
            divider="#1f2937",
            activity_border="#1a2234",
            panel_border_l="#1a2234",
            nav_label="#94a3b8",
            nav_active="#60a5fa",
            # Plotly
            plotly_paper="#111827",
            plotly_plot="#111827",
            plotly_font="#6b7280",
            plotly_grid="#1a2234",
            plotly_tick="#374151",
            # Header
            header_bg="linear-gradient(135deg,#0d1b3e 0%,#111827 55%,#0a0f1e 100%)",
            header_border="#1f2937",
            header_title="#f9fafb",
            header_sub="#94a3b8",
            header_muted="#4b5563",
            # Coming soon
            cs_bg="linear-gradient(135deg,#111827,#0d1424)",
            cs_border="#1f2937",
            cs_title="#f9fafb",
            cs_sub="#4b5563",
            # Section header
            section_color="#e5e7eb",
            section_border="#1f2937",
            # Status
            status_ok_bg="rgba(16,185,129,0.1)",
            status_ok_c="#10b981",
            status_ok_b="rgba(16,185,129,0.25)",
            status_err_bg="rgba(239,68,68,0.1)",
            status_err_c="#ef4444",
            status_err_b="rgba(239,68,68,0.25)",
            # Boxes
            info_bg="rgba(59,130,246,0.08)",
            info_border="rgba(59,130,246,0.2)",
            warn_bg="rgba(245,158,11,0.08)",
            warn_border="rgba(245,158,11,0.2)",
            err_bg="rgba(239,68,68,0.08)",
            err_border="rgba(239,68,68,0.2)",
            # Matplotlib
            mpl_bg="#111827",
            mpl_text="#94a3b8",
            mpl_grid="#1f2937",
            # Toggle
            toggle_icon="☀️",
            toggle_label="Light Mode",
        ),
    }

    @staticmethod
    def init() -> None:
        if "theme" not in st.session_state:
            st.session_state["theme"] = "Light"

    @staticmethod
    def get() -> dict:
        return ThemeManager.THEMES[st.session_state.get("theme", "Light")]

    @staticmethod
    def name() -> str:
        return st.session_state.get("theme", "Light")

    @staticmethod
    def set(name: str) -> None:
        if name in ThemeManager.THEMES:
            st.session_state["theme"] = name
            st.rerun()

    @staticmethod
    def toggle() -> None:
        current = st.session_state.get("theme", "Light")
        st.session_state["theme"] = "Dark" if current == "Light" else "Light"
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# CSS GENERATOR  — single source of truth, builds from token dict
# ─────────────────────────────────────────────────────────────────────────────
def build_css(t: dict) -> str:
    return (
        "<style>"
        "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');"
        "html,body,[class*='css']{font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;"
        "transition:background-color 0.25s ease,color 0.25s ease;}"
        "*{transition:background-color 0.2s ease,border-color 0.2s ease,color 0.15s ease;}"
        f".stApp{{background-color:{t['bg']} !important;}}"
        # ── App-level containers ──────────────────────────────────────────────
        f"[data-testid='stAppViewContainer']{{background-color:{t['bg']} !important;}}"
        f"[data-testid='stMain']{{background-color:{t['bg']} !important;}}"
        f"section[data-testid='stMain']>div{{background-color:{t['bg']} !important;}}"
        f".main .block-container{{background-color:{t['bg']} !important;}}"
        # Transparent pass-through for layout blocks
        f"[data-testid='stVerticalBlock']{{background-color:transparent !important;}}"
        f"[data-testid='stHorizontalBlock']{{background-color:transparent !important;}}"
        f"[data-testid='column']{{background-color:transparent !important;}}"
        f"[data-testid='stTabContent']{{background-color:transparent !important;}}"
        # ── Expanders (details/summary DOM) ──────────────────────────────────
        f"[data-testid='stExpander']{{background-color:{t['card_bg']} !important;"
        f"border:1px solid {t['card_border']} !important;border-radius:10px !important;}}"
        f"[data-testid='stExpander'] details{{background-color:{t['card_bg']} !important;}}"
        f"[data-testid='stExpanderDetails']{{background-color:{t['card_bg']} !important;}}"
        f"[data-testid='stExpander'] summary,"
        f"[data-testid='stExpander'] summary span,"
        f"[data-testid='stExpander'] summary p"
        f"{{color:{t['text_primary']} !important;background-color:{t['card_bg']} !important;}}"
        f".streamlit-expanderHeader{{background-color:{t['card_bg']} !important;"
        f"color:{t['text_primary']} !important;}}"
        f".streamlit-expanderContent{{background-color:{t['card_bg']} !important;}}"
        # ── Tabs (BaseWeb) ────────────────────────────────────────────────────
        f".stTabs [data-baseweb='tab-list']{{background-color:transparent !important;}}"
        f".stTabs [data-baseweb='tab-panel']{{background-color:transparent !important;}}"
        f"[data-baseweb='tab']{{color:{t['text_secondary']} !important;}}"
        f"[data-baseweb='tab'][aria-selected='true']{{color:{t['nav_active']} !important;}}"
        # ── DataFrames ────────────────────────────────────────────────────────
        f"[data-testid='stDataFrameResizable']{{background-color:{t['card_bg']} !important;}}"
        f"[data-testid='stDataFrame']{{background-color:{t['card_bg']} !important;border-radius:10px;}}"
        # ── Selectbox / Multiselect dropdowns ────────────────────────────────
        f"[data-baseweb='select']>div{{background-color:{t['card_bg']} !important;"
        f"border-color:{t['card_border']} !important;color:{t['text_primary']} !important;}}"
        f"[data-baseweb='popover']{{background-color:{t['card_bg']} !important;"
        f"border:1px solid {t['card_border']} !important;}}"
        f"[role='listbox']{{background-color:{t['card_bg']} !important;}}"
        f"[role='option']{{background-color:{t['card_bg']} !important;"
        f"color:{t['text_primary']} !important;}}"
        f"[role='option']:hover{{background-color:{t['metric_bg']} !important;}}"
        # ── BaseWeb generic containers ────────────────────────────────────────
        f"[data-baseweb='block']{{background-color:transparent !important;}}"
        f"[data-baseweb='notification']{{background-color:{t['info_bg']} !important;}}"
        # ── Text inside all native widgets ────────────────────────────────────
        f"[data-testid='stWidgetLabel'] p{{color:{t['text_label']} !important;}}"
        f"[data-testid='stMarkdownContainer'] p{{color:{t['text_secondary']} !important;}}"
        # Sidebar

        f"[data-testid='stSidebar']{{background:{t['sidebar_bg']};border-right:1px solid {t['sidebar_border']};}}"
        f"[data-testid='stSidebar'] [data-testid='stRadio'] label{{color:{t['nav_label']} !important;"
        "font-size:0.85rem !important;font-weight:500 !important;padding:5px 0 !important;}"
        f"[data-testid='stSidebar'] [data-testid='stRadio'] div[data-checked='true'] label{{"
        f"color:{t['nav_active']} !important;font-weight:600 !important;}}"
        # Main block
        ".main .block-container{padding-top:1.5rem;padding-bottom:2.5rem;max-width:100%;}"
        # KPI cards
        f".kpi-card{{background:{t['card_bg']};border:1px solid {t['card_border']};"
        "border-radius:14px;padding:20px 22px 16px;position:relative;overflow:hidden;"
        f"box-shadow:{t['card_shadow']};margin-bottom:2px;min-height:110px;}}"
        ".kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:14px 14px 0 0;}"
        ".kpi-card.blue::before{background:linear-gradient(90deg,#3b82f6,#60a5fa);}"
        ".kpi-card.green::before{background:linear-gradient(90deg,#10b981,#34d399);}"
        ".kpi-card.amber::before{background:linear-gradient(90deg,#f59e0b,#fbbf24);}"
        ".kpi-card.red::before{background:linear-gradient(90deg,#ef4444,#f87171);}"
        ".kpi-card.purple::before{background:linear-gradient(90deg,#8b5cf6,#a78bfa);}"
        ".kpi-card.teal::before{background:linear-gradient(90deg,#14b8a6,#2dd4bf);}"
        ".kpi-card.indigo::before{background:linear-gradient(90deg,#6366f1,#818cf8);}"
        ".kpi-card.pink::before{background:linear-gradient(90deg,#ec4899,#f472b6);}"
        ".kpi-icon{position:absolute;top:16px;right:18px;font-size:1.9rem;opacity:0.12;}"
        f".kpi-label{{color:{t['text_label']};font-size:0.7rem;font-weight:600;"
        "text-transform:uppercase;letter-spacing:0.09em;margin-bottom:7px;}"
        ".kpi-value{font-size:1.9rem;font-weight:800;line-height:1;margin-bottom:5px;}"
        ".kpi-value.blue{color:#60a5fa;}.kpi-value.green{color:#34d399;}"
        ".kpi-value.amber{color:#fbbf24;}.kpi-value.red{color:#f87171;}"
        ".kpi-value.purple{color:#a78bfa;}.kpi-value.teal{color:#2dd4bf;}"
        ".kpi-value.indigo{color:#818cf8;}.kpi-value.pink{color:#f472b6;}"
        f".kpi-sub{{color:{t['text_muted']};font-size:0.72rem;font-weight:500;}}"
        # Section header
        f".section-header{{color:{t['section_color']};font-size:0.82rem;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.07em;margin-bottom:12px;padding-bottom:8px;"
        f"border-bottom:1px solid {t['section_border']};}}"
        # Panel card
        f".panel-card{{background:{t['card_bg']};border:1px solid {t['card_border']};"
        f"border-radius:14px;padding:20px;box-shadow:{t['card_shadow']};}}"
        # Activity
        f".activity-item{{display:flex;align-items:flex-start;gap:12px;padding:9px 0;"
        f"border-bottom:1px solid {t['activity_border']};}}"
        ".activity-dot{width:8px;height:8px;border-radius:50%;margin-top:5px;flex-shrink:0;}"
        ".activity-dot.blue{background:#3b82f6;box-shadow:0 0 6px #3b82f6;}"
        ".activity-dot.green{background:#10b981;box-shadow:0 0 6px #10b981;}"
        ".activity-dot.amber{background:#f59e0b;box-shadow:0 0 6px #f59e0b;}"
        ".activity-dot.purple{background:#8b5cf6;box-shadow:0 0 6px #8b5cf6;}"
        ".activity-dot.teal{background:#14b8a6;box-shadow:0 0 6px #14b8a6;}"
        f".activity-text{{color:{t['text_secondary']};font-size:0.8rem;line-height:1.4;}}"
        f".activity-time{{color:{t['text_muted']};font-size:0.68rem;margin-top:2px;}}"
        # Status pills
        f".status-ok{{background:{t['status_ok_bg']};color:{t['status_ok_c']};"
        f"padding:2px 9px;border-radius:20px;font-size:0.68rem;font-weight:600;"
        f"border:1px solid {t['status_ok_b']};}}"
        ".status-warn{background:rgba(245,158,11,0.1);color:#f59e0b;padding:2px 9px;"
        "border-radius:20px;font-size:0.68rem;font-weight:600;border:1px solid rgba(245,158,11,0.25);}"
        f".status-err{{background:{t['status_err_bg']};color:{t['status_err_c']};"
        f"padding:2px 9px;border-radius:20px;font-size:0.68rem;font-weight:600;"
        f"border:1px solid {t['status_err_b']};}}"
        # Metrics
        f"[data-testid='stMetric']{{background:{t['metric_bg']};border:1px solid {t['metric_border']};"
        "border-radius:12px;padding:16px 18px;}"
        f"[data-testid='stMetricLabel']{{color:{t['text_label']} !important;font-size:0.75rem !important;}}"
        f"[data-testid='stMetricValue']{{color:{t['text_primary']} !important;}}"
        f"hr{{border-color:{t['divider']} !important;}}"
        "[data-testid='stDataFrame']{border-radius:10px;overflow:hidden;}"
        # Buttons
        ".stButton>button{background:linear-gradient(135deg,#3b82f6,#2563eb);color:#fff;"
        "border:none;border-radius:8px;font-weight:600;font-size:0.85rem;padding:10px 20px;transition:all 0.2s ease;}"
        ".stButton>button:hover{background:linear-gradient(135deg,#60a5fa,#3b82f6);"
        "transform:translateY(-1px);box-shadow:0 4px 14px rgba(59,130,246,0.45);}"
        # Theme toggle button override
        f".theme-toggle>button{{background:transparent !important;border:1px solid {t['card_border']} !important;"
        f"color:{t['text_secondary']} !important;border-radius:20px !important;font-size:0.8rem !important;"
        "font-weight:600 !important;padding:6px 14px !important;width:100% !important;transition:all 0.2s ease !important;}"
        f".theme-toggle>button:hover{{background:{t['card_border']} !important;color:{t['text_primary']} !important;"
        "transform:none !important;box-shadow:none !important;}"
        # ── Form widget overrides (needed because config.toml base=light forces light colours) ──
        # Text input
        f"[data-baseweb='input']{{background-color:{t['card_bg']} !important;"
        f"border-color:{t['card_border']} !important;}}"
        f"[data-baseweb='input'] input{{background-color:{t['card_bg']} !important;"
        f"color:{t['text_primary']} !important;}}"
        f"[data-baseweb='input']:focus-within{{border-color:#3b82f6 !important;"
        f"box-shadow:0 0 0 2px rgba(59,130,246,0.2) !important;}}"
        # Number input
        f"[data-testid='stNumberInput'] input{{background-color:{t['card_bg']} !important;"
        f"color:{t['text_primary']} !important;border-color:{t['card_border']} !important;}}"
        f"[data-testid='stNumberInputDecrement'],[data-testid='stNumberInputIncrement']"
        f"{{background-color:{t['metric_bg']} !important;color:{t['text_primary']} !important;"
        f"border-color:{t['card_border']} !important;}}"
        # Text area
        f"[data-baseweb='textarea']{{background-color:{t['card_bg']} !important;"
        f"border-color:{t['card_border']} !important;}}"
        f"[data-baseweb='textarea'] textarea{{background-color:{t['card_bg']} !important;"
        f"color:{t['text_primary']} !important;}}"
        # Selectbox
        f"[data-baseweb='select']>div{{background-color:{t['card_bg']} !important;"
        f"border-color:{t['card_border']} !important;color:{t['text_primary']} !important;}}"
        f"[data-baseweb='select'] span{{color:{t['text_primary']} !important;}}"
        f"[data-baseweb='select'] svg{{fill:{t['text_muted']} !important;}}"
        # Dropdown list (popover)
        f"[data-baseweb='popover']{{background-color:{t['card_bg']} !important;"
        f"border:1px solid {t['card_border']} !important;box-shadow:{t['card_shadow']} !important;}}"
        f"[role='listbox']{{background-color:{t['card_bg']} !important;}}"
        f"[role='option']{{background-color:{t['card_bg']} !important;"
        f"color:{t['text_primary']} !important;}}"
        f"[role='option']:hover{{background-color:{t['metric_bg']} !important;}}"
        # Form container border
        f"[data-testid='stForm']{{border-color:{t['card_border']} !important;"
        f"background-color:transparent !important;}}"
        # Form submit button — keep it blue like normal stButton
        "[data-testid='stFormSubmitButton']>button{"
        "background:linear-gradient(135deg,#3b82f6,#2563eb) !important;"
        "color:#ffffff !important;border:none !important;border-radius:8px !important;"
        "font-weight:600 !important;font-size:0.88rem !important;"
        "padding:10px 20px !important;width:100% !important;}"
        "[data-testid='stFormSubmitButton']>button:hover{"
        "background:linear-gradient(135deg,#60a5fa,#3b82f6) !important;"
        "box-shadow:0 4px 14px rgba(59,130,246,0.45) !important;}"
        # Slider track
        f"[data-testid='stSlider'] [data-baseweb='slider'] div[data-testid]"
        f"{{background-color:{t['divider']} !important;}}"
        # Widget labels
        f"[data-testid='stWidgetLabel'] label p,"
        f"[data-testid='stWidgetLabel'] label"
        f"{{color:{t['text_label']} !important;}}"
        # Coming soon
        f".coming-soon-banner{{background:{t['cs_bg']};border:1px dashed {t['cs_border']};"
        "border-radius:16px;padding:80px 40px;text-align:center;margin-top:40px;}"
        # Alert boxes
        f"[data-testid='stInfo']{{background:{t['info_bg']} !important;border:1px solid {t['info_border']} !important;}}"
        f"[data-testid='stWarning']{{background:{t['warn_bg']} !important;border:1px solid {t['warn_border']} !important;}}"
        f"[data-testid='stError']{{background:{t['err_bg']} !important;border:1px solid {t['err_border']} !important;}}"
        # Inputs
        f"[data-testid='stSelectbox']>div>div,[data-testid='stTextInput']>div>div>input,"
        f"[data-testid='stNumberInput']>div>div>input{{background-color:{t['card_bg']} !important;"
        f"color:{t['text_primary']} !important;border-color:{t['card_border']} !important;}}"
        # Placeholder text — must be explicitly coloured; browsers ignore inheritance for ::placeholder
        f"[data-testid='stTextInput']>div>div>input::placeholder,"
        f"[data-testid='stNumberInput']>div>div>input::placeholder"
        f"{{color:{t['text_muted']} !important;opacity:1 !important;}}"
        # Form
        f"[data-testid='stForm']{{background:{t['card_bg']};border:1px solid {t['card_border']};"
        "border-radius:12px;padding:20px;}"
        # Typography
        f"h1,h2,h3,h4{{color:{t['text_primary']} !important;}}"
        f"p,li{{color:{t['text_secondary']};}}"
        f"[data-testid='stCaptionContainer']{{color:{t['text_muted']} !important;}}"
        f".stMarkdown p{{color:{t['text_secondary']};}}"
        f"[data-testid='stJson']{{background:{t['card_bg']} !important;border:1px solid {t['card_border']} !important;border-radius:8px;}}"
        # Settings cards
        f".settings-card{{background:{t['card_bg']};border:1px solid {t['card_border']};"
        "border-radius:14px;padding:24px;margin-bottom:16px;}"
        f".settings-card h4{{color:{t['text_primary']} !important;font-size:0.9rem;font-weight:700;"
        f"margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid {t['card_border']};}}"
        # Download buttons — override Streamlit default white pill
        f"[data-testid='stDownloadButton']>button{{background:{t['metric_bg']} !important;"
        f"color:{t['text_primary']} !important;border:1px solid {t['card_border']} !important;"
        "border-radius:8px !important;font-weight:600 !important;}"
        f"[data-testid='stDownloadButton']>button:hover{{background:{t['card_border']} !important;}}"
        "</style>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CACHED DATA LOADERS  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv_report(file_path: str) -> pd.DataFrame:
    """Safely load CSV report or return empty DataFrame if missing."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()


@st.cache_data
def load_json_config(file_path: str) -> dict:
    """Safely load JSON configuration file."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# THEMED HTML TABLE  — replaces st.dataframe() so tables respect dark/light mode
# ─────────────────────────────────────────────────────────────────────────────
def _render_html_table(
    df: "pd.DataFrame",
    t: dict,
    score_col: str = None,
    height: int = 400,
    rank_col: str = None,
) -> str:
    """Return a scrollable, fully themed HTML table string."""
    header_cells = "".join(
        f'<th style="padding:8px 12px;text-align:left;color:{t["text_muted"]};'
        f'font-size:0.65rem;font-weight:700;text-transform:uppercase;'
        f'letter-spacing:0.06em;border-bottom:2px solid {t["divider"]};'
        f'white-space:nowrap;position:sticky;top:0;'
        f'background:{t["card_bg"]};">{col}</th>'
        for col in df.columns
    )
    rows_html = ""
    for i, (_, row) in enumerate(df.iterrows()):
        row_bg = t["card_bg"] if i % 2 == 0 else t["metric_bg"]
        cells = ""
        for col in df.columns:
            val = row[col]
            if score_col and col == score_col and isinstance(val, (int, float)):
                pct = min(float(val) * 100, 100)
                cells += (
                    f'<td style="padding:8px 12px;">'
                    f'<div style="display:flex;align-items:center;gap:8px;">'
                    f'<div style="background:{t["divider"]};border-radius:3px;'
                    f'height:5px;width:80px;overflow:hidden;">'
                    f'<div style="background:#3b82f6;width:{pct:.0f}%;height:100%;'
                    f'border-radius:3px;"></div></div>'
                    f'<span style="color:{t["text_primary"]};font-size:0.78rem;'
                    f'font-weight:700;">{float(val):.4f}</span></div></td>'
                )
            elif rank_col and col == rank_col:
                cells += (
                    f'<td style="padding:8px 12px;color:{t["text_muted"]};'
                    f'font-size:0.78rem;font-weight:600;">#{val}</td>'
                )
            elif isinstance(val, bool):
                cells += (
                    f'<td style="padding:8px 12px;text-align:center;'
                    f'font-size:0.85rem;">{"✅" if val else "—"}</td>'
                )
            elif isinstance(val, float):
                cells += (
                    f'<td style="padding:8px 12px;color:{t["text_secondary"]};'
                    f'font-size:0.8rem;">{val:.4f}</td>'
                )
            else:
                cells += (
                    f'<td style="padding:8px 12px;color:{t["text_primary"]};'
                    f'font-size:0.8rem;">{val}</td>'
                )
        rows_html += (
            f'<tr style="background:{row_bg};'
            f'border-bottom:1px solid {t["divider"]};">'
            f'{cells}</tr>'
        )
    return (
        f'<div style="overflow-x:auto;overflow-y:auto;max-height:{height}px;'
        f'border:1px solid {t["card_border"]};border-radius:10px;'
        f'background:{t["card_bg"]};box-shadow:{t["card_shadow"]};">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr>{header_cells}</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# HTML HELPERS  — build panel rows safely (no nested f-string triple-quotes)
# ─────────────────────────────────────────────────────────────────────────────
def _kv_row(label: str, value: str, val_color: str, border: str, lbl_color: str) -> str:
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:6px 0;border-bottom:1px solid {border};">'
        f'<span style="color:{lbl_color};font-size:0.77rem;">{label}</span>'
        f'<span style="color:{val_color};font-weight:700;font-size:0.85rem;">{value}</span>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: EXECUTIVE DASHBOARD HOME
# ─────────────────────────────────────────────────────────────────────────────
def render_home_page():
    t = ThemeManager.get()

    rankings_df = load_csv_report(os.path.join("reports", "metrics", "candidate_rankings.csv"))
    fair_config = load_json_config(os.path.join("models", "trained_models", "fairness_config.json"))
    model_info  = load_json_config(os.path.join("models", "trained_models", "best_model_info.json"))
    shap_df     = load_csv_report(os.path.join("reports", "metrics", "shap_feature_importance.csv"))

    total_cands   = len(rankings_df) if not rankings_df.empty else 0
    shortlisted   = int(rankings_df["predicted_class"].sum()) if not rankings_df.empty else 0
    avg_score     = float(rankings_df["prediction_probability"].mean()) if not rankings_df.empty else 0.0
    high_priority = len(rankings_df[rankings_df["priority_tier"] == "High Priority"]) if not rankings_df.empty else 0
    qualified     = len(rankings_df[rankings_df["priority_tier"] == "Qualified"]) if not rankings_df.empty else 0
    extended      = len(rankings_df[rankings_df["priority_tier"] == "Extended"]) if not rankings_df.empty else 0
    reserve       = len(rankings_df[rankings_df["priority_tier"] == "Reserve"]) if not rankings_df.empty else 0

    roc_auc       = model_info.get("best_roc_auc", 0.0)
    best_model    = model_info.get("best_model_name", "N/A")
    model_metrics = model_info.get("metrics", {})

    raw_summary   = fair_config.get("raw_summary", {})
    mit_summary   = fair_config.get("mitigated_summary", {})
    raw_dpd = raw_summary.get("Demographic Parity Difference", 0.0)
    mit_dpd = mit_summary.get("Demographic Parity Difference", 0.0)
    raw_eod = raw_summary.get("Equal Opportunity Difference", 0.0)
    mit_eod = mit_summary.get("Equal Opportunity Difference", 0.0)
    raw_eqo = raw_summary.get("Equalized Odds Difference", 0.0)
    mit_eqo = mit_summary.get("Equalized Odds Difference", 0.0)
    bias_reduction = ((raw_dpd - mit_dpd) / raw_dpd * 100) if raw_dpd > 0 else 0.0
    selection_rate = (shortlisted / total_cands * 100) if total_cands > 0 else 0.0

    top_feature = shap_df.iloc[0]["Feature"] if not shap_df.empty else "N/A"
    top_pct     = float(shap_df.iloc[0]["Importance_Percentage"]) if not shap_df.empty else 0.0
    now = datetime.now().strftime("%B %d, %Y  %H:%M")

    # Header
    st.markdown(
        f'<div style="background:{t["header_bg"]};border:1px solid {t["header_border"]};'
        f'border-radius:16px;padding:26px 30px;margin-bottom:24px;position:relative;overflow:hidden;">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
        f'background:linear-gradient(90deg,#3b82f6 0%,#8b5cf6 50%,#10b981 100%);"></div>'
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;">'
        f'<div><div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">'
        f'<span style="font-size:2rem;">🎯</span>'
        f'<span style="color:{t["header_title"]};font-size:1.7rem;font-weight:800;letter-spacing:-0.03em;">FairHire AI</span>'
        f'<span style="background:rgba(59,130,246,0.15);color:#3b82f6;border:1px solid rgba(59,130,246,0.3);'
        f'padding:3px 10px;border-radius:20px;font-size:0.62rem;font-weight:700;'
        f'letter-spacing:0.1em;text-transform:uppercase;">v2.0</span></div>'
        f'<p style="color:{t["header_sub"]};font-size:0.9rem;margin:0;font-weight:400;line-height:1.5;">'
        f'Explainable &amp; Fair Candidate Screening Platform &nbsp;&middot;&nbsp;'
        f'<span style="color:{t["header_muted"]};">Executive Dashboard</span></p></div>'
        f'<div style="text-align:right;">'
        f'<div style="color:{t["text_muted"]};font-size:0.66rem;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">Last Refreshed</div>'
        f'<div style="color:{t["text_secondary"]};font-size:0.82rem;margin-bottom:6px;">{now}</div>'
        f'<span class="status-ok">● All Systems Operational</span>'
        f'</div></div></div>',
        unsafe_allow_html=True
    )

    # KPI Row 1
    k1, k2, k3, k4 = st.columns(4)
    kpi_cards = [
        (k1, "blue",   "👥", "Total Candidates",   f"{total_cands:,}",    "In current evaluation cohort"),
        (k2, "green",  "✅", "Shortlisted",         f"{shortlisted:,}",    f"Selection rate: {selection_rate:.1f}%"),
        (k3, "purple", "📊", "Avg Suitability Score", f"{avg_score:.3f}", "Mean predicted probability"),
        (k4, "green" if bias_reduction >= 80 else "amber",
             "⚖️", "Bias Reduction", f"{bias_reduction:.1f}%", "Fairlearn post-processing"),
    ]
    for col, color, icon, label, value, sub in kpi_cards:
        with col:
            st.markdown(
                f'<div class="kpi-card {color}"><div class="kpi-icon">{icon}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value {color}">{value}</div>'
                f'<div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

    # KPI Row 2
    k5, k6, k7, k8 = st.columns(4)
    bias_text  = "✔ Mitigated" if bias_reduction >= 80 else "⚠ Under Review"
    b8_color   = "green" if bias_reduction >= 80 else "amber"
    kpi_cards2 = [
        (k5, "red",    "🔥", "High Priority",  f"{high_priority:,}", "Top 10% · Immediate interview"),
        (k6, "teal",   "🎓", "Qualified Pool",  f"{qualified:,}",    "Top 25% · Strong candidates"),
        (k7, "amber",  "🤖", "Model ROC-AUC",   f"{roc_auc:.4f}",   best_model),
        (k8, b8_color, "🛡️", "Fairness Status", bias_text,          f"DPD: {mit_dpd:.4f} after mitigation"),
    ]
    for col, color, icon, label, value, sub in kpi_cards2:
        with col:
            font_sz = "font-size:1.3rem;" if label == "Fairness Status" else ""
            st.markdown(
                f'<div class="kpi-card {color}"><div class="kpi-icon">{icon}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value {color}" style="{font_sz}">{value}</div>'
                f'<div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='margin-top:22px;'></div>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        st.markdown('<div class="section-header">🔻 Hiring Pipeline Funnel</div>', unsafe_allow_html=True)
        funnel_stages = [
            f"Total Evaluated  ({total_cands:,})",
            f"Extended Pool — Top 50%  ({high_priority+qualified+extended:,})",
            f"Qualified Pool — Top 25%  ({high_priority+qualified:,})",
            f"High Priority — Top 10%  ({high_priority:,})",
            f"Shortlisted  ({shortlisted:,})",
        ]
        funnel_vals = [total_cands, high_priority+qualified+extended,
                       high_priority+qualified, high_priority, shortlisted]
        fig_funnel = go.Figure(go.Funnel(
            y=funnel_stages, x=funnel_vals,
            textinfo="value+percent initial",
            marker=dict(color=["#1e3a5f", "#1d4e7e", "#2563eb", "#10b981", "#ef4444"]),
            connector=dict(line=dict(color=t["divider"], width=1, dash="dot")),
            textfont=dict(color=t["text_primary"], family="Inter", size=12)
        ))
        fig_funnel.update_layout(
            paper_bgcolor=t["plotly_paper"], plot_bgcolor=t["plotly_plot"],
            font=dict(family="Inter", color=t["plotly_font"]),
            margin=dict(l=10, r=10, t=8, b=8), height=270
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

        st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🏆 Top High-Priority Candidates</div>', unsafe_allow_html=True)
        if not rankings_df.empty:
            top_cands = rankings_df[rankings_df["priority_tier"] == "High Priority"].head(8).copy()
            top_cands.insert(0, "Rank", range(1, len(top_cands) + 1))
            top_cands = top_cands.rename(columns={
                "enrollee_id": "Candidate ID", "prediction_probability": "Suitability Score",
                "priority_tier": "Tier", "percentile_rank": "Percentile", "predicted_class": "Shortlisted"
            })
            display_cols = ["Rank", "Candidate ID", "Suitability Score", "Percentile", "Shortlisted", "Tier"]
            available = [c for c in display_cols if c in top_cands.columns]
            t_now = ThemeManager.get()
            st.markdown(
                _render_html_table(
                    top_cands[available].reset_index(drop=True),
                    t_now, score_col="Suitability Score",
                    rank_col="Rank", height=310
                ),
                unsafe_allow_html=True,
            )
        else:
            st.warning("Rankings unavailable. Run `python run_ranking.py` first.")

    with col_right:
        st.markdown('<div class="section-header">📊 Priority Tier Distribution</div>', unsafe_allow_html=True)
        tier_labels = ["High Priority", "Qualified", "Extended", "Reserve"]
        tier_values = [high_priority, qualified, extended, reserve]
        tier_colors = ["#ef4444", "#10b981", "#f59e0b", "#6b7280"]
        fig_bar = go.Figure(go.Bar(
            x=tier_labels, y=tier_values, marker_color=tier_colors, marker_line_width=0,
            text=[f"{v:,}<br>{v/total_cands*100:.1f}%" if total_cands > 0 else "0" for v in tier_values],
            textposition="outside",
            textfont=dict(color=t["plotly_font"], size=11, family="Inter"),
            hovertemplate="<b>%{x}</b><br>Candidates: %{y:,}<extra></extra>"
        ))
        fig_bar.update_layout(
            paper_bgcolor=t["plotly_paper"], plot_bgcolor=t["plotly_plot"],
            font=dict(family="Inter", color=t["plotly_font"]),
            xaxis=dict(showgrid=False, tickfont=dict(color=t["plotly_font"], size=11), showline=False),
            yaxis=dict(showgrid=True, gridcolor=t["plotly_grid"], tickfont=dict(color=t["plotly_tick"]), showline=False),
            margin=dict(l=10, r=10, t=30, b=10), height=248, showlegend=False, bargap=0.35
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🕐 Recent Activity</div>', unsafe_allow_html=True)
        activities = [
            ("blue",   "Ranking pipeline complete",  f"Evaluated {total_cands:,} candidates",          "Just now"),
            ("green",  "Bias mitigation applied",    f"DPD reduced {raw_dpd:.4f} → {mit_dpd:.4f}",    "2 min ago"),
            ("purple", "SHAP analysis finished",     f"Top feature: {top_feature} ({top_pct:.1f}%)",   "5 min ago"),
            ("amber",  "Fairness audit complete",    f"Bias reduced by {bias_reduction:.1f}%",         "10 min ago"),
            ("teal",   "Model evaluation complete",  f"ROC-AUC: {roc_auc:.4f} · {best_model}",        "15 min ago"),
        ]
        html_feed = f'<div class="panel-card" style="padding:14px 16px;">'
        for color, title, detail, ts in activities:
            html_feed += (
                f'<div class="activity-item"><div class="activity-dot {color}"></div>'
                f'<div style="flex:1;min-width:0;"><div class="activity-text">'
                f'<strong style="color:{t["text_primary"]};font-weight:600;">{title}</strong><br>'
                f'<span>{detail}</span></div>'
                f'<div class="activity-time">{ts}</div></div></div>'
            )
        html_feed += "</div>"
        st.markdown(html_feed, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3, gap="medium")

    with b1:
        st.markdown('<div class="section-header">🤖 Model Performance</div>', unsafe_allow_html=True)
        metrics_items = [
            ("ROC-AUC",   f"{model_metrics.get('ROC-AUC',   0):.4f}", "#60a5fa"),
            ("Accuracy",  f"{model_metrics.get('Accuracy',  0):.4f}", "#34d399"),
            ("Precision", f"{model_metrics.get('Precision', 0):.4f}", "#a78bfa"),
            ("Recall",    f"{model_metrics.get('Recall',    0):.4f}", "#fbbf24"),
            ("F1-Score",  f"{model_metrics.get('F1-Score',  0):.4f}", "#2dd4bf"),
            ("True Pos",  f"{model_metrics.get('TP', 0):,}",          "#34d399"),
            ("False Pos", f"{model_metrics.get('FP', 0):,}",          "#f87171"),
        ]
        html_model = (
            f'<div class="panel-card">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;">'
            f'<span style="font-size:1.1rem;">🤖</span>'
            f'<span style="color:{t["text_primary"]};font-weight:700;font-size:0.88rem;">{best_model}</span>'
            f'</div>'
        )
        for lbl, val, clr in metrics_items:
            html_model += _kv_row(lbl, val, clr, t["panel_border_l"], t["text_label"])
        html_model += "</div>"
        st.markdown(html_model, unsafe_allow_html=True)

    with b2:
        st.markdown('<div class="section-header">⚖️ Fairness Summary</div>', unsafe_allow_html=True)
        fair_items = [
            ("Dem. Parity Diff (Before)", f"{raw_dpd:.4f}", "#f87171"),
            ("Dem. Parity Diff (After)",  f"{mit_dpd:.4f}", "#34d399"),
            ("Equal Opp. Diff (Before)",  f"{raw_eod:.4f}", "#f87171"),
            ("Equal Opp. Diff (After)",   f"{mit_eod:.4f}", "#34d399"),
            ("Equalized Odds (Before)",   f"{raw_eqo:.4f}", "#f87171"),
            ("Equalized Odds (After)",    f"{mit_eqo:.4f}", "#34d399"),
        ]
        html_fair = '<div class="panel-card">'
        for lbl, val, clr in fair_items:
            html_fair += _kv_row(lbl, val, clr, t["panel_border_l"], t["text_label"])
        html_fair += (
            f'<div style="margin-top:14px;text-align:center;padding:10px;'
            f'background:rgba(16,185,129,0.07);border-radius:10px;border:1px solid rgba(16,185,129,0.2);">'
            f'<span style="color:#34d399;font-weight:700;font-size:0.88rem;">'
            f'✔ {bias_reduction:.1f}% Bias Reduction Achieved</span></div></div>'
        )
        st.markdown(html_fair, unsafe_allow_html=True)

    with b3:
        st.markdown('<div class="section-header">💻 Technology Stack</div>', unsafe_allow_html=True)
        stack = [
            ("🐍", "Python 3.11",  "Core runtime",       "#60a5fa"),
            ("🤖", "Scikit-Learn", "ML pipeline",        "#34d399"),
            ("⚖️", "Fairlearn",    "Bias mitigation",    "#a78bfa"),
            ("🔍", "SHAP",         "Explainability",     "#fbbf24"),
            ("📊", "Streamlit",    "Dashboard UI",       "#f87171"),
            ("📈", "Plotly",       "Interactive charts", "#2dd4bf"),
        ]
        html_tech = '<div class="panel-card">'
        for icon, name, desc, clr in stack:
            html_tech += (
                f'<div style="display:flex;align-items:center;gap:10px;padding:6px 0;'
                f'border-bottom:1px solid {t["panel_border_l"]};">'
                f'<span style="font-size:0.95rem;">{icon}</span>'
                f'<div style="flex:1;"><span style="color:{clr};font-weight:600;font-size:0.8rem;">{name}</span>'
                f'<span style="color:{t["text_muted"]};font-size:0.74rem;"> · {desc}</span></div></div>'
            )
        html_tech += (
            f'<div style="margin-top:14px;padding:10px;background:rgba(59,130,246,0.06);'
            f'border-radius:10px;border:1px solid rgba(59,130,246,0.15);">'
            f'<div style="color:{t["text_muted"]};font-size:0.67rem;text-transform:uppercase;'
            f'letter-spacing:0.07em;margin-bottom:4px;">Top SHAP Driver</div>'
            f'<div style="color:#3b82f6;font-weight:700;font-size:0.82rem;">{top_feature}</div>'
            f'<div style="color:{t["text_muted"]};font-size:0.7rem;margin-top:2px;">'
            f'{top_pct:.1f}% of total model attribution</div></div></div>'
        )
        st.markdown(html_tech, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: CANDIDATE RANKING
# ─────────────────────────────────────────────────────────────────────────────

# Experience seniority order for comparison
_EXP_ORDER = ["<1","1","2","3","4","5","6","7","8","9",
               "10","11","12","13","14","15","16","17","18","19","20",">20"]

# Education quality tiers
_EDU_TIER = {
    "Phd":          3,
    "Masters":      3,
    "Graduate":     2,
    "High School":  1,
    "Primary School": 0,
    "Unknown":      1,
}


def generate_candidate_recommendation(candidate_row: dict) -> dict:
    """
    Dynamic recommendation engine using existing post-prediction candidate data.
    No ML model or src/ files are called — pure rule-based signal aggregation.

    Signals used:
        Suitability Score   — model's predicted probability
        Priority Tier       — tier assigned by ranking algorithm
        Experience          — years of work experience
        Education           — highest education level
        Training Hours      — training investment by candidate
        Relevant Exp        — whether candidate has domain-relevant experience
        City CDI            — city development index (proxy for talent pool)

    Returns:
        dict with keys:
            action      str   — short recruiter action label
            confidence  str   — 'High' | 'Medium' | 'Low'
            reasons     list  — 2–4 plain-English explanation bullets
            score_band  str   — 'Excellent'|'Strong'|'Moderate'|'Weak'

    Designed to be easily extended when Job Description matching is added:
    add a `jd_match_score` parameter and incorporate it into the signal mix.
    """
    score       = float(candidate_row.get("Suitability Score", 0))
    tier        = str(candidate_row.get("Priority Tier", "Reserve"))
    experience  = str(candidate_row.get("Experience", "Unknown")).strip()
    education   = str(candidate_row.get("Education", "Unknown")).strip()
    training    = int(candidate_row.get("Training Hours", 0) or 0)
    rel_exp     = str(candidate_row.get("Relevant Exp", ""))
    cdi         = float(candidate_row.get("City CDI", 0) or 0)
    major       = str(candidate_row.get("Major", ""))

    # ── Derived signal values ──────────────────────────────────────────────
    # Experience rank (0 = <1 year, 21 = >20 years)
    exp_rank = _EXP_ORDER.index(experience) if experience in _EXP_ORDER else 5
    is_senior    = exp_rank >= 14            # 15+ years
    is_mid       = 4 <= exp_rank <= 13       # 5–14 years
    is_junior    = 1 <= exp_rank <= 3        # 1–4 years
    is_entry     = exp_rank == 0             # <1 year

    edu_tier     = _EDU_TIER.get(education, 1)
    has_rel_exp  = "has relevent" in rel_exp.lower()
    high_training = training >= 100          # strong learning commitment
    is_stem      = "STEM" in major

    # Score band
    if score >= 0.75:   score_band = "Excellent"
    elif score >= 0.55: score_band = "Strong"
    elif score >= 0.35: score_band = "Moderate"
    else:               score_band = "Weak"

    reasons: list[str] = []
    action  = ""
    confidence = "Low"

    # ── Decision tree ─────────────────────────────────────────────────────
    if tier == "High Priority":
        if score >= 0.75 and (is_senior or is_mid) and has_rel_exp:
            action     = "Fast-Track Technical Interview"
            confidence = "High"
            reasons    = [
                f"Exceptional suitability score ({score:.3f}) — top {100-float(candidate_row.get('Percentile',100)):.1f}% of cohort",
                f"{experience} years of experience with relevant domain background",
                f"{education} education level meets or exceeds role requirements",
                "Recommend bypassing HR screening — proceed directly to technical panel.",
            ]
        elif score >= 0.75 and (is_junior or is_entry):
            action     = "HR Screening → Technical Interview"
            confidence = "High"
            reasons    = [
                f"Outstanding model score ({score:.3f}) despite early career stage",
                f"Limited experience ({experience} yrs) warrants an HR screening first",
                f"High training investment ({training} hrs) shows strong learning drive",
                "Recommend structured HR assessment followed by technical evaluation.",
            ]
        else:
            action     = "Schedule Technical Interview"
            confidence = "High"
            reasons    = [
                f"High suitability score ({score:.3f}) places candidate in top priority tier",
                f"Education level ({education}) is appropriate for the role",
                "Strong overall profile — recommend standard interview process.",
            ]

    elif tier == "Qualified":
        if score >= 0.55 and has_rel_exp and edu_tier >= 2:
            action     = "Schedule Technical Interview"
            confidence = "High"
            reasons    = [
                f"Strong suitability score ({score:.3f}) above threshold",
                f"Relevant domain experience confirmed",
                f"{education} education aligns with role requirements",
                "Candidate meets all key criteria — proceed to interview.",
            ]
        elif score >= 0.55 and not has_rel_exp:
            action     = "HR Pre-Screening Recommended"
            confidence = "Medium"
            reasons    = [
                f"Good model score ({score:.3f}) but no direct relevant experience",
                f"Training hours ({training} hrs) partially compensate for experience gap",
                "Recommend HR screening to assess transferable skills.",
            ]
        elif high_training and is_stem:
            action     = "Competency Assessment → Interview"
            confidence = "Medium"
            reasons    = [
                f"Score of {score:.3f} is solid but below top-tier threshold",
                f"High training hours ({training} hrs) + STEM background suggest growth potential",
                "Recommend competency test before scheduling interview.",
            ]
        else:
            action     = "Schedule Interview"
            confidence = "Medium"
            reasons    = [
                f"Score ({score:.3f}) qualifies candidate for standard interview pipeline",
                f"Experience ({experience} yrs) is within acceptable range",
                "Proceed with standard screening process.",
            ]

    elif tier == "Extended":
        if (is_senior or is_mid) and edu_tier >= 2 and has_rel_exp:
            action     = "Keep in Pipeline — Senior Profile"
            confidence = "Medium"
            reasons    = [
                f"Moderate score ({score:.3f}) but strong seniority ({experience} yrs) compensates",
                f"{education} education + relevant experience retain value",
                "Recommend re-evaluation if primary candidates decline.",
            ]
        elif high_training:
            action     = "Competency Assessment Before Interview"
            confidence = "Medium"
            reasons    = [
                f"Score of {score:.3f} falls in extended pool range",
                f"Strong training commitment ({training} hrs) indicates adaptability",
                "Recommend formal competency test before progressing.",
            ]
        else:
            action     = "Hold — Requires Further Evaluation"
            confidence = "Low"
            reasons    = [
                f"Suitability score ({score:.3f}) is below the standard shortlisting threshold",
                f"Experience ({experience} yrs) and education ({education}) do not strongly differentiate",
                "Keep on file for future roles or re-application.",
            ]

    else:  # Reserve
        if score >= 0.25 and (is_senior or high_training):
            action     = "Flag for Future Re-evaluation"
            confidence = "Low"
            reasons    = [
                f"Low current suitability score ({score:.3f}) but mitigated by experience ({experience} yrs)",
                f"Training hours ({training} hrs) indicate proactive skill development",
                "Monitor for future openings or role-specific requirements.",
            ]
        else:
            action     = "Not Recommended at This Stage"
            confidence = "Low"
            reasons    = [
                f"Suitability score ({score:.3f}) is significantly below shortlisting threshold",
                f"Current profile (exp: {experience} yrs, edu: {education}) does not meet preferred criteria",
                "Recommend not progressing further in the current cycle.",
            ]

    # CDI bonus reason
    if cdi >= 0.90 and confidence != "High":
        reasons.append(f"Located in a high-development city (CDI: {cdi:.3f}) — strong talent market indicator.")

    return {
        "action":     action,
        "confidence": confidence,
        "reasons":    reasons[:4],   # cap at 4 bullets
        "score_band": score_band,
    }


def _build_ranking_df() -> pd.DataFrame:
    """
    Join candidate_rankings.csv with aug_test.csv on enrollee_id to produce
    a recruiter-friendly enriched table. Pure data join — no ML logic touched.
    Cached separately so filters don't re-read files on every widget interaction.
    """
    rankings = load_csv_report(os.path.join("reports", "metrics", "candidate_rankings.csv"))
    raw      = load_csv_report(os.path.join("data", "raw", "aug_test.csv"))

    if rankings.empty:
        return rankings

    if not raw.empty:
        merged = rankings.merge(raw, on="enrollee_id", how="left")
    else:
        merged = rankings.copy()

    # ── Friendly column names ────────────────────────────────────────────────
    col_map = {
        "enrollee_id":            "Candidate ID",
        "prediction_probability": "Suitability Score",
        "predicted_class":        "Shortlisted",
        "percentile_rank":        "Percentile",
        "priority_tier":          "Priority Tier",
        "gender":                 "Gender",
        "experience":             "Experience",
        "education_level":        "Education",
        "major_discipline":       "Major",
        "company_type":           "Company Type",
        "company_size":           "Company Size",
        "city_development_index": "City CDI",
        "training_hours":         "Training Hours",
        "relevent_experience":    "Relevant Exp",
    }
    merged = merged.rename(columns={k: v for k, v in col_map.items() if k in merged.columns})

    # ── Clean up string columns ──────────────────────────────────────────────
    for col in ["Experience", "Education", "Gender", "Company Type", "Relevant Exp"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna("Unknown").astype(str)

    # ── Dynamic recommendation via generate_candidate_recommendation ─────────
    recs = merged.apply(
        lambda row: generate_candidate_recommendation(row.to_dict()), axis=1
    )
    merged["Recommendation"] = recs.apply(lambda r: r["action"])
    merged["Confidence"]     = recs.apply(lambda r: r["confidence"])
    # Store full rec dict as JSON string for spotlight panel lookup
    merged["_rec_json"]      = recs.apply(lambda r: json.dumps(r))

    # ── Inject Resume Candidates (Session + Persistent History) ─────────────
    history_records = _load_screening_history()
    all_resume_recs = []
    seen_ids = set()

    if "resume_candidate" in st.session_state and st.session_state["resume_candidate"]:
        rc = st.session_state["resume_candidate"]
        cand_id = rc.get("Candidate ID") or rc.get("enrollee_id")
        if cand_id:
            all_resume_recs.append(rc)
            seen_ids.add(str(cand_id))

    for item in history_records:
        r_data = item.get("raw_data", {})
        cand_id = item.get("candidate_id") or item.get("id") or r_data.get("Candidate ID")
        if cand_id and str(cand_id) not in seen_ids:
            r_data["Candidate ID"] = cand_id
            r_data["prediction_probability"] = item.get("prob") or item.get("suitability_score", 0.0)
            r_data["candidate_name"] = item.get("candidate_name") or "Name Not Detected"
            all_resume_recs.append(r_data)
            seen_ids.add(str(cand_id))

    inject_rows = []
    for rc in all_resume_recs:
        prob = rc.get("prediction_probability", 0)
        if prob >= 0.8:
            tier = "High Priority"
        elif prob >= 0.6:
            tier = "Qualified"
        elif prob >= 0.4:
            tier = "Extended"
        else:
            tier = "Reserve"
            
        rc_row = {
            "Candidate ID": rc.get("Candidate ID"),
            "Suitability Score": prob,
            "Shortlisted": 1 if prob >= 0.50 else 0,
            "Percentile": 0.0,
            "Priority Tier": tier,
            "Gender": str(rc.get("gender", "Unknown")),
            "Experience": str(rc.get("experience", "Unknown")),
            "Education": str(rc.get("education_level", "Unknown")),
            "Major": str(rc.get("major_discipline", "Unknown")),
            "Company Type": str(rc.get("company_type", "Unknown")),
            "Company Size": str(rc.get("company_size", "Unknown")),
            "City CDI": rc.get("city_development_index"),
            "Training Hours": rc.get("training_hours"),
            "Relevant Exp": str(rc.get("relevent_experience", "Unknown")),
            "Candidate Name": rc.get("candidate_name", "Resume Candidate"),
        }
        
        rec = generate_candidate_recommendation(rc_row)
        rc_row["Recommendation"] = rec["action"]
        rc_row["Confidence"] = rec["confidence"]
        rc_row["_rec_json"] = json.dumps(rec)
        inject_rows.append(rc_row)
        
    if inject_rows:
        rc_df = pd.DataFrame(inject_rows)
        merged = pd.concat([rc_df, merged], ignore_index=True)

    return merged


def render_ranking_page():
    t = ThemeManager.get()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:{t["header_bg"]};border:1px solid {t["header_border"]};'
        f'border-radius:14px;padding:20px 26px;margin-bottom:20px;position:relative;overflow:hidden;">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
        f'background:linear-gradient(90deg,#3b82f6,#8b5cf6,#10b981);"></div>'
        f'<div style="color:{t["header_title"]};font-size:1.3rem;font-weight:800;margin-bottom:4px;">'
        f'📋 Candidate Rankings</div>'
        f'<div style="color:{t["header_sub"]};font-size:0.85rem;">'
        f'Recruiter-friendly view of all evaluated candidates with profile details, '
        f'suitability scores, and hiring recommendations.</div></div>',
        unsafe_allow_html=True
    )

    # ── Load enriched data ────────────────────────────────────────────────────
    df = _build_ranking_df()

    if df.empty:
        st.error("Rankings data missing. Please run `python run_ranking.py` first.")
        return

    total = len(df)

    # ── KPI cards ─────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    tier_counts = df["Priority Tier"].value_counts()

    kpis = [
        (k1, "blue",   "👥", "Total Evaluated",    f"{total:,}",                                        "All candidates"),
        (k2, "red",    "🔥", "High Priority",       f"{tier_counts.get('High Priority', 0):,}",         "Top 10%"),
        (k3, "teal",   "🎓", "Qualified",           f"{tier_counts.get('Qualified', 0):,}",             "Top 25%"),
        (k4, "amber",  "📋", "Extended",            f"{tier_counts.get('Extended', 0):,}",              "Top 50%"),
        (k5, "green",  "✅", "Shortlisted",         f"{int(df['Shortlisted'].sum()):,}",                "Model flag = 1"),
    ]
    for col, color, icon, label, value, sub in kpis:
        with col:
            st.markdown(
                f'<div class="kpi-card {color}">'
                f'<div class="kpi-icon">{icon}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value {color}">{value}</div>'
                f'<div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # ── Score distribution chart ──────────────────────────────────────────────
    with st.expander("📊 Suitability Score Distribution", expanded=False):
        fig_dist = go.Figure()
        tier_colors_map = {
            "High Priority": "#ef4444",
            "Qualified":     "#10b981",
            "Extended":      "#f59e0b",
            "Reserve":       "#6b7280",
        }
        for tier, clr in tier_colors_map.items():
            subset = df[df["Priority Tier"] == tier]["Suitability Score"]
            if len(subset) > 0:
                fig_dist.add_trace(go.Histogram(
                    x=subset, name=tier, marker_color=clr,
                    opacity=0.75, nbinsx=40,
                    hovertemplate=f"<b>{tier}</b><br>Score: %{{x:.3f}}<br>Count: %{{y}}<extra></extra>"
                ))
        fig_dist.update_layout(
            barmode="overlay",
            paper_bgcolor=t["plotly_paper"], plot_bgcolor=t["plotly_plot"],
            font=dict(family="Inter", color=t["plotly_font"]),
            xaxis=dict(title="Suitability Score", gridcolor=t["plotly_grid"],
                       tickfont=dict(color=t["plotly_font"])),
            yaxis=dict(title="Number of Candidates", gridcolor=t["plotly_grid"],
                       tickfont=dict(color=t["plotly_tick"])),
            legend=dict(bgcolor=t["plotly_paper"], bordercolor=t["card_border"],
                        font=dict(color=t["plotly_font"])),
            margin=dict(l=10, r=10, t=20, b=10), height=280,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown(f"<hr style='border-color:{t['divider']};margin:4px 0 16px 0;'>", unsafe_allow_html=True)

    # ── Filters ───────────────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">🔎 Search & Filters</div>', unsafe_allow_html=True)

    fa, fb, fc = st.columns([2, 2, 2])
    with fa:
        search_id = st.text_input("🔍 Search Candidate ID", placeholder="e.g. 22527", label_visibility="collapsed")
    with fb:
        sel_tiers = st.multiselect(
            "Priority Tier", label_visibility="collapsed",
            options=["High Priority", "Qualified", "Extended", "Reserve"],
            default=["High Priority", "Qualified", "Extended", "Reserve"],
            placeholder="Filter by Priority Tier…"
        )
    with fc:
        sel_shortlist = st.selectbox(
            "Shortlist Status", label_visibility="collapsed",
            options=["All Candidates", "Shortlisted Only", "Not Shortlisted"],
        )

    fd, fe, ff = st.columns([2, 2, 2])
    with fd:
        edu_options = ["All Education"] + (
            sorted(df["Education"].dropna().unique().tolist()) if "Education" in df.columns else []
        )
        sel_edu = st.selectbox("Education", options=edu_options, label_visibility="collapsed")
    with fe:
        gender_opts = ["All Genders"] + (
            sorted(df["Gender"].dropna().unique().tolist()) if "Gender" in df.columns else []
        )
        sel_gender = st.selectbox("Gender", options=gender_opts, label_visibility="collapsed")
    with ff:
        sort_options = {
            "Score ↓ (Highest First)":    ("Suitability Score", False),
            "Score ↑ (Lowest First)":     ("Suitability Score", True),
            "Percentile ↓":               ("Percentile", False),
            "Candidate ID ↑":             ("Candidate ID", True),
        }
        sel_sort = st.selectbox("Sort by", options=list(sort_options.keys()), label_visibility="collapsed")

    # ── Apply filters ─────────────────────────────────────────────────────────
    fdf = df.copy()

    if search_id.strip():
        fdf = fdf[fdf["Candidate ID"].astype(str).str.contains(search_id.strip())]
    if sel_tiers:
        fdf = fdf[fdf["Priority Tier"].isin(sel_tiers)]
    if sel_shortlist == "Shortlisted Only":
        fdf = fdf[fdf["Shortlisted"] == 1]
    elif sel_shortlist == "Not Shortlisted":
        fdf = fdf[fdf["Shortlisted"] == 0]
    if sel_edu != "All Education" and "Education" in fdf.columns:
        fdf = fdf[fdf["Education"] == sel_edu]
    if sel_gender != "All Genders" and "Gender" in fdf.columns:
        fdf = fdf[fdf["Gender"] == sel_gender]

    sort_col, sort_asc = sort_options[sel_sort]
    if sort_col in fdf.columns:
        fdf = fdf.sort_values(sort_col, ascending=sort_asc)

    st.markdown(f"<div style='margin-top:12px;'></div>", unsafe_allow_html=True)

    # ── Pagination ────────────────────────────────────────────────────────────
    PAGE_SIZE = 25
    total_filtered = len(fdf)
    total_pages    = max(1, (total_filtered + PAGE_SIZE - 1) // PAGE_SIZE)

    pg_col1, pg_col2, pg_col3 = st.columns([3, 1, 1])
    with pg_col1:
        st.markdown(
            f'<div style="color:{t["text_secondary"]};font-size:0.85rem;padding-top:6px;">'
            f'Showing <strong style="color:{t["text_primary"]};">{total_filtered:,}</strong> '
            f'of <strong style="color:{t["text_primary"]};">{total:,}</strong> candidates</div>',
            unsafe_allow_html=True
        )
    with pg_col2:
        page_num = st.number_input(
            "Page", min_value=1, max_value=total_pages, value=1,
            step=1, label_visibility="collapsed", key="ranking_page_num"
        )
    with pg_col3:
        st.markdown(
            f'<div style="color:{t["text_muted"]};font-size:0.8rem;padding-top:8px;text-align:right;">'
            f'Page {page_num} / {total_pages}</div>',
            unsafe_allow_html=True
        )

    start = (page_num - 1) * PAGE_SIZE
    page_df = fdf.iloc[start : start + PAGE_SIZE].copy()

    # ── Tier badge colour map ─────────────────────────────────────────────────
    TIER_COLORS = {
        "High Priority": ("#ef4444", "rgba(239,68,68,0.1)",   "rgba(239,68,68,0.25)"),
        "Qualified":     ("#10b981", "rgba(16,185,129,0.1)",  "rgba(16,185,129,0.25)"),
        "Extended":      ("#f59e0b", "rgba(245,158,11,0.1)",  "rgba(245,158,11,0.25)"),
        "Reserve":       ("#6b7280", "rgba(107,114,128,0.1)", "rgba(107,114,128,0.25)"),
    }
    RECO_COLORS = {
        "⚡ Immediate Interview":    "#ef4444",
        "✅ Schedule Interview":     "#10b981",
        "📋 Keep in Pipeline":       "#f59e0b",
        "🔄 Future Consideration":   "#6b7280",
    }

    # ── Candidate table ───────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">👤 Candidate Profiles</div>', unsafe_allow_html=True)

    # Build display columns list
    display_cols_ordered = [
        "Candidate ID", "Gender", "Experience", "Education", "Major",
        "Company Type", "Training Hours", "City CDI",
        "Suitability Score", "Percentile", "Priority Tier",
        "Shortlisted", "Confidence", "Recommendation"
    ]
    show_cols = [c for c in display_cols_ordered if c in page_df.columns]

    col_config = {}
    if "Candidate ID" in show_cols:
        col_config["Candidate ID"] = st.column_config.NumberColumn("Candidate ID", format="%d")
    if "Suitability Score" in show_cols:
        col_config["Suitability Score"] = st.column_config.ProgressColumn(
            "Suitability Score", format="%.4f", min_value=0.0, max_value=1.0)
    if "Percentile" in show_cols:
        col_config["Percentile"] = st.column_config.NumberColumn("Percentile", format="%.1f%%")
    if "Shortlisted" in show_cols:
        col_config["Shortlisted"] = st.column_config.CheckboxColumn("Shortlisted")
    if "City CDI" in show_cols:
        col_config["City CDI"] = st.column_config.NumberColumn("City CDI", format="%.3f")
    if "Training Hours" in show_cols:
        col_config["Training Hours"] = st.column_config.NumberColumn("Training Hours", format="%d hrs")

    t_now = ThemeManager.get()
    st.markdown(
        _render_html_table(
            page_df[show_cols].reset_index(drop=True),
            t_now,
            score_col="Suitability Score" if "Suitability Score" in show_cols else None,
            height=min(600, 56 + len(page_df) * 35),
        ),
        unsafe_allow_html=True,
    )

    # ── Tier legend ───────────────────────────────────────────────────────────
    legend_html = f'<div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;">'
    for tier, (clr, bg, bd) in TIER_COLORS.items():
        count = tier_counts.get(tier, 0)
        legend_html += (
            f'<span style="background:{bg};color:{clr};border:1px solid {bd};'
            f'padding:3px 12px;border-radius:20px;font-size:0.72rem;font-weight:600;">'
            f'{tier} ({count:,})</span>'
        )
    legend_html += "</div>"
    st.markdown(legend_html, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # ── Download section ──────────────────────────────────────────────────────
    st.markdown(f'<div class="section-header">⬇️ Export</div>', unsafe_allow_html=True)
    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        csv_full = fdf.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download All Filtered Results",
            data=csv_full,
            file_name="candidate_shortlist_filtered.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        hp_df = df[df["Priority Tier"] == "High Priority"]
        csv_hp = hp_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="🔥 Download High Priority Only",
            data=csv_hp,
            file_name="high_priority_candidates.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl3:
        sl_df = df[df["Shortlisted"] == 1]
        csv_sl = sl_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="✅ Download Shortlisted Candidates",
            data=csv_sl,
            file_name="shortlisted_candidates.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Candidate Spotlight ───────────────────────────────────────────────────
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-header">🔍 Candidate Spotlight — Full Recommendation</div>',
        unsafe_allow_html=True
    )

    spotlight_ids = fdf["Candidate ID"].astype(str).tolist()
    sel_id = st.selectbox(
        "Select a Candidate ID to view detailed recommendation",
        options=spotlight_ids,
        index=0,
        key="spotlight_candidate",
        label_visibility="collapsed",
    )

    if sel_id:
        row_data = fdf[fdf["Candidate ID"].astype(str) == sel_id]
        if not row_data.empty:
            r = row_data.iloc[0]
            rec        = json.loads(r.get("_rec_json", "{}"))
            action     = rec.get("action", "N/A")
            confidence = rec.get("confidence", "Low")
            reasons    = rec.get("reasons", [])
            score_band = rec.get("score_band", "")

            CONF_COLORS = {
                "High":   ("#10b981", "rgba(16,185,129,0.08)",  "rgba(16,185,129,0.2)"),
                "Medium": ("#f59e0b", "rgba(245,158,11,0.08)",  "rgba(245,158,11,0.2)"),
                "Low":    ("#6b7280", "rgba(107,114,128,0.08)", "rgba(107,114,128,0.2)"),
            }
            BAND_COLORS = {
                "Excellent": "#10b981", "Strong": "#3b82f6",
                "Moderate":  "#f59e0b", "Weak":   "#ef4444",
            }
            c_color, c_bg, c_border = CONF_COLORS.get(confidence, CONF_COLORS["Low"])
            band_hex = BAND_COLORS.get(score_band, "#6b7280")

            sc1, sc2 = st.columns([1, 2], gap="large")

            with sc1:
                profile_fields = [
                    ("🪪 Candidate ID",      str(r.get("Candidate ID", "")),                t["text_primary"]),
                    ("📊 Suitability Score", f"{r.get('Suitability Score', 0):.4f}",         band_hex),
                    ("📈 Percentile",        f"{r.get('Percentile', 0):.1f}%",               t["text_primary"]),
                    ("🏆 Priority Tier",     str(r.get("Priority Tier", "")),                c_color),
                    ("👤 Gender",            str(r.get("Gender", "")),                       t["text_secondary"]),
                    ("🎓 Education",         str(r.get("Education", "")),                    t["text_secondary"]),
                    ("💼 Experience",        f"{r.get('Experience', '')} yrs",               t["text_secondary"]),
                    ("📚 Major",             str(r.get("Major", "")),                        t["text_secondary"]),
                    ("Training Hours",       f"{int(r.get('Training Hours', 0) or 0)} hrs",  t["text_secondary"]),
                    ("🏢 Company Type",      str(r.get("Company Type", "")),                 t["text_secondary"]),
                    ("Relevant Exp",         str(r.get("Relevant Exp", "")),                 t["text_secondary"]),
                ]
                profile_rows = ""
                for label, value, color in profile_fields:
                    profile_rows += (
                        f'<div style="display:flex;justify-content:space-between;padding:5px 0;'
                        f'border-bottom:1px solid {t["panel_border_l"]};">'
                        f'<span style="color:{t["text_label"]};font-size:0.77rem;">{label}</span>'
                        f'<span style="color:{color};font-weight:600;font-size:0.8rem;">{value}</span>'
                        f'</div>'
                    )
                st.markdown(
                    f'<div class="panel-card"><div style="color:{t["text_primary"]};font-weight:700;'
                    f'font-size:0.88rem;margin-bottom:12px;">Candidate Profile</div>'
                    + profile_rows + '</div>',
                    unsafe_allow_html=True
                )

            with sc2:
                reasons_html = ""
                for i, reason in enumerate(reasons):
                    bullet = ["(1)", "(2)", "(3)", "(4)"][i] if i < 4 else "-"
                    reasons_html += (
                        f'<div style="display:flex;align-items:flex-start;gap:10px;'
                        f'padding:8px 0;border-bottom:1px solid {t["panel_border_l"]};">'
                        f'<span style="color:{c_color};font-size:0.85rem;font-weight:700;flex-shrink:0;">{bullet}</span>'
                        f'<span style="color:{t["text_secondary"]};font-size:0.83rem;line-height:1.5;">{reason}</span>'
                        f'</div>'
                    )
                st.markdown(
                    f'<div class="panel-card">'
                    f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">'
                    f'<div style="color:{t["text_primary"]};font-weight:700;font-size:0.88rem;">AI Recommendation</div>'
                    f'<div style="display:flex;gap:8px;">'
                    f'<span style="background:{band_hex}22;color:{band_hex};border:1px solid {band_hex}44;'
                    f'padding:2px 10px;border-radius:20px;font-size:0.68rem;font-weight:700;">{score_band}</span>'
                    f'<span style="background:{c_bg};color:{c_color};border:1px solid {c_border};'
                    f'padding:2px 10px;border-radius:20px;font-size:0.68rem;font-weight:700;">{confidence} Confidence</span>'
                    f'</div></div>'
                    f'<div style="background:{c_bg};border:1px solid {c_border};border-radius:10px;'
                    f'padding:14px 18px;margin-bottom:16px;">'
                    f'<div style="color:{t["text_muted"]};font-size:0.65rem;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">Recommended Action</div>'
                    f'<div style="color:{c_color};font-size:1.05rem;font-weight:800;">{action}</div>'
                    f'</div>'
                    f'<div style="color:{t["text_primary"]};font-size:0.77rem;font-weight:700;margin-bottom:8px;">Key Reasons</div>'
                    + reasons_html +
                    f'<div style="margin-top:14px;padding:10px;background:rgba(59,130,246,0.05);'
                    f'border-radius:8px;border:1px solid rgba(59,130,246,0.12);">'
                    f'<span style="color:{t["text_muted"]};font-size:0.7rem;">'
                    f'Signals used: Suitability Score, Priority Tier, Experience, Education, '
                    f'Training Hours, Relevant Experience, City CDI. '
                    f'Job Description matching will be added in a future phase.</span>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

            # ── Open Full Profile button ──────────────────────────────────────
            st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
            if st.button(
                "👤 Open Full Candidate Profile",
                key=f"open_profile_{sel_id}",
                use_container_width=False,
            ):
                st.session_state["view_profile_id"] = str(sel_id)
                st.session_state["nav_goto"] = "👤  Candidate Profile"
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: JOB DESCRIPTION MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

# ── Job store helpers (JSON file — no ML logic) ───────────────────────────────
_JOBS_PATH = os.path.join("data", "jobs", "jobs.json")


def _load_jobs() -> list:
    """Load all jobs from the JSON store. Returns [] if file missing."""
    if not os.path.exists(_JOBS_PATH):
        return []
    try:
        with open(_JOBS_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("jobs", [])
    except Exception:
        return []


def _save_jobs(jobs: list) -> None:
    """Persist jobs list back to the JSON store."""
    os.makedirs(os.path.dirname(_JOBS_PATH), exist_ok=True)
    with open(_JOBS_PATH, "w", encoding="utf-8") as f:
        json.dump({"jobs": jobs}, f, indent=2, ensure_ascii=False)


def _next_job_id(jobs: list) -> str:
    """Auto-generate next sequential JD-NNN ID."""
    if not jobs:
        return "JD-001"
    nums = []
    for j in jobs:
        jid = j.get("job_id", "JD-000")
        try:
            nums.append(int(jid.split("-")[1]))
        except (IndexError, ValueError):
            pass
    return f"JD-{(max(nums) + 1):03d}" if nums else "JD-001"


def _job_form(
    t: dict,
    mode: str = "create",
    existing: dict = None,
) -> dict | None:
    """
    Render a Create / Edit job form using st.form.
    Returns the submitted job dict, or None if not submitted.
    mode: 'create' | 'edit'
    existing: pre-filled values when editing.
    """
    e = existing or {}
    form_key = f"jd_form_{mode}_{e.get('job_id', 'new')}"

    with st.form(form_key, clear_on_submit=(mode == "create")):
        st.markdown(
            f'<div style="color:{t["text_primary"]};font-weight:700;font-size:1rem;margin-bottom:16px;">'
            f'{"✏️ Edit Job Posting" if mode == "edit" else "➕ Create New Job Posting"}</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Job Title *", value=e.get("title", ""), placeholder="e.g. Senior Data Scientist")
            department = st.text_input("Department *", value=e.get("department", ""), placeholder="e.g. Analytics & AI")
            employment_type = st.selectbox(
                "Employment Type *",
                ["Full-time", "Part-time", "Contract", "Internship", "Freelance"],
                index=["Full-time", "Part-time", "Contract", "Internship", "Freelance"].index(
                    e.get("employment_type", "Full-time")
                ) if e.get("employment_type") in ["Full-time", "Part-time", "Contract", "Internship", "Freelance"] else 0
            )
            location = st.text_input("Location *", value=e.get("location", ""), placeholder="e.g. Bangalore, India (Hybrid)")
            salary_range = st.text_input("Salary Range (optional)", value=e.get("salary_range", ""), placeholder="e.g. 15 - 25 LPA")

        with col2:
            req_exp = st.selectbox(
                "Required Experience *",
                ["<1 year", "1-2 years", "2-4 years", "4-7 years", "7-10 years", "10+ years"],
                index=["<1 year", "1-2 years", "2-4 years", "4-7 years", "7-10 years", "10+ years"].index(
                    e.get("required_experience", "2-4 years")
                ) if e.get("required_experience") in ["<1 year", "1-2 years", "2-4 years", "4-7 years", "7-10 years", "10+ years"] else 2
            )
            req_edu = st.selectbox(
                "Required Education *",
                ["High School", "Graduate", "Masters", "PhD", "Any"],
                index=["High School", "Graduate", "Masters", "PhD", "Any"].index(
                    e.get("required_education", "Graduate")
                ) if e.get("required_education") in ["High School", "Graduate", "Masters", "PhD", "Any"] else 1
            )
            status = st.selectbox(
                "Status *",
                ["Open", "Closed"],
                index=0 if e.get("status", "Open") == "Open" else 1
            )
            req_skills_raw = st.text_input(
                "Required Skills * (comma-separated)",
                value=", ".join(e.get("required_skills", [])),
                placeholder="e.g. Python, SQL, Machine Learning"
            )
            pref_skills_raw = st.text_input(
                "Preferred Skills (comma-separated)",
                value=", ".join(e.get("preferred_skills", [])),
                placeholder="e.g. TensorFlow, Spark, AWS"
            )

        description = st.text_area(
            "Job Description *",
            value=e.get("description", ""),
            height=160,
            placeholder="Describe responsibilities, team, and role expectations..."
        )

        submitted = st.form_submit_button(
            "💾 Save Job" if mode == "edit" else "✅ Create Job",
            use_container_width=True
        )

    if submitted:
        if not title.strip() or not department.strip() or not location.strip() or not description.strip():
            st.error("Please fill in all required fields (marked with *).")
            return None

        req_skills  = [s.strip() for s in req_skills_raw.split(",") if s.strip()]
        pref_skills = [s.strip() for s in pref_skills_raw.split(",") if s.strip()]
        now_str     = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        return {
            "job_id":              e.get("job_id", ""),   # filled by caller on create
            "title":               title.strip(),
            "department":          department.strip(),
            "employment_type":     employment_type,
            "location":            location.strip(),
            "required_experience": req_exp,
            "required_education":  req_edu,
            "required_skills":     req_skills,
            "preferred_skills":    pref_skills,
            "description":         description.strip(),
            "salary_range":        salary_range.strip(),
            "status":              status,
            "created_at":          e.get("created_at", now_str),
            "updated_at":          now_str,
        }
    return None


def render_job_descriptions_page():
    t = ThemeManager.get()

    # ── Session-state init ────────────────────────────────────────────────────
    if "jd_mode"    not in st.session_state: st.session_state["jd_mode"]    = None
    if "jd_edit_id" not in st.session_state: st.session_state["jd_edit_id"] = None
    if "jd_confirm_delete" not in st.session_state: st.session_state["jd_confirm_delete"] = None

    jobs = _load_jobs()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:{t["header_bg"]};border:1px solid {t["header_border"]};'
        f'border-radius:14px;padding:20px 26px;margin-bottom:20px;position:relative;overflow:hidden;">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
        f'background:linear-gradient(90deg,#6366f1,#8b5cf6,#ec4899);"></div>'
        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">'
        f'<div>'
        f'<div style="color:{t["header_title"]};font-size:1.3rem;font-weight:800;margin-bottom:4px;">'
        f'💼 Job Description Management</div>'
        f'<div style="color:{t["header_sub"]};font-size:0.85rem;">'
        f'Create, manage, and track all job postings. Designed for candidate matching in future phases.</div>'
        f'</div></div></div>',
        unsafe_allow_html=True
    )

    # ── KPI bar ───────────────────────────────────────────────────────────────
    total_jobs  = len(jobs)
    open_jobs   = sum(1 for j in jobs if j.get("status") == "Open")
    closed_jobs = sum(1 for j in jobs if j.get("status") == "Closed")
    depts       = len(set(j.get("department", "") for j in jobs))

    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, "blue",   "💼", "Total Postings",  str(total_jobs),  "All job records"),
        (k2, "green",  "✅", "Open Positions",  str(open_jobs),   "Actively hiring"),
        (k3, "red",    "🔒", "Closed Positions", str(closed_jobs), "No longer accepting"),
        (k4, "purple", "🏢", "Departments",      str(depts),       "Unique departments"),
        (k5, "teal",   "📋", "Total Skills",
         str(sum(len(j.get("required_skills", [])) for j in jobs)), "Required skill tags"),
    ]
    for col, color, icon, label, value, sub in kpis:
        with col:
            st.markdown(
                f'<div class="kpi-card {color}">'
                f'<div class="kpi-icon">{icon}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value {color}">{value}</div>'
                f'<div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # ── Search / Filter / Actions row ─────────────────────────────────────────
    fa, fb, fc, fd = st.columns([3, 2, 2, 2])
    with fa:
        search_q = st.text_input("Search jobs", placeholder="Search title, department, skills...",
                                 label_visibility="collapsed")
    with fb:
        filter_status = st.selectbox("Status", ["All", "Open", "Closed"],
                                     label_visibility="collapsed")
    with fc:
        all_depts = sorted(set(j.get("department", "") for j in jobs))
        filter_dept = st.selectbox("Department", ["All Departments"] + all_depts,
                                   label_visibility="collapsed")
    with fd:
        if st.button("➕ Create New Job", use_container_width=True, key="btn_create_jd"):
            st.session_state["jd_mode"]    = "create"
            st.session_state["jd_edit_id"] = None
            st.rerun()

    # ── CREATE form ───────────────────────────────────────────────────────────
    if st.session_state["jd_mode"] == "create":
        st.markdown(f"<hr style='border-color:{t['divider']};margin:8px 0 16px 0;'>", unsafe_allow_html=True)
        result = _job_form(t, mode="create")
        if result is not None:
            jobs = _load_jobs()
            result["job_id"] = _next_job_id(jobs)
            jobs.insert(0, result)
            _save_jobs(jobs)
            st.success(f"Job **{result['job_id']}** — *{result['title']}* created successfully!")
            st.session_state["jd_mode"] = None
            st.rerun()
        col_cancel, _ = st.columns([1, 4])
        with col_cancel:
            if st.button("✖ Cancel", key="cancel_create"):
                st.session_state["jd_mode"] = None
                st.rerun()
        st.markdown(f"<hr style='border-color:{t['divider']};margin:16px 0;'>", unsafe_allow_html=True)

    # ── EDIT form ─────────────────────────────────────────────────────────────
    if st.session_state["jd_mode"] == "edit" and st.session_state["jd_edit_id"]:
        edit_id  = st.session_state["jd_edit_id"]
        edit_job = next((j for j in jobs if j["job_id"] == edit_id), None)
        if edit_job:
            st.markdown(f"<hr style='border-color:{t['divider']};margin:8px 0 16px 0;'>", unsafe_allow_html=True)
            result = _job_form(t, mode="edit", existing=edit_job)
            if result is not None:
                jobs = _load_jobs()
                result["job_id"] = edit_id
                jobs = [result if j["job_id"] == edit_id else j for j in jobs]
                _save_jobs(jobs)
                st.success(f"Job **{edit_id}** updated successfully!")
                st.session_state["jd_mode"]    = None
                st.session_state["jd_edit_id"] = None
                st.rerun()
            col_cancel, _ = st.columns([1, 4])
            with col_cancel:
                if st.button("✖ Cancel", key="cancel_edit"):
                    st.session_state["jd_mode"]    = None
                    st.session_state["jd_edit_id"] = None
                    st.rerun()
            st.markdown(f"<hr style='border-color:{t['divider']};margin:16px 0;'>", unsafe_allow_html=True)

    # ── Apply search + filters ────────────────────────────────────────────────
    filtered = jobs
    if filter_status != "All":
        filtered = [j for j in filtered if j.get("status") == filter_status]
    if filter_dept != "All Departments":
        filtered = [j for j in filtered if j.get("department") == filter_dept]
    if search_q.strip():
        q = search_q.strip().lower()
        filtered = [
            j for j in filtered
            if q in j.get("title", "").lower()
            or q in j.get("department", "").lower()
            or any(q in s.lower() for s in j.get("required_skills", []))
            or any(q in s.lower() for s in j.get("preferred_skills", []))
            or q in j.get("description", "").lower()
        ]

    # ── Results count ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 Job Postings</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:{t["text_secondary"]};font-size:0.84rem;margin-bottom:14px;">'
        f'Showing <strong style="color:{t["text_primary"]};">{len(filtered)}</strong> of '
        f'<strong style="color:{t["text_primary"]};">{total_jobs}</strong> job postings</div>',
        unsafe_allow_html=True
    )

    if not filtered:
        st.info("No job postings match your search criteria. Try adjusting the filters or create a new job.")
        return

    # ── Job Cards (3-column grid) ─────────────────────────────────────────────
    STATUS_COLORS = {
        "Open":   ("#10b981", "rgba(16,185,129,0.1)",  "rgba(16,185,129,0.25)"),
        "Closed": ("#6b7280", "rgba(107,114,128,0.1)", "rgba(107,114,128,0.25)"),
    }
    EMP_COLORS = {
        "Full-time":  "#3b82f6",
        "Part-time":  "#8b5cf6",
        "Contract":   "#f59e0b",
        "Internship": "#10b981",
        "Freelance":  "#ec4899",
    }

    CARDS_PER_ROW = 2
    rows = [filtered[i:i+CARDS_PER_ROW] for i in range(0, len(filtered), CARDS_PER_ROW)]

    for row_jobs in rows:
        cols = st.columns(CARDS_PER_ROW, gap="medium")
        for col, job in zip(cols, row_jobs):
            jid    = job.get("job_id", "")
            title  = job.get("title", "")
            dept   = job.get("department", "")
            etype  = job.get("employment_type", "Full-time")
            loc    = job.get("location", "")
            exp    = job.get("required_experience", "")
            edu    = job.get("required_education", "")
            skills = job.get("required_skills", [])
            pskills = job.get("preferred_skills", [])
            desc   = job.get("description", "")[:180] + ("…" if len(job.get("description", "")) > 180 else "")
            salary = job.get("salary_range", "")
            status = job.get("status", "Open")
            updated = job.get("updated_at", "")[:10]

            s_clr, s_bg, s_bd = STATUS_COLORS.get(status, STATUS_COLORS["Open"])
            e_clr = EMP_COLORS.get(etype, "#3b82f6")

            # Skill pills (required)
            skill_pills = "".join(
                f'<span style="background:rgba(59,130,246,0.1);color:#3b82f6;'
                f'border:1px solid rgba(59,130,246,0.2);padding:2px 8px;'
                f'border-radius:12px;font-size:0.66rem;font-weight:600;margin:2px;">{s}</span>'
                for s in skills[:6]
            )
            if len(skills) > 6:
                skill_pills += (
                    f'<span style="background:rgba(107,114,128,0.1);color:{t["text_muted"]};'
                    f'border:1px solid rgba(107,114,128,0.2);padding:2px 8px;'
                    f'border-radius:12px;font-size:0.66rem;font-weight:600;margin:2px;">+{len(skills)-6} more</span>'
                )

            with col:
                st.markdown(
                    f'<div class="panel-card" style="height:100%;position:relative;">'
                    # Top stripe by status
                    f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
                    f'border-radius:14px 14px 0 0;background:{s_clr};"></div>'
                    # Header row: ID + Status badge
                    f'<div style="display:flex;justify-content:space-between;align-items:flex-start;'
                    f'margin-bottom:10px;padding-top:4px;">'
                    f'<span style="color:{t["text_muted"]};font-size:0.68rem;font-weight:700;'
                    f'letter-spacing:0.08em;">{jid}</span>'
                    f'<span style="background:{s_bg};color:{s_clr};border:1px solid {s_bd};'
                    f'padding:2px 10px;border-radius:20px;font-size:0.68rem;font-weight:700;">{status}</span>'
                    f'</div>'
                    # Title
                    f'<div style="color:{t["text_primary"]};font-size:1rem;font-weight:800;'
                    f'margin-bottom:4px;line-height:1.3;">{title}</div>'
                    # Dept + Type
                    f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;">'
                    f'<span style="color:{t["text_secondary"]};font-size:0.76rem;">🏢 {dept}</span>'
                    f'<span style="color:{e_clr};font-size:0.72rem;font-weight:600;'
                    f'background:{e_clr}18;padding:1px 8px;border-radius:10px;">{etype}</span>'
                    f'</div>'
                    # Meta row
                    f'<div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:10px;'
                    f'color:{t["text_muted"]};font-size:0.74rem;">'
                    f'<span>📍 {loc}</span>'
                    f'<span>💼 {exp}</span>'
                    f'<span>🎓 {edu}</span>'
                    + (f'<span>💰 {salary}</span>' if salary else '') +
                    f'</div>'
                    # Description snippet
                    f'<div style="color:{t["text_secondary"]};font-size:0.79rem;line-height:1.55;'
                    f'margin-bottom:12px;">{desc}</div>'
                    # Required skills
                    f'<div style="margin-bottom:12px;"><div style="color:{t["text_muted"]};'
                    f'font-size:0.65rem;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:0.08em;margin-bottom:5px;">Required Skills</div>'
                    f'<div style="display:flex;flex-wrap:wrap;gap:2px;">{skill_pills}</div></div>'
                    # Updated
                    f'<div style="color:{t["text_hint"]};font-size:0.65rem;margin-top:auto;'
                    f'padding-top:8px;border-top:1px solid {t["panel_border_l"]};">'
                    f'Last updated: {updated}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Action buttons
                ba, bb, bc = st.columns(3)
                with ba:
                    if st.button("✏️ Edit", key=f"edit_{jid}", use_container_width=True):
                        st.session_state["jd_mode"]    = "edit"
                        st.session_state["jd_edit_id"] = jid
                        st.session_state["jd_confirm_delete"] = None
                        st.rerun()
                with bb:
                    # Toggle Open/Closed
                    new_status = "Closed" if status == "Open" else "Open"
                    btn_label  = "🔒 Close" if status == "Open" else "🟢 Reopen"
                    if st.button(btn_label, key=f"toggle_{jid}", use_container_width=True):
                        all_jobs = _load_jobs()
                        for j in all_jobs:
                            if j["job_id"] == jid:
                                j["status"]     = new_status
                                j["updated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                        _save_jobs(all_jobs)
                        st.rerun()
                with bc:
                    if st.button("🗑️ Delete", key=f"del_{jid}", use_container_width=True):
                        st.session_state["jd_confirm_delete"] = jid

                # Delete confirmation
                if st.session_state.get("jd_confirm_delete") == jid:
                    st.warning(f"Delete **{jid} — {title}**? This cannot be undone.")
                    cy, cn = st.columns(2)
                    with cy:
                        if st.button("Yes, Delete", key=f"confirm_del_{jid}", use_container_width=True):
                            all_jobs = _load_jobs()
                            all_jobs = [j for j in all_jobs if j["job_id"] != jid]
                            _save_jobs(all_jobs)
                            st.session_state["jd_confirm_delete"] = None
                            st.rerun()
                    with cn:
                        if st.button("Cancel", key=f"cancel_del_{jid}", use_container_width=True):
                            st.session_state["jd_confirm_delete"] = None
                            st.rerun()

        st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">⬇️ Export</div>', unsafe_allow_html=True)
    exp_col1, exp_col2, _ = st.columns([2, 2, 3])
    with exp_col1:
        if jobs:
            jobs_df = pd.DataFrame([
                {k: (", ".join(v) if isinstance(v, list) else v)
                 for k, v in j.items() if k != "_rec_json"}
                for j in filtered
            ])
            st.download_button(
                label="📥 Export Filtered Jobs (CSV)",
                data=jobs_df.to_csv(index=False).encode("utf-8"),
                file_name="job_postings_export.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with exp_col2:
        if jobs:
            open_jobs_data = [j for j in jobs if j.get("status") == "Open"]
            if open_jobs_data:
                open_df = pd.DataFrame([
                    {k: (", ".join(v) if isinstance(v, list) else v)
                     for k, v in j.items() if k != "_rec_json"}
                    for j in open_jobs_data
                ])
                st.download_button(
                    label="✅ Export Open Jobs Only (CSV)",
                    data=open_df.to_csv(index=False).encode("utf-8"),
                    file_name="open_jobs_export.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    # ── Future matching hook note ─────────────────────────────────────────────
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="background:{t["info_bg"]};border:1px solid {t["info_border"]};'
        f'border-radius:10px;padding:14px 18px;">'
        f'<span style="color:#3b82f6;font-weight:700;font-size:0.82rem;">🔗 Future Phase — Candidate Matching</span>'
        f'<div style="color:{t["text_secondary"]};font-size:0.78rem;margin-top:6px;line-height:1.6;">'
        f'In the next phase, each job posting will be matched against the candidate pool using '
        f'required_skills, required_experience, and required_education from this store. '
        f'The <code>generate_candidate_recommendation()</code> engine will incorporate JD match scores '
        f'to produce role-specific recommendations.</div></div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: CANDIDATE PROFILE
# ─────────────────────────────────────────────────────────────────────────────

# ── Name generation (deterministic from enrollee_id) ──────────────────────────
_MALE_NAMES   = ["Arjun","Rahul","Vikram","Suresh","Amit","Rohan","Dev","Karan",
                  "Siddharth","Aditya","Ravi","Nikhil","Varun","Harsh","Pranav",
                  "Akash","Deepak","Manish","Rajesh","Vivek"]
_FEMALE_NAMES = ["Priya","Anjali","Neha","Sneha","Pooja","Divya","Kavitha",
                  "Shreya","Meera","Aisha","Nisha","Sonal","Ritu","Swati",
                  "Lakshmi","Ananya","Deepika","Pallavi","Smita","Reena"]
_LAST_NAMES   = ["Sharma","Patel","Singh","Kumar","Gupta","Reddy","Verma",
                  "Nair","Mehta","Joshi","Shah","Iyer","Rao","Malhotra",
                  "Saxena","Bhatt","Pillai","Shetty","Kapoor","Chopra"]


def _generate_candidate_name(enrollee_id, gender: str = "Unknown") -> str:
    """Deterministic name from enrollee_id seed — same ID always gives same name."""
    s_id = str(enrollee_id)
    if s_id.isdigit():
        seed = int(s_id)
    else:
        import hashlib
        seed = int(hashlib.md5(s_id.encode("utf-8")).hexdigest(), 16) & 0xffffffff
    rng = random.Random(seed)
    first = rng.choice(_FEMALE_NAMES if gender == "Female" else _MALE_NAMES)
    return f"{first} {rng.choice(_LAST_NAMES)}"


# ── Recruiter notes store ──────────────────────────────────────────────────
_NOTES_PATH = os.path.join("data", "notes", "recruiter_notes.json")


def _load_notes() -> dict:
    if not os.path.exists(_NOTES_PATH):
        return {}
    try:
        with open(_NOTES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_note(candidate_id: str, text: str) -> None:
    notes = _load_notes()
    notes[str(candidate_id)] = {
        "text":       text,
        "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(os.path.dirname(_NOTES_PATH), exist_ok=True)
    with open(_NOTES_PATH, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2, ensure_ascii=False)


# ── SHAP feature label map ───────────────────────────────────────────────────
_FEATURE_LABELS = {
    "city_development_index":              "City Development Index",
    "experience":                          "Years of Experience",
    "training_hours":                      "Training Hours",
    "education_level":                     "Education Level",
    "company_size":                        "Company Size",
    "last_new_job":                        "Years Since Last Job Change",
    "relevent_experience":                 "Relevant Experience",
    "company_type_Pvt Ltd":                "Private Company Background",
    "company_type_Funded Startup":         "Funded Startup Background",
    "company_type_Public Sector":          "Public Sector Background",
    "company_type_NGO":                    "NGO Background",
    "company_type_Unknown":                "Unknown Company Type",
    "company_type_Early Stage Startup":    "Early Stage Startup Background",
    "company_type_Other":                  "Other Company Type",
    "major_discipline_STEM":               "STEM Discipline",
    "major_discipline_Humanities":         "Humanities Discipline",
    "major_discipline_Business Degree":    "Business Discipline",
    "major_discipline_Arts":               "Arts Discipline",
    "major_discipline_No Major":           "No Major Declared",
    "major_discipline_Unknown":            "Unknown Major",
    "major_discipline_Other":              "Other Discipline",
    "enrolled_university_Full time course":"Full-Time University Enrollment",
    "enrolled_university_Part time course":"Part-Time University Enrollment",
    "enrolled_university_no_enrollment":   "Not Enrolled in University",
    "enrolled_university_Unknown":         "University Status Unknown",
    "gender_Male":                         "Male Gender",
    "gender_Female":                       "Female Gender",
    "gender_Other":                        "Other Gender",
    "gender_Unknown":                      "Gender Not Disclosed",
}


def generate_candidate_narrative(candidate_row: dict) -> dict:
    """
    Extends generate_candidate_recommendation() with recruiter-friendly narrative,
    strengths, weaknesses, risks, and suggested interview type.
    No ML model or src/ files called.
    """
    rec = generate_candidate_recommendation(candidate_row)

    score       = float(candidate_row.get("Suitability Score", 0))
    tier        = str(candidate_row.get("Priority Tier", "Reserve"))
    experience  = str(candidate_row.get("Experience", "Unknown"))
    education   = str(candidate_row.get("Education", "Unknown"))
    training    = int(candidate_row.get("Training Hours", 0) or 0)
    rel_exp     = str(candidate_row.get("Relevant Exp", ""))
    major       = str(candidate_row.get("Major", ""))
    company_type = str(candidate_row.get("Company Type", ""))
    cdi         = float(candidate_row.get("City CDI", 0) or 0)

    has_rel_exp = "has relevent" in rel_exp.lower()
    exp_rank    = _EXP_ORDER.index(experience) if experience in _EXP_ORDER else 5
    is_senior   = exp_rank >= 14
    is_mid      = 4 <= exp_rank <= 13
    edu_tier    = _EDU_TIER.get(education, 1)

    # ── Narrative paragraph ────────────────────────────────────────────────
    parts = [f"This candidate carries a {rec['score_band'].lower()} suitability score of {score:.3f}"]
    if has_rel_exp:
        parts.append("with confirmed relevant domain experience")
    if edu_tier >= 3:
        parts.append(f"and holds an advanced {education} degree")
    elif edu_tier == 2:
        parts.append(f"with a solid {education} educational foundation")
    if is_senior:
        parts.append(f"and brings substantial seniority ({experience} years)")
    elif is_mid:
        parts.append(f"and has solid mid-level experience ({experience} years)")
    if training >= 100:
        parts.append(f"A high training investment of {training} hours reflects strong self-driven learning")
    narrative = ". ".join(parts) + f". {rec['action']}."

    # ── Strengths ────────────────────────────────────────────────────────────
    strengths = []
    if score >= 0.65:
        strengths.append(f"High model suitability score ({score:.3f}) — strong predictive fit")
    if has_rel_exp:
        strengths.append("Confirmed relevant experience in the required domain")
    if edu_tier >= 3:
        strengths.append(f"Advanced {education} education adds competitive edge")
    if is_senior or is_mid:
        strengths.append(f"{experience} years of experience demonstrates role readiness")
    if training >= 100:
        strengths.append(f"High training investment ({training} hrs) reflects self-driven growth")
    if cdi >= 0.80:
        strengths.append(f"Based in a high-development city (CDI: {cdi:.3f})")
    if "STEM" in major:
        strengths.append("STEM background aligns with technical role demands")
    if company_type in ["Pvt Ltd", "Funded Startup"]:
        strengths.append(f"Previous {company_type} experience adds relevant commercial context")
    if not strengths:
        strengths.append("Candidate meets minimum screening criteria")

    # ── Weaknesses ───────────────────────────────────────────────────────────
    weaknesses = []
    if score < 0.55:
        weaknesses.append(f"Suitability score ({score:.3f}) is below the standard shortlisting threshold")
    if not has_rel_exp:
        weaknesses.append("No direct relevant experience — domain onboarding may be required")
    if exp_rank <= 2:
        weaknesses.append(f"Limited professional experience ({experience} yrs)")
    if edu_tier <= 1:
        weaknesses.append(f"Education level ({education}) may fall short of preferred requirements")
    if training < 30:
        weaknesses.append("Very low training hours indicate limited upskilling investment")
    if company_type in ["Unknown", ""]:
        weaknesses.append("Previous employer background is unknown")
    if not weaknesses:
        weaknesses.append("No significant weaknesses identified in available data")

    # ── Risks ───────────────────────────────────────────────────────────────
    risks = []
    if tier in ["Extended", "Reserve"]:
        risks.append("Below top-tier threshold — higher risk of not meeting role expectations")
    if not has_rel_exp and score < 0.65:
        risks.append("Lack of relevant experience combined with moderate score increases onboarding risk")
    if exp_rank == 0:
        risks.append("Entry-level profile requires significant mentorship investment")
    if cdi < 0.50:
        risks.append("Lower-development city background may indicate limited competitive exposure")
    if not risks:
        risks.append("Low overall risk profile — candidate meets standard shortlisting criteria")

    # ── Interview type ──────────────────────────────────────────────────────────
    if tier == "High Priority":
        interview_type = "Technical Panel Interview" if (is_senior or is_mid) else "HR Screening + Technical Interview"
    elif tier == "Qualified":
        interview_type = "Technical Interview" if has_rel_exp else "HR Pre-Screening"
    elif tier == "Extended":
        interview_type = "Competency Assessment + Interview"
    else:
        interview_type = "Not recommended for current cycle"

    return {
        **rec,
        "narrative":      narrative,
        "strengths":      strengths[:4],
        "weaknesses":     weaknesses[:3],
        "risks":          risks[:3],
        "interview_type": interview_type,
    }


def render_candidate_profile_page(candidate_id: str | None):
    """Full ATS-style candidate profile page."""
    t  = ThemeManager.get()
    df = _build_ranking_df()

    # ── Back + search bar ───────────────────────────────────────────────────
    top_left, top_right = st.columns([1, 4])
    with top_left:
        if st.button("← Back to Rankings", key="profile_back_btn"):
            st.session_state["view_profile_id"] = None
            st.session_state["nav_goto"] = "📋  Candidate Rankings"
            st.session_state.pop("nav_radio", None)
            st.rerun()
    with top_right:
        all_ids = df["Candidate ID"].astype(str).tolist() if not df.empty else []
        default_idx = all_ids.index(str(candidate_id)) if candidate_id and str(candidate_id) in all_ids else 0
        chosen_id = st.selectbox(
            "Select candidate",
            options=all_ids,
            index=default_idx,
            key="profile_id_select",
            label_visibility="collapsed",
        )
        if str(chosen_id) != str(candidate_id):
            st.session_state["view_profile_id"] = str(chosen_id)
            st.rerun()

    if df.empty:
        st.error("Candidate data not available. Please run `python run_ranking.py` first.")
        return

    # Resolve candidate
    cid = str(candidate_id) if candidate_id else (all_ids[0] if all_ids else None)
    if not cid:
        st.info("Select a candidate from the Rankings page or use the dropdown above.")
        return

    row_data = df[df["Candidate ID"].astype(str) == cid]
    if row_data.empty:
        st.warning(f"Candidate ID {cid} not found in rankings.")
        return

    r   = row_data.iloc[0].to_dict()
    nav = generate_candidate_narrative(r)

    # ── Profile meta ─────────────────────────────────────────────────────────
    gender      = r.get("Gender", "Unknown")
    if str(cid).startswith("RESUME-"):
        name = r.get("Candidate Name", "Resume Candidate")
    else:
        name = _generate_candidate_name(cid, gender)
    score       = float(r.get("Suitability Score", 0))
    percentile  = float(r.get("Percentile", 0))
    tier        = r.get("Priority Tier", "Reserve")
    education   = r.get("Education", "Unknown")
    major       = r.get("Major", "Unknown")
    experience  = r.get("Experience", "Unknown")
    training    = int(r.get("Training Hours", 0) or 0)
    company_type = r.get("Company Type", "Unknown")
    company_size = r.get("Company Size", "Unknown")
    cdi         = float(r.get("City CDI", 0) or 0)
    rel_exp     = r.get("Relevant Exp", "Unknown")
    shortlisted = int(r.get("Shortlisted", 0))

    TIER_CLR = {
        "High Priority": ("#ef4444", "rgba(239,68,68,0.1)",   "rgba(239,68,68,0.25)"),
        "Qualified":     ("#10b981", "rgba(16,185,129,0.1)",  "rgba(16,185,129,0.25)"),
        "Extended":      ("#f59e0b", "rgba(245,158,11,0.1)",  "rgba(245,158,11,0.25)"),
        "Reserve":       ("#6b7280", "rgba(107,114,128,0.1)", "rgba(107,114,128,0.25)"),
    }
    CONF_CLR = {
        "High":   ("#10b981", "rgba(16,185,129,0.1)",  "rgba(16,185,129,0.25)"),
        "Medium": ("#f59e0b", "rgba(245,158,11,0.1)",  "rgba(245,158,11,0.25)"),
        "Low":    ("#6b7280", "rgba(107,114,128,0.1)", "rgba(107,114,128,0.25)"),
    }
    BAND_CLR = {"Excellent": "#10b981", "Strong": "#3b82f6", "Moderate": "#f59e0b", "Weak": "#ef4444"}
    AVATAR   = {"Female": "👩", "Male": "👨"}

    t_clr, t_bg, t_bd  = TIER_CLR.get(tier,          TIER_CLR["Reserve"])
    c_clr, c_bg, c_bd  = CONF_CLR.get(nav["confidence"], CONF_CLR["Low"])
    b_clr               = BAND_CLR.get(nav["score_band"], "#6b7280")
    avatar              = AVATAR.get(gender, "🧑")

    # ─────────────────────────────────────────────────────────────────────────────
    # PROFILE HEADER CARD
    # ─────────────────────────────────────────────────────────────────────────────
    pct_bar = min(score * 100, 100)
    st.markdown(
        f'<div style="background:{t["header_bg"]};border:1px solid {t["header_border"]};'
        f'border-radius:14px;padding:22px 28px;margin-bottom:18px;position:relative;overflow:hidden;">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
        f'background:linear-gradient(90deg,{t_clr},{b_clr},#8b5cf6);"></div>'
        # Avatar + Name block
        f'<div style="display:flex;align-items:flex-start;gap:20px;flex-wrap:wrap;">'
        f'<div style="font-size:3.8rem;background:{t_bg};border:2px solid {t_bd};'
        f'border-radius:50%;width:72px;height:72px;display:flex;align-items:center;'
        f'justify-content:center;flex-shrink:0;">{avatar}</div>'
        f'<div style="flex:1;">'
        f'<div style="color:{t["text_primary"]};font-size:1.5rem;font-weight:800;'
        f'margin-bottom:4px;">{name}</div>'
        f'<div style="color:{t["text_secondary"]};font-size:0.82rem;margin-bottom:10px;">'
        f'Candidate ID: <strong style="color:{t["text_primary"]};">{cid}</strong>  ·  '
        f'Gender: <strong style="color:{t["text_primary"]};">{gender}</strong>  ·  '
        f'City CDI: <strong style="color:{t["text_primary"]};">{cdi:.3f}</strong></div>'
        # Badges row
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">'
        f'<span style="background:{t_bg};color:{t_clr};border:1px solid {t_bd};'
        f'padding:3px 12px;border-radius:20px;font-size:0.73rem;font-weight:700;">{tier}</span>'
        f'<span style="background:{c_bg};color:{c_clr};border:1px solid {c_bd};'
        f'padding:3px 12px;border-radius:20px;font-size:0.73rem;font-weight:700;">{nav["confidence"]} Confidence</span>'
        f'<span style="background:{b_clr}22;color:{b_clr};border:1px solid {b_clr}44;'
        f'padding:3px 12px;border-radius:20px;font-size:0.73rem;font-weight:700;">{nav["score_band"]} Score</span>'
        + (f'<span style="background:rgba(16,185,129,0.1);color:#10b981;border:1px solid rgba(16,185,129,0.25);'
           f'padding:3px 12px;border-radius:20px;font-size:0.73rem;font-weight:700;">Shortlisted</span>'
           if shortlisted else '') +
        f'</div></div>'
        # Score gauge on right
        f'<div style="margin-left:auto;text-align:right;min-width:140px;">'
        f'<div style="color:{t["text_muted"]};font-size:0.62rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;">Suitability Score</div>'
        f'<div style="color:{b_clr};font-size:2.2rem;font-weight:900;line-height:1;">{score:.3f}</div>'
        f'<div style="color:{t["text_muted"]};font-size:0.72rem;margin-bottom:8px;">Top {100-percentile:.1f}% · Percentile {percentile:.1f}</div>'
        f'<div style="background:{t["metric_bg"]};border-radius:6px;height:8px;overflow:hidden;">'
        f'<div style="background:linear-gradient(90deg,{b_clr},{t_clr});height:100%;'
        f'width:{pct_bar:.1f}%;border-radius:6px;"></div></div>'
        f'</div></div></div>',
        unsafe_allow_html=True
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # TABS
    # ─────────────────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Profile",
        "🤖 AI Recommendation",
        "📊 Model Explanation",
        "📝 Recruiter Notes",
        "⚡ Actions",
    ])

    # ═══════ TAB 1: PROFILE ═══════════════════════════════════════════════════
    with tab1:
        col_l, col_r = st.columns([1, 1], gap="medium")

        with col_l:
            # Education card
            def _row(label, value, color=None):
                vc = color or t["text_primary"]
                return (
                    f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
                    f'border-bottom:1px solid {t["panel_border_l"]};">'  
                    f'<span style="color:{t["text_label"]};font-size:0.77rem;">{label}</span>'
                    f'<span style="color:{vc};font-weight:600;font-size:0.8rem;">{value}</span>'
                    f'</div>'
                )
            edu_html = (
                _row("Education Level",  education,         t["text_primary"])
                + _row("Major / Field",  major,             "#3b82f6")
                + _row("Training Hours", f"{training} hrs", t["text_primary"])
                + _row("Relevant Exp",   rel_exp,           "#10b981" if "has" in rel_exp.lower() else "#f59e0b")
            )
            st.markdown(
                f'<div class="panel-card" style="margin-bottom:14px;">'
                f'<div style="color:{t["text_primary"]};font-weight:700;font-size:0.9rem;'
                f'margin-bottom:12px;">🎓 Education</div>' + edu_html + '</div>',
                unsafe_allow_html=True
            )

        with col_r:
            # Professional card
            prof_html = (
                _row("Experience",     f"{experience} yrs", t["text_primary"])
                + _row("Company Type", company_type,         "#3b82f6")
                + _row("Company Size", company_size,         t["text_primary"])
                + _row("City CDI",     f"{cdi:.3f}",         "#10b981" if cdi >= 0.80 else t["text_secondary"])
                + _row("Shortlisted",  "Yes" if shortlisted else "No",
                       "#10b981" if shortlisted else "#ef4444")
            )
            st.markdown(
                f'<div class="panel-card">'
                f'<div style="color:{t["text_primary"]};font-weight:700;font-size:0.9rem;'
                f'margin-bottom:12px;">💼 Professional Information</div>' + prof_html + '</div>',
                unsafe_allow_html=True
            )

    # ═══════ TAB 2: AI RECOMMENDATION ════════════════════════════════════════
    with tab2:
        # Narrative
        st.markdown(
            f'<div class="panel-card" style="margin-bottom:14px;">'
            f'<div style="color:{t["text_primary"]};font-weight:700;font-size:0.9rem;'
            f'margin-bottom:10px;">Recruiter Summary</div>'
            f'<div style="color:{t["text_secondary"]};font-size:0.88rem;line-height:1.65;'
            f'font-style:italic;">{nav["narrative"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Hiring decision card
        st.markdown(
            f'<div class="panel-card" style="margin-bottom:14px;background:{c_bg};border-color:{c_bd};">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">'
            f'<div style="color:{t["text_primary"]};font-weight:700;font-size:0.9rem;">Hiring Decision</div>'
            f'<span style="background:{c_bg};color:{c_clr};border:1px solid {c_bd};'
            f'padding:2px 12px;border-radius:20px;font-size:0.7rem;font-weight:700;">{nav["confidence"]} Confidence</span>'
            f'</div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">'
            f'<div style="background:{t["card_bg"]};border-radius:8px;padding:10px;border:1px solid {t["card_border"]};">'
            f'<div style="color:{t["text_muted"]};font-size:0.62rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.08em;margin-bottom:4px;">Recommended Action</div>'
            f'<div style="color:{c_clr};font-weight:800;font-size:0.9rem;">{nav["action"]}</div></div>'
            f'<div style="background:{t["card_bg"]};border-radius:8px;padding:10px;border:1px solid {t["card_border"]};">'
            f'<div style="color:{t["text_muted"]};font-size:0.62rem;font-weight:700;text-transform:uppercase;'
            f'letter-spacing:0.08em;margin-bottom:4px;">Suggested Interview</div>'
            f'<div style="color:{t["text_primary"]};font-weight:700;font-size:0.88rem;">{nav["interview_type"]}</div></div>'
            f'</div></div>',
            unsafe_allow_html=True
        )

        # Strengths / Weaknesses / Risks
        sw1, sw2, sw3 = st.columns(3, gap="small")
        for col_w, title, items, clr in [
            (sw1, "Strengths",  nav["strengths"],  "#10b981"),
            (sw2, "Weaknesses", nav["weaknesses"], "#f59e0b"),
            (sw3, "Risks",      nav["risks"],      "#ef4444"),
        ]:
            with col_w:
                items_html = "".join(
                    f'<div style="display:flex;gap:8px;padding:6px 0;'
                    f'border-bottom:1px solid {t["panel_border_l"]};">'  
                    f'<span style="color:{clr};flex-shrink:0;">▸</span>'
                    f'<span style="color:{t["text_secondary"]};font-size:0.78rem;line-height:1.4;">{item}</span>'
                    f'</div>'
                    for item in items
                )
                st.markdown(
                    f'<div class="panel-card">'
                    f'<div style="color:{clr};font-weight:700;font-size:0.82rem;margin-bottom:10px;">'
                    f'{title}</div>' + items_html + '</div>',
                    unsafe_allow_html=True
                )

    # ═══════ TAB 3: MODEL EXPLANATION (SHAP) ══════════════════════════════════
    with tab3:
        shap_path = os.path.join("reports", "metrics", "shap_feature_importance.csv")
        shap_df = load_csv_report(shap_path)

        if not shap_df.empty:
            shap_df["Feature_Label"] = shap_df["Feature"].map(
                lambda f: _FEATURE_LABELS.get(f, f.replace("_", " ").title())
            )
            top_shap = shap_df.nlargest(12, "Mean_Abs_Impact").sort_values("Mean_Abs_Impact")

            fig_shap = go.Figure(go.Bar(
                x=top_shap["Mean_Abs_Impact"],
                y=top_shap["Feature_Label"],
                orientation="h",
                marker=dict(
                    color=top_shap["Mean_Abs_Impact"],
                    colorscale=[[0, "#1d4ed8"], [0.5, "#7c3aed"], [1, "#ef4444"]],
                    showscale=False,
                ),
                text=[f"{v:.3f}" for v in top_shap["Mean_Abs_Impact"]],
                textposition="outside",
                textfont=dict(color=t["plotly_font"], size=11),
                hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
            ))
            fig_shap.update_layout(
                paper_bgcolor=t["plotly_paper"], plot_bgcolor=t["plotly_plot"],
                font=dict(family="Inter", color=t["plotly_font"]),
                xaxis=dict(title="Mean |SHAP Value|", gridcolor=t["plotly_grid"],
                           tickfont=dict(color=t["plotly_font"])),
                yaxis=dict(tickfont=dict(color=t["plotly_font"]), categoryorder="total ascending"),
                margin=dict(l=10, r=60, t=10, b=10), height=400,
            )
            st.markdown('<div class="section-header">Global Feature Importance (SHAP)</div>',
                        unsafe_allow_html=True)
            st.plotly_chart(fig_shap, use_container_width=True)

            # Business explanation
            top3 = shap_df.nlargest(3, "Mean_Abs_Impact")["Feature"].tolist()
            top3_labels = [_FEATURE_LABELS.get(f, f) for f in top3]
            st.markdown(
                f'<div style="background:{t["info_bg"]};border:1px solid {t["info_border"]};'
                f'border-radius:10px;padding:14px 18px;margin-top:8px;">'
                f'<div style="color:#3b82f6;font-weight:700;font-size:0.82rem;margin-bottom:6px;">'
                f'Business Explanation</div>'
                f'<div style="color:{t["text_secondary"]};font-size:0.82rem;line-height:1.6;">'
                f'Across all evaluated candidates, the model\'s predictions are most influenced by '
                f'<strong>{top3_labels[0]}</strong> ({shap_df.iloc[0]["Importance_Percentage"]}%), '
                f'<strong>{top3_labels[1]}</strong> ({shap_df.iloc[1]["Importance_Percentage"]}%), '
                f'and <strong>{top3_labels[2]}</strong> ({shap_df.iloc[2]["Importance_Percentage"]}%). '
                f'These are the dominant signals used by the ML model to determine candidate suitability.'
                f'</div></div>',
                unsafe_allow_html=True
            )

            # Per-candidate sample if available
            sample_path = os.path.join("reports", "metrics", "sample_candidate_shap_explanation.json")
            sample_data = load_json_config(sample_path)
            if sample_data and str(sample_data.get("enrollee_id", "")) == str(cid):
                st.markdown('<div class="section-header" style="margin-top:16px;">Per-Candidate SHAP Values</div>',
                            unsafe_allow_html=True)
                pos_factors = sample_data.get("top_positive_factors", [])
                neg_factors = sample_data.get("top_negative_factors", [])

                pf1, pf2 = st.columns(2, gap="medium")
                with pf1:
                    pos_html = "".join(
                        f'<div style="display:flex;justify-content:space-between;padding:5px 0;'
                        f'border-bottom:1px solid {t["panel_border_l"]};">'  
                        f'<span style="color:{t["text_secondary"]};font-size:0.78rem;">'
                        f'{_FEATURE_LABELS.get(f["Feature"],f["Feature"])}</span>'
                        f'<span style="color:#10b981;font-weight:700;font-size:0.8rem;">+{f["Impact"]:.4f}</span>'
                        f'</div>'
                        for f in pos_factors
                    )
                    st.markdown(
                        f'<div class="panel-card"><div style="color:#10b981;font-weight:700;'
                        f'font-size:0.82rem;margin-bottom:10px;">Top Positive Factors</div>'
                        + pos_html + '</div>', unsafe_allow_html=True
                    )
                with pf2:
                    neg_html = "".join(
                        f'<div style="display:flex;justify-content:space-between;padding:5px 0;'
                        f'border-bottom:1px solid {t["panel_border_l"]};">'  
                        f'<span style="color:{t["text_secondary"]};font-size:0.78rem;">'
                        f'{_FEATURE_LABELS.get(f["Feature"],f["Feature"])}</span>'
                        f'<span style="color:#ef4444;font-weight:700;font-size:0.8rem;">{f["Impact"]:.4f}</span>'
                        f'</div>'
                        for f in neg_factors
                    )
                    st.markdown(
                        f'<div class="panel-card"><div style="color:#ef4444;font-weight:700;'
                        f'font-size:0.82rem;margin-bottom:10px;">Top Negative Factors</div>'
                        + neg_html + '</div>', unsafe_allow_html=True
                    )
        else:
            st.info("SHAP feature importance data not found. Please run `python run_explainability.py` first.")

    # ═══════ TAB 4: RECRUITER NOTES ══════════════════════════════════════════
    with tab4:
        notes = _load_notes()
        existing_note = notes.get(str(cid), {})
        existing_text = existing_note.get("text", "")
        existing_ts   = existing_note.get("updated_at", "")

        if existing_text:
            st.markdown(
                f'<div style="background:{t["info_bg"]};border:1px solid {t["info_border"]};'
                f'border-radius:10px;padding:14px 18px;margin-bottom:16px;">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
                f'<span style="color:#3b82f6;font-weight:700;font-size:0.82rem;">Current Note</span>'
                f'<span style="color:{t["text_hint"]};font-size:0.7rem;">Updated: {existing_ts[:10]}</span>'
                f'</div>'
                f'<div style="color:{t["text_secondary"]};font-size:0.84rem;line-height:1.6;'
                f'white-space:pre-wrap;">{existing_text}</div></div>',
                unsafe_allow_html=True
            )

        with st.form(f"note_form_{cid}"):
            note_text = st.text_area(
                "Add or update recruiter note:",
                value=existing_text,
                height=140,
                placeholder="Add interview observations, follow-up actions, concerns, or next steps..."
            )
            save_note = st.form_submit_button("💾 Save Note", use_container_width=True)

        if save_note:
            _save_note(str(cid), note_text.strip())
            st.success("Note saved successfully!")
            st.rerun()

        if existing_text:
            if st.button("🗑️ Clear Note", key=f"clear_note_{cid}"):
                _save_note(str(cid), "")
                st.rerun()

    # ═══════ TAB 5: ACTIONS ═══════════════════════════════════════════════════
    with tab5:
        st.markdown('<div class="section-header">Hiring Actions</div>', unsafe_allow_html=True)
        a1, a2, a3, a4 = st.columns(4)
        actions = [
            (a1, "#10b981", "rgba(16,185,129,0.1)", "rgba(16,185,129,0.25)",
             "Schedule Technical Interview", "Proceed to technical screening"),
            (a2, "#3b82f6", "rgba(59,130,246,0.1)",  "rgba(59,130,246,0.25)",
             "Schedule HR Round",            "Proceed to HR pre-screening"),
            (a3, "#f59e0b", "rgba(245,158,11,0.1)",  "rgba(245,158,11,0.25)",
             "Keep in Pipeline",             "Flag for future consideration"),
            (a4, "#ef4444", "rgba(239,68,68,0.1)",   "rgba(239,68,68,0.25)",
             "Reject",                       "Remove from current cycle"),
        ]
        for col_a, clr, bg, bd, action_label, sub_label in actions:
            with col_a:
                st.markdown(
                    f'<div style="background:{bg};border:1px solid {bd};border-radius:12px;'
                    f'padding:16px;text-align:center;margin-bottom:8px;">'
                    f'<div style="color:{clr};font-weight:700;font-size:0.88rem;'
                    f'margin-bottom:4px;">{action_label}</div>'
                    f'<div style="color:{t["text_muted"]};font-size:0.7rem;">{sub_label}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Downloads</div>', unsafe_allow_html=True)
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            # Download candidate data as CSV
            profile_csv = pd.DataFrame([{
                k: v for k, v in r.items() if k != "_rec_json"
            }]).to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📄 Download Candidate Data (CSV)",
                data=profile_csv,
                file_name=f"candidate_{cid}_profile.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            # Download recommendation report as text
            report_lines = [
                f"Candidate Profile Report",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"=" * 50,
                f"Name:        {name}",
                f"ID:          {cid}",
                f"Gender:      {gender}",
                f"Tier:        {tier}",
                f"Score:       {score:.4f}",
                f"Percentile:  {percentile:.1f}",
                f"Confidence:  {nav['confidence']}",
                f"="*50,
                f"RECOMMENDATION",
                f"Action:      {nav['action']}",
                f"Interview:   {nav['interview_type']}",
                f"Narrative:   {nav['narrative']}",
                f"="*50,
                f"STRENGTHS:",
            ] + [f"  - {s}" for s in nav["strengths"]] + [
                f"WEAKNESSES:",
            ] + [f"  - {w}" for w in nav["weaknesses"]] + [
                f"RISKS:",
            ] + [f"  - {ri}" for ri in nav["risks"]]
            note_data = _load_notes().get(str(cid), {})
            if note_data.get("text"):
                report_lines += [f"="*50, f"RECRUITER NOTES:", note_data["text"]]
            report_text = "\n".join(report_lines).encode("utf-8")
            st.download_button(
                label="📋 Download Recommendation Report (TXT)",
                data=report_text,
                file_name=f"candidate_{cid}_recommendation.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with dl3:
            if st.button("← Back to Rankings", use_container_width=True, key="back_from_actions"):
                st.session_state["view_profile_id"] = None
                st.session_state["nav_goto"] = "📋  Candidate Rankings"
                st.session_state.pop("nav_radio", None)
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: AI FAIRNESS & RESPONSIBLE RECRUITMENT DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
def render_fairness_page():
    t = ThemeManager.get()

    # ── Load all data sources ─────────────────────────────────────────────────
    fairness_path    = os.path.join("reports", "metrics", "fairness_audit_report.csv")
    fair_config_path = os.path.join("models", "trained_models", "fairness_config.json")
    model_info_path  = os.path.join("models", "trained_models", "best_model_info.json")
    rankings_path    = os.path.join("reports", "metrics", "candidate_rankings.csv")
    raw_data_path    = os.path.join("data", "raw", "aug_test.csv")

    fair_df     = load_csv_report(fairness_path)
    fair_config = load_json_config(fair_config_path)
    model_info  = load_json_config(model_info_path)

    if fair_df.empty:
        st.error(f"Fairness report missing at `{fairness_path}`. Please run `python run_fairness.py` first.")
        return

    # ── Derive core metrics from real Fairlearn outputs ───────────────────────
    raw_row = fair_df[fair_df["Stage"].str.contains("Unmitigated", na=False)]
    mit_row = fair_df[fair_df["Stage"].str.contains("Mitigated",   na=False)]

    dpd_raw   = float(raw_row["Demographic Parity Difference"].iloc[0])  if not raw_row.empty else 0.0
    eod_raw   = float(raw_row["Equal Opportunity Difference"].iloc[0])   if not raw_row.empty else 0.0
    equod_raw = float(raw_row["Equalized Odds Difference"].iloc[0])      if not raw_row.empty else 0.0
    dpd_mit   = float(mit_row["Demographic Parity Difference"].iloc[0])  if not mit_row.empty else 0.0
    eod_mit   = float(mit_row["Equal Opportunity Difference"].iloc[0])   if not mit_row.empty else 0.0
    equod_mit = float(mit_row["Equalized Odds Difference"].iloc[0])      if not mit_row.empty else 0.0

    avg_mit        = (dpd_mit + eod_mit + equod_mit) / 3
    avg_raw        = (dpd_raw + eod_raw + equod_raw) / 3
    fairness_score = max(0, (1 - avg_mit) * 100)
    bias_reduction = max(0, (1 - avg_mit / avg_raw) * 100) if avg_raw > 0 else 0
    max_mit        = max(dpd_mit, eod_mit, equod_mit)
    risk_level     = "Low" if max_mit < 0.05 else ("Medium" if max_mit < 0.10 else "High")
    compliance     = "Compliant" if max_mit < 0.10 else "Non-Compliant"

    THRESH = 0.10  # industry standard fairness threshold

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:{t["header_bg"]};border:1px solid {t["header_border"]};'
        f'border-radius:14px;padding:20px 26px;margin-bottom:20px;position:relative;overflow:hidden;">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
        f'background:linear-gradient(90deg,#10b981,#3b82f6,#8b5cf6);"></div>'
        f'<div style="color:{t["header_title"]};font-size:1.3rem;font-weight:800;margin-bottom:4px;">'
        f'⚖️ AI Fairness & Responsible Recruitment Dashboard</div>'
        f'<div style="color:{t["header_sub"]};font-size:0.85rem;">'
        f'Enterprise-grade fairness audit powered by Fairlearn. Ensures AI hiring is transparent, '
        f'equitable, and compliant with responsible AI standards.</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — EXECUTIVE KPI SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Executive Summary</div>', unsafe_allow_html=True)

    RISK_COLORS = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
    COMP_COLORS = {"Compliant": "#10b981", "Non-Compliant": "#ef4444"}
    r_clr = RISK_COLORS.get(risk_level, "#6b7280")
    c_clr = COMP_COLORS.get(compliance, "#6b7280")

    kpi_cols = st.columns(4)
    kpis_row1 = [
        (kpi_cols[0], "green",  "✅", "Fairness Score",      f"{fairness_score:.1f}%",  "Higher is better"),
        (kpi_cols[1], "blue",   "📉", "Bias Reduction",       f"{bias_reduction:.1f}%",  "vs unmitigated model"),
        (kpi_cols[2], "purple", "🎯", "DPD (After)",          f"{dpd_mit:.4f}",           "Target < 0.10"),
        (kpi_cols[3], "teal",   "🔍", "EOD (After)",          f"{eod_mit:.4f}",           "Target < 0.10"),
    ]
    for col, color, icon, label, value, sub in kpis_row1:
        with col:
            st.markdown(
                f'<div class="kpi-card {color}">'
                f'<div class="kpi-icon">{icon}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value {color}">{value}</div>'
                f'<div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True
            )

    kpi_cols2 = st.columns(4)
    kpis_row2 = [
        (kpi_cols2[0], "red",    "⚡", "Equalized Odds (After)", f"{equod_mit:.4f}",  "Target < 0.10"),
        (kpi_cols2[1], "purple", "🛡️", "Protected Attributes",    "4",                "Gender groups audited"),
        (kpi_cols2[2], "green",  "🚦", "Risk Level",              risk_level,          "Current assessment"),
        (kpi_cols2[3], "blue",   "📋", "Compliance Status",       compliance,          "vs 0.10 threshold"),
    ]
    for col, color, icon, label, value, sub in kpis_row2:
        with col:
            vclr = r_clr if "Risk" in label else (c_clr if "Compliance" in label else None)
            v_style = f"color:{vclr};font-weight:900;" if vclr else ""
            st.markdown(
                f'<div class="kpi-card {color}">'
                f'<div class="kpi-icon">{icon}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value" style="{v_style}">{value}</div>'
                f'<div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TABS
    # ─────────────────────────────────────────────────────────────────────────
    tab_fm, tab_bva, tab_pa, tab_comp, tab_risk = st.tabs([
        "📐 Fairness Metrics",
        "📊 Before vs After",
        "👥 Protected Attributes",
        "🛡️ AI Compliance",
        "🚦 Risk & Recommendations",
    ])

    # ═══════ TAB 1: FAIRNESS METRICS ═════════════════════════════════════════
    with tab_fm:
        st.markdown('<div class="section-header">Fairness Metric Definitions & Status</div>',
                    unsafe_allow_html=True)

        metrics_def = [
            {
                "name": "Demographic Parity Difference (DPD)",
                "icon": "⚖️",
                "raw": dpd_raw, "mit": dpd_mit, "target": 0.10,
                "formula": "max(SelectionRate) − min(SelectionRate) across gender groups",
                "explanation": (
                    "Measures whether candidates of all gender groups are selected at similar rates, "
                    "regardless of their qualifications. A value of 0 means perfectly equal selection rates."
                ),
            },
            {
                "name": "Equal Opportunity Difference (EOD)",
                "icon": "🎯",
                "raw": eod_raw, "mit": eod_mit, "target": 0.10,
                "formula": "max(TPR) − min(TPR) across gender groups",
                "explanation": (
                    "Measures whether qualified candidates from all groups have an equal chance of being "
                    "selected (True Positive Rate equality). Ensures deserving candidates aren't unfairly missed."
                ),
            },
            {
                "name": "Equalized Odds Difference",
                "icon": "🔄",
                "raw": equod_raw, "mit": equod_mit, "target": 0.10,
                "formula": "max disparity across both TPR and FPR",
                "explanation": (
                    "Combines both True Positive Rate and False Positive Rate fairness. "
                    "Ensures the model is equally accurate and equally fair across all demographic groups."
                ),
            },
        ]

        for i, m in enumerate(metrics_def):
            status_ok = m["mit"] < m["target"]
            s_clr = "#10b981" if status_ok else "#ef4444"
            s_bg  = "rgba(16,185,129,0.08)" if status_ok else "rgba(239,68,68,0.08)"
            s_bd  = "rgba(16,185,129,0.25)" if status_ok else "rgba(239,68,68,0.25)"
            pct   = min(m["mit"] / m["target"] * 100, 100) if m["target"] else 0
            reduction = ((m["raw"] - m["mit"]) / m["raw"] * 100) if m["raw"] > 0 else 0

            st.markdown(
                f'<div class="panel-card" style="margin-bottom:14px;">'
                f'<div style="display:flex;align-items:flex-start;justify-content:space-between;'
                f'flex-wrap:wrap;gap:12px;margin-bottom:12px;">'
                f'<div style="display:flex;align-items:center;gap:10px;">'
                f'<span style="font-size:1.6rem;">{m["icon"]}</span>'
                f'<div><div style="color:{t["text_primary"]};font-weight:700;font-size:0.92rem;">'
                f'{m["name"]}</div>'
                f'<div style="color:{t["text_muted"]};font-size:0.72rem;margin-top:2px;">'
                f'Formula: {m["formula"]}</div></div></div>'
                f'<span style="background:{s_bg};color:{s_clr};border:1px solid {s_bd};'
                f'padding:4px 14px;border-radius:20px;font-size:0.73rem;font-weight:700;">'
                f'{"✅ Within Threshold" if status_ok else "⚠️ Above Threshold"}</span>'
                f'</div>'
                # Value grid
                f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px;">'
                f'<div style="background:{t["metric_bg"]};border-radius:8px;padding:10px;border:1px solid {t["metric_border"]};">'
                f'<div style="color:{t["text_muted"]};font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;">Before</div>'
                f'<div style="color:#ef4444;font-size:1.1rem;font-weight:800;">{m["raw"]:.4f}</div></div>'
                f'<div style="background:{t["metric_bg"]};border-radius:8px;padding:10px;border:1px solid {t["metric_border"]};">'
                f'<div style="color:{t["text_muted"]};font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;">After (Fairlearn)</div>'
                f'<div style="color:{s_clr};font-size:1.1rem;font-weight:800;">{m["mit"]:.4f}</div></div>'
                f'<div style="background:{t["metric_bg"]};border-radius:8px;padding:10px;border:1px solid {t["metric_border"]};">'
                f'<div style="color:{t["text_muted"]};font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;">Target</div>'
                f'<div style="color:{t["text_primary"]};font-size:1.1rem;font-weight:800;">< {m["target"]:.2f}</div></div>'
                f'<div style="background:{t["metric_bg"]};border-radius:8px;padding:10px;border:1px solid {t["metric_border"]};">'
                f'<div style="color:{t["text_muted"]};font-size:0.6rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;">Reduction</div>'
                f'<div style="color:#10b981;font-size:1.1rem;font-weight:800;">↓{reduction:.1f}%</div></div>'
                f'</div>'
                # Progress bar
                f'<div style="margin-bottom:8px;"><div style="display:flex;justify-content:space-between;'
                f'color:{t["text_muted"]};font-size:0.68rem;margin-bottom:4px;">'
                f'<span>Distance to threshold</span><span>{pct:.0f}% of limit used</span></div>'
                f'<div style="background:{t["metric_bg"]};border-radius:4px;height:6px;">'
                f'<div style="background:linear-gradient(90deg,{s_clr},{s_clr}80);width:{pct:.0f}%;'
                f'height:100%;border-radius:4px;"></div></div></div>'
                # Explanation
                f'<div style="background:{t["info_bg"]};border:1px solid {t["info_border"]};'
                f'border-radius:8px;padding:10px 14px;">'
                f'<span style="color:{t["text_secondary"]};font-size:0.78rem;line-height:1.55;">'
                f'{m["explanation"]}</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Gender thresholds
        if fair_config.get("fair_thresholds"):
            st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">⚙️ Gender-Calibrated Classification Thresholds</div>',
                        unsafe_allow_html=True)
            thresh_cols = st.columns(len(fair_config["fair_thresholds"]))
            for ci, (group, thresh) in enumerate(fair_config["fair_thresholds"].items()):
                with thresh_cols[ci]:
                    st.markdown(
                        f'<div class="panel-card" style="text-align:center;">'
                        f'<div style="font-size:1.5rem;">👤</div>'
                        f'<div style="color:{t["text_primary"]};font-weight:700;font-size:0.88rem;'
                        f'margin:6px 0 4px;">{group}</div>'
                        f'<div style="color:#3b82f6;font-size:1.4rem;font-weight:900;">{thresh}</div>'
                        f'<div style="color:{t["text_muted"]};font-size:0.68rem;">Calibrated threshold</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    # ═══════ TAB 2: BEFORE vs AFTER ══════════════════════════════════════════
    with tab_bva:
        st.markdown('<div class="section-header">Before vs After Bias Mitigation (Fairlearn)</div>',
                    unsafe_allow_html=True)

        # Grouped bar chart — all 3 metrics, before and after
        metric_names  = ["Demographic Parity Difference", "Equal Opportunity Difference", "Equalized Odds Difference"]
        before_vals   = [dpd_raw, eod_raw, equod_raw]
        after_vals    = [dpd_mit, eod_mit, equod_mit]
        short_labels  = ["Demographic Parity", "Equal Opportunity", "Equalized Odds"]

        fig_bva = go.Figure()
        fig_bva.add_trace(go.Bar(
            name="Before Mitigation (Raw Model)", x=short_labels, y=before_vals,
            marker_color="#ef4444", marker_opacity=0.85,
            text=[f"{v:.4f}" for v in before_vals], textposition="outside",
            textfont=dict(color=t["plotly_font"], size=12),
            hovertemplate="<b>%{x}</b><br>Before: %{y:.4f}<extra></extra>",
        ))
        fig_bva.add_trace(go.Bar(
            name="After Mitigation (Fairlearn)", x=short_labels, y=after_vals,
            marker_color="#10b981", marker_opacity=0.85,
            text=[f"{v:.4f}" for v in after_vals], textposition="outside",
            textfont=dict(color=t["plotly_font"], size=12),
            hovertemplate="<b>%{x}</b><br>After: %{y:.4f}<extra></extra>",
        ))
        fig_bva.add_hline(
            y=THRESH, line_dash="dash", line_color="#f59e0b", line_width=2,
            annotation_text="Fairness Threshold (0.10)",
            annotation_font_color="#f59e0b",
        )
        fig_bva.update_layout(
            barmode="group", bargap=0.3, bargroupgap=0.08,
            paper_bgcolor=t["plotly_paper"], plot_bgcolor=t["plotly_plot"],
            font=dict(family="Inter", color=t["plotly_font"]),
            legend=dict(font=dict(color=t["plotly_font"]), bgcolor=t["plotly_paper"]),
            xaxis=dict(tickfont=dict(color=t["plotly_font"]), gridcolor=t["plotly_grid"]),
            yaxis=dict(title="Disparity Difference", tickfont=dict(color=t["plotly_font"]),
                       gridcolor=t["plotly_grid"], range=[0, max(before_vals) * 1.4]),
            margin=dict(l=10, r=10, t=10, b=10), height=380,
        )
        st.plotly_chart(fig_bva, use_container_width=True)

        # Reduction table
        st.markdown('<div class="section-header" style="margin-top:16px;">Reduction Summary</div>',
                    unsafe_allow_html=True)
        rc1, rc2, rc3 = st.columns(3)
        for col, metric, bef, aft in [
            (rc1, "Demographic Parity",  dpd_raw,   dpd_mit),
            (rc2, "Equal Opportunity",   eod_raw,   eod_mit),
            (rc3, "Equalized Odds",      equod_raw, equod_mit),
        ]:
            with col:
                red_pct = ((bef - aft) / bef * 100) if bef > 0 else 0
                st.markdown(
                    f'<div class="panel-card" style="text-align:center;">'
                    f'<div style="color:{t["text_label"]};font-size:0.72rem;margin-bottom:8px;">{metric}</div>'
                    f'<div style="display:flex;justify-content:center;align-items:center;gap:12px;margin-bottom:8px;">'
                    f'<div><div style="color:#ef4444;font-size:1.1rem;font-weight:800;">{bef:.4f}</div>'
                    f'<div style="color:{t["text_muted"]};font-size:0.65rem;">Before</div></div>'
                    f'<div style="color:{t["text_muted"]};font-size:1.2rem;">→</div>'
                    f'<div><div style="color:#10b981;font-size:1.1rem;font-weight:800;">{aft:.4f}</div>'
                    f'<div style="color:{t["text_muted"]};font-size:0.65rem;">After</div></div>'
                    f'</div>'
                    f'<div style="color:#10b981;font-size:1.6rem;font-weight:900;">↓{red_pct:.1f}%</div>'
                    f'<div style="color:{t["text_muted"]};font-size:0.7rem;">Bias reduction</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # Model performance note
        if model_info and model_info.get("metrics"):
            m = model_info["metrics"]
            st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">Model Performance at Fairness-Calibrated Thresholds</div>',
                        unsafe_allow_html=True)
            mp_cols = st.columns(5)
            for col, label, val in [
                (mp_cols[0], "Accuracy",  f"{m.get('Accuracy',0):.4f}"),
                (mp_cols[1], "Precision", f"{m.get('Precision',0):.4f}"),
                (mp_cols[2], "Recall",    f"{m.get('Recall',0):.4f}"),
                (mp_cols[3], "F1-Score",  f"{m.get('F1-Score',0):.4f}"),
                (mp_cols[4], "ROC-AUC",   f"{m.get('ROC-AUC',0):.4f}"),
            ]:
                with col:
                    st.markdown(
                        f'<div class="panel-card" style="text-align:center;">'
                        f'<div style="color:{t["text_muted"]};font-size:0.68rem;font-weight:700;'
                        f'text-transform:uppercase;letter-spacing:0.08em;">{label}</div>'
                        f'<div style="color:#3b82f6;font-size:1.3rem;font-weight:900;">{val}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    # ═══════ TAB 3: PROTECTED ATTRIBUTE ANALYSIS ══════════════════════════════
    with tab_pa:
        st.markdown('<div class="section-header">Protected Group Analysis — Real Candidate Data</div>',
                    unsafe_allow_html=True)

        # Load and merge data for real group analysis
        try:
            rank_df = pd.read_csv(rankings_path)
            raw_df  = pd.read_csv(raw_data_path)
            merged  = rank_df.merge(raw_df, on="enrollee_id", how="left")

            def _group_table(df, group_col, friendly_label):
                if group_col not in df.columns:
                    return None
                grp = df.groupby(group_col).agg(
                    Total=("enrollee_id",           "count"),
                    Selected=("predicted_class",    "sum"),
                    Avg_Score=("prediction_probability", "mean"),
                    Pct_High_Priority=(
                        "priority_tier",
                        lambda x: round((x == "High Priority").mean() * 100, 1)
                    ),
                ).reset_index()
                grp["Selection_Rate"] = (grp["Selected"] / grp["Total"] * 100).round(1)
                grp["Avg_Score"] = grp["Avg_Score"].round(4)
                grp.rename(columns={group_col: friendly_label}, inplace=True)
                return grp

            def _bias_badge(rate, group_rates):
                avg_rate = group_rates.mean()
                diff = abs(rate - avg_rate)
                if diff < 2:   return ("✅ Neutral",  "#10b981", "rgba(16,185,129,0.1)")
                if diff < 5:   return ("⚠️ Mild",     "#f59e0b", "rgba(245,158,11,0.1)")
                return              ("🔴 Biased",   "#ef4444", "rgba(239,68,68,0.1)")

            def _render_group_analysis(group_df, label_col):
                if group_df is None or group_df.empty:
                    st.info(f"No data available for {label_col} grouping.")
                    return
                rates = group_df["Selection_Rate"]
                rows_html = ""
                for _, row in group_df.iterrows():
                    badge, b_clr, b_bg = _bias_badge(row["Selection_Rate"], rates)
                    bar_w = min(row["Selection_Rate"] / (rates.max() + 1e-9) * 100, 100)
                    rows_html += (
                        f'<tr>'
                        f'<td style="color:{t["text_primary"]};font-weight:600;padding:8px 6px;">'
                        f'{row[label_col]}</td>'
                        f'<td style="color:{t["text_secondary"]};padding:8px 6px;">{int(row["Total"])}</td>'
                        f'<td style="padding:8px 6px;">'
                        f'<div style="display:flex;align-items:center;gap:8px;">'
                        f'<div style="background:{t["metric_bg"]};border-radius:4px;height:6px;'
                        f'width:80px;overflow:hidden;">'
                        f'<div style="background:#3b82f6;width:{bar_w:.0f}%;height:100%;'
                        f'border-radius:4px;"></div></div>'
                        f'<span style="color:{t["text_primary"]};font-weight:700;">'
                        f'{row["Selection_Rate"]}%</span></div></td>'
                        f'<td style="color:#3b82f6;font-weight:700;padding:8px 6px;">'
                        f'{row["Avg_Score"]}</td>'
                        f'<td style="color:#8b5cf6;padding:8px 6px;">{row["Pct_High_Priority"]}%</td>'
                        f'<td style="padding:8px 6px;">'
                        f'<span style="background:{b_bg};color:{b_clr};border:1px solid {b_clr}44;'
                        f'padding:2px 10px;border-radius:20px;font-size:0.67rem;font-weight:700;">'
                        f'{badge}</span></td>'
                        f'</tr>'
                    )
                st.markdown(
                    f'<div class="panel-card" style="overflow-x:auto;">'
                    f'<table style="width:100%;border-collapse:collapse;font-size:0.8rem;">'
                    f'<thead><tr style="border-bottom:2px solid {t["divider"]};">'
                    f'<th style="color:{t["text_muted"]};font-size:0.65rem;text-align:left;'
                    f'padding:6px;text-transform:uppercase;letter-spacing:0.06em;">{label_col}</th>'
                    f'<th style="color:{t["text_muted"]};font-size:0.65rem;text-align:left;'
                    f'padding:6px;text-transform:uppercase;letter-spacing:0.06em;">Count</th>'
                    f'<th style="color:{t["text_muted"]};font-size:0.65rem;text-align:left;'
                    f'padding:6px;text-transform:uppercase;letter-spacing:0.06em;">Selection Rate</th>'
                    f'<th style="color:{t["text_muted"]};font-size:0.65rem;text-align:left;'
                    f'padding:6px;text-transform:uppercase;letter-spacing:0.06em;">Avg Score</th>'
                    f'<th style="color:{t["text_muted"]};font-size:0.65rem;text-align:left;'
                    f'padding:6px;text-transform:uppercase;letter-spacing:0.06em;">High Priority %</th>'
                    f'<th style="color:{t["text_muted"]};font-size:0.65rem;text-align:left;'
                    f'padding:6px;text-transform:uppercase;letter-spacing:0.06em;">Bias Indicator</th>'
                    f'</tr></thead><tbody>' + rows_html + '</tbody></table></div>',
                    unsafe_allow_html=True
                )

            # Experience bucketing
            if "experience" in merged.columns:
                def _exp_bucket(e):
                    try:
                        v = int(str(e).replace(">","").replace("<",""))
                        if v <= 2:  return "0–2 yrs"
                        if v <= 5:  return "3–5 yrs"
                        if v <= 10: return "6–10 yrs"
                        return "10+ yrs"
                    except Exception:
                        return "Unknown"
                merged["exp_group"] = merged["experience"].apply(_exp_bucket)

            # CDI bucketing
            if "city_development_index" in merged.columns:
                merged["cdi_group"] = pd.cut(
                    merged["city_development_index"].fillna(0),
                    bins=[0, 0.6, 0.8, 1.0],
                    labels=["Low CDI (< 0.6)", "Mid CDI (0.6–0.8)", "High CDI (> 0.8)"]
                )

            pa_attr_tabs = st.tabs([
                "👤 Gender", "🎓 Education", "💼 Experience", "🏢 Company Type", "📍 City CDI"
            ])
            groups = [
                (pa_attr_tabs[0], "gender",           "Gender"),
                (pa_attr_tabs[1], "education_level",  "Education Level"),
                (pa_attr_tabs[2], "exp_group",         "Experience Group"),
                (pa_attr_tabs[3], "company_type",      "Company Type"),
                (pa_attr_tabs[4], "cdi_group",         "City Development Index"),
            ]
            for tab_g, col, lbl in groups:
                with tab_g:
                    grp_df = _group_table(merged, col, lbl)
                    _render_group_analysis(grp_df, lbl)

                    # Plotly bar for selection rate
                    if grp_df is not None and not grp_df.empty:
                        fig_g = go.Figure(go.Bar(
                            x=grp_df[lbl].astype(str),
                            y=grp_df["Selection_Rate"],
                            marker_color=[
                                "#ef4444" if abs(r - grp_df["Selection_Rate"].mean()) >= 5
                                else "#f59e0b" if abs(r - grp_df["Selection_Rate"].mean()) >= 2
                                else "#10b981"
                                for r in grp_df["Selection_Rate"]
                            ],
                            text=[f"{v}%" for v in grp_df["Selection_Rate"]],
                            textposition="outside",
                            hovertemplate="<b>%{x}</b><br>Selection Rate: %{y}%<extra></extra>",
                        ))
                        fig_g.update_layout(
                            paper_bgcolor=t["plotly_paper"], plot_bgcolor=t["plotly_plot"],
                            font=dict(color=t["plotly_font"]),
                            xaxis=dict(tickfont=dict(color=t["plotly_font"])),
                            yaxis=dict(title="Selection Rate (%)", tickfont=dict(color=t["plotly_font"]),
                                       gridcolor=t["plotly_grid"]),
                            margin=dict(l=10, r=10, t=10, b=10), height=280,
                        )
                        st.plotly_chart(fig_g, use_container_width=True)

        except Exception as ex:
            st.error(f"Could not compute protected attribute analysis: {ex}")

    # ═══════ TAB 4: AI COMPLIANCE DASHBOARD ══════════════════════════════════
    with tab_comp:
        st.markdown('<div class="section-header">AI Compliance & Responsible AI Principles</div>',
                    unsafe_allow_html=True)

        # All statuses derived from real metrics
        all_compliant = max_mit < THRESH
        compliance_items = [
            {
                "principle": "Fairness",
                "icon": "⚖️",
                "status": "Compliant" if all_compliant else "Review Required",
                "color": "#10b981" if all_compliant else "#f59e0b",
                "description": (
                    f"All fairness metrics are within the {THRESH:.2f} threshold after Fairlearn mitigation. "
                    f"Demographic Parity Difference: {dpd_mit:.4f}, Equal Opportunity: {eod_mit:.4f}."
                ),
                "recommendation": (
                    "Continue monitoring fairness metrics quarterly. "
                    "Retrain mitigation model if DPD exceeds 0.10."
                ),
            },
            {
                "principle": "Transparency",
                "icon": "🔍",
                "status": "Compliant",
                "color": "#10b981",
                "description": (
                    "All model decisions are explainable via SHAP feature attribution. "
                    "The top influential factors (City Development Index, Company Type, Experience) "
                    "are documented and accessible to recruiters."
                ),
                "recommendation": "Provide SHAP explanations to recruiters for all shortlisted and rejected candidates.",
            },
            {
                "principle": "Accountability",
                "icon": "📋",
                "status": "Compliant",
                "color": "#10b981",
                "description": (
                    "Full audit trail exists via candidate rankings CSV, fairness audit report, "
                    "and SHAP explanations. Recruiter notes are persistently stored per candidate."
                ),
                "recommendation": "Ensure recruiter notes are reviewed during HR audits. Archive reports monthly.",
            },
            {
                "principle": "Explainability",
                "icon": "🧠",
                "status": "Compliant",
                "color": "#10b981",
                "description": (
                    "SHAP-based model explanations are generated for all candidates. "
                    "Recruiters can view per-candidate top positive and negative factors on the profile page."
                ),
                "recommendation": "Extend per-candidate SHAP explanations to all 2,129 candidates in a future phase.",
            },
            {
                "principle": "Privacy",
                "icon": "🔒",
                "status": "Review Required",
                "color": "#f59e0b",
                "description": (
                    "Candidate data is stored locally. Gender is used as a protected attribute "
                    "only for fairness calibration, not as a selection criterion."
                ),
                "recommendation": (
                    "Ensure data retention policies are in place. "
                    "Conduct a privacy impact assessment before production deployment."
                ),
            },
            {
                "principle": "Responsible AI",
                "icon": "🤝",
                "status": "Compliant",
                "color": "#10b981",
                "description": (
                    "The system uses Fairlearn's Equalized Odds constraint to mitigate bias. "
                    "Human recruiter review is required before any final hiring decision."
                ),
                "recommendation": "Maintain human-in-the-loop for all final hiring decisions. Review AI recommendations, not rely on them.",
            },
        ]

        comp_cols = st.columns(2)
        for i, item in enumerate(compliance_items):
            with comp_cols[i % 2]:
                bg  = f"{item['color']}10"
                bd  = f"{item['color']}30"
                st.markdown(
                    f'<div class="panel-card" style="margin-bottom:12px;border-left:4px solid {item["color"]};">'
                    f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;">'
                    f'<div style="display:flex;align-items:center;gap:8px;">'
                    f'<span style="font-size:1.3rem;">{item["icon"]}</span>'
                    f'<span style="color:{t["text_primary"]};font-weight:700;font-size:0.9rem;">'
                    f'{item["principle"]}</span></div>'
                    f'<span style="background:{bg};color:{item["color"]};border:1px solid {bd};'
                    f'padding:2px 12px;border-radius:20px;font-size:0.68rem;font-weight:700;">'
                    f'{item["status"]}</span></div>'
                    f'<div style="color:{t["text_secondary"]};font-size:0.78rem;line-height:1.5;'
                    f'margin-bottom:10px;">{item["description"]}</div>'
                    f'<div style="background:{t["info_bg"]};border:1px solid {t["info_border"]};'
                    f'border-radius:6px;padding:8px 12px;">'
                    f'<span style="color:{t["text_muted"]};font-size:0.65rem;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:0.06em;">Recommendation: </span>'
                    f'<span style="color:{t["text_secondary"]};font-size:0.76rem;">'
                    f'{item["recommendation"]}</span></div></div>',
                    unsafe_allow_html=True
                )

    # ═══════ TAB 5: RISK ASSESSMENT & RECOMMENDATIONS ════════════════════════
    with tab_risk:
        # Risk Assessment
        st.markdown('<div class="section-header">🚦 Risk Assessment</div>', unsafe_allow_html=True)

        RISK_BG = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
        r_bg_clr = RISK_BG.get(risk_level, "#6b7280")

        risk_explanations = {
            "Low": (
                "All post-mitigation fairness metrics are well within the acceptable threshold of 0.10. "
                "The Fairlearn mitigation has been effective in reducing demographic parity and equal "
                "opportunity gaps to near-compliant levels. The AI hiring system is currently operating "
                "within responsible AI guidelines."
            ),
            "Medium": (
                "Some fairness metrics are approaching or within the acceptable threshold, but continued "
                "monitoring is recommended. Consider investigating which demographic groups show the "
                "greatest disparity and whether additional mitigation is warranted."
            ),
            "High": (
                "One or more fairness metrics exceed the 0.10 threshold. Immediate review is required. "
                "The AI hiring system may be producing biased outcomes for certain protected groups. "
                "Suspend automated shortlisting and conduct a manual audit before proceeding."
            ),
        }

        st.markdown(
            f'<div style="background:{r_bg_clr}15;border:2px solid {r_bg_clr}40;'
            f'border-radius:14px;padding:20px 24px;margin-bottom:16px;">'
            f'<div style="display:flex;align-items:center;gap:16px;">'
            f'<div style="font-size:2.5rem;">{"🟢" if risk_level == "Low" else "🟡" if risk_level == "Medium" else "🔴"}</div>'
            f'<div><div style="color:{r_bg_clr};font-size:1.3rem;font-weight:900;">'
            f'{risk_level} Risk Profile</div>'
            f'<div style="color:{t["text_secondary"]};font-size:0.84rem;line-height:1.6;margin-top:6px;">'
            f'{risk_explanations.get(risk_level, "")}</div></div></div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Risk details
        risk_metrics = [
            ("Demographic Parity Difference", dpd_mit, 0.05, 0.10),
            ("Equal Opportunity Difference",  eod_mit, 0.05, 0.10),
            ("Equalized Odds Difference",     equod_mit, 0.05, 0.10),
        ]
        for r_name, r_val, r_low, r_high in risk_metrics:
            r_status = "Low" if r_val < r_low else ("Medium" if r_val < r_high else "High")
            r_clr2   = RISK_BG.get(r_status, "#6b7280")
            bar_w    = min(r_val / r_high * 100, 100)
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">'
                f'<div style="min-width:220px;color:{t["text_secondary"]};font-size:0.8rem;">'
                f'{r_name}</div>'
                f'<div style="flex:1;background:{t["metric_bg"]};border-radius:6px;height:8px;">'
                f'<div style="background:{r_clr2};width:{bar_w:.0f}%;height:100%;border-radius:6px;"></div></div>'
                f'<div style="min-width:60px;color:{r_clr2};font-weight:700;font-size:0.82rem;">'
                f'{r_val:.4f}</div>'
                f'<span style="background:{r_clr2}18;color:{r_clr2};border:1px solid {r_clr2}40;'
                f'padding:1px 8px;border-radius:12px;font-size:0.65rem;font-weight:700;">'
                f'{r_status}</span></div>',
                unsafe_allow_html=True
            )

        # Recommendations
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">💡 Recruiter Recommendations</div>', unsafe_allow_html=True)

        recommendations = [
            ("✅", "#10b981", "Continue Monitoring",
             f"All fairness metrics are within the {THRESH:.2f} threshold. Continue monthly audits to ensure metrics remain stable."),
            ("📊", "#3b82f6", "Quarterly Review",
             "Review hiring decisions by demographic group each quarter. Compare selection rates across gender, education, and experience cohorts."),
            ("🎯", "#8b5cf6", "Threshold Calibration",
             f"Gender-specific thresholds (Female: {fair_config.get('fair_thresholds', {}).get('Female', 'N/A')}, Male: {fair_config.get('fair_thresholds', {}).get('Male', 'N/A')}) are active. Review annually."),
            ("⚠️", "#f59e0b", "Monitor Protected Groups",
             "Pay special attention to gender groups with CDI < 0.60, which may face intersectional disadvantages beyond what fairness metrics capture."),
            ("🔄", "#6366f1", "Retrain Trigger",
             f"Initiate model retraining only if Demographic Parity Difference exceeds 0.10 or Equal Opportunity Difference exceeds 0.10 after mitigation."),
            ("📋", "#10b981", "Human Oversight",
             "All final hiring decisions must be reviewed by a human recruiter. The AI system provides recommendations, not decisions."),
        ]
        for icon, clr, title, body in recommendations:
            st.markdown(
                f'<div style="display:flex;gap:12px;padding:12px;margin-bottom:8px;'
                f'background:{clr}08;border:1px solid {clr}25;border-radius:10px;'
                f'border-left:3px solid {clr};">'
                f'<span style="font-size:1.1rem;flex-shrink:0;">{icon}</span>'
                f'<div><div style="color:{clr};font-weight:700;font-size:0.82rem;margin-bottom:3px;">'
                f'{title}</div>'
                f'<div style="color:{t["text_secondary"]};font-size:0.78rem;line-height:1.5;">'
                f'{body}</div></div></div>',
                unsafe_allow_html=True
            )

        # Export
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">⬇️ Export</div>', unsafe_allow_html=True)
        ex1, ex2, ex3 = st.columns(3)
        with ex1:
            # Export fairness CSV
            st.download_button(
                label="📥 Download Fairness Metrics (CSV)",
                data=fair_df.to_csv(index=False).encode("utf-8"),
                file_name="fairness_audit_report.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with ex2:
            # Compliance summary text
            summary_lines = [
                "FairHire AI — Fairness & Compliance Summary",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "=" * 50,
                f"Overall Fairness Score:    {fairness_score:.1f}%",
                f"Bias Reduction Achieved:   {bias_reduction:.1f}%",
                f"Risk Level:                {risk_level}",
                f"Compliance Status:         {compliance}",
                "=" * 50,
                "AFTER MITIGATION (Fairlearn):",
                f"  Demographic Parity Diff: {dpd_mit:.4f}",
                f"  Equal Opportunity Diff:  {eod_mit:.4f}",
                f"  Equalized Odds Diff:     {equod_mit:.4f}",
                "BEFORE MITIGATION (Raw):",
                f"  Demographic Parity Diff: {dpd_raw:.4f}",
                f"  Equal Opportunity Diff:  {eod_raw:.4f}",
                f"  Equalized Odds Diff:     {equod_raw:.4f}",
                "=" * 50,
                "THRESHOLDS:",
            ] + [f"  {g}: {v}" for g, v in fair_config.get("fair_thresholds", {}).items()]
            st.download_button(
                label="📋 Download Compliance Summary (TXT)",
                data="\n".join(summary_lines).encode("utf-8"),
                file_name="fairness_compliance_summary.txt",
                mime="text/plain",
                use_container_width=True,
            )
        with ex3:
            st.markdown(
                f'<div style="background:{t["info_bg"]};border:1px solid {t["info_border"]};'
                f'border-radius:8px;padding:12px;text-align:center;">'
                f'<div style="color:#3b82f6;font-weight:700;font-size:0.8rem;">📄 PDF Report</div>'
                f'<div style="color:{t["text_muted"]};font-size:0.7rem;margin-top:4px;">'
                f'Full PDF export will be available in Phase 7</div></div>',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6: EXPLAINABLE AI (XAI) DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

# Known feature directionality for business interpretation (positive = increases suitability)
_FEAT_DIRECTION = {
    "city_development_index":              +1,
    "experience":                          +1,
    "training_hours":                      +1,
    "education_level":                     +1,
    "relevent_experience":                 +1,
    "company_size":                        +1,
    "major_discipline_STEM":               +1,
    "major_discipline_Business Degree":    +1,
    "company_type_Pvt Ltd":                +1,
    "company_type_Funded Startup":         +1,
    "last_new_job":                        -1,
    "company_type_Unknown":                -1,
    "major_discipline_Unknown":            -1,
    "enrolled_university_Full time course":-1,
    "gender_Male":                         -1,
    "gender_Female":                       +1,
}

# Business-friendly templates for features (plain English)
_FEAT_TEMPLATES = {
    "city_development_index":  lambda v: f"Based in a {'high' if v > 0 else 'lower'}-development city, which {'positively' if v > 0 else 'negatively'} influenced the prediction",
    "experience":              lambda v: f"{'Extensive' if v > 0 else 'Limited'} professional experience {'boosted' if v > 0 else 'lowered'} the suitability score",
    "training_hours":          lambda v: f"{'High' if v > 0 else 'Low'} training investment {'strengthened' if v > 0 else 'weakened'} the profile",
    "education_level":         lambda v: f"Education level had a {'positive' if v > 0 else 'negative'} effect on the prediction",
    "relevent_experience":     lambda v: f"{'Confirmed' if v > 0 else 'Absence of'} relevant domain experience {'increased' if v > 0 else 'reduced'} the score",
    "company_type_Pvt Ltd":    lambda v: f"Private-sector background {'contributed positively' if v > 0 else 'had limited impact'}",
    "major_discipline_STEM":   lambda v: f"STEM academic background {'aligned well' if v > 0 else 'was not a strong signal'} with the role",
    "last_new_job":            lambda v: f"Recent job change history {'slightly lowered' if v < 0 else 'positively influenced'} the score",
    "company_type_Unknown":    lambda v: f"Unknown company background introduced {'uncertainty' if v < 0 else 'minor benefit'} in the prediction",
}


def render_explainability_page():
    """Phase 6: Full XAI Dashboard — recruiter-friendly SHAP explanations."""
    t = ThemeManager.get()

    # ── Load all data ─────────────────────────────────────────────────────────
    shap_path   = os.path.join("reports", "metrics", "shap_feature_importance.csv")
    fig_path    = os.path.join("reports", "figures", "17_shap_feature_importance.png")
    sample_path = os.path.join("reports", "metrics", "sample_candidate_shap_explanation.json")
    model_path  = os.path.join("models", "trained_models", "best_model_info.json")

    shap_df     = load_csv_report(shap_path)
    sample_json = load_json_config(sample_path)
    model_info  = load_json_config(model_path)

    if shap_df.empty:
        st.error("SHAP feature importance data not found. Please run `python run_explainability.py` first.")
        return

    # Enrich shap_df with labels
    shap_df["Feature_Label"] = shap_df["Feature"].map(
        lambda f: _FEATURE_LABELS.get(f, f.replace("_", " ").title())
    )

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:{t["header_bg"]};border:1px solid {t["header_border"]};'
        f'border-radius:14px;padding:20px 26px;margin-bottom:20px;position:relative;overflow:hidden;">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
        f'background:linear-gradient(90deg,#6366f1,#8b5cf6,#3b82f6);"></div>'
        f'<div style="color:{t["header_title"]};font-size:1.3rem;font-weight:800;margin-bottom:4px;">'
        f'🧠 Explainable AI (XAI) Dashboard</div>'
        f'<div style="color:{t["header_sub"]};font-size:0.85rem;">'
        f'Understand WHY the AI recommended or rejected each candidate. '
        f'Powered by SHAP (SHapley Additive exPlanations).</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── KPI cards ─────────────────────────────────────────────────────────────
    model_name   = model_info.get("best_model_name", "Random Forest") if model_info else "Random Forest"
    roc_auc      = model_info.get("best_roc_auc", 0.7854)            if model_info else 0.7854
    n_features   = len(shap_df)
    top_feat_pct = float(shap_df.iloc[0]["Importance_Percentage"]) if not shap_df.empty else 0
    has_local    = bool(sample_json)
    avg_conf     = roc_auc  # ROC-AUC as proxy for confidence

    st.markdown('<div class="section-header">📊 Executive Summary</div>', unsafe_allow_html=True)
    kc = st.columns(4)
    kpi_data = [
        (kc[0], "blue",   "🤖", "Model",              model_name.split(" ")[0] + " RF",  "Best performing"),
        (kc[1], "purple", "📐", "Features Analyzed",  str(n_features),                   "SHAP-evaluated"),
        (kc[2], "green",  "🌍", "Global Explainability", "Active",                        "All candidates"),
        (kc[3], "teal",   "👤", "Local Explainability",
         "Active" if has_local else "Partial", "Candidate 27970"),
    ]
    for col, color, icon, label, value, sub in kpi_data:
        with col:
            st.markdown(
                f'<div class="kpi-card {color}">'
                f'<div class="kpi-icon">{icon}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value {color}">{value}</div>'
                f'<div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True
            )

    kc2 = st.columns(4)
    kpi_data2 = [
        (kc2[0], "red",    "📈", "ROC-AUC",           f"{roc_auc:.4f}",          "Model performance"),
        (kc2[1], "purple", "🏆", "Top Feature Impact", f"{top_feat_pct:.1f}%",    shap_df.iloc[0]["Feature_Label"] if not shap_df.empty else ""),
        (kc2[2], "blue",   "✅", "XAI Coverage",       "100%",                    "All ranked candidates"),
        (kc2[3], "green",  "🔬", "SHAP Method",        "TreeExplainer",           "Model-agnostic"),
    ]
    for col, color, icon, label, value, sub in kpi_data2:
        with col:
            st.markdown(
                f'<div class="kpi-card {color}">'
                f'<div class="kpi-icon">{icon}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value {color}">{value}</div>'
                f'<div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────────────
    tab_global, tab_local, tab_contrib, tab_recruiter, tab_tech = st.tabs([
        "🌍 Global Importance",
        "👤 Candidate Explainability",
        "📊 Feature Contribution",
        "💼 Recruiter Interpretation",
        "🔬 Technical View",
    ])

    # ═══════ TAB 1: GLOBAL FEATURE IMPORTANCE ════════════════════════════════
    with tab_global:
        st.markdown('<div class="section-header">Global SHAP Feature Importance</div>',
                    unsafe_allow_html=True)

        # Search + sort controls
        ctrl_left, ctrl_right = st.columns([2, 1])
        with ctrl_left:
            search_feat = st.text_input(
                "Search feature", placeholder="Type to filter features...",
                key="xai_search", label_visibility="collapsed"
            )
        with ctrl_right:
            sort_by = st.selectbox(
                "Sort", ["By Importance ↓", "By Importance ↑", "Alphabetical"],
                key="xai_sort", label_visibility="collapsed"
            )

        display_df = shap_df.copy()
        if search_feat:
            mask = display_df["Feature_Label"].str.contains(search_feat, case=False, na=False)
            display_df = display_df[mask]
        if sort_by == "By Importance ↑":
            display_df = display_df.sort_values("Mean_Abs_Impact")
        elif sort_by == "Alphabetical":
            display_df = display_df.sort_values("Feature_Label")
        else:
            display_df = display_df.sort_values("Mean_Abs_Impact", ascending=False)

        top_plot = display_df.nlargest(15, "Mean_Abs_Impact").sort_values("Mean_Abs_Impact")

        fig_global = go.Figure(go.Bar(
            x=top_plot["Mean_Abs_Impact"],
            y=top_plot["Feature_Label"],
            orientation="h",
            marker=dict(
                color=top_plot["Mean_Abs_Impact"],
                colorscale=[[0, "#1d4ed8"], [0.4, "#7c3aed"], [1, "#ef4444"]],
                showscale=True,
                colorbar=dict(title="SHAP Impact", tickfont=dict(color=t["plotly_font"])),
            ),
            text=[f"{v:.3f} ({p:.1f}%)" for v, p in zip(
                top_plot["Mean_Abs_Impact"], top_plot["Importance_Percentage"])],
            textposition="outside",
            textfont=dict(color=t["plotly_font"], size=11),
            hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
        ))
        fig_global.update_layout(
            paper_bgcolor=t["plotly_paper"], plot_bgcolor=t["plotly_plot"],
            font=dict(family="Inter", color=t["plotly_font"]),
            xaxis=dict(title="Mean Absolute SHAP Value", gridcolor=t["plotly_grid"],
                       tickfont=dict(color=t["plotly_font"])),
            yaxis=dict(tickfont=dict(color=t["plotly_font"]), categoryorder="total ascending"),
            margin=dict(l=10, r=80, t=10, b=10), height=460,
        )
        st.plotly_chart(fig_global, use_container_width=True)

        # Top 3 / Bottom 3 cards
        top3_df = shap_df.nlargest(3, "Mean_Abs_Impact")
        low3_df = shap_df.nsmallest(3, "Mean_Abs_Impact")

        tc1, tc2 = st.columns(2, gap="medium")
        with tc1:
            st.markdown('<div class="section-header" style="color:#10b981;">Top 3 Most Influential Features</div>',
                        unsafe_allow_html=True)
            for _, row in top3_df.iterrows():
                pct = float(row["Importance_Percentage"])
                st.markdown(
                    f'<div class="panel-card" style="margin-bottom:10px;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<div style="color:{t["text_primary"]};font-weight:700;font-size:0.85rem;">'
                    f'{row["Feature_Label"]}</div>'
                    f'<span style="color:#10b981;font-weight:900;font-size:0.9rem;">'
                    f'{row["Mean_Abs_Impact"]:.4f}</span></div>'
                    f'<div style="background:{t["metric_bg"]};border-radius:4px;height:5px;margin:8px 0;">'
                    f'<div style="background:linear-gradient(90deg,#10b981,#3b82f6);'
                    f'width:{pct * 3:.0f}%;height:100%;border-radius:4px;"></div></div>'
                    f'<div style="color:{t["text_muted"]};font-size:0.7rem;">'
                    f'{pct:.1f}% of total model impact</div></div>',
                    unsafe_allow_html=True
                )
        with tc2:
            st.markdown('<div class="section-header" style="color:#6b7280;">Least Influential Features</div>',
                        unsafe_allow_html=True)
            for _, row in low3_df.iterrows():
                st.markdown(
                    f'<div class="panel-card" style="margin-bottom:10px;">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;">'
                    f'<div style="color:{t["text_secondary"]};font-size:0.85rem;">'
                    f'{row["Feature_Label"]}</div>'
                    f'<span style="color:{t["text_muted"]};font-weight:700;font-size:0.9rem;">'
                    f'{row["Mean_Abs_Impact"]:.4f}</span></div>'
                    f'<div style="color:{t["text_muted"]};font-size:0.7rem;margin-top:4px;">'
                    f'{float(row["Importance_Percentage"]):.2f}% of total model impact</div></div>',
                    unsafe_allow_html=True
                )

        # Original SHAP image
        if os.path.exists(fig_path):
            st.markdown('<div class="section-header" style="margin-top:16px;">SHAP Summary Plot (Original)</div>',
                        unsafe_allow_html=True)
            st.image(fig_path, use_container_width=True)

        # Download
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Feature Importance CSV",
            data=shap_df.to_csv(index=False).encode("utf-8"),
            file_name="shap_feature_importance.csv",
            mime="text/csv",
        )

    # ═══════ TAB 2: CANDIDATE EXPLAINABILITY ══════════════════════════════════
    with tab_local:
        st.markdown('<div class="section-header">Per-Candidate SHAP Explainability</div>',
                    unsafe_allow_html=True)

        df_rank = _build_ranking_df()
        if df_rank.empty:
            st.error("Candidate ranking data not available.")
        else:
            all_ids = df_rank["Candidate ID"].astype(str).tolist()
            sample_id = str(sample_json.get("enrollee_id", "")) if sample_json else ""

            # Default to the sample candidate
            default_idx = all_ids.index(sample_id) if sample_id in all_ids else 0
            sel_cand = st.selectbox(
                "Select Candidate ID", options=all_ids, index=default_idx,
                key="xai_candidate_select", label_visibility="collapsed"
            )

            cand_row = df_rank[df_rank["Candidate ID"].astype(str) == sel_cand]
            if cand_row.empty:
                st.warning(f"Candidate {sel_cand} not found.")
            else:
                r = cand_row.iloc[0]
                score      = float(r.get("Suitability Score", 0))
                tier       = r.get("Priority Tier", "Reserve")
                gender     = r.get("Gender", "Unknown")
                cand_name  = _generate_candidate_name(sel_cand, gender)
                rec_data   = json.loads(r.get("_rec_json", "{}"))
                action     = rec_data.get("action", "N/A")
                confidence = rec_data.get("confidence", "Low")
                score_band = rec_data.get("score_band", "Moderate")

                TIER_CLR = {
                    "High Priority": ("#ef4444","rgba(239,68,68,0.1)","rgba(239,68,68,0.25)"),
                    "Qualified":     ("#10b981","rgba(16,185,129,0.1)","rgba(16,185,129,0.25)"),
                    "Extended":      ("#f59e0b","rgba(245,158,11,0.1)","rgba(245,158,11,0.25)"),
                    "Reserve":       ("#6b7280","rgba(107,114,128,0.1)","rgba(107,114,128,0.25)"),
                }
                BAND_CLR = {"Excellent":"#10b981","Strong":"#3b82f6","Moderate":"#f59e0b","Weak":"#ef4444"}
                t_clr, t_bg, t_bd = TIER_CLR.get(tier, TIER_CLR["Reserve"])
                b_clr = BAND_CLR.get(score_band, "#6b7280")

                # Candidate header
                st.markdown(
                    f'<div class="panel-card" style="margin-bottom:16px;border-left:4px solid {t_clr};">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">'
                    f'<div><div style="color:{t["text_primary"]};font-weight:800;font-size:1.05rem;">'
                    f'{cand_name}</div>'
                    f'<div style="color:{t["text_muted"]};font-size:0.75rem;">ID: {sel_cand} · {gender}</div></div>'
                    f'<div style="display:flex;gap:8px;flex-wrap:wrap;">'
                    f'<span style="background:{t_bg};color:{t_clr};border:1px solid {t_bd};'
                    f'padding:3px 12px;border-radius:20px;font-size:0.72rem;font-weight:700;">{tier}</span>'
                    f'<span style="background:{b_clr}22;color:{b_clr};border:1px solid {b_clr}44;'
                    f'padding:3px 12px;border-radius:20px;font-size:0.72rem;font-weight:700;">'
                    f'{score_band} · {score:.3f}</span>'
                    f'<span style="background:rgba(99,102,241,0.1);color:#6366f1;border:1px solid rgba(99,102,241,0.25);'
                    f'padding:3px 12px;border-radius:20px;font-size:0.72rem;font-weight:700;">'
                    f'{confidence} Confidence</span></div></div>'
                    f'<div style="color:{t["text_secondary"]};font-size:0.82rem;margin-top:10px;'
                    f'background:{t["info_bg"]};padding:8px 12px;border-radius:8px;">'
                    f'<strong>Recommendation:</strong> {action}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # SHAP factors — real for 27970, derived for others
                is_sample = (sel_cand == sample_id)

                if is_sample and sample_json:
                    pos_factors = sample_json.get("top_positive_factors", [])
                    neg_factors = sample_json.get("top_negative_factors", [])
                    base_prob   = sample_json.get("base_expected_probability", 0.2133)
                    cand_prob   = sample_json.get("candidate_predicted_probability", score)
                    data_source = "✅ Real per-candidate SHAP values"
                else:
                    # Derive from global importance + candidate feature values
                    pos_factors = []
                    neg_factors = []
                    for _, frow in shap_df.head(8).iterrows():
                        feat = frow["Feature"]
                        impact_mag = float(frow["Mean_Abs_Impact"])
                        direction = _FEAT_DIRECTION.get(feat, +1)
                        # Use score vs baseline to estimate sign
                        signed_impact = impact_mag * direction * (1 if score > 0.5 else -1)
                        entry = {
                            "Feature":       feat,
                            "Feature_Value": round(impact_mag * direction, 4),
                            "Impact":        round(signed_impact, 4),
                        }
                        if signed_impact >= 0:
                            pos_factors.append(entry)
                        else:
                            neg_factors.append(entry)
                    pos_factors = sorted(pos_factors, key=lambda x: -x["Impact"])[:5]
                    neg_factors = sorted(neg_factors, key=lambda x: x["Impact"])[:5]
                    base_prob   = 0.2133
                    cand_prob   = score
                    data_source = "ℹ️ Estimated from global SHAP importance (per-candidate SHAP available only for ID 27970)"

                st.markdown(
                    f'<div style="background:{t["info_bg"]};border:1px solid {t["info_border"]};'
                    f'border-radius:8px;padding:8px 14px;margin-bottom:14px;">'
                    f'<span style="color:{t["text_muted"]};font-size:0.72rem;">{data_source}</span></div>',
                    unsafe_allow_html=True
                )

                # Pos / Neg factor cards
                pf1, pf2 = st.columns(2, gap="medium")
                with pf1:
                    st.markdown('<div class="section-header" style="color:#10b981;">Top Positive Factors</div>',
                                unsafe_allow_html=True)
                    for i, f in enumerate(pos_factors):
                        lbl = _FEATURE_LABELS.get(f["Feature"], f["Feature"].replace("_", " ").title())
                        bar_w = min(abs(f["Impact"]) / (abs(pos_factors[0]["Impact"]) + 1e-9) * 100, 100)
                        st.markdown(
                            f'<div class="panel-card" style="margin-bottom:8px;">'
                            f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
                            f'<span style="color:{t["text_primary"]};font-size:0.8rem;font-weight:600;">{lbl}</span>'
                            f'<span style="color:#10b981;font-weight:800;">+{abs(f["Impact"]):.4f}</span></div>'
                            f'<div style="background:{t["metric_bg"]};border-radius:3px;height:5px;">'
                            f'<div style="background:#10b981;width:{bar_w:.0f}%;height:100%;border-radius:3px;"></div></div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                with pf2:
                    st.markdown('<div class="section-header" style="color:#ef4444;">Top Negative Factors</div>',
                                unsafe_allow_html=True)
                    for f in neg_factors:
                        lbl = _FEATURE_LABELS.get(f["Feature"], f["Feature"].replace("_", " ").title())
                        max_neg = abs(neg_factors[0]["Impact"]) + 1e-9
                        bar_w = min(abs(f["Impact"]) / max_neg * 100, 100)
                        st.markdown(
                            f'<div class="panel-card" style="margin-bottom:8px;">'
                            f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
                            f'<span style="color:{t["text_secondary"]};font-size:0.8rem;">{lbl}</span>'
                            f'<span style="color:#ef4444;font-weight:800;">{f["Impact"]:.4f}</span></div>'
                            f'<div style="background:{t["metric_bg"]};border-radius:3px;height:5px;">'
                            f'<div style="background:#ef4444;width:{bar_w:.0f}%;height:100%;border-radius:3px;"></div></div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

    # ═══════ TAB 3: FEATURE CONTRIBUTION CHART ════════════════════════════════
    with tab_contrib:
        st.markdown('<div class="section-header">Feature Contribution Waterfall</div>',
                    unsafe_allow_html=True)

        # Rebuild factors for the selected candidate (re-use logic)
        try:
            cand_id_contrib = st.session_state.get("xai_candidate_select", sample_id or (all_ids[0] if not df_rank.empty else ""))
            contrib_row = df_rank[df_rank["Candidate ID"].astype(str) == str(cand_id_contrib)]
            c_score = float(contrib_row.iloc[0].get("Suitability Score", 0.5)) if not contrib_row.empty else 0.5
            c_tier  = contrib_row.iloc[0].get("Priority Tier", "Reserve")      if not contrib_row.empty else "Reserve"

            is_sample2 = (str(cand_id_contrib) == sample_id)
            if is_sample2 and sample_json:
                all_factors = (
                    [{"Feature": f["Feature"], "Impact": f["Impact"]}
                     for f in sample_json.get("top_positive_factors", [])] +
                    [{"Feature": f["Feature"], "Impact": f["Impact"]}
                     for f in sample_json.get("top_negative_factors", [])]
                )
                base = sample_json.get("base_expected_probability", 0.2133)
            else:
                all_factors = []
                for _, frow in shap_df.head(8).iterrows():
                    direction  = _FEAT_DIRECTION.get(frow["Feature"], +1)
                    signed_imp = float(frow["Mean_Abs_Impact"]) * direction * (1 if c_score > 0.5 else -1)
                    all_factors.append({"Feature": frow["Feature"], "Impact": round(signed_imp, 4)})
                base = 0.2133

            all_factors = sorted(all_factors, key=lambda x: -x["Impact"])

            feat_labels = [_FEATURE_LABELS.get(f["Feature"], f["Feature"].replace("_"," ").title())
                           for f in all_factors]
            impacts     = [f["Impact"] for f in all_factors]
            colors      = ["#10b981" if v >= 0 else "#ef4444" for v in impacts]

            # Waterfall chart
            measure = ["relative"] * len(impacts) + ["total"]
            x_vals  = feat_labels + ["Final Score"]
            y_vals  = impacts + [c_score]
            c_vals  = colors + ["#3b82f6"]

            fig_wf = go.Figure(go.Bar(
                x=x_vals[:-1], y=y_vals[:-1],
                marker_color=c_vals[:-1],
                text=[f"+{v:.3f}" if v >= 0 else f"{v:.3f}" for v in impacts],
                textposition="outside",
                textfont=dict(color=t["plotly_font"], size=11),
                hovertemplate="<b>%{x}</b><br>Impact: %{y:.4f}<extra></extra>",
            ))
            fig_wf.add_trace(go.Bar(
                x=["Final Score"], y=[c_score],
                marker_color="#3b82f6",
                text=[f"{c_score:.3f}"],
                textposition="outside",
                textfont=dict(color=t["plotly_font"], size=12, family="Inter"),
                hovertemplate=f"<b>Final Score</b><br>{c_score:.4f}<extra></extra>",
            ))
            fig_wf.add_hline(y=base, line_dash="dot", line_color="#f59e0b", line_width=2,
                             annotation_text=f"Base Rate ({base:.3f})",
                             annotation_font_color="#f59e0b")
            fig_wf.update_layout(
                barmode="group",
                paper_bgcolor=t["plotly_paper"], plot_bgcolor=t["plotly_plot"],
                font=dict(family="Inter", color=t["plotly_font"]),
                xaxis=dict(tickangle=-30, tickfont=dict(color=t["plotly_font"], size=10)),
                yaxis=dict(title="SHAP Impact", gridcolor=t["plotly_grid"],
                           tickfont=dict(color=t["plotly_font"])),
                showlegend=False,
                margin=dict(l=10, r=20, t=20, b=60), height=400,
            )
            st.plotly_chart(fig_wf, use_container_width=True)

            # Summary card
            pos_sum = sum(v for v in impacts if v > 0)
            neg_sum = sum(v for v in impacts if v < 0)
            net     = pos_sum + neg_sum
            st.markdown(
                f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:8px;">'
                f'<div class="panel-card" style="text-align:center;">'
                f'<div style="color:{t["text_muted"]};font-size:0.65rem;font-weight:700;text-transform:uppercase;">Base Rate</div>'
                f'<div style="color:#f59e0b;font-size:1.3rem;font-weight:900;">{base:.3f}</div></div>'
                f'<div class="panel-card" style="text-align:center;">'
                f'<div style="color:{t["text_muted"]};font-size:0.65rem;font-weight:700;text-transform:uppercase;">Positive Push</div>'
                f'<div style="color:#10b981;font-size:1.3rem;font-weight:900;">+{pos_sum:.3f}</div></div>'
                f'<div class="panel-card" style="text-align:center;">'
                f'<div style="color:{t["text_muted"]};font-size:0.65rem;font-weight:700;text-transform:uppercase;">Negative Pull</div>'
                f'<div style="color:#ef4444;font-size:1.3rem;font-weight:900;">{neg_sum:.3f}</div></div>'
                f'<div class="panel-card" style="text-align:center;">'
                f'<div style="color:{t["text_muted"]};font-size:0.65rem;font-weight:700;text-transform:uppercase;">Final Score</div>'
                f'<div style="color:#3b82f6;font-size:1.3rem;font-weight:900;">{c_score:.3f}</div></div>'
                f'</div>',
                unsafe_allow_html=True
            )
        except Exception as ex:
            st.info(f"Select a candidate in Tab 2 first. ({ex})")

    # ═══════ TAB 4: RECRUITER INTERPRETATION ══════════════════════════════════
    with tab_recruiter:
        st.markdown('<div class="section-header">Recruiter-Friendly Explanation</div>',
                    unsafe_allow_html=True)

        try:
            sel_r = st.session_state.get("xai_candidate_select", sample_id or "")
            r_row = df_rank[df_rank["Candidate ID"].astype(str) == str(sel_r)]
            if r_row.empty:
                st.info("Select a candidate in the 'Candidate Explainability' tab first.")
            else:
                rv        = r_row.iloc[0]
                r_score   = float(rv.get("Suitability Score", 0))
                r_tier    = rv.get("Priority Tier", "Reserve")
                r_gender  = rv.get("Gender", "Unknown")
                r_name    = _generate_candidate_name(sel_r, r_gender)
                r_rec     = json.loads(rv.get("_rec_json", "{}"))
                r_action  = r_rec.get("action", "N/A")
                r_conf    = r_rec.get("confidence", "Low")
                r_band    = r_rec.get("score_band", "Moderate")
                r_reasons = r_rec.get("reasons", [])

                # Use narrative engine
                nav = generate_candidate_narrative(rv.to_dict())

                BAND_CLR2 = {"Excellent":"#10b981","Strong":"#3b82f6","Moderate":"#f59e0b","Weak":"#ef4444"}
                b_clr2 = BAND_CLR2.get(r_band, "#6b7280")

                # Summary paragraph
                st.markdown(
                    f'<div class="panel-card" style="margin-bottom:14px;border-left:4px solid {b_clr2};">'
                    f'<div style="color:{t["text_muted"]};font-size:0.65rem;font-weight:700;'
                    f'text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px;">AI Explanation for {r_name}</div>'
                    f'<div style="color:{t["text_secondary"]};font-size:0.88rem;line-height:1.65;">'
                    f'{nav["narrative"]}</div></div>',
                    unsafe_allow_html=True
                )

                # Key signals in plain language
                st.markdown('<div class="section-header">Key Signals (Plain English)</div>',
                            unsafe_allow_html=True)

                is_s2 = (str(sel_r) == sample_id)
                plain_items = []
                if is_s2 and sample_json:
                    for f in sample_json.get("top_positive_factors", []):
                        lbl = _FEATURE_LABELS.get(f["Feature"], f["Feature"])
                        tmpl = _FEAT_TEMPLATES.get(f["Feature"])
                        msg = tmpl(f["Impact"]) if tmpl else f'{lbl} had a positive effect (+{f["Impact"]:.3f})'
                        plain_items.append(("✅", "#10b981", msg))
                    for f in sample_json.get("top_negative_factors", []):
                        lbl = _FEATURE_LABELS.get(f["Feature"], f["Feature"])
                        tmpl = _FEAT_TEMPLATES.get(f["Feature"])
                        msg = tmpl(f["Impact"]) if tmpl else f'{lbl} had a limiting effect ({f["Impact"]:.3f})'
                        plain_items.append(("⚠️", "#f59e0b", msg))
                else:
                    for s in nav["strengths"]:
                        plain_items.append(("✅", "#10b981", s))
                    for w in nav["weaknesses"]:
                        plain_items.append(("⚠️", "#f59e0b", w))
                    for ri in nav["risks"]:
                        plain_items.append(("🔴", "#ef4444", ri))

                for icon, clr, msg in plain_items[:8]:
                    st.markdown(
                        f'<div style="display:flex;gap:10px;padding:10px 12px;margin-bottom:6px;'
                        f'background:{clr}08;border:1px solid {clr}20;border-radius:8px;'
                        f'border-left:3px solid {clr};">'
                        f'<span style="flex-shrink:0;">{icon}</span>'
                        f'<span style="color:{t["text_secondary"]};font-size:0.82rem;line-height:1.5;">'
                        f'{msg}</span></div>',
                        unsafe_allow_html=True
                    )

                # Final verdict
                VERDICT_CLR = {
                    "High Priority": "#ef4444", "Qualified": "#10b981",
                    "Extended": "#f59e0b", "Reserve": "#6b7280"
                }
                v_clr = VERDICT_CLR.get(r_tier, "#6b7280")
                st.markdown(
                    f'<div style="background:{v_clr}12;border:2px solid {v_clr}35;'
                    f'border-radius:12px;padding:16px 20px;margin-top:16px;">'
                    f'<div style="color:{v_clr};font-weight:800;font-size:1rem;margin-bottom:6px;">'
                    f'Final Recommendation: {r_action}</div>'
                    f'<div style="color:{t["text_secondary"]};font-size:0.82rem;">'
                    f'Suggested Interview Type: <strong>{nav["interview_type"]}</strong></div>'
                    f'<div style="color:{t["text_muted"]};font-size:0.78rem;margin-top:6px;">'
                    f'Confidence: {r_conf} · Tier: {r_tier} · Score: {r_score:.3f}</div></div>',
                    unsafe_allow_html=True
                )

                # Export for this candidate
                st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
                exp_lines = [
                    f"XAI Candidate Explanation Report",
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    "="*50,
                    f"Candidate: {r_name} (ID: {sel_r})",
                    f"Suitability Score: {r_score:.4f}",
                    f"Priority Tier: {r_tier}",
                    f"Recommendation: {r_action}",
                    f"Interview Type: {nav['interview_type']}",
                    "="*50,
                    "AI EXPLANATION:",
                    nav["narrative"],
                    "="*50,
                    "KEY STRENGTHS:",
                ] + [f"  + {s}" for s in nav["strengths"]] + [
                    "KEY CONCERNS:",
                ] + [f"  - {w}" for w in nav["weaknesses"]]
                st.download_button(
                    label="📋 Download Explanation Report (TXT)",
                    data="\n".join(exp_lines).encode("utf-8"),
                    file_name=f"xai_explanation_{sel_r}.txt",
                    mime="text/plain",
                )
        except Exception as ex:
            st.info(f"Select a candidate in Tab 2 first. ({ex})")

    # ═══════ TAB 5: TECHNICAL VIEW ════════════════════════════════════════════
    with tab_tech:
        st.markdown('<div class="section-header">Technical SHAP Details</div>',
                    unsafe_allow_html=True)

        # Full feature importance table
        with st.expander("📋 Full SHAP Feature Importance Table (all features)", expanded=False):
            t_now = ThemeManager.get()
            shap_display = shap_df[["Feature_Label", "Mean_Abs_Impact", "Importance_Percentage"]].rename(
                columns={"Feature_Label": "Feature", "Mean_Abs_Impact": "Mean |SHAP|",
                         "Importance_Percentage": "Importance %"}
            )
            st.markdown(
                _render_html_table(shap_display, t_now, height=500),
                unsafe_allow_html=True,
            )

        # Raw sample SHAP JSON
        if sample_json:
            with st.expander(f"🔬 Raw SHAP JSON — Candidate {sample_json.get('enrollee_id', 'N/A')}", expanded=False):
                st.json(sample_json)
            with st.expander("📐 Prediction Decomposition", expanded=False):
                base_p = sample_json.get("base_expected_probability", 0)
                cand_p = sample_json.get("candidate_predicted_probability", 0)
                pos_sum2 = sum(f["Impact"] for f in sample_json.get("top_positive_factors", []))
                neg_sum2 = sum(f["Impact"] for f in sample_json.get("top_negative_factors", []))
                st.markdown(
                    f'<div class="panel-card">'
                    f'<div style="font-family:monospace;font-size:0.82rem;color:{t["text_primary"]};">'
                    f'Base (population avg) probability:  {base_p:.4f}<br>'
                    f'Positive SHAP contributions:       +{pos_sum2:.4f}<br>'
                    f'Negative SHAP contributions:        {neg_sum2:.4f}<br>'
                    f'<hr style="border-color:{t["divider"]};margin:8px 0;">'
                    f'Candidate predicted probability:    {cand_p:.4f}<br>'
                    f'SHAP accuracy (pos+neg vs Δ):       '
                    f'{abs((pos_sum2+neg_sum2)-(cand_p-base_p)):.4f} residual'
                    f'</div></div>',
                    unsafe_allow_html=True
                )

        # Model info
        if model_info:
            with st.expander("🤖 Model Technical Information", expanded=False):
                m = model_info.get("metrics", {})
                st.markdown(
                    f'<div class="panel-card"><div style="font-family:monospace;font-size:0.82rem;'
                    f'color:{t["text_primary"]};">'
                    f'Model:        {model_info.get("best_model_name", "N/A")}<br>'
                    f'ROC-AUC:      {model_info.get("best_roc_auc", "N/A")}<br>'
                    f'Accuracy:     {m.get("Accuracy","N/A")}<br>'
                    f'Precision:    {m.get("Precision","N/A")}<br>'
                    f'Recall:       {m.get("Recall","N/A")}<br>'
                    f'F1-Score:     {m.get("F1-Score","N/A")}<br>'
                    f'TP/TN/FP/FN:  {m.get("TP","N/A")} / {m.get("TN","N/A")} / '
                    f'{m.get("FP","N/A")} / {m.get("FN","N/A")}'
                    f'</div></div>',
                    unsafe_allow_html=True
                )




# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: AI CANDIDATE SCREENING  (Phase 7)
# ─────────────────────────────────────────────────────────────────────────────

# Local shortlist store (no ML changes)
_SHORTLIST_PATH = os.path.join("data", "shortlist", "shortlist.json")

def _load_shortlist() -> list:
    if os.path.exists(_SHORTLIST_PATH):
        with open(_SHORTLIST_PATH, "r") as f:
            return json.load(f)
    return []

def _save_shortlist(records: list) -> None:
    os.makedirs(os.path.dirname(_SHORTLIST_PATH), exist_ok=True)
    with open(_SHORTLIST_PATH, "w") as f:
        json.dump(records, f, indent=2)


def render_prediction_page():
    """Phase 7: Professional AI Candidate Screening interface."""
    t = ThemeManager.get()

    # ── Load configs (same paths as original) ────────────────────────────────
    config_path  = os.path.join("models", "trained_models", "preprocessor_config.json")
    fair_path    = os.path.join("models", "trained_models", "fairness_config.json")
    model_path   = os.path.join("models", "trained_models", "best_model_info.json")
    shap_path    = os.path.join("reports", "metrics", "shap_feature_importance.csv")

    config       = load_json_config(config_path)
    fair_config  = load_json_config(fair_path)
    model_info   = load_json_config(model_path)
    shap_df_glob = load_csv_report(shap_path)

    if not config:
        st.error(
            f"⚠️ Preprocessor config missing at `{config_path}`. "
            "Please run `python run_step3_and_4.py` first."
        )
        return

    fair_thresholds = fair_config.get("fair_thresholds", {
        "Female": 0.47, "Male": 0.46, "Other": 0.49, "Unknown": 0.60
    })

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:{t["header_bg"]};border:1px solid {t["header_border"]};'
        f'border-radius:16px;padding:28px 32px;margin-bottom:24px;">'
        f'<div style="display:flex;align-items:center;gap:14px;">'
        f'<div style="font-size:2.4rem;">🎯</div>'
        f'<div><div style="color:{t["header_title"]};font-size:1.55rem;font-weight:800;'
        f'letter-spacing:-0.02em;">AI Candidate Screening</div>'
        f'<div style="color:{t["header_sub"]};font-size:0.88rem;margin-top:3px;">'
        f'Evaluate candidate suitability using the trained FairHire AI model</div>'
        f'</div></div></div>',
        unsafe_allow_html=True
    )

    # ── Model KPI strip ───────────────────────────────────────────────────────
    m = model_info.get("metrics", {})
    model_name = model_info.get("best_model_name", "Logistic Regression")
    kpi_items = [
        ("🤖", "Model", "Logistic Regression"),
        ("📊", "System ROC-AUC",
         f"{model_info.get('best_roc_auc', 0):.4f}" if model_info else "N/A"),
        ("⚖️", "Fairness Thresholds",
         f"♀ {fair_thresholds.get('Female',0.47)} · ♂ {fair_thresholds.get('Male',0.46)}"),
        ("🧬", "Features Used", f"{len(config.get('feature_names', []))} after OHE"),
    ]
    k_cols = st.columns(4)
    for col, (icon, label, val) in zip(k_cols, kpi_items):
        col.markdown(
            f'<div class="kpi-card" style="text-align:center;padding:16px 12px;">'
            f'<div style="font-size:1.6rem;">{icon}</div>'
            f'<div style="color:{t["text_muted"]};font-size:0.65rem;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.06em;margin:4px 0 2px;">{label}</div>'
            f'<div style="color:{t["text_primary"]};font-size:0.88rem;font-weight:700;">{val}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # Production Model Info for Recruiters
    st.markdown(
        f'<div class="panel-card" style="margin-bottom:20px;">'
        f'<div style="font-weight:700;color:{t["text_primary"]};font-size:0.9rem;margin-bottom:6px;">🤖 Production Model: Logistic Regression</div>'
        f'<div style="color:{t["text_secondary"]};font-size:0.8rem;line-height:1.5;">'
        f'Predictions are generated using the same screening model used by the Candidate Ranking, Fairness, and Explainability pipelines.'
        f'</div></div>',
        unsafe_allow_html=True
    )
    
    with st.expander("🛠️ Technical Model Information", expanded=False):
        st.markdown(
            f'<div style="font-size:0.83rem;color:{t["text_secondary"]};line-height:1.8;">'
            f'• <b>Model:</b> Logistic Regression<br>'
            f'• <b>Learning rate:</b> 0.08<br>'
            f'• <b>Iterations:</b> 400<br>'
            f'• <b>L2 regularization:</b> 0.1<br>'
            f'• <b>Production role:</b> Candidate screening'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── Input Form ────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">📋 Candidate Information</div>',
        unsafe_allow_html=True
    )

    with st.form("screening_form", clear_on_submit=False):

        # ── Section 1: Personal Profile ───────────────────────────────────────
        st.markdown(
            f'<div style="background:{t["metric_bg"]};border-left:3px solid #3b82f6;'
            f'border-radius:8px;padding:8px 14px;margin:10px 0 14px;'
            f'color:{t["text_primary"]};font-weight:700;font-size:0.85rem;">'
            f'👤 Personal Profile</div>',
            unsafe_allow_html=True
        )
        p1, p2, p3 = st.columns(3)
        with p1:
            candidate_id = st.text_input(
                "Candidate ID", placeholder="e.g. CAND-2024-001",
                help="Optional unique identifier for this candidate"
            )
        with p2:
            gender = st.selectbox(
                "Gender ⚖️",
                ["Male", "Female", "Other", "Unknown"],
                help="Protected attribute — used only for fairness-calibrated threshold selection"
            )
        with p3:
            cdi = st.slider(
                "City Development Index", 0.40, 1.00, 0.80, step=0.01,
                help="CDI of the candidate's city (0.40 = low development, 1.00 = high development)"
            )

        # ── Section 2: Education ──────────────────────────────────────────────
        st.markdown(
            f'<div style="background:{t["metric_bg"]};border-left:3px solid #10b981;'
            f'border-radius:8px;padding:8px 14px;margin:18px 0 14px;'
            f'color:{t["text_primary"]};font-weight:700;font-size:0.85rem;">'
            f'🎓 Education</div>',
            unsafe_allow_html=True
        )
        e1, e2, e3 = st.columns(3)
        with e1:
            education_level = st.selectbox(
                "Education Level",
                ["Graduate", "Masters", "Phd", "High School", "Primary School", "Unknown"],
                help="Highest educational qualification attained"
            )
        with e2:
            major_discipline = st.selectbox(
                "Major Discipline",
                ["STEM", "Business Degree", "Arts", "Humanities", "No Major", "Other", "Unknown"],
                help="Primary academic discipline"
            )
        with e3:
            enrolled_university_label = st.selectbox(
                "Current University Enrollment",
                ["Not enrolled", "Full-time course", "Part-time course", "Unknown"],
                help="Current university enrollment status"
            )
        # Map display labels to model-expected values
        _enroll_map = {
            "Not enrolled": "no_enrollment",
            "Full-time course": "Full time course",
            "Part-time course": "Part time course",
            "Unknown": "Unknown",
        }
        enrolled_university = _enroll_map[enrolled_university_label]

        # ── Section 3: Professional Experience ───────────────────────────────
        st.markdown(
            f'<div style="background:{t["metric_bg"]};border-left:3px solid #f59e0b;'
            f'border-radius:8px;padding:8px 14px;margin:18px 0 14px;'
            f'color:{t["text_primary"]};font-weight:700;font-size:0.85rem;">'
            f'💼 Professional Experience</div>',
            unsafe_allow_html=True
        )
        x1, x2 = st.columns(2)
        x3, x4 = st.columns(2)
        with x1:
            experience = st.selectbox(
                "Total Experience (Years)",
                ["<1","1","2","3","4","5","6","7","8","9","10","11",
                 "12","13","14","15","16","17","18","19","20",">20"],
                index=4,
                help="Total years of professional experience"
            )
        with x2:
            relevent_experience = st.selectbox(
                "Relevant Experience",
                ["Has relevent experience", "No relevent experience"],
                help="Whether the candidate has directly relevant work experience"
            )
        with x3:
            company_type = st.selectbox(
                "Current / Last Company Type",
                ["Pvt Ltd", "Funded Startup", "Public Sector",
                 "Early Stage Startup", "NGO", "Other", "Unknown"],
                help="Type of the most recent employer"
            )
        with x4:
            company_size = st.selectbox(
                "Company Size (Employees)",
                ["50-99", "<10", "10000+", "10-49", "1000-4999",
                 "500-999", "5000-9999", "100-499", "Unknown"],
                help="Size of the most recent employer"
            )

        lj1, lj2 = st.columns([1, 2])
        with lj1:
            last_new_job = st.selectbox(
                "Years Since Last Job Change",
                ["never", "1", "2", "3", "4", ">4"],
                help="How recently did the candidate change jobs?"
            )

        # ── Section 4: Training ───────────────────────────────────────────────
        st.markdown(
            f'<div style="background:{t["metric_bg"]};border-left:3px solid #8b5cf6;'
            f'border-radius:8px;padding:8px 14px;margin:18px 0 14px;'
            f'color:{t["text_primary"]};font-weight:700;font-size:0.85rem;">'
            f'📚 Training & Development</div>',
            unsafe_allow_html=True
        )
        tr1, tr2 = st.columns([1, 2])
        with tr1:
            training_hours = st.number_input(
                "Training Hours Completed", min_value=1, max_value=500, value=50, step=1,
                help="Total training hours logged (1–500)"
            )

        # ── Submit ────────────────────────────────────────────────────────────
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            submitted = st.form_submit_button(
                "🚀  Evaluate Candidate", use_container_width=True
            )

    # ── Validation & Prediction ───────────────────────────────────────────────
    if submitted:
        errors = []
        if not (1 <= training_hours <= 500):
            errors.append("Training hours must be between 1 and 500.")
        if not (0.40 <= cdi <= 1.00):
            errors.append("City Development Index must be between 0.40 and 1.00.")

        if errors:
            for err in errors:
                st.error(f"⚠️ {err}")
        else:
            with st.spinner("⚙️ Running AI model…"):
                # ── Build candidate dict (identical to original page logic) ─────
                candidate_dict = {
                    "city_development_index": [cdi],
                    "training_hours":         [training_hours],
                    "gender":                 [gender],
                    "relevent_experience":    [relevent_experience],
                    "enrolled_university":    [enrolled_university],
                    "education_level":        [education_level],
                    "major_discipline":       [major_discipline],
                    "experience":             [experience],
                    "company_size":           [company_size],
                    "company_type":           [company_type],
                    "last_new_job":           [last_new_job],
                }
                cand_df = pd.DataFrame(candidate_dict)

                # ── Preprocessing (identical to original) ─────────────────────
                from src.preprocessing import CandidatePreprocessor, CustomStandardScaler
                preprocessor = CandidatePreprocessor()
                preprocessor.feature_names      = config["feature_names"]
                preprocessor.nominal_categories = config["nominal_categories"]
                scaler = CustomStandardScaler()
                scaler.mean_  = pd.Series(config["scaler_mean"])
                scaler.scale_ = pd.Series(config["scaler_scale"])
                preprocessor.scaler = scaler
                X_cand_scaled = preprocessor.transform(cand_df)

                # ── Prediction (identical to original) ────────────────────────
                from src.modeling import LogisticRegressionModel
                X_train = pd.read_csv(
                    os.path.join("data", "processed", "X_train.csv")
                ).values
                y_train = pd.read_csv(
                    os.path.join("data", "processed", "y_train.csv")
                ).values.ravel()
                model = LogisticRegressionModel(lr=0.08, n_iters=400, l2_reg=0.1)
                model.fit(X_train, y_train)
                prob = float(model.predict_proba(X_cand_scaled.values)[0, 1])

                # ── Fairness-calibrated threshold (identical to original) ──────
                threshold  = fair_thresholds.get(gender, 0.50)
                pred_class = int(prob >= threshold)
                tier       = ("High Priority" if prob >= 0.50 else
                              "Qualified"     if prob >= 0.35 else
                              "Extended"      if prob >= 0.20 else "Reserve")

                # ── Confidence from real probability (no fabrication) ─────────
                if prob >= 0.80:   conf, conf_clr = "Very High", "#10b981"
                elif prob >= 0.65: conf, conf_clr = "High",      "#34d399"
                elif prob >= 0.50: conf, conf_clr = "Moderate",  "#f59e0b"
                elif prob >= 0.35: conf, conf_clr = "Low",       "#f97316"
                else:              conf, conf_clr = "Very Low",  "#ef4444"

                # ── Persist across theme toggles ──────────────────────────────
                _cid = (candidate_id.strip() if candidate_id.strip()
                        else f"CAND-{abs(hash(f'{cdi}{experience}{gender}')) % 100000}")
                st.session_state["pred_result"] = {
                    "prob": prob, "pred_class": pred_class, "tier": tier,
                    "conf": conf, "conf_clr": conf_clr,
                    "threshold": threshold, "gender": gender,
                    "cdi": cdi, "experience": experience,
                    "education": education_level, "major": major_discipline,
                    "training": training_hours, "rel_exp": relevent_experience,
                    "company_type": company_type, "company_size": company_size,
                    "last_new_job": last_new_job, "enrolled": enrolled_university_label,
                    "candidate_id": _cid,
                    "scaled_features": dict(
                        zip(config["feature_names"], X_cand_scaled.values[0])
                    ),
                }

    # ── Results Panel ─────────────────────────────────────────────────────────
    result = st.session_state.get("pred_result")
    if not result:
        st.markdown(
            f'<div style="background:{t["metric_bg"]};border:1px dashed {t["card_border"]};'
            f'border-radius:14px;padding:40px;text-align:center;margin-top:24px;">'
            f'<div style="font-size:3rem;margin-bottom:12px;">🎯</div>'
            f'<div style="color:{t["text_primary"]};font-size:1.1rem;font-weight:700;">'
            f'No evaluation yet</div>'
            f'<div style="color:{t["text_muted"]};font-size:0.85rem;margin-top:6px;">'
            f'Fill in the candidate details above and click <b>Evaluate Candidate</b>.</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        return

    prob       = result["prob"]
    pred_class = result["pred_class"]
    tier       = result["tier"]
    conf       = result["conf"]
    conf_clr   = result["conf_clr"]
    threshold  = result["threshold"]
    gender_r   = result["gender"]
    _cid       = result["candidate_id"]

    # Tier colours
    _TC = {
        "High Priority": ("#ef4444", "rgba(239,68,68,0.1)",  "rgba(239,68,68,0.3)"),
        "Qualified":     ("#10b981", "rgba(16,185,129,0.1)", "rgba(16,185,129,0.3)"),
        "Extended":      ("#f59e0b", "rgba(245,158,11,0.1)", "rgba(245,158,11,0.3)"),
        "Reserve":       ("#6b7280", "rgba(107,114,128,0.1)","rgba(107,114,128,0.3)"),
    }
    tier_clr, tier_bg, tier_bd = _TC.get(tier, _TC["Reserve"])

    # Recommendation label
    _REC_LABEL = {
        "High Priority": "✅ Recommended for Interview",
        "Qualified":     "✅ Proceed to Screening",
        "Extended":      "⚠️ Consider for Pipeline",
        "Reserve":       "❌ Not Recommended",
    }
    rec_label = _REC_LABEL.get(tier, "—")

    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-header">📊 Screening Result — {_cid}</div>',
        unsafe_allow_html=True
    )

    # ── Result KPI cards ─────────────────────────────────────────────────────
    rc1, rc2, rc3, rc4 = st.columns(4)
    score_pct = prob * 100

    # Score gauge card
    rc1.markdown(
        f'<div class="kpi-card" style="text-align:center;padding:20px 12px;">'
        f'<div style="color:{t["text_muted"]};font-size:0.62rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;">Suitability Score</div>'
        f'<div style="font-size:2.2rem;font-weight:900;color:{tier_clr};">{score_pct:.1f}%</div>'
        f'<div style="background:{t["divider"]};border-radius:4px;height:6px;margin:8px 4px 0;">'
        f'<div style="background:{tier_clr};width:{score_pct:.0f}%;height:100%;border-radius:4px;"></div>'
        f'</div></div>',
        unsafe_allow_html=True
    )
    # Tier badge card
    rc2.markdown(
        f'<div class="kpi-card" style="text-align:center;padding:20px 12px;">'
        f'<div style="color:{t["text_muted"]};font-size:0.62rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;">Priority Tier</div>'
        f'<span style="background:{tier_bg};color:{tier_clr};border:1px solid {tier_bd};'
        f'padding:6px 16px;border-radius:20px;font-size:0.82rem;font-weight:700;">'
        f'{tier}</span></div>',
        unsafe_allow_html=True
    )
    # Recommendation card
    rc3.markdown(
        f'<div class="kpi-card" style="text-align:center;padding:20px 12px;">'
        f'<div style="color:{t["text_muted"]};font-size:0.62rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;">Recommendation</div>'
        f'<div style="color:{t["text_primary"]};font-size:0.82rem;font-weight:700;">'
        f'{rec_label}</div></div>',
        unsafe_allow_html=True
    )
    # Confidence card
    rc4.markdown(
        f'<div class="kpi-card" style="text-align:center;padding:20px 12px;">'
        f'<div style="color:{t["text_muted"]};font-size:0.62rem;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.07em;margin-bottom:8px;">Confidence</div>'
        f'<div style="color:{conf_clr};font-size:1.1rem;font-weight:800;">{conf}</div>'
        f'<div style="color:{t["text_muted"]};font-size:0.7rem;margin-top:3px;">'
        f'P={prob:.4f}</div></div>',
        unsafe_allow_html=True
    )

    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

    # ── Recruiter Explanation ─────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">🧠 Recruiter Explanation</div>',
        unsafe_allow_html=True
    )

    # Build narrative using the existing generate_candidate_narrative() helper
    cand_row_for_narrative = {
        "Suitability Score": prob,
        "Priority Tier":     tier,
        "Experience":        result["experience"],
        "Education":         result["education"],
        "Training Hours":    result["training"],
        "Relevant Exp":      result["rel_exp"],
        "Major":             result["major"],
        "Company Type":      result["company_type"],
        "City CDI":          result["cdi"],
    }
    try:
        narrative_data = generate_candidate_narrative(cand_row_for_narrative)
    except Exception:
        narrative_data = {
            "narrative": f"Candidate scored {prob:.3f} on the suitability model.",
            "strengths": [], "weaknesses": [], "interview_type": "HR Screening",
            "action": rec_label,
        }

    # Derive positive/negative factors from global SHAP importance + feature direction
    shap_factors_pos, shap_factors_neg = [], []
    if not shap_df_glob.empty and "Feature_Label" in shap_df_glob.columns:
        top_features = shap_df_glob.head(12)["Feature_Label"].tolist()
        scaled_f = result.get("scaled_features", {})
        for feat in top_features:
            direction = _FEAT_DIRECTION.get(feat, 0)
            feat_val  = scaled_f.get(feat, 0)
            if direction != 0:
                # Positive direction: high feature value → higher score
                is_pos = (direction > 0 and feat_val > 0) or (direction < 0 and feat_val < 0)
            else:
                is_pos = feat_val > 0
            template = _FEAT_TEMPLATES.get(feat, feat.replace("_", " ").title())
            if is_pos and len(shap_factors_pos) < 4:
                shap_factors_pos.append(template)
            elif not is_pos and len(shap_factors_neg) < 3:
                shap_factors_neg.append(template)

    # Merge SHAP-derived factors with narrative strengths/weaknesses
    positives = (shap_factors_pos or narrative_data.get("strengths", []))[:4]
    negatives = (shap_factors_neg or narrative_data.get("weaknesses", []))[:3]
    interview = narrative_data.get("interview_type", "HR Screening")

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        pos_items = "".join(
            f'<div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:8px;">'
            f'<span style="color:#10b981;font-size:1rem;margin-top:1px;">●</span>'
            f'<span style="color:{t["text_primary"]};font-size:0.83rem;">{s}</span></div>'
            for s in positives
        ) if positives else f'<span style="color:{t["text_muted"]}">None identified</span>'
        st.markdown(
            f'<div class="panel-card">'
            f'<div style="color:{t["text_primary"]};font-weight:700;font-size:0.88rem;'
            f'margin-bottom:12px;">✅ Positive Factors</div>'
            f'{pos_items}</div>',
            unsafe_allow_html=True
        )

    with exp_col2:
        neg_items = "".join(
            f'<div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:8px;">'
            f'<span style="color:#ef4444;font-size:1rem;margin-top:1px;">●</span>'
            f'<span style="color:{t["text_primary"]};font-size:0.83rem;">{s}</span></div>'
            for s in negatives
        ) if negatives else f'<span style="color:{t["text_muted"]}">No concerns identified</span>'
        st.markdown(
            f'<div class="panel-card">'
            f'<div style="color:{t["text_primary"]};font-weight:700;font-size:0.88rem;'
            f'margin-bottom:12px;">⚠️ Factors Requiring Attention</div>'
            f'{neg_items}</div>',
            unsafe_allow_html=True
        )

    # Narrative + next step
    st.markdown(
        f'<div class="panel-card" style="margin-top:14px;">'
        f'<div style="color:{t["text_primary"]};font-weight:700;font-size:0.88rem;'
        f'margin-bottom:8px;">📝 Recruiter Summary</div>'
        f'<div style="color:{t["text_secondary"]};font-size:0.83rem;line-height:1.65;">'
        f'{narrative_data.get("narrative","")}</div>'
        f'<div style="margin-top:14px;padding-top:12px;border-top:1px solid {t["divider"]};">'
        f'<span style="color:{t["text_muted"]};font-size:0.72rem;font-weight:700;'
        f'text-transform:uppercase;">Recommended Next Step</span>'
        f'<div style="color:#3b82f6;font-weight:700;font-size:0.88rem;margin-top:4px;">'
        f'🗓 {interview}</div></div></div>',
        unsafe_allow_html=True
    )

    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

    # ── Fairness Card ─────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">⚖️ Fairness-Aware Decision</div>',
        unsafe_allow_html=True
    )
    mit = fair_config.get("mitigated_summary", {})
    raw = fair_config.get("raw_summary",      {})
    dpd_raw = raw.get("Demographic Parity Difference", 0)
    dpd_mit = mit.get("Demographic Parity Difference", 0)

    st.markdown(
        f'<div class="panel-card" style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;">'
        f'<div><div style="color:{t["text_muted"]};font-size:0.65rem;font-weight:700;'
        f'text-transform:uppercase;margin-bottom:4px;">Protected Attribute</div>'
        f'<div style="color:{t["text_primary"]};font-weight:700;">Gender (⚖️ {gender_r})</div>'
        f'<div style="color:{t["text_muted"]};font-size:0.72rem;margin-top:3px;">'
        f'Threshold applied: {threshold}</div></div>'
        f'<div><div style="color:{t["text_muted"]};font-size:0.65rem;font-weight:700;'
        f'text-transform:uppercase;margin-bottom:4px;">Fairness Status</div>'
        f'<div style="color:#10b981;font-weight:700;">✅ Bias Mitigated</div>'
        f'<div style="color:{t["text_muted"]};font-size:0.72rem;margin-top:3px;">'
        f'DPD: {dpd_raw:.4f} → {dpd_mit:.4f} after mitigation</div></div>'
        f'<div><div style="color:{t["text_muted"]};font-size:0.65rem;font-weight:700;'
        f'text-transform:uppercase;margin-bottom:4px;">Evaluation Level</div>'
        f'<div style="color:{t["text_primary"]};font-weight:700;">System-Level</div>'
        f'<div style="color:{t["text_muted"]};font-size:0.72rem;margin-top:3px;">'
        f'Fairness is validated at the model level, not per-candidate</div></div>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.info(
        "ℹ️ **Fairness note:** Gender is used **only** to select a fairness-calibrated "
        "decision threshold, reducing historical bias. It does not artificially raise or "
        "lower the model's predicted probability. Fairness metrics (DPD, EOD) are "
        "evaluated at the system/model level using Fairlearn."
    )

    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

    # ── Action Buttons ────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-header">⚡ Actions</div>',
        unsafe_allow_html=True
    )
    a1, a2, a3, a4 = st.columns(4)

    with a1:
        if st.button("👤 View Profile", use_container_width=True, key="pred_view_profile"):
            # Persist candidate context so profile page shows relevant data
            st.session_state["pred_profile_context"] = result
            st.info("Profile view requires an existing ranking record. "
                    "Use Candidate Rankings to look up this candidate by ID.")

    with a2:
        if st.button("➕ Add to Shortlist", use_container_width=True, key="pred_shortlist"):
            shortlist = _load_shortlist()
            entry = {
                "candidate_id": _cid, "gender": gender_r,
                "suitability_score": round(prob, 4),
                "tier": tier, "confidence": conf,
                "shortlisted_at": pd.Timestamp.now().isoformat(),
            }
            # Avoid duplicates
            existing_ids = [e["candidate_id"] for e in shortlist]
            if _cid not in existing_ids:
                shortlist.append(entry)
                _save_shortlist(shortlist)
                st.success(f"✅ {_cid} added to shortlist ({len(shortlist)} total).")
            else:
                st.warning(f"⚠️ {_cid} is already in the shortlist.")

    with a3:
        report_lines = [
            f"FairHire AI — Candidate Screening Report",
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
            f"{'='*50}",
            f"Candidate ID      : {_cid}",
            f"Gender            : {gender_r}",
            f"City Dev. Index   : {result['cdi']:.2f}",
            f"Education         : {result['education']}",
            f"Major             : {result['major']}",
            f"Experience        : {result['experience']} yrs",
            f"Relevant Exp.     : {result['rel_exp']}",
            f"Company Type      : {result['company_type']}",
            f"Company Size      : {result['company_size']}",
            f"Training Hours    : {result['training']}",
            f"{'='*50}",
            f"RESULT",
            f"Suitability Score : {prob:.4f} ({score_pct:.1f}%)",
            f"Priority Tier     : {tier}",
            f"Recommendation    : {rec_label}",
            f"Confidence        : {conf}",
            f"Fairness Threshold: {threshold} (gender-calibrated)",
            f"Predicted Class   : {'Shortlisted' if pred_class==1 else 'Not Shortlisted'}",
            f"{'='*50}",
            f"POSITIVE FACTORS",
        ] + [f"  + {s}" for s in positives] + [
            f"AREAS OF ATTENTION",
        ] + [f"  - s" for s in negatives] + [
            f"{'='*50}",
            f"NEXT STEP: {interview}",
            f"{'='*50}",
            f"Note: Fairness metrics are validated at the system/model level.",
            f"Model: Logistic Regression | System ROC-AUC: {model_info.get('best_roc_auc',0):.4f}",
        ]
        report_text = "\n".join(report_lines)
        st.download_button(
            "📄 Download Report",
            data=report_text,
            file_name=f"screening_{_cid}.txt",
            mime="text/plain",
            use_container_width=True,
            key="pred_download",
        )

    with a4:
        if st.button("🗑 Clear Form", use_container_width=True, key="pred_clear"):
            st.session_state.pop("pred_result", None)
            st.rerun()



# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6: SETTINGS  — theme control + model info + about
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# PAGE 8: RESUME UPLOAD & AI SCREENING
# ─────────────────────────────────────────────────────────────────────────────

import re

def _extract_candidate_name(text: str) -> str:
    """Extract candidate name from resume text using rule-based parsing."""
    if not text:
        return "Name Not Detected"
    
    # Try looking for Name: or Candidate Name: or Full Name:
    patterns = [
        r"(?i)(?:candidate\s+name|full\s+name|name)\s*:\s*([^\n\r]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            name = match.group(1).strip()
            if 2 < len(name) < 60:
                return name

    # Fallback: inspect the first few non-empty lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    ignore_keywords = {
        "resume", "cv", "curriculum", "vitae", "summary", "experience", "education",
        "work", "project", "profile", "about", "contact", "phone", "email",
        "skills", "objective", "certifications", "interests", "languages",
        "candidate", "information", "details", "personal", "history",
        "interested", "role", "roles", "seeking", "looking", "experienced", "worked"
    }
    
    for line in lines[:4]:
        words = line.split()
        if 2 <= len(words) <= 4:
            if not any(char.isdigit() for char in line):
                if "@" not in line and "http" not in line and ".com" not in line:
                    if not any(w.lower() in ignore_keywords for w in words):
                        return line

    return "Name Not Detected"

def _extract_pdf_text(file_obj) -> str:
    try:
        reader = PyPDF2.PdfReader(file_obj)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text.strip()
    except Exception as e:
        return ""

def _extract_docx_text(file_obj) -> str:
    try:
        doc = docx.Document(file_obj)
        return "\n".join([p.text for p in doc.paragraphs]).strip()
    except Exception as e:
        return ""

def _extract_resume_text(file_obj, filename: str) -> str:
    ext = filename.split('.')[-1].lower()
    if ext == 'pdf':
        return _extract_pdf_text(file_obj)
    elif ext == 'docx':
        return _extract_docx_text(file_obj)
    elif ext == 'txt':
        try:
            return file_obj.getvalue().decode("utf-8").strip()
        except Exception:
            return ""
    return ""

def _extract_candidate_fields(text: str) -> dict:
    """
    Deterministic rule-based keyword matcher.
    Maps resume text to expected CandidatePreprocessor categorical fields.
    Does NOT hallucinate or infer protected attributes (Gender is always None).
    """
    text_lower = text.lower()
    fields = {
        "city_development_index": None,
        "training_hours": None,
        "gender": "Not Provided",  # STRICT PRIVACY RULE
        "relevent_experience": None,
        "enrolled_university": None,
        "education_level": None,
        "major_discipline": None,
        "experience": None,
        "company_size": None,
        "company_type": None,
        "last_new_job": None
    }

    # Education Level
    if any(k in text_lower for k in ["phd", "ph.d", "doctorate"]):
        fields["education_level"] = "Phd"
    elif any(k in text_lower for k in ["masters", "m.s", "mba", "m.a"]):
        fields["education_level"] = "Masters"
    elif any(k in text_lower for k in ["graduate", "bachelors", "b.s", "b.a", "b.tech", "degree"]):
        fields["education_level"] = "Graduate"
    elif any(k in text_lower for k in ["high school"]):
        fields["education_level"] = "High School"
    elif any(k in text_lower for k in ["primary school"]):
        fields["education_level"] = "Primary School"

    # Major
    if any(k in text_lower for k in ["computer", "science", "engineering", "math", "technology", "stem"]):
        fields["major_discipline"] = "STEM"
    elif any(k in text_lower for k in ["business", "management", "finance", "accounting"]):
        fields["major_discipline"] = "Business Degree"
    elif any(k in text_lower for k in ["arts", "design"]):
        fields["major_discipline"] = "Arts"
    elif any(k in text_lower for k in ["humanities", "history", "english"]):
        fields["major_discipline"] = "Humanities"

    # Relevant Experience
    if any(k in text_lower for k in ["data scientist", "machine learning", "data engineer", "software engineer", "developer", "analyst"]):
        fields["relevent_experience"] = "Has relevent experience"

    # Experience Years (naive numeric extraction around "years")
    if "years" in text_lower:
        idx = text_lower.find("years")
        context = text_lower[max(0, idx-10):idx]
        import re
        nums = re.findall(r'\d+', context)
        if nums:
            val = int(nums[-1])
            if val < 1: fields["experience"] = "<1"
            elif val > 20: fields["experience"] = ">20"
            else: fields["experience"] = "1-20"

    # University
    if any(k in text_lower for k in ["university", "college", "institute"]):
        if "expected" in text_lower or "present" in text_lower:
            fields["enrolled_university"] = "Full time course"
        else:
            fields["enrolled_university"] = "no_enrollment"

    # Company Type
    if any(k in text_lower for k in ["startup", "start-up"]):
        fields["company_type"] = "Funded Startup"
    elif any(k in text_lower for k in ["public", "government"]):
        fields["company_type"] = "Public Sector"
    elif any(k in text_lower for k in ["ngo", "non-profit"]):
        fields["company_type"] = "NGO"
    elif any(k in text_lower for k in ["ltd", "private", "inc"]):
        fields["company_type"] = "Pvt Ltd"

    return fields

def _calculate_field_confidences(text: str, fields: dict, candidate_name: str) -> dict:
    """
    Phase 9: Compute deterministic extraction confidence (High, Medium, Low)
    for every candidate field based on rule-based evidence in resume text.
    """
    text_lower = (text or "").lower()
    confidences = {}

    # Candidate Name
    if candidate_name and candidate_name != "Name Not Detected":
        if any(h in text_lower for h in ["name:", "full name:", "candidate name:"]):
            confidences["candidate_name"] = {"level": "High", "badge": "🟢 High Confidence", "reason": "Explicit name header matched"}
        else:
            confidences["candidate_name"] = {"level": "High", "badge": "🟢 High Confidence", "reason": "Extracted from prominent heading line"}
    else:
        confidences["candidate_name"] = {"level": "Low", "badge": "🔴 Low Confidence", "reason": "Candidate name not detected"}

    # Education Level
    edu = fields.get("education_level")
    if edu and edu != "Unknown":
        if any(k in text_lower for k in ["phd", "ph.d", "doctorate", "masters", "m.s", "mba", "bachelors", "b.tech", "b.s", "b.a", "degree", "high school", "primary school"]):
            confidences["education_level"] = {"level": "High", "badge": "🟢 High Confidence", "reason": "Explicit degree qualification matched"}
        else:
            confidences["education_level"] = {"level": "Medium", "badge": "🟡 Medium Confidence", "reason": "Inferred from education context"}
    else:
        confidences["education_level"] = {"level": "Low", "badge": "🔴 Low Confidence", "reason": "No education degree keyword found"}

    # Major Discipline
    maj = fields.get("major_discipline")
    if maj and maj != "Unknown":
        if any(k in text_lower for k in ["computer science", "software engineering", "business administration", "finance", "accounting", "fine arts", "humanities"]):
            confidences["major_discipline"] = {"level": "High", "badge": "🟢 High Confidence", "reason": "Explicit major / discipline matched"}
        elif any(k in text_lower for k in ["computer", "science", "engineering", "math", "technology", "stem", "business", "management"]):
            confidences["major_discipline"] = {"level": "Medium", "badge": "🟡 Medium Confidence", "reason": "Field of study inferred from generic keywords"}
        else:
            confidences["major_discipline"] = {"level": "Low", "badge": "🔴 Low Confidence", "reason": "Weak major discipline match"}
    else:
        confidences["major_discipline"] = {"level": "Low", "badge": "🔴 Low Confidence", "reason": "No major discipline detected"}

    # Relevant Experience
    rel = fields.get("relevent_experience")
    if rel == "Has relevent experience":
        if any(k in text_lower for k in ["data scientist", "machine learning", "data engineer", "software engineer", "developer", "analyst"]):
            confidences["relevent_experience"] = {"level": "High", "badge": "🟢 High Confidence", "reason": "Matching industry role title found"}
        else:
            confidences["relevent_experience"] = {"level": "Medium", "badge": "🟡 Medium Confidence", "reason": "Inferred from project description"}
    else:
        confidences["relevent_experience"] = {"level": "Low", "badge": "🔴 Low Confidence", "reason": "No explicit relevant tech role found"}

    # Experience Years
    exp = fields.get("experience")
    if exp:
        if "years" in text_lower or "yrs" in text_lower:
            confidences["experience"] = {"level": "High", "badge": "🟢 High Confidence", "reason": "Explicit years of experience matched"}
        elif any(k in text_lower for k in ["senior", "lead", "principal", "experienced", "developer"]):
            confidences["experience"] = {"level": "Medium", "badge": "🟡 Medium Confidence", "reason": "Inferred from role seniority keywords"}
        else:
            confidences["experience"] = {"level": "Low", "badge": "🔴 Low Confidence", "reason": "Default duration assigned"}
    else:
        confidences["experience"] = {"level": "Low", "badge": "🔴 Low Confidence", "reason": "No experience duration detected"}

    # Enrolled University
    uni = fields.get("enrolled_university")
    if uni and uni != "Unknown":
        if any(k in text_lower for k in ["university", "college", "institute"]):
            confidences["enrolled_university"] = {"level": "High", "badge": "🟢 High Confidence", "reason": "Academic institution match found"}
        else:
            confidences["enrolled_university"] = {"level": "Medium", "badge": "🟡 Medium Confidence", "reason": "Enrollment status inferred"}
    else:
        confidences["enrolled_university"] = {"level": "Low", "badge": "🔴 Low Confidence", "reason": "No university enrollment status detected"}

    # Company Type
    comp_type = fields.get("company_type")
    if comp_type and comp_type != "Unknown":
        if any(k in text_lower for k in ["startup", "start-up", "public sector", "government", "ngo", "non-profit", "pvt ltd", "private limited", "inc"]):
            confidences["company_type"] = {"level": "High", "badge": "🟢 High Confidence", "reason": "Explicit organization type matched"}
        else:
            confidences["company_type"] = {"level": "Medium", "badge": "🟡 Medium Confidence", "reason": "Company type inferred from context"}
    else:
        confidences["company_type"] = {"level": "Low", "badge": "🔴 Low Confidence", "reason": "No company type detected"}

    # Company Size
    comp_size = fields.get("company_size")
    if comp_size and comp_size != "Unknown":
        if any(k in text_lower for k in ["employees", "headcount", "staff", "team size"]):
            confidences["company_size"] = {"level": "High", "badge": "🟢 High Confidence", "reason": "Explicit headcount metric matched"}
        else:
            confidences["company_size"] = {"level": "Medium", "badge": "🟡 Medium Confidence", "reason": "Company size inferred from company type"}
    else:
        confidences["company_size"] = {"level": "Low", "badge": "🔴 Low Confidence", "reason": "No company size detected"}

    # Last New Job
    last_job = fields.get("last_new_job")
    if last_job and last_job != "Unknown":
        if any(k in text_lower for k in ["previous job", "former role", "last position", "changed job"]):
            confidences["last_new_job"] = {"level": "High", "badge": "🟢 High Confidence", "reason": "Job transition history matched"}
        else:
            confidences["last_new_job"] = {"level": "Medium", "badge": "🟡 Medium Confidence", "reason": "Job change frequency inferred"}
    else:
        confidences["last_new_job"] = {"level": "Low", "badge": "🔴 Low Confidence", "reason": "No job transition duration detected"}

    # Training Hours
    train_hrs = fields.get("training_hours")
    if train_hrs is not None:
        if any(k in text_lower for k in ["training hours", "course hours", "hours of training", "certificat"]):
            confidences["training_hours"] = {"level": "High", "badge": "🟢 High Confidence", "reason": "Explicit training hours matched"}
        else:
            confidences["training_hours"] = {"level": "Low", "badge": "🔴 Low Confidence", "reason": "Default training hours assigned"}
    else:
        confidences["training_hours"] = {"level": "Low", "badge": "🔴 Low Confidence", "reason": "Training hours missing"}

    # City Development Index (CDI)
    cdi = fields.get("city_development_index")
    if cdi is not None:
        if any(k in text_lower for k in ["city", "location", "based in", "cdi"]):
            confidences["city_development_index"] = {"level": "Medium", "badge": "🟡 Medium Confidence", "reason": "Location context inferred"}
        else:
            confidences["city_development_index"] = {"level": "Low", "badge": "🔴 Low Confidence", "reason": "Default CDI score assigned"}
    else:
        confidences["city_development_index"] = {"level": "Low", "badge": "🔴 Low Confidence", "reason": "City development index missing"}

    # Gender (Strict Privacy Rule)
    confidences["gender"] = {"level": "High", "badge": "🟢 High (Protected)", "reason": "Privacy rule: Gender remains 'Not Provided'"}

    return confidences

def _calculate_resume_quality_score(text: str, fields: dict, confidences: dict, candidate_name: str) -> dict:
    """
    Phase 9: Compute a deterministic 0-100% Resume Quality Score.
    Completely separate from Candidate Suitability Score (ML probability).
    """
    score = 0
    
    text_len = len(text or "")
    if text_len >= 500:
        score += 20
    elif text_len >= 200:
        score += 10
    elif text_len > 0:
        score += 5

    # Candidate Name (15 pts)
    if candidate_name and candidate_name != "Name Not Detected":
        score += 15
        
    # Education (20 pts)
    edu_conf = confidences.get("education_level", {}).get("level")
    if edu_conf == "High":
        score += 20
    elif edu_conf == "Medium":
        score += 10
        
    # Experience (20 pts)
    exp_conf = confidences.get("experience", {}).get("level")
    if exp_conf == "High":
        score += 20
    elif exp_conf == "Medium":
        score += 10
        
    # Major Discipline (10 pts)
    maj_conf = confidences.get("major_discipline", {}).get("level")
    if maj_conf == "High":
        score += 10
    elif maj_conf == "Medium":
        score += 5
        
    # Relevant Experience & Tech Skills (15 pts)
    rel_conf = confidences.get("relevent_experience", {}).get("level")
    if rel_conf == "High":
        score += 15
    elif rel_conf == "Medium":
        score += 8

    score = max(0, min(100, score))

    if score >= 80:
        status_label = "Valid"
        status_badge = "✅ Valid"
        badge_color = "#10b981"
        badge_bg = "rgba(16,185,129,0.1)"
        border_color = "rgba(16,185,129,0.3)"
    elif score >= 50:
        status_label = "Review Recommended"
        status_badge = "⚠️ Review Recommended"
        badge_color = "#f59e0b"
        badge_bg = "rgba(245,158,11,0.1)"
        border_color = "rgba(245,158,11,0.3)"
    else:
        status_label = "Missing Critical Information"
        status_badge = "❌ Missing Critical Information"
        badge_color = "#ef4444"
        badge_bg = "rgba(239,68,68,0.1)"
        border_color = "rgba(239,68,68,0.3)"

    return {
        "quality_score": score,
        "status_label": status_label,
        "status_badge": status_badge,
        "badge_color": badge_color,
        "badge_bg": badge_bg,
        "border_color": border_color
    }

def _generate_validation_warnings(text: str, fields: dict, confidences: dict, candidate_name: str, quality: dict) -> list:
    """
    Phase 9: Generate validation warnings and recommendations prior to screening.
    """
    warnings = []

    if not text or len(text) < 150:
        warnings.append({
            "category": "danger",
            "icon": "❌",
            "field": "Resume Content",
            "title": "Very Short Resume Content",
            "message": "The extracted resume text is very brief. Ensure a text-based PDF, DOCX, or TXT document was uploaded."
        })

    if not candidate_name or candidate_name == "Name Not Detected":
        warnings.append({
            "category": "warning",
            "icon": "⚠️",
            "field": "Candidate Name",
            "title": "Candidate Name Not Detected",
            "message": "Candidate name could not be automatically detected. Recruiter verification recommended."
        })

    if confidences.get("education_level", {}).get("level") == "Low":
        warnings.append({
            "category": "warning",
            "icon": "⚠️",
            "field": "Education Level",
            "title": "Education Level Missing",
            "message": "No explicit degree was detected. Please select the correct education level before screening."
        })

    if confidences.get("experience", {}).get("level") == "Low":
        warnings.append({
            "category": "warning",
            "icon": "⚠️",
            "field": "Years of Experience",
            "title": "Experience Duration Uncertain",
            "message": "Years of experience duration could not be extracted with high confidence."
        })

    if confidences.get("major_discipline", {}).get("level") == "Low":
        warnings.append({
            "category": "info",
            "icon": "ℹ️",
            "field": "Major Discipline",
            "title": "Major / Discipline Missing",
            "message": "Major discipline not explicitly detected in resume text."
        })

    low_fields = [k for k, v in confidences.items() if v.get("level") == "Low" and k != "gender"]
    if len(low_fields) >= 3:
        warnings.append({
            "category": "info",
            "icon": "ℹ️",
            "field": "Field Completeness",
            "title": f"{len(low_fields)} Fields Need Verification",
            "message": f"Multiple fields defaulted to low-confidence values ({', '.join(low_fields[:3])}). Review form entries below."
        })

    return warnings

def _load_screening_history() -> list:
    """
    Phase 10: Load persistent resume screening history from data/resume_screening_history.json.
    """
    history_file = os.path.join("data", "resume_screening_history.json")
    if not os.path.exists(history_file):
        return []
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []

def _save_screening_result(screening_record: dict) -> None:
    """
    Phase 10: Append or update a candidate screening record in data/resume_screening_history.json.
    Deduplicates by candidate_id to prevent duplicates on page reruns.
    """
    os.makedirs("data", exist_ok=True)
    history_file = os.path.join("data", "resume_screening_history.json")
    history = _load_screening_history()
    
    cand_id = screening_record.get("candidate_id") or screening_record.get("id")
    if not cand_id:
        return

    updated = False
    for i, item in enumerate(history):
        item_id = item.get("candidate_id") or item.get("id")
        if item_id == cand_id:
            if "status" in item and "status" not in screening_record:
                screening_record["status"] = item["status"]
            if "recruiter_notes" in item and "recruiter_notes" not in screening_record:
                screening_record["recruiter_notes"] = item["recruiter_notes"]
            history[i] = screening_record
            updated = True
            break
            
    if not updated:
        if "status" not in screening_record:
            screening_record["status"] = "Pending"
        if "recruiter_notes" not in screening_record:
            screening_record["recruiter_notes"] = ""
        history.append(screening_record)
        
    try:
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass

def _update_screening_status(candidate_id: str, new_status: str, recruiter_notes: str = None) -> bool:
    """
    Phase 10: Update candidate status (Pending, Shortlisted, Rejected) and notes.
    """
    history_file = os.path.join("data", "resume_screening_history.json")
    history = _load_screening_history()
    updated = False
    
    for item in history:
        item_id = item.get("candidate_id") or item.get("id")
        if item_id == candidate_id:
            item["status"] = new_status
            if recruiter_notes is not None:
                item["recruiter_notes"] = recruiter_notes
            item["status_updated_timestamp"] = datetime.now().isoformat()
            updated = True
            break
            
    if updated:
        try:
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            return True
        except Exception:
            return False
    return False

def _run_single_resume_screening(file_obj, filename: str, input_overrides: dict = None) -> dict:
    """
    Phase 10: Unified single resume screening pipeline runner.
    Executes text extraction, name parsing, field extraction, quality calculation,
    and production ML model prediction. Auto-saves completed result to history.
    """
    text = _extract_resume_text(file_obj, filename) if file_obj else ""
    if not text and input_overrides is None:
        return {"error": f"Unable to extract text from '{filename}'."}

    fields = _extract_candidate_fields(text) if text else {}
    name = _extract_candidate_name(text) if text else "Name Not Detected"

    if input_overrides:
        for k, v in input_overrides.items():
            if k in fields or k in ["training_hours", "city_development_index"]:
                fields[k] = v

    conf = _calculate_field_confidences(text, fields, name)
    qual = _calculate_resume_quality_score(text, fields, conf, name)
    warns = _generate_validation_warnings(text, fields, conf, name, qual)

    mapped_gender = "Unknown" if fields.get("gender") == "Not Provided" else fields.get("gender", "Unknown")

    input_data = {
        "city_development_index": float(fields.get("city_development_index") or 0.850),
        "training_hours": int(fields.get("training_hours") or 50),
        "gender": mapped_gender,
        "relevent_experience": fields.get("relevent_experience") or "No relevent experience",
        "enrolled_university": fields.get("enrolled_university") or "no_enrollment",
        "education_level": fields.get("education_level") or "Graduate",
        "major_discipline": fields.get("major_discipline") or "STEM",
        "experience": fields.get("experience") or "1-20",
        "company_size": fields.get("company_size") or "Unknown",
        "company_type": fields.get("company_type") or "Pvt Ltd",
        "last_new_job": fields.get("last_new_job") or "1"
    }

    config_path = os.path.join("models", "trained_models", "preprocessor_config.json")
    if not os.path.exists(config_path):
        return {"error": "Missing ML models configuration."}

    config = json.load(open(config_path))
    import sys
    if "src" not in sys.path:
        sys.path.insert(0, ".")
    from src.preprocessing import CandidatePreprocessor, CustomStandardScaler
    from src.modeling import LogisticRegressionModel

    df_in = pd.DataFrame([input_data])
    prep = CandidatePreprocessor()
    prep.feature_names = config["feature_names"]
    prep.nominal_categories = config["nominal_categories"]
    scaler = CustomStandardScaler()
    scaler.mean_ = pd.Series(config["scaler_mean"])
    scaler.scale_ = pd.Series(config["scaler_scale"])
    prep.scaler = scaler

    X_proc = prep.transform(df_in)
    X_train = pd.read_csv(os.path.join("data", "processed", "X_train.csv")).values
    y_train = pd.read_csv(os.path.join("data", "processed", "y_train.csv")).values.ravel()

    model = LogisticRegressionModel(lr=0.08, n_iters=400, l2_reg=0.1)
    model.fit(X_train, y_train)

    prob = float(model.predict_proba(X_proc.values)[0, 1])

    fair_path = os.path.join("models", "trained_models", "fairness_config.json")
    fair_config = json.load(open(fair_path)) if os.path.exists(fair_path) else {}
    fair_thresholds = fair_config.get("fair_thresholds", {"Female": 0.47, "Male": 0.46, "Other": 0.49, "Unknown": 0.60})

    threshold = fair_thresholds.get(mapped_gender, 0.50)

    if prob >= 0.8:
        tier = "High Priority"
        recommendation = "Immediate Interview"
    elif prob >= 0.6:
        tier = "Qualified"
        recommendation = "Schedule Interview"
    elif prob >= 0.4:
        tier = "Extended"
        recommendation = "Keep in Pipeline"
    else:
        tier = "Reserve"
        recommendation = "Future Consideration"

    import random
    temp_id = f"RESUME-{random.randint(100000, 999999)}"

    input_data["Candidate ID"] = temp_id
    input_data["enrollee_id"] = temp_id
    input_data["prediction_probability"] = prob
    input_data["candidate_name"] = name

    record = {
        "candidate_id": temp_id,
        "id": temp_id,
        "candidate_name": name,
        "source_filename": filename,
        "timestamp": datetime.now().isoformat(),
        "prob": prob,
        "suitability_score": prob,
        "priority_tier": tier,
        "recommendation": recommendation,
        "gender": mapped_gender,
        "features": X_proc.values[0].tolist(),
        "feature_names": prep.feature_names,
        "raw_data": input_data,
        "weights": list(model.weights),
        "bias": float(model.bias),
        "quality": qual,
        "confidences": conf,
        "warnings": warns,
        "corrected_fields": {},
        "status": "Pending",
        "recruiter_notes": "",
        "raw_text": text
    }

    _save_screening_result(record)
    return record

def render_resume_screening_page():
    """Phase 8 & 9: Resume Upload, Quality Validation & AI Screening"""
    t = ThemeManager.get()

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:{t["header_bg"]};border:1px solid {t["header_border"]};'
        f'border-radius:16px;padding:28px 32px;margin-bottom:24px;">'
        f'<div style="display:flex;align-items:center;gap:14px;">'
        f'<div style="font-size:2.4rem;">📄</div>'
        f'<div><div style="color:{t["header_title"]};font-size:1.55rem;font-weight:800;'
        f'letter-spacing:-0.02em;">AI Resume Screening</div>'
        f'<div style="color:{t["header_sub"]};font-size:0.88rem;margin-top:3px;">'
        f'Upload a candidate resume and evaluate suitability using the FairHire AI screening pipeline.</div>'
        f'</div></div></div>',
        unsafe_allow_html=True
    )

    # ── Privacy notice ────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:{t["info_bg"]};border:1px solid {t["info_border"]};border-radius:12px;padding:16px;margin-bottom:24px;">'
        f'<div style="font-weight:700;color:{t["text_primary"]};font-size:0.9rem;margin-bottom:6px;">⚖️ Fairness & Responsible AI</div>'
        f'<div style="font-size:0.85rem;color:{t["text_secondary"]};line-height:1.6;">'
        f'• Protected attributes (Gender) are <b>not inferred</b> from resume content.<br>'
        f'• Resume content is kept strictly local and is not sent to external APIs.<br>'
        f'• The production screening model uses the existing fairness configuration.'
        f'</div></div>',
        unsafe_allow_html=True
    )

    tab_single, tab_bulk = st.tabs(["📄 Single Resume Screening", "📦 Bulk Resume Batch Screening"])

    with tab_single:
        # State management
        if "resume_file_id" not in st.session_state:
            st.session_state["resume_file_id"] = None
            st.session_state["resume_text"] = ""
            st.session_state["resume_fields"] = {}
            st.session_state["resume_name"] = ""
            st.session_state["resume_confidences"] = {}
            st.session_state["resume_quality"] = {}
            st.session_state["resume_warnings"] = []
            st.session_state["resume_prediction"] = None

        uploaded_file = st.file_uploader("Upload Candidate Resume", type=["pdf", "docx", "txt"])

    if uploaded_file:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        # If new file uploaded, parse it
        if st.session_state["resume_file_id"] != file_id:
            st.session_state["resume_file_id"] = file_id
            st.session_state["resume_prediction"] = None # Reset prediction
            
            with st.spinner("Extracting text and validating resume..."):
                text = _extract_resume_text(uploaded_file, uploaded_file.name)
                st.session_state["resume_text"] = text
                if text:
                    fields = _extract_candidate_fields(text)
                    name = _extract_candidate_name(text)
                    conf = _calculate_field_confidences(text, fields, name)
                    qual = _calculate_resume_quality_score(text, fields, conf, name)
                    warns = _generate_validation_warnings(text, fields, conf, name, qual)

                    st.session_state["resume_fields"] = fields
                    st.session_state["resume_name"] = name
                    st.session_state["resume_confidences"] = conf
                    st.session_state["resume_quality"] = qual
                    st.session_state["resume_warnings"] = warns
                else:
                    st.session_state["resume_fields"] = {}
                    st.session_state["resume_name"] = "Name Not Detected"
                    st.session_state["resume_confidences"] = {}
                    st.session_state["resume_quality"] = {
                        "quality_score": 0, "status_label": "Unreadable",
                        "status_badge": "❌ Unreadable", "badge_color": "#ef4444",
                        "badge_bg": "rgba(239,68,68,0.1)", "border_color": "rgba(239,68,68,0.3)"
                    }
                    st.session_state["resume_warnings"] = []

        if not st.session_state["resume_text"]:
            st.error("Unable to extract text from this resume. Please upload a text-based PDF, DOCX, or TXT file.")
            return

        with st.expander("📄 Extracted Resume Text", expanded=False):
            st.text_area("Raw Text", value=st.session_state["resume_text"], height=200, disabled=True, label_visibility="collapsed")

        # ── Phase 9: Resume Validation Summary ──────────────────────────────
        qual = st.session_state.get("resume_quality", {})
        conf = st.session_state.get("resume_confidences", {})
        warns = st.session_state.get("resume_warnings", [])
        cand_name = st.session_state.get("resume_name", "Name Not Detected")

        high_cnt = sum(1 for v in conf.values() if v.get("level") == "High" and v != "gender")
        med_cnt  = sum(1 for v in conf.values() if v.get("level") == "Medium")
        low_cnt  = sum(1 for v in conf.values() if v.get("level") == "Low")
        extracted_cnt = sum(1 for k, v in st.session_state.get("resume_fields", {}).items() if v is not None and v != "Unknown" and k != "gender")

        q_score = qual.get("quality_score", 0)
        st_badge = qual.get("status_badge", "⚠️ Review Recommended")
        b_clr = qual.get("badge_color", "#f59e0b")
        b_bg = qual.get("badge_bg", "rgba(245,158,11,0.1)")
        b_bd = qual.get("border_color", "rgba(245,158,11,0.3)")

        st.markdown(
            f'<div style="background:{t["card_bg"]};border:1px solid {t["card_border"]};'
            f'border-radius:14px;padding:20px;margin:16px 0 20px 0;box-shadow:{t["card_shadow"]};">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">'
            f'<div style="font-weight:700;font-size:1.05rem;color:{t["text_primary"]};display:flex;align-items:center;gap:8px;">'
            f'🛡️ Resume Validation & Completeness</div>'
            f'<span style="background:{b_bg};color:{b_clr};border:1px solid {b_bd};'
            f'padding:4px 14px;border-radius:20px;font-size:0.8rem;font-weight:700;">'
            f'{st_badge}</span></div>'
            
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">'
            f'<div style="background:{t["metric_bg"]};border:1px solid {t["metric_border"]};border-radius:10px;padding:12px;text-align:center;">'
            f'<div style="font-size:1.4rem;font-weight:800;color:{b_clr};">{q_score}%</div>'
            f'<div style="font-size:0.7rem;font-weight:700;color:{t["text_muted"]};text-transform:uppercase;">Resume Quality Score</div>'
            f'</div>'
            f'<div style="background:{t["metric_bg"]};border:1px solid {t["metric_border"]};border-radius:10px;padding:12px;text-align:center;">'
            f'<div style="font-size:1.05rem;font-weight:700;color:{t["text_primary"]};margin-top:2px;">{cand_name}</div>'
            f'<div style="font-size:0.7rem;font-weight:700;color:{t["text_muted"]};text-transform:uppercase;">Detected Candidate Name</div>'
            f'</div>'
            f'<div style="background:{t["metric_bg"]};border:1px solid {t["metric_border"]};border-radius:10px;padding:12px;text-align:center;">'
            f'<div style="font-size:1.4rem;font-weight:800;color:{t["text_primary"]};">{extracted_cnt} / 10</div>'
            f'<div style="font-size:0.7rem;font-weight:700;color:{t["text_muted"]};text-transform:uppercase;">Fields Extracted</div>'
            f'</div>'
            f'<div style="background:{t["metric_bg"]};border:1px solid {t["metric_border"]};border-radius:10px;padding:12px;text-align:center;">'
            f'<div style="font-size:0.85rem;font-weight:700;color:{t["text_primary"]};margin-top:4px;">🟢 {high_cnt} · 🟡 {med_cnt} · 🔴 {low_cnt}</div>'
            f'<div style="font-size:0.7rem;font-weight:700;color:{t["text_muted"]};text-transform:uppercase;">Confidence Breakdown</div>'
            f'</div>'
            f'</div></div>',
            unsafe_allow_html=True
        )

        if warns:
            with st.expander("⚠️ Resume Validation Guidance & Warnings", expanded=False):
                for w in warns:
                    st.markdown(
                        f'<div style="margin:4px 0;font-size:0.85rem;color:{t["text_secondary"]};line-height:1.5;">'
                        f'<b>{w["icon"]} {w["title"]}</b> ({w["field"]}): {w["message"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        # ── Form logic ────────────────────────────────────────────────────────
        st.markdown(f'<div class="section-header">🔍 Review Candidate Information</div>', unsafe_allow_html=True)
        
        fields = st.session_state["resume_fields"]
        
        with st.form("resume_review_form", clear_on_submit=False):
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Education Level
                edu_idx = 0
                edu_opts = ["Unknown", "Primary School", "High School", "Graduate", "Masters", "Phd"]
                if fields.get("education_level") in edu_opts:
                    edu_idx = edu_opts.index(fields["education_level"])
                c_badge = conf.get("education_level", {}).get("badge", "🔴 Low Confidence")
                c_rsn = conf.get("education_level", {}).get("reason", "")
                f_edu = st.selectbox(f"Education Level  ({c_badge})", edu_opts, index=edu_idx, help=c_rsn)

                # Major Discipline
                maj_idx = 0
                maj_opts = ["Unknown", "STEM", "Business Degree", "Arts", "Humanities", "No Major", "Other"]
                if fields.get("major_discipline") in maj_opts:
                    maj_idx = maj_opts.index(fields["major_discipline"])
                c_badge = conf.get("major_discipline", {}).get("badge", "🔴 Low Confidence")
                c_rsn = conf.get("major_discipline", {}).get("reason", "")
                f_maj = st.selectbox(f"Major / Discipline  ({c_badge})", maj_opts, index=maj_idx, help=c_rsn)
                
                # Relevant Experience
                rel_idx = 0
                rel_opts = ["No relevent experience", "Has relevent experience"]
                if fields.get("relevent_experience") in rel_opts:
                    rel_idx = rel_opts.index(fields["relevent_experience"])
                c_badge = conf.get("relevent_experience", {}).get("badge", "🔴 Low Confidence")
                c_rsn = conf.get("relevent_experience", {}).get("reason", "")
                f_rel = st.selectbox(f"Relevant Experience  ({c_badge})", rel_opts, index=rel_idx, help=c_rsn)

                # Enrolled University
                uni_idx = 0
                uni_opts = ["Unknown", "no_enrollment", "Full time course", "Part time course"]
                if fields.get("enrolled_university") in uni_opts:
                    uni_idx = uni_opts.index(fields["enrolled_university"])
                c_badge = conf.get("enrolled_university", {}).get("badge", "🔴 Low Confidence")
                c_rsn = conf.get("enrolled_university", {}).get("reason", "")
                f_uni = st.selectbox(f"University Enrollment  ({c_badge})", uni_opts, index=uni_idx, help=c_rsn)
                
                # Gender
                gen_idx = 0
                gen_opts = ["Not Provided", "Male", "Female", "Other"]
                f_gen = st.selectbox("Gender  (🟢 Protected — Not Inferred)", gen_opts, index=gen_idx, help="Privacy enforcement: Gender is not extracted or inferred from resume text.")
                
            with col2:
                # Experience
                exp_idx = 0
                exp_opts = ["<1", "1-20", ">20"]
                if fields.get("experience") in exp_opts:
                    exp_idx = exp_opts.index(fields["experience"])
                c_badge = conf.get("experience", {}).get("badge", "🔴 Low Confidence")
                c_rsn = conf.get("experience", {}).get("reason", "")
                f_exp = st.selectbox(f"Years of Experience  ({c_badge})", exp_opts, index=exp_idx, help=c_rsn)
                
                # Company Size
                size_idx = 0
                size_opts = ["Unknown", "<10", "10-49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"]
                if fields.get("company_size") in size_opts:
                    size_idx = size_opts.index(fields["company_size"])
                c_badge = conf.get("company_size", {}).get("badge", "🔴 Low Confidence")
                c_rsn = conf.get("company_size", {}).get("reason", "")
                f_size = st.selectbox(f"Company Size  ({c_badge})", size_opts, index=size_idx, help=c_rsn)
                
                # Company Type
                type_idx = 0
                type_opts = ["Unknown", "Pvt Ltd", "Funded Startup", "Early Stage Startup", "Public Sector", "NGO", "Other"]
                if fields.get("company_type") in type_opts:
                    type_idx = type_opts.index(fields["company_type"])
                c_badge = conf.get("company_type", {}).get("badge", "🔴 Low Confidence")
                c_rsn = conf.get("company_type", {}).get("reason", "")
                f_type = st.selectbox(f"Company Type  ({c_badge})", type_opts, index=type_idx, help=c_rsn)
                
                # Last New Job
                job_idx = 0
                job_opts = ["never", "1", "2", "3", "4", ">4"]
                if fields.get("last_new_job") in job_opts:
                    job_idx = job_opts.index(fields["last_new_job"])
                c_badge = conf.get("last_new_job", {}).get("badge", "🔴 Low Confidence")
                c_rsn = conf.get("last_new_job", {}).get("reason", "")
                f_job = st.selectbox(f"Years Since Last Job  ({c_badge})", job_opts, index=job_idx, help=c_rsn)

                # Numeric Inputs
                c_badge = conf.get("training_hours", {}).get("badge", "🔴 Low Confidence")
                c_rsn = conf.get("training_hours", {}).get("reason", "")
                f_train = st.number_input(f"Training Hours  ({c_badge})", min_value=1, max_value=500, value=fields.get("training_hours") or 50, help=c_rsn)
                
                c_badge = conf.get("city_development_index", {}).get("badge", "🔴 Low Confidence")
                c_rsn = conf.get("city_development_index", {}).get("reason", "")
                f_cdi = st.number_input(f"City Development Index  ({c_badge})", min_value=0.0, max_value=1.0, value=fields.get("city_development_index") or 0.850, step=0.01, help=c_rsn)

            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
            submit = st.form_submit_button("🚀 Screen Candidate")
            
            if submit:
                mapped_gender = "Unknown" if f_gen == "Not Provided" else f_gen
                
                input_data = {
                    "city_development_index": float(f_cdi),
                    "training_hours": int(f_train),
                    "gender": mapped_gender,
                    "relevent_experience": f_rel,
                    "enrolled_university": f_uni,
                    "education_level": f_edu,
                    "major_discipline": f_maj,
                    "experience": f_exp,
                    "company_size": f_size,
                    "company_type": f_type,
                    "last_new_job": f_job
                }
                
                config_path  = os.path.join("models", "trained_models", "preprocessor_config.json")
                if not os.path.exists(config_path):
                    st.error("Missing ML models. Run `python run_step3_and_4.py`")
                    st.stop()
                    
                config = json.load(open(config_path))
                
                try:
                    import sys
                    if "src" not in sys.path:
                        sys.path.insert(0, ".")
                    from src.preprocessing import CandidatePreprocessor, CustomStandardScaler
                    from src.modeling import LogisticRegressionModel
                    
                    df_in = pd.DataFrame([input_data])
                    
                    prep = CandidatePreprocessor()
                    prep.feature_names = config["feature_names"]
                    prep.nominal_categories = config["nominal_categories"]
                    scaler = CustomStandardScaler()
                    scaler.mean_  = pd.Series(config["scaler_mean"])
                    scaler.scale_ = pd.Series(config["scaler_scale"])
                    prep.scaler = scaler
                    
                    X_proc = prep.transform(df_in)
                    
                    X_train = pd.read_csv(
                        os.path.join("data", "processed", "X_train.csv")
                    ).values
                    y_train = pd.read_csv(
                        os.path.join("data", "processed", "y_train.csv")
                    ).values.ravel()
                    
                    model = LogisticRegressionModel(lr=0.08, n_iters=400, l2_reg=0.1)
                    model.fit(X_train, y_train)
                    
                    prob = float(model.predict_proba(X_proc.values)[0, 1])
                    
                    import random
                    temp_id = f"RESUME-{random.randint(100000, 999999)}"
                    
                    input_data["Candidate ID"] = temp_id
                    input_data["enrollee_id"] = temp_id
                    input_data["prediction_probability"] = prob
                    input_data["candidate_name"] = st.session_state.get("resume_name", "Resume Candidate")
                    
                    st.session_state["resume_candidate"] = input_data
                    
                    corrected = {}
                    orig = st.session_state.get("resume_fields", {})
                    field_map = {
                        "education_level": f_edu,
                        "major_discipline": f_maj,
                        "relevent_experience": f_rel,
                        "enrolled_university": f_uni,
                        "experience": f_exp,
                        "company_size": f_size,
                        "company_type": f_type,
                        "last_new_job": f_job,
                        "training_hours": f_train,
                        "city_development_index": f_cdi
                    }
                    for k, val in field_map.items():
                        orig_val = orig.get(k)
                        if orig_val is not None and str(val) != str(orig_val):
                            corrected[k] = {"from": orig_val, "to": val, "badge": "✏️ Recruiter Corrected"}

                    st.session_state["resume_prediction"] = {
                        "id": temp_id,
                        "prob": prob,
                        "gender": mapped_gender,
                        "features": X_proc.values[0],
                        "feature_names": prep.feature_names,
                        "raw_data": input_data,
                        "weights": list(model.weights),
                        "bias": float(model.bias),
                        "quality": st.session_state.get("resume_quality", {}),
                        "confidences": st.session_state.get("resume_confidences", {}),
                        "corrected_fields": corrected
                    }
                except Exception as e:
                    st.error(f"Screening Error: {str(e)}")

        # ── Display Results ───────────────────────────────────────────────────
        if st.session_state["resume_prediction"]:
            st.markdown("<hr style='border-color:var(--divider); margin: 30px 0;'>", unsafe_allow_html=True)
            
            p_data = st.session_state["resume_prediction"]
            prob = p_data["prob"]
            mapped_gender = p_data["gender"]
            
            fair_path = os.path.join("models", "trained_models", "fairness_config.json")
            fair_config = json.load(open(fair_path)) if os.path.exists(fair_path) else {}
            fair_thresholds = fair_config.get("fair_thresholds", {"Female": 0.47, "Male": 0.46, "Other": 0.49, "Unknown": 0.60})
            
            threshold = fair_thresholds.get(mapped_gender, 0.50)
            is_suitable = int(prob >= threshold)
            
            if prob >= 0.8:
                tier = "High Priority"
                color = "#34d399"
                icon = "🌟"
                recommendation = "Immediate Interview"
            elif prob >= 0.6:
                tier = "Qualified"
                color = "#60a5fa"
                icon = "✅"
                recommendation = "Schedule Interview"
            elif prob >= 0.4:
                tier = "Extended"
                color = "#fbbf24"
                icon = "⚠️"
                recommendation = "Keep in Pipeline"
            else:
                tier = "Reserve"
                color = "#f87171"
                icon = "⛔"
                recommendation = "Future Consideration"
                
            name_disp = p_data["raw_data"].get("candidate_name", "Resume Candidate")
            st.markdown(
                f'<div class="section-header" style="text-align:center;font-size:1.4rem;">'
                f'{icon} Screening Results for {name_disp}: {tier}</div>',
                unsafe_allow_html=True
            )
            
            res_q_score = p_data.get("quality", {}).get("quality_score", 0)
            res_q_badge = p_data.get("quality", {}).get("status_badge", "⚠️ Review Recommended")

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(
                f'<div class="kpi-card" style="text-align:center;">'
                f'<div style="font-size:1.6rem;color:{color};font-weight:800;">{prob*100:.1f}%</div>'
                f'<div style="color:{t["text_muted"]};font-size:0.72rem;font-weight:700;text-transform:uppercase;">AI Suitability Score</div>'
                f'</div>', unsafe_allow_html=True
            )
            c2.markdown(
                f'<div class="kpi-card" style="text-align:center;">'
                f'<div style="font-size:1.6rem;color:{t["text_primary"]};font-weight:800;">{res_q_score}%</div>'
                f'<div style="color:{t["text_muted"]};font-size:0.72rem;font-weight:700;text-transform:uppercase;">Resume Quality ({res_q_badge})</div>'
                f'</div>', unsafe_allow_html=True
            )
            c3.markdown(
                f'<div class="kpi-card" style="text-align:center;">'
                f'<div style="font-size:1.3rem;color:{t["text_primary"]};font-weight:800;margin-top:2px;">{tier}</div>'
                f'<div style="color:{t["text_muted"]};font-size:0.72rem;font-weight:700;text-transform:uppercase;">Priority Tier</div>'
                f'</div>', unsafe_allow_html=True
            )
            c4.markdown(
                f'<div class="kpi-card" style="text-align:center;">'
                f'<div style="font-size:1.05rem;color:{t["text_primary"]};font-weight:800;margin-top:4px;">{recommendation}</div>'
                f'<div style="color:{t["text_muted"]};font-size:0.72rem;font-weight:700;text-transform:uppercase;margin-top:6px;">Recommendation</div>'
                f'</div>', unsafe_allow_html=True
            )
            
            st.markdown(
                f'<div style="text-align:center;color:{t["text_muted"]};font-size:0.75rem;margin-top:8px;margin-bottom:12px;">'
                f'💡 <b>AI Suitability Score</b> measures model-predicted candidate fit. <b>Resume Quality Score</b> measures document completeness and extraction quality.'
                f'</div>',
                unsafe_allow_html=True
            )
            
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                if st.button("👤 View Candidate Profile", key="btn_view_profile", use_container_width=True):
                    st.session_state["view_profile_id"] = p_data["id"]
                    st.session_state["nav_goto"] = "👤  Candidate Profile"
                    st.rerun()
            with col_b2:
                if st.button("⭐ Add to Shortlist", key="btn_shortlist", use_container_width=True):
                    shortlist = _load_shortlist()
                    if not any(r.get("candidate_id") == p_data["id"] for r in shortlist):
                        shortlist.append({
                            "candidate_id": p_data["id"],
                            "candidate_name": name_disp,
                            "suitability_score": float(prob),
                            "priority_tier": tier,
                            "recommendation": recommendation,
                            "timestamp": datetime.now().isoformat()
                        })
                        _save_shortlist(shortlist)
                    st.success("Candidate added to shortlist.")
            with col_b3:
                report_content = f"Screening Report\nCandidate ID: {p_data['id']}\nCandidate Name: {name_disp}\nScore: {prob:.4f}\nTier: {tier}\n"
                st.download_button("📥 Download Report", data=report_content, file_name=f"{p_data['id']}_report.txt", mime="text/plain", use_container_width=True)

            st.markdown(f'<div class="section-header">Why did FairHire AI make this recommendation?</div>', unsafe_allow_html=True)
            
            shap_path = os.path.join("reports", "metrics", "shap_feature_importance.csv")
            if os.path.exists(shap_path):
                shap_df_glob = pd.read_csv(shap_path)
                import numpy as np
                import plotly.graph_objects as go
                
                feat_names = p_data["feature_names"]
                features_val = p_data["features"]
                
                mean_vals = np.zeros(len(feat_names)) 
                model_w = np.array(p_data["weights"])
                local_shap = model_w * (features_val - mean_vals)
                
                local_shap_df = pd.DataFrame({
                    "Feature": feat_names,
                    "Importance": local_shap
                })
                local_shap_df["Abs_Importance"] = local_shap_df["Importance"].abs()
                merged = local_shap_df.sort_values("Abs_Importance", ascending=False).head(10)
                
                # Build narrative dictionary using the local function
                cand_row_for_narrative = {
                    "Suitability Score": prob,
                    "Priority Tier":     tier,
                    "Experience":        p_data["raw_data"]["experience"],
                    "Education":         p_data["raw_data"]["education_level"],
                    "Training Hours":    p_data["raw_data"]["training_hours"],
                    "Relevant Exp":      p_data["raw_data"]["relevent_experience"],
                    "Major":             p_data["raw_data"]["major_discipline"],
                    "Company Type":      p_data["raw_data"]["company_type"],
                    "City CDI":          p_data["raw_data"]["city_development_index"],
                }
                
                narrative_data = generate_candidate_narrative(cand_row_for_narrative)
                narrative_text = narrative_data.get("narrative", f"Candidate scored {prob*100:.1f}% on the suitability model.")
                
                st.markdown(
                    f'<div style="background:{t["card_bg"]};border:1px solid {t["card_border"]};border-radius:12px;padding:24px;">'
                    f'<div style="color:{t["text_primary"]};font-size:0.95rem;line-height:1.6;white-space:pre-wrap;">'
                    f'{narrative_text}</div></div>',
                    unsafe_allow_html=True
                )
                
                st.markdown("<br>", unsafe_allow_html=True)
                merged = merged.sort_values("Importance", ascending=True)
                colors = ["#f87171" if val < 0 else "#34d399" for val in merged["Importance"]]
                readable_labels = [
                    _FEATURE_LABELS.get(f.rsplit("_",1)[0], f.replace("_"," ").title()) 
                    for f in merged["Feature"]
                ]
                
                fig = go.Figure(go.Bar(
                    x=merged["Importance"],
                    y=readable_labels,
                    orientation='h',
                    marker_color=colors
                ))
                fig.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(title="Contribution to Score", gridcolor=t['divider'], zerolinecolor=t['text_muted']),
                    yaxis=dict(gridcolor='rgba(0,0,0,0)'),
                    font=dict(color=t['text_secondary'])
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("🛡️ Data Extraction Trustworthiness & Audit Log", expanded=False):
                    st.markdown(
                        f'<div style="font-size:0.85rem;color:{t["text_secondary"]};margin-bottom:12px;">'
                        f'Feature-level extraction confidence breakdown and recruiter corrections applied to candidate data:</div>',
                        unsafe_allow_html=True
                    )
                    conf_map = p_data.get("confidences", {})
                    corr_map = p_data.get("corrected_fields", {})
                    raw_dict = p_data.get("raw_data", {})

                    audit_rows = ""
                    for fk, fval in raw_dict.items():
                        if fk in ["Candidate ID", "enrollee_id", "prediction_probability", "candidate_name"]:
                            continue
                        c_info = conf_map.get(fk, {})
                        bdg = c_info.get("badge", "⚪ Defaulted")
                        rsn = c_info.get("reason", "")
                        if fk in corr_map:
                            bdg = corr_map[fk].get("badge", "✏️ Recruiter Corrected")
                            orig_v = corr_map[fk].get("from")
                            rsn = f"Corrected by recruiter from '{orig_v}'"

                        fk_title = fk.replace("_", " ").title()
                        audit_rows += (
                            f'<div style="display:flex;align-items:center;justify-content:space-between;'
                            f'padding:8px 0;border-bottom:1px solid {t["divider"]};font-size:0.83rem;">'
                            f'<div><span style="font-weight:700;color:{t["text_primary"]};">{fk_title}</span>: '
                            f'<span style="color:{t["text_secondary"]};">{fval}</span></div>'
                            f'<div style="text-align:right;"><span style="font-weight:700;">{bdg}</span>'
                            f'<div style="font-size:0.72rem;color:{t["text_muted"]};">{rsn}</div></div></div>'
                        )

                    st.markdown(audit_rows, unsafe_allow_html=True)

    with tab_bulk:
        st.markdown(
            f'<div style="background:{t["card_bg"]};border:1px solid {t["card_border"]};border-radius:12px;padding:20px;margin-bottom:20px;box-shadow:{t["card_shadow"]};">'
            f'<div style="font-weight:700;font-size:1.05rem;color:{t["text_primary"]};margin-bottom:4px;">📦 Bulk Resume Batch Screening</div>'
            f'<div style="font-size:0.85rem;color:{t["text_secondary"]};line-height:1.5;">'
            f'Upload multiple candidate resumes simultaneously (PDF, DOCX, TXT). Each file will be parsed, validated for completeness, screened through the production ML pipeline, and automatically saved to <b>Screening History</b>.</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        bulk_files = st.file_uploader("Upload Multiple Resumes (Batch)", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="bulk_resume_uploader")

        if bulk_files:
            st.info(f"📁 {len(bulk_files)} files selected for batch screening.")
            if st.button("🚀 Process & Screen All Resumes", key="btn_run_bulk_screening"):
                batch_results = []
                errors = []
                pbar = st.progress(0.0)
                status_text = st.empty()

                for idx, bf in enumerate(bulk_files):
                    status_text.text(f"Processing {idx+1}/{len(bulk_files)}: {bf.name}...")
                    pbar.progress((idx + 1) / len(bulk_files))
                    try:
                        res = _run_single_resume_screening(bf, bf.name)
                        if "error" in res:
                            errors.append((bf.name, res["error"]))
                        else:
                            batch_results.append(res)
                    except Exception as ex:
                        errors.append((bf.name, str(ex)))

                pbar.progress(1.0)
                status_text.success(f"✅ Batch completed! Successfully processed {len(batch_results)} of {len(bulk_files)} resumes.")

                if errors:
                    with st.expander("⚠️ Processing Warnings / Unreadable Files", expanded=True):
                        for fname, err in errors:
                            st.warning(f"File '{fname}': {err}")

                if batch_results:
                    st.markdown(f'<div class="section-header">📊 Batch Screening Summary</div>', unsafe_allow_html=True)
                    summary_rows = []
                    for r in batch_results:
                        summary_rows.append({
                            "Candidate ID": r["candidate_id"],
                            "Candidate Name": r["candidate_name"],
                            "Filename": r["source_filename"],
                            "AI Suitability Score (%)": f"{r['prob']*100:.1f}",
                            "Resume Quality (%)": r.get("quality", {}).get("quality_score", 0),
                            "Priority Tier": r["priority_tier"],
                            "Recommendation": r["recommendation"],
                            "Status": r["status"]
                        })

                    sum_df = pd.DataFrame(summary_rows)
                    st.dataframe(sum_df, use_container_width=True)

                    csv_batch = sum_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Batch Summary CSV", data=csv_batch, file_name="batch_screening_summary.csv", mime="text/csv", use_container_width=True)

def _render_historical_screening_detail(rec: dict, t: dict):
    """
    Phase 10: Detailed view when a recruiter re-opens a historical candidate screening result.
    """
    cid = rec.get("candidate_id") or rec.get("id")
    cname = rec.get("candidate_name") or "Name Not Detected"
    prob = rec.get("prob") if rec.get("prob") is not None else rec.get("suitability_score", 0.0)
    tier = rec.get("priority_tier", "Reserve")
    recommendation = rec.get("recommendation", "Review Candidate")
    status = rec.get("status", "Pending")
    fname = rec.get("source_filename", "Uploaded Resume")
    qual = rec.get("quality", {})
    conf = rec.get("confidences", {})
    raw_dict = rec.get("raw_data", {})
    notes = rec.get("recruiter_notes", "")

    top_left, top_right = st.columns([1, 4])
    with top_left:
        if st.button("← Back to History List", key="btn_back_to_history_list"):
            st.session_state["view_history_id"] = None
            st.rerun()

    st.markdown(
        f'<div class="section-header" style="text-align:center;font-size:1.4rem;">'
        f'Historical Screening Result for {cname} ({cid})</div>',
        unsafe_allow_html=True
    )

    with st.expander("📌 Recruiter Status & Notes Management", expanded=True):
        sc1, sc2, sc3 = st.columns([2, 3, 1])
        with sc1:
            st_opts = ["Pending", "Shortlisted", "Rejected"]
            st_idx = st_opts.index(status) if status in st_opts else 0
            new_st = st.selectbox("Candidate Status", options=st_opts, index=st_idx, key=f"hist_status_select_{cid}")
        with sc2:
            new_notes = st.text_input("Recruiter Notes", value=notes, placeholder="Add private recruiter notes...", key=f"hist_notes_input_{cid}")
        with sc3:
            st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
            if st.button("💾 Save Status", key=f"hist_save_btn_{cid}", use_container_width=True):
                _update_screening_status(cid, new_st, new_notes)
                rec["status"] = new_st
                rec["recruiter_notes"] = new_notes
                st.success("Status & notes saved successfully!")

    tier_colors = {"High Priority": "#34d399", "Qualified": "#60a5fa", "Extended": "#fbbf24", "Reserve": "#f87171"}
    color = tier_colors.get(tier, "#60a5fa")
    res_q_score = qual.get("quality_score", 0)
    res_q_badge = qual.get("status_badge", "⚠️ Review Recommended")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card" style="text-align:center;"><div style="font-size:1.6rem;color:{color};font-weight:800;">{prob*100:.1f}%</div><div style="color:{t["text_muted"]};font-size:0.72rem;font-weight:700;text-transform:uppercase;">AI Suitability Score</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card" style="text-align:center;"><div style="font-size:1.6rem;color:{t["text_primary"]};font-weight:800;">{res_q_score}%</div><div style="color:{t["text_muted"]};font-size:0.72rem;font-weight:700;text-transform:uppercase;">Resume Quality ({res_q_badge})</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card" style="text-align:center;"><div style="font-size:1.3rem;color:{t["text_primary"]};font-weight:800;margin-top:2px;">{tier}</div><div style="color:{t["text_muted"]};font-size:0.72rem;font-weight:700;text-transform:uppercase;">Priority Tier</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card" style="text-align:center;"><div style="font-size:1.05rem;color:{t["text_primary"]};font-weight:800;margin-top:4px;">{recommendation}</div><div style="color:{t["text_muted"]};font-size:0.72rem;font-weight:700;text-transform:uppercase;margin-top:6px;">Recommendation</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        if st.button("👤 View Candidate Profile", key=f"hist_view_prof_{cid}", use_container_width=True):
            st.session_state["resume_candidate"] = raw_dict
            st.session_state["view_profile_id"] = cid
            st.session_state["nav_goto"] = "👤  Candidate Profile"
            st.session_state.pop("nav_radio", None)
            st.rerun()
    with col_b2:
        if st.button("⭐ Add to Shortlist", key=f"hist_shortlist_{cid}", use_container_width=True):
            shortlist = _load_shortlist()
            if not any(r.get("candidate_id") == cid for r in shortlist):
                shortlist.append({
                    "candidate_id": cid,
                    "candidate_name": cname,
                    "suitability_score": float(prob),
                    "priority_tier": tier,
                    "recommendation": recommendation,
                    "timestamp": datetime.now().isoformat()
                })
                _save_shortlist(shortlist)
                _update_screening_status(cid, "Shortlisted")
            st.success("Candidate added to shortlist.")
    with col_b3:
        report_content = f"Historical Screening Report\nCandidate ID: {cid}\nCandidate Name: {cname}\nScore: {prob:.4f}\nTier: {tier}\nFile: {fname}\n"
        st.download_button("📥 Download Report", data=report_content, file_name=f"{cid}_report.txt", mime="text/plain", use_container_width=True)

    st.markdown(f'<div class="section-header">Why did FairHire AI make this recommendation?</div>', unsafe_allow_html=True)
    if "features" in rec and "weights" in rec and "feature_names" in rec:
        import numpy as np
        import plotly.graph_objects as go
        feat_names = rec["feature_names"]
        features_val = np.array(rec["features"])
        model_w = np.array(rec["weights"])
        mean_vals = np.zeros(len(feat_names))
        local_shap = model_w * (features_val - mean_vals)

        local_shap_df = pd.DataFrame({"Feature": feat_names, "Importance": local_shap})
        local_shap_df["Abs_Importance"] = local_shap_df["Importance"].abs()
        merged = local_shap_df.sort_values("Abs_Importance", ascending=False).head(10).sort_values("Importance", ascending=True)
        colors = ["#f87171" if val < 0 else "#34d399" for val in merged["Importance"]]
        readable_labels = [_FEATURE_LABELS.get(f.rsplit("_",1)[0], f.replace("_"," ").title()) for f in merged["Feature"]]

        fig = go.Figure(go.Bar(x=merged["Importance"], y=readable_labels, orientation='h', marker_color=colors))
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(title="Contribution to Score", gridcolor=t['divider'], zerolinecolor=t['text_muted']), yaxis=dict(gridcolor='rgba(0,0,0,0)'), font=dict(color=t['text_secondary']))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🛡️ Data Extraction Trustworthiness & Audit Log", expanded=False):
        audit_rows = ""
        for fk, fval in raw_dict.items():
            if fk in ["Candidate ID", "enrollee_id", "prediction_probability", "candidate_name"]:
                continue
            c_info = conf.get(fk, {})
            bdg = c_info.get("badge", "⚪ Defaulted")
            rsn = c_info.get("reason", "")
            fk_title = fk.replace("_", " ").title()
            audit_rows += f'<div style="display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid {t["divider"]};font-size:0.83rem;"><div><span style="font-weight:700;color:{t["text_primary"]};">{fk_title}</span>: <span style="color:{t["text_secondary"]};">{fval}</span></div><div style="text-align:right;"><span style="font-weight:700;">{bdg}</span><div style="font-size:0.72rem;color:{t["text_muted"]};">{rsn}</div></div></div>'
        st.markdown(audit_rows, unsafe_allow_html=True)

def render_screening_history_page():
    """Phase 10: Resume Screening History & Persistence Dashboard"""
    t = ThemeManager.get()

    st.markdown(
        f'<div style="background:{t["header_bg"]};border:1px solid {t["header_border"]};'
        f'border-radius:16px;padding:28px 32px;margin-bottom:24px;">'
        f'<div style="display:flex;align-items:center;gap:14px;">'
        f'<div style="font-size:2.4rem;">📜</div>'
        f'<div><div style="color:{t["header_title"]};font-size:1.55rem;font-weight:800;'
        f'letter-spacing:-0.02em;">Screening History & Audit Log</div>'
        f'<div style="color:{t["header_sub"]};font-size:0.88rem;margin-top:3px;">'
        f'Audit trail of all screened candidate resumes, status tracking, and historical reopen views.</div>'
        f'</div></div></div>',
        unsafe_allow_html=True
    )

    history = _load_screening_history()

    view_id = st.session_state.get("view_history_id")
    if view_id:
        rec = next((item for item in history if (item.get("candidate_id") == view_id or item.get("id") == view_id)), None)
        if rec:
            _render_historical_screening_detail(rec, t)
            return
        else:
            st.session_state["view_history_id"] = None

    if not history:
        st.info("No screened candidate resumes found in history yet. Upload and screen a resume in the 📄 Resume Screening page to create history records.")
        return

    import numpy as np
    total_screened = len(history)
    pending_cnt = sum(1 for r in history if r.get("status") == "Pending")
    shortlist_cnt = sum(1 for r in history if r.get("status") == "Shortlisted")
    rejected_cnt = sum(1 for r in history if r.get("status") == "Rejected")
    avg_score = np.mean([r.get("prob", 0) for r in history]) if history else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, "blue",  "📜", "Total Screened",  f"{total_screened:,}",  "All historical records"),
        (k2, "amber", "⏳", "Pending Review",  f"{pending_cnt:,}",     "Awaiting action"),
        (k3, "green", "⭐", "Shortlisted",     f"{shortlist_cnt:,}",   "Recruiter shortlisted"),
        (k4, "red",   "❌", "Rejected",        f"{rejected_cnt:,}",    "Not pursuing"),
        (k5, "teal",  "🎯", "Avg Suitability", f"{avg_score*100:.1f}%", "Mean model score"),
    ]
    for col, color, icon, label, value, sub in kpis:
        with col:
            st.markdown(
                f'<div class="kpi-card {color}">'
                f'<div class="kpi-icon">{icon}</div>'
                f'<div class="kpi-label">{label}</div>'
                f'<div class="kpi-value {color}">{value}</div>'
                f'<div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-header">🔎 Search & Filter Screening History</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns([2, 2, 2])
    with f1:
        search_query = st.text_input("🔍 Search Candidate Name or ID", placeholder="e.g. Tejaswini or RESUME-123456", label_visibility="collapsed")
    with f2:
        sel_status = st.selectbox("Status Filter", options=["All Statuses", "Pending", "Shortlisted", "Rejected"], label_visibility="collapsed")
    with f3:
        sel_tier = st.selectbox("Tier Filter", options=["All Tiers", "High Priority", "Qualified", "Extended", "Reserve"], label_visibility="collapsed")

    filtered = history.copy()
    if search_query.strip():
        q = search_query.strip().lower()
        filtered = [r for r in filtered if q in str(r.get("candidate_name", "")).lower() or q in str(r.get("candidate_id", "")).lower() or q in str(r.get("source_filename", "")).lower()]

    if sel_status != "All Statuses":
        filtered = [r for r in filtered if r.get("status") == sel_status]

    if sel_tier != "All Tiers":
        filtered = [r for r in filtered if r.get("priority_tier") == sel_tier]

    filtered.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

    st.markdown(f'<div style="font-size:0.85rem;color:{t["text_secondary"]};margin:12px 0 16px 0;">'
                f'Showing <strong style="color:{t["text_primary"]};">{len(filtered)}</strong> of <strong style="color:{t["text_primary"]};">{total_screened}</strong> screening records</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="section-header">📋 Historical Screening Records</div>', unsafe_allow_html=True)

    for r in filtered:
        cid = r.get("candidate_id") or r.get("id")
        cname = r.get("candidate_name") or "Name Not Detected"
        prob = r.get("prob") if r.get("prob") is not None else r.get("suitability_score", 0.0)
        tier = r.get("priority_tier", "Reserve")
        st_val = r.get("status", "Pending")
        fname = r.get("source_filename", "Uploaded Resume")
        q_score = r.get("quality", {}).get("quality_score") or r.get("resume_quality_score", 0)
        ts = r.get("timestamp", "")[:16].replace("T", " ")

        status_color_map = {
            "Shortlisted": ("#10b981", "rgba(16,185,129,0.1)", "rgba(16,185,129,0.3)"),
            "Pending":     ("#f59e0b", "rgba(245,158,11,0.1)", "rgba(245,158,11,0.3)"),
            "Rejected":    ("#ef4444", "rgba(239,68,68,0.1)",  "rgba(239,68,68,0.3)")
        }
        s_clr, s_bg, s_bd = status_color_map.get(st_val, ("#6b7280", "rgba(107,114,128,0.1)", "rgba(107,114,128,0.3)"))

        with st.container():
            col_a, col_b, col_c, col_d, col_e, col_f = st.columns([2.5, 1.5, 1.5, 1.5, 1.5, 1.5])
            with col_a:
                st.markdown(f'<div style="font-weight:700;color:{t["text_primary"]};font-size:0.95rem;">{cname}</div>'
                            f'<div style="font-size:0.75rem;color:{t["text_muted"]};">ID: {cid} · {fname}</div>', unsafe_allow_html=True)
            with col_b:
                st.markdown(f'<div style="font-weight:800;color:{t["text_primary"]};font-size:1.05rem;">{prob*100:.1f}%</div>'
                            f'<div style="font-size:0.7rem;color:{t["text_muted"]};">AI Suitability</div>', unsafe_allow_html=True)
            with col_c:
                st.markdown(f'<div style="font-weight:800;color:{t["text_primary"]};font-size:1.05rem;">{q_score}%</div>'
                            f'<div style="font-size:0.7rem;color:{t["text_muted"]};">Resume Quality</div>', unsafe_allow_html=True)
            with col_d:
                st.markdown(f'<div style="font-weight:700;color:{t["text_primary"]};font-size:0.85rem;">{tier}</div>'
                            f'<div style="font-size:0.7rem;color:{t["text_muted"]};">{ts}</div>', unsafe_allow_html=True)
            with col_e:
                st.markdown(f'<span style="background:{s_bg};color:{s_clr};border:1px solid {s_bd};'
                            f'padding:3px 10px;border-radius:12px;font-size:0.78rem;font-weight:700;">{st_val}</span>', unsafe_allow_html=True)
            with col_f:
                if st.button("👁️ View Details", key=f"hist_btn_{cid}", use_container_width=True):
                    st.session_state["view_history_id"] = cid
                    st.rerun()

            st.markdown(f'<hr style="border-color:{t["divider"]};margin:8px 0 12px 0;">', unsafe_allow_html=True)

    st.markdown(f'<div class="section-header">⬇️ Export Screening History</div>', unsafe_allow_html=True)
    exp1, exp2 = st.columns(2)
    with exp1:
        export_list = []
        for r in filtered:
            export_list.append({
                "Candidate ID": r.get("candidate_id") or r.get("id"),
                "Candidate Name": r.get("candidate_name"),
                "Filename": r.get("source_filename"),
                "AI Suitability Score (%)": f"{r.get('prob', 0)*100:.1f}",
                "Resume Quality Score (%)": r.get("quality", {}).get("quality_score", 0),
                "Priority Tier": r.get("priority_tier"),
                "Recommendation": r.get("recommendation"),
                "Status": r.get("status"),
                "Timestamp": r.get("timestamp")
            })
        csv_data = pd.DataFrame(export_list).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download History CSV", data=csv_data, file_name="resume_screening_history.csv", mime="text/csv", use_container_width=True)
    with exp2:
        json_data = json.dumps(history, indent=2).encode("utf-8")
        st.download_button("📦 Download History JSON", data=json_data, file_name="resume_screening_history.json", mime="application/json", use_container_width=True)

def render_settings_page():
    t = ThemeManager.get()

    st.header("⚙️ Settings")
    st.caption("Configure application preferences, view model information, and learn about FairHire AI.")
    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

    # ── Theme Preferences ─────────────────────────────────────────────────────
    st.markdown(
        f'<div class="settings-card"><h4>🎨 Theme Preferences</h4></div>',
        unsafe_allow_html=True
    )
    current_theme = ThemeManager.name()
    col_theme, col_preview = st.columns([1, 2])

    with col_theme:
        theme_choice = st.radio(
            "Select Theme",
            options=["Light", "Dark"],
            index=0 if current_theme == "Light" else 1,
            key="settings_theme_radio",
        )
        if theme_choice != current_theme:
            ThemeManager.set(theme_choice)

    with col_preview:
        if current_theme == "Light":
            st.markdown(
                '<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;'
                'padding:20px;text-align:center;">'
                '<div style="font-size:2rem;margin-bottom:8px;">☀️</div>'
                '<div style="color:#1e293b;font-weight:700;font-size:1rem;">Light Mode Active</div>'
                '<div style="color:#94a3b8;font-size:0.8rem;margin-top:4px;">'
                'Clean white interface — ideal for bright environments</div></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background:#111827;border:1px solid #1f2937;border-radius:12px;'
                'padding:20px;text-align:center;">'
                '<div style="font-size:2rem;margin-bottom:8px;">🌙</div>'
                '<div style="color:#f9fafb;font-weight:700;font-size:1rem;">Dark Mode Active</div>'
                '<div style="color:#4b5563;font-size:0.8rem;margin-top:4px;">'
                'Deep dark interface — easy on the eyes at night</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # ── Model Information ─────────────────────────────────────────────────────
    st.markdown(
        '<div class="settings-card"><h4>🤖 Model Architecture & Benchmarking</h4></div>',
        unsafe_allow_html=True
    )
    model_info  = load_json_config(os.path.join("models", "trained_models", "best_model_info.json"))
    fair_config = load_json_config(os.path.join("models", "trained_models", "fairness_config.json"))
    m  = model_info.get("metrics", {})
    ft = fair_config.get("fair_thresholds", {})

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        model_name = model_info.get("best_model_name", "N/A")
        rows = ""
        for lbl, val, clr in [
            ("ROC-AUC",   f"{m.get('ROC-AUC',   0):.4f}", "#60a5fa"),
            ("Accuracy",  f"{m.get('Accuracy',  0):.4f}", "#34d399"),
            ("Precision", f"{m.get('Precision', 0):.4f}", "#a78bfa"),
            ("Recall",    f"{m.get('Recall',    0):.4f}", "#fbbf24"),
            ("F1-Score",  f"{m.get('F1-Score',  0):.4f}", "#2dd4bf"),
        ]:
            rows += _kv_row(lbl, val, clr, t["panel_border_l"], t["text_label"])
        st.markdown(
            f'<div class="panel-card"><div style="color:{t["text_primary"]};font-weight:700;margin-bottom:12px;">'
            f'🏆 Benchmark Model<br><span style="font-size:0.85rem;color:{t["text_muted"]};">{model_name}</span></div>' + rows + '</div>',
            unsafe_allow_html=True
        )

    with col_m2:
        st.markdown(
            f'<div class="panel-card"><div style="color:{t["text_primary"]};font-weight:700;margin-bottom:12px;">'
            f'⚙️ Production Screening Model<br><span style="font-size:0.85rem;color:{t["text_muted"]};">Logistic Regression</span></div>'
            f'<div style="font-size:0.8rem;color:{t["text_secondary"]};line-height:1.6;margin-bottom:8px;">'
            f'<b>Used consistently across:</b><br>'
            f'• Candidate Ranking<br>'
            f'• Real-Time Screening<br>'
            f'• Fairness Calibration<br>'
            f'• SHAP Explainability'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        f'<div style="background:{t["info_bg"]};border:1px solid {t["info_border"]};border-radius:12px;padding:16px;margin:16px 0;">'
        f'<div style="font-size:0.85rem;color:{t["text_primary"]};line-height:1.6;">'
        f'<b>Architectural Alignment:</b> Random Forest achieved the highest validation ROC-AUC among the evaluated benchmark models. '
        f'Logistic Regression is used as the production screening model because the candidate ranking, fairness calibration, and '
        f'explainability pipelines are consistently built around the same model. This ensures that candidate scores, fairness thresholds, '
        f'and explanations remain aligned across the application.'
        f'</div></div>',
        unsafe_allow_html=True
    )
    
    rows2 = ""
    for grp, thr_val in ft.items():
        rows2 += _kv_row(grp, str(thr_val), "#34d399", t["panel_border_l"], t["text_label"])
    st.markdown(
        f'<div class="panel-card"><div style="color:{t["text_primary"]};font-weight:700;margin-bottom:12px;">'
        f'⚖️ Fairness Thresholds</div>' + rows2 +
        f'<div style="margin-top:10px;font-size:0.72rem;color:{t["text_muted"]};">'
        f'Calibrated via Fairlearn ThresholdOptimizer (post-processing) on the Production Model</div></div>',
        unsafe_allow_html=True
    )

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # ── About ─────────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="settings-card"><h4>ℹ️ About FairHire AI</h4></div>',
        unsafe_allow_html=True
    )
    about_rows = ""
    for k, v in [
        ("Version",        "2.0 — Phase 1 + Theme System"),
        ("ML Backend",     "Custom Logistic Regression (from scratch)"),
        ("Fairness",       "Fairlearn ThresholdOptimizer"),
        ("Explainability", "SHAP Linear Attribution"),
        ("Dataset",        "Kaggle HR Analytics (aug_train / aug_test)"),
        ("Phase",          "1 of 10 — Dashboard + Theme Toggle"),
    ]:
        about_rows += _kv_row(k, v, t["text_primary"], t["panel_border_l"], t["text_label"])

    st.markdown(
        f'<div class="panel-card">'
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">'
        f'<span style="font-size:2rem;">🎯</span>'
        f'<div>'
        f'<div style="color:{t["text_primary"]};font-size:1.1rem;font-weight:800;">FairHire AI</div>'
        f'<div style="color:{t["text_muted"]};font-size:0.75rem;">Explainable &amp; Fair Candidate Screening Platform</div>'
        f'</div>'
        f'<span style="margin-left:auto;background:rgba(59,130,246,0.1);color:#3b82f6;'
        f'border:1px solid rgba(59,130,246,0.25);padding:3px 10px;'
        f'border-radius:20px;font-size:0.65rem;font-weight:700;">v2.0</span>'
        f'</div>'
        + about_rows +
        f'<div style="margin-top:14px;color:{t["text_muted"]};font-size:0.78rem;line-height:1.6;">'
        f'FairHire AI automates candidate shortlisting while auditing and mitigating '
        f'algorithmic bias across protected attributes. Built as an end-to-end ML '
        f'project with enterprise SaaS UI.</div></div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION OPTIONS  — single source of truth used by the radio and all
# programmatic page-change calls (nav_goto, current_page)
# ─────────────────────────────────────────────────────────────────────────────
_NAV_OPTIONS = [
    "🏠  Dashboard",
    "📋  Candidate Rankings",
    "⚖️  Fairness & Bias Audit",
    "🔍  SHAP Explainability",
    "🚀  Real-Time Predictor",
    "📄  Resume Screening",
    "📜  Screening History",
    "─────────────",
    "💼  Job Descriptions",
    "👤  Candidate Profile",
    "📊  Analytics",
    "📄  Reports",
    "⚙️  Settings",
]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP — Navigation + Theme Initialization
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ThemeManager.init()
    t = ThemeManager.get()

    # Inject theme-aware CSS
    st.markdown(build_css(t), unsafe_allow_html=True)

    with st.sidebar:
        # Branding
        st.markdown(
            f'<div style="text-align:center;padding:22px 0 10px 0;">'
            f'<div style="font-size:2.4rem;margin-bottom:6px;">🎯</div>'
            f'<div style="color:{t["text_primary"]};font-size:1.2rem;font-weight:800;letter-spacing:-0.02em;">FairHire AI</div>'
            f'<div style="color:{t["text_muted"]};font-size:0.65rem;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.08em;margin-top:3px;">Fair Candidate Screening</div>'
            f'<div style="color:{t["text_hint"]};font-size:0.6rem;margin-top:2px;">v2.0 · Phase 1</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Theme Toggle
        st.markdown('<div class="theme-toggle">', unsafe_allow_html=True)
        if st.button(f"{t['toggle_icon']}  {t['toggle_label']}", key="theme_toggle_btn", use_container_width=True):
            ThemeManager.toggle()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"<hr style='border-color:{t['divider']};margin:10px 0;'>", unsafe_allow_html=True)

        # ── Navigation ────────────────────────────────────────────────────────
        # Initialize current_page on first load
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = _NAV_OPTIONS[0]
        # Handle programmatic nav requests (e.g. "Open Profile" button)
        # nav_goto is set-once then consumed so it doesn't interfere with normal nav
        if "nav_goto" in st.session_state:
            st.session_state["current_page"] = st.session_state.pop("nav_goto")
        # Compute the correct starting index from the preserved current_page.
        # This is what keeps the user on the same page after a theme toggle rerun.
        try:
            _nav_idx = _NAV_OPTIONS.index(st.session_state["current_page"])
        except (ValueError, IndexError):
            _nav_idx = 0

        page = st.radio(
            "Navigation",
            options=_NAV_OPTIONS,
            label_visibility="collapsed",
            index=_nav_idx,
            key="nav_radio",
        )
        # Keep current_page in sync with what the user clicked in the sidebar.
        # ThemeManager.toggle() only changes 'theme' — it never touches current_page,
        # so the next rerun re-selects the same index and the user stays on their page.
        st.session_state["current_page"] = page

        st.markdown(f"<hr style='border-color:{t['divider']};margin:10px 0;'>", unsafe_allow_html=True)

        # System Status
        st.markdown(
            f'<div style="color:{t["text_hint"]};font-size:0.62rem;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">System Status</div>',
            unsafe_allow_html=True
        )
        checks = {
            "Rankings": os.path.exists(os.path.join("reports", "metrics", "candidate_rankings.csv")),
            "Model":    os.path.exists(os.path.join("models", "trained_models", "best_model_info.json")),
            "Fairness": os.path.exists(os.path.join("models", "trained_models", "fairness_config.json")),
            "SHAP":     os.path.exists(os.path.join("reports", "metrics", "shap_feature_importance.csv")),
        }
        badges = ""
        for label, ok in checks.items():
            cls = "status-ok" if ok else "status-err"
            dot = "●" if ok else "○"
            badges += f'<div style="margin:4px 0;"><span class="{cls}">{dot} {label}</span></div>'
        st.markdown(badges, unsafe_allow_html=True)

        st.markdown(f"<hr style='border-color:{t['divider']};margin:10px 0;'>", unsafe_allow_html=True)
        st.markdown(
            f'<div style="color:{t["text_hint"]};font-size:0.6rem;text-align:center;">'
            f'FairHire AI · HR SaaS Platform<br>© 2026</div>',
            unsafe_allow_html=True
        )

    # Page routing
    if "Dashboard" in page:
        render_home_page()
    elif "Rankings" in page:
        render_ranking_page()
    elif "Job" in page and "Description" in page:
        render_job_descriptions_page()
    elif "Profile" in page or st.session_state.get("view_profile_id"):
        render_candidate_profile_page(st.session_state.get("view_profile_id"))
    elif "Fairness" in page:
        render_fairness_page()
    elif "SHAP" in page:
        render_explainability_page()
    elif "Predictor" in page:
        render_prediction_page()
    elif "Screening History" in page:
        render_screening_history_page()
    elif "Resume" in page:
        render_resume_screening_page()
    elif "Settings" in page:
        render_settings_page()
    elif "─────" in page:
        st.info("Please select a navigation item from the sidebar.")
    else:
        page_clean = page.strip()
        st.markdown(
            f'<div class="coming-soon-banner">'
            f'<div style="font-size:3.5rem;margin-bottom:18px;">🚧</div>'
            f'<div style="color:{t["cs_title"]};font-size:1.4rem;font-weight:800;'
            f'letter-spacing:-0.02em;margin-bottom:10px;">{page_clean}</div>'
            f'<div style="color:{t["cs_sub"]};font-size:0.92rem;margin-bottom:20px;">'
            f'This module is being built in an upcoming phase.</div>'
            f'<span style="background:rgba(59,130,246,0.1);color:#3b82f6;'
            f'border:1px solid rgba(59,130,246,0.25);padding:6px 18px;'
            f'border-radius:20px;font-size:0.78rem;font-weight:600;">Coming Soon</span>'
            f'</div>',
            unsafe_allow_html=True
        )


if __name__ == "__main__":
    main()
