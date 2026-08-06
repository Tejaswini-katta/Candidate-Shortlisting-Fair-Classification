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
        f".stApp{{background-color:{t['bg']};}}"
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
            st.dataframe(
                top_cands[available].reset_index(drop=True),
                column_config={
                    "Suitability Score": st.column_config.ProgressColumn("Suitability Score", format="%.4f", min_value=0.0, max_value=1.0),
                    "Percentile": st.column_config.NumberColumn("Percentile", format="%.1f%%"),
                    "Shortlisted": st.column_config.CheckboxColumn("Shortlisted"),
                    "Rank": st.column_config.NumberColumn("Rank", format="#%d"),
                    "Candidate ID": st.column_config.NumberColumn("Candidate ID", format="%d"),
                },
                use_container_width=True, height=310, hide_index=True
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

    st.dataframe(
        page_df[show_cols].reset_index(drop=True),
        column_config=col_config,
        use_container_width=True,
        height=min(600, 56 + len(page_df) * 35),
        hide_index=True,
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
                st.session_state["nav_default_idx"] = 6   # index of Candidate Profile in nav
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
    rng = random.Random(int(str(enrollee_id)))
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
    name        = _generate_candidate_name(cid, gender)
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
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: FAIRNESS DASHBOARD  (unchanged logic, theme-aware matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
def render_fairness_page():
    t = ThemeManager.get()

    st.header("⚖️ Algorithmic Fairness & Bias Mitigation Audit")
    st.caption("Audit Demographic Parity, Equal Opportunity, and Equalized Odds across protected candidate Gender groups.")

    fairness_path    = os.path.join("reports", "metrics", "fairness_audit_report.csv")
    fair_config_path = os.path.join("models", "trained_models", "fairness_config.json")
    fair_df    = load_csv_report(fairness_path)
    fair_config = load_json_config(fair_config_path)

    if fair_df.empty:
        st.error(f"Fairness report missing at `{fairness_path}`. Please run `python run_fairness.py` first.")
        return

    st.markdown("### 📊 Disparity Metrics Comparison (Before vs After Mitigation)")
    st.dataframe(fair_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(t["mpl_bg"])
    ax.set_facecolor(t["mpl_bg"])
    plot_data = fair_df.set_index("Stage").T
    plot_data.plot(kind="bar", ax=ax, color=["#e76f51", "#2a9d8f"], width=0.6)
    plt.title("Disparity Metrics Reduction Across Gender Groups",
              fontsize=12, fontweight="bold", color=t["mpl_text"])
    plt.ylabel("Disparity Difference Magnitude", fontsize=10, color=t["mpl_text"])
    ax.tick_params(colors=t["mpl_text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(t["mpl_grid"])
    plt.xticks(rotation=15, ha="right", color=t["mpl_text"])
    plt.grid(axis="y", linestyle="--", alpha=0.4, color=t["mpl_grid"])
    legend = ax.get_legend()
    if legend:
        for txt in legend.get_texts():
            txt.set_color(t["mpl_text"])
        legend.get_frame().set_facecolor(t["mpl_bg"])
        legend.get_frame().set_edgecolor(t["mpl_grid"])
    st.pyplot(fig)
    plt.close()

    if "fair_thresholds" in fair_config:
        st.markdown("### ⚙️ Group-Specific Calibrated Classification Thresholds")
        st.json(fair_config["fair_thresholds"])

    st.markdown("---")
    st.markdown("### 📖 Metric Definitions")
    st.markdown("""
    - **Demographic Parity Difference**: $\\max(SR) - \\min(SR)$. Measures selection rate equality regardless of true labels.
    - **Equal Opportunity Difference**: $\\max(TPR) - \\min(TPR)$. Measures True Positive Rate (Recall) equality for qualified candidates.
    - **Equalized Odds Difference**: Measures maximum disparity across both True Positive Rate and False Positive Rate.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: SHAP EXPLAINABILITY  (unchanged, bug fix retained)
# ─────────────────────────────────────────────────────────────────────────────
def render_explainability_page():
    st.header("🔍 Explainable AI (SHAP Feature Importance & Attributions)")
    st.caption("Inspect global model feature dependencies and local candidate decision attribution scores.")

    shap_path   = os.path.join("reports", "metrics", "shap_feature_importance.csv")
    fig_path    = os.path.join("reports", "figures", "17_shap_feature_importance.png")
    sample_path = os.path.join("reports", "metrics", "sample_candidate_shap_explanation.json")

    shap_df     = load_csv_report(shap_path)
    sample_json = load_json_config(sample_path)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📈 Global Feature Importance Rankings")
        if not shap_df.empty:
            st.dataframe(shap_df.head(15), use_container_width=True, height=400)
        else:
            st.warning("SHAP feature importance report missing.")
    with col2:
        st.markdown("### 🖼️ SHAP Summary Visualization")
        if os.path.exists(fig_path):
            st.image(fig_path, width="content")
        else:
            st.warning("SHAP summary plot missing.")

    st.markdown("---")
    st.markdown("### 👤 Sample Candidate Local SHAP Decision Breakdown")
    if sample_json:
        st.json(sample_json)
    else:
        st.info("Sample candidate explanation JSON missing.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: REAL-TIME PREDICTION  (unchanged ML logic)
# ─────────────────────────────────────────────────────────────────────────────
def render_prediction_page():
    st.header("👤 Real-Time Single Candidate Shortlisting Evaluator")
    st.caption("Input candidate profile attributes to compute shortlisting suitability score, predicted class, and assigned priority tier.")

    config_path = os.path.join("models", "trained_models", "preprocessor_config.json")
    fair_path   = os.path.join("models", "trained_models", "fairness_config.json")
    config      = load_json_config(config_path)
    fair_config = load_json_config(fair_path)

    if not config:
        st.error(f"Preprocessor config missing at `{config_path}`. Please run preprocessing first.")
        return

    with st.form("candidate_form"):
        st.subheader("Candidate Background Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            cdi            = st.slider("City Development Index (CDI):", 0.40, 1.00, 0.92, step=0.01)
            training_hours = st.number_input("Training Hours Completed:", min_value=1, max_value=500, value=36)
            gender         = st.selectbox("Gender (Protected Attribute):", ["Male", "Female", "Other", "Unknown"])
        with c2:
            experience = st.selectbox("Total Experience (Years):",
                ["<1","1","2","3","4","5","6","7","8","9","10","11",
                 "12","13","14","15","16","17","18","19","20",">20"])
            last_new_job        = st.selectbox("Years Since Last Job Change:", ["never","1","2","3","4",">4"])
            relevent_experience = st.selectbox("Relevant Work Experience:", ["Has relevent experience","No relevent experience"])
        with c3:
            education_level = st.selectbox("Education Level:", ["Graduate","Masters","High School","Phd","Primary School","Unknown"])
            company_size    = st.selectbox("Company Size:", ["50-99","<10","10000+","10-49","1000-4999","500-999","5000-9999","100-499","Unknown"])
            company_type    = st.selectbox("Company Type:", ["Pvt Ltd","Funded Startup","Public Sector","Early Stage Startup","NGO","Other","Unknown"])
        submit_button = st.form_submit_button("🚀 Evaluate Candidate Suitability", use_container_width=True)

    if submit_button:
        candidate_dict = {
            "city_development_index": [cdi], "training_hours": [training_hours],
            "gender": [gender], "relevent_experience": [relevent_experience],
            "enrolled_university": ["no_enrollment"], "education_level": [education_level],
            "major_discipline": ["STEM"], "experience": [experience],
            "company_size": [company_size], "company_type": [company_type],
            "last_new_job": [last_new_job]
        }
        cand_df = pd.DataFrame(candidate_dict)

        from src.preprocessing import CandidatePreprocessor, CustomStandardScaler
        preprocessor = CandidatePreprocessor()
        preprocessor.feature_names      = config["feature_names"]
        preprocessor.nominal_categories = config["nominal_categories"]
        scaler = CustomStandardScaler()
        scaler.mean_  = pd.Series(config["scaler_mean"])
        scaler.scale_ = pd.Series(config["scaler_scale"])
        preprocessor.scaler = scaler
        X_cand_scaled = preprocessor.transform(cand_df)

        from src.modeling import LogisticRegressionModel
        X_train = pd.read_csv(os.path.join("data", "processed", "X_train.csv")).values
        y_train = pd.read_csv(os.path.join("data", "processed", "y_train.csv")).values.ravel()
        model = LogisticRegressionModel(lr=0.08, n_iters=400, l2_reg=0.1)
        model.fit(X_train, y_train)
        prob = float(model.predict_proba(X_cand_scaled.values)[0, 1])

        threshold = fair_config.get("fair_thresholds", {}).get(gender, 0.5)
        pred_class = int(prob >= threshold)
        tier = ("High Priority" if prob >= 0.50 else
                "Qualified"     if prob >= 0.35 else
                "Extended"      if prob >= 0.20 else "Reserve")

        st.markdown("---")
        st.subheader("📊 Candidate Prediction Analysis Results")
        res1, res2, res3 = st.columns(3)
        res1.metric("Predicted Suitability Score", f"{prob:.4f}")
        res2.metric("Fairness-Calibrated Flag", "Shortlisted (1)" if pred_class == 1 else "Not Shortlisted (0)")
        res3.metric("Assigned Recruitment Tier", tier)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6: SETTINGS  — theme control + model info + about
# ─────────────────────────────────────────────────────────────────────────────
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
        '<div class="settings-card"><h4>🤖 Model Information</h4></div>',
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
            f'{model_name}</div>' + rows + '</div>',
            unsafe_allow_html=True
        )

    with col_m2:
        rows2 = ""
        for grp, thr_val in ft.items():
            rows2 += _kv_row(grp, str(thr_val), "#34d399", t["panel_border_l"], t["text_label"])
        st.markdown(
            f'<div class="panel-card"><div style="color:{t["text_primary"]};font-weight:700;margin-bottom:12px;">'
            f'⚖️ Fairness Thresholds</div>' + rows2 +
            f'<div style="margin-top:10px;font-size:0.72rem;color:{t["text_muted"]};">'
            f'Calibrated via Fairlearn ThresholdOptimizer (post-processing)</div></div>',
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

        # Navigation
        page = st.radio(
            "Navigation",
            options=[
                "🏠  Dashboard",
                "📋  Candidate Rankings",
                "⚖️  Fairness & Bias Audit",
                "🔍  SHAP Explainability",
                "🚀  Real-Time Predictor",
                "─────────────",
                "💼  Job Descriptions",
                "👤  Candidate Profile",
                "📊  Analytics",
                "📄  Reports",
                "⚙️  Settings",
            ],
            label_visibility="collapsed",
            index=st.session_state.pop("nav_default_idx", 0),
        )

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
