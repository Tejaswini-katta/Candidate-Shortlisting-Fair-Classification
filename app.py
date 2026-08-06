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
# PAGE 2: CANDIDATE RANKING  (unchanged logic)
# ─────────────────────────────────────────────────────────────────────────────
def render_ranking_page():
    st.header("🎯 Candidate Shortlisting & Probability Ranking Dashboard")
    st.caption("Rank candidates in the test cohort by predicted job change suitability and priority recruitment tiers.")

    rankings_path = os.path.join("reports", "metrics", "candidate_rankings.csv")
    df = load_csv_report(rankings_path)

    if df.empty:
        st.error(f"Ranking report file missing at `{rankings_path}`. Please run `python run_ranking.py` first.")
        return

    total_candidates = len(df)
    high_prio = len(df[df["priority_tier"] == "High Priority"])
    qualified = len(df[df["priority_tier"] == "Qualified"])
    extended  = len(df[df["priority_tier"] == "Extended"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Candidates Evaluated", f"{total_candidates:,}")
    c2.metric("High Priority (Top 10%)", f"{high_prio:,}")
    c3.metric("Qualified Pool (Top 25%)", f"{qualified:,}")
    c4.metric("Extended Pool (Top 50%)", f"{extended:,}")

    st.markdown("---")

    col_filter1, col_filter2 = st.columns([2, 2])
    with col_filter1:
        search_id = st.text_input("🔍 Search by Candidate Enrollee ID:", placeholder="e.g. 22527")
    with col_filter2:
        selected_tiers = st.multiselect(
            "Filter by Priority Tier:",
            options=["High Priority", "Qualified", "Extended", "Reserve"],
            default=["High Priority", "Qualified"]
        )

    filtered_df = df.copy()
    if selected_tiers:
        filtered_df = filtered_df[filtered_df["priority_tier"].isin(selected_tiers)]
    if search_id.strip():
        filtered_df = filtered_df[filtered_df["enrollee_id"].astype(str).str.contains(search_id.strip())]

    st.markdown(f"**Displaying {len(filtered_df):,} out of {total_candidates:,} candidates**")
    st.dataframe(
        filtered_df,
        column_config={
            "enrollee_id": st.column_config.NumberColumn("Candidate ID", format="%d"),
            "prediction_probability": st.column_config.ProgressColumn(
                "Suitability Score", format="%.4f", min_value=0.0, max_value=1.0),
            "predicted_class": st.column_config.NumberColumn("Shortlist Flag", format="%d"),
            "percentile_rank": st.column_config.NumberColumn("Top Percentile", format="%.2f%%"),
            "priority_tier": "Priority Tier"
        },
        use_container_width=True, height=450
    )
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Candidate Shortlist (CSV)",
        data=csv_data, file_name="candidate_shortlist_export.csv", mime="text/csv"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: FAIRNESS DASHBOARD  (unchanged logic, theme-aware matplotlib)
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
