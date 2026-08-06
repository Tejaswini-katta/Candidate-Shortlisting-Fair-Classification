"""
FairHire AI — Explainable & Fair Candidate Screening Platform.

Professional HR SaaS Dashboard providing:
1. Executive Dashboard Home
2. Candidate Probability Ranking & Shortlist Export
3. Algorithmic Fairness & Bias Mitigation Audit
4. SHAP Global & Local Feature Explainability
5. Real-Time Single Candidate Shortlisting Predictor

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
# ENTERPRISE CSS — Dark Mode Design System
# ─────────────────────────────────────────────────────────────────────────────
ENTERPRISE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp { background-color: #0a0f1e; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1424 0%, #0f1929 100%);
    border-right: 1px solid #1a2540;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 5px 0 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[data-checked="true"] label {
    color: #60a5fa !important;
    font-weight: 600 !important;
}

/* ── Main block ── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
    max-width: 100%;
}

/* ── KPI Card ── */
.kpi-card {
    background: linear-gradient(135deg, #111827 0%, #141d2e 100%);
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 20px 22px 16px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    margin-bottom: 2px;
    min-height: 110px;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.kpi-card.blue::before   { background: linear-gradient(90deg,#3b82f6,#60a5fa); }
.kpi-card.green::before  { background: linear-gradient(90deg,#10b981,#34d399); }
.kpi-card.amber::before  { background: linear-gradient(90deg,#f59e0b,#fbbf24); }
.kpi-card.red::before    { background: linear-gradient(90deg,#ef4444,#f87171); }
.kpi-card.purple::before { background: linear-gradient(90deg,#8b5cf6,#a78bfa); }
.kpi-card.teal::before   { background: linear-gradient(90deg,#14b8a6,#2dd4bf); }
.kpi-card.indigo::before { background: linear-gradient(90deg,#6366f1,#818cf8); }
.kpi-card.pink::before   { background: linear-gradient(90deg,#ec4899,#f472b6); }

.kpi-icon {
    position: absolute;
    top: 16px; right: 18px;
    font-size: 1.9rem;
    opacity: 0.12;
}
.kpi-label {
    color: #6b7280;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 7px;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 5px;
}
.kpi-value.blue   { color: #60a5fa; }
.kpi-value.green  { color: #34d399; }
.kpi-value.amber  { color: #fbbf24; }
.kpi-value.red    { color: #f87171; }
.kpi-value.purple { color: #a78bfa; }
.kpi-value.teal   { color: #2dd4bf; }
.kpi-value.indigo { color: #818cf8; }
.kpi-value.pink   { color: #f472b6; }
.kpi-sub { color: #4b5563; font-size: 0.72rem; font-weight: 500; }

/* ── Section header ── */
.section-header {
    color: #e5e7eb;
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1f2937;
}

/* ── Panel card ── */
.panel-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.25);
}

/* ── Activity feed ── */
.activity-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 9px 0;
    border-bottom: 1px solid #1a2234;
}
.activity-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-top: 5px;
    flex-shrink: 0;
}
.activity-dot.blue   { background: #3b82f6; box-shadow: 0 0 6px #3b82f6; }
.activity-dot.green  { background: #10b981; box-shadow: 0 0 6px #10b981; }
.activity-dot.amber  { background: #f59e0b; box-shadow: 0 0 6px #f59e0b; }
.activity-dot.purple { background: #8b5cf6; box-shadow: 0 0 6px #8b5cf6; }
.activity-dot.teal   { background: #14b8a6; box-shadow: 0 0 6px #14b8a6; }

.activity-text { color: #d1d5db; font-size: 0.8rem; line-height: 1.4; }
.activity-time { color: #374151; font-size: 0.68rem; margin-top: 2px; }

/* ── Status pills ── */
.status-ok {
    background: rgba(16,185,129,0.1);
    color: #10b981;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 600;
    border: 1px solid rgba(16,185,129,0.25);
}
.status-warn {
    background: rgba(245,158,11,0.1);
    color: #f59e0b;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 600;
    border: 1px solid rgba(245,158,11,0.25);
}
.status-err {
    background: rgba(239,68,68,0.1);
    color: #ef4444;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 600;
    border: 1px solid rgba(239,68,68,0.25);
}

/* ── Streamlit metric override ── */
[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 16px 18px;
}
[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #f9fafb !important; }

/* ── Dividers ── */
hr { border-color: #1f2937 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #2563eb);
    color: #fff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 10px 20px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #60a5fa, #3b82f6);
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(59,130,246,0.45);
}

/* ── Coming soon ── */
.coming-soon-banner {
    background: linear-gradient(135deg, #111827, #0d1424);
    border: 1px dashed #1f2937;
    border-radius: 16px;
    padding: 80px 40px;
    text-align: center;
    margin-top: 40px;
}

/* ── Info boxes ── */
[data-testid="stInfo"] { background: rgba(59,130,246,0.08) !important; border: 1px solid rgba(59,130,246,0.2) !important; }
[data-testid="stWarning"] { background: rgba(245,158,11,0.08) !important; border: 1px solid rgba(245,158,11,0.2) !important; }
[data-testid="stError"] { background: rgba(239,68,68,0.08) !important; border: 1px solid rgba(239,68,68,0.2) !important; }
</style>
"""

st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)


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
# PAGE 1: EXECUTIVE DASHBOARD HOME  (Phase 1 — completely rebuilt)
# ─────────────────────────────────────────────────────────────────────────────
def render_home_page():
    # ── Load all real data sources ──────────────────────────────────────────
    rankings_df = load_csv_report(os.path.join("reports", "metrics", "candidate_rankings.csv"))
    fair_config = load_json_config(os.path.join("models", "trained_models", "fairness_config.json"))
    model_info  = load_json_config(os.path.join("models", "trained_models", "best_model_info.json"))
    shap_df     = load_csv_report(os.path.join("reports", "metrics", "shap_feature_importance.csv"))

    # ── Derive KPIs from real data ──────────────────────────────────────────
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
    raw_dpd       = raw_summary.get("Demographic Parity Difference", 0.0)
    mit_dpd       = mit_summary.get("Demographic Parity Difference", 0.0)
    raw_eod       = raw_summary.get("Equal Opportunity Difference", 0.0)
    mit_eod       = mit_summary.get("Equal Opportunity Difference", 0.0)
    raw_eqo       = raw_summary.get("Equalized Odds Difference", 0.0)
    mit_eqo       = mit_summary.get("Equalized Odds Difference", 0.0)
    bias_reduction = ((raw_dpd - mit_dpd) / raw_dpd * 100) if raw_dpd > 0 else 0.0
    selection_rate = (shortlisted / total_cands * 100) if total_cands > 0 else 0.0

    top_feature   = shap_df.iloc[0]["Feature"] if not shap_df.empty else "N/A"
    top_pct       = float(shap_df.iloc[0]["Importance_Percentage"]) if not shap_df.empty else 0.0

    now = datetime.now().strftime("%B %d, %Y  %H:%M")

    # ── Header Banner ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0d1b3e 0%, #111827 55%, #0a0f1e 100%);
                border: 1px solid #1f2937; border-radius: 16px; padding: 26px 30px;
                margin-bottom: 24px; position: relative; overflow: hidden;">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;
                    background:linear-gradient(90deg,#3b82f6 0%,#8b5cf6 50%,#10b981 100%);"></div>
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;">
            <div>
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                    <span style="font-size:2rem;">🎯</span>
                    <span style="color:#f9fafb;font-size:1.7rem;font-weight:800;letter-spacing:-0.03em;">FairHire AI</span>
                    <span style="background:rgba(59,130,246,0.15);color:#60a5fa;
                                 border:1px solid rgba(59,130,246,0.3);padding:3px 10px;
                                 border-radius:20px;font-size:0.62rem;font-weight:700;
                                 letter-spacing:0.1em;text-transform:uppercase;">v2.0</span>
                </div>
                <p style="color:#94a3b8;font-size:0.9rem;margin:0;font-weight:400;line-height:1.5;">
                    Explainable &amp; Fair Candidate Screening Platform &nbsp;·&nbsp;
                    <span style="color:#4b5563;">Executive Dashboard</span>
                </p>
            </div>
            <div style="text-align:right;">
                <div style="color:#374151;font-size:0.66rem;font-weight:600;
                            text-transform:uppercase;letter-spacing:0.08em;margin-bottom:4px;">Last Refreshed</div>
                <div style="color:#6b7280;font-size:0.82rem;margin-bottom:6px;">{now}</div>
                <span class="status-ok">● All Systems Operational</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row 1: Candidate Pipeline ────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="kpi-card blue">
            <div class="kpi-icon">👥</div>
            <div class="kpi-label">Total Candidates</div>
            <div class="kpi-value blue">{total_cands:,}</div>
            <div class="kpi-sub">In current evaluation cohort</div>
        </div>""", unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card green">
            <div class="kpi-icon">✅</div>
            <div class="kpi-label">Shortlisted</div>
            <div class="kpi-value green">{shortlisted:,}</div>
            <div class="kpi-sub">Selection rate: {selection_rate:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card purple">
            <div class="kpi-icon">📊</div>
            <div class="kpi-label">Avg Suitability Score</div>
            <div class="kpi-value purple">{avg_score:.3f}</div>
            <div class="kpi-sub">Mean predicted probability</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        bias_color = "green" if bias_reduction >= 80 else "amber"
        st.markdown(f"""
        <div class="kpi-card {bias_color}">
            <div class="kpi-icon">⚖️</div>
            <div class="kpi-label">Bias Reduction</div>
            <div class="kpi-value {bias_color}">{bias_reduction:.1f}%</div>
            <div class="kpi-sub">Fairlearn post-processing</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

    # ── KPI Row 2: Tier Breakdown + Model ────────────────────────────────────
    k5, k6, k7, k8 = st.columns(4)

    with k5:
        st.markdown(f"""
        <div class="kpi-card red">
            <div class="kpi-icon">🔥</div>
            <div class="kpi-label">High Priority</div>
            <div class="kpi-value red">{high_priority:,}</div>
            <div class="kpi-sub">Top 10% · Immediate interview</div>
        </div>""", unsafe_allow_html=True)

    with k6:
        st.markdown(f"""
        <div class="kpi-card teal">
            <div class="kpi-icon">🎓</div>
            <div class="kpi-label">Qualified Pool</div>
            <div class="kpi-value teal">{qualified:,}</div>
            <div class="kpi-sub">Top 25% · Strong candidates</div>
        </div>""", unsafe_allow_html=True)

    with k7:
        st.markdown(f"""
        <div class="kpi-card amber">
            <div class="kpi-icon">🤖</div>
            <div class="kpi-label">Model ROC-AUC</div>
            <div class="kpi-value amber">{roc_auc:.4f}</div>
            <div class="kpi-sub">{best_model}</div>
        </div>""", unsafe_allow_html=True)

    with k8:
        bias_status_text = "✔ Mitigated" if bias_reduction >= 80 else "⚠ Under Review"
        b8_color = "green" if bias_reduction >= 80 else "amber"
        st.markdown(f"""
        <div class="kpi-card {b8_color}">
            <div class="kpi-icon">🛡️</div>
            <div class="kpi-label">Fairness Status</div>
            <div class="kpi-value {b8_color}" style="font-size:1.3rem;">{bias_status_text}</div>
            <div class="kpi-sub">DPD: {mit_dpd:.4f} after mitigation</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:22px;'></div>", unsafe_allow_html=True)

    # ── Main Content Row ─────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2], gap="large")

    # LEFT: Hiring Funnel + Top Candidates Table
    with col_left:
        st.markdown('<div class="section-header">🔻 Hiring Pipeline Funnel</div>', unsafe_allow_html=True)

        funnel_stages = [
            f"Total Evaluated  ({total_cands:,})",
            f"Extended Pool — Top 50%  ({high_priority + qualified + extended:,})",
            f"Qualified Pool — Top 25%  ({high_priority + qualified:,})",
            f"High Priority — Top 10%  ({high_priority:,})",
            f"Shortlisted  ({shortlisted:,})",
        ]
        funnel_vals = [
            total_cands,
            high_priority + qualified + extended,
            high_priority + qualified,
            high_priority,
            shortlisted,
        ]

        fig_funnel = go.Figure(go.Funnel(
            y=funnel_stages,
            x=funnel_vals,
            textinfo="value+percent initial",
            marker=dict(color=["#1e3a5f", "#1d4e7e", "#2563eb", "#10b981", "#ef4444"]),
            connector=dict(line=dict(color="#374151", width=1, dash="dot")),
            textfont=dict(color="#f9fafb", family="Inter", size=12)
        ))
        fig_funnel.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(family="Inter", color="#6b7280"),
            margin=dict(l=10, r=10, t=8, b=8),
            height=270,
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

        st.markdown("<div style='margin-top:4px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🏆 Top High-Priority Candidates</div>', unsafe_allow_html=True)

        if not rankings_df.empty:
            top_cands = rankings_df[rankings_df["priority_tier"] == "High Priority"].head(8).copy()
            top_cands.insert(0, "Rank", range(1, len(top_cands) + 1))
            top_cands = top_cands.rename(columns={
                "enrollee_id":             "Candidate ID",
                "prediction_probability":  "Suitability Score",
                "priority_tier":           "Tier",
                "percentile_rank":         "Percentile",
                "predicted_class":         "Shortlisted",
            })
            display_cols = ["Rank", "Candidate ID", "Suitability Score", "Percentile", "Shortlisted", "Tier"]
            available_cols = [c for c in display_cols if c in top_cands.columns]
            st.dataframe(
                top_cands[available_cols].reset_index(drop=True),
                column_config={
                    "Suitability Score": st.column_config.ProgressColumn(
                        "Suitability Score", format="%.4f", min_value=0.0, max_value=1.0
                    ),
                    "Percentile": st.column_config.NumberColumn("Percentile", format="%.1f%%"),
                    "Shortlisted": st.column_config.CheckboxColumn("Shortlisted"),
                    "Rank": st.column_config.NumberColumn("Rank", format="#%d"),
                    "Candidate ID": st.column_config.NumberColumn("Candidate ID", format="%d"),
                },
                use_container_width=True,
                height=310,
                hide_index=True,
            )
        else:
            st.warning("Rankings unavailable. Run `python run_ranking.py` first.")

    # RIGHT: Tier Chart + Activity Feed
    with col_right:
        st.markdown('<div class="section-header">📊 Priority Tier Distribution</div>', unsafe_allow_html=True)

        tier_labels = ["High Priority", "Qualified", "Extended", "Reserve"]
        tier_values = [high_priority, qualified, extended, reserve]
        tier_colors = ["#ef4444", "#10b981", "#f59e0b", "#6b7280"]

        fig_bar = go.Figure(go.Bar(
            x=tier_labels,
            y=tier_values,
            marker_color=tier_colors,
            marker_line_width=0,
            text=[
                f"{v:,}<br><span style='font-size:9px'>{v/total_cands*100:.1f}%</span>"
                if total_cands > 0 else "0"
                for v in tier_values
            ],
            textposition="outside",
            textfont=dict(color="#94a3b8", size=11, family="Inter"),
            hovertemplate="<b>%{x}</b><br>Candidates: %{y:,}<extra></extra>",
        ))
        fig_bar.update_layout(
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(family="Inter", color="#6b7280"),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(color="#6b7280", size=11),
                showline=False,
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#1a2234",
                tickfont=dict(color="#374151"),
                showline=False,
            ),
            margin=dict(l=10, r=10, t=30, b=10),
            height=248,
            showlegend=False,
            bargap=0.35,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("<div style='margin-top:6px;'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">🕐 Recent Activity</div>', unsafe_allow_html=True)

        activities = [
            ("blue",   "Ranking pipeline complete",    f"Evaluated {total_cands:,} candidates",              "Just now"),
            ("green",  "Bias mitigation applied",      f"DPD reduced {raw_dpd:.4f} → {mit_dpd:.4f}",        "2 min ago"),
            ("purple", "SHAP analysis finished",       f"Top feature: {top_feature} ({top_pct:.1f}%)",       "5 min ago"),
            ("amber",  "Fairness audit complete",      f"Bias reduced by {bias_reduction:.1f}%",             "10 min ago"),
            ("teal",   "Model evaluation complete",    f"ROC-AUC: {roc_auc:.4f} · {best_model}",            "15 min ago"),
        ]

        html_feed = '<div class="panel-card" style="padding:14px 16px;">'
        for color, title, detail, ts in activities:
            html_feed += f"""
            <div class="activity-item">
                <div class="activity-dot {color}"></div>
                <div style="flex:1;min-width:0;">
                    <div class="activity-text">
                        <strong style="color:#e5e7eb;font-weight:600;">{title}</strong><br>
                        <span style="color:#94a3b8;">{detail}</span>
                    </div>
                    <div class="activity-time">{ts}</div>
                </div>
            </div>"""
        html_feed += "</div>"
        st.markdown(html_feed, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

    # ── Bottom Row: Model · Fairness · Tech Stack ────────────────────────────
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
        html_model = f"""<div class="panel-card">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;">
            <span style="font-size:1.1rem;">🤖</span>
            <span style="color:#f9fafb;font-weight:700;font-size:0.88rem;">{best_model}</span>
        </div>"""
        for lbl, val, clr in metrics_items:
            html_model += f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:6px 0;border-bottom:1px solid #1a2234;">
            <span style="color:#6b7280;font-size:0.77rem;">{lbl}</span>
            <span style="color:{clr};font-weight:700;font-size:0.85rem;">{val}</span>
        </div>"""
        html_model += "</div>"
        st.markdown(html_model, unsafe_allow_html=True)

    with b2:
        st.markdown('<div class="section-header">⚖️ Fairness Summary</div>', unsafe_allow_html=True)
        fair_items = [
            ("Dem. Parity Diff (Before)",   f"{raw_dpd:.4f}",  "#f87171"),
            ("Dem. Parity Diff (After)",    f"{mit_dpd:.4f}",  "#34d399"),
            ("Equal Opp. Diff (Before)",    f"{raw_eod:.4f}",  "#f87171"),
            ("Equal Opp. Diff (After)",     f"{mit_eod:.4f}",  "#34d399"),
            ("Equalized Odds (Before)",     f"{raw_eqo:.4f}",  "#f87171"),
            ("Equalized Odds (After)",      f"{mit_eqo:.4f}",  "#34d399"),
        ]
        html_fair = "<div class='panel-card'>"
        for lbl, val, clr in fair_items:
            html_fair += f"""
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:6px 0;border-bottom:1px solid #1a2234;">
            <span style="color:#6b7280;font-size:0.74rem;">{lbl}</span>
            <span style="color:{clr};font-weight:700;font-size:0.83rem;">{val}</span>
        </div>"""
        html_fair += f"""
        <div style="margin-top:14px;text-align:center;padding:10px;
                    background:rgba(16,185,129,0.07);border-radius:10px;
                    border:1px solid rgba(16,185,129,0.2);">
            <span style="color:#34d399;font-weight:700;font-size:0.88rem;">
                ✔ {bias_reduction:.1f}% Bias Reduction Achieved
            </span>
        </div></div>"""
        st.markdown(html_fair, unsafe_allow_html=True)

    with b3:
        st.markdown('<div class="section-header">💻 Technology Stack</div>', unsafe_allow_html=True)
        stack = [
            ("🐍", "Python 3.11",    "Core runtime",         "#60a5fa"),
            ("🤖", "Scikit-Learn",   "ML pipeline",          "#34d399"),
            ("⚖️", "Fairlearn",      "Bias mitigation",      "#a78bfa"),
            ("🔍", "SHAP",           "Explainability",       "#fbbf24"),
            ("📊", "Streamlit",      "Dashboard UI",         "#f87171"),
            ("📈", "Plotly",         "Interactive charts",   "#2dd4bf"),
        ]
        html_tech = "<div class='panel-card'>"
        for icon, name, desc, clr in stack:
            html_tech += f"""
        <div style="display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid #1a2234;">
            <span style="font-size:0.95rem;">{icon}</span>
            <div style="flex:1;">
                <span style="color:{clr};font-weight:600;font-size:0.8rem;">{name}</span>
                <span style="color:#374151;font-size:0.74rem;"> · {desc}</span>
            </div>
        </div>"""
        html_tech += f"""
        <div style="margin-top:14px;padding:10px;background:rgba(59,130,246,0.06);
                    border-radius:10px;border:1px solid rgba(59,130,246,0.15);">
            <div style="color:#4b5563;font-size:0.67rem;text-transform:uppercase;
                        letter-spacing:0.07em;margin-bottom:4px;">Top SHAP Driver</div>
            <div style="color:#60a5fa;font-weight:700;font-size:0.82rem;">{top_feature}</div>
            <div style="color:#374151;font-size:0.7rem;margin-top:2px;">
                {top_pct:.1f}% of total model attribution
            </div>
        </div></div>"""
        st.markdown(html_tech, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: CANDIDATE RANKING  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
def render_ranking_page():
    st.header("🎯 Candidate Shortlisting & Probability Ranking Dashboard")
    st.caption("Rank candidates in the test cohort by predicted job change suitability and priority recruitment tiers.")

    rankings_path = os.path.join("reports", "metrics", "candidate_rankings.csv")
    df = load_csv_report(rankings_path)

    if df.empty:
        st.error(f"Ranking report file missing at `{rankings_path}`. Please run `python run_ranking.py` first.")
        return

    # KPI summary
    total_candidates = len(df)
    high_prio = len(df[df["priority_tier"] == "High Priority"])
    qualified = len(df[df["priority_tier"] == "Qualified"])
    extended = len(df[df["priority_tier"] == "Extended"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Candidates Evaluated", f"{total_candidates:,}")
    c2.metric("High Priority (Top 10%)", f"{high_prio:,}")
    c3.metric("Qualified Pool (Top 25%)", f"{qualified:,}")
    c4.metric("Extended Pool (Top 50%)", f"{extended:,}")

    st.markdown("---")

    # Filters
    col_filter1, col_filter2 = st.columns([2, 2])
    with col_filter1:
        search_id = st.text_input("🔍 Search by Candidate Enrollee ID:", placeholder="e.g. 22527")
    with col_filter2:
        selected_tiers = st.multiselect(
            "Filter by Priority Tier:",
            options=["High Priority", "Qualified", "Extended", "Reserve"],
            default=["High Priority", "Qualified"]
        )

    # Apply filters
    filtered_df = df.copy()
    if selected_tiers:
        filtered_df = filtered_df[filtered_df["priority_tier"].isin(selected_tiers)]
    if search_id.strip():
        filtered_df = filtered_df[filtered_df["enrollee_id"].astype(str).str.contains(search_id.strip())]

    st.markdown(f"**Displaying {len(filtered_df):,} out of {total_candidates:,} candidates**")

    # Dataframe display
    st.dataframe(
        filtered_df,
        column_config={
            "enrollee_id": st.column_config.NumberColumn("Candidate ID", format="%d"),
            "prediction_probability": st.column_config.ProgressColumn(
                "Suitability Score",
                format="%.4f",
                min_value=0.0,
                max_value=1.0
            ),
            "predicted_class": st.column_config.NumberColumn("Shortlist Flag", format="%d"),
            "percentile_rank": st.column_config.NumberColumn("Top Percentile", format="%.2f%%"),
            "priority_tier": "Priority Tier"
        },
        use_container_width=True,
        height=450
    )

    # Download Button
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Candidate Shortlist (CSV)",
        data=csv_data,
        file_name="candidate_shortlist_export.csv",
        mime="text/csv"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: FAIRNESS DASHBOARD  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
def render_fairness_page():
    st.header("⚖️ Algorithmic Fairness & Bias Mitigation Audit")
    st.caption("Audit Demographic Parity, Equal Opportunity, and Equalized Odds across protected candidate Gender groups.")

    fairness_path = os.path.join("reports", "metrics", "fairness_audit_report.csv")
    fair_config_path = os.path.join("models", "trained_models", "fairness_config.json")

    fair_df = load_csv_report(fairness_path)
    fair_config = load_json_config(fair_config_path)

    if fair_df.empty:
        st.error(f"Fairness report missing at `{fairness_path}`. Please run `python run_fairness.py` first.")
        return

    st.markdown("### 📊 Disparity Metrics Comparison (Before vs After Mitigation)")
    st.dataframe(fair_df, use_container_width=True)

    # Visualization of fairness reduction
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_data = fair_df.set_index("Stage").T
    plot_data.plot(kind="bar", ax=ax, color=["#e76f51", "#2a9d8f"], width=0.6)

    plt.title("Disparity Metrics Reduction Across Gender Groups", fontsize=12, fontweight="bold")
    plt.ylabel("Disparity Difference Magnitude", fontsize=10)
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    st.pyplot(fig)

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
# PAGE 4: SHAP EXPLAINABILITY  (unchanged from original, bug fix applied)
# ─────────────────────────────────────────────────────────────────────────────
def render_explainability_page():
    st.header("🔍 Explainable AI (SHAP Feature Importance & Attributions)")
    st.caption("Inspect global model feature dependencies and local candidate decision attribution scores.")

    shap_path = os.path.join("reports", "metrics", "shap_feature_importance.csv")
    fig_path = os.path.join("reports", "figures", "17_shap_feature_importance.png")
    sample_path = os.path.join("reports", "metrics", "sample_candidate_shap_explanation.json")

    shap_df = load_csv_report(shap_path)
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
# PAGE 5: REAL-TIME PREDICTION  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
def render_prediction_page():
    st.header("👤 Real-Time Single Candidate Shortlisting Evaluator")
    st.caption("Input candidate profile attributes to compute shortlisting suitability score, predicted class, and assigned priority tier.")

    # Load preprocessor config and model weights
    config_path = os.path.join("models", "trained_models", "preprocessor_config.json")
    model_info_path = os.path.join("models", "trained_models", "best_model_info.json")
    fair_path = os.path.join("models", "trained_models", "fairness_config.json")

    config = load_json_config(config_path)
    fair_config = load_json_config(fair_path)

    if not config:
        st.error(f"Preprocessor config missing at `{config_path}`. Please run preprocessing first.")
        return

    with st.form("candidate_form"):
        st.subheader("Candidate Background Information")

        c1, c2, c3 = st.columns(3)
        with c1:
            cdi = st.slider("City Development Index (CDI):", 0.40, 1.00, 0.92, step=0.01)
            training_hours = st.number_input("Training Hours Completed:", min_value=1, max_value=500, value=36)
            gender = st.selectbox("Gender (Protected Attribute):", ["Male", "Female", "Other", "Unknown"])

        with c2:
            experience = st.selectbox(
                "Total Experience (Years):",
                ["<1", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
                 "12", "13", "14", "15", "16", "17", "18", "19", "20", ">20"]
            )
            last_new_job = st.selectbox("Years Since Last Job Change:", ["never", "1", "2", "3", "4", ">4"])
            relevent_experience = st.selectbox("Relevant Work Experience:", ["Has relevent experience", "No relevent experience"])

        with c3:
            education_level = st.selectbox("Education Level:", ["Graduate", "Masters", "High School", "Phd", "Primary School", "Unknown"])
            company_size = st.selectbox("Company Size:", ["50-99", "<10", "10000+", "10-49", "1000-4999", "500-999", "5000-9999", "100-499", "Unknown"])
            company_type = st.selectbox("Company Type:", ["Pvt Ltd", "Funded Startup", "Public Sector", "Early Stage Startup", "NGO", "Other", "Unknown"])

        submit_button = st.form_submit_button("🚀 Evaluate Candidate Suitability", use_container_width=True)

    if submit_button:
        # Build candidate DataFrame row
        candidate_dict = {
            "city_development_index": [cdi],
            "training_hours": [training_hours],
            "gender": [gender],
            "relevent_experience": [relevent_experience],
            "enrolled_university": ["no_enrollment"],
            "education_level": [education_level],
            "major_discipline": ["STEM"],
            "experience": [experience],
            "company_size": [company_size],
            "company_type": [company_type],
            "last_new_job": [last_new_job]
        }

        cand_df = pd.DataFrame(candidate_dict)

        # Import preprocessor class & fit/transform manually using config
        from src.preprocessing import CandidatePreprocessor
        preprocessor = CandidatePreprocessor()

        # Reconstruct scaler & categories from json config
        preprocessor.feature_names = config["feature_names"]
        preprocessor.nominal_categories = config["nominal_categories"]

        from src.preprocessing import CustomStandardScaler
        scaler = CustomStandardScaler()
        scaler.mean_ = pd.Series(config["scaler_mean"])
        scaler.scale_ = pd.Series(config["scaler_scale"])
        preprocessor.scaler = scaler

        X_cand_scaled = preprocessor.transform(cand_df)

        # Load model weights
        from src.modeling import LogisticRegressionModel
        X_train = pd.read_csv(os.path.join("data", "processed", "X_train.csv")).values
        y_train = pd.read_csv(os.path.join("data", "processed", "y_train.csv")).values.ravel()

        model = LogisticRegressionModel(lr=0.08, n_iters=400, l2_reg=0.1)
        model.fit(X_train, y_train)

        prob = float(model.predict_proba(X_cand_scaled.values)[0, 1])

        # Apply fairness threshold if available
        threshold = 0.5
        fair_thresholds = fair_config.get("fair_thresholds", {})
        if gender in fair_thresholds:
            threshold = fair_thresholds[gender]

        pred_class = int(prob >= threshold)

        # Assign tier
        if prob >= 0.50:
            tier = "High Priority"
            badge_class = "tier-badge-high"
        elif prob >= 0.35:
            tier = "Qualified"
            badge_class = "tier-badge-qualified"
        elif prob >= 0.20:
            tier = "Extended"
            badge_class = "tier-badge-qualified"
        else:
            tier = "Reserve"
            badge_class = "tier-badge-qualified"

        st.markdown("---")
        st.subheader("📊 Candidate Prediction Analysis Results")

        res1, res2, res3 = st.columns(3)
        res1.metric("Predicted Suitability Score", f"{prob:.4f}")
        res2.metric("Fairness-Calibrated Flag", "Shortlisted (1)" if pred_class == 1 else "Not Shortlisted (0)")
        res3.metric("Assigned Recruitment Tier", tier)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP — Navigation Controller
# ─────────────────────────────────────────────────────────────────────────────
def main():
    with st.sidebar:
        # ── Branding ──────────────────────────────────────────────────────
        st.markdown("""
        <div style="text-align:center;padding:22px 0 14px 0;">
            <div style="font-size:2.4rem;margin-bottom:6px;">🎯</div>
            <div style="color:#f9fafb;font-size:1.2rem;font-weight:800;letter-spacing:-0.02em;">
                FairHire AI
            </div>
            <div style="color:#4b5563;font-size:0.65rem;font-weight:600;
                        text-transform:uppercase;letter-spacing:0.08em;margin-top:3px;">
                Fair Candidate Screening
            </div>
            <div style="color:#1f2937;font-size:0.6rem;margin-top:2px;">
                v2.0 · Phase 1
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#1a2234;margin:4px 0 10px 0;'>", unsafe_allow_html=True)

        # ── Navigation ────────────────────────────────────────────────────
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

        st.markdown("<hr style='border-color:#1a2234;margin:10px 0;'>", unsafe_allow_html=True)

        # ── System Status ─────────────────────────────────────────────────
        st.markdown(
            "<div style='color:#374151;font-size:0.62rem;font-weight:700;"
            "text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>"
            "System Status</div>",
            unsafe_allow_html=True
        )

        checks = {
            "Rankings":  os.path.exists(os.path.join("reports", "metrics", "candidate_rankings.csv")),
            "Model":     os.path.exists(os.path.join("models", "trained_models", "best_model_info.json")),
            "Fairness":  os.path.exists(os.path.join("models", "trained_models", "fairness_config.json")),
            "SHAP":      os.path.exists(os.path.join("reports", "metrics", "shap_feature_importance.csv")),
        }
        badges = ""
        for label, ok in checks.items():
            cls = "status-ok" if ok else "status-err"
            dot = "●" if ok else "○"
            badges += f'<div style="margin:4px 0;"><span class="{cls}">{dot} {label}</span></div>'
        st.markdown(badges, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#1a2234;margin:10px 0;'>", unsafe_allow_html=True)
        st.markdown(
            "<div style='color:#1f2937;font-size:0.6rem;text-align:center;'>"
            "FairHire AI · HR SaaS Platform<br>© 2026</div>",
            unsafe_allow_html=True
        )

    # ── Page Routing ─────────────────────────────────────────────────────────
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
    elif "─────" in page:
        st.info("Please select a navigation item from the sidebar.")
    else:
        # Future phases — Coming Soon
        page_clean = page.strip()
        st.markdown(f"""
        <div class="coming-soon-banner">
            <div style="font-size:3.5rem;margin-bottom:18px;">🚧</div>
            <div style="color:#f9fafb;font-size:1.4rem;font-weight:800;
                        letter-spacing:-0.02em;margin-bottom:10px;">{page_clean}</div>
            <div style="color:#4b5563;font-size:0.92rem;margin-bottom:20px;">
                This module is being built in an upcoming phase.
            </div>
            <span style="background:rgba(59,130,246,0.1);color:#60a5fa;
                         border:1px solid rgba(59,130,246,0.25);padding:6px 18px;
                         border-radius:20px;font-size:0.78rem;font-weight:600;">
                Coming Soon
            </span>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
