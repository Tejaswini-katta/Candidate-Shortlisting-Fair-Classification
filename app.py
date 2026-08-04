"""
AI-Powered Candidate Shortlisting & Fair Classification Dashboard.

Streamlit multi-page web application providing:
1. Home Overview & ML Pipeline Architecture
2. Interactive Candidate Probability Ranking & Shortlist Export
3. Fairlearn Bias Audit & Demographic Parity Dashboard
4. SHAP Global & Local Feature Explainability Portal
5. Real-Time Single Candidate Shortlisting Predictor

Run locally:
    streamlit run app.py
"""

import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


# Set page config as early as possible
st.set_page_config(
    page_title="AI Candidate Shortlisting & Fair ML",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for modern UI aesthetic
STYLING_CSS = """
<style>
    /* Global Container Padding & Colors */
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    /* Header Gradient & Cards */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .metric-card h3 {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 6px;
        font-weight: 500;
    }
    
    .metric-card p {
        color: #38bdf8;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Tier Badge Styling */
    .tier-badge-high {
        background-color: rgba(231, 111, 81, 0.2);
        color: #e76f51;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    
    .tier-badge-qualified {
        background-color: rgba(42, 157, 143, 0.2);
        color: #2a9d8f;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
"""

st.markdown(STYLING_CSS, unsafe_allow_html=True)


# Data loading helpers with cache
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


# -----------------------------------------------------------------------------
# PAGE 1: HOME OVERVIEW
# -----------------------------------------------------------------------------
def render_home_page():
    st.markdown("""
    <div class="main-header">
        <h1 style="color: #38bdf8; margin-bottom: 8px;">AI-Powered Candidate Shortlisting & Fair Classification</h1>
        <p style="color: #cbd5e1; font-size: 1.1rem; margin: 0;">
            An end-to-end Machine Learning pipeline that automates candidate shortlisting while actively auditing and mitigating bias across sensitive attributes (Gender), powered by Fairlearn and SHAP explainability.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h3>TRAINING CANDIDATES</h3><p>19,158</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>ENGINEERED FEATURES</h3><p>29</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>TOP MODEL ROC-AUC</h3><p>0.7854</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h3>BIAS REDUCTION</h3><p>88.3%</p></div>', unsafe_allow_html=True)

    st.markdown("### 🛠️ ML Pipeline & Architectural Workflow")
    st.markdown("""
    1. **Data Ingestion & Structural Inspection**: Load Kaggle HR Analytics dataset (`aug_train.csv` & `aug_test.csv`) retaining candidate tracking IDs (`enrollee_id`).
    2. **Exploratory Data Analysis**: Univariate & bivariate distribution visualizer across demographic and candidate features.
    3. **Data Preprocessing & Encoding**: Categorical missing value imputation (`'Unknown'`), Ordinal Mapping, One-Hot Encoding, and Z-Score scaling.
    4. **Model Comparison**: Train and compare Logistic Regression, KNN, Random Forest, and Gradient Boosting.
    5. **Fairness Audit (Fairlearn)**: Evaluate Demographic Parity, Equal Opportunity, and apply post-processing threshold tuning.
    6. **SHAP Explainability**: Compute global feature importance rankings and local candidate attribution breakdowns.
    7. **Probability Ranking Engine**: Rank candidates descending by suitability probability into 4 priority recruitment tiers.
    """)

    st.markdown("### 💻 Technology Stack")
    st.markdown("""
    - **Core**: Python 3.11, Pandas, NumPy, Scikit-Learn
    - **Fairness & Explainability**: Fairlearn, SHAP, Matplotlib, Seaborn
    - **Web Interface**: Streamlit Dashboard
    """)


# -----------------------------------------------------------------------------
# PAGE 2: CANDIDATE RANKING
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# PAGE 3: FAIRNESS DASHBOARD
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# PAGE 4: EXPLAINABILITY
# -----------------------------------------------------------------------------
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
            st.image(fig_path, use_column_width=True)
        else:
            st.warning("SHAP summary plot missing.")

    st.markdown("---")
    st.markdown("### 👤 Sample Candidate Local SHAP Decision Breakdown")
    if sample_json:
        st.json(sample_json)
    else:
        st.info("Sample candidate explanation JSON missing.")


# -----------------------------------------------------------------------------
# PAGE 5: CANDIDATE PREDICTION
# -----------------------------------------------------------------------------
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
                ["<1", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", ">20"]
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


# -----------------------------------------------------------------------------
# MAIN APP NAVIGATION CONTROLLER
# -----------------------------------------------------------------------------
def main():
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio(
        "Select Dashboard View:",
        [
            "1. Home & Pipeline Overview",
            "2. Candidate Probability Ranking",
            "3. Fairness & Bias Audit",
            "4. SHAP Model Explainability",
            "5. Real-Time Candidate Predictor"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.info("AI-Powered Candidate Shortlisting Project v1.0")

    if page.startswith("1"):
        render_home_page()
    elif page.startswith("2"):
        render_ranking_page()
    elif page.startswith("3"):
        render_fairness_page()
    elif page.startswith("4"):
        render_explainability_page()
    elif page.startswith("5"):
        render_prediction_page()


if __name__ == "__main__":
    main()
