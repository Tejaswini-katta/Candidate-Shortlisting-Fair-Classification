# AI-Powered Candidate Shortlisting Using Fair Classification Models

An end-to-end Machine Learning system that automates candidate shortlisting while minimizing algorithmic bias across sensitive attributes (**Gender**), incorporating Fairlearn metric framing, SHAP explainability, and an interactive Streamlit dashboard.

[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)

---

## 📌 Project Overview
Traditional candidate selection methods can perpetuate unconscious bias. This project develops a fair classification system using the Kaggle **HR Analytics: Job Change of Data Scientists** dataset ($N = 19,158$). It evaluates candidate suitability probabilities while explicitly auditing and mitigating demographic bias across protected attributes (**Gender**) using group-calibrated decision threshold optimization.

---

## 🛠️ Key Architectural Features
1. **Modular Codebase**: Production Python architecture with dedicated modules for data loading, EDA, preprocessing, model comparison, fairness auditing, SHAP explainability, and candidate ranking.
2. **Data Leakage Protection**: Preprocessing scalers and categorical encoders are fitted **strictly on training data** and applied to validation and test sets.
3. **Multi-Model Benchmark**: Trains and compares Logistic Regression, K-Nearest Neighbors, Random Forest, and Gradient Boosting models across Stratified Cross-Validation metrics.
4. **Fairness Audit & Mitigation**: Evaluates Demographic Parity, Equal Opportunity, and Equalized Odds using Fairlearn metric framing. Enforces group-specific threshold optimization to reduce demographic selection disparities by **88.3%**.
5. **Explainable AI (SHAP Framework)**: Derives global feature importance rankings and candidate-level decision attributions ($f_i(x) = w_i \cdot (x_i - \bar{x}_i)$).
6. **Actionable Ranking Engine**: Ranks test candidates ($N = 2,129$) into 4 priority recruitment tiers (High Priority, Qualified, Extended, Reserve).
7. **Streamlit Web Dashboard**: 5-page interactive UI for shortlisting exploration, fairness metric visualization, SHAP explanations, and real-time candidate suitability scoring.

---

## 📂 Repository Structure

```
Candidate-Shortlisting-Fair-Classification/
│
├── .streamlit/
│   └── config.toml                 # Custom dark theme configuration
│
├── data/
│   ├── raw/                        # Raw Kaggle CSV datasets (aug_train.csv, aug_test.csv)
│   └── processed/                  # Processed & scaled data matrices (X_train, X_val, X_test)
│
├── src/                            # Production Python Source Package
│   ├── __init__.py
│   ├── data_loader.py              # Raw dataset loading & structural checks
│   ├── eda.py                      # Exploratory Data Analysis & visual plotting
│   ├── preprocessing.py            # Missing value imputation, encoding & Z-Score scaling
│   ├── modeling.py                 # Logistic Regression, KNN & ensemble classifiers
│   ├── fairness.py                 # Fairlearn audit & threshold optimization engine
│   ├── explainability.py           # SHAP global & local feature attributions
│   └── ranking.py                  # Candidate probability scoring & tier ranker
│
├── models/
│   └── trained_models/             # Preprocessor, model & fairness JSON artifacts
│
├── reports/
│   ├── figures/                    # 19 Publication-quality visual plots (EDA, SHAP, Ranking)
│   └── metrics/                    # Performance, Fairness, SHAP & Ranking CSV reports
│
├── notebooks/                      # Jupyter Notebooks for walkthroughs
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 05_Candidate_Ranking.ipynb
│
├── app.py                          # Streamlit 5-Page Interactive Application
├── run_step3_and_4.py              # Preprocessing & Model Training Runner
├── run_fairness.py                 # Fairness Audit & Bias Mitigation Runner
├── run_explainability.py           # SHAP Explainability Runner
├── run_ranking.py                  # Candidate Ranking Engine Runner
├── requirements.txt                # Dependency requirements
├── PROJECT_PROGRESS.md             # Progress tracker
└── README.md                       # Documentation
```

---

## 📊 End-to-End Execution & Results

### 1. Data Cleaning & Feature Preprocessing
- **Categorical Imputation**: Assigned missing values (`gender`, `company_type`, `company_size`) to explicit `'Unknown'` categories to prevent data loss.
- **Ordinal Encoding**: Explicit integer mapping for `experience` ($<1 \rightarrow 0, >20 \rightarrow 21$), `last_new_job`, `education_level`, `company_size`, and `relevent_experience`.
- **Nominal Encoding**: One-Hot Encoding for `gender`, `enrolled_university`, `major_discipline`, and `company_type`.
- **Feature Scaling**: Z-Score `StandardScaler` fit on `X_train`.

### 2. Model Performance Benchmark
Evaluated on Stratified Validation Set ($N = 3,831$):

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Top Model |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Benchmark** | 0.7541 | 0.5596 | 0.0639 | 0.1147 | **0.7854** | ⭐ **Best** |
| **Gradient Boosting (XGBoost)** | 0.7507 | 0.0000 | 0.0000 | 0.0000 | **0.7839** | Runner-up |
| **Logistic Regression** | **0.7669** | **0.5735** | **0.2534** | **0.3515** | **0.7682** | Top Linear |
| **KNN Classifier** | 0.7502 | 0.4983 | 0.3152 | 0.3861 | 0.7164 | Baseline |

### Model Architecture

The project evaluates multiple machine learning models during benchmarking. Random Forest achieved the highest validation ROC-AUC of 0.7854.

For the deployed candidate screening workflow (including **Resume Screening**, **Real-Time Prediction**, and **Candidate Ranking**), Logistic Regression is used consistently. This ensures that predictions, fairness thresholds, and SHAP explanations remain consistent across the application. The **AI Resume Screening & Validation** module processes uploaded documents entirely in memory/session state without altering the existing training or fairness artifacts, ensuring strict data immutability. It includes deterministic field-level extraction confidence scoring (High/Medium/Low), a 0–100% Resume Quality Score, recruiter edit tracking, and data trustworthiness auditing.

### 3. Algorithmic Fairness Audit (Fairlearn Framework)
Post-processing decision threshold optimization across **Gender** groups ($N = 3,831$ Validation Candidates):

| Fairness Metric | Raw Baseline Model | Fair Mitigated Model | Disparity Reduction |
| :--- | :---: | :---: | :---: |
| **Demographic Parity Difference** | 0.1088 | **0.0127** | **88.3% Reduction** |
| **Equal Opportunity Difference** | 0.1719 | **0.0692** | **59.7% Reduction** |
| **Equalized Odds Difference** | 0.1719 | **0.0692** | **59.7% Reduction** |

- **Calibrated Decision Thresholds**:
  - `Female`: $t = 0.4700$
  - `Male`: $t = 0.4600$
  - `Other`: $t = 0.4900$
  - `Unknown`: $t = 0.6000$

### 4. Explainable AI (SHAP Framework)
- **Top Global Predictors**: `city_development_index` (28.2% weight), `company_type_Unknown` (13.3%), `company_type_Pvt Ltd` (8.8%), `major_discipline_Unknown` (8.6%), and `experience` (6.4%).
- **Generated Figures**: [`reports/figures/17_shap_feature_importance.png`](file:///c:/Users/tejas/OneDrive/Documents/Candidate-Shortlisting-Fair-Classification/reports/figures/17_shap_feature_importance.png).

### 5. Candidate Probability Ranking Engine ($N = 2,129$ Testing Candidates)
- **High Priority (Top 10%)**: 213 candidates (Top suitability scores $\ge 0.50$)
- **Qualified (Top 25% Pool)**: 320 candidates
- **Extended (Top 50% Pool)**: 532 candidates
- **Reserve Pool**: 1,064 candidates

---

## 🚀 Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/your-username/Candidate-Shortlisting-Fair-Classification.git
cd Candidate-Shortlisting-Fair-Classification
pip install -r requirements.txt
```

### 2. Execute Full Pipeline
```bash
# Preprocessing & Model Comparison
python run_step3_and_4.py

# Fairness Audit & Mitigation
python run_fairness.py

# SHAP Explainability
python run_explainability.py

# Candidate Probability Ranking
python run_ranking.py
```

### 3. Launch Streamlit Web Application
```bash
streamlit run app.py
```

---

## ⚖️ Ethical Considerations & Responsible AI
1. **Human-in-the-Loop Oversight**: Automated suitability scores and priority tiers serve strictly as decision support tools for recruiters, not autonomous rejection filters.
2. **Mitigating Historical Representation Gaps**: By tuning group-specific decision boundaries, the system ensures qualified candidates from underrepresented demographic backgrounds receive equal shortlisting opportunities.
3. **Data Privacy & Compliance**: Personal identity numbers (`enrollee_id`) are kept isolated from model training features to preserve privacy.

---

## 📄 License
This project is open-source under the **MIT License**.
