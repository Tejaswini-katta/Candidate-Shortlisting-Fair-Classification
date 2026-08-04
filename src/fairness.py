"""
Fairness Analysis & Bias Audit Module for Candidate Shortlisting.

Audits candidate classification models across protected attributes (Gender) using
Fairlearn metric framing concepts:
- Selection Rate
- Demographic Parity Difference
- Equal Opportunity Difference (TPR equality)
- Equalized Odds Difference (TPR + FPR equality)

Provides threshold optimization bias mitigation to reduce disparate demographic impact.
"""

import os
import json
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np


def compute_group_metrics(y_true: np.ndarray, y_pred: np.ndarray, sensitive_attr: np.ndarray) -> pd.DataFrame:
    """
    Compute Selection Rate, True Positive Rate (Recall), and False Positive Rate
    for each demographic subgroup.
    """
    unique_groups = np.unique(sensitive_attr)
    metrics_list = []

    for group in unique_groups:
        idx = (sensitive_attr == group)
        y_t = y_true[idx]
        y_p = y_pred[idx]

        n_samples = len(y_t)
        if n_samples == 0:
            continue

        selection_rate = np.mean(y_p)

        # TP, FP, TN, FN
        tp = np.sum((y_t == 1) & (y_p == 1))
        fp = np.sum((y_t == 0) & (y_p == 1))
        fn = np.sum((y_t == 1) & (y_p == 0))
        tn = np.sum((y_t == 0) & (y_p == 0))

        tpr = tp / (tp + fn + 1e-7)  # Equal Opportunity metric
        fpr = fp / (fp + tn + 1e-7)  # False Positive Rate

        metrics_list.append({
            "Subgroup": group,
            "Sample Count": n_samples,
            "Selection Rate": round(float(selection_rate), 4),
            "TPR (Equal Opportunity)": round(float(tpr), 4),
            "FPR": round(float(fpr), 4)
        })

    return pd.DataFrame(metrics_list)


def compute_fairness_summary(group_metrics_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate summary disparity metrics across demographic groups:
    - Demographic Parity Difference: Max(Selection Rate) - Min(Selection Rate)
    - Equal Opportunity Difference: Max(TPR) - Min(TPR)
    - Equalized Odds Difference: Max(Max(TPR) - Min(TPR), Max(FPR) - Min(FPR))
    """
    sel_rates = group_metrics_df["Selection Rate"].values
    tprs = group_metrics_df["TPR (Equal Opportunity)"].values
    fprs = group_metrics_df["FPR"].values

    demographic_parity_diff = np.max(sel_rates) - np.min(sel_rates)
    equal_opportunity_diff = np.max(tprs) - np.min(tprs)
    fpr_diff = np.max(fprs) - np.min(fprs)
    equalized_odds_diff = max(equal_opportunity_diff, fpr_diff)

    return {
        "Demographic Parity Difference": round(float(demographic_parity_diff), 4),
        "Equal Opportunity Difference": round(float(equal_opportunity_diff), 4),
        "Equalized Odds Difference": round(float(equalized_odds_diff), 4)
    }


def optimize_fair_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sensitive_attr: np.ndarray,
    target_metric: str = "demographic_parity"
) -> Dict[str, float]:
    """
    Mitigate algorithmic bias using group-specific decision threshold adjustment.
    Finds optimal prediction threshold per demographic group to minimize demographic parity difference.
    """
    unique_groups = np.unique(sensitive_attr)
    thresholds = {}

    # Target selection rate to equalize across groups (mean unmitigated selection rate)
    base_selection_rate = np.mean(y_proba >= 0.5)

    for group in unique_groups:
        idx = (sensitive_attr == group)
        group_proba = y_proba[idx]

        if len(group_proba) == 0:
            thresholds[group] = 0.5
            continue

        best_t = 0.5
        min_diff = 1.0

        for t in np.linspace(0.1, 0.9, 81):
            rate = np.mean(group_proba >= t)
            diff = abs(rate - base_selection_rate)
            if diff < min_diff:
                min_diff = diff
                best_t = t

        thresholds[str(group)] = round(float(best_t), 4)

    return thresholds


def apply_fair_thresholds(
    y_proba: np.ndarray,
    sensitive_attr: np.ndarray,
    thresholds: Dict[str, float]
) -> np.ndarray:
    """Apply group-specific classification thresholds."""
    y_pred_fair = np.zeros(len(y_proba), dtype=int)

    for group, t in thresholds.items():
        idx = (sensitive_attr.astype(str) == str(group))
        y_pred_fair[idx] = (y_proba[idx] >= t).astype(int)

    return y_pred_fair


def run_fairness_audit(
    processed_dir: str = os.path.join("data", "processed"),
    raw_train_path: str = os.path.join("data", "raw", "aug_train.csv"),
    models_dir: str = os.path.join("models", "trained_models"),
    reports_dir: str = os.path.join("reports", "metrics")
) -> Dict[str, Any]:
    """
    Execute full fairness audit:
    1. Reconstruct protected attribute ('gender') for validation set.
    2. Audit baseline model prediction fairness across gender groups.
    3. Perform threshold optimization bias mitigation.
    4. Export before & after fairness comparison reports.
    """
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 5: FAIRNESS ANALYSIS & BIAS AUDIT (FAIRLEARN FRAMEWORK)")
    print("=" * 60)

    # 1. Load Processed Validation Data & Raw Datasets
    X_val = pd.read_csv(os.path.join(processed_dir, "X_val.csv"))
    y_val = pd.read_csv(os.path.join(processed_dir, "y_val.csv")).values.ravel()
    val_ids = pd.read_csv(os.path.join(processed_dir, "val_enrollee_ids.csv")).values.ravel()

    raw_train = pd.read_csv(raw_train_path)

    # Map enrollee_id to raw gender feature
    id_to_gender = dict(zip(raw_train["enrollee_id"], raw_train["gender"].fillna("Unknown")))
    gender_val = np.array([id_to_gender.get(eid, "Unknown") for eid in val_ids])

    print(f"[INFO] Audit Sample Size: {len(y_val)} candidates")
    print("[INFO] Protected Attribute (Gender) Subgroup Counts:")
    for g, count in pd.Series(gender_val).value_counts().items():
        print(f"   - {g}: {count} candidates ({(count/len(gender_val))*100:.1f}%)")

    # Re-run model probabilities for audit
    from src.modeling import LogisticRegressionModel
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv")).values
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).values.ravel()

    model = LogisticRegressionModel(lr=0.08, n_iters=400, l2_reg=0.1)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_val.values)[:, 1]
    y_pred_raw = (y_proba >= 0.5).astype(int)

    # 2. Raw Baseline Model Fairness Audit
    raw_group_metrics = compute_group_metrics(y_val, y_pred_raw, gender_val)
    raw_summary = compute_fairness_summary(raw_group_metrics)

    print("\n--- Raw Model Subgroup Performance ---")
    print(raw_group_metrics.to_string(index=False))

    print("\n--- Raw Model Disparity Summary ---")
    for metric_name, val in raw_summary.items():
        print(f"   - {metric_name}: {val:.4f}")

    # 3. Bias Mitigation: Group-Specific Threshold Tuning
    fair_thresholds = optimize_fair_thresholds(y_val, y_proba, gender_val)
    y_pred_mitigated = apply_fair_thresholds(y_proba, gender_val, fair_thresholds)

    mitigated_group_metrics = compute_group_metrics(y_val, y_pred_mitigated, gender_val)
    mitigated_summary = compute_fairness_summary(mitigated_group_metrics)

    print("\n--- Mitigated Model Subgroup Performance ---")
    print(mitigated_group_metrics.to_string(index=False))

    print("\n--- Mitigated Model Disparity Summary ---")
    for metric_name, val in mitigated_summary.items():
        print(f"   - {metric_name}: {val:.4f}")

    # 4. Save Audit Reports
    comparison_df = pd.DataFrame([
        {"Stage": "Raw Model (Unmitigated)", **raw_summary},
        {"Stage": "Fair Model (Mitigated)", **mitigated_summary}
    ])
    comparison_path = os.path.join(reports_dir, "fairness_audit_report.csv")
    comparison_df.to_csv(comparison_path, index=False)

    fair_info = {
        "fair_thresholds": fair_thresholds,
        "raw_summary": raw_summary,
        "mitigated_summary": mitigated_summary
    }
    with open(os.path.join(models_dir, "fairness_config.json"), "w") as f:
        json.dump(fair_info, f, indent=2)

    print(f"\n[OK] Saved fairness audit report to: {comparison_path}")
    print(f"[OK] Saved mitigation thresholds to: {os.path.join(models_dir, 'fairness_config.json')}")
    print("=" * 60)

    return {
        "raw_summary": raw_summary,
        "mitigated_summary": mitigated_summary,
        "fair_thresholds": fair_thresholds
    }


if __name__ == "__main__":
    run_fairness_audit()
