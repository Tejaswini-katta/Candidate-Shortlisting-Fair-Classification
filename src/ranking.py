"""
Candidate Probability Ranking & Shortlisting Engine.

Scores test candidates, applies fairness-calibrated decision thresholds,
ranks candidates by predicted suitability probability, and buckets them into
actionable hiring priority tiers:
- Top 10%: High Priority
- Top 25%: Qualified
- Top 50%: Extended
- Remaining: Reserve
"""

import os
import json
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class CandidateRankingEngine:
    """
    Production candidate ranking engine.
    Computes probabilities, applies fairness thresholds, and assigns priority tiers.
    """

    def __init__(
        self,
        processed_dir: str = os.path.join("data", "processed"),
        raw_test_path: str = os.path.join("data", "raw", "aug_test.csv"),
        models_dir: str = os.path.join("models", "trained_models")
    ):
        self.processed_dir = processed_dir
        self.raw_test_path = raw_test_path
        self.models_dir = models_dir

    def load_artifacts_and_data(self) -> Tuple[Any, pd.DataFrame, pd.Series, pd.Series]:
        """Load trained model, test feature matrix, enrollee IDs, and raw gender series."""
        # Load test feature matrix and IDs from data/processed
        X_test_df = pd.read_csv(os.path.join(self.processed_dir, "X_test.csv"))
        test_ids = pd.read_csv(os.path.join(self.processed_dir, "test_enrollee_ids.csv")).values.ravel()

        # Load raw test dataset to extract sensitive attribute (gender)
        raw_test = pd.read_csv(self.raw_test_path)
        raw_gender = raw_test["gender"].fillna("Unknown")

        # Load trained model
        from src.modeling import LogisticRegressionModel
        X_train = pd.read_csv(os.path.join(self.processed_dir, "X_train.csv")).values
        y_train = pd.read_csv(os.path.join(self.processed_dir, "y_train.csv")).values.ravel()

        model = LogisticRegressionModel(lr=0.08, n_iters=400, l2_reg=0.1)
        model.fit(X_train, y_train)

        return model, X_test_df, test_ids, raw_gender

    def rank_candidates(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Execute probability prediction, fairness thresholding, sorting, and tier assignment.
        """
        model, X_test_df, test_ids, raw_gender = self.load_artifacts_and_data()

        # Predict probability scores
        probas = model.predict_proba(X_test_df.values)[:, 1]

        # Check for fairness config thresholds from Step 5
        fairness_config_path = os.path.join(self.models_dir, "fairness_config.json")
        fair_thresholds = None
        if os.path.exists(fairness_config_path):
            with open(fairness_config_path, "r") as f:
                config = json.load(f)
                fair_thresholds = config.get("fair_thresholds", None)

        # Assign predicted class using group-specific thresholds if available
        predicted_classes = []
        for prob, gender in zip(probas, raw_gender):
            threshold = 0.5
            if fair_thresholds and str(gender) in fair_thresholds:
                threshold = fair_thresholds[str(gender)]
            predicted_classes.append(int(prob >= threshold))

        # Build raw ranking dataframe
        df = pd.DataFrame({
            "enrollee_id": test_ids,
            "prediction_probability": probas.round(4),
            "predicted_class": predicted_classes,
            "gender": raw_gender
        })

        # Sort candidates descending by prediction probability
        df = df.sort_values(by="prediction_probability", ascending=False).reset_index(drop=True)

        # Calculate Percentile Rank (100.0 = Top candidate, 0.0 = Bottom candidate)
        total_candidates = len(df)
        df["rank"] = range(1, total_candidates + 1)
        df["percentile_rank"] = (((total_candidates - df["rank"] + 1) / total_candidates) * 100).round(2)

        # Assign Priority Tiers based on percentiles
        p10_cutoff = np.percentile(df["prediction_probability"], 90)
        p25_cutoff = np.percentile(df["prediction_probability"], 75)
        p50_cutoff = np.percentile(df["prediction_probability"], 50)

        def assign_tier(prob: float) -> str:
            if prob >= p10_cutoff:
                return "High Priority"
            elif prob >= p25_cutoff:
                return "Qualified"
            elif prob >= p50_cutoff:
                return "Extended"
            else:
                return "Reserve"

        df["priority_tier"] = df["prediction_probability"].apply(assign_tier)

        # Clean display columns
        ranking_df = df[[
            "enrollee_id", "prediction_probability", "predicted_class",
            "percentile_rank", "priority_tier"
        ]]

        top10_df = ranking_df[ranking_df["priority_tier"] == "High Priority"].reset_index(drop=True)
        top25_df = ranking_df[ranking_df["priority_tier"].isin(["High Priority", "Qualified"])].reset_index(drop=True)
        top50_df = ranking_df[ranking_df["priority_tier"].isin(["High Priority", "Qualified", "Extended"])].reset_index(drop=True)

        return ranking_df, top10_df, top25_df, top50_df


def plot_ranking_visualizations(
    ranking_df: pd.DataFrame,
    figures_dir: str = os.path.join("reports", "figures")
) -> None:
    """Generate probability distribution and priority tier bar plot figures."""
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Probability Distribution Plot
    plt.figure(figsize=(10, 5))
    ax = sns.histplot(data=ranking_df, x="prediction_probability", kde=True, color="#2b5c8f", bins=30)

    # Add cutoffs
    p10_val = np.percentile(ranking_df["prediction_probability"], 90)
    p25_val = np.percentile(ranking_df["prediction_probability"], 75)
    p50_val = np.percentile(ranking_df["prediction_probability"], 50)

    plt.axvline(p10_val, color="#e76f51", linestyle="--", linewidth=1.8, label=f"Top 10% Cutoff ({p10_val:.3f})")
    plt.axvline(p25_val, color="#2a9d8f", linestyle="--", linewidth=1.8, label=f"Top 25% Cutoff ({p25_val:.3f})")
    plt.axvline(p50_val, color="#e9c46a", linestyle="--", linewidth=1.8, label=f"Top 50% Cutoff ({p50_val:.3f})")

    plt.title("Candidate Shortlisting Probability Distribution", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Job Change / Suitability Probability", fontsize=11)
    plt.ylabel("Candidate Count", fontsize=11)
    plt.legend(loc="upper right")
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "18_candidate_probability_distribution.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # 2. Priority Tier Distribution Plot
    tier_order = ["High Priority", "Qualified", "Extended", "Reserve"]
    tier_counts = ranking_df["priority_tier"].value_counts().reindex(tier_order).fillna(0)

    plt.figure(figsize=(9, 5))
    palette = ["#e76f51", "#2a9d8f", "#e9c46a", "#a8ded5"]
    bars = plt.bar(tier_counts.index, tier_counts.values, color=palette, edgecolor="#1a365d")

    plt.title("Candidate Priority Tier Distribution (Test Cohort)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Priority Tier", fontsize=11)
    plt.ylabel("Number of Candidates", fontsize=11)

    total = len(ranking_df)
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 15,
            f"{int(height)}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "19_candidate_tier_distribution.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()


def save_ranking_reports(
    ranking_df: pd.DataFrame,
    top10_df: pd.DataFrame,
    top25_df: pd.DataFrame,
    top50_df: pd.DataFrame,
    reports_dir: str = os.path.join("reports", "metrics")
) -> None:
    """Save candidate ranking CSV artifacts to reports/metrics/."""
    os.makedirs(reports_dir, exist_ok=True)

    ranking_df.to_csv(os.path.join(reports_dir, "candidate_rankings.csv"), index=False)
    top10_df.to_csv(os.path.join(reports_dir, "top10_candidates.csv"), index=False)
    top25_df.to_csv(os.path.join(reports_dir, "top25_candidates.csv"), index=False)
    top50_df.to_csv(os.path.join(reports_dir, "top50_candidates.csv"), index=False)


def run_ranking_pipeline(
    processed_dir: str = os.path.join("data", "processed"),
    raw_test_path: str = os.path.join("data", "raw", "aug_test.csv"),
    models_dir: str = os.path.join("models", "trained_models"),
    reports_dir: str = os.path.join("reports", "metrics"),
    figures_dir: str = os.path.join("reports", "figures")
) -> Dict[str, Any]:
    """Execute candidate probability ranking pipeline."""
    print("=" * 60)
    print("STEP 7: CANDIDATE PROBABILITY RANKING ENGINE")
    print("=" * 60)

    engine = CandidateRankingEngine(processed_dir, raw_test_path, models_dir)
    ranking_df, top10_df, top25_df, top50_df = engine.rank_candidates()

    print(f"[INFO] Successfully ranked {len(ranking_df)} candidates in testing cohort.")

    # Save reports
    save_ranking_reports(ranking_df, top10_df, top25_df, top50_df, reports_dir)
    print(f"[OK] Saved candidate_rankings.csv to: {reports_dir}")
    print(f"[OK] Saved top10_candidates.csv ({len(top10_df)} candidates)")
    print(f"[OK] Saved top25_candidates.csv ({len(top25_df)} candidates)")
    print(f"[OK] Saved top50_candidates.csv ({len(top50_df)} candidates)")

    # Plot figures
    plot_ranking_visualizations(ranking_df, figures_dir)
    print(f"[OK] Saved 18_candidate_probability_distribution.png to: {figures_dir}")
    print(f"[OK] Saved 19_candidate_tier_distribution.png to: {figures_dir}")

    tier_counts = ranking_df["priority_tier"].value_counts().to_dict()

    print("=" * 60)

    return {
        "total_candidates": len(ranking_df),
        "ranking_df": ranking_df,
        "tier_counts": tier_counts,
        "top10_count": len(top10_df),
        "top25_count": len(top25_df),
        "top50_count": len(top50_df)
    }


if __name__ == "__main__":
    run_ranking_pipeline()
