"""
Runner script to execute Step 6 Explainable AI & Model Attribution.
"""

from src.explainability import run_explainability_pipeline


def main():
    results = run_explainability_pipeline()

    print("\n" + "=" * 60)
    print("EXPLAINABLE AI (SHAP) SUMMARY REPORT")
    print("=" * 60)
    print(f"Attribution Method: {results['method_used']}")

    top_row = results["importance_df"].iloc[0]
    print(f"Top Predictor: '{top_row['Feature']}' (Impact: {top_row['Importance_Percentage']:.2f}%)")

    sample_exp = results["sample_explanation"]
    print(f"\nSample Candidate Enrollee ID: {sample_exp['enrollee_id']}")
    print(f"Predicted Probability Score: {sample_exp['candidate_predicted_probability']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
