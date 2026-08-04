"""
Runner script for Step 3 (Preprocessing) & Step 4 (Model Training & Evaluation).
"""

from src.preprocessing import run_preprocessing_pipeline
from src.modeling import train_and_compare_models


def main():
    print("\n>>> EXECUTING STEP 3: PREPROCESSING PIPELINE <<<")
    prep_res = run_preprocessing_pipeline()
    print(f"Preprocessed features count: {prep_res['feature_count']}")

    print("\n>>> EXECUTING STEP 4: MODEL TRAINING & COMPARISON <<<")
    model_res = train_and_compare_models()
    print(f"\nBest Model Identified: {model_res['best_model']} (ROC-AUC: {model_res['best_roc_auc']:.4f})")


if __name__ == "__main__":
    main()
