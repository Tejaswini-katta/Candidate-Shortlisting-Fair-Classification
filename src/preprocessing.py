"""
Data Preprocessing & Feature Engineering Module for Candidate Shortlisting.

This module provides a production-grade, self-contained preprocessing pipeline.
It handles categorical missing values, ordinal encoding, nominal One-Hot Encoding,
stratified train/validation splitting, and feature scaling using Pandas & NumPy,
preventing data leakage and removing strict external C-extension dependencies.
"""

import os
import json
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np


# Ordinal mappings for candidate features
ORDINAL_MAPPINGS = {
    "experience": {
        "<1": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
        "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
        "11": 11, "12": 12, "13": 13, "14": 14, "15": 15,
        "16": 16, "17": 17, "18": 18, "19": 19, "20": 20, ">20": 21
    },
    "last_new_job": {
        "never": 0, "1": 1, "2": 2, "3": 3, "4": 4, ">4": 5
    },
    "education_level": {
        "Primary School": 0, "High School": 1, "Graduate": 2, "Masters": 3, "Phd": 4
    },
    "company_size": {
        "<10": 0, "10-49": 1, "50-99": 2, "100-499": 3,
        "500-999": 4, "1000-4999": 5, "5000-9999": 6, "10000+": 7
    },
    "relevent_experience": {
        "No relevent experience": 0, "Has relevent experience": 1
    }
}

# Categorical features for One-Hot Encoding
NOMINAL_FEATURES = [
    "gender", "enrolled_university", "major_discipline", "company_type"
]


class CustomStandardScaler:
    """StandardScaler implementation using NumPy to prevent data leakage."""

    def __init__(self):
        self.mean_: Optional[pd.Series] = None
        self.scale_: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> "CustomStandardScaler":
        """Compute column-wise mean and standard deviation strictly on training data."""
        self.mean_ = df.mean()
        self.scale_ = df.std(ddof=0)
        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score normalization: (X - mean) / std."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("CustomStandardScaler must be fitted before calling transform().")
        return (df - self.mean_) / self.scale_

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


class CandidatePreprocessor:
    """
    Production candidate preprocessing pipeline.
    Handles ordinal encoding, nominal One-Hot Encoding, missing value imputation,
    and feature scaling strictly fitting stats on training data.
    """

    def __init__(self):
        self.scaler: Optional[CustomStandardScaler] = None
        self.nominal_categories: Dict[str, List[str]] = {}
        self.feature_names: List[str] = []
        self.numeric_cols: List[str] = ["city_development_index", "training_hours"]
        self.ordinal_cols: List[str] = list(ORDINAL_MAPPINGS.keys())

    def _clean_and_impute_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute categorical missing values with 'Unknown'."""
        df_clean = df.copy()
        cat_cols = NOMINAL_FEATURES + ["education_level", "company_size", "gender", "major_discipline", "company_type", "enrolled_university"]
        for col in cat_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna("Unknown")
        return df_clean

    def _encode_ordinals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map ordinal categorical strings to numerical ranks."""
        df_ord = df.copy()
        for col, mapping in ORDINAL_MAPPINGS.items():
            if col in df_ord.columns:
                df_ord[col] = df_ord[col].map(mapping).fillna(-1).astype(float)
        return df_ord

    def fit(self, train_df: pd.DataFrame) -> "CandidatePreprocessor":
        """Fit categories for One-Hot Encoding and scaler stats on training set."""
        train_clean = self._clean_and_impute_categoricals(train_df)
        train_ord = self._encode_ordinals(train_clean)

        # Store unique categories per nominal feature strictly from training set
        self.nominal_categories = {}
        for col in NOMINAL_FEATURES:
            cats = sorted(train_ord[col].astype(str).unique().tolist())
            self.nominal_categories[col] = cats

        # Construct fitted features DataFrame
        encoded_blocks = []
        for col in NOMINAL_FEATURES:
            for cat in self.nominal_categories[col]:
                col_name = f"{col}_{cat}"
                encoded_blocks.append(pd.Series((train_ord[col].astype(str) == cat).astype(float), name=col_name))

        feature_df = pd.concat(
            [train_ord[["city_development_index", "training_hours"] + self.ordinal_cols].reset_index(drop=True)] +
            [b.reset_index(drop=True) for b in encoded_blocks],
            axis=1
        )

        self.feature_names = list(feature_df.columns)
        self.scaler = CustomStandardScaler()
        self.scaler.fit(feature_df)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataset using fitted categories and scaler."""
        if self.scaler is None:
            raise RuntimeError("Preprocessor must be fitted before transform().")

        df_clean = self._clean_and_impute_categoricals(df)
        df_ord = self._encode_ordinals(df_clean)

        encoded_blocks = []
        for col in NOMINAL_FEATURES:
            for cat in self.nominal_categories[col]:
                col_name = f"{col}_{cat}"
                encoded_blocks.append(pd.Series((df_ord[col].astype(str) == cat).astype(float), name=col_name))

        feature_df = pd.concat(
            [df_ord[["city_development_index", "training_hours"] + self.ordinal_cols].reset_index(drop=True)] +
            [b.reset_index(drop=True) for b in encoded_blocks],
            axis=1
        )

        scaled_df = self.scaler.transform(feature_df)
        return scaled_df

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train_df).transform(train_df)


def stratified_train_val_split(
    df: pd.DataFrame,
    target_col: str = "target",
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform Stratified Train/Validation split preserving target class ratios.
    """
    np.random.seed(random_state)
    train_indices = []
    val_indices = []

    for label, group in df.groupby(target_col):
        shuffled = group.sample(frac=1.0, random_state=random_state)
        n_val = int(len(shuffled) * val_size)
        val_indices.extend(shuffled.index[:n_val])
        train_indices.extend(shuffled.index[n_val:])

    train_df = df.loc[train_indices].sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    val_df = df.loc[val_indices].sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return train_df, val_df


def run_preprocessing_pipeline(
    train_raw_path: str = os.path.join("data", "raw", "aug_train.csv"),
    test_raw_path: str = os.path.join("data", "raw", "aug_test.csv"),
    processed_dir: str = os.path.join("data", "processed"),
    models_dir: str = os.path.join("models", "trained_models"),
    reports_dir: str = os.path.join("reports", "metrics"),
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Execute full preprocessing pipeline.
    """
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 3: DATA PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 60)

    # 1. Load Datasets
    train_raw = pd.read_csv(train_raw_path)
    test_raw = pd.read_csv(test_raw_path)

    print(f"[INFO] Raw Train shape: {train_raw.shape}")
    print(f"[INFO] Raw Test shape:  {test_raw.shape}")

    # 2. Duplicate Removal
    initial_train_len = len(train_raw)
    train_raw = train_raw.drop_duplicates().reset_index(drop=True)
    dups_removed = initial_train_len - len(train_raw)
    print(f"[INFO] Duplicate rows removed: {dups_removed}")

    # 3. Stratified Split on Raw Train
    train_data, val_data = stratified_train_val_split(train_raw, target_col="target", val_size=test_size, random_state=random_state)

    # Extract targets and tracking IDs
    y_train = train_data["target"].astype(int)
    y_val = val_data["target"].astype(int)

    train_ids = train_data["enrollee_id"].copy()
    val_ids = val_data["enrollee_id"].copy()
    test_ids = test_raw["enrollee_id"].copy()

    X_train_raw = train_data.drop(columns=["target", "enrollee_id"])
    X_val_raw = val_data.drop(columns=["target", "enrollee_id"])
    X_test_raw = test_raw.drop(columns=["enrollee_id"])

    print(f"[INFO] Stratified Split -> Train: {X_train_raw.shape[0]}, Val: {X_val_raw.shape[0]}")

    # 4. Preprocessing Fit & Transform
    preprocessor = CandidatePreprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train_raw)
    X_val_scaled = preprocessor.transform(X_val_raw)
    X_test_scaled = preprocessor.transform(X_test_raw)

    # Assertions for verification
    assert X_train_scaled.isnull().sum().sum() == 0, "Missing values remain in X_train!"
    assert X_val_scaled.isnull().sum().sum() == 0, "Missing values remain in X_val!"
    assert X_test_scaled.isnull().sum().sum() == 0, "Missing values remain in X_test!"
    assert list(X_train_scaled.columns) == list(X_val_scaled.columns) == list(X_test_scaled.columns), "Columns mismatch!"

    print(f"[SUCCESS] Processed {X_train_scaled.shape[1]} features with 0 remaining missing values.")

    # 5. Save Processed Datasets
    X_train_scaled.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    X_val_scaled.to_csv(os.path.join(processed_dir, "X_val.csv"), index=False)
    y_val.to_csv(os.path.join(processed_dir, "y_val.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)

    train_ids.to_csv(os.path.join(processed_dir, "train_enrollee_ids.csv"), index=False)
    val_ids.to_csv(os.path.join(processed_dir, "val_enrollee_ids.csv"), index=False)
    test_ids.to_csv(os.path.join(processed_dir, "test_enrollee_ids.csv"), index=False)

    print(f"[OK] Saved processed datasets to: {processed_dir}")

    # 6. Save Metadata & Config Artifacts
    metadata = {
        "feature_names": preprocessor.feature_names,
        "nominal_categories": preprocessor.nominal_categories,
        "scaler_mean": preprocessor.scaler.mean_.to_dict(),
        "scaler_scale": preprocessor.scaler.scale_.to_dict(),
        "ordinal_mappings": ORDINAL_MAPPINGS
    }
    with open(os.path.join(models_dir, "preprocessor_config.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Saved preprocessor configuration artifact to: {models_dir}")

    # 7. Save Reports
    missing_summary = train_raw.isnull().sum()
    missing_summary_df = pd.DataFrame({
        "Feature": missing_summary.index,
        "Missing Count": missing_summary.values,
        "Missing Percentage (%)": (missing_summary.values / len(train_raw) * 100).round(2)
    })
    missing_summary_df.to_csv(os.path.join(reports_dir, "missing_value_summary.csv"), index=False)

    feature_summary = pd.DataFrame({
        "Feature_Index": range(len(preprocessor.feature_names)),
        "Feature_Name": preprocessor.feature_names
    })
    feature_summary.to_csv(os.path.join(reports_dir, "feature_list_after_preprocessing.csv"), index=False)

    print(f"[OK] Saved reports to: {reports_dir}")
    print("=" * 60)

    return {
        "X_train_shape": X_train_scaled.shape,
        "X_val_shape": X_val_scaled.shape,
        "X_test_shape": X_test_scaled.shape,
        "feature_count": len(preprocessor.feature_names),
        "features": preprocessor.feature_names
    }


if __name__ == "__main__":
    run_preprocessing_pipeline()
