"""
Data Loader Module for Candidate Shortlisting and Fair Classification.

This module provides reusable, production-ready functions to locate,
load, and perform initial structural inspections on raw training and
testing datasets while maintaining candidate tracking IDs (enrollee_id).
"""

import os
from typing import Tuple, Dict, Any
import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset into a pandas DataFrame with error handling.

    Parameters
    ----------
    file_path : str
        The relative or absolute file path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded pandas DataFrame.

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    ValueError
        If the loaded file is empty or corrupted.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at path: {file_path}")

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError(f"Dataset at {file_path} is empty.")
        return df
    except Exception as exc:
        raise RuntimeError(f"Error loading CSV file from '{file_path}': {str(exc)}") from exc


def load_raw_data(
    train_path: str = os.path.join("data", "raw", "aug_train.csv"),
    test_path: str = os.path.join("data", "raw", "aug_test.csv"),
    fallback_dir: str = "."
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both raw training and test datasets.
    Includes a fallback check to the root directory if raw data directory is not yet populated.

    Parameters
    ----------
    train_path : str
        Path to the training CSV file.
    test_path : str
        Path to the testing CSV file.
    fallback_dir : str
        Fallback directory to search if primary paths are missing.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing (train_df, test_df).
    """
    # Resolve train path
    if not os.path.exists(train_path):
        fallback_train = os.path.join(fallback_dir, "aug_train.csv")
        if os.path.exists(fallback_train):
            train_path = fallback_train

    # Resolve test path
    if not os.path.exists(test_path):
        fallback_test = os.path.join(fallback_dir, "aug_test.csv")
        if os.path.exists(fallback_test):
            test_path = fallback_test

    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)

    return train_df, test_df


def inspect_dataset_structure(df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict[str, Any]:
    """
    Inspect the shape, columns, missing values, duplicates, and enrollee ID presence.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to inspect.
    dataset_name : str
        Name tag for logging and output summary.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing summary statistics.
    """
    has_enrollee_id = "enrollee_id" in df.columns
    total_rows, total_cols = df.shape
    duplicate_rows = df.duplicated().sum()
    duplicate_enrollees = df["enrollee_id"].duplicated().sum() if has_enrollee_id else 0

    missing_summary = df.isnull().sum()
    missing_percentage = (missing_summary / total_rows) * 100

    missing_stats = pd.DataFrame({
        "Missing Values": missing_summary,
        "Percentage (%)": missing_percentage.round(2)
    })
    missing_stats = missing_stats[missing_stats["Missing Values"] > 0]

    summary_info = {
        "dataset_name": dataset_name,
        "total_rows": total_rows,
        "total_cols": total_cols,
        "duplicate_rows": duplicate_rows,
        "duplicate_enrollee_ids": duplicate_enrollees,
        "has_enrollee_id": has_enrollee_id,
        "missing_summary": missing_stats,
        "column_types": df.dtypes.to_dict()
    }

    return summary_info
