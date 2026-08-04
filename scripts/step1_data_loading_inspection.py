"""
Step 1: Data Loading and Structural Inspection Script.

This script executes the data loading module, verifies the directory setup,
and performs structural inspection on raw training and testing datasets.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from setup_project_structure import setup_directories, copy_raw_data
except ImportError:
    from scripts.setup_project_structure import setup_directories, copy_raw_data

from src.data_loader import load_raw_data, inspect_dataset_structure


def run_step_1():
    """Execute Step 1 workflow."""
    print("=" * 60)
    print("STEP 1: PROJECT SETUP & DATA LOADING INSPECTION")
    print("=" * 60)

    # 1. Setup project directories & copy data
    setup_directories()
    copy_raw_data()

    # 2. Load raw datasets
    train_df, test_df = load_raw_data()

    print(f"\n[INFO] Successfully loaded Training Data: {train_df.shape[0]} rows, {train_df.shape[1]} columns")
    print(f"[INFO] Successfully loaded Test Data:     {test_df.shape[0]} rows, {test_df.shape[1]} columns")

    # 3. Inspect Train dataset
    train_summary = inspect_dataset_structure(train_df, dataset_name="Training Dataset (aug_train.csv)")
    print("\n" + "-" * 50)
    print(f"Summary for {train_summary['dataset_name']}:")
    print("-" * 50)
    print(f"Total Records (Rows):       {train_summary['total_rows']}")
    print(f"Total Features (Columns):  {train_summary['total_cols']}")
    print(f"Duplicate Rows:            {train_summary['duplicate_rows']}")
    print(f"Duplicate Candidate IDs:   {train_summary['duplicate_enrollee_ids']}")
    print(f"Enrollee ID Present:       {train_summary['has_enrollee_id']}")

    print("\nMissing Values Summary (Train):")
    if not train_summary['missing_summary'].empty:
        print(train_summary['missing_summary'].to_string())
    else:
        print("No missing values found.")

    # 4. Inspect Test dataset
    test_summary = inspect_dataset_structure(test_df, dataset_name="Testing Dataset (aug_test.csv)")
    print("\n" + "-" * 50)
    print(f"Summary for {test_summary['dataset_name']}:")
    print("-" * 50)
    print(f"Total Records (Rows):       {test_summary['total_rows']}")
    print(f"Total Features (Columns):  {test_summary['total_cols']}")
    print(f"Duplicate Rows:            {test_summary['duplicate_rows']}")
    print(f"Duplicate Candidate IDs:   {test_summary['duplicate_enrollee_ids']}")
    print(f"Enrollee ID Present:       {test_summary['has_enrollee_id']}")

    print("\nMissing Values Summary (Test):")
    if not test_summary['missing_summary'].empty:
        print(test_summary['missing_summary'].to_string())
    else:
        print("No missing values found.")

    print("\n" + "=" * 60)
    print("[SUCCESS] Step 1 Data Loading & Inspection Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_step_1()
