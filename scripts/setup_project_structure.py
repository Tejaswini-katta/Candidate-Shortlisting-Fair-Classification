"""
Project Structure Setup Script.
Creates necessary project subdirectories and copies raw data files to data/raw/.
"""

import os
import shutil


def setup_directories():
    """Create project directories if they do not exist."""
    directories = [
        os.path.join("data", "raw"),
        os.path.join("data", "processed"),
        "notebooks",
        "src",
        "models",
        "reports"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"[OK] Directory verified/created: {directory}")


def copy_raw_data():
    """Copy root raw CSV files to data/raw directory."""
    raw_dir = os.path.join("data", "raw")

    for file_name in ["aug_train.csv", "aug_test.csv"]:
        src_path = file_name
        dest_path = os.path.join(raw_dir, file_name)

        if os.path.exists(src_path) and not os.path.exists(dest_path):
            shutil.copy(src_path, dest_path)
            print(f"[OK] Copied {src_path} -> {dest_path}")
        elif os.path.exists(dest_path):
            print(f"[OK] Raw file already present: {dest_path}")
        else:
            print(f"[WARNING] Source file missing: {src_path}")


if __name__ == "__main__":
    setup_directories()
    copy_raw_data()
