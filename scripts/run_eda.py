"""
Script to execute complete EDA and save figures to reports/figures/.
"""

import os
import pandas as pd
from src.data_loader import load_raw_data
from src.eda import (
    set_plot_style,
    plot_univariate_categorical,
    plot_univariate_numerical,
    plot_target_vs_categorical,
    plot_target_vs_numerical,
    plot_correlation_heatmap
)


def run_eda_pipeline():
    print("=" * 60)
    print("RUNNING COMPLETE EDA & GENERATING FIGURES FOR REPORTS")
    print("=" * 60)

    # Set aesthetics
    set_plot_style()

    # Load raw data
    train_df, _ = load_raw_data()

    # Figures destination directory
    fig_dir = os.path.join("reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Target Distribution
    plot_univariate_categorical(
        train_df,
        column="target",
        title="Target Variable Distribution (0: Retained, 1: Looking for Change)",
        save_path=os.path.join(fig_dir, "01_target_distribution.png")
    )
    print("[OK] Saved 01_target_distribution.png")

    # 2. Protected Attribute: Gender Distribution
    plot_univariate_categorical(
        train_df,
        column="gender",
        title="Protected Attribute: Gender Distribution",
        save_path=os.path.join(fig_dir, "02_gender_distribution.png")
    )
    print("[OK] Saved 02_gender_distribution.png")

    # 3. Categorical Distributions
    cat_cols = [
        ("education_level", "03_education_level_distribution.png", "Education Level Distribution"),
        ("relevent_experience", "04_relevant_experience_distribution.png", "Relevant Experience Distribution"),
        ("company_size", "05_company_size_distribution.png", "Company Size Distribution"),
        ("company_type", "06_company_type_distribution.png", "Company Type Distribution"),
        ("major_discipline", "07_major_discipline_distribution.png", "Major Discipline Distribution"),
    ]

    for col, filename, title in cat_cols:
        plot_univariate_categorical(
            train_df,
            column=col,
            title=title,
            save_path=os.path.join(fig_dir, filename)
        )
        print(f"[OK] Saved {filename}")

    # 4. Numerical Distributions
    plot_univariate_numerical(
        train_df,
        column="city_development_index",
        title="City Development Index (CDI)",
        save_path=os.path.join(fig_dir, "08_cdi_distribution.png")
    )
    print("[OK] Saved 08_cdi_distribution.png")

    plot_univariate_numerical(
        train_df,
        column="training_hours",
        title="Training Hours Completed",
        save_path=os.path.join(fig_dir, "09_training_hours_distribution.png")
    )
    print("[OK] Saved 09_training_hours_distribution.png")

    # 5. Relationship Analysis vs Target
    plot_target_vs_categorical(
        train_df,
        column="gender",
        title="Target Rate vs Gender (Protected Attribute)",
        save_path=os.path.join(fig_dir, "10_target_vs_gender.png")
    )
    print("[OK] Saved 10_target_vs_gender.png")

    plot_target_vs_categorical(
        train_df,
        column="education_level",
        title="Target Rate vs Education Level",
        save_path=os.path.join(fig_dir, "11_target_vs_education.png")
    )
    print("[OK] Saved 11_target_vs_education.png")

    plot_target_vs_categorical(
        train_df,
        column="relevent_experience",
        title="Target Rate vs Relevant Experience",
        save_path=os.path.join(fig_dir, "12_target_vs_relevant_experience.png")
    )
    print("[OK] Saved 12_target_vs_relevant_experience.png")

    plot_target_vs_categorical(
        train_df,
        column="company_size",
        title="Target Rate vs Company Size",
        save_path=os.path.join(fig_dir, "13_target_vs_company_size.png")
    )
    print("[OK] Saved 13_target_vs_company_size.png")

    # Numerical vs Target (Violin plots)
    plot_target_vs_numerical(
        train_df,
        column="city_development_index",
        title="City Development Index across Target Status",
        save_path=os.path.join(fig_dir, "14_target_vs_cdi.png")
    )
    print("[OK] Saved 14_target_vs_cdi.png")

    plot_target_vs_numerical(
        train_df,
        column="training_hours",
        title="Training Hours across Target Status",
        save_path=os.path.join(fig_dir, "15_target_vs_training_hours.png")
    )
    print("[OK] Saved 15_target_vs_training_hours.png")

    # Correlation matrix for existing numeric features
    num_cols = ["city_development_index", "training_hours", "target"]
    plot_correlation_heatmap(
        train_df,
        numerical_cols=num_cols,
        save_path=os.path.join(fig_dir, "16_correlation_heatmap.png")
    )
    print("[OK] Saved 16_correlation_heatmap.png")

    print("=" * 60)
    print(f"[SUCCESS] EDA Figures saved in: {fig_dir}")
    print("=" * 60)


if __name__ == "__main__":
    run_eda_pipeline()
