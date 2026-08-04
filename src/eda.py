"""
Exploratory Data Analysis (EDA) Module for Candidate Shortlisting.

Provides reusable visual and statistical analysis functions for candidate features,
demographic distributions, and target variable relationships. Saves output figures
to reports/figures/.
"""

import os
from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def set_plot_style(palette: str = "viridis", style: str = "whitegrid") -> None:
    """Set global visual style and aesthetics for matplotlib/seaborn plots."""
    sns.set_theme(style=style)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 1.0


def _ensure_dir(file_path: Optional[str]) -> None:
    """Helper function to create parent directories for figure saving."""
    if file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)


def plot_univariate_categorical(
    df: pd.DataFrame,
    column: str,
    title: str,
    save_path: Optional[str] = None,
    order: Optional[List[str]] = None
) -> None:
    """
    Generate and display a count plot for a categorical feature.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset dataframe.
    column : str
        Categorical feature column name.
    title : str
        Plot title.
    save_path : Optional[str]
        File path to save the generated figure.
    order : Optional[List[str]]
        Specific ordering for category categories.
    """
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(
        data=df,
        x=column,
        hue=column,
        order=order if order else df[column].value_counts().index,
        palette="crest",
        legend=False
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel(column.replace("_", " ").title(), fontsize=12)
    plt.ylabel("Candidate Count", fontsize=12)
    plt.xticks(rotation=30, ha="right")

    # Annotate percentages on top of bars
    total = len(df[column].dropna())
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            percentage = f"{(height / total) * 100:.1f}%"
            ax.annotate(
                f"{int(height)}\n({percentage})",
                (p.get_x() + p.get_width() / 2., height),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points"
            )

    plt.tight_layout()
    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_univariate_numerical(
    df: pd.DataFrame,
    column: str,
    title: str,
    save_path: Optional[str] = None
) -> None:
    """
    Generate distribution histogram and box plot side-by-side for a numerical feature.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset dataframe.
    column : str
        Numerical feature column name.
    title : str
        Plot title.
    save_path : Optional[str]
        File path to save figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram + KDE
    sns.histplot(data=df, x=column, kde=True, ax=axes[0], color="#2b5c8f")
    axes[0].set_title(f"{title} - Distribution (KDE)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(column.replace("_", " ").title(), fontsize=11)
    axes[0].set_ylabel("Density / Count", fontsize=11)

    # Boxplot for outlier check
    sns.boxplot(data=df, x=column, ax=axes[1], color="#e76f51")
    axes[1].set_title(f"{title} - Box Plot (Outlier Check)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel(column.replace("_", " ").title(), fontsize=11)

    plt.tight_layout()
    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_target_vs_categorical(
    df: pd.DataFrame,
    column: str,
    target_col: str = "target",
    title: str = "",
    save_path: Optional[str] = None
) -> None:
    """
    Plot normalized proportion of target variable across categories of a feature.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset dataframe.
    column : str
        Categorical feature name.
    target_col : str
        Target column name (default 'target').
    title : str
        Plot title.
    save_path : Optional[str]
        File path to save figure.
    """
    plt.figure(figsize=(10, 5))

    # Calculate proportion of target=1 vs target=0
    prop_df = df.groupby(column)[target_col].value_counts(normalize=True).unstack().fillna(0)
    ax = prop_df.plot(kind="bar", stacked=True, figsize=(10, 5), color=["#2a9d8f", "#e76f51"])

    plt.title(title if title else f"Target Rate by {column.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    plt.xlabel(column.replace("_", " ").title(), fontsize=12)
    plt.ylabel("Proportion of Candidates", fontsize=12)
    plt.legend(["Not Looking (0)", "Looking for Change (1)"], title="Target Status", loc="upper right")
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_target_vs_numerical(
    df: pd.DataFrame,
    column: str,
    target_col: str = "target",
    title: str = "",
    save_path: Optional[str] = None
) -> None:
    """
    Plot violin plot comparing distribution of a numerical feature across target classes.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset dataframe.
    column : str
        Numerical feature name.
    target_col : str
        Target column name.
    title : str
        Plot title.
    save_path : Optional[str]
        File path to save figure.
    """
    plt.figure(figsize=(9, 5))
    sns.violinplot(
        data=df,
        x=target_col,
        y=column,
        hue=target_col,
        palette=["#2a9d8f", "#e76f51"],
        inner="quartile",
        legend=False
    )
    plt.title(title if title else f"{column.replace('_', ' ').title()} by Target Status", fontsize=13, fontweight="bold")
    plt.xlabel("Candidate Target Status (0 = Retained, 1 = Looking for Change)", fontsize=11)
    plt.ylabel(column.replace("_", " ").title(), fontsize=11)

    plt.tight_layout()
    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    numerical_cols: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Generate correlation matrix heatmap for numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset dataframe.
    numerical_cols : List[str]
        List of numerical feature names.
    save_path : Optional[str]
        File path to save figure.
    """
    plt.figure(figsize=(8, 6))
    corr = df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="vlag", vmin=-1, vmax=1, linewidths=0.5)
    plt.title("Correlation Matrix (Numerical Candidate Features)", fontsize=13, fontweight="bold")

    plt.tight_layout()
    _ensure_dir(save_path)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()
