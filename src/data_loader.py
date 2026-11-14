"""Data Loading, Preprocessing, and Exploratory Data Analysis module.

This module handles:
1. Loading the UCI Heart Disease dataset (handling missing values encoded as '?').
2. Collapsing multi-class target (0-4) into binary classification (0: No Disease, 1: Disease).
3. Clinically-grounded imputation (Median for discrete vessel counts, Mode for categorical defect).
4. Feature encoding for machine learning pipelines.
5. Generating foundational EDA visualizations.
"""

from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Standard column names for the 14-attribute UCI Cleveland Heart Disease dataset
FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]


def load_raw_data(filepath: str | Path = "data/raw/heart.csv") -> pd.DataFrame:
    """Load raw dataset and convert missing markers ('?') into standard NaN values.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        pd.DataFrame with '?' converted to np.nan and numeric columns cast.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path.resolve()}. "
            "Please ensure heart.csv exists in data/raw/."
        )

    # na_values='?' ensures string placeholders are immediately parsed as NaN
    df = pd.read_csv(path, na_values="?")

    # Ensure standard column naming
    if list(df.columns) != FEATURE_NAMES and len(df.columns) == len(FEATURE_NAMES):
        df.columns = FEATURE_NAMES

    return df


def clean_and_impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle target binary collapsing and justify clinical imputation.

    Imputation Rationale:
    - 'ca' (number of major vessels 0-3): Discrete count. We use the median
      (0.0), preserving the discrete whole-number property without adding floating bias.
    - 'thal' (thalassemia condition): Categorical blood condition. We use the mode
      (3.0 = normal), the dominant clinical baseline.
    - 'target': Collapses original 0-4 values into binary (0: No disease, 1: Disease).

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame with no null values and a binary 'target' column.
    """
    df_clean = df.copy()

    # 1. Target Collapsing: 0 = healthy (0), 1-4 = presence of heart disease (1)
    if "num" in df_clean.columns:
        df_clean["target"] = (df_clean["num"] > 0).astype(int)
        df_clean = df_clean.drop(columns=["num"])
    elif "target" in df_clean.columns:
        df_clean["target"] = (df_clean["target"] > 0).astype(int)

    # 2. Convert object/string columns to numeric float before imputation
    for col in ["ca", "thal"]:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # 3. Clinical Imputation
    if "ca" in df_clean.columns and df_clean["ca"].isna().sum() > 0:
        ca_median = df_clean["ca"].median()
        df_clean["ca"] = df_clean["ca"].fillna(ca_median)

    if "thal" in df_clean.columns and df_clean["thal"].isna().sum() > 0:
        thal_mode = df_clean["thal"].mode()[0]
        df_clean["thal"] = df_clean["thal"].fillna(thal_mode)

    return df_clean


def encode_features(
    df: pd.DataFrame, drop_first: bool = False
) -> Tuple[pd.DataFrame, list]:
    """One-hot encode multi-category nominal features.

    Features like 'cp' (chest pain type) and 'restecg' are nominal categories;
    treating them as ordinal numbers would mislead distance/tree splits.

    Args:
        df: Cleaned DataFrame.
        drop_first: Whether to drop the first dummy level (recommended for linear models
                    to prevent multicollinearity; can keep False for tree models).

    Returns:
        Tuple of (encoded DataFrame, list of feature names without target).
    """
    df_encoded = df.copy()

    # Nominal categorical features requiring one-hot encoding
    nominal_cols = ["cp", "restecg", "slope", "thal"]
    cols_to_encode = [c for c in nominal_cols if c in df_encoded.columns]

    df_encoded = pd.get_dummies(
        df_encoded, columns=cols_to_encode, drop_first=drop_first, dtype=int
    )

    feature_cols = [c for c in df_encoded.columns if c != "target"]
    return df_encoded, feature_cols


def generate_eda_plots(df_clean: pd.DataFrame, output_dir: str = "eda_plots") -> None:
    """Generate and save essential EDA figures for student inspection.

    Figures generated:
    1. target_balance.png: Class balance barplot.
    2. feature_distributions.png: Histograms of key clinical markers.
    3. correlation_heatmap.png: Full correlation matrix with the target.

    Args:
        df_clean: Cleaned DataFrame (with 'target' and without missing values).
        output_dir: Directory where plots will be saved.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # --- Plot 1: Target Balance ---
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df_clean["target"].value_counts().sort_index()
    labels = ["No Disease (0)", "Disease (1)"]
    bars = ax.bar(labels, counts.values, color=["#2ecc71", "#e74c3c"], width=0.55)

    # Annotate bar percentages
    total = len(df_clean)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height} ({height/total*100:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
        )
    ax.set_title("Target Distribution (Heart Disease Presence)", fontsize=13, pad=12)
    ax.set_ylabel("Patient Count")
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig(out_path / "target_balance.png", dpi=300)
    plt.close()

    # --- Plot 2: Key Continuous Feature Distributions by Target ---
    key_continuous = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, feature in enumerate(key_continuous):
        sns.kdeplot(
            data=df_clean,
            x=feature,
            hue="target",
            common_norm=False,
            palette={0: "#2ecc71", 1: "#e74c3c"},
            fill=True,
            alpha=0.35,
            ax=axes[idx],
        )
        axes[idx].set_title(f"Distribution of {feature}", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel(feature)

    # 6th plot shows exercise angina vs target
    sns.countplot(
        data=df_clean,
        x="exang",
        hue="target",
        palette={0: "#2ecc71", 1: "#e74c3c"},
        ax=axes[5],
    )
    axes[5].set_title("Exercise Induced Angina vs Target", fontsize=11, fontweight="bold")
    axes[5].set_xticks([0, 1])
    axes[5].set_xticklabels(["No (0)", "Yes (1)"])

    plt.suptitle("Clinical Feature Distributions Stratified by Target", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path / "feature_distributions.png", dpi=300)
    plt.close()

    # --- Plot 3: Correlation Heatmap ---
    plt.figure(figsize=(11, 9))
    corr = df_clean.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.5,
        vmax=0.5,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Matrix of Clinical Indicators", fontsize=14, pad=12)
    plt.tight_layout()
    plt.savefig(out_path / "correlation_heatmap.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    print("--- Stage 2: Data Pipeline & EDA Run ---")
    raw_df = load_raw_data("data/raw/heart.csv")
    print(f"\n1. Raw Shape: {raw_df.shape}")
    print("\n2. Data Types & Non-Null Counts:")
    print(raw_df.info())

    print("\n3. Missing Values Detected:")
    missing_counts = raw_df.isna().sum()
    print(missing_counts[missing_counts > 0])

    cleaned_df = clean_and_impute_data(raw_df)
    print("\n4. Cleaned & Imputed Dataset:")
    print(f"Shape: {cleaned_df.shape}")
    print(f"Remaining Missing Values: {cleaned_df.isna().sum().sum()}")
    print("\nTarget Value Counts:")
    print(cleaned_df["target"].value_counts())

    encoded_df, features = encode_features(cleaned_df)
    print(f"\n5. One-Hot Encoded Shape: {encoded_df.shape}")
    print(f"Total Features for ML: {len(features)}")

    print("\n6. Generating EDA plots in 'eda_plots/'...")
    generate_eda_plots(cleaned_df, output_dir="eda_plots")
    print("Plots saved successfully!")
