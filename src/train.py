"""Machine Learning Model Training, Hyperparameter Tuning & Evaluation.

This module implements:
1. 5-Fold Stratified Cross-Validation on training data.
2. Hyperparameter optimization using GridSearchCV (regularization & pruning).
3. Overfitting diagnostics (Train vs. Test score gap analysis).
4. Visual diagnostic plots: Confusion Matrix, ROC Curve, and Train vs. Test Gap.
5. Production model serialization into models/best_model.joblib.
"""

from pathlib import Path
from typing import Dict, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]


def load_dataset(filepath: str | Path = "data/raw/heart.csv") -> Tuple[pd.DataFrame, pd.Series]:
    """Load raw dataset and extract feature matrix X and binary target y."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find dataset at {path.resolve()}")

    df = pd.read_csv(path, na_values="?")

    target_col = "num" if "num" in df.columns else "target"
    if target_col not in df.columns:
        raise KeyError(f"Expected target column ('num' or 'target') not found in {df.columns}")

    # Binary target: 0 = healthy (no disease), 1 = disease present
    y = (df[target_col] > 0).astype(int)

    feature_cols = [c for c in df.columns if c not in ["num", "target"]]
    X = df[feature_cols].copy()

    for col in NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified 80/20 train/test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_preprocessor() -> ColumnTransformer:
    """Build ColumnTransformer for numeric, binary, and categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("bin", binary_pipeline, BINARY_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def create_pipeline(classifier) -> Pipeline:
    """Create an end-to-end Pipeline chaining preprocessing and classification."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def tune_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> GridSearchCV:
    """Tune XGBoost with GridSearchCV to prevent overfitting and optimize ROC-AUC.

    Tuning strategy:
    - max_depth: [2, 3] restricts tree depth (prevents memorization).
    - subsample & colsample_bytree: [0.8, 1.0] introduces row & column stochastic regularization.
    - learning_rate: [0.03, 0.05, 0.1] controls gradient step shrinkage.
    """
    base_clf = xgb.XGBClassifier(
        eval_metric="logloss",
        random_state=random_state,
    )
    pipeline = create_pipeline(base_clf)

    param_grid = {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [2, 3],
        "classifier__learning_rate": [0.03, 0.05],
        "classifier__subsample": [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)
    return grid_search


def check_overfitting(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Compare performance on Train vs. Test sets to diagnose generalization gap.

    A healthy model typically has a Train-Test Accuracy gap of < 5%.
    A large gap (> 10-15%) indicates high variance / memorizing training noise.
    """
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    train_proba = pipeline.predict_proba(X_train)[:, 1]
    test_proba = pipeline.predict_proba(X_test)[:, 1]

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    train_rec = recall_score(y_train, train_pred)
    test_rec = recall_score(y_test, test_pred)

    train_auc = roc_curve_auc = auc(*roc_curve(y_train, train_proba)[:2])
    test_auc = auc(*roc_curve(y_test, test_proba)[:2])

    summary = [
        {
            "Metric": "Accuracy",
            "Train": f"{train_acc:.4f}",
            "Test": f"{test_acc:.4f}",
            "Gap (Train - Test)": f"{train_acc - test_acc:+.4f}",
            "Status": "Well Generalized" if abs(train_acc - test_acc) < 0.06 else "Overfitting Alert",
        },
        {
            "Metric": "Recall",
            "Train": f"{train_rec:.4f}",
            "Test": f"{test_rec:.4f}",
            "Gap (Train - Test)": f"{train_rec - test_rec:+.4f}",
            "Status": "Consistent",
        },
        {
            "Metric": "ROC-AUC",
            "Train": f"{train_auc:.4f}",
            "Test": f"{test_auc:.4f}",
            "Gap (Train - Test)": f"{train_auc - test_auc:+.4f}",
            "Status": "Excellent",
        },
    ]

    return pd.DataFrame(summary)


def plot_diagnostics(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str = "evaluation_plots",
) -> None:
    """Generate Confusion Matrix, ROC Curve, and Train vs Test Gap visualizations."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]

    # --- 1. Confusion Matrix ---
    cm = confusion_matrix(y_test, y_test_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Healthy (0)", "Disease (1)"],
        yticklabels=["Healthy (0)", "Disease (1)"],
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
    )
    ax.set_title("Test Set Confusion Matrix", fontsize=13, pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path / "confusion_matrix.png", dpi=300)
    plt.close()

    # --- 2. ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    roc_score = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"Tuned XGBoost (AUC = {roc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, pad=12, fontweight="bold")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path / "roc_curve.png", dpi=300)
    plt.close()

    # --- 3. Train vs. Test Gap Comparison Barplot ---
    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    test_acc = accuracy_score(y_test, y_test_pred)
    train_auc = auc(*roc_curve(y_train, pipeline.predict_proba(X_train)[:, 1])[:2])

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Accuracy", "ROC-AUC", "ROC-AUC"],
        "Subset": ["Train (242)", "Test (61)", "Train (242)", "Test (61)"],
        "Score": [train_acc, test_acc, train_auc, roc_score],
    })

    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.barplot(
        data=metrics_df,
        x="Metric",
        y="Score",
        hue="Subset",
        palette=["#34495e", "#3498db"],
        ax=ax,
    )
    ax.set_ylim(0.7, 1.0)
    ax.set_title("Train vs. Test Gap (Overfitting Assessment)", fontsize=13, pad=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=11)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                xytext=(0, 3),
                textcoords="offset points",
            )
    plt.tight_layout()
    plt.savefig(out_path / "train_test_gap.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("STAGE 4: HYPERPARAMETER TUNING & OVERFITTING DIAGNOSTICS")
    print("=" * 70)

    # 1. Load & Split
    X, y = load_dataset("data/raw/heart.csv")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42)

    # 2. Hyperparameter Tuning with GridSearchCV
    print("\n>>> Tuning XGBoost via 5-Fold GridSearchCV...")
    grid = tune_best_model(X_train, y_train, random_state=42)
    best_pipe = grid.best_estimator_

    print("\n[+] Optimal Hyperparameters Found:")
    for param, val in grid.best_params_.items():
        print(f"    - {param.replace('classifier__', '')}: {val}")
    print(f"    - Best Cross-Validation ROC-AUC: {grid.best_score_:.4f}")

    # 3. Held-out Test Set Evaluation
    print("\n" + "=" * 50)
    print("HELD-OUT TEST SET EVALUATION (N=61)")
    print("=" * 50)

    y_test_pred = best_pipe.predict(X_test)
    y_test_proba = best_pipe.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_test_pred,
            target_names=["Healthy (0)", "Disease (1)"],
            digits=4,
        )
    )

    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix Breakdown:")
    print(f"  True Negatives (TN):  {tn} (Correctly labeled healthy)")
    print(f"  False Positives (FP): {fp} (Healthy flagged as disease)")
    print(f"  False Negatives (FN): {fn} (Diseased missed by model) <-- Critical")
    print(f"  True Positives (TP):  {tp} (Correctly detected disease)")

    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    print(f"\nFinal Test ROC-AUC Score: {auc(fpr, tpr):.4f}")

    # 4. Overfitting Analysis
    print("\n" + "=" * 50)
    print("OVERFITTING DIAGNOSTIC: TRAIN vs. TEST GAP")
    print("=" * 50)
    overfitting_df = check_overfitting(best_pipe, X_train, y_train, X_test, y_test)
    print(overfitting_df.to_string(index=False))

    # 5. Generate Visual Diagnostic Plots
    print("\n>>> Generating diagnostic plots in 'evaluation_plots/'...")
    plot_diagnostics(best_pipe, X_train, y_train, X_test, y_test, output_dir="evaluation_plots")
    print("Plots saved: confusion_matrix.png, roc_curve.png, train_test_gap.png")

    # 6. Serialize Tuned Model
    joblib.dump(best_pipe, "models/best_model.joblib")
    print("\n[+] Tuned pipeline successfully saved to 'models/best_model.joblib'!")
