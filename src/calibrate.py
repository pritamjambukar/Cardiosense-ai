"""Model Calibration Assessment & Reliability Curve Generation.

This module evaluates the clinical probability calibration of the final XGBoost model:
1. Calculates the Brier Score Loss and Brier Skill Score (BSS) against a clinical baseline.
2. Computes the Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
3. Evaluates calibration on both the held-out test split (N=61) and 5-fold CV out-of-fold predictions (N=242).
4. Generates publication-grade reliability diagrams and probability histograms saved to evaluation_plots/calibration_curve.png.
"""

from pathlib import Path
import sys
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# Ensure project root is available for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train import load_dataset, split_data


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 5,
    strategy: str = "uniform",
) -> Tuple[float, float]:
    """Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    ECE = sum(|B_b| / N * |acc(B_b) - conf(B_b)|)
    MCE = max(|acc(B_b) - conf(B_b)|)
    """
    if strategy == "uniform":
        bins = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
        bins[0] = 0.0
        bins[-1] = 1.0
    else:
        raise ValueError(f"Unknown binning strategy: {strategy}")

    bin_assignments = np.digitize(y_prob, bins) - 1
    # Clip to valid range [0, n_bins - 1]
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    ece = 0.0
    mce = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = bin_assignments == i
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            diff = abs(bin_acc - bin_conf)
            ece += (bin_count / n) * diff
            mce = max(mce, diff)

    return ece, mce


def evaluate_calibration(
    model_path: str | Path = "models/best_model.joblib",
    output_plot_path: str | Path = "evaluation_plots/calibration_curve.png",
) -> Dict[str, float]:
    """Evaluate probability calibration on test and cross-validation sets and save plots."""
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = joblib.load(model_path)

    # 1. Test set probabilities
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]
    test_brier = brier_score_loss(y_test, y_test_prob)

    # Clinical reference baseline: predicting the training prevalence for all patients
    base_rate = float(y_train.mean())
    baseline_brier = brier_score_loss(y_test, [base_rate] * len(y_test))
    test_bss = 1.0 - (test_brier / baseline_brier)

    test_ece, test_mce = compute_expected_calibration_error(y_test.values, y_test_prob, n_bins=5)

    # 2. 5-Fold Cross-Validation Out-of-Fold Probabilities (N=242 for higher statistical power)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_prob = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    cv_brier = brier_score_loss(y_train, cv_prob)
    cv_ece, cv_mce = compute_expected_calibration_error(y_train.values, cv_prob, n_bins=8)

    # 3. Calibration Curve Points
    test_prob_true, test_prob_pred = calibration_curve(y_test, y_test_prob, n_bins=5, strategy="uniform")
    cv_prob_true, cv_prob_pred = calibration_curve(y_train, cv_prob, n_bins=8, strategy="uniform")

    # 4. Generate Publication-Quality Dual-Panel Plot
    fig, (ax_curve, ax_hist) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(7.5, 7.0),
        gridspec_kw={"height_ratios": [3.0, 1.2]},
        sharex=True,
    )
    plt.subplots_adjust(hspace=0.08)

    # Stylistic setup
    bg_color = "#FFFFFF"
    fig.patch.set_facecolor(bg_color)
    ax_curve.set_facecolor(bg_color)
    ax_hist.set_facecolor(bg_color)

    # --- TOP PANEL: RELIABILITY CURVE ---
    # Perfect calibration diagonal
    ax_curve.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="#888888",
        linewidth=1.2,
        label="Perfect Calibration (Ideal)",
    )

    # 5-Fold Cross-Validation curve (N=242)
    ax_curve.plot(
        cv_prob_pred,
        cv_prob_true,
        marker="s",
        markersize=5,
        color="#2563EB",
        linewidth=1.6,
        alpha=0.85,
        label=f"5-Fold CV Out-of-Fold (N=242, Brier={cv_brier:.3f}, ECE={cv_ece:.1%})",
    )

    # Held-out Test Set curve (N=61)
    ax_curve.plot(
        test_prob_pred,
        test_prob_true,
        marker="o",
        markersize=6,
        color="#881337",
        linewidth=2.0,
        label=f"Held-Out Test Set (N=61, Brier={test_brier:.3f}, ECE={test_ece:.1%})",
    )

    ax_curve.set_ylabel("Observed Fraction of Positives", fontsize=9, fontweight="bold", color="#111111", labelpad=8)
    ax_curve.set_ylim(-0.02, 1.02)
    ax_curve.set_xlim(-0.02, 1.02)
    ax_curve.set_title(
        "CardioSense AI — Clinical Probability Calibration Diagram\n"
        f"Test Brier Loss: {test_brier:.4f} (Skill Score: +{test_bss:.1%} vs Prevalence Baseline)",
        fontsize=10,
        fontweight="bold",
        color="#111111",
        pad=14,
    )

    ax_curve.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="#E5E5E5", fontsize=8)
    ax_curve.grid(True, linestyle=":", alpha=0.5, color="#D1D5DB")
    ax_curve.tick_params(colors="#4B5563", labelsize=8)

    # Clinical Risk Tiers annotations
    ax_curve.axvspan(0.0, 0.35, alpha=0.04, color="#1E4D38", label="_LowRisk")
    ax_curve.axvspan(0.35, 0.65, alpha=0.04, color="#B45309", label="_MedRisk")
    ax_curve.axvspan(0.65, 1.0, alpha=0.04, color="#881337", label="_HighRisk")

    ax_curve.text(0.17, 0.05, "Low Risk (<35%)", color="#1E4D38", fontsize=7.5, ha="center", style="italic")
    ax_curve.text(0.50, 0.05, "Intermediate (35-65%)", color="#B45309", fontsize=7.5, ha="center", style="italic")
    ax_curve.text(0.82, 0.05, "Elevated (≥65%)", color="#881337", fontsize=7.5, ha="center", style="italic")

    # --- BOTTOM PANEL: PREDICTED PROBABILITY HISTOGRAM ---
    bins_hist = np.linspace(0, 1, 21)
    ax_hist.hist(
        y_test_prob,
        bins=bins_hist,
        color="#881337",
        alpha=0.6,
        edgecolor="#881337",
        linewidth=0.8,
        label="Test Set Density",
    )
    ax_hist.set_xlabel("Mean Predicted Probability", fontsize=9, fontweight="bold", color="#111111", labelpad=8)
    ax_hist.set_ylabel("Patient Count", fontsize=9, fontweight="bold", color="#111111", labelpad=8)
    ax_hist.set_xlim(-0.02, 1.02)
    ax_hist.grid(True, linestyle=":", alpha=0.5, color="#D1D5DB")
    ax_hist.tick_params(colors="#4B5563", labelsize=8)
    ax_hist.legend(loc="upper center", frameon=False, fontsize=8)

    out_file = Path(output_plot_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    metrics = {
        "test_brier": float(test_brier),
        "baseline_brier": float(baseline_brier),
        "test_bss": float(test_bss),
        "test_ece": float(test_ece),
        "test_mce": float(test_mce),
        "cv_brier": float(cv_brier),
        "cv_ece": float(cv_ece),
        "cv_mce": float(cv_mce),
    }

    return metrics


if __name__ == "__main__":
    print("Evaluating probability calibration of models/best_model.joblib...")
    m = evaluate_calibration()
    print("-" * 55)
    print(f"Held-Out Test Brier Score:       {m['test_brier']:.4f}")
    print(f"Prevalence Baseline Brier Score: {m['baseline_brier']:.4f}")
    print(f"Brier Skill Score (BSS):         {m['test_bss']:.2%}")
    print(f"Test Expected Calibration Error: {m['test_ece']:.2%}")
    print(f"Test Maximum Calibration Error:  {m['test_mce']:.2%}")
    print(f"5-Fold CV Out-of-Fold Brier:     {m['cv_brier']:.4f}")
    print(f"5-Fold CV Calibration Error:     {m['cv_ece']:.2%}")
    print("-" * 55)
    print("Calibration diagram saved to evaluation_plots/calibration_curve.png")

