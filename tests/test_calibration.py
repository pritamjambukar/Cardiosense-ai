"""Unit tests for Probability Calibration and Brier Score evaluation."""

import numpy as np
import pytest
from src.calibrate import compute_expected_calibration_error, evaluate_calibration


def test_ece_computation_perfect_calibration():
    """When probabilities match true labels perfectly, ECE must be 0."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    ece, mce = compute_expected_calibration_error(y_true, y_prob, n_bins=2)
    assert ece == pytest.approx(0.0, abs=1e-5)
    assert mce == pytest.approx(0.0, abs=1e-5)


def test_model_calibration_metrics():
    """Verify that the trained model delivers strong clinical calibration:
    1. Brier score must be substantially lower than prevalence baseline (< 0.15 vs 0.248).
    2. Brier Skill Score (BSS) must show at least 40% improvement.
    3. Expected Calibration Error (ECE) must remain within clinically acceptable limits (< 15%).
    """
    metrics = evaluate_calibration()

    assert metrics["test_brier"] < 0.15, f"Brier score {metrics['test_brier']} too high"
    assert metrics["test_bss"] > 0.40, f"Brier Skill Score {metrics['test_bss']:.2%} below 40%"
    assert metrics["cv_ece"] < 0.12, f"CV Expected Calibration Error {metrics['cv_ece']:.2%} exceeds 12%"

