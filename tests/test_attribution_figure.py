"""Unit tests for SHAP Attribution Waterfall Figure generation.

Verifies:
1. build_attribution_figure returns a matplotlib.figure.Figure without raising exceptions.
2. Evaluates across 3 distinct patient profiles (including unusual categories).
3. Verifies title and axis labels are cleanly formatted without invalid CSS kwargs.
4. Verifies the returned figure contains the waterfall (axes present, labels containing '=').
5. Verifies no figures leak in memory after calling plt.close(fig).
6. Verifies direct support for precomputed shap.Explanation objects.
"""

import matplotlib.figure
import matplotlib.pyplot as plt
import pytest
from src.explain import build_attribution_figure, load_model_and_metadata

PROFILE_NORMAL = {
    "age": 45, "sex": 0, "cp": 2, "trestbps": 120, "chol": 190,
    "fbs": 0, "restecg": 0, "thalach": 175, "exang": 0, "oldpeak": 0.0,
    "slope": 1, "ca": 0, "thal": 3,
}

PROFILE_INTERMEDIATE = {
    "age": 58, "sex": 1, "cp": 3, "trestbps": 138, "chol": 245,
    "fbs": 0, "restecg": 1, "thalach": 142, "exang": 0, "oldpeak": 1.2,
    "slope": 2, "ca": 1, "thal": 3,
}

PROFILE_UNUSUAL_CATEGORIES = {
    "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286,
    "fbs": 1, "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 2.5,
    "slope": 3, "ca": 3, "thal": 6,  # Unusual categories: slope=3, thal=6 (fixed defect), restecg=2 (LVH)
}


@pytest.mark.parametrize(
    "profile_dict, label",
    [
        (PROFILE_NORMAL, "Normal Profile"),
        (PROFILE_INTERMEDIATE, "Intermediate Profile"),
        (PROFILE_UNUSUAL_CATEGORIES, "Unusual Categories Profile"),
    ],
)
def test_build_attribution_figure_profiles(profile_dict, label):
    """build_attribution_figure must generate a valid Figure without raising for diverse clinical profiles."""
    figns_before = set(plt.get_fignums())
    fig = build_attribution_figure(profile_dict, title="ATTRIBUTION VECTOR: TEST CASE")

    assert isinstance(fig, matplotlib.figure.Figure), f"Expected Figure object for {label}"
    assert len(fig.axes) > 0, f"Expected non-empty axes for {label}"

    ax = fig.axes[0]
    title_text = ax.get_title()
    assert "ATTRIBUTION VECTOR" in title_text

    # Verify waterfall plot structure: y-axis contains feature assignment labels with '='
    ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
    assert any("=" in text for text in ytick_labels), f"Expected waterfall labels with '=' in {ytick_labels}"

    # Verify memory management: figure is cleaned up and no leak occurs
    fig_num = fig.number
    assert fig_num in plt.get_fignums()
    plt.close(fig)
    assert fig_num not in plt.get_fignums()
    assert set(plt.get_fignums()) == figns_before


def test_build_attribution_figure_with_explanation_object():
    """Verify build_attribution_figure accepts a pre-computed shap.Explanation directly."""
    import pandas as pd
    import shap

    pipeline, classifier, feature_names = load_model_and_metadata("models/best_model.joblib")
    preprocessor = pipeline.named_steps["preprocessor"]

    df = pd.DataFrame([PROFILE_INTERMEDIATE])
    transformed = preprocessor.transform(df)
    transformed_df = pd.DataFrame(transformed, columns=feature_names)

    explainer = shap.TreeExplainer(classifier)
    patient_shap = explainer(transformed_df)[0]

    fig = build_attribution_figure(patient_shap, title="ATTRIBUTION VECTOR: PRECOMPUTED SHAP")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) > 0

    ax = fig.axes[0]
    ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
    assert any("=" in text for text in ytick_labels)

    fig_num = fig.number
    plt.close(fig)
    assert fig_num not in plt.get_fignums()
