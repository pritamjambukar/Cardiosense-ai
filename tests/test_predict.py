"""Unit tests for predict() Inference, Range Validation, and Risk Stratification.

Verifies:
1. Probability output is strictly bounded in [0.0, 1.0].
2. Risk label is strictly in {'Low', 'Medium', 'High'}.
3. Out-of-range or physiologically impossible inputs are rejected with ValueError.
4. Missing required features raise clear validation errors.
"""

import pytest
from src.predict import predict


@pytest.fixture
def valid_patient_profile() -> dict:
    """Standard valid baseline patient dictionary."""
    return {
        "age": 52,
        "sex": 1,
        "cp": 2,
        "trestbps": 128,
        "chol": 215,
        "fbs": 0,
        "restecg": 0,
        "thalach": 162,
        "exang": 0,
        "oldpeak": 0.4,
        "slope": 1,
        "ca": 0,
        "thal": 3,
    }


def test_predict_probability_bounds(valid_patient_profile):
    """Ensure predicted disease probability is strictly in [0.0, 1.0]."""
    result = predict(valid_patient_profile)
    prob = result["probability"]
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0, f"Probability {prob} out of [0.0, 1.0] bounds"


def test_predict_risk_label_categories(valid_patient_profile):
    """Ensure risk tier is strictly one of 'Low', 'Medium', or 'High'."""
    result = predict(valid_patient_profile)
    assert result["risk_tier"] in {"Low", "Medium", "High"}
    assert result["prediction"] in {0, 1}
    assert "clinical_recommendation" in result


def test_predict_rejects_missing_keys():
    """Ensure missing patient features immediately raise ValueError."""
    incomplete_profile = {"age": 55, "sex": 1}
    with pytest.raises(ValueError, match="Missing required clinical features"):
        predict(incomplete_profile)


@pytest.mark.parametrize(
    "invalid_feature, bad_value",
    [
        ("age", -5),            # Negative age
        ("age", 150),           # Impossible human age
        ("trestbps", 350),      # Fatal blood pressure
        ("trestbps", 30),       # Unviable blood pressure
        ("chol", -10),          # Negative cholesterol
        ("chol", 900),          # Out of range
        ("sex", 2),             # Invalid categorical sex
        ("cp", 5),              # Invalid chest pain type
        ("fbs", 3),             # Invalid blood sugar flag
        ("restecg", 4),         # Invalid ECG code
        ("thalach", 10),        # Unviable heart rate
        ("thalach", 300),       # Impossible heart rate
        ("oldpeak", -2.0),      # Negative ST depression
        ("oldpeak", 15.0),      # Impossibly high ST depression
        ("slope", 0),           # Invalid slope code
        ("ca", 5),              # Max fluoroscopy vessels is 3
        ("thal", 2),            # Valid thal codes are 3, 6, 7
    ],
)
def test_predict_rejects_out_of_range_inputs(valid_patient_profile, invalid_feature, bad_value):
    """Ensure physiologically impossible or invalid values raise ValueError."""
    bad_profile = valid_patient_profile.copy()
    bad_profile[invalid_feature] = bad_value

    with pytest.raises(ValueError, match=f"(Out of clinical range|Invalid categorical value)"):
        predict(bad_profile)
