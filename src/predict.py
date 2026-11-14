"""Inference and Risk Stratification module for Heart Disease Prediction.

This module provides:
1. `predict(patient_dict)`: Accepts clinical patient attributes as a dictionary and returns:
   - Disease Probability (0.0 to 1.0)
   - Risk Tier: 'Low', 'Medium', or 'High'
   - Binary Decision (0 = Healthy, 1 = Disease Present)
   - Clinical Action Recommendation
2. Robust input validation and caching of the serialized model pipeline.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import pandas as pd

# Expected clinical features required by the pipeline
EXPECTED_FEATURES = [
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
]

# Clinically acceptable ranges for input validation
VALIDATION_RANGES = {
    "age": (1, 120),
    "sex": {0, 1},
    "cp": {1, 2, 3, 4},
    "trestbps": (50, 260),
    "chol": (50, 650),
    "fbs": {0, 1},
    "restecg": {0, 1, 2},
    "thalach": (40, 250),
    "exang": {0, 1},
    "oldpeak": (0.0, 10.0),
    "slope": {1, 2, 3},
    "ca": {0, 1, 2, 3},
    "thal": {3, 6, 7},
}


def validate_patient_dict(patient_dict: Dict[str, Any]) -> None:
    """Validate that input patient attributes exist and fall within physiological ranges.

    Raises:
        ValueError: If required keys are missing or values fall outside clinical bounds.
    """
    missing_keys = [f for f in EXPECTED_FEATURES if f not in patient_dict]
    if missing_keys:
        raise ValueError(f"Missing required clinical features: {missing_keys}")

    for feature, allowed in VALIDATION_RANGES.items():
        val = patient_dict[feature]
        try:
            val_num = float(val)
        except (ValueError, TypeError):
            raise ValueError(f"Feature '{feature}' must be numeric, received: {val}")

        if isinstance(allowed, tuple):
            min_val, max_val = allowed
            if not (min_val <= val_num <= max_val):
                raise ValueError(
                    f"Out of clinical range for '{feature}': {val_num}. "
                    f"Allowed range is [{min_val}, {max_val}]."
                )
        elif isinstance(allowed, set):
            if int(val_num) not in allowed or val_num != int(val_num):
                raise ValueError(
                    f"Invalid categorical value for '{feature}': {val_num}. "
                    f"Allowed values are {sorted(list(allowed))}."
                )

# Global cache so the pipeline isn't reloaded from disk on every prediction call
_CACHED_PIPELINE: Optional[object] = None


def get_pipeline(model_path: str = "models/best_model.joblib") -> object:
    """Load and cache the trained Pipeline from disk."""
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {path.resolve()}. "
                "Please run `python src/train.py` first to train and serialize the model."
            )
        _CACHED_PIPELINE = joblib.load(path)
    return _CACHED_PIPELINE


def determine_risk_tier(probability: float) -> Tuple[str, str, str]:
    """Stratify continuous probability into clinically meaningful risk categories.

    Thresholds:
    - Low Risk (< 0.35): Minimal clinical suspicion. Recommend standard wellness and routine monitoring.
    - Medium Risk (0.35 - 0.65): Borderline / intermediate suspicion. Recommend non-invasive evaluation.
    - High Risk (>= 0.65): Significant clinical suspicion. Recommend expedited cardiologist referral.
    """
    if probability < 0.35:
        return (
            "Low",
            "#2ecc71",
            "Low likelihood of coronary artery disease. Maintain heart-healthy diet, exercise, and routine checkups.",
        )
    elif probability < 0.65:
        return (
            "Medium",
            "#f39c12",
            "Moderate / borderline risk profile. Secondary non-invasive assessment (e.g., stress ECG, lipid panel review) advised.",
        )
    else:
        return (
            "High",
            "#e74c3c",
            "Elevated risk indicators detected. Strongly recommend scheduling a formal consultation with a board-certified cardiologist.",
        )


def predict(
    patient_dict: Dict[str, Any],
    model_path: str = "models/best_model.joblib",
) -> Dict[str, Any]:
    """Predict heart disease probability and risk tier from a patient attribute dictionary.

    Args:
        patient_dict: Dictionary containing the 13 clinical attributes:
            - age (int/float): Age in years
            - sex (int/float): 1 = Male, 0 = Female
            - cp (int/float): Chest pain type (1, 2, 3, or 4)
            - trestbps (float): Resting blood pressure in mm Hg
            - chol (float): Serum cholesterol in mg/dl
            - fbs (int/float): Fasting blood sugar > 120 mg/dl (1 or 0)
            - restecg (int/float): Resting ECG (0, 1, or 2)
            - thalach (float): Max heart rate achieved
            - exang (int/float): Exercise-induced angina (1 or 0)
            - oldpeak (float): ST depression induced by exercise
            - slope (int/float): Slope of peak exercise ST segment (1, 2, or 3)
            - ca (float): Number of major vessels (0 to 3)
            - thal (float): Thalassemia status (3 = normal, 6 = fixed, 7 = reversible)
        model_path: Path to serialized model artifact.

    Returns:
        Dict with keys:
            - 'probability': float (0.0 to 1.0)
            - 'probability_pct': str (e.g., '78.4%')
            - 'prediction': int (0 = No Disease, 1 = Disease)
            - 'risk_tier': str ('Low', 'Medium', or 'High')
            - 'risk_color': str (hex color code for UI rendering)
            - 'clinical_recommendation': str
            - 'patient_df': pd.DataFrame (1-row dataframe ready for SHAP)
    """
    # Defensive input validation
    validate_patient_dict(patient_dict)

    pipeline = get_pipeline(model_path)

    # Build 1-row DataFrame with explicit feature ordering
    patient_df = pd.DataFrame([{col: float(patient_dict[col]) for col in EXPECTED_FEATURES}])

    # Predict using the full leak-free pipeline (preprocessor + classifier)
    probabilities = pipeline.predict_proba(patient_df)[0]
    disease_prob = float(probabilities[1])
    binary_pred = int(disease_prob >= 0.50)

    risk_label, risk_color, recommendation = determine_risk_tier(disease_prob)

    return {
        "probability": disease_prob,
        "probability_pct": f"{disease_prob * 100:.1f}%",
        "prediction": binary_pred,
        "prediction_label": "Disease Present" if binary_pred == 1 else "No Disease Detected",
        "risk_tier": risk_label,
        "risk_color": risk_color,
        "clinical_recommendation": recommendation,
        "patient_df": patient_df,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("STAGE 6: INFERENCE PIPELINE & RISK STRATIFICATION TEST")
    print("=" * 70)

    # Case 1: Healthy Patient Profile (Low Risk)
    healthy_patient = {
        "age": 45,
        "sex": 0,          # Female
        "cp": 2,           # Atypical angina
        "trestbps": 120,   # Normal BP
        "chol": 190,       # Healthy cholesterol
        "fbs": 0,          # Normal blood sugar
        "restecg": 0,      # Normal ECG
        "thalach": 172,    # High max heart rate
        "exang": 0,        # No exercise angina
        "oldpeak": 0.0,    # Zero ST depression
        "slope": 1,        # Upsloping ST segment
        "ca": 0,           # 0 blocked vessels
        "thal": 3,         # Normal perfusion
    }

    # Case 2: Borderline Patient Profile (Medium Risk)
    borderline_patient = {
        "age": 58,
        "sex": 1,          # Male
        "cp": 3,           # Non-anginal pain
        "trestbps": 138,   # Slightly elevated BP
        "chol": 245,       # Elevated cholesterol
        "fbs": 0,
        "restecg": 1,      # ST-T wave abnormality
        "thalach": 142,    # Moderate heart rate
        "exang": 0,
        "oldpeak": 1.2,    # Mild ST depression
        "slope": 2,        # Flat slope
        "ca": 1,           # 1 blocked vessel
        "thal": 3,         # Normal perfusion
    }

    # Case 3: High-Risk Cardiac Profile (High Risk)
    high_risk_patient = {
        "age": 67,
        "sex": 1,          # Male
        "cp": 4,           # Asymptomatic chest pain (silent ischemia)
        "trestbps": 160,   # Stage 2 hypertension
        "chol": 286,       # High cholesterol
        "fbs": 1,          # High fasting blood sugar
        "restecg": 2,      # Left ventricular hypertrophy
        "thalach": 108,    # Poor max heart rate
        "exang": 1,        # Exercise-induced angina
        "oldpeak": 2.5,    # Significant ST depression
        "slope": 2,        # Flat ST slope
        "ca": 3,           # 3 major vessels blocked
        "thal": 7,         # Reversible defect
    }

    for idx, (label, profile) in enumerate([
        ("Profile A (Healthy)", healthy_patient),
        ("Profile B (Borderline)", borderline_patient),
        ("Profile C (High Risk)", high_risk_patient),
    ], 1):
        result = predict(profile)
        print(f"\n[{idx}] {label}:")
        print(f"    - Probability:  {result['probability_pct']} ({result['probability']:.4f})")
        print(f"    - Risk Tier:    {result['risk_tier']}")
        print(f"    - Prediction:   {result['prediction_label']}")
        print(f"    - Advice:       {result['clinical_recommendation']}")
