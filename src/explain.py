"""SHAP Model Interpretability & Plain-Language Clinical Explainer.

This module provides:
1. Global Feature Importance via SHAP TreeExplainer (beeswarm summary plot).
2. Local Individual Prediction Explanation (waterfall attribution plot).
3. Plain-English translation of top risk factors.
4. Integration with Google Gemini for conversational patient-friendly summaries.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from src.train import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    load_dataset,
    split_data,
)

# Friendly clinical labels for patient-facing explanations
CLINICAL_FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex at birth (1=Male, 0=Female)",
    "cp_1.0": "Typical Angina Chest Pain",
    "cp_2.0": "Atypical Angina Chest Pain",
    "cp_3.0": "Non-Anginal Chest Pain",
    "cp_4.0": "Asymptomatic Chest Pain",
    "trestbps": "Resting Blood Pressure",
    "chol": "Serum Cholesterol",
    "fbs": "Fasting Blood Sugar > 120 mg/dl",
    "restecg_0.0": "Normal Resting ECG",
    "restecg_1.0": "ST-T Wave Abnormality ECG",
    "restecg_2.0": "Left Ventricular Hypertrophy ECG",
    "thalach": "Maximum Heart Rate Achieved",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "Exercise ST Depression",
    "slope_1.0": "Upsloping ST Segment",
    "slope_2.0": "Flat ST Segment",
    "slope_3.0": "Downsloping ST Segment",
    "ca": "Number of Major Blocked Vessels",
    "thal_3.0": "Normal Thalassemia Blood Flow",
    "thal_6.0": "Fixed Thalassemia Defect",
    "thal_7.0": "Reversible Thalassemia Defect",
}


def load_model_and_metadata(
    model_path: str = "models/best_model.joblib",
) -> Tuple[object, object, List[str]]:
    """Load serialized pipeline, extract preprocessor and classifier, and recover transformed feature names."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path.resolve()}. Please run src/train.py first.")

    pipeline = joblib.load(path)
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names = NUMERIC_FEATURES + BINARY_FEATURES + cat_names

    return pipeline, classifier, feature_names


def compute_shap_explanations(
    classifier,
    preprocessor,
    X_test: pd.DataFrame,
    feature_names: List[str],
) -> Tuple[shap.TreeExplainer, shap.Explanation, pd.DataFrame]:
    """Compute SHAP values for holdout test samples using TreeExplainer."""
    X_test_transformed = preprocessor.transform(X_test)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(X_test_df)

    return explainer, shap_values, X_test_df


def plot_global_importance(
    shap_values: shap.Explanation,
    output_path: str = "shap_plots/shap_global_summary.png",
) -> None:
    """Generate and save SHAP Beeswarm Summary Plot."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=12, show=False)
    plt.title("SHAP Global Feature Importance (Test Set)", fontsize=13, pad=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_individual_explanation(
    shap_values: shap.Explanation,
    patient_idx: int = 0,
    output_path: str = "shap_plots/shap_patient_waterfall.png",
) -> None:
    """Generate and save SHAP Waterfall plot for a specific patient."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(shap_values[patient_idx], max_display=10, show=False)
    plt.title(f"SHAP Waterfall Attribution: Patient #{patient_idx}", fontsize=13, pad=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_attribution_figure(
    patient_input: dict | pd.DataFrame | shap.Explanation,
    pipeline: object | None = None,
    classifier: object | None = None,
    feature_names: List[str] | None = None,
    max_display: int = 8,
    title: str = "ATTRIBUTION VECTOR: PATIENT CASE",
) -> plt.Figure:
    """Build a standalone matplotlib Figure containing the SHAP waterfall attribution plot.

    Args:
        patient_input: Either a patient dict, single-row DataFrame, or precomputed shap.Explanation.
        pipeline: Optional preloaded scikit-learn Pipeline. If None, loaded from models/best_model.joblib.
        classifier: Optional preloaded classifier. If None, extracted from pipeline.
        feature_names: Optional transformed feature names list.
        max_display: Maximum number of top contributing features to render in waterfall.
        title: Title string (rendered cleanly without invalid CSS-style kwargs).

    Returns:
        plt.Figure: The rendered matplotlib figure object ready for st.pyplot.
    """
    if isinstance(patient_input, shap.Explanation):
        patient_shap = patient_input
    else:
        # Resolve model assets if not provided
        if pipeline is None or classifier is None or feature_names is None:
            pipeline, classifier, feature_names = load_model_and_metadata("models/best_model.joblib")

        if isinstance(patient_input, dict):
            df = pd.DataFrame([patient_input])
        else:
            df = patient_input

        preprocessor = pipeline.named_steps["preprocessor"]
        transformed = preprocessor.transform(df)
        transformed_df = pd.DataFrame(transformed, columns=feature_names)

        explainer = shap.TreeExplainer(classifier)
        patient_shap = explainer(transformed_df)[0]

    fig, ax = plt.subplots(figsize=(8, 5.0))
    shap.plots.waterfall(patient_shap, max_display=max_display, show=False)
    ax.set_title(title, fontsize=10, fontweight="bold", color="#333333", pad=12)
    plt.tight_layout()
    return fig


def extract_patient_risk_drivers(
    shap_values: shap.Explanation,
    patient_idx: int = 0,
    top_k: int = 4,
) -> Dict[str, list]:
    """Extract top positive (risk-increasing) and negative (protective) features for a patient."""
    patient_explanation = shap_values[patient_idx]
    values = patient_explanation.values
    names = patient_explanation.feature_names
    data_vals = patient_explanation.data

    contributions = []
    for name, val, raw_val in zip(names, values, data_vals):
        friendly_name = CLINICAL_FEATURE_LABELS.get(name, name)
        contributions.append({
            "feature": friendly_name,
            "raw_name": name,
            "shap_value": val,
            "patient_value": raw_val,
        })

    # Sort: positive SHAP pushes towards Disease, negative pushes towards Healthy
    risk_factors = sorted([c for c in contributions if c["shap_value"] > 0], key=lambda x: x["shap_value"], reverse=True)[:top_k]
    protective_factors = sorted([c for c in contributions if c["shap_value"] < 0], key=lambda x: x["shap_value"])[:top_k]

    return {
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
    }


def explain_prediction_plain_english(
    patient_record: pd.Series,
    probability: float,
    risk_drivers: Dict[str, list],
) -> str:
    """Produce a deterministic, plain-language clinical summary."""
    status = "ELEVATED RISK of Heart Disease" if probability >= 0.50 else "LOW RISK / Baseline"
    summary_lines = [
        f"**Prediction**: {status} (Estimated Probability: {probability:.1%})\n",
        "**Key Risk Drivers (Factors pushing risk higher):**",
    ]

    for item in risk_drivers["risk_factors"]:
        summary_lines.append(f"- **{item['feature']}**: Contributed strongly to higher risk (SHAP impact: +{item['shap_value']:.2f}).")

    if risk_drivers["protective_factors"]:
        summary_lines.append("\n**Key Protective Factors (Factors lowering risk):**")
        for item in risk_drivers["protective_factors"]:
            summary_lines.append(f"- **{item['feature']}**: Favorable indicator (SHAP impact: {item['shap_value']:.2f}).")

    summary_lines.append(
        "\n*Medical Disclaimer: This AI-generated interpretation is an educational screening aid "
        "and not a clinical diagnosis. Always consult a board-certified cardiologist for diagnostic confirmation.*"
    )

    return "\n".join(summary_lines)


def query_gemini_explainer(
    probability: float,
    risk_drivers: Dict[str, list],
    patient_raw_dict: dict,
    api_key: str | None = None,
) -> str:
    """Send structured clinical findings to Google Gemini for a conversational, patient-friendly explanation."""
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key or key == "your_gemini_api_key_here":
        return explain_prediction_plain_english(pd.Series(patient_raw_dict), probability, risk_drivers)

    try:
        from google import genai
        client = genai.Client(api_key=key)

        prompt = f"""
You are an expert cardiologist assisting a patient in understanding an ML heart disease screening result.
Explain these results in plain, empathetic, accessible language that a non-medical person can easily understand.

Patient Clinical Data:
- Model Predicted Probability of Heart Disease: {probability:.1%}
- Key Risk Factors (factors raising risk):
{', '.join([f"{item['feature']} (impact: +{item['shap_value']:.2f})" for item in risk_drivers['risk_factors']])}
- Key Protective Factors (factors lowering risk):
{', '.join([f"{item['feature']} (impact: {item['shap_value']:.2f})" for item in risk_drivers['protective_factors']])}

Instructions:
1. Explain clearly what these specific factors mean (e.g. explain ST depression, maximum heart rate, or chest pain in plain words).
2. Suggest 2-3 practical, proactive lifestyle questions or topics the patient can discuss with their doctor.
3. Conclude with a strict medical disclaimer: educational tool, not a clinical diagnosis.
Keep response concise, supportive, and under 250 words.
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return (
            explain_prediction_plain_english(pd.Series(patient_raw_dict), probability, risk_drivers)
            + f"\n\n*(Note: Gemini API fallback activated due to: {e})*"
        )


if __name__ == "__main__":
    print("=" * 70)
    print("STAGE 5: SHAP GLOBAL & LOCAL EXPLAINABILITY")
    print("=" * 70)

    # 1. Load Data & Model
    X, y = load_dataset("data/raw/heart.csv")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42)

    pipeline, classifier, feature_names = load_model_and_metadata("models/best_model.joblib")
    preprocessor = pipeline.named_steps["preprocessor"]

    # 2. Compute SHAP Values
    print("\n>>> Computing SHAP values on holdout test set (TreeExplainer)...")
    explainer, shap_values, X_test_df = compute_shap_explanations(
        classifier, preprocessor, X_test, feature_names
    )

    # 3. Save Global Beeswarm Plot
    print("\n>>> Generating SHAP Global Summary Plot...")
    plot_global_importance(shap_values, "shap_plots/shap_global_summary.png")
    print("Saved: shap_plots/shap_global_summary.png")

    # 4. Save Local Waterfall Plot for Patient #0
    print("\n>>> Generating Individual SHAP Waterfall Plot for Patient #0...")
    plot_individual_explanation(shap_values, patient_idx=0, output_path="shap_plots/shap_patient_waterfall.png")
    print("Saved: shap_plots/shap_patient_waterfall.png")

    # 5. Extract Patient #0 Risk Factors & Plain English Summary
    patient_0_raw = X_test.iloc[0]
    patient_0_prob = pipeline.predict_proba(X_test.iloc[[0]])[0, 1]
    risk_drivers = extract_patient_risk_drivers(shap_values, patient_idx=0)

    print("\n" + "=" * 50)
    print(f"PATIENT #0 CLINICAL SHAP SUMMARY (Actual Class: {y_test.iloc[0]})")
    print("=" * 50)
    plain_summary = explain_prediction_plain_english(patient_0_raw, patient_0_prob, risk_drivers)
    print(plain_summary)
