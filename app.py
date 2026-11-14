"""CardioSense AI — Clinical Risk Stratification & Interpretability Engine.

Hardened Production Interface:
- Input validation with user-facing clinical error messaging.
- Cached pipeline loading via st.cache_resource with missing-artifact safeguards.
- Graceful offline fallback when LLM API keys are unconfigured or calls fail.
- Zero-logging of patient personal/clinical entries (HIPAA / Privacy compliant).
- Medical disclaimer persistently visible across all views.
"""

import html
import logging
import os
from pathlib import Path
import sys

# Configure silent / secure operational logging (NEVER log patient clinical data)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("cardiosense")

# Ensure root directory is accessible for imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.chatbot import handle_health_chat
from src.explain import CLINICAL_FEATURE_LABELS, build_attribution_figure, load_model_and_metadata
from src.predict import EXPECTED_FEATURES, predict, validate_patient_dict

# Load environment variables (.env)
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="CardioSense — Clinical Assessment Engine",
    page_icon="—",
    layout="wide",
    initial_sidebar_state="expanded",
)


def resolve_gemini_api_key(manual_entry: str | None = None) -> str | None:
    """Resolve Gemini API key using a secure precedence order:
    1. Streamlit Cloud Secrets (st.secrets['GEMINI_API_KEY'])
    2. Local Environment variable (.env / os.environ)
    3. Manual password entry in UI (if provided)
    """
    # 1. Check Streamlit Cloud Secrets safely
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            key = str(st.secrets["GEMINI_API_KEY"]).strip()
            if key and key != "your_gemini_api_key_here":
                return key
    except Exception:
        pass

    # 2. Fallback to local environment variable (.env)
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key and env_key.strip() != "your_gemini_api_key_here":
        return env_key.strip()

    # 3. Fallback to manual entry from UI
    if manual_entry and manual_entry.strip() and manual_entry.strip() != "your_gemini_api_key_here":
        return manual_entry.strip()

    return None


def inject_custom_styles():
    """Inject custom editorial CSS stylesheet."""
    css_path = PROJECT_ROOT / "styles.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


inject_custom_styles()


@st.cache_resource(show_spinner=False)
def get_model_assets():
    """Safely load and cache the trained pipeline, classifier, and feature metadata."""
    model_file = PROJECT_ROOT / "models" / "best_model.joblib"
    if not model_file.exists():
        return None, None, None
    try:
        return load_model_and_metadata(str(model_file))
    except Exception as e:
        logger.error(f"Error loading model artifact: {e}")
        return None, None, None


# --- Preset Clinical Profiles for Quick Demonstration ---
PRESETS = {
    "Select a demonstration profile...": None,
    "Profile A: Normal Baseline (Low Risk)": {
        "age": 45, "sex": 0, "cp": 2, "trestbps": 120, "chol": 190,
        "fbs": 0, "restecg": 0, "thalach": 175, "exang": 0, "oldpeak": 0.0,
        "slope": 1, "ca": 0, "thal": 3,
    },
    "Profile B: Intermediate Risk Profile (Medium Risk)": {
        "age": 58, "sex": 1, "cp": 3, "trestbps": 138, "chol": 245,
        "fbs": 0, "restecg": 1, "thalach": 142, "exang": 0, "oldpeak": 1.2,
        "slope": 2, "ca": 1, "thal": 3,
    },
    "Profile C: Elevated Risk Profile (High Risk)": {
        "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286,
        "fbs": 1, "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 2.5,
        "slope": 2, "ca": 3, "thal": 7,
    },
}

# --- Sidebar Assessment Form ---
st.sidebar.markdown('<div class="section-label">CLINICAL PARAMETERS</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="section-title">Patient Intake</div>', unsafe_allow_html=True)

preset_choice = st.sidebar.selectbox("Demonstration Presets", list(PRESETS.keys()))
default_vals = PRESETS.get(preset_choice) or PRESETS["Profile B: Intermediate Risk Profile (Medium Risk)"]

with st.sidebar.form("assessment_form"):
    st.markdown('<div class="section-label">I. DEMOGRAPHICS & VITALS</div>', unsafe_allow_html=True)
    age = st.slider(
        "Age (Years)",
        min_value=20,
        max_value=95,
        value=int(default_vals["age"]),
        help="Patient age in years.",
    )
    sex_label = st.radio(
        "Sex Assigned at Birth",
        options=["Male", "Female"],
        index=0 if default_vals["sex"] == 1 else 1,
        help="Biological sex recorded at hospital admission.",
    )
    sex = 1 if sex_label == "Male" else 0

    trestbps = st.slider(
        "Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=210,
        value=int(default_vals["trestbps"]),
        help="Resting arterial blood pressure upon admission. Clinical baseline is < 120 mm Hg.",
    )
    chol = st.slider(
        "Serum Cholesterol (mg/dl)",
        min_value=100,
        max_value=580,
        value=int(default_vals["chol"]),
        help="Total serum cholesterol level in mg/dl.",
    )
    fbs_label = st.radio(
        "Fasting Blood Sugar > 120 mg/dl",
        options=["No (Normal Range)", "Yes (Elevated Glycemia)"],
        index=1 if default_vals["fbs"] == 1 else 0,
        help="Fasting blood glucose > 120 mg/dl indicator.",
    )
    fbs = 1 if "Yes" in fbs_label else 0

    st.markdown('<div class="section-label" style="margin-top: 16px;">II. EXERCISE PHYSIOLOGY</div>', unsafe_allow_html=True)
    cp_options = {
        "Type 1: Typical Angina": 1,
        "Type 2: Atypical Angina": 2,
        "Type 3: Non-Anginal Discomfort": 3,
        "Type 4: Asymptomatic (Silent Ischemia Risk)": 4,
    }
    cp_default_key = [k for k, v in cp_options.items() if v == default_vals["cp"]][0]
    cp_label = st.selectbox(
        "Chest Pain Classification",
        options=list(cp_options.keys()),
        index=list(cp_options.keys()).index(cp_default_key),
        help="Subjective chest discomfort classification.",
    )
    cp = cp_options[cp_label]

    thalach = st.slider(
        "Maximum Exertional Heart Rate (bpm)",
        min_value=60,
        max_value=220,
        value=int(default_vals["thalach"]),
        help="Maximum heart rate achieved during standardized treadmill stress protocol.",
    )
    exang_label = st.radio(
        "Exercise-Induced Angina",
        options=["Absent", "Present"],
        index=1 if default_vals["exang"] == 1 else 0,
        help="Onset of angina pectoris provoked specifically by physical exertion.",
    )
    exang = 1 if exang_label == "Present" else 0

    oldpeak = st.slider(
        "Exertional ST Depression (mm)",
        min_value=0.0,
        max_value=6.5,
        value=float(default_vals["oldpeak"]),
        step=0.1,
        help="ST segment depression relative to baseline resting ECG.",
    )

    slope_options = {
        "Upsloping (Physiologic Response)": 1,
        "Flat (Equivocal / Ischemic Risk)": 2,
        "Downsloping (Severe Ischemia Marker)": 3,
    }
    slope_default_key = [k for k, v in slope_options.items() if v == default_vals["slope"]][0]
    slope_label = st.selectbox(
        "Peak ST Segment Slope",
        options=list(slope_options.keys()),
        index=list(slope_options.keys()).index(slope_default_key),
        help="Morphology of ST segment slope at peak physical exertion.",
    )
    slope = slope_options[slope_label]

    st.markdown('<div class="section-label" style="margin-top: 16px;">III. ELECTROCARDIOGRAPHY & IMAGING</div>', unsafe_allow_html=True)
    restecg_options = {
        "Normal Baseline ECG": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy (LVH)": 2,
    }
    restecg_default_key = [k for k, v in restecg_options.items() if v == default_vals["restecg"]][0]
    restecg_label = st.selectbox(
        "Resting Electrocardiography",
        options=list(restecg_options.keys()),
        index=list(restecg_options.keys()).index(restecg_default_key),
        help="Baseline 12-lead ECG reading prior to exertion.",
    )
    restecg = restecg_options[restecg_label]

    ca = st.slider(
        "Major Vessels Colored by Fluoroscopy",
        min_value=0,
        max_value=3,
        value=int(default_vals["ca"]),
        help="Number of major coronary vessels showing luminal diameter narrowing.",
    )

    thal_options = {
        "Normal Perfusion Scan": 3,
        "Fixed Defect (Infarction Scar)": 6,
        "Reversible Defect (Stress Ischemia)": 7,
    }
    thal_default_key = [k for k, v in thal_options.items() if v == default_vals["thal"]][0]
    thal_label = st.selectbox(
        "Myocardial Scintigraphy (Thallium)",
        options=list(thal_options.keys()),
        index=list(thal_options.keys()).index(thal_default_key),
        help="Nuclear myocardial perfusion scintigraphy evaluation.",
    )
    thal = thal_options[thal_label]

    submit_button = st.form_submit_button("Run Clinical Stratification", use_container_width=True)

# Sidebar persistent disclaimer
st.sidebar.markdown(
    """
    <div style="font-size: 0.74rem; color: #888888; border-top: 1px solid #E5E5E5; padding-top: 14px; margin-top: 24px;">
        <strong>NOTICE:</strong> Educational decision-support tool. Not a clinical diagnosis. Consult a physician for diagnostic advice.
    </div>
    """,
    unsafe_allow_html=True,
)

# Current Patient Record
current_patient = {
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
}

# --- Editorial Hero Section ---
st.markdown(
    """
    <div class="hero-wrapper">
        <div class="hero-title">CardioSense</div>
        <div class="hero-subtitle">Clinical Risk Stratification & Interpretability Engine</div>
    </div>
    <div class="editorial-disclaimer">
        <strong>Medical Notice:</strong> This software is an academic engineering decision-support model built for educational analysis. 
        It does not formulate diagnostic verdicts or clinical treatment plans. Patients experiencing acute chest pressure, 
        dyspnea, or cardiac distress must contact emergency clinical services immediately.
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Hardened Inference & Pipeline Execution ---
pipeline, classifier, feature_names = get_model_assets()

if pipeline is None:
    st.markdown(
        """
        <div class="editorial-card" style="border-left: 3px solid #881337;">
            <div class="section-label" style="color: #881337;">MODEL INITIALIZATION REQUIRED</div>
            <div style="font-size: 0.90rem; color: #333333; margin-top: 8px;">
                The trained machine learning pipeline was not located at <code>models/best_model.joblib</code>.<br>
                Please execute <code>python src/train.py</code> in your project terminal to train and serialize the model.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Validate inputs with friendly UI error handling
try:
    validate_patient_dict(current_patient)
    result = predict(current_patient)
    prob = result["probability"]
    tier = result["risk_tier"]

    # Compute SHAP values for the patient
    preprocessor = pipeline.named_steps["preprocessor"]
    patient_df = result["patient_df"]
    patient_transformed = preprocessor.transform(patient_df)
    patient_transformed_df = pd.DataFrame(patient_transformed, columns=feature_names)

    explainer = shap.TreeExplainer(classifier)
    patient_shap = explainer(patient_transformed_df)[0]

    # Extract risk drivers for explainability & chatbot context
    values = patient_shap.values
    names = patient_shap.feature_names
    contributions = []
    for name, val in zip(names, values):
        friendly = CLINICAL_FEATURE_LABELS.get(name, name)
        contributions.append({"feature": friendly, "shap": val})

    top_risk_factors = sorted([c for c in contributions if c["shap"] > 0], key=lambda x: x["shap"], reverse=True)[:4]
    top_protective_factors = sorted([c for c in contributions if c["shap"] < 0], key=lambda x: x["shap"])[:4]

except ValueError as val_err:
    err_escaped = html.escape(str(val_err))
    st.markdown(
        f"""
        <div class="editorial-card" style="border-left: 3px solid #881337;">
            <div class="section-label" style="color: #881337;">INPUT VALIDATION NOTICE</div>
            <div style="font-size: 0.90rem; color: #333333; margin-top: 8px;">
                {err_escaped}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()
except Exception as err:
    logger.error(f"Inference exception: {type(err).__name__}")
    st.markdown(
        """
        <div class="editorial-card" style="border-left: 3px solid #881337;">
            <div class="section-label" style="color: #881337;">PROCESSING ERROR</div>
            <div style="font-size: 0.90rem; color: #333333; margin-top: 8px;">
                An unexpected error occurred during risk computation. Please verify input entries and refresh.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# --- Navigation Tabs ---
tab1, tab2 = st.tabs(["I. Assessment & Interpretability", "II. Health Educator Dialogue"])

# ==============================================================================
# TAB 1: ASSESSMENT & INTERPRETABILITY
# ==============================================================================
with tab1:
    badge_class = f"badge-{tier.lower()}"
    tier_escaped = html.escape(str(tier))
    pred_label_escaped = html.escape(str(result["prediction_label"]))
    prob_pct_escaped = html.escape(str(result["probability_pct"]))
    rec_escaped = html.escape(str(result["clinical_recommendation"]))

    col_res1, col_res2, col_res3 = st.columns([1.1, 1.1, 1.8])

    with col_res1:
        st.markdown(
            f"""
            <div class="editorial-card">
                <div class="metric-caption">Risk Stratification</div>
                <div style="margin: 14px 0 8px 0;"><span class="{badge_class}">{tier_escaped} Risk</span></div>
                <div style="font-size: 0.80rem; color: #777777;">Binary Verdict: {pred_label_escaped}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_res2:
        st.markdown(
            f"""
            <div class="editorial-card">
                <div class="metric-caption">Disease Probability</div>
                <div class="metric-number">{prob_pct_escaped}</div>
                <div style="font-size: 0.80rem; color: #777777;">Calibrated Model Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_res3:
        st.markdown(
            f"""
            <div class="editorial-card">
                <div class="metric-caption">Clinical Guidance</div>
                <div style="font-size: 0.88rem; line-height: 1.5; color: #333333; margin-top: 10px;">
                    {rec_escaped}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 3-Segment Architectural Risk Track
    gauge_active_low = "active-low" if tier == "Low" else ""
    gauge_active_med = "active-medium" if tier == "Medium" else ""
    gauge_active_high = "active-high" if tier == "High" else ""

    st.markdown(
        f"""
        <div style="margin-top: 8px;">
            <div class="gauge-track">
                <div class="gauge-segment {gauge_active_low}"></div>
                <div class="gauge-segment {gauge_active_med}"></div>
                <div class="gauge-segment {gauge_active_high}"></div>
            </div>
            <div class="gauge-labels">
                <span>Low Risk (&lt; 35%)</span>
                <span>Intermediate (35% – 65%)</span>
                <span>Elevated Risk (&ge; 65%)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- SHAP Waterfall Visualization ---
    st.markdown('<div class="section-title" style="margin-top: 36px;">Patient Feature Attribution</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size: 0.86rem; color: #666666; margin-bottom: 20px;'>"
        "The SHAP waterfall plot demonstrates the mathematical shift from the baseline population mean to this patient's calibrated probability score."
        "</div>",
        unsafe_allow_html=True,
    )

    col_shap_fig, col_shap_list = st.columns([1.5, 1.2])

    with col_shap_fig:
        fig = build_attribution_figure(patient_shap, max_display=8, title="ATTRIBUTION VECTOR: PATIENT CASE")
        st.pyplot(fig)
        plt.close(fig)

    with col_shap_list:
        st.markdown('<div class="section-label">ATTRIBUTION BREAKDOWN</div>', unsafe_allow_html=True)

        st.markdown(
            "<div style='font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #881337; margin-top: 8px;'>"
            "Risk-Elevating Indicators"
            "</div>",
            unsafe_allow_html=True,
        )
        if top_risk_factors:
            shap_html = "<div class='shap-list'>"
            for r in top_risk_factors:
                feat_escaped = html.escape(str(r["feature"]))
                shap_html += f"<div class='shap-row'><span>{feat_escaped}</span><span class='shap-val-pos'>+{r['shap']:.2f}</span></div>"
            shap_html += "</div>"
            st.markdown(shap_html, unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size: 0.82rem; color: #777777;'>No significant risk-elevating factors detected.</div>", unsafe_allow_html=True)

        st.markdown(
            "<div style='font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #1E4D38; margin-top: 18px;'>"
            "Protective Clinical Factors"
            "</div>",
            unsafe_allow_html=True,
        )
        if top_protective_factors:
            shap_html = "<div class='shap-list'>"
            for p in top_protective_factors:
                feat_escaped = html.escape(str(p["feature"]))
                shap_html += f"<div class='shap-row'><span>{feat_escaped}</span><span class='shap-val-neg'>{p['shap']:.2f}</span></div>"
            shap_html += "</div>"
            st.markdown(shap_html, unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size: 0.82rem; color: #777777;'>No significant protective factors detected.</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="font-size: 0.76rem; color: #999999; border-top: 1px solid #EAEAEA; padding-top: 16px; margin-top: 32px;">
            CardioSense Assessment Engine &bull; Educational Decision-Support Model &bull; Diagnostic Confirmation Required by Licensed Physician
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# TAB 2: HEALTH EDUCATOR DIALOGUE
# ==============================================================================
with tab2:
    st.markdown('<div class="section-title">Health-Education Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size: 0.86rem; color: #666666; margin-bottom: 20px;'>"
        "Conversational guidance interpreting physiological indicators, cardiovascular lifestyle principles, and questions for clinical consultations."
        "</div>",
        unsafe_allow_html=True,
    )

    # Secure API key resolution (Precedence: st.secrets -> .env -> manual password input)
    resolved_key = resolve_gemini_api_key()
    user_api_key = None

    if resolved_key:
        st.caption("🔒 Gemini API key loaded via secure configuration (Streamlit Secrets / Environment).")
        active_key = resolved_key
    else:
        with st.expander("🔑 Configure Gemini API Key (Optional)", expanded=False):
            user_api_key = st.text_input(
                "Gemini API Key (Free tier from Google AI Studio):",
                type="password",
                placeholder="AIzaSy...",
                help="Enter your Gemini key to activate live generative dialogue. Otherwise, structured educational responses are provided.",
            )
        active_key = resolve_gemini_api_key(manual_entry=user_api_key)

    risk_text = ", ".join([f"{r['feature']} (+{r['shap']:.2f})" for r in top_risk_factors]) or "None"
    protective_text = ", ".join([f"{p['feature']} ({p['shap']:.2f})" for p in top_protective_factors]) or "None"

    # Multi-turn chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    col_reset1, col_reset2 = st.columns([1, 4])
    with col_reset1:
        if st.button("Reset Dialogue"):
            st.session_state.messages = []
            st.rerun()

    # Display Chat History with strict HTML escaping to prevent XSS
    for msg in st.session_state.messages:
        escaped_body = html.escape(str(msg["content"])).replace("\n", "<br>")
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div class="chat-card-user">
                    <div class="chat-speaker-label">Inquiry</div>
                    <div>{escaped_body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="chat-card-assistant">
                    <div class="chat-speaker-label">Clinical Health Educator</div>
                    <div>{escaped_body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Initial Welcome message if empty
    if not st.session_state.messages:
        welcome_msg = (
            f"Welcome. I have reviewed your current screening results: <strong>{tier_escaped} Risk</strong> "
            f"(calibrated probability: <strong>{prob_pct_escaped}</strong>).<br><br>"
            "You may ask for clarification regarding:<br>"
            "&bull; The physiological meaning of exertional ST depression or fluoroscopy findings.<br>"
            "&bull; Evidence-based cardiovascular lifestyle and dietary principles.<br>"
            "&bull; Specific inquiry formulations to discuss with your primary cardiologist.<br><br>"
            "<em>(Notice: This assistant provides health-education only and does not issue diagnoses or treatment prescriptions.)</em>"
        )
        st.markdown(
            f"""
            <div class="chat-card-assistant">
                <div class="chat-speaker-label">Clinical Health Educator</div>
                <div>{welcome_msg}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Chat Input with Per-Session Message Capping (Defensive Quota & Anti-Abuse Control)
    MAX_SESSION_INQUIRIES = 10
    user_inquiry_count = sum(1 for m in st.session_state.messages if m["role"] == "user")
    remaining_inquiries = max(0, MAX_SESSION_INQUIRIES - user_inquiry_count)

    if user_inquiry_count >= MAX_SESSION_INQUIRIES:
        st.markdown(
            f"""
            <div class="editorial-card" style="border-left: 3px solid #B45309; margin-top: 16px;">
                <div class="section-label" style="color: #B45309;">SESSION INQUIRY LIMIT REACHED ({MAX_SESSION_INQUIRIES}/{MAX_SESSION_INQUIRIES})</div>
                <div style="font-size: 0.88rem; color: #333333; margin-top: 8px;">
                    You have reached the maximum allowed inquiries for this screening session.
                    To prevent automated API overuse and prioritize in-person clinical consultations, additional inquiries are paused.
                    Please discuss these metrics directly with your physician or click <strong>Reset Dialogue</strong> to begin a new session.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"Inquiry balance: {remaining_inquiries} of {MAX_SESSION_INQUIRIES} queries remaining in this screening session.")
        user_query = st.chat_input("Submit an inquiry regarding your screening assessment...")

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})

            chat_patient_context = {
                "age": age,
                "sex_label": sex_label,
                "trestbps": trestbps,
                "chol": chol,
                "fbs_label": fbs_label,
                "thalach": thalach,
                "exang_label": exang_label,
                "oldpeak": oldpeak,
                "slope_label": slope_label,
                "restecg_label": restecg_label,
                "ca": ca,
                "thal_label": thal_label,
                "probability_pct": result['probability_pct'],
                "risk_tier": tier,
                "risk_drivers": risk_text,
                "protective_drivers": protective_text,
            }

            response_text = handle_health_chat(
                user_query=user_query,
                patient_context=chat_patient_context,
                chat_history=st.session_state.messages,
                api_key=active_key if active_key != "your_gemini_api_key_here" else None,
            )

            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.rerun()

    st.markdown(
        """
        <div style="font-size: 0.76rem; color: #999999; border-top: 1px solid #EAEAEA; padding-top: 16px; margin-top: 32px;">
            Educational Dialogue Assistant &bull; Never Prescribes &bull; Never Diagnoses &bull; Emergency Distress Requires Immediate 911 Call
        </div>
        """,
        unsafe_allow_html=True,
    )
