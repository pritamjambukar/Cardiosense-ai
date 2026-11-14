"""Conversational Health Educator Chatbot Module with Clinical Safety Guardrails.

This module implements:
1. Strict clinical safety guardrails for emergency symptoms, prescription requests, and diagnostic inquiries.
2. Comprehensive multi-regional emergency routing (112 India / 911 US).
3. Robust keyword and phrase interception for emergency variants and typos.
4. Structured patient context formatting for grounded LLM responses.
5. Integration with Google Gemini with deterministic, safe offline fallbacks.
6. Zero leakage of internal exception details to the user interface.
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("cardiosense.chatbot")

SYSTEM_INSTRUCTION = """You are a health-education assistant. You receive a patient's inputs, the
model's risk probability, and top SHAP factors. Explain them in simple
language, suggest general lifestyle information and when to see a doctor.
Never diagnose, never prescribe, never contradict a doctor. If the user
describes chest pain, shortness of breath, or emergency symptoms, tell
them to seek emergency care immediately."""

# Emergency symptom keywords & variants requiring immediate clinical intervention
EMERGENCY_KEYWORDS = [
    # Canonical symptoms
    "chest pain",
    "shortness of breath",
    "tightness in chest",
    "radiating pain",
    "passing out",
    "dizziness",
    "heart attack",
    "pressure in chest",
    "difficulty breathing",
    # Variant phrasings & colloquial expressions
    "chest hurts",
    "chest hurt",
    "chest pressure",
    "chest ache",
    "chest tight",
    "can't breathe",
    "cant breathe",
    "cannot breathe",
    "trouble breathing",
    "hard to breathe",
    "short of breath",
    "left arm pain",
    "arm pain",
    "pain radiating",
    "jaw pain",
    "passed out",
    "loss of consciousness",
    "cardiac arrest",
    "fainting",
    # Common typos & phonetic variants
    "chezt pain",
    "chect pain",
    "shorness of breath",
    "hart attack",
]

# Clinical boundary triggers: Prescription requests
PRESCRIPTION_KEYWORDS = [
    "prescribe",
    "prescription",
    "medication",
    "medicine",
    "medicines",
    "pills",
    "drugs",
    "dosage",
    "dose",
    "what should i take",
    "what medicine",
    "what pill",
    "give me a prescription",
    "can you prescribe",
    "suggest some medicine",
    "which tablets",
    "tablets",
]

# Clinical boundary triggers: Definitive diagnosis requests
DIAGNOSIS_KEYWORDS = [
    "am i definitely sick",
    "do i definitely have heart disease",
    "diagnose me",
    "confirm my diagnosis",
    "am i sick",
    "do i have heart disease",
    "tell me if i have disease",
    "do i have disease",
    "is it confirmed that i have",
    "give me a diagnosis",
    "what is my diagnosis",
]


def check_emergency_symptoms(query: str) -> bool:
    """Check if query describes urgent or acute emergency cardiac symptoms."""
    q = query.lower()
    return any(kw in q for kw in EMERGENCY_KEYWORDS)


def check_prescription_request(query: str) -> bool:
    """Check if query is asking for prescription or drug recommendations."""
    q = query.lower()
    return any(kw in q for kw in PRESCRIPTION_KEYWORDS)


def check_diagnostic_request(query: str) -> bool:
    """Check if query is demanding a definitive medical diagnosis."""
    q = query.lower()
    return any(kw in q for kw in DIAGNOSIS_KEYWORDS)


def format_patient_context(patient_context: Dict[str, Any]) -> str:
    """Format patient readings and model predictions into grounded clinical text."""
    return f"""
[PATIENT SCREENING ASSESSMENT]:
- Demographics: Age {patient_context.get('age', 'N/A')}, Sex {patient_context.get('sex_label', 'N/A')}
- Vitals: BP {patient_context.get('trestbps', 'N/A')} mm Hg, Cholesterol {patient_context.get('chol', 'N/A')} mg/dl
- Exercise Physiology: Max HR {patient_context.get('thalach', 'N/A')} bpm, ST Depression {patient_context.get('oldpeak', 'N/A')} mm
- Model Predicted Probability: {patient_context.get('probability_pct', 'N/A')}
- Stratified Risk Category: {patient_context.get('risk_tier', 'N/A')} Risk
- Top Risk-Increasing Drivers: {patient_context.get('risk_drivers', 'None')}
- Top Protective Drivers: {patient_context.get('protective_drivers', 'None')}
"""


def handle_health_chat(
    user_query: str,
    patient_context: Dict[str, Any],
    chat_history: Optional[List[Dict[str, str]]] = None,
    api_key: Optional[str] = None,
    genai_client: Optional[Any] = None,
) -> str:
    """Process a patient query through safety guardrails and generate health-education responses.

    Args:
        user_query: The patient's prompt.
        patient_context: Dictionary with patient inputs, model probabilities, and SHAP drivers.
        chat_history: Optional list of past turns ({'role': 'user'/'assistant', 'content': '...'}).
        api_key: Optional Gemini API key.
        genai_client: Optional pre-configured client (useful for unit testing and mocking).

    Returns:
        String response that strictly enforces medical boundaries.
    """
    # 1. Guardrail 1: Immediate Emergency Detection (Executes strictly before LLM or fallback)
    if check_emergency_symptoms(user_query):
        return (
            "🚨 **URGENT EMERGENCY NOTIFICATION**: You have reported acute symptoms "
            "(such as chest discomfort, breathing difficulty, or radiating pain) that may signal an active cardiovascular emergency. "
            "**Please contact emergency medical services immediately: Call 112 (India) or 911 (US) or proceed to the nearest emergency department.** "
            "Do not wait or rely on digital screening tools during acute physical distress."
        )

    # 2. Guardrail 2: Prescription Refusal (Executes strictly before LLM or fallback)
    if check_prescription_request(user_query):
        return (
            "**Medication Notice**: As a health-education assistant, I **cannot prescribe medications, suggest dosages, or recommend specific pharmaceuticals**. "
            "Only a licensed physician or pharmacist who has physically examined you and reviewed your medical history can safely prescribe drugs. "
            "Please schedule an appointment with your doctor to discuss whether medication is appropriate for your heart health."
        )

    # 3. Guardrail 3: Definitive Diagnosis Refusal (Executes strictly before LLM or fallback)
    if check_diagnostic_request(user_query):
        return (
            "**Diagnostic Boundary Notice**: This software is an **educational screening model, not a definitive diagnostic test**. "
            "I cannot definitively state whether you have heart disease or confirm any pathology. "
            "Cardiovascular diagnosis requires clinical evaluation by a board-certified cardiologist, including comprehensive blood tests, ECG, and imaging."
        )

    # 4. Standard Educational Consultation via Gemini or Fallback
    active_key = api_key or os.getenv("GEMINI_API_KEY")

    if genai_client is not None or (active_key and active_key != "your_gemini_api_key_here"):
        try:
            if genai_client is None:
                from google import genai
                client = genai.Client(api_key=active_key)
            else:
                client = genai_client

            history_lines = []
            if chat_history:
                for msg in chat_history:
                    history_lines.append(f"{msg['role'].upper()}: {msg['content']}")
            history_text = "\n".join(history_lines)

            context_block = format_patient_context(patient_context)
            full_prompt = f"""{SYSTEM_INSTRUCTION}

{context_block}

CONVERSATION HISTORY:
{history_text}

USER: {user_query}
ASSISTANT:"""

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
            )
            return response.text
        except Exception as e:
            # Secure error logging: Log only exception type, never leak internal tracebacks to users
            logger.error(f"Generative service unavailable: {type(e).__name__}")
            return _generate_fallback_response(user_query, patient_context, service_unavailable=True)
    else:
        return _generate_fallback_response(user_query, patient_context)


def _generate_fallback_response(
    query: str,
    patient_context: Dict[str, Any],
    service_unavailable: bool = False,
) -> str:
    """Deterministic, educational fallback when live LLM is unconfigured or unavailable."""
    tier = patient_context.get("risk_tier", "Assessment")
    prob = patient_context.get("probability_pct", "N/A")
    risk_drivers = patient_context.get("risk_drivers", "Not specified")
    protective = patient_context.get("protective_drivers", "Not specified")

    text = (
        f"**Health-Education Summary**: In your screening profile ({tier} Risk, {prob}), "
        f"the primary factors pushing risk higher are: {risk_drivers}. Favorable indicators are: {protective}.\n\n"
        f"Regarding your inquiry *'{query}'*:\n"
        f"- Evidence-based cardiovascular wellness highlights 150 minutes of weekly moderate aerobic activity, a Mediterranean-style nutritional pattern, and routine monitoring of arterial blood pressure.\n"
        f"- Bring your ECG stress test results and lipid panels to your primary care physician or cardiologist for individualized medical guidance.\n\n"
        f"*(Notice: Educational information only. Never substitute for doctor consultation.)*"
    )
    if service_unavailable:
        text += (
            "\n\n*(Notice: Live conversational service is temporarily unavailable. "
            "A standardized clinical health-education summary has been provided above.)*"
        )
    return text
