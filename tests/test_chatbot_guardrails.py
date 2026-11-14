"""Unit tests for Chatbot Clinical Safety Guardrails, Rule-Based Fallbacks, and LLM Mocking.

Verifies:
1. Prompts mentioning acute symptoms ('chest pain', 'shortness of breath', 'chest hurts', 'can't breathe',
   'heart attack', 'left arm pain', typos like 'chezt pain') trigger emergency guidance with local numbers (112 India / 911 US).
2. Prompts asking for medication ('prescribe me medicine', 'dosage') trigger prescription refusal.
3. Prompts asking for definitive pathology ('am I definitely sick?', 'diagnose me') trigger diagnostic boundary warnings.
4. Emergency, prescription, and diagnostic queries NEVER invoke the LLM (assert_not_called).
5. Normal educational queries pass cleanly to the LLM (mocked for offline deterministic testing).
6. Rule-based offline fallback paths work deterministically when:
   - No API key is configured (api_key=None)
   - LLM API call raises a network, quota, or operational exception
   - Raw exception details are never leaked to user responses
7. Patient clinical context is correctly bound.
"""

from unittest.mock import MagicMock
import pytest
from src.chatbot import handle_health_chat


@pytest.fixture
def mock_patient_context() -> dict:
    """Mock patient clinical context passed into the chatbot."""
    return {
        "age": 58,
        "sex_label": "Male",
        "trestbps": 138,
        "chol": 245,
        "thalach": 142,
        "oldpeak": 1.2,
        "probability_pct": "51.8%",
        "risk_tier": "Medium",
        "risk_drivers": "Chest Pain (+0.42), Blocked Vessels (+0.35)",
        "protective_drivers": "Max Heart Rate (-0.21)",
    }


def test_emergency_chest_pain_guardrail(mock_patient_context):
    """User reporting chest pain must receive an immediate emergency warning with 112 India and 911 US numbers."""
    query = "I am feeling severe chest pain and pressure right now. What should I do?"
    response = handle_health_chat(query, mock_patient_context)

    # Must contain emergency alert keywords
    assert "EMERGENCY" in response.upper()
    # Must list both 112 (India) and 911 (US) emergency dispatch numbers
    assert "112" in response
    assert "911" in response
    # Must NOT provide a diagnosis or prescription
    assert "I diagnose you" not in response
    assert "take aspirin" not in response.lower()


def test_emergency_shortness_of_breath_guardrail(mock_patient_context):
    """User reporting acute shortness of breath must be directed to emergency care with local emergency numbers."""
    query = "I have sudden shortness of breath and dizziness. Am I okay?"
    response = handle_health_chat(query, mock_patient_context)

    assert "EMERGENCY" in response.upper()
    assert "112" in response
    assert "911" in response


def test_prescription_refusal_guardrail(mock_patient_context):
    """Chatbot must explicitly refuse to prescribe drugs or dosages."""
    query = "Can you prescribe me medicine or tell me which pills I should take for my cholesterol?"
    response = handle_health_chat(query, mock_patient_context)

    assert any(phrase in response.lower() for phrase in ["cannot prescribe", "never prescribe", "cannot recommend"])
    assert any(phrase in response.lower() for phrase in ["doctor", "physician", "pharmacist"])


def test_diagnostic_refusal_guardrail(mock_patient_context):
    """Chatbot must refuse to deliver a definitive medical diagnosis."""
    query = "Look at my probability of 51.8%. Am I definitely sick with heart disease?"
    response = handle_health_chat(query, mock_patient_context)

    assert any(phrase in response.lower() for phrase in ["not a definitive", "screening model", "educational", "cannot definitively"])
    assert any(phrase in response.lower() for phrase in ["cardiologist", "physician", "doctor"])


# ==============================================================================
# FORMAL VERIFICATION: GUARDRAILS NEVER CALL THE LLM (ASSERT_NOT_CALLED)
# ==============================================================================

@pytest.mark.parametrize(
    "forbidden_query, expected_indicator",
    [
        # Emergency variant phrasings
        ("My chest hurts really bad when I sit down", "112"),
        ("I can't breathe and feel tight inside", "112"),
        ("I cant breathe at all", "911"),
        ("I think I am having a heart attack right now", "EMERGENCY"),
        ("There is severe left arm pain shooting down my wrist", "EMERGENCY"),
        ("I have terrible chezt pain", "112"),  # Common typo
        ("I have severe chect pain", "911"),   # Common typo
        # Prescription variant phrasings
        ("Please prescribe me some statin tablets", "cannot prescribe"),
        ("What dosage of medicine should I take tonight?", "cannot prescribe"),
        ("What pills should I take for this?", "cannot prescribe"),
        # Diagnostic demands
        ("Please diagnose me based on these tests", "not a definitive"),
        ("Am I definitely sick with cardiac disease?", "not a definitive"),
        ("Confirm my diagnosis please", "not a definitive"),
    ],
)
def test_guardrails_never_call_llm_with_variant_phrasings(mock_patient_context, forbidden_query, expected_indicator):
    """Verify that acute emergencies, prescription requests, and diagnostic demands
    NEVER invoke the LLM endpoint under any circumstance."""
    mock_client = MagicMock()

    response = handle_health_chat(
        user_query=forbidden_query,
        patient_context=mock_patient_context,
        genai_client=mock_client,
    )

    # Formal invariant: The LLM generation endpoint must NOT have been called
    mock_client.models.generate_content.assert_not_called()
    assert mock_client.models.generate_content.call_count == 0

    # Safety response verification
    assert expected_indicator.lower() in response.lower()


def test_educational_query_with_mocked_llm(mock_patient_context):
    """Ensure normal educational queries invoke the LLM with patient context and return formatted responses."""
    query = "What dietary lifestyle changes can help support a healthy cholesterol level?"

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = (
        "Focus on dietary fiber, reducing saturated fats, and engaging in 150 minutes of weekly aerobic exercise. "
        "Consult your physician for personalized lipid management."
    )
    mock_client.models.generate_content.return_value = mock_response

    response = handle_health_chat(
        user_query=query,
        patient_context=mock_patient_context,
        genai_client=mock_client,
    )

    mock_client.models.generate_content.assert_called_once()
    assert "dietary fiber" in response
    assert "Consult your physician" in response


# ==============================================================================
# RULE-BASED FALLBACK PATH TESTS
# ==============================================================================

def test_fallback_when_no_llm_key(mock_patient_context):
    """Ensure chatbot provides structured, safe educational response when api_key=None."""
    query = "What does exertional ST depression mean?"
    response = handle_health_chat(
        user_query=query,
        patient_context=mock_patient_context,
        api_key=None,
        genai_client=None,
    )

    assert "Health-Education Summary" in response
    assert "Medium Risk" in response
    assert "51.8%" in response
    assert "Chest Pain (+0.42)" in response
    assert "Max Heart Rate (-0.21)" in response
    assert "physician" in response.lower()
    assert "Never substitute for doctor consultation" in response


def test_fallback_emergency_still_intercepted_without_api_key(mock_patient_context):
    """Even without an API key, emergency symptom guardrail executes before any fallback."""
    query = "I have crushing chest pain and feel like I'm having a heart attack."
    response = handle_health_chat(
        user_query=query,
        patient_context=mock_patient_context,
        api_key=None,
        genai_client=None,
    )

    assert "URGENT EMERGENCY NOTIFICATION" in response
    assert "112" in response
    assert "911" in response
    assert "Health-Education Summary" not in response


def test_fallback_prescription_still_refused_without_api_key(mock_patient_context):
    """Even without an API key, prescription requests are refused before any fallback."""
    query = "What dosage of statin medication should I take?"
    response = handle_health_chat(
        user_query=query,
        patient_context=mock_patient_context,
        api_key=None,
        genai_client=None,
    )

    assert "Medication Notice" in response
    assert "cannot prescribe" in response.lower()
    assert "physician" in response.lower()
    assert "Health-Education Summary" not in response


def test_fallback_diagnosis_still_refused_without_api_key(mock_patient_context):
    """Even without an API key, diagnostic confirmation demands are refused before fallback."""
    query = "Do I definitely have heart disease based on these numbers?"
    response = handle_health_chat(
        user_query=query,
        patient_context=mock_patient_context,
        api_key=None,
        genai_client=None,
    )

    assert "Diagnostic Boundary Notice" in response
    assert "not a definitive diagnostic test" in response.lower()
    assert "Health-Education Summary" not in response


def test_fallback_graceful_recovery_when_llm_api_fails(mock_patient_context):
    """When the LLM API throws an unexpected runtime exception (e.g., quota or timeout),
    the app must catch it gracefully and return the safe rule-based summary without leaking
    raw exception strings or tracebacks to the user."""
    query = "Can you explain my resting blood pressure reading?"

    # Mock client that raises a connection/quota error with sensitive internal details
    failing_client = MagicMock()
    failing_client.models.generate_content.side_effect = RuntimeError("503 Service Unavailable: High server load at backend cluster us-central1-c")

    response = handle_health_chat(
        user_query=query,
        patient_context=mock_patient_context,
        genai_client=failing_client,
    )

    # Must provide safe summary
    assert "Health-Education Summary" in response
    assert "51.8%" in response
    assert "physician" in response.lower()
    # Must inform user that live conversational service is temporarily unavailable
    assert "temporarily unavailable" in response
    # CRITICAL SECURITY CHECK: Raw exception text must NEVER be leaked to the UI
    assert "503 Service Unavailable" not in response
    assert "us-central1-c" not in response
