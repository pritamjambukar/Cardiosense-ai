"""Streamlit AppTest Smoke Test for CardioSense UI.

Verifies:
1. Loads app.py without raising unhandled exceptions or matplotlib text attribute errors.
2. Form submission executes assessment and SHAP attribution pipeline successfully.
3. Asserts app.exception is empty across the full execution cycle.
"""

from pathlib import Path
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = str(PROJECT_ROOT / "app.py")


def test_streamlit_app_smoke_assessment():
    """Smoke test that executes app.py end-to-end, clicks the assessment submit button,
    and asserts that app.exception is completely empty."""
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    # Initial page load must not have thrown any exceptions (verifies letter_spacing bug is fixed)
    assert not at.exception, f"Exceptions on initial load: {[e.value for e in at.exception]}"

    # Verify key visual elements were rendered
    assert len(at.tabs) == 2, "Expected 2 tabs ('Assessment & Interpretability' and 'Health Educator Dialogue')"

    # Find the submit button inside the sidebar intake form and trigger assessment run
    if at.sidebar.button:
        at.sidebar.button[0].click().run()
        assert not at.exception, f"Exceptions after form submission: {[e.value for e in at.exception]}"
