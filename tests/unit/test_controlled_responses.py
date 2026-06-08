# Unit tests — ControlledResponseManager
# Cases to cover:
#   - Each response type returns a non-empty string
#   - No response contains medical advice keywords
#   - Urgency response contains reference to medical team / SAMU
#   - All messages are in French

import re
import pytest
from src.safety.controlled_responses import ControlledResponseManager

# Keywords that must NEVER appear in any controlled response
FORBIDDEN_MEDICAL_KEYWORDS = [
    r"\bdiagnostic\s+est\b",
    r"\bvous\s+avez\b.*\bmaladie\b",
    r"\bprenez\s+\d+\s*mg\b",
    r"\bdosage\s+recommandé\b",
    r"\bpronostic\s+vital\b",
    r"\bchance\s+de\s+survie\b",
]

CRM = ControlledResponseManager()

ALL_METHODS = [
    CRM.get_urgency_redirect,
    CRM.get_refuse_medical_advice,
    CRM.get_ask_reformulation,
    CRM.get_no_source_found,
    CRM.get_out_of_scope,
    CRM.get_api_unavailable,
    CRM.get_generic_error,
]


class TestControlledResponsesNonEmpty:
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_returns_non_empty_string(self, method):
        result = method()
        assert isinstance(result, str)
        assert len(result.strip()) > 0


class TestControlledResponsesNoMedicalAdvice:
    @pytest.mark.parametrize("method", ALL_METHODS)
    @pytest.mark.parametrize("pattern", FORBIDDEN_MEDICAL_KEYWORDS)
    def test_no_forbidden_medical_content(self, method, pattern):
        result = method().lower()
        assert not re.search(pattern, result), (
            f"{method.__name__} contains forbidden pattern: {pattern!r}"
        )


class TestUrgencyRedirect:
    def test_contains_emergency_reference(self):
        result = CRM.get_urgency_redirect().lower()
        assert any(kw in result for kw in ["samu", "15", "équipe médicale", "urgence"])

    def test_contains_medical_team_reference(self):
        result = CRM.get_urgency_redirect()
        assert "équipe" in result.lower()


class TestRefuseMedicalAdvice:
    def test_redirects_to_medical_team(self):
        result = CRM.get_refuse_medical_advice().lower()
        assert "équipe" in result or "soignante" in result

    def test_explains_system_limits(self):
        result = CRM.get_refuse_medical_advice().lower()
        assert "personnalisé" in result or "autorisé" in result


class TestAllResponsesInFrench:
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_contains_french_words(self, method):
        result = method().lower()
        french_markers = ["vous", "je", "votre", "équipe", "le ", "la ", "les ", "un ", "une "]
        assert any(marker in result for marker in french_markers), (
            f"{method.__name__} does not appear to be in French"
        )

#   - All messages are in French
