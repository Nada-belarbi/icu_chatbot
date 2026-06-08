# Unit tests — PostGenerationChecker (Phase 13)
# Cases:
#   - Safe general answer → accepted unchanged, was_modified = False
#   - "Il va survivre" → rejected, was_modified = True
#   - "Donnez-lui ce médicament" → rejected
#   - Dosage (e.g. "prenez 500mg par jour") → rejected
#   - Direct recommendation → rejected
#   - was_modified flag always correctly set

import pytest
from src.safety.post_checker import PostGenerationChecker


@pytest.fixture
def checker():
    return PostGenerationChecker()


class TestSafeAnswers:
    def test_safe_answer_returned_unchanged(self, checker):
        response = (
            "Le respirateur mécanique est un appareil qui aide le patient à respirer. "
            "Pour plus d'informations, consultez l'équipe médicale."
        )
        result, modified = checker.check(response)
        assert result == response
        assert modified is False

    def test_was_modified_false_for_safe(self, checker):
        _, modified = checker.check("Voici une information générale sur la réanimation.")
        assert modified is False

    def test_empty_string_is_safe(self, checker):
        result, modified = checker.check("")
        assert result == ""
        assert modified is False


class TestDangerousAnswers:
    def test_survival_prediction_rejected(self, checker):
        result, modified = checker.check("Il va survivre sans problème.")
        assert modified is True
        assert len(result) > 0

    def test_survival_prediction_elle(self, checker):
        result, modified = checker.check("Elle va s'en sortir rapidement.")
        assert modified is True

    def test_dosage_rejected(self, checker):
        result, modified = checker.check(
            "Le médecin prescrit 500mg par jour de ce traitement."
        )
        assert modified is True

    def test_direct_treatment_instruction_rejected(self, checker):
        result, modified = checker.check("Donnez-lui ce médicament immédiatement.")
        assert modified is True

    def test_recommendation_rejected(self, checker):
        result, modified = checker.check(
            "Je vous conseille de prendre ce traitement."
        )
        assert modified is True

    def test_should_take_rejected(self, checker):
        result, modified = checker.check("Vous devriez prendre ce médicament.")
        assert modified is True

    def test_modified_answer_is_non_empty_string(self, checker):
        result, _ = checker.check("Il va survivre.")
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_diagnostic_statement_rejected(self, checker):
        result, modified = checker.check(
            "C'est probablement une pneumonie bactérienne."
        )
        assert modified is True
