# Unit tests — RuleFilter
# Cases:
#   - prognosis query → (prognosis, True)
#   - treatment query → (treatment, True)
#   - urgency query → (urgency, True)
#   - general question → (allowed, False)
#   - urgency wins over other categories in same query
#   - case-insensitive matching

import pytest
from src.safety.rule_filter import RuleFilter


@pytest.fixture
def rf():
    return RuleFilter()


class TestRuleFilterBlocked:
    def test_prognosis_blocked(self, rf):
        cat, blocked = rf.filter("Est-ce que mon père va survivre ?")
        assert cat == "prognosis"
        assert blocked is True

    def test_treatment_blocked(self, rf):
        cat, blocked = rf.filter("Quel médicament donner ?")
        assert cat == "treatment"
        assert blocked is True

    def test_urgency_blocked(self, rf):
        cat, blocked = rf.filter("Il ne respire plus")
        assert cat == "urgency"
        assert blocked is True

    def test_diagnosis_blocked(self, rf):
        cat, blocked = rf.filter("Est-ce que j'ai une infection pulmonaire ?")
        assert cat == "diagnosis"
        assert blocked is True

    def test_personal_data_blocked(self, rf):
        cat, blocked = rf.filter("Pouvez-vous accéder à mon dossier médical ?")
        assert cat == "personal_data"
        assert blocked is True


class TestRuleFilterAllowed:
    def test_general_question_allowed(self, rf):
        cat, blocked = rf.filter("Qu'est-ce qu'un respirateur ?")
        assert cat == "allowed"
        assert blocked is False

    def test_information_question_allowed(self, rf):
        cat, blocked = rf.filter("Comment fonctionne la réanimation ?")
        assert cat == "allowed"
        assert blocked is False

    def test_visitor_question_allowed(self, rf):
        cat, blocked = rf.filter("Quels sont les horaires de visite ?")
        assert cat == "allowed"
        assert blocked is False


class TestRuleFilterPriority:
    def test_urgency_wins_over_treatment(self, rf):
        # Contains both urgency and treatment keywords
        cat, blocked = rf.filter("arrêt cardiaque, dois-je donner un médicament ?")
        assert cat == "urgency"
        assert blocked is True

    def test_urgency_wins_over_prognosis(self, rf):
        cat, blocked = rf.filter("il est inconscient, va-t-il survivre ?")
        assert cat == "urgency"
        assert blocked is True


class TestRuleFilterCaseInsensitive:
    def test_uppercase_urgency(self, rf):
        cat, blocked = rf.filter("URGENCE MAINTENANT")
        assert cat == "urgency"
        assert blocked is True

    def test_mixed_case_treatment(self, rf):
        cat, blocked = rf.filter("Puis-Je Arrêter le Traitement ?")
        assert cat == "treatment"
        assert blocked is True

#   - "Qu'est-ce qu'un respirateur ?" → allowed
#   - Multiple rule categories in one query
