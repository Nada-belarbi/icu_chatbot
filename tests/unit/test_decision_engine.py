# Unit tests — DecisionEngine
# All tests use MockQueryClassifier — no model needed.
# Cases:
#   - urgency rule  → redirect  (classifier NOT called)
#   - non-urgency rule block → refuse  (classifier NOT called)
#   - high-confidence information_generale → allow_rag
#   - high-confidence sensible → allow_rag
#   - low confidence (below threshold) → ask_reformulation
#   - avis_medical_personnalise label → refuse
#   - hors_perimetre label → refuse
#   - urgence label from classifier → redirect
#   - rule takes priority over classifier

import pytest
from src.safety.decision_engine import (
    DecisionEngine,
    DECISION_ALLOW_RAG,
    DECISION_REFUSE,
    DECISION_REDIRECT,
    DECISION_ASK_REFORMULATION,
)
from src.safety.rule_filter import RuleFilter
from src.safety.query_classifier import MockQueryClassifier
from config.safety_config import (
    LABEL_GENERAL, LABEL_SENSITIVE, LABEL_MEDICAL_ADVICE,
    LABEL_EMERGENCY, LABEL_OUT_OF_SCOPE,
)

THRESHOLD = 0.70


def make_engine(label=LABEL_GENERAL, score=0.95):
    return DecisionEngine(
        rule_filter=RuleFilter(),
        classifier=MockQueryClassifier(label=label, score=score),
        confidence_threshold=THRESHOLD,
    )


class TestRuleFilterLayer:
    def test_urgency_keyword_returns_redirect(self):
        engine = make_engine()
        assert engine.decide("il ne respire plus") == DECISION_REDIRECT

    def test_urgency_beats_classifier(self):
        engine = make_engine(label=LABEL_GENERAL, score=0.99)
        assert engine.decide("urgence arrêt cardiaque") == DECISION_REDIRECT

    def test_prognosis_keyword_returns_refuse(self):
        engine = make_engine()
        assert engine.decide("va-t-il survivre ?") == DECISION_REFUSE

    def test_treatment_keyword_returns_refuse(self):
        engine = make_engine()
        assert engine.decide("quel médicament donner ?") == DECISION_REFUSE

    def test_diagnosis_keyword_returns_refuse(self):
        engine = make_engine()
        assert engine.decide("est-ce que j'ai une infection ?") == DECISION_REFUSE


class TestClassifierLayer:
    def test_information_generale_high_confidence_allows_rag(self):
        engine = make_engine(label=LABEL_GENERAL, score=0.95)
        assert engine.decide("qu'est-ce qu'un respirateur ?") == DECISION_ALLOW_RAG

    def test_sensible_high_confidence_allows_rag(self):
        engine = make_engine(label=LABEL_SENSITIVE, score=0.85)
        assert engine.decide("pourquoi la sédation ?") == DECISION_ALLOW_RAG

    def test_medical_advice_label_returns_refuse(self):
        engine = make_engine(label=LABEL_MEDICAL_ADVICE, score=0.92)
        assert engine.decide("c'est grave ce qu'il a ?") == DECISION_REFUSE

    def test_out_of_scope_label_returns_refuse(self):
        engine = make_engine(label=LABEL_OUT_OF_SCOPE, score=0.88)
        assert engine.decide("quel est le menu ?") == DECISION_REFUSE

    def test_emergency_label_returns_redirect(self):
        engine = make_engine(label=LABEL_EMERGENCY, score=0.97)
        assert engine.decide("il perd connaissance") == DECISION_REDIRECT

    def test_low_confidence_returns_ask_reformulation(self):
        engine = make_engine(label=LABEL_GENERAL, score=0.45)
        assert engine.decide("question ambiguë") == DECISION_ASK_REFORMULATION

    def test_score_exactly_at_threshold_allows_rag(self):
        engine = make_engine(label=LABEL_GENERAL, score=THRESHOLD)
        assert engine.decide("question claire") == DECISION_ALLOW_RAG

    def test_score_just_below_threshold_asks_reformulation(self):
        engine = make_engine(label=LABEL_GENERAL, score=THRESHOLD - 0.01)
        assert engine.decide("question floue") == DECISION_ASK_REFORMULATION


class TestCustomThreshold:
    def test_custom_threshold_respected(self):
        engine = DecisionEngine(
            rule_filter=RuleFilter(),
            classifier=MockQueryClassifier(label=LABEL_GENERAL, score=0.60),
            confidence_threshold=0.50,
        )
        assert engine.decide("question") == DECISION_ALLOW_RAG

    def test_custom_high_threshold_triggers_reformulation(self):
        engine = DecisionEngine(
            rule_filter=RuleFilter(),
            classifier=MockQueryClassifier(label=LABEL_GENERAL, score=0.80),
            confidence_threshold=0.90,
        )
        assert engine.decide("question") == DECISION_ASK_REFORMULATION
#   - avis_medical_personnalise → refuse
#   - hors_perimetre → refuse
