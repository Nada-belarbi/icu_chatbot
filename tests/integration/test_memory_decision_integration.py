# Integration test — Phase 8 checkpoint
# Verifies that ConversationMemoryManager.get_context_string() is correctly
# prepended to the classifier input inside DecisionEngine.decide().
#
# This test uses a spy classifier that records the text it receives,
# so we can assert the context was included.

import pytest
from src.safety.decision_engine import DecisionEngine, DECISION_ALLOW_RAG
from src.safety.rule_filter import RuleFilter
from src.safety.query_classifier import BaseQueryClassifier, MockQueryClassifier
from src.memory.conversation_memory import ConversationMemoryManager
from config.safety_config import LABEL_GENERAL


class SpyClassifier(BaseQueryClassifier):
    """Records the last text passed to classify() for assertion."""

    def __init__(self):
        self.last_text = ""

    def classify(self, text: str) -> tuple[str, float]:
        self.last_text = text
        return (LABEL_GENERAL, 0.95)


class TestMemoryIntegration:
    def test_context_prepended_to_classifier_input(self):
        spy = SpyClassifier()
        engine = DecisionEngine(
            rule_filter=RuleFilter(),
            classifier=spy,
            confidence_threshold=0.70,
        )
        mem = ConversationMemoryManager(history=[], max_turns=3)
        mem.add_user_message("mon père est en réanimation")
        mem.add_assistant_message("Je comprends votre inquiétude.")

        context = mem.get_context_string()
        # Neutral question: no blocked rule, no safe-allow pattern → classifier called
        engine.decide("les soins intensifs en reanimation", context_string=context)

        assert "mon père est en réanimation" in spy.last_text
        assert "les soins intensifs en reanimation" in spy.last_text

    def test_context_appears_before_current_question(self):
        spy = SpyClassifier()
        engine = DecisionEngine(
            rule_filter=RuleFilter(),
            classifier=spy,
            confidence_threshold=0.70,
        )
        mem = ConversationMemoryManager(history=[], max_turns=3)
        mem.add_user_message("il est sous respirateur")
        mem.add_assistant_message("Le respirateur assiste la respiration.")

        context = mem.get_context_string()
        # Neutral question — doesn't trigger rules or safe-allow bypass
        engine.decide("la surveillance medicale en reanimation", context_string=context)

        pos_context = spy.last_text.find("il est sous respirateur")
        pos_question = spy.last_text.find("la surveillance medicale en reanimation")
        assert pos_context < pos_question

    def test_empty_context_passes_text_unchanged(self):
        spy = SpyClassifier()
        engine = DecisionEngine(
            rule_filter=RuleFilter(),
            classifier=spy,
            confidence_threshold=0.70,
        )
        # Neutral question — no rule match, no safe-allow bypass → classifier called
        engine.decide("les soins intensifs en reanimation", context_string="")
        assert spy.last_text == "les soins intensifs en reanimation"

    def test_context_limited_to_max_turns(self):
        spy = SpyClassifier()
        engine = DecisionEngine(
            rule_filter=RuleFilter(),
            classifier=spy,
            confidence_threshold=0.70,
        )
        mem = ConversationMemoryManager(history=[], max_turns=2)
        # Add 3 turns — oldest should be dropped
        mem.add_user_message("Q0 très ancien")
        mem.add_assistant_message("R0")
        mem.add_user_message("Q1 récent")
        mem.add_assistant_message("R1")
        mem.add_user_message("Q2 le plus récent")
        mem.add_assistant_message("R2")

        context = mem.get_context_string()
        engine.decide("nouvelle question", context_string=context)

        assert "Q0 très ancien" not in spy.last_text
        assert "Q1 récent" in spy.last_text or "Q2 le plus récent" in spy.last_text

    def test_rule_filter_uses_raw_text_not_context(self):
        """Urgency in context history must NOT trigger redirect for a safe question."""
        engine = DecisionEngine(
            rule_filter=RuleFilter(),
            classifier=MockQueryClassifier.as_general(),
            confidence_threshold=0.70,
        )
        mem = ConversationMemoryManager(history=[], max_turns=3)
        # Previous turn mentioned urgency — but current question is safe
        mem.add_user_message("il ne respirait plus mais il va mieux maintenant")
        mem.add_assistant_message("Je suis soulagé d'apprendre cela.")

        context = mem.get_context_string()
        # Safe question — rule filter should NOT be triggered by the old urgency in context
        decision = engine.decide("quels sont les horaires de visite ?", context_string=context)
        assert decision == DECISION_ALLOW_RAG
