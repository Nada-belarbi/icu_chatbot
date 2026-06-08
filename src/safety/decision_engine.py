# DecisionEngine
#
# Responsibilities:
#   - Combine RuleFilter output + CamemBERT (label, score) into a single decision
#   - Possible outputs: allow_rag | refuse | redirect | ask_reformulation

from config.safety_config import (
    LABEL_EMERGENCY,
    LABEL_MEDICAL_ADVICE,
    LABEL_OUT_OF_SCOPE,
)
from config.settings import CAMEMBERT_CONFIDENCE_THRESHOLD
from src.safety.rule_filter import RuleFilter
from src.safety.query_classifier import BaseQueryClassifier

# Decision constants
DECISION_ALLOW_RAG = "allow_rag"
DECISION_REFUSE = "refuse"
DECISION_REDIRECT = "redirect"
DECISION_ASK_REFORMULATION = "ask_reformulation"

_REFUSE_LABELS = {LABEL_MEDICAL_ADVICE, LABEL_OUT_OF_SCOPE}


class DecisionEngine:
    """
    Single entry point for the safety routing logic.

    Decision order
    --------------
    1. RuleFilter — fast keyword check
       - ``urgency``  → ``redirect``  (classifier skipped)
       - any other blocked category → ``refuse``  (classifier skipped)
    2. QueryClassifier — semantic understanding
       - score < threshold → ``ask_reformulation``
       - label ``urgence`` → ``redirect``
       - label in {``avis_medical_personnalise``, ``hors_perimetre``} → ``refuse``
       - label in {``information_generale``, ``sensible``} → ``allow_rag``
    3. Default fallback → ``refuse``  (safety-first)

    Parameters
    ----------
    rule_filter:
        Pre-instantiated ``RuleFilter``.
    classifier:
        Any ``BaseQueryClassifier`` implementation
        (``MockQueryClassifier`` during dev, ``CamembertQueryClassifier`` in prod).
    confidence_threshold:
        Minimum classifier score to trust the label.
        Defaults to the value in ``.env`` / ``settings.py``.
    """

    def __init__(
        self,
        rule_filter: RuleFilter,
        classifier: BaseQueryClassifier,
        confidence_threshold: float = CAMEMBERT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._rule_filter = rule_filter
        self._classifier = classifier
        self._threshold = confidence_threshold

    def decide(self, text: str) -> str:
        """
        Route *text* to one of the four decision constants.

        Parameters
        ----------
        text:
            Cleaned (and optionally context-prefixed) user input.

        Returns
        -------
        str
            One of: ``allow_rag``, ``refuse``, ``redirect``,
            ``ask_reformulation``.
        """
        # --- Step 1: rule-based pre-filter ---
        category, blocked = self._rule_filter.filter(text)
        if blocked:
            if category == "urgency":
                return DECISION_REDIRECT
            return DECISION_REFUSE

        # --- Step 2: semantic classifier ---
        label, score = self._classifier.classify(text)

        if score < self._threshold:
            return DECISION_ASK_REFORMULATION

        if label == LABEL_EMERGENCY:
            return DECISION_REDIRECT

        if label in _REFUSE_LABELS:
            return DECISION_REFUSE

        # information_generale or sensible with sufficient confidence
        if score >= self._threshold:
            return DECISION_ALLOW_RAG

        # Safety-first default
        return DECISION_REFUSE

#   - Decision logic:
#       urgency (any source)         → redirect
#       low confidence score         → ask_reformulation
#       information_generale + high  → allow_rag
#       avis_medical/hors_perimetre  → refuse
#   - Confidence threshold read from config/settings.py
