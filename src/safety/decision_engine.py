# DecisionEngine
#
# Responsibilities:
#   - Combine RuleFilter output + CamemBERT (label, score) into a single decision
#   - Possible outputs: allow_rag | refuse | redirect | ask_reformulation

from config.safety_config import (
    LABEL_EMERGENCY,
    LABEL_GENERAL,
    LABEL_MEDICAL_ADVICE,
    LABEL_OUT_OF_SCOPE,
    LABEL_SENSITIVE,
)
from config.settings import CAMEMBERT_CONFIDENCE_THRESHOLD
from src.safety.rule_filter import RuleFilter
from src.safety.query_classifier import BaseQueryClassifier

# Decision constants
DECISION_ALLOW_RAG = "allow_rag"
DECISION_REFUSE = "refuse"          # avis médical personnalisé
DECISION_REFUSE_OOS = "refuse_oos"   # hors périmètre
DECISION_REDIRECT = "redirect"
DECISION_ASK_REFORMULATION = "ask_reformulation"

_REFUSE_LABELS = {LABEL_MEDICAL_ADVICE}
_REFUSE_OOS_LABELS = {LABEL_OUT_OF_SCOPE}
_ALLOW_LABELS = {LABEL_GENERAL, LABEL_SENSITIVE}


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
       - label in {``information_generale``, ``sensible_reformulable``} → ``allow_rag``
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

    def decide(self, text: str, context_string: str = "") -> str:
        """
        Route *text* to one of the four decision constants.

        Parameters
        ----------
        text:
            Cleaned user input (output of ``TextCleaner.clean()``).
        context_string:
            Optional conversation history prefix produced by
            ``ConversationMemoryManager.get_context_string()``.
            When non-empty it is prepended to *text* before calling the
            classifier, so ambiguous follow-up questions are classified
            with their conversation context.  The rule filter always
            operates on raw *text* only (no context contamination).

        Returns
        -------
        str
            One of: ``allow_rag``, ``refuse``, ``redirect``,
            ``ask_reformulation``.
        """
        # --- Step 1: rule-based pre-filter (on raw text only) ---
        category, blocked = self._rule_filter.filter(text)
        if blocked:
            if category == "urgency":
                return DECISION_REDIRECT
            if category == "out_of_scope":
                return DECISION_REFUSE_OOS
            return DECISION_REFUSE

        # force_allow: clearly educational question, skip CamemBERT
        if category == "force_allow":
            return DECISION_ALLOW_RAG

        # --- Step 2: semantic classifier (with memory context) ---
        classifier_input = (
            f"{context_string}\nUtilisateur: {text}" if context_string else text
        )
        label, score = self._classifier.classify(classifier_input)

        if score < self._threshold:
            return DECISION_ASK_REFORMULATION

        if label == LABEL_EMERGENCY:
            return DECISION_REDIRECT

        if label in _REFUSE_LABELS:
            return DECISION_REFUSE

        if label in _REFUSE_OOS_LABELS:
            return DECISION_REFUSE_OOS

        if label in _ALLOW_LABELS:
            return DECISION_ALLOW_RAG

        # Safety-first default
        return DECISION_REFUSE

#   - Decision logic:
#       urgency (any source)         → redirect
#       low confidence score         → ask_reformulation
#       information_generale + high  → allow_rag
#       avis_medical/hors_perimetre  → refuse
#   - Confidence threshold read from config/settings.py
