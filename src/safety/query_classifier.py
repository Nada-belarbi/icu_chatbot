# CamembertQueryClassifier
#
# Responsibilities:
#   - Load fine-tuned CamemBERT model from models/camembert_classifier/
#   - Classify a cleaned query into one of:
#     information_generale | sensible | avis_medical_personnalise | urgence | hors_perimetre
#
# This file defines:
#   - BaseQueryClassifier  : interface every classifier must satisfy
#   - MockQueryClassifier  : test double used during phases 6-9 (no model required)
#   - CamembertQueryClassifier : real classifier (implemented in Phase 18)

from __future__ import annotations
from abc import ABC, abstractmethod

from config.safety_config import LABEL_GENERAL, ALL_LABELS


class BaseQueryClassifier(ABC):
    """
    Interface that every query classifier must implement.

    Both ``MockQueryClassifier`` (used in tests and early phases) and
    ``CamembertQueryClassifier`` (used in production) satisfy this contract.
    """

    @abstractmethod
    def classify(self, text: str) -> tuple[str, float]:
        """
        Classify *text* into one of the five safety labels.

        Parameters
        ----------
        text:
            Cleaned (and optionally context-prefixed) user input.

        Returns
        -------
        tuple[str, float]
            ``(label, score)`` where *label* is one of the constants in
            ``safety_config.ALL_LABELS`` and *score* is the confidence in
            ``[0.0, 1.0]``.
        """


class MockQueryClassifier(BaseQueryClassifier):
    """
    Deterministic test double for ``BaseQueryClassifier``.

    Used during phases 6-9 and in all unit / integration tests that should
    not depend on a trained CamemBERT model.

    Parameters
    ----------
    label:
        The label to return on every ``classify()`` call.
        Defaults to ``information_generale``.
    score:
        The confidence score to return.  Defaults to ``0.95``.
    """

    def __init__(
        self,
        label: str = LABEL_GENERAL,
        score: float = 0.95,
    ) -> None:
        if label not in ALL_LABELS:
            raise ValueError(
                f"Invalid label {label!r}. Must be one of: {ALL_LABELS}"
            )
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"Score must be in [0.0, 1.0], got {score}")
        self._label = label
        self._score = score

    def classify(self, text: str) -> tuple[str, float]:
        """Return the configured ``(label, score)`` regardless of *text*."""
        return (self._label, self._score)

    # Convenience factory methods for the most common test scenarios
    @classmethod
    def as_general(cls, score: float = 0.95) -> "MockQueryClassifier":
        from config.safety_config import LABEL_GENERAL
        return cls(label=LABEL_GENERAL, score=score)

    @classmethod
    def as_emergency(cls, score: float = 0.99) -> "MockQueryClassifier":
        from config.safety_config import LABEL_EMERGENCY
        return cls(label=LABEL_EMERGENCY, score=score)

    @classmethod
    def as_medical_advice(cls, score: float = 0.92) -> "MockQueryClassifier":
        from config.safety_config import LABEL_MEDICAL_ADVICE
        return cls(label=LABEL_MEDICAL_ADVICE, score=score)

    @classmethod
    def as_out_of_scope(cls, score: float = 0.88) -> "MockQueryClassifier":
        from config.safety_config import LABEL_OUT_OF_SCOPE
        return cls(label=LABEL_OUT_OF_SCOPE, score=score)

    @classmethod
    def as_low_confidence(cls, score: float = 0.45) -> "MockQueryClassifier":
        """Returns general label with score below the default threshold (0.70)."""
        from config.safety_config import LABEL_GENERAL
        return cls(label=LABEL_GENERAL, score=score)


# ---------------------------------------------------------------------------
# CamembertQueryClassifier — implemented in Phase 18
# ---------------------------------------------------------------------------
class CamembertQueryClassifier(BaseQueryClassifier):  # pragma: no cover
    """
    Production classifier backed by a fine-tuned CamemBERT model.
    Implemented in Phase 18 once the model is trained via notebook 02.
    """

    def classify(self, text: str) -> tuple[str, float]:
        raise NotImplementedError("CamembertQueryClassifier not yet implemented (Phase 18).")

#       information_generale | sensible | avis_medical_personnalise | urgence | hors_perimetre
#   - Return (label: str, confidence_score: float)
#   - In early development: swap with a MockClassifier that returns hardcoded results
#
# NOTE: The mock classifier must have the same interface as the real one
