# CamembertQueryClassifier
#
# Responsibilities:
#   - Load fine-tuned CamemBERT model from models/camembert_classifier/
#   - Classify a cleaned query into one of:
#     information_generale | sensible_reformulable | avis_medical_personnalise | urgence | hors_perimetre
#
# This file defines:
#   - BaseQueryClassifier       : interface every classifier must satisfy
#   - MockQueryClassifier       : test double used during phases 6-9 (no model required)
#   - CamembertQueryClassifier  : real classifier (CamemBERT fine-tuned)
#   - SVMQueryClassifier        : TF-IDF + LinearSVC baseline (specialist for hors_perimetre)
#   - HybridQueryClassifier     : CamemBERT primary + SVM safety-net for hors_perimetre

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.safety_config import LABEL_GENERAL, LABEL_OUT_OF_SCOPE, ALL_LABELS
from config.settings import CLASSIFIER_PATH
from config.label_schema import ID_TO_LABEL

_BASELINE_SVM_PATH = Path(CLASSIFIER_PATH).parent / "baseline_tfidf_linear_svm_augmented"


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
    Loads a local model from ``CLASSIFIER_PATH`` and serves inference only.
    """

    def __init__(self, model_dir=None, max_length: int = 192) -> None:
        self._model_dir = model_dir or CLASSIFIER_PATH
        self._max_length = max_length

        if not self._model_dir.exists():
            raise FileNotFoundError(
                f"Camembert model directory not found: {self._model_dir}"
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._model_dir)
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def _id_to_label(self, pred_id: int) -> str:
        # Prefer runtime model mapping if present, then fallback to shared schema.
        model_id2label = getattr(self._model.config, "id2label", None) or {}
        if pred_id in model_id2label:
            return str(model_id2label[pred_id])
        if str(pred_id) in model_id2label:
            return str(model_id2label[str(pred_id)])
        return ID_TO_LABEL[pred_id]

    def classify(self, text: str) -> tuple[str, float]:
        encoded = self._tokenizer(
            [text],
            truncation=True,
            padding=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = self._model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)

        confidence, pred_ids = torch.max(probs, dim=-1)
        pred_id = int(pred_ids.item())
        label = self._id_to_label(pred_id)

        if label not in ALL_LABELS:
            # Safety-first fallback to a known safe contract.
            return (LABEL_GENERAL, float(confidence.item()))

        return (label, float(confidence.item()))

#       information_generale | sensible | avis_medical_personnalise | urgence | hors_perimetre
#   - Return (label: str, confidence_score: float)
#   - In early development: swap with a MockClassifier that returns hardcoded results
#
# NOTE: The mock classifier must have the same interface as the real one


# ---------------------------------------------------------------------------
# SVMQueryClassifier — TF-IDF + LinearSVC baseline (hors_perimetre specialist)
# ---------------------------------------------------------------------------
class SVMQueryClassifier(BaseQueryClassifier):  # pragma: no cover
    """
    Baseline TF-IDF + LinearSVC classifier.

    Used as a safety-net specialist for the ``hors_perimetre`` class inside
    ``HybridQueryClassifier``.  Can also be used standalone.

    Parameters
    ----------
    model_path:
        Directory containing ``model.joblib`` and ``label_encoder.joblib``.
        Defaults to ``models/baseline_tfidf_linear_svm_augmented``.
    """

    def __init__(self, model_path: Path | str | None = None) -> None:
        import joblib

        self._model_path = Path(model_path) if model_path else _BASELINE_SVM_PATH
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"SVM model directory not found: {self._model_path}"
            )
        self._pipeline = joblib.load(self._model_path / "model.joblib")
        self._label_encoder = joblib.load(self._model_path / "label_encoder.joblib")

    def classify(self, text: str) -> tuple[str, float]:
        """Return ``(label, confidence)`` using the SVM pipeline.

        LinearSVC does not produce calibrated probabilities; confidence is
        derived from the decision-function distance (sigmoid-normalized) for
        the predicted class so it stays in ``[0, 1]``.
        """
        import numpy as np

        raw = self._pipeline.predict([text])[0]
        # model.predict() may return string labels directly or encoded integers
        if isinstance(raw, str):
            label = raw
        else:
            label = self._label_encoder.inverse_transform([int(raw)])[0]

        # Confidence: sigmoid of the max decision-function score
        try:
            decision = self._pipeline.decision_function([text])[0]
            max_score = float(decision.max())
            confidence = float(1.0 / (1.0 + np.exp(-max_score)))
        except Exception:
            confidence = 0.80  # safe fallback

        if label not in ALL_LABELS:
            return (LABEL_GENERAL, confidence)
        return (label, confidence)


# ---------------------------------------------------------------------------
# HybridQueryClassifier — CamemBERT + SVM (conflict-free OR logic)
# ---------------------------------------------------------------------------
class HybridQueryClassifier(BaseQueryClassifier):  # pragma: no cover
    """
    Production classifier that combines CamemBERT and the SVM baseline.

    **Strategy (conflict-free)**:
    - CamemBERT is the primary classifier for all five labels.
    - The SVM baseline is consulted exclusively for ``hors_perimetre``
      detection, which CamemBERT systematically misses (0 % recall).
    - If the SVM predicts ``hors_perimetre``, that label overrides
      CamemBERT regardless of CamemBERT's prediction.
    - In all other cases, CamemBERT's output is used unchanged.

    This avoids any conflict because the SVM only intervenes for one class
    and the two classifiers never disagree on the other four labels.

    Parameters
    ----------
    camembert_dir:
        CamemBERT model directory.  Defaults to ``CLASSIFIER_PATH``.
    svm_dir:
        SVM model directory.  Defaults to
        ``models/baseline_tfidf_linear_svm_augmented``.
    """

    def __init__(
        self,
        camembert_dir: Path | str | None = None,
        svm_dir: Path | str | None = None,
    ) -> None:
        self._camembert = CamembertQueryClassifier(model_dir=camembert_dir)
        self._svm = SVMQueryClassifier(model_path=svm_dir)

    def classify(self, text: str) -> tuple[str, float]:
        # Step 1: CamemBERT primary classification
        cam_label, cam_conf = self._camembert.classify(text)

        # Fast path: CamemBERT already detected hors_perimetre
        if cam_label == LABEL_OUT_OF_SCOPE:
            return cam_label, cam_conf

        # Step 2: SVM safety-net exclusively for hors_perimetre
        svm_label, svm_conf = self._svm.classify(text)
        if svm_label == LABEL_OUT_OF_SCOPE:
            return LABEL_OUT_OF_SCOPE, svm_conf

        # Step 3: No hors_perimetre signal — trust CamemBERT
        return cam_label, cam_conf
