"""Canonical label schema for ICU query classification.

This module is intentionally independent from the runtime safety rules so the
same label contract can be reused by datasets, training scripts, notebooks, and
future runtime classifiers.
"""

from __future__ import annotations

LABEL_GENERAL = "information_generale"
LABEL_SENSITIVE = "sensible_reformulable"
LABEL_MEDICAL_ADVICE = "avis_medical_personnalise"
LABEL_EMERGENCY = "urgence"
LABEL_OUT_OF_SCOPE = "hors_perimetre"

LABEL_ORDER = [
    LABEL_GENERAL,
    LABEL_SENSITIVE,
    LABEL_MEDICAL_ADVICE,
    LABEL_EMERGENCY,
    LABEL_OUT_OF_SCOPE,
]

LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

CRITICAL_PAIRS = {
    (LABEL_EMERGENCY, LABEL_GENERAL),
    (LABEL_EMERGENCY, LABEL_SENSITIVE),
    (LABEL_EMERGENCY, LABEL_MEDICAL_ADVICE),
    (LABEL_MEDICAL_ADVICE, LABEL_GENERAL),
    (LABEL_MEDICAL_ADVICE, LABEL_SENSITIVE),
    (LABEL_OUT_OF_SCOPE, LABEL_GENERAL),
    (LABEL_OUT_OF_SCOPE, LABEL_SENSITIVE),
}

REQUIRED_DATASET_COLUMNS = ["text", "label", "topic", "source"]
