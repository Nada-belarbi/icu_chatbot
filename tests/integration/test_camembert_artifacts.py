from pathlib import Path

import pytest
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.label_schema import LABEL_ORDER

MODEL_DIR = Path("models/camembert_classifier")


@pytest.mark.skipif(
    not (MODEL_DIR / "config.json").exists(),
    reason="CamemBERT model has not been trained yet.",
)
def test_trained_camembert_artifact_loads():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    assert tokenizer is not None
    assert model.config.num_labels == len(LABEL_ORDER)


@pytest.mark.skipif(
    not (MODEL_DIR / "config.json").exists(),
    reason="CamemBERT model has not been trained yet.",
)
def test_trained_camembert_label_mapping_matches_schema():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]

    assert labels == LABEL_ORDER
