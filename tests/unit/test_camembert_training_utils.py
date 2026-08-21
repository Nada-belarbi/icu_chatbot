from pathlib import Path

import pandas as pd
import pytest

from src.ml.camembert_trainer import (
    CamembertTrainingConfig,
    compute_class_weights,
    load_csv_split,
)


def test_training_config_defaults_are_strong_but_reasonable():
    cfg = CamembertTrainingConfig()

    assert cfg.model_name == "camembert-base"
    assert cfg.max_length == 192
    assert cfg.epochs >= 4
    assert cfg.use_weighted_loss is True


def test_load_csv_split_maps_labels(tmp_path: Path):
    path = tmp_path / "sample.csv"
    pd.DataFrame(
        [
            {
                "text": "Qu'est-ce qu'un respirateur ?",
                "label": "information_generale",
                "topic": "ventilation",
                "source": "unit",
            },
            {
                "text": "Mon proche ne respire plus",
                "label": "urgence",
                "topic": "monitoring",
                "source": "unit",
            },
        ]
    ).to_csv(path, index=False)

    df = load_csv_split(path, "unit")

    assert "labels" in df.columns
    assert df["labels"].tolist() == [0, 3]


def test_load_csv_split_rejects_unknown_label(tmp_path: Path):
    path = tmp_path / "bad.csv"
    pd.DataFrame(
        [
            {
                "text": "test",
                "label": "sensible",
                "topic": "x",
                "source": "unit",
            }
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="unknown labels"):
        load_csv_split(path, "unit")


def test_compute_class_weights_returns_one_weight_per_label():
    weights = compute_class_weights([0, 0, 1, 2, 3, 4])

    assert len(weights) == 5
