from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.label_schema import ID_TO_LABEL
from src.ml.classifier_metrics import (
    classification_report_df,
    compute_classifier_metrics,
    confusion_matrix_df,
    critical_errors_df,
)
from src.ml.camembert_trainer import load_csv_split


def predict_camembert(
    model_dir: Path,
    texts: list[str],
    batch_size: int = 16,
    max_length: int = 192,
) -> tuple[list[str], list[float]]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    labels: list[str] = []
    confidences: list[float] = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)

        confidence, pred_ids = torch.max(probs, dim=-1)
        labels.extend([ID_TO_LABEL[int(idx)] for idx in pred_ids.cpu().tolist()])
        confidences.extend([float(score) for score in confidence.cpu().tolist()])

    return labels, confidences


def evaluate_camembert_split(
    model_dir: Path,
    csv_path: Path,
    reports_dir: Path,
    split_name: str,
    batch_size: int = 16,
    max_length: int = 192,
) -> dict:
    df = load_csv_split(csv_path, split_name)
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].astype(str).tolist()

    y_pred, confidences = predict_camembert(
        model_dir=model_dir,
        texts=texts,
        batch_size=batch_size,
        max_length=max_length,
    )

    metrics = compute_classifier_metrics(y_true, y_pred)
    report = classification_report_df(y_true, y_pred)
    matrix = confusion_matrix_df(y_true, y_pred)
    critical = critical_errors_df(
        texts=texts,
        y_true=y_true,
        y_pred=y_pred,
        topics=df["topic"].astype(str).tolist() if "topic" in df else None,
        sources=df["source"].astype(str).tolist() if "source" in df else None,
        confidences=confidences,
    )

    predictions = df.copy()
    predictions["predicted_label"] = y_pred
    predictions["confidence"] = confidences
    predictions["is_correct"] = predictions["label"] == predictions["predicted_label"]

    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{split_name}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report.to_csv(reports_dir / f"{split_name}_classification_report.csv")
    matrix.to_csv(reports_dir / f"{split_name}_confusion_matrix.csv")
    critical.to_csv(reports_dir / f"{split_name}_critical_errors.csv", index=False)
    predictions.to_csv(reports_dir / f"{split_name}_predictions.csv", index=False)
    return metrics


def evaluate_camembert_model(
    model_dir: Path,
    val_path: Path,
    test_path: Path,
    reports_dir: Path,
    batch_size: int = 16,
    max_length: int = 192,
) -> dict[str, dict]:
    return {
        "val": evaluate_camembert_split(
            model_dir=model_dir,
            csv_path=val_path,
            reports_dir=reports_dir,
            split_name="val",
            batch_size=batch_size,
            max_length=max_length,
        ),
        "test": evaluate_camembert_split(
            model_dir=model_dir,
            csv_path=test_path,
            reports_dir=reports_dir,
            split_name="test",
            batch_size=batch_size,
            max_length=max_length,
        ),
    }
