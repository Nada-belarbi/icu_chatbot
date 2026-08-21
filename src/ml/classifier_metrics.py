from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config.label_schema import CRITICAL_PAIRS, LABEL_ORDER


def count_critical_errors(y_true, y_pred) -> int:
    """Count unsafe misroutes, such as emergency predicted as general."""
    return int(sum((true, pred) in CRITICAL_PAIRS for true, pred in zip(y_true, y_pred)))


def compute_classifier_metrics(y_true, y_pred) -> dict:
    """Compute global, per-class, and safety-focused metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="weighted",
        zero_division=0,
    )
    p_cls, r_cls, f_cls, s_cls = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average=None,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "critical_errors": count_critical_errors(y_true, y_pred),
    }
    for idx, label in enumerate(LABEL_ORDER):
        metrics[f"precision_{label}"] = float(p_cls[idx])
        metrics[f"recall_{label}"] = float(r_cls[idx])
        metrics[f"f1_{label}"] = float(f_cls[idx])
        metrics[f"support_{label}"] = int(s_cls[idx])
    return metrics


def classification_report_df(y_true, y_pred) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).T


def confusion_matrix_df(y_true, y_pred) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    return pd.DataFrame(matrix, index=LABEL_ORDER, columns=LABEL_ORDER)


def critical_errors_df(
    texts,
    y_true,
    y_pred,
    topics=None,
    sources=None,
    confidences=None,
) -> pd.DataFrame:
    rows = []
    for idx, (text, true_label, pred_label) in enumerate(zip(texts, y_true, y_pred)):
        if (true_label, pred_label) not in CRITICAL_PAIRS:
            continue
        rows.append(
            {
                "text": text,
                "true_label": true_label,
                "predicted_label": pred_label,
                "topic": topics[idx] if topics is not None else "",
                "source": sources[idx] if sources is not None else "",
                "confidence": confidences[idx] if confidences is not None else None,
            }
        )
    return pd.DataFrame(rows)
