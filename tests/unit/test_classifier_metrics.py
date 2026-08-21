from src.ml.classifier_metrics import (
    classification_report_df,
    compute_classifier_metrics,
    confusion_matrix_df,
    count_critical_errors,
    critical_errors_df,
)


def test_count_critical_errors():
    y_true = ["urgence", "avis_medical_personnalise", "information_generale"]
    y_pred = ["information_generale", "sensible_reformulable", "information_generale"]

    assert count_critical_errors(y_true, y_pred) == 2


def test_compute_classifier_metrics_contains_safety_fields():
    y_true = ["urgence", "avis_medical_personnalise", "information_generale"]
    y_pred = ["information_generale", "sensible_reformulable", "information_generale"]

    metrics = compute_classifier_metrics(y_true, y_pred)

    assert "macro_f1" in metrics
    assert "recall_urgence" in metrics
    assert metrics["critical_errors"] == 2


def test_report_and_confusion_matrix_shapes():
    y_true = ["urgence", "avis_medical_personnalise", "information_generale"]
    y_pred = ["information_generale", "sensible_reformulable", "information_generale"]

    report = classification_report_df(y_true, y_pred)
    matrix = confusion_matrix_df(y_true, y_pred)

    assert "urgence" in report.index
    assert matrix.shape == (5, 5)


def test_critical_errors_df_keeps_context():
    texts = ["alerte", "scanner", "general"]
    y_true = ["urgence", "avis_medical_personnalise", "information_generale"]
    y_pred = ["information_generale", "sensible_reformulable", "information_generale"]

    df = critical_errors_df(
        texts=texts,
        y_true=y_true,
        y_pred=y_pred,
        topics=["monitoring", "imaging", "general"],
        sources=["manual", "manual", "manual"],
        confidences=[0.9, 0.8, 0.7],
    )

    assert len(df) == 2
    assert set(["text", "true_label", "predicted_label", "topic", "source", "confidence"]).issubset(df.columns)
