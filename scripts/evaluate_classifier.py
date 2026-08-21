"""Évalue de façon unifiée tous les classifieurs baseline (originaux et
augmentés) déjà entraînés sous `models/`, sur val / test / tests de sécurité
manuels — équivalent CLI de la section évaluation du notebook
`03_baseline_augmented_rulefilter.ipynb`. Réutilise TEL QUEL le code déjà
testé de `src/evaluation/baseline_augmented_utils.py` (`evaluate_systems`,
`comparison_table`, `save_evaluation_outputs`, `best_system_for_split`).

Ce script NE modifie ET N'entraîne AUCUN modèle : il charge uniquement les
artefacts déjà sauvegardés (`model.joblib`) et écrit des rapports sous
`reports/baselines/`.

Usage:
    python scripts/evaluate_classifier.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.baseline_augmented_utils import (  # noqa: E402
    best_system_for_split,
    build_manual_safety_tests,
    comparison_table,
    evaluate_systems,
    load_split,
    save_evaluation_outputs,
)

DATA_PROCESSED = ROOT / "data" / "processed"
DATA_AUG = ROOT / "data" / "augmentation"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports" / "baselines"

VAL_PATH = DATA_PROCESSED / "val.csv"
TEST_PATH = DATA_PROCESSED / "test.csv"
MANUAL_TEST_PATH = DATA_AUG / "manual_safety_tests.csv"

# name -> (model_dir, use_rulefilter)
MODEL_DIRS = {
    "original_logreg": (MODELS_DIR / "baseline_tfidf_logreg", False),
    "original_linear_svm": (MODELS_DIR / "baseline_tfidf_linear_svm", False),
    "augmented_logreg_rulefilter": (MODELS_DIR / "baseline_tfidf_logreg_augmented", True),
    "augmented_linear_svm_rulefilter": (MODELS_DIR / "baseline_tfidf_linear_svm_augmented", True),
}


def main() -> None:
    systems = {}
    for name, (model_dir, use_rulefilter) in MODEL_DIRS.items():
        model_path = model_dir / "model.joblib"
        if not model_path.exists():
            print(f"(ignoré) {name}: {model_path} introuvable")
            continue
        systems[name] = {"model": joblib.load(model_path), "use_rulefilter": use_rulefilter}

    if not systems:
        raise SystemExit("Aucun modèle baseline trouvé sous models/. Rien à évaluer.")

    if not VAL_PATH.exists() or not TEST_PATH.exists():
        raise SystemExit("data/processed/val.csv ou test.csv introuvable.")

    val_df = load_split(VAL_PATH, "val")
    test_df = load_split(TEST_PATH, "test")
    manual_test_df = (
        load_split(MANUAL_TEST_PATH, "manual_safety")
        if MANUAL_TEST_PATH.exists()
        else build_manual_safety_tests()
    )

    splits = {"val": val_df, "test": test_df, "manual_safety": manual_test_df}

    metrics_df, critical_df, predictions_df = evaluate_systems(systems, splits)
    comparison_df = comparison_table(metrics_df)
    save_evaluation_outputs(REPORTS_DIR, comparison_df, metrics_df, predictions_df, critical_df)

    print(f"Systèmes évalués : {list(systems)}")
    print(comparison_df.to_string(index=False))
    print(f"Meilleur système (val) : {best_system_for_split(metrics_df, 'val')}")
    print(f"Rapports écrits sous   : {REPORTS_DIR}")


if __name__ == "__main__":
    main()
