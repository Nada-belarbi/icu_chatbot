"""Entraîne de façon reproductible les modèles baseline TF-IDF "augmentés"
(LogisticRegression OvR et LinearSVC, avec filtre par règles au moment de la
prédiction) — équivalent CLI de la section entraînement du notebook
`03_baseline_augmented_rulefilter.ipynb`. Réutilise TEL QUEL le code déjà
testé de `src/evaluation/baseline_augmented_utils.py`
(`make_augmented_models`, `save_augmented_model`) — mêmes hyperparamètres,
même `RANDOM_STATE`, aucune logique réinventée.

ATTENTION : par défaut, ce script écrit dans
`models/baseline_tfidf_logreg_augmented/` et
`models/baseline_tfidf_linear_svm_augmented/`, qui sont les modèles
ACTUELLEMENT chargés en production par `SVMQueryClassifier`
(`src/safety/query_classifier.py`). Utilisez `--output-suffix` pour écrire
dans des dossiers séparés sans toucher aux modèles de production.

Prérequis : `data/processed/train_augmented.csv` doit exister (générez-le
avec `python scripts/build_dataset.py`).

Usage:
    python scripts/train_baseline.py --output-suffix _rebuild
    python scripts/train_baseline.py   # écrase les modèles de production
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.baseline_augmented_utils import (  # noqa: E402
    load_split,
    make_augmented_models,
    save_augmented_model,
)

DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

TRAIN_AUG_PATH = DATA_PROCESSED / "train_augmented.csv"
VAL_PATH = DATA_PROCESSED / "val.csv"
TEST_PATH = DATA_PROCESSED / "test.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraîne les baselines TF-IDF augmentées.")
    parser.add_argument(
        "--output-suffix",
        default="",
        help=(
            "Suffixe ajouté aux dossiers de sortie (ex: '_rebuild' -> "
            "models/baseline_tfidf_logreg_augmented_rebuild/). Vide par "
            "défaut = écrase les modèles de production."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not TRAIN_AUG_PATH.exists():
        raise SystemExit(
            f"{TRAIN_AUG_PATH} introuvable. Lancez d'abord "
            "`python scripts/build_dataset.py` pour le générer."
        )
    if not VAL_PATH.exists() or not TEST_PATH.exists():
        raise SystemExit("data/processed/val.csv ou test.csv introuvable.")

    train_aug_df = load_split(TRAIN_AUG_PATH, "train_augmented")
    x_train_aug = train_aug_df["text"].astype(str)
    y_train_aug = train_aug_df["label"]

    (
        logreg,
        svm,
        logreg_vectorizer_params,
        logreg_classifier_params,
        svm_vectorizer_params,
        svm_classifier_params,
    ) = make_augmented_models()

    print(f"Entraînement LogReg (OvR) sur {len(train_aug_df):,} exemples...")
    logreg.fit(x_train_aug, y_train_aug)

    print(f"Entraînement LinearSVC sur {len(train_aug_df):,} exemples...")
    svm.fit(x_train_aug, y_train_aug)

    logreg_dir = MODELS_DIR / f"baseline_tfidf_logreg_augmented{args.output_suffix}"
    svm_dir = MODELS_DIR / f"baseline_tfidf_linear_svm_augmented{args.output_suffix}"

    save_augmented_model(
        logreg,
        logreg_dir,
        "logreg_ovr_tfidf",
        logreg_vectorizer_params,
        logreg_classifier_params,
        TRAIN_AUG_PATH,
        VAL_PATH,
        TEST_PATH,
    )
    save_augmented_model(
        svm,
        svm_dir,
        "linear_svc_tfidf",
        svm_vectorizer_params,
        svm_classifier_params,
        TRAIN_AUG_PATH,
        VAL_PATH,
        TEST_PATH,
    )

    print(f"Modèle LogReg sauvegardé    : {logreg_dir}")
    print(f"Modèle LinearSVC sauvegardé : {svm_dir}")


if __name__ == "__main__":
    main()
