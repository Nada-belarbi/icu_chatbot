"""Reconstruit de façon reproductible les jeux de données d'augmentation
manuelle utilisés par les modèles baseline TF-IDF "augmentés".

Ce script est l'équivalent CLI, scriptable et reproductible, des cellules de
préparation de données du notebook `03_baseline_augmented_rulefilter.ipynb`.
Il réutilise TEL QUEL le code déjà testé de
`src/evaluation/baseline_augmented_utils.py` (`build_manual_augmentation`,
`build_manual_safety_tests`, `build_train_augmented`) — aucune logique n'est
réinventée ici.

Sécurité : `data/processed/train.csv`, `val.csv` et `test.csv` (les splits
d'origine, jamais retouchés) sont vérifiés par hash MD5 avant et après
exécution ; le script s'arrête avec une erreur s'ils ont été modifiés.

Fichiers (re)générés (jamais les splits d'origine) :
  - data/augmentation/manual_train_examples.csv
  - data/augmentation/manual_safety_tests.csv
  - data/processed/train_augmented.csv

Usage:
    python scripts/build_dataset.py                # écrit les 3 fichiers
    python scripts/build_dataset.py --check-only    # valide sans rien écrire
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.baseline_augmented_utils import (  # noqa: E402
    build_manual_augmentation,
    build_manual_safety_tests,
    build_train_augmented,
    file_md5,
    load_split,
)

DATA_PROCESSED = ROOT / "data" / "processed"
DATA_AUG = ROOT / "data" / "augmentation"

TRAIN_PATH = DATA_PROCESSED / "train.csv"
VAL_PATH = DATA_PROCESSED / "val.csv"
TEST_PATH = DATA_PROCESSED / "test.csv"
MANUAL_AUG_PATH = DATA_AUG / "manual_train_examples.csv"
MANUAL_TEST_PATH = DATA_AUG / "manual_safety_tests.csv"
TRAIN_AUG_PATH = DATA_PROCESSED / "train_augmented.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruit train_augmented.csv et les fichiers d'augmentation manuelle."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Calcule et affiche les tailles sans rien écrire sur le disque.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for required in (TRAIN_PATH, VAL_PATH, TEST_PATH):
        if not required.exists():
            raise SystemExit(f"Fichier requis manquant : {required}")

    train_hash_before = file_md5(TRAIN_PATH)
    val_hash_before = file_md5(VAL_PATH)
    test_hash_before = file_md5(TEST_PATH)

    train_df = load_split(TRAIN_PATH, "train")
    manual_aug_df = build_manual_augmentation()
    manual_test_df = build_manual_safety_tests()
    train_aug_df = build_train_augmented(train_df, manual_aug_df)

    print(f"train.csv                 : {len(train_df):,} lignes")
    print(f"augmentation manuelle      : {len(manual_aug_df):,} lignes")
    print(f"tests de sécurité manuels  : {len(manual_test_df):,} lignes")
    print(f"train_augmented.csv (total): {len(train_aug_df):,} lignes")

    if args.check_only:
        print("--check-only : aucun fichier écrit.")
    else:
        DATA_AUG.mkdir(parents=True, exist_ok=True)
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        manual_aug_df.to_csv(MANUAL_AUG_PATH, index=False)
        manual_test_df.to_csv(MANUAL_TEST_PATH, index=False)
        train_aug_df.to_csv(TRAIN_AUG_PATH, index=False)
        print(f"Écrit : {MANUAL_AUG_PATH}")
        print(f"Écrit : {MANUAL_TEST_PATH}")
        print(f"Écrit : {TRAIN_AUG_PATH}")

    # Garantie : ce script ne doit jamais modifier les splits d'origine.
    assert file_md5(TRAIN_PATH) == train_hash_before, "train.csv a été modifié !"
    assert file_md5(VAL_PATH) == val_hash_before, "val.csv a été modifié !"
    assert file_md5(TEST_PATH) == test_hash_before, "test.csv a été modifié !"
    print("OK : train.csv / val.csv / test.csv non modifiés.")


if __name__ == "__main__":
    main()
