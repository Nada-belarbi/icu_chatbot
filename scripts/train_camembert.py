from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.camembert_trainer import CamembertTrainingConfig, train_camembert


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune CamemBERT classifier.")
    parser.add_argument("--train", default="data/processed/train_augmented.csv")
    parser.add_argument("--val", default="data/processed/val.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--output-dir", default="models/camembert_classifier")
    parser.add_argument("--reports-dir", default="reports/camembert")
    parser.add_argument("--model-name", default="camembert-base")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--no-weighted-loss", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CamembertTrainingConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        seed=args.seed,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        early_stopping_patience=args.early_stopping_patience,
        use_weighted_loss=not args.no_weighted_loss,
    )

    _, val_metrics, test_metrics = train_camembert(
        train_path=ROOT / args.train,
        val_path=ROOT / args.val,
        test_path=ROOT / args.test,
        output_dir=ROOT / args.output_dir,
        reports_dir=ROOT / args.reports_dir,
        config=config,
    )

    print("=== Validation metrics ===")
    print(val_metrics)
    print("=== Test metrics ===")
    print(test_metrics)


if __name__ == "__main__":
    main()
