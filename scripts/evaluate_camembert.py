from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.camembert_evaluator import evaluate_camembert_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained CamemBERT model.")
    parser.add_argument("--model-dir", default="models/camembert_classifier")
    parser.add_argument("--reports-dir", default="reports/camembert")
    parser.add_argument("--val", default="data/processed/val.csv")
    parser.add_argument("--test", default="data/processed/test.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_camembert_model(
        model_dir=ROOT / args.model_dir,
        val_path=ROOT / args.val,
        test_path=ROOT / args.test,
        reports_dir=ROOT / args.reports_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print("VAL:", metrics["val"])
    print("TEST:", metrics["test"])


if __name__ == "__main__":
    main()
