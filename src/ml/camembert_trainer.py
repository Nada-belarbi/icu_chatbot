from __future__ import annotations

import inspect
import json
import random
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from config.label_schema import ID_TO_LABEL, LABEL_ORDER, LABEL_TO_ID, REQUIRED_DATASET_COLUMNS
from src.ml.classifier_metrics import compute_classifier_metrics


@dataclass(frozen=True)
class CamembertTrainingConfig:
    model_name: str = "camembert-base"
    max_length: int = 192
    seed: int = 42
    learning_rate: float = 2e-5
    epochs: int = 6
    train_batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    early_stopping_patience: int = 2
    metric_for_best_model: str = "macro_f1"
    use_weighted_loss: bool = True


class WeightedLossTrainer(Trainer):
    """Trainer variant using class weights for imbalanced safety labels."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
            if self.class_weights is not None
            else None
        )
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_training_environment() -> dict[str, str]:
    required = {
        "torch": "torch",
        "transformers": "transformers",
        "datasets": "datasets",
        "accelerate": "accelerate",
    }
    versions: dict[str, str] = {}
    missing: list[str] = []

    for package_name, distribution_name in required.items():
        try:
            versions[package_name] = version(distribution_name)
        except PackageNotFoundError:
            missing.append(package_name)

    if missing:
        packages = " ".join(missing)
        raise RuntimeError(
            "Dépendances manquantes pour entraîner CamemBERT: "
            f"{packages}. Installe-les dans le kernel du notebook avec "
            f"`%pip install {packages}`, puis redémarre le kernel."
        )

    trainer_signature = inspect.signature(Trainer.__init__)
    if (
        "processing_class" not in trainer_signature.parameters
        and "tokenizer" not in trainer_signature.parameters
    ):
        raise RuntimeError(
            "Incompatible transformers version: Trainer.__init__ accepts neither "
            "`processing_class` nor `tokenizer`."
        )

    training_args_signature = inspect.signature(TrainingArguments.__init__)
    if (
        "eval_strategy" not in training_args_signature.parameters
        and "evaluation_strategy" not in training_args_signature.parameters
    ):
        raise RuntimeError(
            "Incompatible transformers version: TrainingArguments accepts neither "
            "`eval_strategy` nor `evaluation_strategy`."
        )

    versions["cuda_available"] = str(torch.cuda.is_available())
    versions["device"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    )
    return versions


def load_csv_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {split_name} split: {path}")

    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_DATASET_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{split_name} missing columns: {missing}")

    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].ne("")].reset_index(drop=True)

    unknown_labels = sorted(set(df["label"]) - set(LABEL_ORDER))
    if unknown_labels:
        raise ValueError(f"{split_name} contains unknown labels: {unknown_labels}")

    df["labels"] = df["label"].map(LABEL_TO_ID).astype(int)
    return df


def load_dataset_dict(train_path: Path, val_path: Path, test_path: Path) -> DatasetDict:
    train_df = load_csv_split(train_path, "train")
    val_df = load_csv_split(val_path, "validation")
    test_df = load_csv_split(test_path, "test")

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "validation": Dataset.from_pandas(val_df, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False),
        }
    )


def tokenize_dataset(dataset: DatasetDict, tokenizer, max_length: int) -> DatasetDict:
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize, batched=True)
    keep_cols = {"input_ids", "attention_mask", "labels"}
    remove_cols = [c for c in tokenized["train"].column_names if c not in keep_cols]
    return tokenized.remove_columns(remove_cols)


def compute_class_weights(train_label_ids: list[int]) -> torch.Tensor:
    if not train_label_ids:
        raise ValueError("Cannot compute class weights from an empty training split.")

    label_ids = np.array(train_label_ids, dtype=int)
    counts = np.bincount(label_ids, minlength=len(LABEL_ORDER))
    total = counts.sum()
    weights = np.zeros(len(LABEL_ORDER), dtype=float)
    present = counts > 0
    weights[present] = total / (len(LABEL_ORDER) * counts[present])
    return torch.tensor(weights, dtype=torch.float)


def build_model(model_name: str):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_ORDER),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )


def make_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred_ids = np.argmax(logits, axis=-1)

        y_true = [ID_TO_LABEL[int(label_id)] for label_id in labels]
        y_pred = [ID_TO_LABEL[int(pred_id)] for pred_id in pred_ids]
        metrics = compute_classifier_metrics(y_true, y_pred)

        return {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_recall": metrics["macro_recall"],
            "weighted_f1": metrics["weighted_f1"],
            "critical_errors": metrics["critical_errors"],
            "recall_urgence": metrics["recall_urgence"],
            "recall_avis_medical_personnalise": metrics[
                "recall_avis_medical_personnalise"
            ],
            "recall_hors_perimetre": metrics["recall_hors_perimetre"],
        }

    return compute_metrics


def _training_arguments_kwargs(
    output_dir: Path,
    reports_dir: Path,
    config: CamembertTrainingConfig,
) -> dict:
    kwargs = {
        "output_dir": str(output_dir),
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": config.epochs,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "load_best_model_at_end": True,
        "metric_for_best_model": config.metric_for_best_model,
        "greater_is_better": True,
        "save_total_limit": 2,
        "fp16": torch.cuda.is_available(),
        "report_to": ["tensorboard"],
        "logging_dir": str(reports_dir / "tensorboard"),
        "seed": config.seed,
        "data_seed": config.seed,
        "remove_unused_columns": True,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    return kwargs


def _trainer_tokenizer_kwargs(tokenizer) -> dict:
    signature = inspect.signature(Trainer.__init__)
    if "processing_class" in signature.parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def train_camembert(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    output_dir: Path,
    reports_dir: Path,
    config: CamembertTrainingConfig | None = None,
):
    config = config or CamembertTrainingConfig()
    check_training_environment()
    seed_everything(config.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw_dataset = load_dataset_dict(train_path, val_path, test_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenized = tokenize_dataset(raw_dataset, tokenizer, config.max_length)

    class_weights = (
        compute_class_weights(raw_dataset["train"]["labels"])
        if config.use_weighted_loss
        else None
    )

    model = build_model(config.model_name)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    args = TrainingArguments(
        **_training_arguments_kwargs(output_dir, reports_dir, config)
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            )
        ],
        class_weights=class_weights,
        **_trainer_tokenizer_kwargs(tokenizer),
    )

    trainer.train()

    val_metrics = trainer.evaluate(tokenized["validation"], metric_key_prefix="val")
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metadata = {
        "training_config": asdict(config),
        "label_order": LABEL_ORDER,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": {str(k): v for k, v in ID_TO_LABEL.items()},
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "train_path": str(train_path.as_posix()),
        "val_path": str(val_path.as_posix()),
        "test_path": str(test_path.as_posix()),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "label_mapping.json").write_text(
        json.dumps(
            {
                "label_order": LABEL_ORDER,
                "label_to_id": LABEL_TO_ID,
                "id_to_label": {str(k): v for k, v in ID_TO_LABEL.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (reports_dir / "camembert_val_metrics.json").write_text(
        json.dumps(val_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (reports_dir / "camembert_test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return trainer, val_metrics, test_metrics
