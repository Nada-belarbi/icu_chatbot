from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.label_schema import LABEL_OUT_OF_SCOPE


def out_of_scope_topic_issues(df: pd.DataFrame) -> pd.DataFrame:
    required = {"text", "label", "topic", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing dataset columns for quality check: {sorted(missing)}")

    mask = (df["label"] == LABEL_OUT_OF_SCOPE) & (df["topic"] != LABEL_OUT_OF_SCOPE)
    return df.loc[mask, ["text", "label", "topic", "source"]].copy()


def quality_summary(df: pd.DataFrame) -> dict:
    issues = out_of_scope_topic_issues(df)
    return {
        "rows": int(len(df)),
        "out_of_scope_topic_issues": int(len(issues)),
        "out_of_scope_topic_issue_rate": float(len(issues) / len(df)) if len(df) else 0.0,
    }


def write_train_without_out_of_scope_topic_issues(
    input_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    issues = out_of_scope_topic_issues(df)
    cleaned = df.drop(index=issues.index).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False, encoding="utf-8")
    return cleaned
