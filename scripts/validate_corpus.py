"""
scripts/validate_corpus.py

Validate data/rag_documents/sources.csv (the corpus manifest) and the raw
documents it references. Independent of the indexing pipeline — read-only.

Checks:
  - missing document (local_filename set but file absent under raw/)
  - empty / near-empty document
  - duplicate source_id
  - missing official_url
  - missing title
  - missing organization
  - status not in the allowed indexable set (informational, not a failure)

Usage:
    python scripts/validate_corpus.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

INDEXABLE_STATUSES = {"APPROVED", "APPROVED_CONTEXTUAL", "APPROVED_LEGACY"}
MIN_DOCUMENT_CHARS = 50


def main() -> int:
    import pandas as pd

    manifest_path = PROJECT_ROOT / "data" / "rag_documents" / "sources.csv"
    raw_dir = PROJECT_ROOT / "data" / "rag_documents" / "raw"

    if not manifest_path.exists():
        print(f"FAIL — manifest not found: {manifest_path}")
        return 1

    df = pd.read_csv(manifest_path)
    issues: list[str] = []

    required_cols = ["source_id", "title", "organization", "official_url", "status", "local_filename"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"FAIL — manifest missing required columns: {missing_cols}")
        return 1

    dup_ids = df["source_id"][df["source_id"].duplicated()].unique().tolist()
    if dup_ids:
        issues.append(f"duplicate source_id: {dup_ids}")

    missing_url = df.loc[df["official_url"].isna() | (df["official_url"].astype(str).str.strip() == ""), "source_id"].tolist()
    if missing_url:
        issues.append(f"missing official_url ({len(missing_url)} rows): {missing_url[:10]}{' ...' if len(missing_url) > 10 else ''}")

    missing_title = df.loc[df["title"].isna() | (df["title"].astype(str).str.strip() == ""), "source_id"].tolist()
    if missing_title:
        issues.append(f"missing title ({len(missing_title)} rows): {missing_title[:10]}{' ...' if len(missing_title) > 10 else ''}")

    missing_org = df.loc[df["organization"].isna() | (df["organization"].astype(str).str.strip() == ""), "source_id"].tolist()
    if missing_org:
        issues.append(f"missing organization ({len(missing_org)} rows): {missing_org[:10]}{' ...' if len(missing_org) > 10 else ''}")

    has_local = df[df["local_filename"].notna() & (df["local_filename"].astype(str).str.strip() != "")]
    missing_files, empty_files = [], []
    for row in has_local.itertuples():
        doc_path = raw_dir / str(row.local_filename)
        if not doc_path.exists():
            missing_files.append(row.source_id)
            continue
        try:
            size = doc_path.stat().st_size
            if size < MIN_DOCUMENT_CHARS:
                empty_files.append(row.source_id)
        except OSError:
            missing_files.append(row.source_id)
    if missing_files:
        issues.append(f"local_filename set but file absent ({len(missing_files)} rows): {missing_files[:10]}{' ...' if len(missing_files) > 10 else ''}")
    if empty_files:
        issues.append(f"file present but near-empty (<{MIN_DOCUMENT_CHARS} bytes) ({len(empty_files)} rows): {empty_files[:10]}{' ...' if len(empty_files) > 10 else ''}")

    non_indexable_statuses = sorted(set(df["status"]) - INDEXABLE_STATUSES)

    print("=== Corpus validation ===")
    print(f"Manifest rows       : {len(df)}")
    print(f"Indexable statuses   : {sorted(INDEXABLE_STATUSES)}")
    print(f"Other statuses found : {non_indexable_statuses} (excluded from indexing by design, not a failure)")
    print()

    if issues:
        print(f"FAIL — {len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("PASS — no manifest/document issues found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
