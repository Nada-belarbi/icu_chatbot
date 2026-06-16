from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LEGACY_DATASET = ROOT / "data" / "questions_dataset" / "dataset_clean.csv"
SOURCE_INDEX = ROOT / "data" / "classifier_dataset" / "raw_sources" / "source_index.csv"

CLASSIFIER_DIR = ROOT / "data" / "classifier_dataset"
RAW_WEAK_SEED_PATH = CLASSIFIER_DIR / "raw_sources" / "manual_icu" / "weak_seed_dataset.csv"
MASTER_PATH = CLASSIFIER_DIR / "processed" / "dataset_master.csv"
TRAIN_PATH = CLASSIFIER_DIR / "processed" / "train.csv"
VAL_PATH = CLASSIFIER_DIR / "processed" / "val.csv"
TEST_PATH = CLASSIFIER_DIR / "processed" / "test.csv"
FROZEN_PATH = CLASSIFIER_DIR / "processed" / "frozen_safety_test.csv"
ANNOTATION_QUEUE_PATH = CLASSIFIER_DIR / "annotation" / "annotation_queue.csv"
ADJUDICATION_LOG_PATH = CLASSIFIER_DIR / "annotation" / "adjudication_log.csv"

MASTER_COLUMNS = [
    "query_id",
    "text_raw",
    "text_clean",
    "label_primary",
    "expected_decision",
    "risk_tags",
    "icu_topic",
    "source_dataset",
    "source_type",
    "is_synthetic",
    "family_id",
    "difficulty",
    "requires_context",
    "contains_pii",
    "contains_patient_data",
    "split",
    "review_status",
    "annotator_1",
    "annotator_2",
    "final_label",
    "notes",
]

DECISION_MAP = {
    "information_generale": "allow_rag",
    "sensible": "allow_generalized_rag",
    "sensible_reformulable": "allow_generalized_rag",
    "avis_medical_personnalise": "refuse",
    "urgence": "redirect_urgent",
    "hors_perimetre": "refuse",
    "ambigu": "ask_reformulation",
}


def _normalize(text: str) -> str:
    txt = unicodedata.normalize("NFKD", str(text))
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    txt = txt.lower()
    txt = re.sub(r"[^a-z0-9\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _guess_topic(text: str) -> str:
    t = _normalize(text)
    if any(k in t for k in ["intub", "respirateur", "ventilation"]):
        return "ventilation"
    if any(k in t for k in ["sedation", "sédation", "reveil", "réveil"]):
        return "sedation"
    if any(k in t for k in ["sepsis", "septique", "infection"]):
        return "infection_sepsis"
    if any(k in t for k in ["saturation", "oxygene", "oxygène"]):
        return "oxygen_saturation"
    if any(k in t for k in ["douleur"]):
        return "pain"
    if any(k in t for k in ["visite", "famille", "proche"]):
        return "visits"
    if any(k in t for k in ["pronostic", "surviv"]):
        return "prognosis"
    if any(k in t for k in ["medicament", "médicament", "dose", "traitement"]):
        return "medication"
    if any(k in t for k in ["dossier", "donnees", "données", "carte vitale"]):
        return "administrative"
    return "communication_team"


def _guess_risk_tags(label: str, text: str) -> str:
    t = _normalize(text)
    tags: list[str] = []

    if label in {"information_generale"}:
        tags.append("general_explanation")
    if label in {"hors_perimetre"}:
        tags.append("out_of_scope")
    if label in {"urgence"}:
        tags.append("urgent_symptom")
    if label in {"avis_medical_personnalise"}:
        tags.append("patient_specific")

    if any(k in t for k in ["diagnostic", "maladie"]):
        tags.append("diagnosis")
    if any(k in t for k in ["traitement", "medicament", "médicament"]):
        tags.append("treatment")
    if any(k in t for k in ["dose", "mg", "ml"]):
        tags.append("dosage")
    if any(k in t for k in ["surviv", "pronostic", "grave"]):
        tags.append("prognosis")
    if any(k in t for k in ["resultat", "résultat", "analyse", "saturation"]):
        tags.append("test_result_interpretation")
    if any(k in t for k in ["mon pere", "ma mere", "mon proche", "ma soeur", "mon frere"]):
        tags.append("patient_specific")

    uniq = sorted(set(tags))
    return ",".join(uniq)


def _family_id(label: str, text: str) -> str:
    n = _normalize(text)
    tokens = n.split()
    short = "_".join(tokens[:6])
    return f"{label}__{short}"[:120]


def _difficulty(label: str, text: str) -> str:
    t = _normalize(text)
    if label == "urgence":
        return "hard"
    if label == "avis_medical_personnalise":
        return "hard"
    if any(k in t for k in ["et dans son cas", "combien de temps", "est ce grave"]):
        return "hard"
    if len(t.split()) > 20:
        return "medium"
    return "easy"


def _load_public_sources() -> pd.DataFrame:
    if not SOURCE_INDEX.exists():
        return pd.DataFrame(columns=["text_raw", "source_dataset", "source_type", "is_synthetic", "language"])

    idx = pd.read_csv(SOURCE_INDEX)
    frames: list[pd.DataFrame] = []
    for _, row in idx.iterrows():
        dataset_name = str(row.get("source_dataset", "")).strip()
        path = CLASSIFIER_DIR / "raw_sources" / dataset_name / "ingested.csv"
        if not path.exists():
            raw_path = Path(str(row.get("path", "")))
            if raw_path:
                if raw_path.is_absolute():
                    path = raw_path
                else:
                    # Backward compatibility: source_index paths may be notebook-relative.
                    path = (ROOT / "notebooks" / raw_path).resolve()
        if not path.exists():
            continue
        src = pd.read_csv(path)
        if "text_raw" not in src.columns:
            continue
        for col in ["source_dataset", "source_type", "is_synthetic", "language"]:
            if col not in src.columns:
                if col == "source_dataset":
                    src[col] = str(row.get("source_dataset", "unknown_source"))
                elif col == "source_type":
                    src[col] = "public_untyped"
                elif col == "is_synthetic":
                    src[col] = False
                else:
                    src[col] = "unknown"
        src = src[["text_raw", "source_dataset", "source_type", "is_synthetic", "language"]].copy()
        src["text_raw"] = src["text_raw"].astype(str).str.strip()
        src = src[src["text_raw"] != ""]
        frames.append(src)

    if not frames:
        return pd.DataFrame(columns=["text_raw", "source_dataset", "source_type", "is_synthetic", "language"])

    all_src = pd.concat(frames, ignore_index=True)
    all_src = all_src.drop_duplicates(subset=["text_raw", "source_dataset"])
    return all_src.reset_index(drop=True)


def build() -> None:
    if not LEGACY_DATASET.exists():
        raise FileNotFoundError(f"Missing legacy dataset: {LEGACY_DATASET}")

    df = pd.read_csv(LEGACY_DATASET)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Legacy dataset must have columns: text,label")

    df = df[["text", "label"]].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].copy().reset_index(drop=True)

    records = []
    for i, row in df.iterrows():
        text = row["text"]
        label = str(row["label"])
        records.append(
            {
                "query_id": f"weak_seed_{i:06d}",
                "text_raw": text,
                "text_clean": text,
                "label_primary": label,
                "expected_decision": DECISION_MAP.get(label, "ask_reformulation"),
                "risk_tags": _guess_risk_tags(label, text),
                "icu_topic": _guess_topic(text),
                "source_dataset": "manual_weak_seed",
                "source_type": "weak_seed",
                "is_synthetic": True,
                "family_id": _family_id(label, text),
                "difficulty": _difficulty(label, text),
                "requires_context": label in {"sensible", "sensible_reformulable", "ambigu"},
                "contains_pii": False,
                "contains_patient_data": label in {"avis_medical_personnalise", "urgence", "sensible", "sensible_reformulable"},
                "split": "",
                "review_status": "not_validated",
                "annotator_1": "",
                "annotator_2": "",
                "final_label": "",
                "notes": "Imported from legacy weak seed dataset; requires manual review.",
            }
        )

    public_df = _load_public_sources()
    for i, row in public_df.iterrows():
        text = str(row["text_raw"])
        source_dataset = str(row.get("source_dataset", "unknown_source"))
        source_type = str(row.get("source_type", "public_untyped"))
        language = str(row.get("language", "unknown"))
        records.append(
            {
                "query_id": f"public_{i:06d}",
                "text_raw": text,
                "text_clean": text,
                "label_primary": "information_generale",
                "expected_decision": "allow_rag",
                "risk_tags": "general_explanation",
                "icu_topic": _guess_topic(text),
                "source_dataset": source_dataset,
                "source_type": source_type,
                "is_synthetic": False,
                "family_id": _family_id("information_generale", text),
                "difficulty": "easy",
                "requires_context": False,
                "contains_pii": False,
                "contains_patient_data": False,
                "split": "",
                "review_status": "not_validated",
                "annotator_1": "",
                "annotator_2": "",
                "final_label": "",
                "notes": f"Imported from public source ({source_dataset}, lang={language}); label is provisional and must be reviewed.",
            }
        )

    master = pd.DataFrame(records, columns=MASTER_COLUMNS)
    master = master.drop_duplicates(subset=["text_clean", "source_dataset", "label_primary"]).reset_index(drop=True)

    RAW_WEAK_SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANNOTATION_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)

    df.rename(columns={"text": "text_raw", "label": "label_primary"}).to_csv(RAW_WEAK_SEED_PATH, index=False)
    master.to_csv(MASTER_PATH, index=False)

    queue = master[(master["review_status"] == "not_validated") & (master["split"] == "")].copy()
    queue = queue.sort_values(["source_dataset", "query_id"]).reset_index(drop=True)
    queue.to_csv(ANNOTATION_QUEUE_PATH, index=False)

    if not ADJUDICATION_LOG_PATH.exists():
        pd.DataFrame(
            columns=["query_id", "annotator_1_label", "annotator_2_label", "adjudicated_label", "adjudicator", "notes"]
        ).to_csv(ADJUDICATION_LOG_PATH, index=False)

    pd.DataFrame(columns=MASTER_COLUMNS).to_csv(TRAIN_PATH, index=False)
    pd.DataFrame(columns=MASTER_COLUMNS).to_csv(VAL_PATH, index=False)
    pd.DataFrame(columns=MASTER_COLUMNS).to_csv(TEST_PATH, index=False)
    pd.DataFrame(columns=MASTER_COLUMNS).to_csv(FROZEN_PATH, index=False)

    print(f"Saved weak seed raw source: {RAW_WEAK_SEED_PATH}")
    print(f"Saved dataset master: {MASTER_PATH}")
    print(f"Rows in dataset_master: {len(master)}")
    print(f"Rows in annotation_queue: {len(queue)}")


if __name__ == "__main__":
    build()
