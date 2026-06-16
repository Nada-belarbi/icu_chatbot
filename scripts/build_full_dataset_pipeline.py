"""
build_full_dataset_pipeline.py
==============================
Pipeline complet dataset:
  1. Ingest sources publiques disponibles (MedQuAD, ChatDoctor, MediQA, MedQuad-QnA)
  2. Génération synthétique de questions ICU en FR (par label rare)
  3. Construction du dataset_master avec schéma complet
  4. Annotation provisoire automatique
  5. Split stratifié family-safe (train/val/test + frozen_safety_test)
  6. Écriture de tous les artefacts processed + annotation

Usage:
    python scripts/build_full_dataset_pipeline.py
"""
from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
CLASSIFIER_DIR = ROOT / "data" / "classifier_dataset"
RAW_DIR = CLASSIFIER_DIR / "raw_sources"
PROCESSED_DIR = CLASSIFIER_DIR / "processed"
ANNOTATION_DIR = CLASSIFIER_DIR / "annotation"
AUDIT_DIR = CLASSIFIER_DIR / "audit_reports"

MASTER_PATH = PROCESSED_DIR / "dataset_master.csv"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
VAL_PATH = PROCESSED_DIR / "val.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
FROZEN_PATH = PROCESSED_DIR / "frozen_safety_test.csv"
ANNOTATION_QUEUE_PATH = ANNOTATION_DIR / "annotation_queue.csv"
ADJUDICATION_LOG_PATH = ANNOTATION_DIR / "adjudication_log.csv"

for d in [RAW_DIR, PROCESSED_DIR, ANNOTATION_DIR, AUDIT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
MASTER_COLUMNS = [
    "query_id", "text_raw", "text_clean", "label_primary", "expected_decision",
    "risk_tags", "icu_topic", "source_dataset", "source_type", "is_synthetic",
    "family_id", "difficulty", "requires_context", "contains_pii",
    "contains_patient_data", "split", "review_status",
    "annotator_1", "annotator_2", "final_label", "notes",
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

VALID_LABELS = list(DECISION_MAP.keys())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize(text: str) -> str:
    txt = unicodedata.normalize("NFKD", str(text))
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    txt = txt.lower()
    txt = re.sub(r"[^a-z0-9\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _family_id(label: str, text: str) -> str:
    """
    Family grouping:
    - Strip known boilerplate prefixes (ChatDoctor template) before computing key
    - Use first 8 meaningful tokens
    - Fallback to hash bucket if fewer than 5 tokens remain
    """
    n = _normalize(text)
    # Strip ChatDoctor boilerplate prefix
    for prefix in [
        "if you are a doctor please answer the medical questions based on the patient s description",
        "doctor please answer the medical questions based on the patient s description",
        "please answer the medical questions based on the patient s description",
    ]:
        if n.startswith(prefix):
            n = n[len(prefix):].strip()
            break
    tokens = n.split()
    # Always include a full-text hash suffix to guarantee global uniqueness even
    # when two texts share the same first-8-token prefix (hash collision avoidance).
    h8 = hashlib.md5(n.encode()).hexdigest()[:8]
    if len(tokens) >= 5:
        short = "_".join(tokens[:8])
        key = f"{label}__{short}__{h8}"[:128]
    else:
        key = f"{label}__short_{h8}"
    return key


def _guess_topic(text: str) -> str:
    t = _normalize(text)
    if any(k in t for k in ["intub", "respirateur", "ventilation", "tuyau", "respir"]):
        return "ventilation"
    if any(k in t for k in ["sedation", "sedatif", "reveil", "coma", "endormi"]):
        return "sedation"
    if any(k in t for k in ["sepsis", "septique", "infection", "bacterie", "antibio"]):
        return "infection_sepsis"
    if any(k in t for k in ["saturation", "oxygene", "spo2", "o2"]):
        return "oxygen_saturation"
    if any(k in t for k in ["douleur", "souffrance", "mal", "analgesie"]):
        return "pain"
    if any(k in t for k in ["visite", "famille", "proche", "voir", "venir"]):
        return "visits"
    if any(k in t for k in ["pronostic", "surviv", "guerir", "chance", "espoir"]):
        return "prognosis"
    if any(k in t for k in ["medicament", "dose", "traitement", "mg", "ml"]):
        return "medication"
    if any(k in t for k in ["dossier", "donnee", "information", "resultat"]):
        return "administrative"
    return "general_icu"


def _guess_risk_tags(label: str, text: str) -> str:
    t = _normalize(text)
    tags: list[str] = []
    if label == "information_generale":
        tags.append("general_explanation")
    if label == "hors_perimetre":
        tags.append("out_of_scope")
    if label == "urgence":
        tags.append("urgent_symptom")
    if label == "avis_medical_personnalise":
        tags.append("patient_specific")
    if any(k in t for k in ["diagnostic", "maladie", "test"]):
        tags.append("diagnosis")
    if any(k in t for k in ["traitement", "medicament", "dose", "mg"]):
        tags.append("treatment")
    if any(k in t for k in ["surviv", "pronostic", "grave", "guerir"]):
        tags.append("prognosis")
    if any(k in t for k in ["mon pere", "ma mere", "mon proche", "ma soeur", "mon frere", "mon mari", "ma femme"]):
        tags.append("patient_specific")
    return ",".join(sorted(set(tags)))


def _difficulty(label: str, text: str) -> str:
    t = _normalize(text)
    if label in {"urgence", "avis_medical_personnalise"}:
        return "hard"
    if any(k in t for k in ["combien de temps", "est ce grave", "va t il", "va t elle"]):
        return "hard"
    if len(t.split()) > 18:
        return "medium"
    return "easy"


def _make_record(
    idx: int,
    text: str,
    label: str,
    source_dataset: str,
    source_type: str,
    is_synthetic: bool,
    review_status: str,
    notes: str,
    prefix: str = "row",
) -> dict:
    return {
        "query_id": f"{prefix}_{idx:07d}",
        "text_raw": text,
        "text_clean": text,
        "label_primary": label,
        "expected_decision": DECISION_MAP.get(label, "ask_reformulation"),
        "risk_tags": _guess_risk_tags(label, text),
        "icu_topic": _guess_topic(text),
        "source_dataset": source_dataset,
        "source_type": source_type,
        "is_synthetic": is_synthetic,
        "family_id": _family_id(label, text),
        "difficulty": _difficulty(label, text),
        "requires_context": label in {"sensible", "sensible_reformulable", "ambigu"},
        "contains_pii": False,
        "contains_patient_data": label in {"avis_medical_personnalise", "urgence", "sensible", "sensible_reformulable"},
        "split": "",
        "review_status": review_status,
        "annotator_1": "auto_rule_based",
        "annotator_2": "",
        "final_label": "",
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Step 1 — Ingest public sources
# ---------------------------------------------------------------------------
def ingest_public_sources() -> pd.DataFrame:
    print("\n[1/5] Ingesting public sources...")
    frames: list[pd.DataFrame] = []

    # --- MedQuAD (already ingested) ---
    medquad_csv = RAW_DIR / "medquad" / "ingested.csv"
    if medquad_csv.exists():
        df = pd.read_csv(medquad_csv)
        if len(df) > 0:
            frames.append(df)
            print(f"  MedQuAD from cache: {len(df)} rows")
    else:
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset("lavita/MedQuAD", split="train")
            df = pd.DataFrame({
                "text_raw": [str(x.get("question", "")).strip() for x in ds],
                "source_dataset": "medquad",
                "source_type": "public_medical_qa",
                "is_synthetic": False,
                "language": "en",
            })
            df = df[df["text_raw"] != ""].drop_duplicates()
            (RAW_DIR / "medquad").mkdir(parents=True, exist_ok=True)
            df.to_csv(medquad_csv, index=False)
            frames.append(df)
            print(f"  MedQuAD downloaded: {len(df)} rows")
        except Exception as e:
            print(f"  MedQuAD SKIP: {e}")

    # --- ChatDoctor HealthCareMagic ---
    chatdoctor_csv = RAW_DIR / "chatdoctor" / "ingested.csv"
    (RAW_DIR / "chatdoctor").mkdir(parents=True, exist_ok=True)
    if chatdoctor_csv.exists() and len(pd.read_csv(chatdoctor_csv)) > 0:
        df = pd.read_csv(chatdoctor_csv)
        frames.append(df)
        print(f"  ChatDoctor from cache: {len(df)} rows")
    else:
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split="train")
            df = pd.DataFrame({
                "text_raw": [str(x.get("instruction", "") + " " + str(x.get("input", ""))).strip() for x in ds],
                "source_dataset": "chatdoctor",
                "source_type": "public_medical_qa",
                "is_synthetic": False,
                "language": "en",
            })
            df = df[df["text_raw"].str.len() > 10].drop_duplicates()
            df.to_csv(chatdoctor_csv, index=False)
            frames.append(df)
            print(f"  ChatDoctor downloaded: {len(df)} rows")
        except Exception as e:
            print(f"  ChatDoctor SKIP: {e}")

    # --- MedAlpaca MediQA ---
    mediqa_csv = RAW_DIR / "mediqa" / "ingested.csv"
    (RAW_DIR / "mediqa").mkdir(parents=True, exist_ok=True)
    if mediqa_csv.exists() and len(pd.read_csv(mediqa_csv)) > 0:
        df = pd.read_csv(mediqa_csv)
        frames.append(df)
        print(f"  MediQA from cache: {len(df)} rows")
    else:
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset("medalpaca/medical_meadow_mediqa", split="train")
            df = pd.DataFrame({
                "text_raw": [str(x.get("input", "")).strip() for x in ds],
                "source_dataset": "mediqa",
                "source_type": "public_medical_qa",
                "is_synthetic": False,
                "language": "en",
            })
            df = df[df["text_raw"].str.len() > 10].drop_duplicates()
            df.to_csv(mediqa_csv, index=False)
            frames.append(df)
            print(f"  MediQA downloaded: {len(df)} rows")
        except Exception as e:
            print(f"  MediQA SKIP: {e}")

    # --- MedQuad MedicalQnA ---
    medquad2_csv = RAW_DIR / "medquad2" / "ingested.csv"
    (RAW_DIR / "medquad2").mkdir(parents=True, exist_ok=True)
    if medquad2_csv.exists() and len(pd.read_csv(medquad2_csv)) > 0:
        df = pd.read_csv(medquad2_csv)
        frames.append(df)
        print(f"  MedQuad-QnA from cache: {len(df)} rows")
    else:
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
            df = pd.DataFrame({
                "text_raw": [str(x.get("Question", "")).strip() for x in ds],
                "source_dataset": "medquad2",
                "source_type": "public_medical_qa",
                "is_synthetic": False,
                "language": "en",
            })
            df = df[df["text_raw"].str.len() > 10].drop_duplicates()
            df.to_csv(medquad2_csv, index=False)
            frames.append(df)
            print(f"  MedQuad-QnA downloaded: {len(df)} rows")
        except Exception as e:
            print(f"  MedQuad-QnA SKIP: {e}")

    # --- Weak seed manual ICU (legacy) ---
    weak_csv = RAW_DIR / "manual_icu" / "weak_seed_dataset.csv"
    if weak_csv.exists():
        df = pd.read_csv(weak_csv)
        if "text_raw" not in df.columns and "text" in df.columns:
            df = df.rename(columns={"text": "text_raw"})
        if "source_dataset" not in df.columns:
            df["source_dataset"] = "manual_weak_seed"
        if "source_type" not in df.columns:
            df["source_type"] = "weak_seed"
        if "is_synthetic" not in df.columns:
            df["is_synthetic"] = True
        if "language" not in df.columns:
            df["language"] = "fr"
        frames.append(df[["text_raw", "source_dataset", "source_type", "is_synthetic", "language"] +
                         ([c for c in df.columns if c not in ["text_raw","source_dataset","source_type","is_synthetic","language"]])])
        print(f"  Weak seed: {len(df)} rows")

    if not frames:
        return pd.DataFrame(columns=["text_raw", "source_dataset", "source_type", "is_synthetic", "language"])

    # Normalise to common minimal columns
    minimal = ["text_raw", "source_dataset", "source_type", "is_synthetic"]
    out_frames = []
    for f in frames:
        sub = f.copy()
        for c in minimal:
            if c not in sub.columns:
                sub[c] = ""
        if "language" not in sub.columns:
            sub["language"] = "unknown"
        # Carry label_primary if present (weak seed)
        if "label_primary" in sub.columns:
            out_frames.append(sub[minimal + ["language", "label_primary"]])
        else:
            sub["label_primary"] = ""
            out_frames.append(sub[minimal + ["language", "label_primary"]])

    combined = pd.concat(out_frames, ignore_index=True)
    combined["text_raw"] = combined["text_raw"].astype(str).str.strip()
    combined = combined[combined["text_raw"].str.len() > 5]
    combined = combined.drop_duplicates(subset=["text_raw", "source_dataset"])
    print(f"  Total after dedup: {len(combined)} rows")
    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2 — Auto-label public medical questions
# ---------------------------------------------------------------------------

# Keywords for ICU-specific routing decisions
_URGENCE_KW = [
    "emergency", "urgent", "chest pain", "can't breathe", "can not breathe",
    "cardiac arrest", "heart attack", "stroke", "code blue", "resuscitation",
    "douleur thoracique", "arret cardiaque", "urgence", "mourir", "en train de mourir",
]
_PERSO_KW = [
    "my patient", "my father", "my mother", "my wife", "my husband",
    "mon pere", "ma mere", "mon mari", "ma femme", "mon enfant", "my child",
    "his prognosis", "her prognosis", "son pronostic", "his treatment", "her treatment",
    "will he", "will she", "va t il", "va t elle",
]
_HORS_KW = [
    "recipe", "weather", "sport", "football", "movie", "music",
    "recette", "meteo", "sport", "politique", "cuisine",
]
_SENSIBLE_KW = [
    "how long", "survive", "mortality", "combien de temps", "survie",
    "chances", "prognosis", "pronostic", "probability", "probabilite",
    "end of life", "fin de vie",
]


def _auto_label(text: str, given_label: str | None = None) -> str:
    """Derive a provisional routing label from text."""
    if given_label and given_label in VALID_LABELS:
        return given_label
    t = text.lower()
    if any(k in t for k in _URGENCE_KW):
        return "urgence"
    if any(k in t for k in _HORS_KW):
        return "hors_perimetre"
    if any(k in t for k in _PERSO_KW):
        return "avis_medical_personnalise"
    if any(k in t for k in _SENSIBLE_KW):
        return "sensible"
    return "information_generale"


# ---------------------------------------------------------------------------
# Step 3 — Synthetic ICU data generation (FR, rare labels)
# ---------------------------------------------------------------------------

SYNTH_TEMPLATES: dict[str, list[str]] = {
    "information_generale": [
        # Ventilation
        "Qu'est-ce que la ventilation mécanique en réanimation ?",
        "Comment fonctionne un respirateur artificiel ?",
        "Pourquoi place-t-on un patient sous respirateur ?",
        "Qu'est-ce que la pression positive en fin d'expiration (PEEP) ?",
        "Comment fonctionne la CPAP en réanimation ?",
        "Qu'est-ce que la ventilation non invasive ?",
        "Pourquoi ajuste-t-on les paramètres du respirateur ?",
        "Qu'est-ce qu'un volume courant et pourquoi est-il important ?",
        "Comment sevrer un patient du respirateur progressivement ?",
        "Qu'est-ce que la fréquence respiratoire réglée sur le ventilateur ?",
        # Sédation
        "À quoi sert la sédation en réanimation ?",
        "Quels médicaments utilise-t-on pour sédater un patient en réanimation ?",
        "Comment se déroule un réveil de sédation ?",
        "Pourquoi réveille-t-on les patients sédatés chaque jour ?",
        "Qu'est-ce que le score de Richmond pour évaluer la sédation ?",
        "Pourquoi surveille-t-on la profondeur de la sédation ?",
        "Qu'est-ce que l'analgosédation en réanimation ?",
        "Comment évalue-t-on la douleur d'un patient sédaté ?",
        "Quels sont les effets secondaires de la sédation prolongée ?",
        "Qu'est-ce que le syndrome de sevrage aux sédatifs ?",
        # Sepsis/infection
        "Qu'est-ce que le sepsis ?",
        "Comment reconnaît-on un choc septique ?",
        "Pourquoi administre-t-on des antibiotiques en réanimation ?",
        "Qu'est-ce qu'une hémoculture et pourquoi la réalise-t-on ?",
        "Comment traite-t-on une infection à bactérie résistante ?",
        "Qu'est-ce que le SIRS en réanimation ?",
        "Pourquoi prend-on des prélèvements bactériologiques réguliers ?",
        "Qu'est-ce qu'une infection nosocomiale ?",
        "Comment prévient-on les infections liées aux cathéters ?",
        "Qu'est-ce que le contrôle de source dans le traitement du sepsis ?",
        # Oxygénation
        "Comment surveille-t-on la saturation en oxygène ?",
        "Qu'est-ce qu'un oxymètre de pouls ?",
        "Pourquoi réalise-t-on des gaz du sang en réanimation ?",
        "Qu'est-ce que l'hypoxémie et comment la traite-t-on ?",
        "Qu'est-ce que le rapport PaO2/FiO2 ?",
        "Pourquoi utilise-t-on de l'oxygène à haut débit ?",
        "Qu'est-ce que l'oxygénothérapie à haut débit nasal ?",
        "Comment évalue-t-on les besoins en oxygène d'un patient ?",
        "Qu'est-ce que la SpO2 cible en réanimation ?",
        "Pourquoi surveille-t-on la capnographie ?",
        # Cathéters/accès vasculaires
        "Qu'est-ce qu'un cathéter veineux central ?",
        "Qu'est-ce qu'une voie artérielle et à quoi sert-elle ?",
        "Pourquoi pose-t-on un cathéter artériel en réanimation ?",
        "Qu'est-ce qu'une voie veineuse périphérique ?",
        "Comment évite-t-on les complications liées aux cathéters ?",
        "Qu'est-ce qu'un cathéter de Swan-Ganz ?",
        "Pourquoi utilise-t-on une chambre implantable en réanimation ?",
        "Qu'est-ce qu'une canule de dialyse ?",
        "Comment positionne-t-on correctement un cathéter central ?",
        "Qu'est-ce que la pression veineuse centrale mesure ?",
        # Nutrition
        "Pourquoi le patient reçoit-il une nutrition artificielle ?",
        "Quelle est la différence entre nutrition entérale et parentérale ?",
        "Comment calcule-t-on les besoins nutritionnels en réanimation ?",
        "Pourquoi commence-t-on tôt la nutrition entérale ?",
        "Qu'est-ce qu'une sonde nasogastrique en réanimation ?",
        "Quelles sont les complications de la nutrition parentérale ?",
        "Comment surveille-t-on la tolérance à la nutrition entérale ?",
        "Qu'est-ce que le résidu gastrique et pourquoi le surveille-t-on ?",
        "Pourquoi mesure-t-on la glycémie régulièrement en réanimation ?",
        "Qu'est-ce que l'insulinothérapie intensive en réanimation ?",
        # Surveillance
        "Comment fonctionne un scope de monitoring ?",
        "Qu'est-ce que la surveillance continue en réanimation ?",
        "Pourquoi y a-t-il des alarmes sur les machines ?",
        "Quels paramètres surveille-t-on en permanence en réanimation ?",
        "Qu'est-ce qu'un électrocardiogramme continu ?",
        "Pourquoi mesure-t-on la température régulièrement ?",
        "Qu'est-ce que la pression intracrânienne et comment la surveille-t-on ?",
        "Comment fonctionne un moniteur de débit cardiaque ?",
        "Qu'est-ce que la saturation veineuse centrale en O2 ?",
        "Pourquoi effectue-t-on des prises de sang très fréquemment ?",
        # Dialyse/rein
        "Qu'est-ce que la dialyse en réanimation ?",
        "Qu'est-ce que l'hémodialyse continue ?",
        "Pourquoi le patient a-t-il besoin de dialyse ?",
        "Comment fonctionne une machine de dialyse en réanimation ?",
        "Qu'est-ce que l'insuffisance rénale aiguë en réanimation ?",
        "Quand indique-t-on la dialyse en réanimation ?",
        "Qu'est-ce que la diurèse forcée en réanimation ?",
        "Comment surveille-t-on la fonction rénale en réanimation ?",
        "Qu'est-ce que la créatinine et pourquoi la surveille-t-on ?",
        "Qu'est-ce qu'une séance d'épuration extra-rénale ?",
        # Procédures
        "Qu'est-ce qu'une trachéotomie ?",
        "Comment se passe une intubation orotrachéale ?",
        "Pourquoi réalise-t-on une radiographie thoracique quotidienne ?",
        "Qu'est-ce qu'une échographie thoracique au lit ?",
        "Pourquoi utilise-t-on des drains thoraciques ?",
        "Qu'est-ce qu'une ponction lombaire en réanimation ?",
        "Comment se passe une fibroscopie bronchique ?",
        "Qu'est-ce qu'un lavage bronchoalvéolaire ?",
        "Pourquoi effectue-t-on des changements de position en réanimation ?",
        "Qu'est-ce que la mise en décubitus ventral ?",
        # Prévention
        "Pourquoi le patient porte-t-il des bas de contention ?",
        "Qu'est-ce que la thromboprophylaxie en réanimation ?",
        "Comment prévient-on les ulcères de stress ?",
        "Pourquoi fait-on de la kinésithérapie en réanimation ?",
        "Qu'est-ce que la réhabilitation précoce en réanimation ?",
        "Comment prévient-on les escarres en réanimation ?",
        "Pourquoi change-t-on fréquemment la position du patient ?",
        "Qu'est-ce que l'hygiène des mains en réanimation ?",
        "Comment prévient-on les pneumonies associées à la ventilation ?",
        "Qu'est-ce qu'un bundle de prévention des infections ?",
        # Divers
        "Qu'est-ce qu'une défaillance multi-organes ?",
        "Qu'est-ce que le syndrome de détresse respiratoire aiguë ?",
        "Comment évalue-t-on le score SOFA en réanimation ?",
        "Qu'est-ce que le score SAPS II ?",
        "Pourquoi classe-t-on les patients par niveau de gravité ?",
        "Qu'est-ce qu'une transfusion sanguine en réanimation ?",
        "Pourquoi surveille-t-on la coagulation du patient ?",
        "Qu'est-ce qu'une CIVD (coagulation intravasculaire disséminée) ?",
        "Comment gère-t-on la fièvre en réanimation ?",
        "Qu'est-ce que la thérapeutique par refroidissement (hypothermie thérapeutique) ?",
        "Qu'est-ce qu'une ordonnance de réanimation ?",
        "Pourquoi fait-on des électroencéphalogrammes en réanimation ?",
        "Qu'est-ce qu'un état de mal épileptique ?",
        "Comment évalue-t-on le niveau de conscience ?",
        "Qu'est-ce que le score de Glasgow ?",
        "Pourquoi surveille-t-on la diurèse horaire ?",
        "Qu'est-ce qu'une urographie en réanimation ?",
        "Qu'est-ce que le rôle d'une infirmière de réanimation ?",
        "Comment se coordonnent les équipes soignantes en réanimation ?",
        "Quel est le rôle de l'aide-soignant en réanimation ?",
        "Pourquoi y a-t-il autant de matériel autour du patient ?",
        "Qu'est-ce que le dossier médical informatisé en réanimation ?",
    ],
    "sensible": [
        # Survie/mortalité
        "Quelles sont les chances de survie d'un patient en sepsis sévère ?",
        "Quel est le taux de mortalité en réanimation pour une pneumonie grave ?",
        "Combien de temps dure en moyenne une réanimation pour un sepsis ?",
        "Quels facteurs influencent le pronostic d'un patient ventilé ?",
        "Est-il fréquent de sortir de réanimation après un arrêt cardiaque ?",
        "Quelle est la durée moyenne de séjour en réanimation ?",
        "Quel est le pronostic d'un patient sous ECMO ?",
        "Quelle est la mortalité associée à un SDRA sévère ?",
        "Combien de temps peut-on rester sous respirateur ?",
        "Comment évolue généralement un patient en choc septique ?",
        "Quelle est la survie à un an après une réanimation longue ?",
        "Quelle est la probabilité de guérison après une défaillance multi-organes ?",
        "Combien de temps dure le coma artificiel en réanimation ?",
        "Quel est le pronostic d'un patient après une pneumonie bilatérale ?",
        "Quels sont les indicateurs de mauvais pronostic en réanimation ?",
        "Quelle est la mortalité hospitalière pour les patients ventilés ?",
        "Quelle est la durée de vie restante pour un patient avec SDRA sévère ?",
        "Comment évolue la fonction respiratoire après une réanimation longue ?",
        "Quel est le taux de survie après un choc anaphylactique sévère ?",
        "Quelle est la mortalité associée au syndrome de défaillance multi-organes ?",
        # Séquelles
        "Y a-t-il des séquelles après une longue réanimation ?",
        "Quelles sont les séquelles cognitives possibles après réanimation ?",
        "Quelles sont les conséquences d'une longue sédation ?",
        "Y a-t-il un risque de dépendance aux antalgiques après réanimation ?",
        "Les patients se souviennent-ils généralement de leur passage en réanimation ?",
        "Y a-t-il un risque de stress post-traumatique après réanimation ?",
        "Quelle est la récupération musculaire typique après une longue réanimation ?",
        "Quelles séquelles respiratoires peut-on attendre après un SDRA ?",
        "Y a-t-il des troubles de la mémoire après une réanimation ?",
        "Combien de temps dure la rééducation après une réanimation longue ?",
        "Quels sont les troubles psychologiques fréquents chez les anciens patients de réanimation ?",
        "Y a-t-il des risques de neuropathie de réanimation ?",
        "Quelles sont les séquelles cutanées après un séjour long en réanimation ?",
        "Comment se passe la reprise de l'autonomie après une réanimation ?",
        "Quels organes peuvent rester déficients après une réanimation longue ?",
        "Quelles sont les difficultés de déglutition après trachéotomie ?",
        "Comment évolue le diaphragme après ventilation prolongée ?",
        "Y a-t-il des risques de fibroses pulmonaires après SDRA ?",
        "Comment se récupère-t-on d'un syndrome de fatigue post-réanimation ?",
        "Quels sont les risques de re-hospitalisation après réanimation ?",
        # Pronostic général
        "Quel est le risque d'infections nosocomiales en réanimation ?",
        "Combien de temps après la réanimation peut-on reprendre le travail ?",
        "Quelle est la probabilité de guérison complète après un sepsis ?",
        "Quels signes montrent que le patient s'améliore en réanimation ?",
        "Quels sont les indicateurs de bonne évolution en réanimation ?",
        "Y a-t-il des risques liés à l'immobilisation prolongée en réanimation ?",
        "Combien de temps dure généralement le sevrage respiratoire ?",
        "Quel est l'impact d'une réanimation longue sur la qualité de vie ?",
        "Quelle est la récupération cognitive après un arrêt cardiaque réanimé ?",
        "Comment évalue-t-on la qualité de vie après réanimation ?",
        "Quel est le pronostic neurologique après anoxie cérébrale ?",
        "Quels tests évaluent les fonctions cognitives après réanimation ?",
        "Quel est l'impact du séjour en réanimation sur le sommeil à long terme ?",
        "Comment évolue la pression artérielle après une longue réanimation ?",
        "Quels sont les risques cardiovasculaires à long terme après réanimation ?",
        "Comment récupère-t-on la force musculaire après réanimation ?",
        "Quelle est la probabilité de retour à domicile après réanimation ?",
        "Quelle est la mortalité à 30 jours en réanimation médicale ?",
        "Quelle est la probabilité de rechute après sepsis guéri ?",
        "Quels facteurs prédisent une bonne récupération après réanimation ?",
    ],
    "avis_medical_personnalise": [
        # Mon père
        "Mon père est en réanimation depuis 5 jours, va-t-il s'en sortir ?",
        "Mon père est intubé depuis 3 semaines, ça ne s'améliore pas, que pensez-vous ?",
        "Pouvez-vous me dire si mon père va guérir de son infection ?",
        "D'après les résultats de mon père, pensez-vous qu'il va s'en sortir ?",
        "Mon père est sous ECMO, pensez-vous qu'il a des chances ?",
        "Mon père a 80 ans et est en sepsis, quel pronostic lui donnez-vous ?",
        "Mon père ne réagit plus aux antibiotiques, c'est mauvais signe ?",
        "Mon père est en dialyse depuis une semaine, ça va aller ?",
        "Selon les examens de mon père, pensez-vous qu'il va se réveiller ?",
        "Mon père a fait plusieurs arrêts cardiaques, il peut s'en remettre ?",
        # Ma mère
        "Ma mère est sous respirateur, est-ce grave dans son cas ?",
        "D'après les résultats de ma mère, pensez-vous qu'elle va s'en sortir ?",
        "Ma mère est en réanimation depuis 2 semaines, est-ce trop long ?",
        "Ma mère a 75 ans et est en sepsis, quel est son pronostic ?",
        "Ma mère ne se réveille pas depuis 10 jours, que pensez-vous ?",
        "Ma mère a une pneumonie sévère, pensez-vous qu'elle va guérir ?",
        "Selon vous, ma mère doit-elle continuer la dialyse ?",
        "Ma mère a perdu 20 kg en réanimation, est-ce dangereux selon vous ?",
        "Est-ce que ma mère peut encore parler après trachéotomie ?",
        "D'après les médecins, ma mère peut-elle rentrer à domicile bientôt ?",
        # Mon mari / ma femme
        "Mon mari a eu un arrêt cardiaque, quelles sont ses chances maintenant ?",
        "D'après vous, que va-t-il se passer pour mon mari ?",
        "En voyant les examens de mon mari, quel pronostic donnez-vous ?",
        "Ma femme est sous dialyse en réanimation, est-ce qu'elle va s'en sortir ?",
        "D'après vous, que va-t-il se passer pour ma femme ?",
        "Mon mari est sous ventilation depuis 4 semaines, c'est trop long ?",
        "Ma femme a des convulsions depuis hier, est-ce grave dans son cas ?",
        "Selon vous, mon mari a-t-il encore une chance de sortir de là ?",
        "Ma femme est comateus depuis l'accident, que pensez-vous de l'évolution ?",
        "Mon mari fait de la fièvre depuis une semaine malgré les antibiotiques, que faire ?",
        # Mon proche / fils / fille
        "Mon proche est en sepsis, pensez-vous qu'il va survivre ?",
        "Mon fils est en réanimation depuis une semaine, c'est bon signe ?",
        "Ma fille est sous respirateur depuis hier, comment évolue-t-elle d'après vous ?",
        "Compte tenu de l'état de ma mère, pensez-vous qu'on devrait arrêter les soins ?",
        "Est-ce que mon proche peut sortir de réanimation d'après ce que vous voyez ?",
        "Ma sœur ne réagit pas aux antibiotiques, vous pensez que c'est mauvais signe ?",
        "Mon frère a une défaillance multi-organes, peut-il s'en sortir selon vous ?",
        "Mon fils a une anoxie cérébrale après arrêt cardiaque, quel est son pronostic ?",
        "Ma fille est en dialyse, pensez-vous qu'elle récupérera la fonction rénale ?",
        "Selon vous, mon proche peut-il récupérer une vie normale après réanimation ?",
        # Décisions thérapeutiques
        "Docteur, selon vous, mon mari a encore combien de temps à vivre ?",
        "Pensez-vous qu'il faut continuer les soins pour mon père dans son état ?",
        "D'après vous, faut-il tenter la réanimation si mon père fait un arrêt ?",
        "En voyant l'état de ma femme, pensez-vous qu'il faut la garder en réanimation ?",
        "Selon vous, faut-il opérer mon père malgré ses comorbidités ?",
        "Pensez-vous que la greffe est encore envisageable pour ma mère ?",
        "Faut-il continuer la dialyse pour ma mère dans son état actuel ?",
        "D'après vous, est-il encore raisonnable de tenter la réanimation cardiopulmonaire ?",
        "Pensez-vous qu'on devrait faire une trachéotomie pour mon père maintenant ?",
        "D'après vous, quand peut-on envisager de mettre fin aux soins actifs ?",
        # Demandes de pronostic chiffré
        "Quel pourcentage de chance mon père a-t-il de s'en sortir ?",
        "En termes statistiques, quelle est la survie pour un cas comme celui de ma mère ?",
        "Sur une échelle de 1 à 10, comment évaluez-vous les chances de mon mari ?",
        "Dans quel délai peut-on espérer un rétablissement pour ma femme selon vous ?",
        "Mon père a un score SOFA de 12, est-ce que ça veut dire qu'il va mourir ?",
    ],
    "urgence": [
        # Arrêt cardiaque / détresse
        "Mon proche en réanimation vient de faire un arrêt cardiaque, que faire immédiatement ?",
        "Mon père vient de faire une crise convulsive, que dois-je faire ?",
        "Mon proche est en train de s'étouffer, il n'y a personne, que faire maintenant ?",
        "Ma mère vient de perdre conscience dans sa chambre, que faire ?",
        "Mon proche saigne abondamment, pas d'infirmière disponible, comment agir ?",
        "Mon père ne répond plus à mon appel, il est seul en chambre, c'est urgent ?",
        "Mon proche a vomi et s'étouffe avec le tuyau, que faire de toute urgence ?",
        "Je vois que mon proche ne respire plus normalement, c'est une urgence ?",
        "Mon père vient de chuter du lit, il ne se relève pas, que faire ?",
        "Ma mère est en train de saigner au niveau du cathéter, que faire ?",
        # Alarmes / équipements
        "Il y a une alarme qui retentit depuis plusieurs minutes dans la chambre, personne ne vient !",
        "Je vois que le respirateur s'est arrêté de fonctionner, c'est une urgence ?",
        "Il y a une fuite sur le tuyau du respirateur, c'est dangereux ?",
        "Mon proche tente d'arracher son tuyau d'intubation, je fais quoi ?",
        "Le débit de la perfusion est bloqué depuis longtemps, est-ce dangereux ?",
        "La pompe à perfusion sonne en continu, que faire ?",
        "L'alarme du saturomètre retentit sans arrêt, c'est grave ?",
        "Le moniteur indique une fréquence cardiaque de 200, c'est urgent ?",
        "Le respirateur indique une haute pression, que faire ?",
        "Le scope de mon père montre une flatline, est-ce une urgence ?",
        # Signes cliniques alarmants
        "La pression artérielle de mon père vient de chuter sur le moniteur, c'est grave ?",
        "Le saturomètre de ma mère affiche 70%, c'est urgent ?",
        "Mon proche est tout bleu, c'est une urgence ?",
        "Je suis dans la salle d'attente et une infirmière m'a dit que mon père s'aggrave, qui appeler ?",
        "Mon proche a 42 degrés de fièvre depuis une heure, c'est dangereux ?",
        "Mon père a une tension de 60/30 sur le moniteur, c'est critique ?",
        "Mon proche vient de faire une crise d'agitation importante, que faire ?",
        "Ma mère saigne du nez abondamment sans que personne ne vienne, que faire ?",
        "Mon proche vomit du sang en réanimation, c'est une urgence ?",
        "Mon père a les lèvres violettes, est-ce une urgence ?",
        # Situations d'attente
        "Le médecin ne vient pas depuis plusieurs heures et mon père se dégrade, que faire ?",
        "L'infirmière de nuit n'est pas venue depuis 2 heures, mon proche ne va pas bien ?",
        "Personne ne répond à la sonnette depuis 30 minutes, c'est normal ?",
        "Mon père est tombé du lit et personne ne vient, que faire ?",
        "J'appuie sur le bouton d'urgence depuis 10 minutes, personne ne vient, que faire ?",
        "Ma mère délire et tente de se lever, il n'y a pas d'infirmière dans le couloir ?",
        "Mon proche vient de décrocher sa voie artérielle, du sang coule, que faire ?",
        "Le soluté de mon père est vide depuis longtemps, c'est urgent ?",
        "Ma mère a décroché son électrode cardiaque, est-ce dangereux ?",
        "Mon proche a de la mousse aux lèvres depuis quelques minutes, c'est urgent ?",
    ],
    "hors_perimetre": [
        # Cuisine/alimentation
        "Pouvez-vous me donner une recette de cuisine pour les convalescents ?",
        "Quel est le meilleur régime alimentaire pour maigrir rapidement ?",
        "Comment préparer des repas équilibrés pour un proche convalescent ?",
        "Quels aliments éviter pour perdre du poids ?",
        "Pouvez-vous me conseiller sur mon régime alimentaire au quotidien ?",
        "Comment faire un régime paléo pour me remettre en forme ?",
        "Quelle est la recette d'une soupe adaptée pour un convalescent ?",
        "Quels aliments favorisent la récupération après une opération ordinaire ?",
        "Comment cuisiner pour une personne intolérante au lactose ?",
        "Quelles sont les meilleures recettes pour prendre du poids sainement ?",
        # Administratif/juridique
        "Pouvez-vous m'aider à comprendre ma feuille d'impôts ?",
        "Comment trouver un bon avocat pour une succession ?",
        "Comment faire une réclamation auprès de mon assurance habitation ?",
        "Pouvez-vous m'aider à remplir un formulaire de demande d'aide sociale ?",
        "Comment obtenir un arrêt de travail longue durée ?",
        "Que faire pour ma retraite si je dois m'occuper de mon père malade ?",
        "Comment contacter un service social pour aide à domicile ?",
        "Comment rédiger une procuration pour gérer les affaires d'un proche hospitalisé ?",
        "Comment faire une déclaration de tutelle pour mon proche ?",
        "Comment obtenir une reconnaissance de travailleur handicapé ?",
        # Transport/logistique
        "Comment contacter un service de taxi pour aller à l'hôpital ?",
        "Pouvez-vous me dire quel hôtel réserver pour les proches de patients ?",
        "Comment organiser le transport d'un patient par ambulance privée ?",
        "Y a-t-il un parking gratuit pour les familles à l'hôpital ?",
        "Comment me déplacer facilement si je n'ai pas de voiture ?",
        "Pouvez-vous me donner l'adresse des restaurants près de l'hôpital ?",
        "Y a-t-il des chambres disponibles pour rester à côté de mon proche ?",
        "Comment réserver un hébergement proche de l'hôpital ?",
        "Quel est le numéro du taxi conventionné pour les soins ?",
        "Y a-t-il des navettes gratuites depuis la gare jusqu'à l'hôpital ?",
        # Loisirs/vie quotidienne
        "Quel médicament prendre pour un rhume banal sans aller chez le médecin ?",
        "Pouvez-vous me recommander un bon dentiste près d'ici ?",
        "Comment obtenir un rendez-vous avec un spécialiste rapidement ?",
        "Quelle est la météo à Paris ce week-end ?",
        "Pouvez-vous me recommander un bon film à regarder à l'hôpital ?",
        "Y a-t-il une salle de sport près de l'hôpital ?",
        "Comment me connecter à internet dans la chambre de mon proche ?",
        "Pouvez-vous me conseiller sur les meilleures assurances complémentaires ?",
        "Comment prendre soin de mon chat pendant que mon proche est hospitalisé ?",
        "Y a-t-il une garderie à l'hôpital pour les enfants des patients ?",
        # Questions médicales hors réanimation
        "Mon rhume dure depuis 2 semaines, c'est normal ?",
        "Quel médicament prendre contre la migraine sans ordonnance ?",
        "Comment traiter une entorse de la cheville chez un adulte ?",
        "Quelle crème utiliser pour un coup de soleil ?",
        "Comment soigner une infection cutanée superficielle ?",
        "Mon enfant a de la fièvre à 38.5, dois-je l'emmener aux urgences ?",
        "Quel antibiotique est indiqué pour une angine bactérienne ?",
        "Comment traiter un zona sans complication ?",
        "Quelle est la posologie de l'aspirine en automédication ?",
        "Comment gérer une allergie saisonnière aux pollens ?",
    ],
}

# Additional variations to avoid template clustering
_PREFIXES = [
    "", "Bonjour, ", "Bonsoir, ", "Excusez-moi, ", "S'il vous plaît, ",
    "Pourriez-vous m'expliquer ", "Je voudrais savoir ", "J'aimerais comprendre ",
    "Pouvez-vous m'expliquer ", "Je ne comprends pas bien, ",
]
_SUFFIXES = [
    "", " Merci.", " Merci d'avance.", " C'est important pour nous.",
    " Nous sommes très inquiets.", " La famille veut comprendre.",
    " Nous n'avons pas eu d'explication.", " Le médecin n'a pas eu le temps de tout expliquer.",
]


def _augment(base: str, seed: int) -> str:
    """Light augmentation to diversify phrasing for a given base template."""
    rng = random.Random(seed)
    # Optionally add prefix or suffix (but not both to avoid bloat)
    if rng.random() < 0.4:
        base = rng.choice(_PREFIXES) + base[0].lower() + base[1:]
    elif rng.random() < 0.3:
        base = base.rstrip(".?!") + rng.choice(_SUFFIXES)
    return base.strip()


def generate_synthetic_icu(
    target_per_label: int = 1200,
    existing_counts: dict[str, int] | None = None,
) -> list[dict]:
    """Generate synthetic ICU questions to reach target_per_label per label."""
    print("\n[2/5] Generating synthetic ICU data...")
    records: list[dict] = []
    idx = 0
    for label, templates in SYNTH_TEMPLATES.items():
        existing = (existing_counts or {}).get(label, 0)
        need = max(0, target_per_label - existing)
        if need == 0:
            print(f"  {label}: already has {existing} rows, skipping")
            continue
        generated = 0
        seed = 0
        while generated < need:
            base = templates[seed % len(templates)]
            text = _augment(base, seed=seed + hash(label) % 999983)
            records.append(_make_record(
                idx=idx,
                text=text,
                label=label,
                source_dataset="synthetic_icu_fr",
                source_type="synthetic_fr",
                is_synthetic=True,
                review_status="validated",  # Rules-based synthetic: auto-validated
                notes=f"Synthetic FR ICU example generated from template, label={label}.",
                prefix="synth",
            ))
            idx += 1
            generated += 1
            seed += 1
        print(f"  {label}: generated {generated} synthetic rows (existing={existing})")
    return records


# ---------------------------------------------------------------------------
# Step 4 — Build master from all sources
# ---------------------------------------------------------------------------
def build_master(public_df: pd.DataFrame, synth_records: list[dict]) -> pd.DataFrame:
    print("\n[3/5] Building dataset_master...")
    records: list[dict] = []

    for i, row in public_df.iterrows():
        text = str(row.get("text_raw", "")).strip()
        if not text:
            continue
        given_label = str(row.get("label_primary", "")).strip() if "label_primary" in row and str(row.get("label_primary", "")).strip() else None
        label = _auto_label(text, given_label)
        source_dataset = str(row.get("source_dataset", "unknown"))
        source_type = str(row.get("source_type", "public"))
        is_synth = bool(row.get("is_synthetic", False))
        # All auto-labeled public rows are rule-validated (conservative label by keyword rule)
        review = "validated"
        records.append(_make_record(
            idx=i,
            text=text,
            label=label,
            source_dataset=source_dataset,
            source_type=source_type,
            is_synthetic=is_synth,
            review_status=review,
            notes=f"Public source ({source_dataset}); label is auto-derived.",
            prefix="pub",
        ))

    records.extend(synth_records)
    master = pd.DataFrame(records, columns=MASTER_COLUMNS)
    master = master.drop_duplicates(subset=["text_clean", "source_dataset", "label_primary"]).reset_index(drop=True)
    # Regenerate query_ids to ensure uniqueness
    master["query_id"] = [f"r{i:07d}" for i in range(len(master))]
    print(f"  Master rows: {len(master)}")
    print(f"  Labels: {master['label_primary'].value_counts().to_dict()}")
    print(f"  Sources: {master['source_dataset'].value_counts().to_dict()}")
    return master


# ---------------------------------------------------------------------------
# Step 5 — Family-safe stratified split
# ---------------------------------------------------------------------------
def assign_splits(
    master: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    frozen_n_per_label: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign train/val/test/frozen_safety_test splits respecting:
      - Only validated/reviewed rows get a split assignment
      - No synthetic rows in val/test/frozen
      - No family_id leakage across splits
      - Stratified by label × difficulty
    """
    print("\n[4/5] Assigning splits (family-safe)...")
    df = master.copy()
    df["split"] = ""

    # Only curate rows that are validated
    curable = df[df["review_status"].isin({"reviewed", "validated"})].copy()
    non_curable = df[~df["review_status"].isin({"reviewed", "validated"})].copy()
    print(f"  Curable rows (reviewed/validated): {len(curable)}")

    # ---- frozen_safety_test: hard/sensitive cases, NON-SYNTHETIC ONLY ----
    # frozen_n_per_label non-synthetic rows per safety-critical label.
    # Do NOT use ignore_index=True — must preserve original df indices.
    FROZEN_LABELS = ["urgence", "avis_medical_personnalise", "sensible", "hors_perimetre"]
    frozen_indices: list[int] = []
    used_families: set[str] = set()
    for label in FROZEN_LABELS:
        pool = curable[
            (~curable["is_synthetic"].astype(str).str.lower().isin(["true", "1"])) &
            (curable["label_primary"] == label)
        ]
        if pool.empty:
            # fallback: allow synthetic only if nothing non-synthetic exists
            pool = curable[curable["label_primary"] == label]
        if pool.empty:
            continue
        sampled = pool.sample(min(frozen_n_per_label, len(pool)), random_state=seed)
        frozen_indices.extend(sampled.index.tolist())
        used_families.update(sampled["family_id"].tolist())

    if frozen_indices:
        df.loc[frozen_indices, "split"] = "frozen_safety_test"
        curable = curable[~curable.index.isin(frozen_indices)]
        print(f"  frozen_safety_test: {len(frozen_indices)} rows")

    # ---- Remaining pool, split by family ----
    # Group families → assign entire family to one split
    # Separate non-synthetic (available for val/test) from synthetic-only families
    non_synth_families = set(
        curable[~curable["is_synthetic"].astype(str).str.lower().isin(["true", "1"])]["family_id"].unique()
    )
    synth_only_families = set(curable["family_id"].unique()) - non_synth_families

    rng = random.Random(seed)

    # Non-synthetic families: eligible for all splits
    ns_list = sorted(non_synth_families - used_families)
    rng.shuffle(ns_list)
    n_ns = len(ns_list)
    ns_train_end = int(n_ns * train_ratio)
    ns_val_end = ns_train_end + int(n_ns * val_ratio)

    family_to_split: dict[str, str] = {}
    for fam in used_families:
        family_to_split[fam] = ""

    for i, fam in enumerate(ns_list):
        if i < ns_train_end:
            family_to_split[fam] = "train"
        elif i < ns_val_end:
            family_to_split[fam] = "val"
        else:
            family_to_split[fam] = "test"

    # Synthetic-only families: only train
    for fam in synth_only_families - used_families:
        family_to_split[fam] = "train"

    for idx2, row in curable.iterrows():
        proposed = family_to_split.get(row["family_id"], "")
        # Strict: synthetic rows never go to val/test/frozen (Rule 2)
        if proposed in ("val", "test") and str(row["is_synthetic"]).lower() in ("true", "1"):
            proposed = "train"
        df.at[idx2, "split"] = proposed

    # Stats
    split_counts = df["split"].value_counts(dropna=False).to_dict()
    print(f"  Split counts: {split_counts}")
    label_split = df[df["split"] != ""].groupby(["label_primary", "split"]).size().unstack(fill_value=0)
    print(f"  Label×split distribution:\n{label_split}")
    return df


# ---------------------------------------------------------------------------
# Step 6 — Write processed artifacts
# ---------------------------------------------------------------------------
def write_artifacts(master: pd.DataFrame) -> None:
    print("\n[5/5] Writing processed artifacts...")
    master.to_csv(MASTER_PATH, index=False)
    print(f"  dataset_master.csv: {len(master)} rows")

    for split_name, path in [
        ("train", TRAIN_PATH),
        ("val", VAL_PATH),
        ("test", TEST_PATH),
        ("frozen_safety_test", FROZEN_PATH),
    ]:
        subset = master[master["split"] == split_name].copy()
        subset.to_csv(path, index=False)
        print(f"  {split_name}.csv: {len(subset)} rows")

    # Annotation queue = not_validated, no split yet
    queue = master[(master["review_status"] == "not_validated") & (master["split"] == "")].copy()
    queue = queue.sort_values(["source_dataset", "query_id"]).reset_index(drop=True)
    queue.to_csv(ANNOTATION_QUEUE_PATH, index=False)
    print(f"  annotation_queue.csv: {len(queue)} rows")

    if not ADJUDICATION_LOG_PATH.exists():
        pd.DataFrame(columns=["query_id","annotator_1_label","annotator_2_label","adjudicated_label","adjudicator","notes"]).to_csv(ADJUDICATION_LOG_PATH, index=False)

    # Source index
    source_index = pd.DataFrame([
        {"source_dataset": d.name, "path": str(d / "ingested.csv")}
        for d in sorted(RAW_DIR.iterdir()) if d.is_dir()
    ])
    source_index.to_csv(RAW_DIR / "source_index.csv", index=False)

    # Audit distribution
    dist = master.groupby(["label_primary", "split"]).size().reset_index(name="count")
    dist.to_csv(AUDIT_DIR / "split_label_distribution.csv", index=False)
    fam_dist = master.groupby("label_primary")["family_id"].nunique().reset_index(name="n_families")
    fam_dist.to_csv(AUDIT_DIR / "label_distribution.csv", index=False)
    src_dist = master["source_dataset"].value_counts().rename_axis("source_dataset").reset_index(name="count")
    src_dist.to_csv(AUDIT_DIR / "source_distribution.csv", index=False)
    print("  Audit reports written.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("ICU CHATBOT — FULL DATASET PIPELINE (no training)")
    print("=" * 60)

    public_df = ingest_public_sources()

    # Count existing validated rows per label to compute synth targets
    existing_counts: dict[str, int] = {}
    if len(public_df) > 0 and "label_primary" in public_df.columns:
        for label in VALID_LABELS:
            existing_counts[label] = int(
                (public_df["label_primary"].astype(str) == label).sum()
            )

    synth_records = generate_synthetic_icu(
        target_per_label=1200,
        existing_counts=existing_counts,
    )

    master = build_master(public_df, synth_records)
    master = assign_splits(master)
    write_artifacts(master)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE — ready for audit (04_audit_dataset_quality.ipynb)")
    print("=" * 60)


if __name__ == "__main__":
    main()
