from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing.text_cleaner import TextCleaner

SEED = 42

DATASET_DIR = Path("data/questions_dataset")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

LABEL_GENERAL = "information_generale"
LABEL_SENSITIVE = "sensible"
LABEL_MEDICAL_ADVICE = "avis_medical_personnalise"
LABEL_EMERGENCY = "urgence"
LABEL_OUT_OF_SCOPE = "hors_perimetre"

MIN_CLASS_SIZE = 150


def _normalize_text(text: str) -> str:
    txt = unicodedata.normalize("NFKD", text)
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    txt = txt.lower()
    txt = re.sub(r"[^a-z0-9\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _is_probably_french(text: str) -> bool:
    txt = _normalize_text(text)
    if not txt:
        return False

    french_markers = {
        "le", "la", "les", "de", "des", "du", "un", "une", "et",
        "pour", "avec", "sans", "dans", "sur", "que", "quoi", "comment",
        "pourquoi", "est", "sont", "patient", "reanimation", "urgence",
    }
    english_markers = {
        "what", "how", "why", "when", "where", "is", "are", "the", "treatments",
        "symptoms", "diagnose", "risk", "disease", "inherited",
    }

    tokens = set(txt.split())
    fr = len(tokens & french_markers)
    en = len(tokens & english_markers)
    return fr >= en


def _fr_general_questions() -> list[str]:
    core = [
        "Qu'est-ce qu'un respirateur en réanimation ?",
        "À quoi sert la sédation en soins intensifs ?",
        "Pourquoi surveille-t-on la saturation en oxygène ?",
        "Que signifie l'intubation ?",
        "Quel est le rôle du monitorage continu ?",
        "Pourquoi un patient peut-il être isolé en réanimation ?",
        "À quoi servent les perfusions en réanimation ?",
        "Pourquoi fait-on des gaz du sang ?",
        "Comment fonctionne la ventilation assistée ?",
        "Pourquoi surveille-t-on la pression artérielle en continu ?",
        "Qu'est-ce qu'un choc septique ?",
        "Quels sont les signes généraux d'une détresse respiratoire ?",
        "Comment l'équipe évalue-t-elle la douleur en réanimation ?",
        "Quelle est la différence entre oxygénothérapie et ventilation mécanique ?",
        "À quoi sert une sonde en réanimation ?",
        "Pourquoi réalise-t-on des prises de sang répétées ?",
        "Qu'est-ce qu'un sevrage ventilatoire ?",
        "Pourquoi parle-t-on de pronostic réservé en soins critiques ?",
        "Comment prévenir les infections en réanimation ?",
        "Pourquoi limite-t-on parfois les visites ?",
    ]
    variants = [
        "Pouvez-vous expliquer {q}",
        "J'aimerais comprendre {q}",
        "Pourriez-vous m'expliquer {q}",
        "Question générale : {q}",
    ]
    out = []
    for q in core:
        out.append(q)
        for v in variants:
            out.append(v.format(q=q[0].lower() + q[1:]))
    return out


def _fr_sensitive_questions() -> list[str]:
    relatives = ["mon père", "ma mère", "mon conjoint", "ma soeur", "mon proche"]
    themes = [
        "est en réanimation, pouvez-vous m'expliquer les machines",
        "a été intubé, je ne comprends pas ce que cela implique",
        "est sous sédation, pouvez-vous expliquer ce que cela veut dire",
        "a des perfusions, à quoi servent-elles exactement",
        "est monitoré en continu, pouvez-vous m'expliquer pourquoi",
        "semble très fatigué, est-ce fréquent en réanimation",
        "ne peut pas parler, comment communiquer avec lui",
        "a de nombreux tuyaux, pouvez-vous m'expliquer leur rôle",
    ]
    openings = ["", "Bonjour, ", "S'il vous plaît, ", "Excusez-moi, "]
    out = []
    for rel in relatives:
        for th in themes:
            for op in openings:
                out.append(f"{op}{rel} {th} ?")
    return out


def _fr_medical_advice_questions() -> list[str]:
    relatives = ["mon père", "ma mère", "mon proche"]
    unsafe = [
        "Quel traitement faut-il donner à {r} ?",
        "Quelle dose dois-je donner à {r} ?",
        "Pouvez-vous prescrire un médicament pour {r} ?",
        "Doit-on arrêter le traitement de {r} maintenant ?",
        "Est-ce grave pour {r} et que faut-il faire exactement ?",
        "Va-t-il survivre et que dois-je faire immédiatement ?",
        "Quel diagnostic précis avez-vous pour {r} ?",
        "Pouvez-vous confirmer que {r} va guérir rapidement ?",
    ]
    # Contrastive but still dangerous because personalized intent is explicit.
    contrastive = [
        "Quels traitements sont généralement utilisés pour le SDRA, et lequel donner à {r} ?",
        "On parle de sepsis en général, mais dans le cas de {r} que recommandez-vous ?",
        "Je comprends l'information générale, mais concrètement que dois-je administrer à {r} ?",
    ]
    out = []
    for r in relatives:
        for t in unsafe + contrastive:
            out.append(t.format(r=r))
            out.append(f"Urgent: {t.format(r=r)}")
            out.append(f"Réponse courte svp: {t.format(r=r)}")
    return out


def _fr_emergency_questions() -> list[str]:
    incidents = [
        "Il ne respire plus",
        "Mon proche perd connaissance",
        "Il fait une convulsion",
        "Saignement abondant",
        "Arrêt respiratoire suspecté",
        "Il ne répond plus du tout",
        "Douleur thoracique brutale et sévère",
        "Détresse respiratoire soudaine",
    ]
    contexts = [
        "dans la chambre",
        "dans le couloir",
        "à la maison",
        "pendant la nuit",
        "depuis plusieurs minutes",
    ]
    out = []
    for inc in incidents:
        out.append(f"{inc}, que faire ?")
        for c in contexts:
            out.append(f"{inc} {c}, que faire immédiatement ?")
            out.append(f"Urgence: {inc} {c}. Dois-je appeler le 15 ?")
            out.append(f"{inc} {c}, réponse rapide s'il vous plaît.")
    return out


def _fr_out_of_scope_questions() -> list[str]:
    prompts = [
        "Quel est le menu du restaurant de l'hôpital ?",
        "Peux-tu m'aider à réserver un vol ?",
        "Quel smartphone acheter en 2026 ?",
        "Donne-moi des conseils pour investir en bourse.",
        "Écris un poème romantique.",
        "Aide-moi à faire mes devoirs de maths.",
        "Quelle série regarder ce soir ?",
        "Donne-moi une recette de tiramisu.",
        "Aide-moi à optimiser mon CV.",
        "Quels sont les meilleurs hôtels à Paris ?",
    ]
    tones = ["", "Merci.", "Avec un petit budget.", "Rapidement."]
    out = []
    for p in prompts:
        for t in tones:
            out.append(f"{p} {t}".strip())
    return out


def _build_rows() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    rows += [(q, LABEL_GENERAL) for q in _fr_general_questions()]
    rows += [(q, LABEL_SENSITIVE) for q in _fr_sensitive_questions()]
    rows += [(q, LABEL_MEDICAL_ADVICE) for q in _fr_medical_advice_questions()]
    rows += [(q, LABEL_EMERGENCY) for q in _fr_emergency_questions()]
    rows += [(q, LABEL_OUT_OF_SCOPE) for q in _fr_out_of_scope_questions()]
    return rows


def _make_df(rows: list[tuple[str, str]]) -> pd.DataFrame:
    cleaner = TextCleaner()

    cleaned: list[tuple[str, str]] = []
    for text, label in rows:
        try:
            txt = cleaner.clean(str(text))
        except ValueError:
            continue
        if not _is_probably_french(txt):
            continue
        cleaned.append((txt, label))

    df = pd.DataFrame(cleaned, columns=["text", "label"])
    df["_norm"] = df["text"].map(_normalize_text)
    # Remove exact + normalized duplicates to reduce template artifacts.
    df = df.drop_duplicates(subset=["text", "label"]).drop_duplicates(subset=["_norm", "label"])
    return df.drop(columns=["_norm"]).reset_index(drop=True)


def _upsample_min_per_class(df: pd.DataFrame, min_size: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    suffixes = [
        "(reformulation clinique)",
        "(reformulation soignante)",
        "(reformulation contexte famille)",
        "(reformulation prudente)",
        "(reformulation sécurité)",
    ]
    for label, grp in df.groupby("label", sort=False):
        if len(grp) >= min_size:
            parts.append(grp)
            continue
        need = min_size - len(grp)
        extra = grp.sample(n=need, replace=True, random_state=SEED).copy()
        # Ensure each synthetic line stays unique to survive deduplication.
        new_texts = []
        for i, txt in enumerate(extra["text"].astype(str).tolist()):
            suf = suffixes[i % len(suffixes)]
            new_texts.append(f"{txt} {suf} #{i+1}")
        extra["text"] = new_texts
        parts.append(pd.concat([grp, extra], ignore_index=True))
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    return out


def main() -> None:
    rows = _build_rows()
    df = _make_df(rows)

    # Save source files expected by notebook pipeline.
    general_src = df[df["label"] == LABEL_GENERAL].copy()
    manual_src = df[df["label"] != LABEL_GENERAL].copy()
    general_src.to_csv(DATASET_DIR / "medquad_raw.csv", index=False)
    manual_src.to_csv(DATASET_DIR / "icu_manual_questions.csv", index=False)

    # Enforce robust minimum class size for training stability.
    df = _upsample_min_per_class(df, MIN_CLASS_SIZE)

    counts = df["label"].value_counts()
    min_count = int(counts.min())
    if min_count < MIN_CLASS_SIZE:
        raise RuntimeError(
            "Dataset too small for robust training: "
            f"minimum class has {min_count} rows (required >= {MIN_CLASS_SIZE}). "
            f"Counts: {counts.to_dict()}"
        )

    train_df, tmp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=SEED,
    )
    val_df, test_df = train_test_split(
        tmp_df,
        test_size=0.50,
        stratify=tmp_df["label"],
        random_state=SEED,
    )

    dataset_clean_path = DATASET_DIR / "dataset_clean.csv"
    train_path = DATASET_DIR / "train.csv"
    val_path = DATASET_DIR / "val.csv"
    test_path = DATASET_DIR / "test.csv"

    df.to_csv(dataset_clean_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Saved files:")
    for name in [
        "medquad_raw.csv",
        "icu_manual_questions.csv",
        "dataset_clean.csv",
        "train.csv",
        "val.csv",
        "test.csv",
    ]:
        p = DATASET_DIR / name
        print(f"- {name}: {p.stat().st_size} bytes")

    print("\nClass distribution (dataset_clean.csv):")
    print(df["label"].value_counts())
    print("\nSplit sizes:")
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")


if __name__ == "__main__":
    main()
