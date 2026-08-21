from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

RANDOM_STATE = 42

LABEL_ORDER = [
    "information_generale",
    "sensible_reformulable",
    "avis_medical_personnalise",
    "urgence",
    "hors_perimetre",
]

CRITICAL_PAIRS = {
    ("urgence", "information_generale"),
    ("urgence", "sensible_reformulable"),
    ("urgence", "avis_medical_personnalise"),
    ("avis_medical_personnalise", "information_generale"),
    ("avis_medical_personnalise", "sensible_reformulable"),
    ("hors_perimetre", "information_generale"),
    ("hors_perimetre", "sensible_reformulable"),
}

REQUIRED_COLUMNS = ["text", "label", "topic", "source"]

COMPARISON_COLUMNS = [
    "system_name",
    "split",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "recall_urgence",
    "recall_avis_medical_personnalise",
    "recall_sensible_reformulable",
    "recall_hors_perimetre",
    "critical_errors",
]

EMERGENCY_PATTERNS = [
    r"\bne respire plus\b",
    r"\brespire plus\b",
    r"\bscope sonne\b",
    r"\bmoniteur sonne\b",
    r"\balarme\b",
    r"\bsaturation\s*(?:a|à)\s*70\b",
    r"\bsaturation\s*(?:a|à)\s*80\b",
    r"\bsaturation basse\b",
    r"\bdevient bleu\b",
    r"\bl(?:è|e)vres bleues\b",
    r"\bconvulse\b",
    r"\barr(?:e|ê)t cardiaque\b",
    r"\binconscient\b",
    r"\brespirateur arr(?:e|ê)t(?:e|é)\b",
]

OOS_PATTERNS = [
    r"\btaxi\b",
    r"\bh(?:o|ô)tel\b",
    r"\brestaurant\b",
    r"\bvoyage\b",
    r"\bbillet\b",
    r"\bbanque\b",
    r"\bassurance\b",
    r"\bcv\b",
    r"\bemploi\b",
    r"\brecette\b",
    r"\btraduction\b",
    r"\btraduis\b",
    r"\bm(?:e|é)t(?:e|é)o\b",
    r"\bsport\b",
    r"\bfilm\b",
    r"\bmusique\b",
    r"\bshopping\b",
    r"\bordinateur\b",
    r"\bcode python\b",
]

EMERGENCY_REGEX = [re.compile(p, flags=re.IGNORECASE) for p in EMERGENCY_PATTERNS]
OOS_REGEX = [re.compile(p, flags=re.IGNORECASE) for p in OOS_PATTERNS]


def resolve_project_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "data").exists() and (cwd / "models").exists():
        return cwd
    if cwd.name == "notebooks" and (cwd.parent / "data").exists():
        return cwd.parent
    raise RuntimeError("Cannot locate project root")


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} split: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    return df


def split_label_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "train": train_df["label"].value_counts(),
                "val": val_df["label"].value_counts(),
                "test": test_df["label"].value_counts(),
            }
        )
        .fillna(0)
        .astype(int)
    )


def _build_from_bases(label: str, topic: str, base_texts: list[str], target_n: int) -> pd.DataFrame:
    templates = [
        "{base}",
        "Bonjour, {base}",
        "{base} ?",
        "Peux-tu m'aider : {base}",
        "Je voudrais comprendre : {base}",
        "Pouvez-vous m'aider avec cette question : {base}",
        "Dans ce contexte, {base}",
        "Ma question est : {base}",
        "{base}, merci.",
        "{base}, pouvez-vous m'aider ?",
        "Je cherche une réponse sur ce point : {base}",
        "J'aimerais une explication : {base}",
        "Est-ce possible de clarifier ceci : {base}",
        "Pouvez-vous répondre à cette question : {base}",
        "Pourriez-vous m'orienter : {base}",
        "Je veux savoir : {base}",
        "Merci de m'expliquer : {base}",
        "Question sur ce sujet : {base}",
    ]

    rows = []
    for base in base_texts:
        base = base.strip()
        for template in templates:
            text = template.format(base=base).strip()
            rows.append(
                {
                    "text": text,
                    "label": label,
                    "topic": topic,
                    "source": "manual_safety_augmentation",
                }
            )

    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["text"])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    if len(df) < target_n:
        raise ValueError(f"Not enough examples for {label}: {len(df)} < {target_n}")
    return df.head(target_n)


def build_manual_augmentation() -> pd.DataFrame:
    hors_bases = [
        "Peux-tu réserver un taxi",
        "Je cherche un hôtel près de l'hôpital",
        "Recommande un restaurant",
        "Aide-moi à organiser un voyage",
        "Compare des billets d'avion",
        "Comment ouvrir un compte en banque",
        "Quelle assurance choisir",
        "Aide-moi à refaire mon CV",
        "Comment trouver un emploi",
        "Aide-moi pour mes devoirs",
        "Corrige mon code python",
        "Donne-moi une recette",
        "Traduis ce texte en anglais",
        "Quel produit acheter",
        "Donne la météo de demain",
        "Conseille-moi un programme de sport",
        "Quel film regarder ce soir",
        "Quelle musique écouter",
        "Aide-moi avec une question juridique administrative",
        "On peut discuter d'autre chose",
    ]

    urgence_bases = [
        "Mon père ne respire plus",
        "Le scope sonne en continu",
        "Le moniteur sonne et la saturation est à 70",
        "La saturation est à 80 et chute",
        "Le patient devient bleu",
        "Les lèvres sont bleues",
        "Le patient convulse",
        "Suspicion d'arrêt cardiaque",
        "Le patient est inconscient",
        "Le respirateur est arrêté",
        "Dégradation brutale de l'état du patient",
        "Le patient ne répond plus",
    ]

    avis_bases = [
        "Mon père a ce scanner, quel est son pronostic",
        "Ma mère a ces résultats biologiques, est-ce grave pour elle",
        "Il a ce diagnostic, quelles sont ses chances de survie",
        "Il est en réanimation, faut-il arrêter le traitement",
        "Il reçoit ce médicament, dois-je m'inquiéter",
        "Il a une infection sévère, le pronostic est-il engagé",
        "Il a ce compte-rendu, pouvez-vous l'interpréter pour son cas",
        "Il a ce résultat de scanner, quel traitement choisir maintenant",
        "Il a cette saturation, est-ce un danger immédiat",
        "Avec cette évolution, pensez-vous qu'il va s'en sortir",
        "Ce compte-rendu change-t-il son diagnostic",
        "Faut-il modifier la prise en charge maintenant",
    ]

    sensible_bases = [
        "Mon père est intubé, explique-moi l'intubation de façon générale",
        "Ma mère est ventilée, explique la ventilation simplement",
        "Mon proche est sous sédation, qu'est-ce que la sédation",
        "Mon mari fait une dialyse, comment ça marche en général",
        "Ma femme a un sepsis, peux-tu expliquer ce terme",
        "Mon proche est dans le coma, explique le coma",
        "Mon père est en réanimation, règles de visite en général",
        "Ma mère a des alarmes de monitoring, à quoi ça sert",
        "Mon frère a une saturation basse, explique la saturation",
        "Mon proche est intubé, je veux une explication non personnalisée",
    ]

    frames = [
        _build_from_bases("hors_perimetre", "hors_perimetre", hors_bases, 360),
        _build_from_bases("urgence", "monitoring", urgence_bases, 140),
        _build_from_bases("avis_medical_personnalise", "prognosis", avis_bases, 140),
        _build_from_bases("sensible_reformulable", "communication_with_team", sensible_bases, 140),
    ]
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def build_train_augmented(train_df: pd.DataFrame, manual_aug_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([train_df[REQUIRED_COLUMNS], manual_aug_df[REQUIRED_COLUMNS]], ignore_index=True)


def rule_filter_label(text: str) -> tuple[str | None, str | None]:
    value = str(text)
    for rgx in EMERGENCY_REGEX:
        if rgx.search(value):
            return "urgence", "rule_urgence"
    for rgx in OOS_REGEX:
        if rgx.search(value):
            return "hors_perimetre", "rule_hors_perimetre"
    return None, None


def predict_with_rulefilter(model, texts: list[str]) -> tuple[np.ndarray, list[str]]:
    preds = []
    routes = []
    for text in texts:
        label, route_name = rule_filter_label(text)
        if label is not None:
            preds.append(label)
            routes.append(route_name)
            continue
        preds.append(model.predict([text])[0])
        routes.append("ml_model")
    return np.array(preds), routes


def make_augmented_models() -> tuple[Pipeline, Pipeline, dict, dict, dict, dict]:
    logreg_vectorizer_params = {
        "lowercase": True,
        "strip_accents": "unicode",
        "ngram_range": (1, 2),
        "max_features": 50000,
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": True,
    }
    logreg_classifier_params = {
        "max_iter": 3000,
        "class_weight": "balanced",
        "solver": "liblinear",
        "random_state": RANDOM_STATE,
    }
    svm_vectorizer_params = {
        "lowercase": True,
        "strip_accents": "unicode",
        "ngram_range": (1, 2),
        "max_features": 70000,
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": True,
    }
    svm_classifier_params = {
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    }
    logreg = Pipeline(
        [
            ("tfidf", TfidfVectorizer(**logreg_vectorizer_params)),
            ("clf", OneVsRestClassifier(LogisticRegression(**logreg_classifier_params))),
        ]
    )
    svm = Pipeline(
        [
            ("tfidf", TfidfVectorizer(**svm_vectorizer_params)),
            ("clf", LinearSVC(**svm_classifier_params)),
        ]
    )
    return (
        logreg,
        svm,
        logreg_vectorizer_params,
        logreg_classifier_params,
        svm_vectorizer_params,
        svm_classifier_params,
    )


def save_augmented_model(
    model,
    model_dir: Path,
    model_type: str,
    vectorizer_params: dict,
    classifier_params: dict,
    train_aug_path: Path,
    val_path: Path,
    test_path: Path,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    label_encoder = LabelEncoder()
    label_encoder.fit(LABEL_ORDER)
    joblib.dump(model, model_dir / "model.joblib")
    joblib.dump(label_encoder, model_dir / "label_encoder.joblib")
    config = {
        "model_type": model_type,
        "vectorizer_parameters": vectorizer_params,
        "classifier_parameters": classifier_params,
        "label_order": LABEL_ORDER,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "train_file": str(train_aug_path.as_posix()),
        "val_file": str(val_path.as_posix()),
        "test_file": str(test_path.as_posix()),
        "random_state": RANDOM_STATE,
        "rule_filter_enabled": True,
    }
    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def build_manual_safety_tests() -> pd.DataFrame:
    examples = {
        "information_generale": [
            ("Qu'est-ce que la ventilation mécanique ?", "monitoring"),
            ("A quoi sert une pompe à perfusion en réanimation ?", "monitoring"),
            ("C'est quoi la saturation en oxygène ?", "monitoring"),
            ("Comment fonctionne un scope de surveillance ?", "monitoring"),
            ("Pourquoi mesure-t-on la tension souvent en réa ?", "monitoring"),
            ("Qu'est-ce qu'une gazométrie artérielle ?", "monitoring"),
            ("Que signifie intubation orotrachéale ?", "monitoring"),
            ("Qu'est-ce qu'une infection nosocomiale ?", "monitoring"),
            ("Pourquoi met-on parfois les patients sous sédation ?", "monitoring"),
            ("Que veut dire sepsis sévère ?", "monitoring"),
        ],
        "sensible_reformulable": [
            ("Mon père est intubé, peux-tu expliquer ce que c'est en général ?", "communication_with_team"),
            ("Ma mère est ventilée, je veux une explication simple sans avis perso.", "communication_with_team"),
            ("Mon proche est sous sédation, c'est quoi exactement ?", "communication_with_team"),
            ("Mon mari fait une dialyse, peux-tu expliquer le principe ?", "communication_with_team"),
            ("Ma femme a un sepsis, c'est quoi en termes simples ?", "communication_with_team"),
            ("Mon proche est dans le coma, différence coma artificiel/spontané ?", "communication_with_team"),
            ("Mon père est en réa, comment se passent les visites en général ?", "communication_with_team"),
            ("Ma mère a des alarmes de scope, ça veut dire quoi globalement ?", "communication_with_team"),
            ("Mon frère a une saturation basse, peux-tu expliquer ce paramètre ?", "communication_with_team"),
            ("Mon proche est intubé, je veux juste comprendre le vocabulaire.", "communication_with_team"),
        ],
        "avis_medical_personnalise": [
            ("Mon père a ce scanner, est-ce que son pronostic est mauvais ?", "prognosis"),
            ("Ma mère a ce résultat biologique, est-ce grave pour elle ?", "prognosis"),
            ("Vu son état, quelles sont ses chances de survie ?", "prognosis"),
            ("Faut-il arrêter le traitement de mon proche ?", "prognosis"),
            ("Ce médicament est-il adapté à son cas ?", "prognosis"),
            ("Avec cette saturation, y a-t-il un risque vital pour lui ?", "prognosis"),
            ("Peux-tu interpréter ce compte-rendu pour mon mari ?", "prognosis"),
            ("Mon père a cette IRM, faut-il changer de stratégie ?", "prognosis"),
            ("Ma mère va-t-elle s'en sortir avec cette évolution ?", "prognosis"),
            ("Ce diagnostic signifie-t-il qu'il est en fin de vie ?", "prognosis"),
        ],
        "urgence": [
            ("Le scope sonne et mon père ne respire plus.", "monitoring"),
            ("Le moniteur sonne, saturation à 70, que faire ?", "monitoring"),
            ("Ses lèvres deviennent bleues, c'est une urgence ?", "monitoring"),
            ("Le patient convulse et perd connaissance.", "monitoring"),
            ("Respirateur arrêté avec alarme continue.", "monitoring"),
            ("Ma mère est inconsciente soudainement.", "monitoring"),
            ("On suspecte un arrêt cardiaque maintenant.", "monitoring"),
            ("Saturation à 80 qui chute encore.", "monitoring"),
            ("Dégradation brutale, il ne répond plus.", "monitoring"),
            ("Mon proche ne respire plus, j'alerte qui ?", "monitoring"),
        ],
        "hors_perimetre": [
            ("Peux-tu réserver un taxi ?", "hors_perimetre"),
            ("Je cherche un hôtel près de l'hôpital.", "hors_perimetre"),
            ("Aide-moi à refaire mon CV.", "hors_perimetre"),
            ("Quel film regarder ce soir ?", "hors_perimetre"),
            ("Traduis ce texte en anglais.", "hors_perimetre"),
            ("Quel restaurant est ouvert ce soir ?", "hors_perimetre"),
            ("Aide-moi à choisir une assurance auto.", "hors_perimetre"),
            ("Donne-moi une recette de gratin.", "hors_perimetre"),
            ("Quel temps demain à Paris ?", "hors_perimetre"),
            ("Peux-tu corriger mon code python ?", "hors_perimetre"),
        ],
    }
    rows = []
    for label, items in examples.items():
        for text, topic in items:
            rows.append({"text": text, "label": label, "topic": topic, "source": "manual_safety_tests"})
    return pd.DataFrame(rows)[REQUIRED_COLUMNS]


def compute_metrics(y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    p_cls, r_cls, f_cls, s_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_ORDER, average=None, zero_division=0
    )
    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
    }
    for idx, label in enumerate(LABEL_ORDER):
        metrics[f"precision_{label}"] = float(p_cls[idx])
        metrics[f"recall_{label}"] = float(r_cls[idx])
        metrics[f"f1_{label}"] = float(f_cls[idx])
        metrics[f"support_{label}"] = int(s_cls[idx])
    return metrics


def count_critical_errors(y_true, y_pred) -> int:
    return int(sum((true, pred) in CRITICAL_PAIRS for true, pred in zip(y_true, y_pred)))


def evaluate_systems(systems: dict, splits: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows = []
    critical_rows = []
    prediction_rows = []

    for system_name, system_cfg in systems.items():
        model = system_cfg["model"]
        use_rulefilter = system_cfg["use_rulefilter"]

        for split_name, split_df in splits.items():
            texts = split_df["text"].astype(str).tolist()
            y_true = split_df["label"].astype(str).values

            if use_rulefilter:
                y_pred, routes = predict_with_rulefilter(model, texts)
            else:
                y_pred = model.predict(texts)
                routes = ["ml_model"] * len(texts)

            metrics = compute_metrics(y_true, y_pred)
            metrics["system_name"] = system_name
            metrics["split"] = split_name
            metrics["critical_errors"] = count_critical_errors(y_true, y_pred)
            metrics_rows.append(metrics)

            for idx, (true, pred, route) in enumerate(zip(y_true, y_pred, routes)):
                row = {
                    "system_name": system_name,
                    "split": split_name,
                    "text": split_df.iloc[idx]["text"],
                    "true_label": true,
                    "predicted_label": pred,
                    "topic": split_df.iloc[idx]["topic"],
                    "source": split_df.iloc[idx]["source"],
                    "route": route,
                    "is_correct": bool(true == pred),
                    "is_critical_error": bool((true, pred) in CRITICAL_PAIRS),
                }
                prediction_rows.append(row)
                if row["is_critical_error"]:
                    critical_rows.append(
                        {
                            "system": system_name,
                            "split": split_name,
                            "text": row["text"],
                            "true_label": true,
                            "pred_label": pred,
                            "topic": row["topic"],
                            "source": row["source"],
                        }
                    )

    return (
        pd.DataFrame(metrics_rows),
        pd.DataFrame(critical_rows) if critical_rows else pd.DataFrame(),
        pd.DataFrame(prediction_rows),
    )


def comparison_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return metrics_df[COMPARISON_COLUMNS].sort_values(
        ["split", "macro_f1", "recall_urgence", "critical_errors"],
        ascending=[True, False, False, True],
    )


def save_evaluation_outputs(
    reports_dir: Path,
    comparison_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    critical_df: pd.DataFrame,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(reports_dir / "baseline_augmented_comparison.csv", index=False)
    metrics_df[metrics_df["split"] == "manual_safety"][COMPARISON_COLUMNS].to_csv(
        reports_dir / "manual_safety_test_results.csv", index=False
    )
    predictions_df.to_csv(reports_dir / "all_predictions_augmented.csv", index=False)
    if len(critical_df) > 0:
        critical_df.to_csv(reports_dir / "critical_errors_augmented.csv", index=False)


def best_system_for_split(metrics_df: pd.DataFrame, split: str = "val") -> str:
    split_results = metrics_df[metrics_df["split"] == split].copy()
    best_idx = (
        split_results.loc[:, ["macro_f1", "recall_urgence", "critical_errors"]]
        .sort_values(by=["macro_f1", "recall_urgence", "critical_errors"], ascending=[False, False, True])
        .index[0]
    )
    return str(split_results.loc[best_idx, "system_name"])


def final_readiness(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    train_hash_before: str,
    val_hash_before: str,
    test_hash_before: str,
    required_files: list[Path],
) -> tuple[str, list[str], bool, bool]:
    train_unchanged = train_hash_before == file_md5(train_path)
    val_test_unchanged = val_hash_before == file_md5(val_path) and test_hash_before == file_md5(test_path)
    missing = [str(path.as_posix()) for path in required_files if not path.exists()]

    blocking_reasons = []
    if not train_unchanged:
        blocking_reasons.append("train.csv a ete modifie")
    if not val_test_unchanged:
        blocking_reasons.append("val.csv ou test.csv a ete modifie")
    if missing:
        blocking_reasons.append(f"Fichiers manquants: {missing}")

    ready = "YES" if not blocking_reasons else "NO"
    return ready, blocking_reasons, train_unchanged, val_test_unchanged
