# Safety rules, classifier class definitions, and controlled response messages
# Centralises all safety-related constants so they are easy to audit and update

# ---------------------------------------------------------------------------
# Classifier label constants
# ---------------------------------------------------------------------------
LABEL_GENERAL = "information_generale"
LABEL_SENSITIVE = "sensible"
LABEL_MEDICAL_ADVICE = "avis_medical_personnalise"
LABEL_EMERGENCY = "urgence"
LABEL_OUT_OF_SCOPE = "hors_perimetre"

ALL_LABELS = [
    LABEL_GENERAL,
    LABEL_SENSITIVE,
    LABEL_MEDICAL_ADVICE,
    LABEL_EMERGENCY,
    LABEL_OUT_OF_SCOPE,
]

# ---------------------------------------------------------------------------
# Rule-based keyword lists (lowercased; checked before the classifier)
# ---------------------------------------------------------------------------

URGENCY_KEYWORDS = [
    "urgence", "urgent", "mourir", "mort", "décède", "décéder",
    "inconscient", "ne répond plus", "arrêt cardiaque", "arrêt respiratoire",
    "hémorragie", "saignement abondant", "convulsion", "convulsions",
    "appel le 15", "samu", "pompiers", "18", "15", "112",
    "ne respire plus", "ne respire pas", "s'arrête de respirer",
    "perd connaissance", "perd la conscience",
]

DIAGNOSIS_KEYWORDS = [
    "diagnostiquer", "ai-je", "est-ce que j'ai", "suis-je atteint",
    "c'est quoi ma maladie", "quelle maladie",
]

TREATMENT_KEYWORDS = [
    "dois-je prendre", "quelle dose", "combien de médicament",
    "prescrire", "prescription", "ordonnance", "quel traitement pour moi",
    "puis-je arrêter", "peut-on arrêter le traitement",
    "médicament donner", "quel médicament", "quels médicaments",
    "quel traitement", "quels traitements",
]

PROGNOSIS_KEYWORDS = [
    "vais-je survivre", "vais-je guérir", "combien de temps il me reste",
    "chance de survie", "pronostic vital", "espérance de vie",
    "va survivre", "va-t-il survivre", "va-t-elle survivre",
    "va guérir", "va-t-il guérir", "va-t-elle guérir",
    "survivre", "chances de s'en sortir",
]

PERSONAL_DATA_KEYWORDS = [
    "mon dossier", "mon numéro de sécurité sociale", "ma carte vitale",
    "mes données personnelles", "mon adresse",
]

# ---------------------------------------------------------------------------
# Dangerous response patterns (used by PostGenerationChecker)
# Regex patterns matched against the LLM's raw output.
# ---------------------------------------------------------------------------
DANGEROUS_RESPONSE_PATTERNS = [
    # Diagnostic statements
    r"vous\s+avez\s+(certainement|probablement|sûrement)\s+\w+",
    r"il\s+s['']agit\s+d['']une?\s+\w+",
    r"c['']est\s+(certainement|probablement)\s+\w+",
    # Dosage / prescriptions
    r"\d+\s*(mg|ml|g|µg)\s*(par jour|\/jour|\/j|toutes les)",
    r"prenez\s+\d+",
    r"dose\s+(recommandée|habituelle|standard)\s+est",
    # Prognosis / survival predictions
    r"vous\s+avez\s+\d+\s*%\s+(de chance|de survie)",
    r"il\s+(vous\s+)?reste\s+\d+",
    r"il\s+va\s+(survivre|s['']en\s+sortir|guérir)",
    r"elle\s+va\s+(survivre|s['']en\s+sortir|guérir)",
    # Direct medical advice / treatment instructions
    r"je\s+vous\s+(conseille|recommande)\s+de\s+(prendre|arrêter|augmenter|diminuer)",
    r"vous\s+devriez\s+(prendre|arrêter|augmenter|diminuer)",
    r"donnez[- ]lui\s+(ce|un|des|le|la|les|\w+\s+(médicament|traitement|dose|comprimé))",
]

