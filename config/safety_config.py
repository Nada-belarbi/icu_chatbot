# Safety rules, classifier class definitions, and controlled response messages.
# Label constants are imported from the shared schema used by notebooks,
# training, evaluation, and runtime safety modules.

from config.label_schema import (
    LABEL_EMERGENCY,
    LABEL_GENERAL,
    LABEL_MEDICAL_ADVICE,
    LABEL_ORDER,
    LABEL_OUT_OF_SCOPE,
    LABEL_SENSITIVE,
)

# ---------------------------------------------------------------------------
# Classifier label constants
# ---------------------------------------------------------------------------
ALL_LABELS = list(LABEL_ORDER)

# ---------------------------------------------------------------------------
# Rule-based keyword lists (lowercased; checked before the classifier)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# IMMEDIATE_DANGER_PHRASES — subset of urgency used even for educational questions
# Only unambiguous, present-tense, life-threatening situations.
# These ALWAYS trigger urgency regardless of sentence structure.
# ---------------------------------------------------------------------------
IMMEDIATE_DANGER_PHRASES = [
    # Arrêt cardio-respiratoire
    "arrêt cardiaque", "arret cardiaque",
    "arrêt respiratoire", "arret respiratoire",
    # Détresse respiratoire active
    "ne respire plus", "ne respire pas",
    "s'arrête de respirer", "s arrete de respirer",
    "n'arrive plus à respirer", "n arrive plus a respirer",
    # Perte de conscience active
    "perd connaissance", "perd la conscience",
    "a perdu connaissance", "vient de perdre connaissance",
    # Signes visuels immédiats
    "il devient bleu", "elle devient bleue",
    # Hémorragie
    "hémorragie", "hemorragie", "saignement abondant",
    # Convulsions
    "convulsion", "convulsions", "fait des convulsions",
    # Appels urgents actifs
    "appelle le 15", "appeler le 15", "appel le 15",
    "samu", "pompiers",
    # Aggravation immédiate constatée
    "s'aggrave maintenant", "saggrave maintenant",
    "vient de faire un malaise",
]

# Full urgency keyword list — same as above but may include broader terms
# NOT used for educational questions (see IMMEDIATE_DANGER_PHRASES above)
URGENCY_KEYWORDS = IMMEDIATE_DANGER_PHRASES + [
    # Détresse respiratoire décrite
    "du mal à respirer", "a du mal a respirer", "a du mal à respirer",
    # Perte de conscience décrite
    "inconscient",
    # Malaise actif (non éducatif)
    "fait un malaise",
    # Aggravation immédiate
    "état s'aggrave maintenant", "etat s'aggrave maintenant",
]

DIAGNOSIS_KEYWORDS = [
    "diagnostiquer", "ai-je", "est-ce que j'ai", "suis-je atteint",
    "c'est quoi ma maladie", "quelle maladie",
    # Ajouts pour couvrir l'interprétation de valeurs médicales
    "taux d'oxygène", "taux d oxygene", "saturation en oxygene",
    "interpréter ses résultats", "interpreter ses resultats",
    "résultats sanguins", "resultats sanguins",
    "scanner est inquiétant", "scanner est inquietant",
    "les médecins ont raison", "les medecins ont raison",
    "avis médical", "avis medical",
]

TREATMENT_KEYWORDS = [
    "dois-je prendre", "quelle dose", "combien de médicament",
    "combien de medicament",
    "prescrire", "prescription", "ordonnance", "quel traitement pour moi",
    "puis-je arrêter", "peut-on arrêter le traitement",
    "médicament donner", "quel médicament", "quels médicaments",
    "medicament donner", "quel medicament", "quels medicaments",
    "donnez-lui ce medicament", "donnez lui ce medicament",
    "quel traitement", "quels traitements",
    # Ajouts pour couvrir les questions de traitement de la spec
    "traitement est le bon", "traitement est-il le bon",
    "traitement doit être changé", "traitement doit etre change",
    "changer le traitement", "changer son traitement",
    "faut-il accepter", "faut il accepter",
    "dois-je accepter", "doit-il accepter",
]

PROGNOSIS_KEYWORDS = [
    "vais-je survivre", "vais-je guérir", "combien de temps il me reste",
    "chance de survie", "pronostic vital", "espérance de vie",
    "va survivre", "va-t-il survivre", "va-t-elle survivre",
    "va mourir", "va-t-il mourir", "va-t-elle mourir",
    "va guérir", "va-t-il guérir", "va-t-elle guérir",
    "survivre", "chances de s'en sortir",
    # Ajouts pour couvrir les questions de pronostic de la spec
    "va rester en réanimation", "va rester en reanimation",
    "état s'aggrave", "etat s'aggrave", "s'aggrave",
    "est-ce grave", "c'est grave", "c est grave",
    "mauvais signe", "bon signe", "est-ce un bon", "est-ce un mauvais",
    "signe que ça va", "signe que ca va",
    # Taux de survie / pronostic statistique
    "taux de survie", "taux exact de survie", "taux de mortalite", "taux de mortalité",
]

PERSONAL_DATA_KEYWORDS = [
    "mon dossier", "mon numéro de sécurité sociale", "ma carte vitale",
    "mes données personnelles", "mon adresse",
    "mon numero de securite sociale", "numero de securite sociale", "mes donnees personnelles",
]

OUT_OF_SCOPE_KEYWORDS = [
    "taxi", "hotel", "hôtel", "restaurant", "voyage", "billet d'avion",
    "billet de train", "billet paris", "billet pour", "réserver un vol",
    "prix d'un vol", "combien coûte un billet", "prix du billet",
    "banque", "assurance", "cv", "emploi", "recette", "traduction",
    "meteo", "météo", "film", "shopping",
    "code python", "devoirs", "maths", "juridique", "administratif",
    "réserver une chambre", "louer une voiture",
]

# ---------------------------------------------------------------------------
# Safe allow patterns — educational questions about ICU concepts
# These bypass CamemBERT when they match but none of the blocked rules match.
# Format: list of lowercase substring patterns (no regex).
# ---------------------------------------------------------------------------
SAFE_ALLOW_PATTERNS = [
    # Formal French: qu'est-ce que...
    "qu'est-ce qu'un service",
    "qu'est-ce qu'une",
    "qu'est-ce que la",
    "qu'est-ce que le",
    "qu'est-ce que les",
    "qu'est-ce qu'un",
    # Formal: à quoi sert...
    "à quoi sert",
    "à quoi servent",
    "a quoi sert",
    "a quoi servent",
    # Formal: que signifie...
    "que signifie",
    "que veut dire",
    # Formal: quel est le rôle...
    "quel est le rôle",
    "quel est le role",
    # Formal: pourquoi un/les...
    "pourquoi un patient",
    "pourquoi les patients",
    "pourquoi les visites",
    "pourquoi les proches",
    # Formal: pourquoi un/une/le/la (general concept questions)
    "pourquoi un ",
    "pourquoi une ",
    "pourquoi le ",
    "pourquoi la ",
    "pourquoi les ",
    "pourquoi est-ce que",
    "pourquoi est-ce qu",
    # Formal: comment...
    "comment les proches",
    "comment se déroule",
    "comment se deroule",
    "comment fonctionne",
    "comment est",
    # Informal French: c'est quoi...
    "c'est quoi",
    "c est quoi",
    "c'est quoi le",
    "c'est quoi la",
    "c'est quoi un",
    "c'est quoi une",
    # Informal: pourquoi mon/ma proche...
    "pourquoi mon",
    "pourquoi ma",
    # Pouvez-vous expliquer...
    "pouvez-vous expliquer",
    "pouvez-vous m'expliquer",
    "pouvez-vous me dire ce que",
    "pouvez-vous me dire ce qu",
    # Expliquez-moi...
    "expliquez-moi",
    "expliquez moi",
    "expliquer",
    # J'aimerais / je voudrais comprendre...
    "j'aimerais comprendre",
    "j aimerais comprendre",
    "je voudrais comprendre",
    "je voulais savoir",
    "je souhaiterais savoir",
    # Informations sur...
    "informations sur",
    "information sur",
    "en quoi consiste",
    # Parlez-moi de...
    "parlez-moi",
    "parlez moi",
    # Quelle différence / comparaison entre équipements ICU
    "quelle différence",
    "quelle difference",
    "quelles différences",
    "quelles differences",
    "différence entre",
    "difference entre",
    # Puis-je / est-ce que je peux (questions pratiques familles)
    "puis-je",
    "puis je",
    "est-ce que je peux",
    "est ce que je peux",
    "est-il possible de",
    "est il possible de",
    # Qu'arrive-t-il / que se passe-t-il
    "qu'arrive-t-il",
    "que se passe-t-il",
    "qu'est-ce qui se passe",
    # Quel est le fonctionnement / la procedure
    "fonctionnement de",
    "procédure de",
    "procedure de",
    # Est-ce utile / est-ce bénéfique (questions pratiques familles en réanimation)
    "est-ce utile",
    "est ce utile",
    "est-ce bénéfique",
    "est ce benefique",
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

