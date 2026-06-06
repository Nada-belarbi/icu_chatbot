# CamembertQueryClassifier
#
# Responsibilities:
#   - Load fine-tuned CamemBERT model from models/camembert_classifier/
#   - Classify a cleaned query into one of:
#       information_generale | sensible | avis_medical_personnalise | urgence | hors_perimetre
#   - Return (label: str, confidence_score: float)
#   - In early development: swap with a MockClassifier that returns hardcoded results
#
# NOTE: The mock classifier must have the same interface as the real one
