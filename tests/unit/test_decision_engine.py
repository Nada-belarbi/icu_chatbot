# Unit tests — DecisionEngine
# Cases to cover:
#   - urgency rule → redirect (regardless of CamemBERT score)
#   - CamemBERT score < threshold → ask_reformulation
#   - information_generale + high score → allow_rag
#   - avis_medical_personnalise → refuse
#   - hors_perimetre → refuse
