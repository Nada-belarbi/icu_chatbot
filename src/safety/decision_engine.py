# DecisionEngine
#
# Responsibilities:
#   - Combine RuleFilter output + CamemBERT (label, score) into a single decision
#   - Possible outputs: allow_rag | refuse | redirect | ask_reformulation
#   - Decision logic:
#       urgency (any source)         → redirect
#       low confidence score         → ask_reformulation
#       information_generale + high  → allow_rag
#       avis_medical/hors_perimetre  → refuse
#   - Confidence threshold read from config/settings.py
