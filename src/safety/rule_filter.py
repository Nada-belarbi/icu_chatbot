# RuleFilter
#
# Responsibilities:
#   - Fast keyword/regex-based detection BEFORE calling CamemBERT or the LLM
#   - Detects: urgency, diagnosis, treatment, prognosis, personal data, out-of-scope
#   - Returns: (category: str, blocked: bool)
#   - Rules are defined in config/safety_config.py — never hardcoded here
