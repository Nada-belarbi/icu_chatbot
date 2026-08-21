# RuleFilter
#
# Responsibilities:
#   - Fast keyword/regex-based detection BEFORE calling CamemBERT or the LLM
#   - Detects: urgency, diagnosis, treatment, prognosis, personal data, out-of-scope
#   - Also detects: force_allow (clearly educational questions that bypass CamemBERT)
#
# Logic order (changed from v1):
#   1. Check SAFE_ALLOW_PATTERNS first.
#   2. If educational question detected:
#       - Check immediate-danger phrases first (unambiguous life-threatening situations).
#       - Check all other deterministic safety rules.
#       - Only a question free from every blocking rule becomes force_allow.
#   3. If not educational: check all blocking rules normally (urgency first).

import re
from config.safety_config import (
    IMMEDIATE_DANGER_PHRASES,
    URGENCY_KEYWORDS,
    DIAGNOSIS_KEYWORDS,
    TREATMENT_KEYWORDS,
    PROGNOSIS_KEYWORDS,
    PERSONAL_DATA_KEYWORDS,
    OUT_OF_SCOPE_KEYWORDS,
    SAFE_ALLOW_PATTERNS,
)

# Full blocking rules checked for non-educational queries — urgency MUST be first
_RULES: list[tuple[str, list[str]]] = [
    ("urgency",       URGENCY_KEYWORDS),
    ("diagnosis",     DIAGNOSIS_KEYWORDS),
    ("treatment",     TREATMENT_KEYWORDS),
    ("prognosis",     PROGNOSIS_KEYWORDS),
    ("out_of_scope",  OUT_OF_SCOPE_KEYWORDS),
    ("personal_data", PERSONAL_DATA_KEYWORDS),
]


def _normalize(text: str) -> str:
    """Lowercase + normalize typographic apostrophes."""
    return (
        text.lower()
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u02bc", "'")
    )


class RuleFilter:
    """
    Deterministic pre-filter based on keyword lists from ``safety_config``.

    Returns
    -------
    tuple[str, bool]
        ``(category, blocked)``

        - *category* is one of: ``urgency``, ``diagnosis``, ``treatment``,
          ``prognosis``, ``personal_data``, ``out_of_scope``, ``allowed``,
          ``force_allow``.
        - *blocked* is ``True`` for every category except ``allowed`` and
          ``force_allow``.

    For clearly educational questions (matching SAFE_ALLOW_PATTERNS),
    unambiguous immediate-danger phrases still trigger urgency and every other
    deterministic safety rule remains active. Educational wording must never
    turn a request for diagnosis, treatment, prognosis, personal data, or an
    unrelated service into an allowed query.
    """

    def filter(self, text: str) -> tuple[str, bool]:
        lower = _normalize(text)

        # ----------------------------------------------------------------
        # Step 1 — detect educational / explanatory question
        # ----------------------------------------------------------------
        is_educational = any(pat.lower() in lower for pat in SAFE_ALLOW_PATTERNS)

        if is_educational:
            # Even for educational questions, unambiguous immediate danger
            # (e.g. "il ne respire plus") must still trigger urgency.
            for kw in IMMEDIATE_DANGER_PHRASES:
                if re.search(re.escape(kw.lower()), lower):
                    return ("urgency", True)

            # A question may sound explanatory while still requesting an
            # individual diagnosis, a treatment, a prognosis, private data or
            # an unrelated task. Do not let the wording bypass those gates.
            for category, keywords in _RULES[1:]:
                for kw in keywords:
                    if re.search(re.escape(kw.lower()), lower):
                        return (category, True)

            # Safe educational question — skip classifier entirely.
            return ("force_allow", False)

        # ----------------------------------------------------------------
        # Step 2 — non-educational: check all blocking rules (urgency first)
        # ----------------------------------------------------------------
        for category, keywords in _RULES:
            for kw in keywords:
                pattern = re.escape(kw.lower())
                if re.search(pattern, lower):
                    return (category, True)

        return ("allowed", False)

#   - Returns: (category: str, blocked: bool)
#   - Rules are defined in config/safety_config.py — never hardcoded here
