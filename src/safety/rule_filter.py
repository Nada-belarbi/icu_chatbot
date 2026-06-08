# RuleFilter
#
# Responsibilities:
#   - Fast keyword/regex-based detection BEFORE calling CamemBERT or the LLM
#   - Detects: urgency, diagnosis, treatment, prognosis, personal data, out-of-scope

import re
from config.safety_config import (
    URGENCY_KEYWORDS,
    DIAGNOSIS_KEYWORDS,
    TREATMENT_KEYWORDS,
    PROGNOSIS_KEYWORDS,
    PERSONAL_DATA_KEYWORDS,
)

# Ordered list of (category, keyword_list) — urgency MUST be first
_RULES: list[tuple[str, list[str]]] = [
    ("urgency",       URGENCY_KEYWORDS),
    ("diagnosis",     DIAGNOSIS_KEYWORDS),
    ("treatment",     TREATMENT_KEYWORDS),
    ("prognosis",     PROGNOSIS_KEYWORDS),
    ("personal_data", PERSONAL_DATA_KEYWORDS),
]


class RuleFilter:
    """
    Deterministic pre-filter based on keyword lists from ``safety_config``.

    Returns
    -------
    tuple[str, bool]
        ``(category, blocked)``

        - *category* is one of: ``urgency``, ``diagnosis``, ``treatment``,
          ``prognosis``, ``personal_data``, ``allowed``.
        - *blocked* is ``True`` for every category except ``allowed``.

    Urgency always takes priority over any other matching category.
    Matching is case-insensitive.
    """

    def filter(self, text: str) -> tuple[str, bool]:
        """
        Screen *text* against all keyword rules.

        Parameters
        ----------
        text:
            Cleaned user input (output of ``TextCleaner.clean()``).

        Returns
        -------
        tuple[str, bool]
            ``(category, blocked)``.
        """
        lower = text.lower()

        for category, keywords in _RULES:
            for kw in keywords:
                # Keyword may contain spaces — use simple substring match
                # with word-boundary awareness via \b where possible
                pattern = re.escape(kw.lower())
                if re.search(pattern, lower):
                    return (category, True)

        return ("allowed", False)

#   - Returns: (category: str, blocked: bool)
#   - Rules are defined in config/safety_config.py — never hardcoded here
