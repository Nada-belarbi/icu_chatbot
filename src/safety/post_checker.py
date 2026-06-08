# PostGenerationChecker
#
# Responsibilities:
#   - Scan the LLM response BEFORE it is shown to the user
#   - Detect and reject responses that contain:
#       - Personalized medical advice
#       - Diagnosis or prognosis statements
#       - Survival predictions or duration estimates
#       - Treatment recommendations
#       - Overly affirmative phrasing

from __future__ import annotations

import re

from config.safety_config import DANGEROUS_RESPONSE_PATTERNS
from src.safety.controlled_responses import ControlledResponseManager


class PostGenerationChecker:
    """
    Last-resort safety gate applied to every LLM-generated answer.

    Scans for dangerous content using the regex patterns defined in
    ``config/safety_config.py``.  If any pattern matches, the raw answer
    is replaced by a pre-written controlled response.
    """

    def __init__(self) -> None:
        self._patterns = [
            re.compile(p, re.IGNORECASE | re.UNICODE)
            for p in DANGEROUS_RESPONSE_PATTERNS
        ]
        self._responses = ControlledResponseManager()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, response: str) -> tuple[str, bool]:
        """
        Scan *response* for dangerous content.

        Returns
        -------
        (safe_response, was_modified)
            If the response is safe, returns it unchanged and ``False``.
            If a dangerous pattern is found, returns the controlled refusal
            and ``True``.
        """
        for pattern in self._patterns:
            if pattern.search(response):
                return self._responses.get_refuse_medical_advice(), True
        return response, False

#       - No documentary source
#       - Out-of-scope content
#   - If unsafe: replace with a controlled refusal/redirection message
#   - Returns: (safe_response: str, was_modified: bool)
