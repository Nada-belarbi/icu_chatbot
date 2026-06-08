# TextCleaner
#
# Responsibilities:
#   - Remove unnecessary whitespace and normalize encoding
#   - Preserve medical terminology and personal phrasing (e.g. "mon père")

import re
import unicodedata


class TextCleaner:
    """
    Lightweight text normalizer for user input.

    Rules applied (in order):
    1. Reject empty / whitespace-only input with ValueError
    2. NFC Unicode normalization (accents stay intact: é, è, à, ç …)
    3. Collapse multiple consecutive spaces into one
    4. Strip leading and trailing whitespace

    Intentionally NOT applied:
    - Lowercasing (classifier handles casing internally)
    - Punctuation removal (destroys sentence structure)
    - Stopword removal (removes context needed for safety classification)
    """

    def clean(self, text: str) -> str:
        """
        Normalize *text* and return the cleaned string.

        Raises:
            ValueError: if *text* is empty or contains only whitespace.
        """
        if not text or not text.strip():
            raise ValueError("Input text is empty or contains only whitespace.")

        # Step 1 — Unicode NFC normalization (preserves accented characters)
        text = unicodedata.normalize("NFC", text)

        # Step 2 — Collapse multiple spaces (including non-breaking spaces) to one
        text = re.sub(r"[ \t\u00a0]+", " ", text)

        # Step 3 — Strip edges
        text = text.strip()

        return text

#   - Raise or return a controlled error on empty input
#   - No semantic transformation — cleaning only
