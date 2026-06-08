# Unit tests — TextCleaner
# Cases to cover:
#   - empty string → ValueError
#   - whitespace-only → ValueError
#   - multiple spaces → collapsed to one
#   - leading/trailing whitespace → stripped
#   - personal phrasing ("mon père") → preserved unchanged
#   - medical term ("réanimation") → not altered
#   - NFC normalization → accents intact

import pytest
from src.preprocessing.text_cleaner import TextCleaner


@pytest.fixture
def cleaner():
    return TextCleaner()


class TestTextCleanerEmptyInput:
    def test_empty_string_raises(self, cleaner):
        with pytest.raises(ValueError):
            cleaner.clean("")

    def test_whitespace_only_raises(self, cleaner):
        with pytest.raises(ValueError):
            cleaner.clean("   ")

    def test_tab_only_raises(self, cleaner):
        with pytest.raises(ValueError):
            cleaner.clean("\t\t")


class TestTextCleanerWhitespace:
    def test_strips_leading_trailing(self, cleaner):
        assert cleaner.clean("  bonjour  ") == "bonjour"

    def test_collapses_multiple_spaces(self, cleaner):
        assert cleaner.clean("bonjour   monde") == "bonjour monde"

    def test_collapses_tabs_to_space(self, cleaner):
        assert cleaner.clean("bonjour\tmonde") == "bonjour monde"

    def test_mixed_whitespace(self, cleaner):
        assert cleaner.clean("  bonjour   monde  ") == "bonjour monde"


class TestTextCleanerPreservation:
    def test_personal_phrasing_preserved(self, cleaner):
        text = "mon père est intubé"
        assert cleaner.clean(text) == text

    def test_medical_term_preserved(self, cleaner):
        text = "réanimation polyvalente"
        assert cleaner.clean(text) == text

    def test_accents_preserved(self, cleaner):
        text = "échéance à prévoir"
        assert cleaner.clean(text) == text

    def test_punctuation_preserved(self, cleaner):
        text = "Est-ce grave ?"
        assert cleaner.clean(text) == text

    def test_casing_preserved(self, cleaner):
        text = "SAMU"
        assert cleaner.clean(text) == text


class TestTextCleanerUnicode:
    def test_nfc_normalization(self, cleaner):
        # NFD form of é (e + combining acute) → should normalize to NFC é
        nfd_e = "e\u0301"
        result = cleaner.clean(nfd_e + "lève")
        assert result == "élève"

    def test_non_breaking_space_collapsed(self, cleaner):
        # U+00A0 non-breaking space → collapsed to regular space
        text = "bonjour\u00a0monde"
        assert cleaner.clean(text) == "bonjour monde"

#   - medical terminology → not stripped
#   - encoding edge cases
