# Unit tests — BaseQueryClassifier interface + MockQueryClassifier
# Cases:
#   - MockQueryClassifier satisfies BaseQueryClassifier interface
#   - Default label/score returned correctly
#   - Configurable label and score work
#   - Invalid label raises ValueError
#   - Invalid score raises ValueError
#   - Factory convenience methods return correct labels

import pytest
from src.safety.query_classifier import BaseQueryClassifier, MockQueryClassifier
from config.safety_config import (
    LABEL_GENERAL, LABEL_SENSITIVE, LABEL_MEDICAL_ADVICE,
    LABEL_EMERGENCY, LABEL_OUT_OF_SCOPE, ALL_LABELS,
)


class TestInterface:
    def test_mock_is_subclass_of_base(self):
        assert issubclass(MockQueryClassifier, BaseQueryClassifier)

    def test_mock_instance_satisfies_interface(self):
        clf = MockQueryClassifier()
        assert isinstance(clf, BaseQueryClassifier)

    def test_classify_returns_tuple(self):
        clf = MockQueryClassifier()
        result = clf.classify("quelque chose")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_classify_label_is_string(self):
        clf = MockQueryClassifier()
        label, _ = clf.classify("test")
        assert isinstance(label, str)

    def test_classify_score_is_float(self):
        clf = MockQueryClassifier()
        _, score = clf.classify("test")
        assert isinstance(score, float)


class TestDefaults:
    def test_default_label_is_general(self):
        clf = MockQueryClassifier()
        label, _ = clf.classify("n'importe quoi")
        assert label == LABEL_GENERAL

    def test_default_score_is_095(self):
        clf = MockQueryClassifier()
        _, score = clf.classify("n'importe quoi")
        assert score == 0.95

    def test_text_does_not_affect_output(self):
        clf = MockQueryClassifier()
        assert clf.classify("a") == clf.classify("question très longue et complexe")


class TestConfigurable:
    @pytest.mark.parametrize("label", ALL_LABELS)
    def test_all_valid_labels_accepted(self, label):
        clf = MockQueryClassifier(label=label, score=0.80)
        result_label, _ = clf.classify("test")
        assert result_label == label

    def test_custom_score_returned(self):
        clf = MockQueryClassifier(score=0.42)
        _, score = clf.classify("test")
        assert score == 0.42

    def test_score_boundary_zero(self):
        clf = MockQueryClassifier(score=0.0)
        _, score = clf.classify("test")
        assert score == 0.0

    def test_score_boundary_one(self):
        clf = MockQueryClassifier(score=1.0)
        _, score = clf.classify("test")
        assert score == 1.0


class TestValidation:
    def test_invalid_label_raises(self):
        with pytest.raises(ValueError, match="Invalid label"):
            MockQueryClassifier(label="inventé")

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError, match="Score must be"):
            MockQueryClassifier(score=1.1)

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError, match="Score must be"):
            MockQueryClassifier(score=-0.1)


class TestFactoryMethods:
    def test_as_general(self):
        label, score = MockQueryClassifier.as_general().classify("q")
        assert label == LABEL_GENERAL
        assert score == 0.95

    def test_as_emergency(self):
        label, score = MockQueryClassifier.as_emergency().classify("q")
        assert label == LABEL_EMERGENCY
        assert score == 0.99

    def test_as_medical_advice(self):
        label, score = MockQueryClassifier.as_medical_advice().classify("q")
        assert label == LABEL_MEDICAL_ADVICE
        assert score == 0.92

    def test_as_out_of_scope(self):
        label, score = MockQueryClassifier.as_out_of_scope().classify("q")
        assert label == LABEL_OUT_OF_SCOPE
        assert score == 0.88

    def test_as_low_confidence(self):
        _, score = MockQueryClassifier.as_low_confidence().classify("q")
        assert score < 0.70  # below default CAMEMBERT_CONFIDENCE_THRESHOLD
