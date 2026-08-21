from pathlib import Path

import pytest

from src.safety.query_classifier import CamembertQueryClassifier


class _DummyTensor:
    def __init__(self, value):
        self._value = value

    def to(self, _device):
        return self

    def item(self):
        return self._value


class _DummyTokenizer:
    def __call__(self, texts, truncation, padding, max_length, return_tensors):
        assert isinstance(texts, list)
        assert truncation is True
        assert padding is True
        assert max_length > 0
        assert return_tensors == "pt"
        return {"input_ids": _DummyTensor(1), "attention_mask": _DummyTensor(1)}


class _DummyModel:
    def __init__(self, id2label):
        class _Cfg:
            pass

        self.config = _Cfg()
        self.config.id2label = id2label
        self._eval_called = False

    def eval(self):
        self._eval_called = True

    def to(self, _device):
        return self

    def __call__(self, **_encoded):
        class _Out:
            pass

        out = _Out()
        out.logits = "dummy_logits"
        return out


def test_camembert_classifier_requires_model_dir(tmp_path: Path):
    missing = tmp_path / "missing_model"
    with pytest.raises(FileNotFoundError):
        CamembertQueryClassifier(model_dir=missing)


def test_camembert_classifier_returns_model_label(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True)

    dummy_model = _DummyModel({0: "information_generale"})

    monkeypatch.setattr(
        "src.safety.query_classifier.AutoTokenizer.from_pretrained",
        lambda _path: _DummyTokenizer(),
    )
    monkeypatch.setattr(
        "src.safety.query_classifier.AutoModelForSequenceClassification.from_pretrained",
        lambda _path: dummy_model,
    )
    monkeypatch.setattr("src.safety.query_classifier.torch.cuda.is_available", lambda: False)

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("src.safety.query_classifier.torch.no_grad", lambda: _NoGrad())
    monkeypatch.setattr("src.safety.query_classifier.torch.softmax", lambda logits, dim: "dummy_probs")

    def _fake_max(_probs, dim):
        return _DummyTensor(0.93), _DummyTensor(0)

    monkeypatch.setattr("src.safety.query_classifier.torch.max", _fake_max)

    clf = CamembertQueryClassifier(model_dir=model_dir)
    label, score = clf.classify("Question test")

    assert label == "information_generale"
    assert score == pytest.approx(0.93)
    assert dummy_model._eval_called is True


def test_camembert_classifier_fallback_to_general_on_unknown_label(monkeypatch, tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True)

    dummy_model = _DummyModel({0: "label_inconnu"})

    monkeypatch.setattr(
        "src.safety.query_classifier.AutoTokenizer.from_pretrained",
        lambda _path: _DummyTokenizer(),
    )
    monkeypatch.setattr(
        "src.safety.query_classifier.AutoModelForSequenceClassification.from_pretrained",
        lambda _path: dummy_model,
    )
    monkeypatch.setattr("src.safety.query_classifier.torch.cuda.is_available", lambda: False)

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("src.safety.query_classifier.torch.no_grad", lambda: _NoGrad())
    monkeypatch.setattr("src.safety.query_classifier.torch.softmax", lambda logits, dim: "dummy_probs")
    monkeypatch.setattr(
        "src.safety.query_classifier.torch.max",
        lambda _probs, dim: (_DummyTensor(0.51), _DummyTensor(0)),
    )

    clf = CamembertQueryClassifier(model_dir=model_dir)
    label, score = clf.classify("Question test")

    assert label == "information_generale"
    assert score == pytest.approx(0.51)
