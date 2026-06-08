# Unit tests — GuardrailsManager (Phase 14)

import pytest
from unittest.mock import MagicMock
from src.safety.guardrails import GuardrailsManager
from src.safety.post_checker import PostGenerationChecker
from src.safety.decision_engine import DECISION_ALLOW_RAG, DECISION_REFUSE


@pytest.fixture
def mock_rag_chain():
    chain = MagicMock()
    chain.generate.return_value = (
        "Le respirateur est un appareil médical qui aide à la ventilation.",
        ["sfar.pdf"],
    )
    return chain


@pytest.fixture
def post_checker():
    return PostGenerationChecker()


@pytest.fixture
def guardrails(mock_rag_chain, post_checker):
    return GuardrailsManager(rag_chain=mock_rag_chain, post_checker=post_checker)


SAMPLE_DOCS = [{"text": "Contenu médical.", "source": "sfar.pdf", "score": 0.9}]


class TestPreGenerationGate:
    def test_wrong_decision_blocks_llm(self, guardrails, mock_rag_chain):
        guardrails.run("question", DECISION_REFUSE, SAMPLE_DOCS)
        mock_rag_chain.generate.assert_not_called()

    def test_wrong_decision_returns_refusal(self, guardrails):
        answer, sources, _ = guardrails.run("question", DECISION_REFUSE, SAMPLE_DOCS)
        assert len(answer) > 0
        assert sources == []

    def test_empty_docs_blocks_llm(self, guardrails, mock_rag_chain):
        guardrails.run("question", DECISION_ALLOW_RAG, [])
        mock_rag_chain.generate.assert_not_called()

    def test_empty_docs_returns_no_source(self, guardrails):
        answer, sources, _ = guardrails.run("question", DECISION_ALLOW_RAG, [])
        assert len(answer) > 0
        assert sources == []


class TestNormalPath:
    def test_returns_answer_and_sources(self, guardrails):
        answer, sources, _ = guardrails.run("question", DECISION_ALLOW_RAG, SAMPLE_DOCS)
        assert len(answer) > 0
        assert "sfar.pdf" in sources

    def test_was_modified_false_for_safe_answer(self, guardrails):
        _, _, was_modified = guardrails.run("question", DECISION_ALLOW_RAG, SAMPLE_DOCS)
        assert was_modified is False

    def test_unsafe_llm_output_triggers_post_check(self, mock_rag_chain, post_checker):
        mock_rag_chain.generate.return_value = (
            "Il va survivre sans problème.",
            ["sfar.pdf"],
        )
        gm = GuardrailsManager(rag_chain=mock_rag_chain, post_checker=post_checker)
        answer, _, was_modified = gm.run("question", DECISION_ALLOW_RAG, SAMPLE_DOCS)
        assert was_modified is True
        assert len(answer) > 0

    def test_rag_chain_exception_returns_api_unavailable(self, mock_rag_chain, post_checker):
        mock_rag_chain.generate.side_effect = RuntimeError("network error")
        gm = GuardrailsManager(rag_chain=mock_rag_chain, post_checker=post_checker)
        answer, sources, _ = gm.run("question", DECISION_ALLOW_RAG, SAMPLE_DOCS)
        assert len(answer) > 0
        assert sources == []
