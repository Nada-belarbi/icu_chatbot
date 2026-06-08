# Unit tests — RAGChain (Phase 12)
# Uses a RunnableLambda mock to avoid real OpenAI calls.
# (MagicMock is callable but not a Runnable — LCEL wraps it as RunnableLambda
#  which calls mock(input), not mock.invoke(input), causing StrOutputParser to fail.)

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from src.rag.rag_chain import RAGChain


def _make_llm(content: str = "Voici une information générale sur le respirateur."):
    """Return a minimal LCEL-compatible mock that returns a fixed AIMessage."""
    return RunnableLambda(lambda _: AIMessage(content=content))


def _make_raising_llm(exc: Exception):
    def _raise(_):
        raise exc
    return RunnableLambda(_raise)


@pytest.fixture
def chain():
    return RAGChain(llm=_make_llm())


SAMPLE_DOCS = [
    {"text": "Le respirateur aide à ventiler.", "source": "sfar.pdf", "score": 0.9},
    {"text": "La sédation réduit l'anxiété.", "source": "sfar.pdf", "score": 0.8},
]


class TestGenerate:
    def test_returns_tuple(self, chain):
        answer, sources = chain.generate("qu'est-ce qu'un respirateur ?", SAMPLE_DOCS)
        assert isinstance(answer, str)
        assert isinstance(sources, list)

    def test_answer_is_non_empty(self, chain):
        answer, _ = chain.generate("qu'est-ce qu'un respirateur ?", SAMPLE_DOCS)
        assert len(answer.strip()) > 0

    def test_sources_contains_file_names(self, chain):
        _, sources = chain.generate("qu'est-ce qu'un respirateur ?", SAMPLE_DOCS)
        assert "sfar.pdf" in sources

    def test_sources_deduplicated(self, chain):
        """Multiple docs with same source → source appears only once."""
        _, sources = chain.generate("question", SAMPLE_DOCS)
        assert sources.count("sfar.pdf") == 1

    def test_multiple_sources(self, chain):
        docs = [
            {"text": "Contenu A.", "source": "doc_a.pdf", "score": 0.9},
            {"text": "Contenu B.", "source": "doc_b.pdf", "score": 0.8},
        ]
        _, sources = chain.generate("question", docs)
        assert set(sources) == {"doc_a.pdf", "doc_b.pdf"}

    def test_empty_docs_returns_empty_without_calling_llm(self):
        """Empty docs → early return before LLM is ever called."""
        called = []
        llm = RunnableLambda(lambda _: called.append(True) or AIMessage(content="x"))
        c = RAGChain(llm=llm)
        answer, sources = c.generate("question", [])
        assert answer == ""
        assert sources == []
        assert called == [], "LLM must NOT be called for empty docs"

    def test_llm_exception_propagates(self):
        chain_err = RAGChain(llm=_make_raising_llm(RuntimeError("API error")))
        with pytest.raises(RuntimeError):
            chain_err.generate("question", SAMPLE_DOCS)
