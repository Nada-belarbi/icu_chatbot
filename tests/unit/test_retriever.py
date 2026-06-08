# Unit tests — Retriever (Phase 11)
# Uses FakeEmbeddings + real Chroma in tmp_path.

import pytest
from pathlib import Path
from langchain_core.documents import Document
from src.rag.vector_store import VectorStoreManager
from src.rag.retriever import Retriever


class FakeEmbeddings:
    _DIM = 16

    def embed_documents(self, texts):
        import hashlib
        return [
            [int(hashlib.md5(t.encode()).hexdigest()[i : i + 2], 16) / 255.0 for i in range(0, self._DIM * 2, 2)]
            for t in texts
        ]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.fixture
def indexed_vsm(tmp_path):
    """VectorStoreManager with 3 pre-indexed documents."""
    vsm = VectorStoreManager(
        chroma_path=tmp_path / "chroma",
        embeddings=FakeEmbeddings(),
        collection_name="test_retriever",
    )
    docs = [
        Document(page_content="Le respirateur mécanique aide à la ventilation.", metadata={"source": "rea.txt"}),
        Document(page_content="La sédation est utilisée pour réduire la douleur.", metadata={"source": "sedation.txt"}),
        Document(page_content="Le monitoring permet de surveiller les constantes vitales.", metadata={"source": "monitoring.txt"}),
    ]
    vsm.index_documents(docs)
    return vsm


class TestRetrieverSearch:
    def test_returns_list(self, indexed_vsm):
        retriever = Retriever(vector_store_manager=indexed_vsm, top_k=2)
        results = retriever.search("respirateur")
        assert isinstance(results, list)

    def test_returns_up_to_top_k(self, indexed_vsm):
        retriever = Retriever(vector_store_manager=indexed_vsm, top_k=2)
        results = retriever.search("médical")
        assert len(results) <= 2

    def test_result_has_required_keys(self, indexed_vsm):
        retriever = Retriever(vector_store_manager=indexed_vsm, top_k=3)
        results = retriever.search("respirateur")
        for r in results:
            assert "text" in r
            assert "source" in r
            assert "score" in r

    def test_result_text_is_non_empty(self, indexed_vsm):
        retriever = Retriever(vector_store_manager=indexed_vsm, top_k=3)
        results = retriever.search("sédation")
        assert all(len(r["text"].strip()) > 0 for r in results)

    def test_result_source_is_string(self, indexed_vsm):
        retriever = Retriever(vector_store_manager=indexed_vsm, top_k=3)
        results = retriever.search("monitoring")
        assert all(isinstance(r["source"], str) for r in results)

    def test_result_score_is_float(self, indexed_vsm):
        retriever = Retriever(vector_store_manager=indexed_vsm, top_k=3)
        results = retriever.search("monitoring")
        assert all(isinstance(r["score"], float) for r in results)

    def test_no_vsm_returns_empty(self):
        retriever = Retriever(vector_store_manager=None)
        assert retriever.search("question") == []

    def test_threshold_filters_results(self, indexed_vsm):
        # threshold=0.0 means only exact matches (distance <= 0), which should return nothing
        retriever = Retriever(vector_store_manager=indexed_vsm, top_k=5, similarity_threshold=0.0)
        results = retriever.search("respirateur")
        # No result should have score <= 0 (unless exact match)
        assert all(r["score"] <= 0.0 for r in results)
