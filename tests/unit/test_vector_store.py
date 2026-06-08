# Unit tests — VectorStoreManager (Phase 10)
# Uses a FakeEmbeddings to avoid downloading the real model.

import pytest
from pathlib import Path
from langchain_core.documents import Document
from src.rag.vector_store import VectorStoreManager


# ---------------------------------------------------------------------------
# Fake embeddings — deterministic, no model download
# ---------------------------------------------------------------------------
class FakeEmbeddings:
    """Minimal duck-type implementation of LangChain Embeddings interface."""

    _DIM = 16

    def embed_documents(self, texts):
        import hashlib
        vectors = []
        for text in texts:
            digest = hashlib.md5(text.encode()).hexdigest()
            vec = [int(digest[i : i + 2], 16) / 255.0 for i in range(0, self._DIM * 2, 2)]
            vectors.append(vec)
        return vectors

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.fixture
def vsm(tmp_path):
    return VectorStoreManager(
        chroma_path=tmp_path / "chroma",
        embeddings=FakeEmbeddings(),
        collection_name="test_collection",
    )


@pytest.fixture
def sample_docs():
    return [
        Document(page_content="Le respirateur aide à respirer.", metadata={"source": "doc_a.txt"}),
        Document(page_content="La sédation réduit la douleur.", metadata={"source": "doc_b.txt"}),
        Document(page_content="Le monitoring cardiaque est essentiel.", metadata={"source": "doc_c.txt"}),
    ]


class TestIndexDocuments:
    def test_index_returns_chroma_store(self, vsm, sample_docs):
        store = vsm.index_documents(sample_docs)
        assert store is not None

    def test_empty_docs_raises(self, vsm):
        with pytest.raises(ValueError, match="empty"):
            vsm.index_documents([])

    def test_chroma_dir_created(self, tmp_path, sample_docs):
        chroma_path = tmp_path / "new_chroma"
        vsm = VectorStoreManager(
            chroma_path=chroma_path,
            embeddings=FakeEmbeddings(),
            collection_name="test",
        )
        vsm.index_documents(sample_docs)
        assert chroma_path.exists()


class TestGetStore:
    def test_get_store_after_index(self, vsm, sample_docs):
        vsm.index_documents(sample_docs)
        store = vsm.get_store()
        assert store is not None

    def test_get_store_missing_raises(self, tmp_path):
        vsm = VectorStoreManager(
            chroma_path=tmp_path / "missing",
            embeddings=FakeEmbeddings(),
            collection_name="test",
        )
        with pytest.raises(FileNotFoundError):
            vsm.get_store()

    def test_get_store_returns_searchable_store(self, vsm, sample_docs):
        vsm.index_documents(sample_docs)
        store = vsm.get_store()
        results = store.similarity_search("respirateur", k=1)
        assert len(results) >= 1
