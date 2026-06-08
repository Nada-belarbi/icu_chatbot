# Unit tests — DocumentLoader (Phase 10)

import pytest
from pathlib import Path
from src.rag.document_loader import DocumentLoader


@pytest.fixture
def txt_dir(tmp_path):
    """tmp_path with a single TXT document."""
    (tmp_path / "sfar_2024.txt").write_text(
        "Le respirateur mécanique est un appareil médical essentiel. " * 30,
        encoding="utf-8",
    )
    return tmp_path


class TestLoadEmptyOrMissing:
    def test_missing_directory_returns_empty(self, tmp_path):
        loader = DocumentLoader(documents_path=tmp_path / "nonexistent")
        assert loader.load() == []

    def test_empty_directory_returns_empty(self, tmp_path):
        loader = DocumentLoader(documents_path=tmp_path)
        assert loader.load() == []

    def test_unsupported_extension_ignored(self, tmp_path):
        (tmp_path / "data.csv").write_text("a,b,c")
        (tmp_path / "notes.xlsx").write_bytes(b"fake")
        loader = DocumentLoader(documents_path=tmp_path)
        assert loader.load() == []


class TestLoadTxt:
    def test_returns_documents(self, txt_dir):
        loader = DocumentLoader(documents_path=txt_dir, chunk_size=100, chunk_overlap=10)
        docs = loader.load()
        assert len(docs) > 0

    def test_source_metadata_attached(self, txt_dir):
        loader = DocumentLoader(documents_path=txt_dir, chunk_size=100, chunk_overlap=10)
        docs = loader.load()
        assert all("source" in d.metadata for d in docs)
        assert all(d.metadata["source"] == "sfar_2024.txt" for d in docs)

    def test_chunks_have_page_content(self, txt_dir):
        loader = DocumentLoader(documents_path=txt_dir, chunk_size=100, chunk_overlap=10)
        docs = loader.load()
        assert all(len(d.page_content.strip()) > 0 for d in docs)

    def test_chunk_size_respected(self, txt_dir):
        loader = DocumentLoader(documents_path=txt_dir, chunk_size=100, chunk_overlap=0)
        docs = loader.load()
        # Chunks may slightly exceed chunk_size at word boundaries — allow 20 % margin
        assert all(len(d.page_content) <= 120 for d in docs)

    def test_multiple_files(self, tmp_path):
        for i in range(3):
            (tmp_path / f"doc_{i}.txt").write_text(
                f"Document numéro {i}. Contenu médical. " * 20, encoding="utf-8"
            )
        loader = DocumentLoader(documents_path=tmp_path, chunk_size=50, chunk_overlap=5)
        docs = loader.load()
        sources = {d.metadata["source"] for d in docs}
        assert len(sources) == 3

    def test_mixed_extensions_only_txt_loaded(self, tmp_path):
        (tmp_path / "valid.txt").write_text("Contenu valide. " * 20, encoding="utf-8")
        (tmp_path / "ignore.csv").write_text("a,b,c")
        loader = DocumentLoader(documents_path=tmp_path, chunk_size=50, chunk_overlap=5)
        docs = loader.load()
        assert all(d.metadata["source"] == "valid.txt" for d in docs)
