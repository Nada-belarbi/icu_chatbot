# DocumentLoader
#
# Responsibilities:
#   - Load documents from data/documents/ (PDF, TXT, DOCX)
#   - Split into chunks (size and overlap configurable in settings.py)
#   - Return list of LangChain Document objects with source metadata
#   - Must be run separately from the runtime (index once, reuse Chroma)

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import DOCUMENTS_PATH, CHUNK_SIZE, CHUNK_OVERLAP

_log = logging.getLogger(__name__)

# File extensions handled natively; PDF/DOCX require optional deps.
_SUPPORTED = {".txt", ".pdf", ".docx"}


class DocumentLoader:
    """
    Load and chunk documents from *documents_path*.

    Supported formats: TXT (always), PDF (requires ``pypdf``),
    DOCX (requires ``docx2txt``).  Unsupported or unreadable files are
    skipped with a warning — they never cause the whole load to fail.
    """

    def __init__(
        self,
        documents_path: Path | str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self._path = Path(documents_path) if documents_path else Path(DOCUMENTS_PATH)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size if chunk_size is not None else CHUNK_SIZE,
            chunk_overlap=chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> list[Document]:
        """Return all chunks from all documents in *documents_path*."""
        if not self._path.exists():
            _log.warning("Documents path does not exist: %s", self._path)
            return []

        docs: list[Document] = []
        for file in sorted(self._path.iterdir()):
            if file.suffix.lower() not in _SUPPORTED:
                _log.debug("Skipping unsupported file: %s", file.name)
                continue
            try:
                chunks = self._load_file(file)
                docs.extend(chunks)
                _log.info("Loaded %d chunks from %s", len(chunks), file.name)
            except Exception as exc:
                _log.error("Failed to load %s: %s", file.name, exc)
        return docs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_file(self, path: Path) -> list[Document]:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return self._load_txt(path)
        if suffix == ".pdf":
            return self._load_pdf(path)
        if suffix == ".docx":
            return self._load_docx(path)
        return []

    def _load_txt(self, path: Path) -> list[Document]:
        from langchain_community.document_loaders import TextLoader

        raw = TextLoader(str(path), encoding="utf-8").load()
        self._attach_source(raw, path.name)
        return self._splitter.split_documents(raw)

    def _load_pdf(self, path: Path) -> list[Document]:
        try:
            from langchain_community.document_loaders import PyPDFLoader
        except ImportError:
            _log.warning("pypdf not installed — skipping %s", path.name)
            return []
        raw = PyPDFLoader(str(path)).load()
        self._attach_source(raw, path.name)
        return self._splitter.split_documents(raw)

    def _load_docx(self, path: Path) -> list[Document]:
        try:
            from langchain_community.document_loaders import Docx2txtLoader
        except ImportError:
            _log.warning("docx2txt not installed — skipping %s", path.name)
            return []
        raw = Docx2txtLoader(str(path)).load()
        self._attach_source(raw, path.name)
        return self._splitter.split_documents(raw)

    @staticmethod
    def _attach_source(docs: list[Document], source: str) -> None:
        for doc in docs:
            doc.metadata["source"] = source
