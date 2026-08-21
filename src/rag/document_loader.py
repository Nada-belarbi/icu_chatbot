# DocumentLoader
#
# Responsibilities:
#   - Load documents from data/rag_documents/raw/ (HTML, PDF, TXT, DOCX)
#   - Recurse into subdirectories (SRLF/, HAS/, ICUS/, NHS/, ...)
#   - Split into chunks (size and overlap configurable in settings.py)
#   - Return list of LangChain Document objects with source metadata (org/file)
#   - Must be run separately from the runtime (index once, reuse Chroma)

from __future__ import annotations

import logging
import warnings
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import RAG_DOCUMENTS_PATH, CHUNK_SIZE, CHUNK_OVERLAP

_log = logging.getLogger(__name__)

# File extensions supported; HTML/PDF/DOCX require optional deps.
_SUPPORTED = {".txt", ".pdf", ".docx", ".html"}


class DocumentLoader:
    """
    Load and chunk documents from *documents_path* recursively.

    Supported formats: HTML (requires ``beautifulsoup4``), TXT (always),
    PDF (requires ``pypdf``), DOCX (requires ``docx2txt``).
    Unsupported or unreadable files are skipped with a warning — they
    never cause the whole load to fail.

    Source metadata is stored as ``"ORG/filename.ext"`` (path relative
    to *documents_path*) so the UI can display the originating collection.
    """

    def __init__(
        self,
        documents_path: Path | str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self._path = Path(documents_path) if documents_path else Path(RAG_DOCUMENTS_PATH)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size if chunk_size is not None else CHUNK_SIZE,
            chunk_overlap=chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> list[Document]:
        """Return all chunks from all documents in *documents_path* (recursive)."""
        if not self._path.exists():
            _log.warning("Documents path does not exist: %s", self._path)
            return []

        docs: list[Document] = []
        for file in sorted(self._path.rglob("*")):
            if not file.is_file():
                continue
            if file.suffix.lower() not in _SUPPORTED:
                _log.debug("Skipping unsupported file: %s", file.name)
                continue
            try:
                chunks = self._load_file(file)
                docs.extend(chunks)
                rel = file.relative_to(self._path)
                _log.info("Loaded %d chunks from %s", len(chunks), rel)
            except Exception as exc:
                _log.error("Failed to load %s: %s", file.name, exc)
        return docs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_file(self, path: Path) -> list[Document]:
        # Source = relative path from root, e.g. "SRLF/SRLF-001.html"
        try:
            source = str(path.relative_to(self._path)).replace("\\", "/")
        except ValueError:
            source = path.name

        suffix = path.suffix.lower()
        if suffix == ".html":
            raw = self._load_html_raw(path)
        elif suffix == ".txt":
            raw = self._load_txt_raw(path)
        elif suffix == ".pdf":
            raw = self._load_pdf_raw(path)
        elif suffix == ".docx":
            raw = self._load_docx_raw(path)
        else:
            return []

        if not raw:
            return []
        self._attach_source(raw, source)
        return self._splitter.split_documents(raw)

    def _load_html_raw(self, path: Path) -> list[Document]:
        try:
            from langchain_community.document_loaders import BSHTMLLoader
        except ImportError:
            _log.warning("beautifulsoup4 not installed — skipping %s", path.name)
            return []
        # Try UTF-8 first, then common fallback encodings
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    loader = BSHTMLLoader(
                        str(path),
                        open_encoding=encoding,
                        get_text_separator="\n",
                    )
                    return loader.load()
            except (UnicodeDecodeError, LookupError):
                continue
        _log.warning("Could not decode %s with any known encoding — skipping", path.name)
        return []

    def _load_txt_raw(self, path: Path) -> list[Document]:
        from langchain_community.document_loaders import TextLoader
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return TextLoader(str(path), encoding="utf-8").load()

    def _load_pdf_raw(self, path: Path) -> list[Document]:
        try:
            from langchain_community.document_loaders import PyPDFLoader
        except ImportError:
            _log.warning("pypdf not installed — skipping %s", path.name)
            return []
        return PyPDFLoader(str(path)).load()

    def _load_docx_raw(self, path: Path) -> list[Document]:
        try:
            from langchain_community.document_loaders import Docx2txtLoader
        except ImportError:
            _log.warning("docx2txt not installed — skipping %s", path.name)
            return []
        return Docx2txtLoader(str(path)).load()

    @staticmethod
    def _attach_source(docs: list[Document], source: str) -> None:
        for doc in docs:
            doc.metadata["source"] = source
