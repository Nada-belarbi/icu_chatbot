# VectorStoreManager
#
# Responsibilities:
#   - Create or load a persistent Chroma collection from data/chroma_db/
#   - Embed documents using a sentence-transformers model (local, no API cost)
#   - Expose methods: index_documents(), get_store()

from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config.settings import CHROMA_DB_PATH, EMBEDDING_MODEL, CHROMA_COLLECTION_NAME

_log = logging.getLogger(__name__)


def _default_embeddings():
    """Return HuggingFaceEmbeddings with the configured model (lazy import)."""
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


class VectorStoreManager:
    """
    Create or load a persistent Chroma vector store.

    Parameters
    ----------
    chroma_path:
        Directory for the Chroma SQLite DB.  Defaults to ``CHROMA_DB_PATH``.
    embeddings:
        Any LangChain-compatible embeddings object.  Pass a fake/mock
        implementation in unit tests to avoid downloading the real model.
        Defaults to ``HuggingFaceEmbeddings(EMBEDDING_MODEL)``.
    collection_name:
        Chroma collection name.  Defaults to ``CHROMA_COLLECTION_NAME``.
    """

    def __init__(
        self,
        chroma_path: Path | str | None = None,
        embeddings=None,
        collection_name: str | None = None,
    ) -> None:
        self._chroma_path = Path(chroma_path) if chroma_path else Path(CHROMA_DB_PATH)
        self._collection_name = collection_name or CHROMA_COLLECTION_NAME
        # Defer real embedding model loading so tests can inject a fake
        self._embeddings = embeddings if embeddings is not None else _default_embeddings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_documents(self, docs: list[Document]) -> Chroma:
        """
        Build a Chroma collection from *docs* and persist it to disk.

        Raises ``ValueError`` if *docs* is empty.
        """
        if not docs:
            raise ValueError("Cannot index an empty document list.")
        self._chroma_path.mkdir(parents=True, exist_ok=True)
        store = Chroma.from_documents(
            documents=docs,
            embedding=self._embeddings,
            collection_name=self._collection_name,
            persist_directory=str(self._chroma_path),
        )
        _log.info(
            "Indexed %d chunks into Chroma collection '%s' at %s",
            len(docs),
            self._collection_name,
            self._chroma_path,
        )
        return store

    def get_store(self) -> Chroma:
        """
        Load an existing Chroma collection from disk.

        Raises ``FileNotFoundError`` if the collection has not been indexed yet.
        """
        if not self._chroma_path.exists():
            raise FileNotFoundError(
                f"Chroma DB not found at {self._chroma_path}. "
                "Run index_documents() or the indexing notebook first."
            )
        return Chroma(
            embedding_function=self._embeddings,
            collection_name=self._collection_name,
            persist_directory=str(self._chroma_path),
        )
