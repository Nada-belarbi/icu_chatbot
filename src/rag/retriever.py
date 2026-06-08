# Retriever
#
# Responsibilities:
#   - Query the Chroma store and return the top-k most relevant document chunks
#   - top_k read from config/settings.py
#   - Return list of (document_chunk: str, source: str, score: float)
#   - If no relevant document found above a similarity threshold → signal absence

from __future__ import annotations

import logging

from config.settings import CHROMA_TOP_K

_log = logging.getLogger(__name__)


class Retriever:
    """
    Query a Chroma vector store and return the top-k most relevant chunks.

    Parameters
    ----------
    vector_store_manager:
        ``VectorStoreManager`` instance that owns the Chroma collection.
    top_k:
        Number of results to return.  Defaults to ``CHROMA_TOP_K``.
    similarity_threshold:
        Minimum relevance score to include a result (0 = no filtering).
        Chroma returns L2-distance-based scores; lower = more similar, so
        this acts as an *upper* bound on the distance.  Set to ``float("inf")``
        to disable filtering (default).
    """

    def __init__(
        self,
        vector_store_manager=None,
        top_k: int | None = None,
        similarity_threshold: float = float("inf"),
    ) -> None:
        self._vsm = vector_store_manager
        self._top_k = top_k if top_k is not None else CHROMA_TOP_K
        self._threshold = similarity_threshold
        self._store = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str) -> list[dict]:
        """
        Return the top-k most relevant chunks for *query*.

        Each result is a ``dict`` with keys ``text``, ``source``, ``score``.
        Returns an empty list if the store is empty or all scores exceed the
        similarity threshold.  Never raises.
        """
        if self._vsm is None:
            _log.warning("Retriever has no VectorStoreManager — returning empty list")
            return []
        try:
            store = self._get_store()
            raw = store.similarity_search_with_score(query, k=self._top_k)
            results = []
            for doc, score in raw:
                if score <= self._threshold:
                    results.append(
                        {
                            "text": doc.page_content,
                            "source": doc.metadata.get("source", "unknown"),
                            "score": float(score),
                        }
                    )
            return results
        except Exception as exc:
            _log.error("Retriever.search() failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_store(self):
        if self._store is None:
            self._store = self._vsm.get_store()
        return self._store
