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


def _build_query_variants(query: str) -> list[str]:
    """Return semantically equivalent query variants for robust retrieval."""
    variants = [query]
    lower = query.lower()

    # Respiratory support acronyms are often under-indexed in multilingual corpora.
    # Expand explicitly to improve recall on CPAP/BiPAP/VNI questions.
    if any(token in lower for token in (
        "cpap",
        "bipap",
        "vni",
        "niv",
        "ventilation non invasive",
        "ventilation non-invasive",
        "non invasive ventilation",
        "non-invasive ventilation",
    )):
        variants.extend([
            f"{query} ventilation non invasive VNI NIV",
            f"{query} CPAP BiPAP pression positive continue pression positive a deux niveaux",
            f"{query} non-invasive ventilation CPAP BiPAP",
        ])

    # Deduplicate while preserving order.
    return list(dict.fromkeys(v.strip() for v in variants if v.strip()))


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
            candidates: list[dict] = []
            lower_query = query.lower()
            is_niv_query = any(token in lower_query for token in (
                "cpap", "bipap", "vni", "niv", "ventilation non invasive", "ventilation non-invasive"
            ))

            for variant in _build_query_variants(query):
                raw = store.similarity_search_with_score(variant, k=self._top_k)
                for doc, score in raw:
                    score_val = float(score)
                    if score_val > self._threshold:
                        continue

                    text = doc.page_content
                    if is_niv_query:
                        low_text = text.lower()
                        niv_markers = (
                            "cpap", "bipap", "vni", "niv", "non-invasive ventilation",
                            "non invasive ventilation", "pression positive", "ventilation"
                        )
                        if not any(marker in low_text for marker in niv_markers):
                            continue

                    candidates.append(
                        {
                            "text": text,
                            "source": doc.metadata.get("source", "unknown"),
                            "score": score_val,
                        }
                    )

            if not candidates:
                return []

            # Merge variants, keep best scores first, and diversify source coverage.
            candidates.sort(key=lambda x: x["score"])
            results: list[dict] = []
            seen_chunks: set[tuple[str, str]] = set()
            per_source_count: dict[str, int] = {}
            max_per_source = 2

            for item in candidates:
                key = (item["source"], item["text"])
                if key in seen_chunks:
                    continue
                if per_source_count.get(item["source"], 0) >= max_per_source:
                    continue

                seen_chunks.add(key)
                per_source_count[item["source"]] = per_source_count.get(item["source"], 0) + 1
                results.append(item)

                if len(results) >= self._top_k:
                    return results

            # If source diversification was too restrictive, fill remaining slots.
            for item in candidates:
                key = (item["source"], item["text"])
                if key in seen_chunks:
                    continue
                seen_chunks.add(key)
                results.append(item)
                if len(results) >= self._top_k:
                    break

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
