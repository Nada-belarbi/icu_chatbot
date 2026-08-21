# GuardrailsManager  (LangChain-based)
#
# Three levels:
#   1. PRE-generation  — block LLM call if decision != allow_rag or no sources found
#   2. DURING generation — inject strict system prompt:
#       - answer only from retrieved documents
#       - no individualized medical advice
#       - no diagnosis / treatment / prognosis
#       - redirect to medical team when uncertain
#   3. POST-generation — delegate to PostGenerationChecker before returning

from __future__ import annotations

import logging

from src.safety.decision_engine import DECISION_ALLOW_RAG
from src.safety.controlled_responses import ControlledResponseManager

_log = logging.getLogger(__name__)


class GuardrailsManager:
    """
    Three-level guardrail wrapping the full RAG generation path.

    Parameters
    ----------
    rag_chain:
        ``RAGChain`` instance that calls the LLM.
    post_checker:
        ``PostGenerationChecker`` instance for post-generation scanning.
    """

    def __init__(self, rag_chain, post_checker) -> None:
        self._rag_chain = rag_chain
        self._post_checker = post_checker
        self._responses = ControlledResponseManager()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self, query: str, decision: str, docs: list[dict], history: list[dict] | None = None
    ) -> tuple[str, list[str], bool]:
        """
        Execute the full guarded RAG path.

        Parameters
        ----------
        query:
            Cleaned user question.
        decision:
            Routing decision from ``DecisionEngine``.
        docs:
            Retrieved document chunks from ``Retriever``.

        Returns
        -------
        (answer, sources, was_modified)
            *was_modified* is ``True`` if ``PostGenerationChecker`` replaced
            the LLM answer.
        """
        # Level 1 — pre-generation gate
        if decision != DECISION_ALLOW_RAG:
            _log.warning(
                "GuardrailsManager.run() called with decision=%r — blocked", decision
            )
            return self._responses.get_refuse_medical_advice(), [], False

        if not docs:
            return self._responses.get_no_source_found(), [], False

        # Level 2 — RAGChain (system prompt injected inside RAGChain)
        try:
            answer, sources = self._rag_chain.generate(query, docs, history=history)
        except Exception as exc:
            _log.error("RAGChain.generate() failed in GuardrailsManager: %s", exc)
            return self._responses.get_api_unavailable(), [], False

        # Level 3 — post-generation scan
        answer, was_modified = self._post_checker.check(answer)
        return answer, sources, was_modified

