# ChatbotPipeline — central orchestrator
#
# Single entry point called by app.py for every user message.
# Calls each module in the correct order and handles all fallback cases.
#
# Call order:
#   1. TextCleaner.clean()
#   2. ConversationMemoryManager.get_context_string()   (last 2-3 turns only)
#   3. RuleFilter.filter()
#   4. QueryClassifier.classify()                       (mock or real CamemBERT)
#   5. DecisionEngine.decide()
#   6. Route by decision:
#        allow_rag → Retriever.search() → RAGChain.generate() → PostGenerationChecker.check()
#        refuse / redirect / ask_reformulation → ControlledResponseManager
#   7. AnonymizedLogger.log()

from __future__ import annotations

import logging

from src.preprocessing.text_cleaner import TextCleaner
from src.memory.conversation_memory import ConversationMemoryManager
from src.safety.rule_filter import RuleFilter
from src.safety.query_classifier import BaseQueryClassifier, MockQueryClassifier
from src.safety.decision_engine import (
    DecisionEngine,
    DECISION_ALLOW_RAG,
    DECISION_REFUSE,
    DECISION_REFUSE_OOS,
    DECISION_REDIRECT,
    DECISION_ASK_REFORMULATION,
)
from src.safety.controlled_responses import ControlledResponseManager
from src.safety.guardrails import GuardrailsManager
from config.settings import MEMORY_MAX_TURNS, CAMEMBERT_CONFIDENCE_THRESHOLD

_log = logging.getLogger(__name__)

# Response type tags used by AnonymizedLogger (Phase 15)
RTYPE_INFORMATIVE = "informative"
RTYPE_REFUSE = "refuse"
RTYPE_REDIRECT = "redirect"
RTYPE_REFORMULATION = "reformulation"
RTYPE_ERROR = "error"


# Greetings — handled before any safety gate
_GREETINGS = frozenset([
    "bonjour", "bonsoir", "salut", "hello", "hi", "coucou",
    "bonjour !", "bonsoir !", "salut !",
    "bonjour je", "bonjour j", "bonsoir je",
])


def _is_greeting(text: str) -> bool:
    """Return True if *text* is a greeting with no substantive question."""
    lower = text.lower().strip()
    return any(lower == g or lower.startswith(g + " ") for g in _GREETINGS)


class PipelineResult:
    """Value object returned by ChatbotPipeline.run()."""

    __slots__ = ("answer", "sources", "decision", "response_type")

    def __init__(
        self,
        answer: str,
        sources: list[str],
        decision: str,
        response_type: str,
    ) -> None:
        self.answer = answer
        self.sources = sources
        self.decision = decision
        self.response_type = response_type


class ChatbotPipeline:
    """
    Single entry point for every user message.

    ``app.py`` calls ``run()`` and displays ``result.answer``.
    No business logic belongs in ``app.py``.

    Parameters
    ----------
    classifier:
        Any ``BaseQueryClassifier`` implementation.
        Defaults to ``MockQueryClassifier`` (Phase 6-9).
        Replaced by ``CamembertQueryClassifier`` in Phase 18.
    retriever:
        ``Retriever`` instance (Phase 11).  ``None`` until then.
    rag_chain:
        ``RAGChain`` instance (Phase 12).  ``None`` until then.
    post_checker:
        ``PostGenerationChecker`` instance (Phase 13).  ``None`` until then.
    logger:
        ``AnonymizedLogger`` instance (Phase 15).  ``None`` until then.
    """

    def __init__(
        self,
        classifier: BaseQueryClassifier | None = None,
        retriever=None,
        rag_chain=None,
        post_checker=None,
        logger=None,
    ) -> None:
        self._cleaner = TextCleaner()
        self._responses = ControlledResponseManager()
        self._rule_filter = RuleFilter()
        self._classifier = classifier or MockQueryClassifier()
        self._engine = DecisionEngine(
            rule_filter=self._rule_filter,
            classifier=self._classifier,
            confidence_threshold=CAMEMBERT_CONFIDENCE_THRESHOLD,
        )
        # Phase 11-15 stubs — wired in later phases
        self._retriever = retriever
        self._rag_chain = rag_chain
        self._post_checker = post_checker
        self._logger = logger
        self._guardrails = (
            GuardrailsManager(rag_chain=self._rag_chain, post_checker=self._post_checker)
            if self._rag_chain is not None and self._post_checker is not None
            else None
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        user_input: str,
        memory: ConversationMemoryManager | None = None,
    ) -> PipelineResult:
        """
        Process one user message end-to-end.

        Never raises — all exceptions are caught and returned as a
        ``generic_error`` controlled response.

        Parameters
        ----------
        user_input:
            Raw text from the UI input box.
        memory:
            Session memory manager.  Pass ``None`` in unit tests.

        Returns
        -------
        PipelineResult
        """
        try:
            return self._run(user_input, memory)
        except Exception as exc:
            _log.exception("Unexpected error in ChatbotPipeline.run(): %s", exc)
            return PipelineResult(
                answer=self._responses.get_generic_error(),
                sources=[],
                decision="error",
                response_type=RTYPE_ERROR,
            )

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _run(
        self,
        user_input: str,
        memory: ConversationMemoryManager | None,
    ) -> PipelineResult:

        # --- Step 1: clean input ---
        try:
            clean_text = self._cleaner.clean(user_input)
        except ValueError:
            return PipelineResult(
                answer=self._responses.get_ask_reformulation(),
                sources=[],
                decision=DECISION_ASK_REFORMULATION,
                response_type=RTYPE_REFORMULATION,
            )

        # --- Step 1b: greeting shortcut ---
        if _is_greeting(clean_text):
            result = PipelineResult(
                answer=self._responses.get_greeting(),
                sources=[],
                decision="greeting",
                response_type=RTYPE_INFORMATIVE,
            )
            if memory is not None:
                memory.add_user_message(clean_text)
                memory.add_assistant_message(result.answer)
            self._log_interaction(clean_text, result)
            return result

        # --- Step 2: get conversation context ---
        history = memory.get_history() if memory else []
        context_string = memory.get_context_string() if memory else ""

        # --- Step 3 + 4 + 5: decide ---
        decision = self._engine.decide(clean_text, context_string=context_string)

        # --- Step 6: route ---
        result = self._route(clean_text, decision, history=history)

        # --- Step 7: update memory ---
        # Only store the assistant's message when it's a real RAG answer.
        # Controlled responses (urgency redirect, refuse, ask_reformulation)
        # contain boilerplate text (e.g. "SAMU", "urgence médicale") that
        # would poison the classifier context on the next turn.
        if memory is not None:
            memory.add_user_message(clean_text)
            if result.decision == DECISION_ALLOW_RAG:
                memory.add_assistant_message(result.answer)

        # --- Step 7b: log (Phase 15 — no-op until wired) ---
        self._log_interaction(clean_text, result)

        return result

    def _route(self, clean_text: str, decision: str, history: list[dict] | None = None) -> PipelineResult:
        """Map decision to the appropriate response path."""

        if decision == DECISION_REDIRECT:
            return PipelineResult(
                answer=self._responses.get_urgency_redirect(),
                sources=[],
                decision=decision,
                response_type=RTYPE_REDIRECT,
            )

        if decision == DECISION_REFUSE:
            return PipelineResult(
                answer=self._responses.get_refuse_medical_advice(),
                sources=[],
                decision=decision,
                response_type=RTYPE_REFUSE,
            )

        if decision == DECISION_REFUSE_OOS:
            return PipelineResult(
                answer=self._responses.get_out_of_scope(),
                sources=[],
                decision=decision,
                response_type=RTYPE_REFUSE,
            )

        if decision == DECISION_ASK_REFORMULATION:
            return PipelineResult(
                answer=self._responses.get_ask_reformulation(),
                sources=[],
                decision=decision,
                response_type=RTYPE_REFORMULATION,
            )

        if decision == DECISION_ALLOW_RAG:
            return self._rag_path(clean_text, history=history)

        # Safety-first default
        _log.warning("Unknown decision %r — defaulting to refuse", decision)
        return PipelineResult(
            answer=self._responses.get_refuse_medical_advice(),
            sources=[],
            decision=DECISION_REFUSE,
            response_type=RTYPE_REFUSE,
        )

    def _rag_path(self, clean_text: str, history: list[dict] | None = None) -> PipelineResult:
        """
        RAG path — Retriever → RAGChain → PostGenerationChecker.
        Until Phases 11-13 are wired, returns a placeholder.
        """
        # Phase 11 — Retriever (stub)
        if self._retriever is None:
            return PipelineResult(
                answer=self._responses.get_no_source_found(),
                sources=[],
                decision=DECISION_ALLOW_RAG,
                response_type=RTYPE_REFUSE,
            )

        # Enrich retrieval query with last user message to resolve follow-up references
        # (e.g. "Et pourquoi on en met une ?" needs "trachéotomie" context to retrieve correctly)
        retrieval_query = clean_text
        if history:
            last_user = next(
                (m["content"] for m in reversed(history) if m["role"] == "user"),
                None,
            )
            if last_user:
                retrieval_query = f"{last_user} {clean_text}"

        docs = self._retriever.search(retrieval_query)
        if not docs:
            return PipelineResult(
                answer=self._responses.get_no_source_found(),
                sources=[],
                decision=DECISION_ALLOW_RAG,
                response_type=RTYPE_REFUSE,
            )

        # Phase 12 — RAGChain (stub)
        if self._rag_chain is None:
            return PipelineResult(
                answer=self._responses.get_api_unavailable(),
                sources=[],
                decision=DECISION_ALLOW_RAG,
                response_type=RTYPE_ERROR,
            )

        if self._guardrails is not None:
            answer, sources, was_modified = self._guardrails.run(
                query=clean_text,
                decision=DECISION_ALLOW_RAG,
                docs=docs,
                history=history,
            )
            response_type = RTYPE_REFUSE if was_modified else RTYPE_INFORMATIVE
            return PipelineResult(
                answer=answer,
                sources=sources,
                decision=DECISION_ALLOW_RAG,
                response_type=response_type,
            )

        try:
            answer, sources = self._rag_chain.generate(clean_text, docs, history=history)
        except Exception as exc:
            _log.error("RAGChain.generate() failed: %s", exc)
            return PipelineResult(
                answer=self._responses.get_api_unavailable(),
                sources=[],
                decision=DECISION_ALLOW_RAG,
                response_type=RTYPE_ERROR,
            )

        if self._post_checker is not None:
            answer, was_modified = self._post_checker.check(answer)
            if was_modified:
                return PipelineResult(
                    answer=answer,
                    sources=[],
                    decision=DECISION_ALLOW_RAG,
                    response_type=RTYPE_REFUSE,
                )

        return PipelineResult(
            answer=answer,
            sources=sources,
            decision=DECISION_ALLOW_RAG,
            response_type=RTYPE_INFORMATIVE,
        )

    def _log_interaction(self, clean_text: str, result: PipelineResult) -> None:
        """Phase 15 — no-op until AnonymizedLogger is wired."""
        if self._logger is not None:
            self._logger.log(
                raw_text=clean_text,
                decision=result.decision,
                sources=result.sources,
                response_type=result.response_type,
            )

#   8. ConversationMemoryManager.add_*()
#
# Fallback cases (must never raise an exception to the UI):
#   - TextCleaner raises ValueError (empty input)  → return ask_reformulation response
#   - Retriever returns empty list (no docs found) → return no_source_found response
#   - OpenAI API call fails                        → return api_unavailable response
#   - PostGenerationChecker rejects the answer     → return refuse_medical_advice response
#   - Any unexpected exception                     → return generic_error response + log
#
# Interface:
#   pipeline = ChatbotPipeline()
#   response, sources, decision = pipeline.run(user_input, session_memory)
