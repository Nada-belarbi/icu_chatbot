# Integration tests — ChatbotPipeline skeleton (Phase 9)
# RAG path (Phases 11-13) not yet wired — tested with None stubs.
# All tests use MockQueryClassifier — no model, no API calls.

import pytest
from src.pipeline.chatbot_pipeline import (
    ChatbotPipeline,
    PipelineResult,
    RTYPE_REDIRECT,
    RTYPE_REFUSE,
    RTYPE_REFORMULATION,
    RTYPE_ERROR,
)
from src.safety.query_classifier import MockQueryClassifier
from src.memory.conversation_memory import ConversationMemoryManager
from src.safety.decision_engine import (
    DECISION_REDIRECT, DECISION_REFUSE, DECISION_REFUSE_OOS, DECISION_ASK_REFORMULATION, DECISION_ALLOW_RAG,
)
from config.safety_config import (
    LABEL_GENERAL, LABEL_MEDICAL_ADVICE, LABEL_EMERGENCY, LABEL_OUT_OF_SCOPE,
)


def make_pipeline(label=LABEL_GENERAL, score=0.95):
    return ChatbotPipeline(classifier=MockQueryClassifier(label=label, score=score))


class TestPipelineReturnsResult:
    def test_returns_pipeline_result(self):
        result = make_pipeline().run("qu'est-ce qu'un respirateur ?")
        assert isinstance(result, PipelineResult)

    def test_result_has_answer(self):
        result = make_pipeline().run("qu'est-ce qu'un respirateur ?")
        assert isinstance(result.answer, str)
        assert len(result.answer.strip()) > 0

    def test_result_has_sources_list(self):
        result = make_pipeline().run("qu'est-ce qu'un respirateur ?")
        assert isinstance(result.sources, list)


class TestRoutingDecisions:
    def test_urgency_keyword_returns_redirect(self):
        result = make_pipeline().run("il ne respire plus")
        assert result.decision == DECISION_REDIRECT
        assert result.response_type == RTYPE_REDIRECT

    def test_blocked_keyword_returns_refuse(self):
        result = make_pipeline().run("quel médicament donner ?")
        assert result.decision == DECISION_REFUSE
        assert result.response_type == RTYPE_REFUSE

    def test_medical_advice_label_returns_refuse(self):
        result = make_pipeline(label=LABEL_MEDICAL_ADVICE, score=0.92).run(
            "c'est grave ce qu'il a ?"
        )
        assert result.decision == DECISION_REFUSE

    def test_out_of_scope_label_returns_refuse(self):
        result = make_pipeline(label=LABEL_OUT_OF_SCOPE, score=0.88).run(
            "quel est le menu du restaurant ?"
        )
        assert result.decision == DECISION_REFUSE_OOS

    def test_emergency_label_returns_redirect(self):
        result = make_pipeline(label=LABEL_EMERGENCY, score=0.97).run(
            "il perd connaissance"
        )
        assert result.decision == DECISION_REDIRECT

    def test_low_confidence_returns_reformulation(self):
        result = make_pipeline(label=LABEL_GENERAL, score=0.40).run("question floue")
        assert result.decision == DECISION_ASK_REFORMULATION
        assert result.response_type == RTYPE_REFORMULATION

    def test_allow_rag_without_retriever_returns_no_source(self):
        result = make_pipeline(label=LABEL_GENERAL, score=0.95).run(
            "qu'est-ce qu'un respirateur ?"
        )
        assert result.decision == DECISION_ALLOW_RAG


class TestEmptyInput:
    def test_empty_string_returns_reformulation(self):
        result = make_pipeline().run("")
        assert result.decision == DECISION_ASK_REFORMULATION

    def test_whitespace_only_returns_reformulation(self):
        result = make_pipeline().run("   ")
        assert result.decision == DECISION_ASK_REFORMULATION


class TestMemoryIntegration:
    def test_memory_updated_after_run(self):
        mem = ConversationMemoryManager(history=[], max_turns=3)
        make_pipeline().run("qu'est-ce qu'un respirateur ?", memory=mem)
        history = mem.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_empty_input_does_not_update_memory(self):
        mem = ConversationMemoryManager(history=[], max_turns=3)
        make_pipeline().run("", memory=mem)
        assert mem.get_history() == []

    def test_none_memory_does_not_crash(self):
        result = make_pipeline().run("question", memory=None)
        assert isinstance(result, PipelineResult)


class TestExceptionSafety:
    def test_unexpected_exception_returns_generic_error(self):
        class BrokenClassifier(MockQueryClassifier):
            def classify(self, text):
                raise RuntimeError("simulated crash")

        pipeline = ChatbotPipeline(classifier=BrokenClassifier())
        # Neutral question: no rules, no safe-allow bypass, not a greeting → reaches classifier
        result = pipeline.run("les soins intensifs en reanimation")
        assert result.response_type == RTYPE_ERROR
        assert len(result.answer) > 0
#   - Sensitive question → ask_reformulation
#   - Forbidden question → refuse (no LLM call)
#   - Urgency → redirect (no LLM call)
#   - Out-of-scope → refuse (no LLM call)
#   - Ambiguous question + conversation memory context
#   - Question with no matching document → controlled response
