"""
scripts/run_smoke_tests.py

Fast, read-only checks that the core components actually load and respond.
Not a replacement for the pytest suite — meant to run in a few seconds/minutes
as a quick "is this environment usable at all" gate.

Usage:
    python scripts/run_smoke_tests.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Avoid multi-minute hangs when there is no internet access: the embedding
# model is already cached locally, so HuggingFace Hub connectivity checks
# (HEAD requests) are not needed to load it.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _check(label: str, fn) -> bool:
    start = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - start
        print(f"{label:.<40}PASS  ({elapsed:.2f}s)")
        return True
    except Exception as exc:  # smoke test: report, never crash the runner
        elapsed = time.perf_counter() - start
        print(f"{label:.<40}FAIL  ({elapsed:.2f}s) — {exc}")
        return False


def main() -> int:
    results = []

    def check_rule_filter():
        from src.safety.rule_filter import RuleFilter
        category, blocked = RuleFilter().filter("Qu'est-ce qu'un respirateur ?")
        assert blocked is False, f"expected an allowed educational question, got category={category}"

    def check_decision_engine_controlled_path():
        from src.safety.rule_filter import RuleFilter
        from src.safety.query_classifier import MockQueryClassifier
        from src.safety.decision_engine import DecisionEngine, DECISION_REDIRECT
        engine = DecisionEngine(rule_filter=RuleFilter(), classifier=MockQueryClassifier())
        decision = engine.decide("Il ne respire plus")
        assert decision == DECISION_REDIRECT, f"expected {DECISION_REDIRECT}, got {decision}"

    def check_controlled_response():
        from src.safety.controlled_responses import ControlledResponseManager
        message = ControlledResponseManager.get_urgency_redirect()
        assert isinstance(message, str) and len(message) > 0

    def check_classifier_loads():
        from src.safety.query_classifier import CamembertQueryClassifier
        clf = CamembertQueryClassifier()
        label, score = clf.classify("Qu'est-ce qu'un respirateur ?")
        assert isinstance(label, str) and 0.0 <= score <= 1.0

    def check_chroma_loads():
        from config.settings import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        collection = client.get_collection(CHROMA_COLLECTION_NAME)
        assert collection.count() > 0, "Chroma collection is empty"

    def check_retrieval_returns_metadata():
        from src.rag.vector_store import VectorStoreManager
        from src.rag.retriever import Retriever
        retriever = Retriever(vector_store_manager=VectorStoreManager(), top_k=3)
        results_ = retriever.search("ventilation mécanique")
        assert isinstance(results_, list)
        if results_:
            assert "source" in results_[0] and "score" in results_[0]

    print("=== Smoke tests ===")
    results.append(_check("RuleFilter allows educational question", check_rule_filter))
    results.append(_check("DecisionEngine redirects emergencies", check_decision_engine_controlled_path))
    results.append(_check("ControlledResponseManager returns text", check_controlled_response))
    results.append(_check("Chroma collection loads and is non-empty", check_chroma_loads))
    results.append(_check("Retriever returns metadata (source/score)", check_retrieval_returns_metadata))
    results.append(_check("CamemBERT classifier loads (slow)", check_classifier_loads))

    print()
    if all(results):
        print("PASS — all smoke tests passed.")
        return 0
    print(f"FAIL — {results.count(False)}/{len(results)} smoke test(s) failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
