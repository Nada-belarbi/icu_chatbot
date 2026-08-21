"""
scripts/validate_environment.py

Validate that the local environment is correctly configured before running the
app, the indexing pipeline, or the tests. Never raises a raw traceback: every
check prints a clear PASS/FAIL line.

Usage:
    python scripts/validate_environment.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    suffix = f" — {detail}" if detail and not ok else ""
    print(f"{label:.<35}{status}{suffix}")
    return ok


def main() -> int:
    print("=== Environment validation ===")
    all_ok = True

    env_path = PROJECT_ROOT / ".env"
    all_ok &= _check(".env file present", env_path.exists(),
                      "copy .env.example to .env and fill in OPENAI_API_KEY")

    try:
        from config import settings  # noqa: F401  (import triggers required-var validation)
        all_ok &= _check("Required environment variables", True)
    except EnvironmentError as exc:
        all_ok &= _check("Required environment variables", False, str(exc))
        # Remaining checks need `settings` to be importable; stop here.
        print("\nNOT READY — fix the environment variable(s) above first.")
        return 1

    all_ok &= _check("CLASSIFIER_PATH exists", settings.CLASSIFIER_PATH.exists(),
                      f"expected {settings.CLASSIFIER_PATH}")
    all_ok &= _check("RAG_DOCUMENTS_PATH exists", settings.RAG_DOCUMENTS_PATH.exists(),
                      f"expected {settings.RAG_DOCUMENTS_PATH}")
    all_ok &= _check("CHROMA_DB_PATH exists", settings.CHROMA_DB_PATH.exists(),
                      f"expected {settings.CHROMA_DB_PATH} — run scripts/index_rag_documents.py")

    manifest_path = PROJECT_ROOT / "data" / "rag_documents" / "sources.csv"
    all_ok &= _check("Corpus manifest present", manifest_path.exists(), str(manifest_path))

    print()
    print("READY" if all_ok else "NOT READY")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
