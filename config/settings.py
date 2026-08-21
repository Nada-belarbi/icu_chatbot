# Central configuration — paths, thresholds, model names, API settings
# Loaded at startup; all other modules import from here

import os
from pathlib import Path
from dotenv import load_dotenv

# Locate project root (parent of config/) and load .env
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _get_required(key: str) -> str:
    """Return env var value or raise if missing/empty."""
    value = os.getenv(key, "").strip()
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Copy .env.example to .env and fill in the value."
        )
    return value


def _get_float(key: str, default: float, min_val: float, max_val: float) -> float:
    raw = os.getenv(key, str(default))
    try:
        value = float(raw)
    except ValueError:
        raise EnvironmentError(f"Environment variable '{key}' must be a number, got: {raw!r}")
    if not (min_val <= value <= max_val):
        raise EnvironmentError(
            f"Environment variable '{key}' must be between {min_val} and {max_val}, got: {value}"
        )
    return value


def _get_int(key: str, default: int, min_val: int, max_val: int) -> int:
    raw = os.getenv(key, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise EnvironmentError(f"Environment variable '{key}' must be an integer, got: {raw!r}")
    if not (min_val <= value <= max_val):
        raise EnvironmentError(
            f"Environment variable '{key}' must be between {min_val} and {max_val}, got: {value}"
        )
    return value


# ---------------------------------------------------------------------------
# API / LLM
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = _get_required("OPENAI_API_KEY")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
CLASSIFIER_PATH: Path = _PROJECT_ROOT / os.getenv(
    "CLASSIFIER_PATH", "models/camembert_classifier"
)

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
CHROMA_DB_PATH: Path = _PROJECT_ROOT / os.getenv("CHROMA_PATH", "data/rag_documents/chroma_db")
DOCUMENTS_PATH: Path = _PROJECT_ROOT / os.getenv("DOCUMENTS_PATH", "data/documents")
RAG_DOCUMENTS_PATH: Path = _PROJECT_ROOT / os.getenv("RAG_DOCUMENTS_PATH", "data/rag_documents/raw")
DATASET_PATH: Path = _PROJECT_ROOT / os.getenv("DATASET_PATH", "data/questions_dataset")
LOG_DIR: Path = _PROJECT_ROOT / os.getenv("LOG_DIR", "logs")

# ---------------------------------------------------------------------------
# Tuning / thresholds
# ---------------------------------------------------------------------------
CAMEMBERT_CONFIDENCE_THRESHOLD: float = _get_float(
    "CAMEMBERT_CONFIDENCE_THRESHOLD", default=0.70, min_val=0.0, max_val=1.0
)
CHROMA_TOP_K: int = _get_int("CHROMA_TOP_K", default=8, min_val=1, max_val=20)
MEMORY_MAX_TURNS: int = _get_int("MEMORY_MAX_TURNS", default=3, min_val=1, max_val=10)
CHUNK_SIZE: int = _get_int("CHUNK_SIZE", default=500, min_val=100, max_val=2000)
CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", default=50, min_val=0, max_val=500)
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "icu_documents")


# ---------------------------------------------------------------------------
# Self-test (python config/settings.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Settings loaded ===")
    print(f"  OPENAI_MODEL             : {OPENAI_MODEL}")
    print(f"  EMBEDDING_MODEL          : {EMBEDDING_MODEL}")
    print(f"  CLASSIFIER_PATH          : {CLASSIFIER_PATH}")
    print(f"  CHROMA_DB_PATH           : {CHROMA_DB_PATH}")
    print(f"  DOCUMENTS_PATH           : {DOCUMENTS_PATH}")
    print(f"  DATASET_PATH             : {DATASET_PATH}")
    print(f"  LOG_DIR                  : {LOG_DIR}")
    print(f"  CAMEMBERT_CONFIDENCE     : {CAMEMBERT_CONFIDENCE_THRESHOLD}")
    print(f"  CHROMA_TOP_K             : {CHROMA_TOP_K}")
    print(f"  MEMORY_MAX_TURNS         : {MEMORY_MAX_TURNS}")
    print("OK — no errors")

