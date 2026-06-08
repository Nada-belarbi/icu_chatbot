# AnonymizedLogger
#
# Responsibilities:
#   - Log each interaction to logs/ in JSON/CSV format
#   - Fields to log:
#       timestamp | anonymized_query_hash | predicted_class | confidence_score
#       decision | sources_used | response_type | post_check_modified
#   - NEVER log: raw user text, patient names, personal data, IP addresses
#   - Log rotation: configurable max file size / daily rotation

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.settings import LOG_DIR

_module_log = logging.getLogger(__name__)


class AnonymizedLogger:
    """
    Write one JSON line per interaction to ``logs/interactions.jsonl``.

    Raw user text is **never** stored — only its SHA-256 hash.
    Log files are rotated when they exceed *max_bytes* (default 5 MB),
    keeping *backup_count* (default 7) rotated files.

    Parameters
    ----------
    log_dir:
        Directory for log files.  Defaults to ``LOG_DIR``.
    max_bytes:
        Maximum size of the active log file before rotation.
    backup_count:
        Number of backup files to keep.
    """

    def __init__(
        self,
        log_dir: Path | str | None = None,
        max_bytes: int = 5_000_000,
        backup_count: int = 7,
    ) -> None:
        self._log_dir = Path(log_dir) if log_dir else Path(LOG_DIR)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        log_file = self._log_dir / "interactions.jsonl"
        handler = RotatingFileHandler(
            str(log_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        # Dedicated internal logger — one per instance, no propagation
        self._logger = logging.getLogger(f"icu.audit.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.addHandler(handler)
        self._logger.propagate = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        raw_text: str,
        decision: str,
        sources: list[str],
        response_type: str,
        predicted_class: str = "",
        confidence_score: float = 0.0,
        post_check_modified: bool = False,
    ) -> None:
        """
        Write one anonymized JSON log entry.

        Parameters
        ----------
        raw_text:
            Original user input — hashed before storage, never stored as-is.
        decision:
            Routing decision (allow_rag / refuse / redirect / ask_reformulation).
        sources:
            Source filenames used in the RAG answer (empty for refusals).
        response_type:
            One of the RTYPE_* constants from ``chatbot_pipeline``.
        predicted_class:
            Label returned by the classifier (empty string if not applicable).
        confidence_score:
            Classifier confidence (0.0 if not applicable).
        post_check_modified:
            Whether ``PostGenerationChecker`` replaced the LLM answer.
        """
        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "query_hash": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            "predicted_class": predicted_class,
            "confidence_score": round(float(confidence_score), 4),
            "decision": decision,
            "sources_used": list(sources),
            "response_type": response_type,
            "post_check_modified": post_check_modified,
        }
        try:
            self._logger.info(json.dumps(entry, ensure_ascii=False))
        except Exception as exc:  # pragma: no cover
            _module_log.error("AnonymizedLogger failed to write entry: %s", exc)

