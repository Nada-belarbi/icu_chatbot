# Unit tests — AnonymizedLogger (Phase 15)

import json
import pytest
from pathlib import Path
from src.audit.anonymized_logger import AnonymizedLogger


@pytest.fixture
def logger(tmp_path):
    return AnonymizedLogger(log_dir=tmp_path / "logs")


@pytest.fixture
def log_file(tmp_path):
    return tmp_path / "logs" / "interactions.jsonl"


def _write_entry(logger):
    logger.log(
        raw_text="Qu'est-ce qu'un respirateur ?",
        decision="allow_rag",
        sources=["sfar.pdf"],
        response_type="informative",
        predicted_class="information_generale",
        confidence_score=0.95,
        post_check_modified=False,
    )


class TestLogFileCreation:
    def test_log_dir_created(self, tmp_path):
        log_dir = tmp_path / "new_logs"
        AnonymizedLogger(log_dir=log_dir)
        assert log_dir.exists()

    def test_log_file_created_after_first_entry(self, logger, log_file):
        _write_entry(logger)
        # Flush the handler
        for h in logger._logger.handlers:
            h.flush()
        assert log_file.exists()


class TestLogEntryContent:
    def _read_last_entry(self, log_file) -> dict:
        for h in log_file.parent.parent.iterdir():
            pass  # ensure file exists
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        return json.loads(lines[-1])

    def test_entry_is_valid_json(self, logger, log_file):
        _write_entry(logger)
        for h in logger._logger.handlers:
            h.flush()
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            json.loads(line)  # must not raise

    def test_raw_text_not_in_log(self, logger, log_file):
        _write_entry(logger)
        for h in logger._logger.handlers:
            h.flush()
        content = log_file.read_text(encoding="utf-8")
        assert "respirateur" not in content

    def test_all_required_fields_present(self, logger, log_file):
        _write_entry(logger)
        for h in logger._logger.handlers:
            h.flush()
        entry = json.loads(log_file.read_text(encoding="utf-8").strip().splitlines()[-1])
        required = {
            "timestamp", "query_hash", "predicted_class",
            "confidence_score", "decision", "sources_used",
            "response_type", "post_check_modified",
        }
        assert required.issubset(entry.keys())

    def test_query_hash_is_sha256(self, logger, log_file):
        _write_entry(logger)
        for h in logger._logger.handlers:
            h.flush()
        entry = json.loads(log_file.read_text(encoding="utf-8").strip().splitlines()[-1])
        assert len(entry["query_hash"]) == 64  # SHA-256 hex digest

    def test_sources_is_list(self, logger, log_file):
        _write_entry(logger)
        for h in logger._logger.handlers:
            h.flush()
        entry = json.loads(log_file.read_text(encoding="utf-8").strip().splitlines()[-1])
        assert isinstance(entry["sources_used"], list)

    def test_post_check_modified_is_bool(self, logger, log_file):
        _write_entry(logger)
        for h in logger._logger.handlers:
            h.flush()
        entry = json.loads(log_file.read_text(encoding="utf-8").strip().splitlines()[-1])
        assert isinstance(entry["post_check_modified"], bool)

    def test_multiple_entries(self, logger, log_file):
        _write_entry(logger)
        _write_entry(logger)
        for h in logger._logger.handlers:
            h.flush()
        lines = [l for l in log_file.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
        assert len(lines) == 2


class TestAnonymization:
    def test_same_text_same_hash(self, tmp_path):
        a = AnonymizedLogger(log_dir=tmp_path / "a")
        b = AnonymizedLogger(log_dir=tmp_path / "b")
        log_a = tmp_path / "a" / "interactions.jsonl"
        log_b = tmp_path / "b" / "interactions.jsonl"

        for lg, lf in [(a, log_a), (b, log_b)]:
            lg.log("même question", "refuse", [], "refuse")
            for h in lg._logger.handlers:
                h.flush()

        entry_a = json.loads(log_a.read_text(encoding="utf-8").strip())
        entry_b = json.loads(log_b.read_text(encoding="utf-8").strip())
        assert entry_a["query_hash"] == entry_b["query_hash"]

    def test_different_text_different_hash(self, logger, log_file):
        logger.log("question A", "allow_rag", [], "informative")
        logger.log("question B", "allow_rag", [], "informative")
        for h in logger._logger.handlers:
            h.flush()
        lines = [l for l in log_file.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
        hashes = [json.loads(l)["query_hash"] for l in lines]
        assert hashes[0] != hashes[1]
