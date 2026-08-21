"""
scripts/pre_deploy_check.py

Aggregate pre-deployment gate. Runs the other validation scripts plus a
lightweight secret scan and prints a single PASS/FAIL summary per gate.

Usage:
    python scripts/pre_deploy_check.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

# Matches a real OpenAI-style secret key, not the placeholder "sk-..." in .env.example.
SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9]{20,}")
SCAN_EXTENSIONS = {".py", ".md", ".txt", ".ini", ".toml", ".yml", ".yaml", ".json", ".ipynb"}
SCAN_EXCLUDE_DIRS = {".venv", ".git", "node_modules", "__pycache__", ".pytest_cache"}


def _run_script(script: str) -> bool:
    result = subprocess.run([PYTHON, str(PROJECT_ROOT / "scripts" / script)], cwd=PROJECT_ROOT)
    return result.returncode == 0


def _scan_for_secrets() -> list[str]:
    hits = []
    for path in PROJECT_ROOT.rglob("*"):
        if path.suffix not in SCAN_EXTENSIONS or not path.is_file():
            continue
        if any(part in SCAN_EXCLUDE_DIRS for part in path.parts):
            continue
        if path.name == ".env.example":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if SECRET_PATTERN.search(text):
            hits.append(str(path.relative_to(PROJECT_ROOT)))
    return hits


def _report(label: str, ok: bool) -> bool:
    print(f"{label:.<22}{'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("=== Pre-deployment check ===\n")

    gates = {}
    gates["Environment"] = _run_script("validate_environment.py")
    gates["Manifest"] = _run_script("validate_corpus.py")
    gates["Smoke tests"] = _run_script("run_smoke_tests.py")

    secret_hits = _scan_for_secrets()
    gates["Secrets scan"] = len(secret_hits) == 0

    print()
    for label, ok in gates.items():
        _report(label, ok)
    if secret_hits:
        print("  Possible secret(s) found in:")
        for hit in secret_hits:
            print(f"    - {hit}")

    print()
    if all(gates.values()):
        print("READY FOR DEPLOYMENT")
        return 0
    print("NOT READY")
    return 1


if __name__ == "__main__":
    sys.exit(main())
