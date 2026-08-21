# Security Policy

## Scope

This repository is a research prototype (POC/MVS) for an ICU family-information assistant. It is
**not a certified medical device** and has **not** been deployed to a real hospital environment.

## Supported versions

No stable release exists yet; only the current `main` branch is maintained.

## Reporting a vulnerability

If you find a security issue (e.g. a way to bypass the safety pipeline, extract another user's
conversation data, or leak a secret), please report it privately to the project maintainer
rather than opening a public issue.

## Secret handling

- API keys and other secrets live only in `.env` (git-ignored) or in the deployment platform's
  secret store (e.g. GitHub Actions secrets). They must never be committed.
- `.github/workflows/security.yml` runs a secret scan (Gitleaks) and a dependency vulnerability
  scan (`pip-audit`) on every push/pull request and weekly.
- If a secret is ever committed, rotating the key is not sufficient — the commit must be scrubbed
  from Git history (`git filter-repo` or equivalent) since the old value remains readable in past
  commits otherwise.

## Data privacy

- Users should not enter information that identifies a specific patient (name, exact dates,
  medical record numbers).
- Conversation history is encrypted at rest (`src/memory/conversation_repository.py`, Fernet) and
  scoped per authenticated user (`src/auth/local_auth.py`).
- Logs (`logs/interactions.jsonl`) store only anonymized SHA-256 query hashes, never raw text.

## Known prototype limitations

- The LLM call depends on the OpenAI API — no data-sovereignty guarantee for a real hospital
  deployment in the current architecture.
- No formal penetration test or third-party security audit has been performed.
- See `evaluation_outputs/final_metrics.json` and `notebooks/FINAL_PROJECT_EVALUATION.ipynb` for
  measured safety metrics (routing accuracy, unsafe RAG leakage rate, critical safety failure
  rate) — these are real, measured numbers, not assumptions.
