# Deployment

> **Status: this prototype has not been deployed to any real hosting target.** This document
> describes the steps that are real and executable today (build, prepare data, run locally /
> via Docker) and clearly marks what is not yet decided.

## Prerequisites

- Python 3.12, or Docker (see [README.md](../../README.md#15-docker))
- An OpenAI API key
- The Chroma index built (`python scripts/index_rag_documents.py`)
- The CamemBERT classifier trained/restored under `models/camembert_classifier/`

## Environment

Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY`. See `.env.example` for the full,
real list of variables (paths, thresholds, chunking parameters).

## Secrets

- Never commit `.env` or any file containing a real `sk-...` key.
- In CI, secrets must be provided via the platform's secret store (GitHub Actions secrets),
  never hardcoded in a workflow file.

## Build

Local (no Docker):

```bash
pip install -r requirements.txt
```

Docker:

```bash
docker compose build
```

## Chroma preparation

```bash
python scripts/validate_corpus.py   # PASS/FAIL report on data/rag_documents/sources.csv
python scripts/index_rag_documents.py
```

## Model preparation

The CamemBERT checkpoint must exist under `models/camembert_classifier/` before starting the
app (see `notebooks/04_camembert_training_pipeline.ipynb` to train it, or restore it from the
committed Git LFS copy of `models/camembert_classifier/model.safetensors`).

## Docker

```bash
docker compose up --build
```

## Deploy

No hosting target (cloud provider, on-prem server, PaaS) has been chosen for this prototype yet,
and no continuous-deployment workflow is configured — only continuous integration (`ci.yml`,
`security.yml`) runs automatically. Deployment, when a target is selected, is a manual step:
`docker compose build` / `docker compose up` (see above).

## Health check

```bash
curl http://localhost:8501/_stcore/health
```

Also run, from inside the running environment:

```bash
python scripts/run_smoke_tests.py
```

## Rollback

Until a real registry/deployment target exists: keep the previous working Docker image tag
(`icu-chatbot:<previous-sha-or-tag>`) available locally and re-run `docker compose up` pointing
at it if the new deployment fails its health check.

## Logs

`src/audit/anonymized_logger.py` writes one JSON line per request to `logs/interactions.jsonl`
(see `README.md` section 17); logs are excluded from version control.
