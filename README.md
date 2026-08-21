# ICU Information Assistant

> ⚠️ **This system never provides individualized medical advice, diagnosis, treatment or
> prognosis.** It is a prototype (POC/MVS) and is **not suitable for real hospital deployment**
> in its current form (OpenAI API data-sovereignty constraints, no clinical validation).

## Table of contents

1. [Overview](#1-overview)
2. [Main features](#2-main-features)
3. [Safety scope](#3-safety-scope)
4. [Architecture](#4-architecture)
5. [Technology stack](#5-technology-stack)
6. [Project structure](#6-project-structure)
7. [Prerequisites](#7-prerequisites)
8. [Installation](#8-installation)
9. [Environment configuration](#9-environment-configuration)
10. [Building the knowledge base](#10-building-the-knowledge-base)
11. [Loading the classifier](#11-loading-the-classifier)
12. [Running the application](#12-running-the-application)
13. [Running tests](#13-running-tests)
14. [Evaluation](#14-evaluation)
15. [Docker](#15-docker)
16. [Deployment](#16-deployment)
17. [Logging](#17-logging)
18. [Troubleshooting](#18-troubleshooting)
19. [Security and privacy](#19-security-and-privacy)
20. [Known limitations](#20-known-limitations)
21. [License](#21-license)

---

## 1. Overview

An assistant providing **general information** about ICU (intensive/critical care) practices to
family members of patients, in French. It uses a Retrieval-Augmented Generation (RAG) pipeline
grounded in an institutional medical corpus (learned societies, hospitals, health authorities),
with a multi-layer safety pipeline that systematically refuses personalized medical advice,
diagnosis, treatment or prognosis requests, and redirects emergencies immediately.

## 2. Main features

- RAG-based general medical information retrieval, grounded in cited institutional sources
- Chroma vector database (local, persisted on disk)
- CamemBERT query classification (French, fine-tuned) with a deterministic keyword `RuleFilter`
- `DecisionEngine` routing to `allow_rag` / `refuse` / `refuse_oos` / `redirect` / `ask_reformulation`
- Controlled response templates for every non-`allow_rag` outcome
- LLM generation (OpenAI API) grounded only in retrieved sources
- Post-generation safety checking (`PostGenerationChecker`) before any response is shown
- Conversation memory (short context window fed to the classifier, not the LLM directly)
- Source citations displayed with every RAG-based answer
- Local authentication (scrypt password hashing) and encrypted, per-user conversation history

## 3. Safety scope

| The assistant DOES | The assistant NEVER does |
|---|---|
| Explain general ICU concepts (ventilation, monitoring, visiting policies, etc.) | Give a diagnosis, treatment or prognosis for a specific patient |
| Cite the institutional documents it used | Answer emergencies conversationally (it redirects instead) |
| Refuse or redirect when a request is out of scope or unsafe | Answer when the corpus has no relevant source (it abstains) |
| Ask the user to rephrase when confidence is low | Use the LLM's own medical knowledge outside the retrieved context |

## 4. Architecture

There are two separate pipelines: an offline **knowledge-base build** pipeline (run when the
corpus or model changes) and the **runtime** pipeline (run once per user message).

```text
User question
      │
      ▼
 TextCleaner
      │
      ▼  ┌─ ConversationMemoryManager (last N turns injected as classifier context)
      ▼
 RuleFilter ──────────────────────────────────────────► redirect / refuse / refuse_oos
      │ (not blocked)
      ▼
 CamembertQueryClassifier  (loads from models/camembert_classifier/)
      │ (label + confidence score)
      ▼
 DecisionEngine
      ├─ allow_rag  ──► Retriever (queries Chroma) ──► RAGChain (OpenAI) ──► GuardrailsManager /
      │                                                                      PostGenerationChecker
      ├─ refuse / refuse_oos / redirect / ask_reformulation ──► ControlledResponseManager
      │
      ▼
 AnonymizedLogger (every request, regardless of branch)
```

Detailed diagrams (sequence, security chain, document pipeline) are in
[docs/figures/architecture/](docs/figures/architecture/). A full narrative description of the
architecture is in the academic report (`report_v2/`).

## 5. Technology stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| RAG orchestration | LangChain (`langchain-chroma`, `langchain-openai`) |
| Vector store | ChromaDB (local, persisted) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (local) |
| LLM | OpenAI API (prototype only) |
| Query classifier | `HybridQueryClassifier` in production: CamemBERT (fine-tuned, French) as primary, with a TF-IDF + LinearSVC baseline as a safety-net specialist for the `hors_perimetre` class |
| Auth | scrypt password hashing (`src/auth/local_auth.py`) |
| Conversation storage | Fernet-encrypted local storage (`src/memory/conversation_repository.py`) |

## 6. Project structure

```text
icu_chatbot/
├── app.py                       # Streamlit entry point (composition root only)
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Testing / linting / notebooks / training
├── pyproject.toml               # Ruff / Black / Mypy configuration
├── pytest.ini
├── .env.example
│
├── config/
│   ├── settings.py              # Paths, thresholds, model names — loaded from .env
│   ├── safety_config.py         # Rule keywords, labels, thresholds
│   └── label_schema.py          # Classifier label constants
│
├── src/
│   ├── frontend/                # Streamlit UI (ui.py, styles.py, icons.py, formatting.py)
│   ├── auth/                    # LocalAuthService (scrypt)
│   ├── memory/                  # ConversationMemoryManager, ConversationRepository (encrypted)
│   ├── preprocessing/           # TextCleaner
│   ├── safety/                  # RuleFilter, QueryClassifier(s), DecisionEngine,
│   │                            # ControlledResponseManager, GuardrailsManager, PostGenerationChecker
│   ├── rag/                     # DocumentLoader, VectorStoreManager, Retriever, RAGChain
│   ├── audit/                   # AnonymizedLogger
│   ├── pipeline/                # ChatbotPipeline — single orchestrator, only class app.py calls
│   ├── ml/                      # CamemBERT training/evaluation utilities (used by notebooks/scripts)
│   └── evaluation/               # Baseline classifier evaluation utilities
│
├── data/
│   ├── rag_documents/
│   │   ├── raw/                 # Source documents, organized by organization (SRLF, HAS, NHS, ...)
│   │   ├── chroma_db/           # Chroma persistent vector store (committed via git so a fresh clone needs no reindexing)
│   │   └── sources.csv          # Corpus manifest — source of truth for provenance/approval status
│   ├── processed/               # Classifier train/val/test/dataset_master CSVs
│   ├── augmentation/            # Manual safety-test and augmentation examples
│   └── auth/, conversations/, private_conversations/   # Local user + encrypted conversation data
│
├── models/
│   └── camembert_classifier/    # Fine-tuned model weights (model.safetensors committed via Git LFS)
│
├── notebooks/                   # Training / evaluation only — never required by the runtime app
│   ├── 01_icu_classifier_dataset_analysis.ipynb
│   ├── 02_baseline_text_classifiers.ipynb
│   ├── 03_baseline_augmented_rulefilter.ipynb
│   ├── 04_camembert_training_pipeline.ipynb
│   ├── 05_camembert_evaluation_report.ipynb
│   ├── 06_safety_pipeline_audit.ipynb
│   ├── 07_index_rag_documents.ipynb
│   └── FINAL_PROJECT_EVALUATION.ipynb   # Official quantitative evaluation notebook
│
├── scripts/                     # CLI entry points (dataset build, training, evaluation, indexing)
│
├── tests/
│   ├── unit/                    # One test module per component (20 files)
│   ├── integration/              # Multi-component tests (pipeline, memory+decision, artifacts)
│   └── security/                 # Adversarial / safety-critical tests
│
├── evaluation_outputs/           # final_metrics.json, data_availability.csv, error_cases.csv, ...
├── docs/                          # Figures, tables and deployment/troubleshooting notes
├── report_v2/                     # Academic report (LaTeX sources + compiled PDF), delivered separately
└── logs/                          # Anonymized interaction logs (not committed)
```

> **Note:** `tests/` is currently organized as `unit/` / `integration/` / `security/`. The
> `security/` folder is the safety-critical adversarial suite (equivalent to a `safety` marker,
> already declared in `pytest.ini`).

## 7. Prerequisites

- Python 3.12
- [Git LFS](https://git-lfs.com/) (`git lfs install`) — the trained CamemBERT checkpoint
  (`models/camembert_classifier/model.safetensors`, ~420MB) is stored via Git LFS since it
  exceeds GitHub's 100MB per-file limit; without Git LFS installed *before* cloning, you get a
  small pointer file instead of the real weights
- An OpenAI API key (for the LLM generation step only — retrieval and classification run locally)
- ~2 GB free disk space (CamemBERT weights + embedding model + Chroma index)

## 8. Installation

```bash
git lfs install        # once per machine, before cloning
git clone <repo-url>
cd icu_chatbot

python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
# add -r requirements-dev.txt for tests, linting, and notebooks
```

## 9. Environment configuration

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # Linux / macOS
```

Edit `.env` and set `OPENAI_API_KEY` (required — the app raises a clear
`EnvironmentError` at startup if it is missing, see `config/settings.py`). All other variables
have working defaults; see `.env.example` for the full list (paths, `CAMEMBERT_CONFIDENCE_THRESHOLD`,
`CHROMA_TOP_K`, `CHUNK_SIZE`/`CHUNK_OVERLAP`, `MEMORY_MAX_TURNS`).

## 10. Building the knowledge base

The Chroma index (`data/rag_documents/chroma_db/`) and the corpus manifest
(`data/rag_documents/sources.csv`) are already committed to this repository — a fresh clone
does **not** need to rebuild the index to run the app. Re-run the build only after adding or
changing documents:

```bash
python scripts/index_rag_documents.py
```

This loads and chunks the source documents (`CHUNK_SIZE`/`CHUNK_OVERLAP` from `.env`), embeds them
locally, and persists the collection to `data/rag_documents/chroma_db/`. Commit the updated index
afterward if you want the change to reach other clones of this repository.

## 11. Loading the classifier

The runtime loads the already-trained CamemBERT checkpoint from `models/camembert_classifier/`
(committed via Git LFS, see [Prerequisites](#7-prerequisites)) — no notebook is required to run
the app. To retrain or re-evaluate the classifier, see
`notebooks/04_camembert_training_pipeline.ipynb` and `notebooks/05_camembert_evaluation_report.ipynb`,
or the equivalent CLI scripts `scripts/train_camembert.py` / `scripts/evaluate_camembert.py`.

## 12. Running the application

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). The Chroma index and the CamemBERT model must
already exist (see sections 10 and 11) before the first run.

## 13. Running tests

```bash
pytest                     # all 252 tests
pytest tests/unit/
pytest tests/security/     # adversarial / safety-critical suite
pytest --cov=src --cov-report=term-missing
```

## 14. Evaluation

The official, reproducible quantitative evaluation lives in
[`notebooks/FINAL_PROJECT_EVALUATION.ipynb`](notebooks/FINAL_PROJECT_EVALUATION.ipynb). It computes
every metric from real project artifacts only (no invented numbers) and exports:

- `evaluation_outputs/final_metrics.json` — single source of numbers for any report
- `evaluation_outputs/data_availability.csv`, `error_cases.csv`, `missing_evaluation_data.csv`
- `docs/tables/*.csv` and `docs/figures/evaluation/*.svg`/`*.png`

## 15. Docker

A `Dockerfile` and `docker-compose.yml` are provided. The image installs `requirements.txt` and
copies only `app.py`, `config/`, `src/` and `scripts/`; `.env`, `data/`, `models/` and `logs/` are
mounted as volumes (see `docker-compose.yml`) so the Chroma index, the CamemBERT weights and
conversation data survive container rebuilds and are never baked into the image.

```bash
docker compose build
docker compose up
```

This has not been verified end-to-end in every environment (no running Docker daemon was
available when this section was last checked) — verify locally with the commands above before
relying on it for a real run.

## 16. Deployment

This prototype has **not** been deployed to a hospital environment and is not currently packaged
for one (see [Known limitations](#20-known-limitations)). See `docs/deployment/DEPLOYMENT.md` for
the real, executable steps (build, prepare data, run locally or via Docker) and what is not yet
decided (hosting target).

## 17. Logging

`src/audit/anonymized_logger.py` writes one JSON line per request to `logs/interactions.jsonl`:
timestamp, a SHA-256 hash of the query (never the raw text), predicted label, confidence,
decision, sources used, response type, and whether the post-generation checker modified the
response. Logs are excluded from version control (`.gitignore`).

## 18. Troubleshooting

| Symptom | Probable cause | Solution |
|---|---|---|
| `EnvironmentError: OPENAI_API_KEY is not set` at startup | `.env` missing or empty key | `copy .env.example .env` and fill in the key |
| Classifier fails to load / errors mentioning a tiny or invalid `model.safetensors` | Repository was cloned without Git LFS, so the file is a small text pointer instead of the real weights | Install [Git LFS](https://git-lfs.com/), then run `git lfs pull` in the repo |
| Every answer is "no source found" | Chroma index missing, empty, or unreadable by the installed `chromadb` version | Re-run `python scripts/index_rag_documents.py`; confirm `chromadb`/`langchain-chroma` versions match the pins in `requirements.txt` |
| Classifier fails to load | `models/camembert_classifier/` missing or incomplete | Train it via `notebooks/04_camembert_training_pipeline.ipynb` or restore the checkpoint |
| Streamlit shows stale UI after editing `src/frontend/` | An old `streamlit run` process is still serving the previous version | Stop the old process (check the port with `Get-NetTCPConnection -LocalPort 8501`) and restart |
| Slow first response (~60-90s) | CamemBERT + embeddings cold start | Expected on first load per process; subsequent requests are fast |

## 19. Security and privacy

- Do not enter information that could identify a specific patient (name, exact dates, medical
  record numbers) — the assistant is not designed to store or process such data safely.
- Conversation history is encrypted at rest (`ConversationRepository`) and scoped per authenticated
  user; logs never store raw user text, only anonymized hashes.
- Never commit `.env`, API keys, or trained model weights to version control (see `.gitignore`).

## 20. Known limitations

- General information only — not diagnostic, not therapeutic, not prognostic.
- Answers are limited to what the indexed corpus covers; see `data/rag_documents/sources.csv` for
  the full source manifest and its approval statuses.
- Prototype / MVS stage — not a certified medical device, not validated for clinical use, not
  suitable for real hospital deployment (OpenAI API data-sovereignty constraints).
- Quantitative results (accuracy, safety rates, known gaps) are tracked in
  `evaluation_outputs/final_metrics.json` and `docs/tables/`/`docs/figures/evaluation/` — see the
  academic report (`report_v2/`) for the full analysis.

## 21. License

See `LICENSE` — all rights reserved until an explicit open-source license is chosen. Do not
redistribute.
