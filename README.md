# Chatbot conversationnel pour la réanimation

A conversational AI assistant providing **general medical information** to family members of ICU patients.  
Built with a hybrid RAG pipeline and a strict multi-layer safety system.

> ⚠️ **This system never provides individualized medical advice.**  
> It is a prototype only. It is **not suitable for real hospital deployment** in its current form (OpenAI API data sovereignty constraints).

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Project structure](#3-project-structure)
4. [Safety pipeline](#4-safety-pipeline)
5. [Document strategy](#5-document-strategy)
6. [Fallback & error handling](#6-fallback--error-handling)
7. [Installation](#7-installation)
8. [Usage](#8-usage)
9. [Running tests](#9-running-tests)
10. [Development order](#10-development-order)
11. [Ethical & regulatory limits](#11-ethical--regulatory-limits)

---

## 1. Project overview

Family members of ICU patients face high emotional stress and limited access to clear medical information. This chatbot aims to:

- Answer **general** questions about ICU care in plain French
- Prepare families for conversations with healthcare professionals
- Systematically **refuse** personalized medical advice, diagnosis, prognosis, and treatment requests
- Redirect emergencies to the medical team immediately

**Tech stack**

| Layer | Technology |
|---|---|
| UI | Streamlit |
| RAG orchestration | LangChain |
| Vector store | ChromaDB (local) |
| Embeddings | sentence-transformers (local) |
| LLM | OpenAI API (prototype only) |
| Query classifier | CamemBERT (fine-tuned, French) |
| Memory | Custom `ConversationMemoryManager` (`st.session_state`) |

---

## 2. Architecture

There are **two separate pipelines**: an offline setup pipeline (run once) and the runtime pipeline (run per user message).

### Offline pipeline (run once before starting the app)

```
data/questions_dataset/          data/documents/
         │                               │
         ▼ 01_prepare_dataset.ipynb      ▼ 03_index_documents.ipynb
  dataset_clean.csv              document_loader.py → vector_store.py
         │                               │
         ▼ 02_train_camembert.ipynb      ▼
  models/camembert_classifier/   data/chroma_db/
         │                               │
         └──────────┬────────────────────┘
                    ▼
             loaded at runtime by
     query_classifier.py   retriever.py
```

### Runtime pipeline (per user message)

```
User question
      │
      ▼
 TextCleaner
      │
      ▼  ┌─ ConversationMemoryManager (last N turns injected as context)
      ▼
 RuleFilter ──────────────────────────────────────────► BLOCK / REDIRECT
      │ (not blocked)
      ▼
 CamembertQueryClassifier  (loads from models/camembert_classifier/)
      │ (label + confidence score)
      ▼
 DecisionEngine
      ├─ allow_rag  ──► Retriever (queries chroma_db/) ──► RAGChain ──► PostGenerationChecker ──► Response
      ├─ refuse          ──► ControlledResponseManager
      ├─ redirect        ──► ControlledResponseManager (emergency)
      └─ ask_reformulation ► ControlledResponseManager
                                                                               │
                                                                               ▼
                                                                       AnonymizedLogger
```

**Key design principles:**
- The LLM is only called after all safety gates pass — never automatically
- `chroma_db/` and `models/camembert_classifier/` are the bridges between offline and runtime — both must exist before launching `app.py`
- Conversation memory feeds the **classifier** as context, not the LLM — it is a safety feature, not just UX
  - Keep **only the last 2–3 cleaned messages** — injecting too much history degrades classifier accuracy
  - Implemented as a plain Python list in `st.session_state`, not LangChain memory (LangChain memory API changes frequently across versions)
- No user accounts or persistent profiles — `st.session_state` handles session scope; memory is wiped on tab close
- `ChatbotPipeline` in `src/pipeline/chatbot_pipeline.py` is the single orchestrator — `app.py` calls only this

---

## 3. Project structure

```
icu_chatbot/
│
├── app.py                                # Streamlit entry point
├── .env.example                          # Environment variable template
├── .gitignore
├── requirements.txt
├── pytest.ini
│
├── config/
│   ├── settings.py                       # Paths, thresholds, model names, API settings
│   └── safety_config.py                  # Safety rules, classifier classes, refusal messages
│
├── data/
│   ├── documents/                        # Validated source documents for the RAG (PDF, TXT, DOCX)
│   ├── questions_dataset/                # Annotated dataset for CamemBERT fine-tuning (CSV)
│   └── chroma_db/                        # Chroma persistent vector store (auto-generated, not committed)
│
├── src/
│   ├── memory/
│   │   └── conversation_memory.py        # Session-scoped conversation history (no PII, no persistence)
│   │
│   ├── preprocessing/
│   │   └── text_cleaner.py               # Input normalization (whitespace, encoding)
│   │
│   ├── safety/
│   │   ├── controlled_responses.py       # All pre-written refusal / redirection messages
│   │   ├── rule_filter.py                # Fast keyword/regex rules — no ML cost
│   │   ├── query_classifier.py           # CamemBERT classifier (+ MockClassifier for early dev)
│   │   ├── decision_engine.py            # Combines rule output + ML label + score → action
│   │   ├── guardrails.py                 # Pre / during / post LLM controls (LangChain)
│   │   └── post_checker.py               # Last gate: scans response before display
│   │
│   ├── rag/
│   │   ├── document_loader.py            # Loads and chunks source documents
│   │   ├── vector_store.py               # Creates / loads Chroma collection
│   │   ├── retriever.py                  # Top-k similarity search
│   │   └── rag_chain.py                  # LangChain chain: docs → prompt → OpenAI → answer + sources
│   │
│   ├── audit/
│   │   └── anonymized_logger.py          # Logs decisions anonymously (no raw user text stored)
│   │                                       # NOTE: named 'audit' not 'logging' to avoid
│   │                                       # shadowing Python's stdlib logging module
│   ├── pipeline/
│   │   └── chatbot_pipeline.py           # Central orchestrator — single entry point for app.py
│   │                                       # Calls: cleaner→memory→rule_filter→classifier→
│   │                                       #        decision_engine→retriever→rag_chain→
│   │                                       #        post_checker→logger
│   │                                       # Handles all fallback cases internally
│   │
│   └── evaluation/
│       ├── test_questions.py             # 50-question validated bank (5 categories)
│       └── evaluator.py                  # Runs pipeline on test bank, computes metrics
│
├── models/
│   └── camembert_classifier/             # Saved fine-tuned model weights (not committed to git)
│
├── notebooks/
│   ├── 01_prepare_dataset.ipynb          # Clean + annotate + split the classifier dataset
│   ├── 02_train_camembert.ipynb          # Fine-tune CamemBERT, evaluate, save model
│   ├── 03_index_documents.ipynb          # Load documents, embed, index into Chroma
│   ├── 04_test_pipeline_modules.ipynb    # Validate every src/ module in isolation + in sequence
│   └── 05_evaluate_chatbot.ipynb         # Full end-to-end evaluation on 50-question test bank
│
└── tests/
    ├── unit/                             # One test file per component
    │   ├── test_text_cleaner.py
    │   ├── test_rule_filter.py
    │   ├── test_decision_engine.py
    │   ├── test_memory_manager.py
    │   ├── test_post_checker.py
    │   └── test_controlled_responses.py
    ├── integration/
    │   └── test_pipeline.py              # Full pipeline with MockClassifier + MockOpenAI
    └── security/
        └── test_safety.py                # Adversarial inputs, prompt injection attempts
```

---

## 4. Safety pipeline

> The safety module stubs in `src/safety/` are intentionally kept separate from
> `src/pipeline/chatbot_pipeline.py`. Each can be developed, mocked, and tested
> in isolation. The pipeline wires them together.

The system uses **5 decision outcomes** — the LLM is never called for the last four:

| Decision | Trigger | Output |
|---|---|---|
| `allow_rag` | General question, high confidence | RAG → LLM → post-check → response |
| `refuse` | Medical advice / out-of-scope | Controlled refusal message |
| `redirect` | Emergency situation | Immediate redirect to medical team |
| `ask_reformulation` | Low classifier confidence | Ask user to rephrase |

**Three-level guardrails:**
1. **Pre-generation** — blocks LLM call if safety conditions not met
2. **During generation** — strict system prompt (no diagnosis, no treatment, source-only)
3. **Post-generation** — `PostGenerationChecker` scans response before display; replaces if unsafe

**Classifier classes (CamemBERT):**
- `information_generale`
- `sensible`
- `avis_medical_personnalise`
- `urgence`
- `hors_perimetre`

---

## 5. Document strategy

The RAG system is only as safe as the documents it cites. All documents placed in `data/documents/` must meet these criteria:

| Criterion | Requirement |
|---|---|
| **Medical validation** | Reviewed and approved by a qualified healthcare professional |
| **Source** | Official bodies only (HAS, SFAR, CHU guidelines, validated patient information sheets) |
| **Date** | Each document must include a production or last-review date |
| **Traceability** | Filename must encode the source and date, e.g. `SFAR_sedation_guide_2024.pdf` |
| **Language** | French only |
| **No web content** | Never index arbitrary web pages or unvalidated online text |
| **Versioning** | When a document is updated, the old version is archived and the index is rebuilt (re-run notebook 03) |

> If `data/documents/` is empty or no relevant chunk is found for a query,
> the chatbot returns a controlled `no_source_found` message — it never invents an answer.

---

## 6. Fallback & error handling

`ChatbotPipeline` catches all failure modes internally. The UI (`app.py`) always receives a safe string response — it never crashes.

| Failure | Fallback response |
|---|---|
| Empty / whitespace user input | `ask_reformulation` message |
| Chroma DB empty or not initialised | `no_source_found` controlled message |
| Retriever returns no results | `no_source_found` controlled message |
| OpenAI API call fails (timeout, quota) | `api_unavailable` controlled message |
| `PostGenerationChecker` rejects the answer | `refuse_medical_advice` controlled message |
| Any unexpected exception | `generic_error` controlled message + anonymous error log |

All fallback messages are defined in `src/safety/controlled_responses.py`. The pipeline logs the failure type (not the user's raw text) via `AnonymizedLogger`.

---

## 7. Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd icu_chatbot

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
copy .env.example .env        # Windows
# cp .env.example .env        # Linux / macOS
# Edit .env and add your OPENAI_API_KEY
```

### One-time offline setup (run in order before first launch)

| Step | Notebook | What it produces |
|---|---|---|
| 1 | `notebooks/01_prepare_dataset.ipynb` | `data/questions_dataset/dataset_clean.csv` |
| 2 | `notebooks/02_train_camembert.ipynb` | `models/camembert_classifier/` |
| 3 | `notebooks/03_index_documents.ipynb` | `data/chroma_db/` |

> Steps 1 and 2 require a labelled dataset in `data/questions_dataset/`.  
> Step 3 requires at least one document in `data/documents/`.  
> Steps 1–3 only need to be re-run when data or documents change.

---

## 8. Usage

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**First run:** place validated documents in `data/documents/` and run the indexing step above before starting the app.

---

## 9. Running tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Security tests only
pytest tests/security/

# With coverage
pytest --cov=src
```

---

## 10. Development order

Follow this commit sequence to minimise blockers:

| # | Commit | Why this position |
|---|---|---|
| 1 | Project structure | Foundation for all imports |
| 2 | `config/settings.py` + `safety_config.py` | Every module imports from here |
| 3 | `controlled_responses.py` | Every safety gate needs refusal messages |
| 4 | `text_cleaner.py` | First step of every request |
| 5 | `rule_filter.py` | Fastest safety layer, fully testable alone |
| 6 | `query_classifier.py` — **MockClassifier** | `DecisionEngine` depends on classifier interface — define it early |
| 7 | `conversation_memory.py` | Simple list in `st.session_state`, no LangChain dependency |
| 8 | `decision_engine.py` | Now has all its dependencies (rule_filter + mock classifier) |
| 9 | `chatbot_pipeline.py` (skeleton) | Wire the modules built so far; grow it as you add more |
| 10 | `document_loader.py` + `vector_store.py` | RAG prerequisite — run notebook 03 after this |
| 11 | `retriever.py` | RAG prerequisite |
| 12 | `rag_chain.py` | Core generation |
| 13 | `post_checker.py` | Last safety gate |
| 14 | `guardrails.py` | Wraps the full chain |
| 15 | `app.py` (Streamlit UI) | Calls only `ChatbotPipeline.run()` |
| 16 | Dataset preparation (notebook 01) | Build + annotate the classifier training data |
| 17 | `02_train_camembert.ipynb` | Train the real classifier |
| 18 | `query_classifier.py` — real CamemBERT | Drop-in replacement for mock |
| 19 | `evaluator.py` + final tests | Measure and document |

> **Key rule:** `MockClassifier` and `CamembertQueryClassifier` share the same interface (`classify(text) -> (label, score)`). Swap them without touching any other module.

---

## 11. Ethical & regulatory limits

- This system **does not process personally identifiable health data**
- Logs store only anonymized hashes — never raw user text
- OpenAI API is used for the prototype only; a production deployment in a hospital environment would require a **100% local model** (data sovereignty)
- The system is not a medical device and must never be presented as one
- All responses citing this system should be reviewed by a healthcare professional before any clinical use
