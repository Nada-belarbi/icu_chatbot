# Development To-Do List — ICU Chatbot

Progress legend: `[ ]` not started · `[~]` in progress · `[x]` done

---

## PHASE 0 — Project bootstrap

### 0.1 Environment setup
- [ ] Create and activate Python virtual environment (`.venv`)
- [ ] Run `pip install -r requirements.txt`
- [ ] Copy `.env.example` → `.env` and add `OPENAI_API_KEY`
- [ ] Verify Python ≥ 3.10

### 0.2 Git setup
- [ ] `git init` inside `icu_chatbot/`
- [ ] First commit: `initial project structure`
- [ ] Create branches: `dev`, then feature branches per phase

**✓ Exit check:** `python -c "import langchain, openai, chromadb, streamlit, transformers"` runs without error.

---

## PHASE 1 — Configuration

### 1.1 `config/settings.py`
- [ ] Load `.env` with `python-dotenv`
- [ ] Define constants:
  - `OPENAI_API_KEY` (from env)
  - `OPENAI_MODEL` (default: `gpt-4o-mini`)
  - `CAMEMBERT_MODEL_PATH` → `models/camembert_classifier/`
  - `CAMEMBERT_CONFIDENCE_THRESHOLD` (default: `0.70`)
  - `CHROMA_DB_PATH` → `data/chroma_db/`
  - `DOCUMENTS_PATH` → `data/documents/`
  - `DATASET_PATH` → `data/questions_dataset/`
  - `CHROMA_TOP_K` (default: `5`)
  - `MEMORY_MAX_TURNS` (default: `5`)
  - `LOG_DIR` → `logs/`
- [ ] Validate that required values are set (raise on missing API key)
- [ ] Validate that numeric thresholds are in expected ranges

### 1.2 `config/safety_config.py`
- [ ] Define classifier label constants:
  `LABEL_GENERAL`, `LABEL_SENSITIVE`, `LABEL_MEDICAL_ADVICE`, `LABEL_EMERGENCY`, `LABEL_OUT_OF_SCOPE`
- [ ] Define rule keyword lists per category (urgency, diagnosis, treatment, prognosis, personal data)
- [ ] Define all controlled response message strings (in French)
- [ ] Define dangerous response patterns list (used by `PostGenerationChecker`)

**Commit:** `add configuration files`

**✓ Tests (manual):**
- Run `python config/settings.py` → no error, all paths printed
- All threshold values in valid range
- No secret values committed to git

---

## PHASE 2 — Controlled responses

### 2.1 `src/safety/controlled_responses.py`
- [ ] Create `ControlledResponseManager` class
- [ ] Implement response methods:
  - `get_urgency_redirect()` → direct to emergency / medical team
  - `get_refuse_medical_advice()` → explain system limits
  - `get_ask_reformulation()` → invite rephrasing
  - `get_no_source_found()` → no matching document
  - `get_out_of_scope()` → outside perimeter
- [ ] All messages: empathetic French, no medical content, redirect to professional when needed

**Commit:** `add controlled responses`

**✓ Unit tests** (`tests/unit/test_controlled_responses.py`):
- [ ] Each method returns a non-empty string
- [ ] No response contains medical advice keywords
- [ ] Urgency response contains reference to medical team
- [ ] All messages are in French

---

## PHASE 3 — Preprocessing

### 3.1 `src/preprocessing/text_cleaner.py`
- [ ] Create `TextCleaner` class
- [ ] Implement `clean(text: str) -> str`:
  - Strip leading/trailing whitespace
  - Normalize multiple spaces to single space
  - Normalize Unicode (NFC)
  - Preserve medical terms and personal phrasing ("mon père", "ma mère")
- [ ] Raise `ValueError` (or return controlled error) on empty / whitespace-only input

**Commit:** `add text cleaner`

**✓ Unit tests** (`tests/unit/test_text_cleaner.py`):
- [ ] Empty string → raises `ValueError`
- [ ] `"  bonjour  "` → `"bonjour"`
- [ ] Multiple internal spaces → single space
- [ ] `"mon père est intubé"` → personal phrasing preserved unchanged
- [ ] Medical term (e.g. `"réanimation"`) → not altered

---

## PHASE 4 — Conversation memory

> **Why memory matters here:** short follow-up questions like "Est-ce grave ?" or
> "Combien de temps ?" are meaningless without context. Memory feeds the classifier
> with prior turns so ambiguous follow-ups are classified as personalized/prognosis
> and refused — not allowed through as general questions. This is a **safety feature**,
> not just a UX comfort feature.
>
> **No user accounts needed.** Streamlit `st.session_state` is already scoped per
> browser tab. Memory is in-memory only and wiped when the session ends. Persistent
> user profiles would create GDPR risk with no benefit in this use case.

### 4.1 `src/memory/conversation_memory.py`
- [ ] Create `ConversationMemoryManager` class — **plain Python, no LangChain dependency**
  - Use a simple `list` stored in `st.session_state` (avoids LangChain version drift)
- [ ] Implement:
  - `add_user_message(text: str)`
  - `add_assistant_message(text: str)`
  - `get_history() -> list[dict]` — returns last N turns only
  - `get_context_string() -> str` — formats last **2–3 turns** as a short string prefix for the classifier
  - `clear()` — wipe all history
- [ ] Hard limit: `MEMORY_MAX_TURNS = 3` (default in `.env.example`) — more history degrades classifier accuracy
- [ ] No disk persistence — in-memory only
- [ ] No storage of PII

**Commit:** `add conversation memory manager`

**✓ Unit tests** (`tests/unit/test_memory_manager.py`):
- [ ] Add user message → appears in `get_history()`
- [ ] Add assistant message → appears in `get_history()`
- [ ] History capped at `MEMORY_MAX_TURNS`
- [ ] `clear()` → `get_history()` returns empty
- [ ] Two separate instances share no state

---

## PHASE 5 — Rule-based safety filter

### 5.1 `src/safety/rule_filter.py`
- [ ] Create `RuleFilter` class
- [ ] Load keyword lists from `safety_config.py`
- [ ] Implement `filter(text: str) -> tuple[str, bool]`:
  - Returns `(category, blocked)` where category ∈ `{urgency, diagnosis, treatment, prognosis, personal_data, out_of_scope, allowed}`
  - `blocked = True` for everything except `allowed`
- [ ] Rules use case-insensitive regex matching
- [ ] Urgency always takes priority over other categories

**Commit:** `add rule-based safety filter`

**✓ Unit tests** (`tests/unit/test_rule_filter.py`):
- [ ] `"Est-ce que mon père va survivre ?"` → `(prognosis, True)`
- [ ] `"Quel médicament donner ?"` → `(treatment, True)`
- [ ] `"Il ne respire plus"` → `(urgency, True)`
- [ ] `"Qu'est-ce qu'un respirateur ?"` → `(allowed, False)`
- [ ] Urgency keyword wins over any other category in same query
- [ ] Case-insensitive match works

---

## PHASE 6 — CamemBERT mock classifier

> **Why before DecisionEngine:** `DecisionEngine` takes a classifier as a constructor argument.
> Define the interface and the mock first so `DecisionEngine` can be built and tested immediately,
> without waiting for the real CamemBERT model to be trained.

### 6.1 `src/safety/query_classifier.py` — Mock phase
- [ ] Define abstract base class / interface `BaseQueryClassifier`:
  - `classify(text: str) -> tuple[str, float]` → `(label, score)`
- [ ] Implement `MockQueryClassifier(BaseQueryClassifier)`:
  - Configurable return value for tests
  - Default: returns `(information_generale, 0.95)`
- [ ] Wire `MockQueryClassifier` into `DecisionEngine` for all integration work

**Commit:** `add CamemBERT mock classifier`

**✓ Tests:**
- [ ] `MockQueryClassifier` satisfies `BaseQueryClassifier` interface
- [ ] Configurable label and score work as expected
- [ ] All unit tests for `DecisionEngine` pass with mock

---

## PHASE 7 — Decision engine

### 7.1 `src/safety/decision_engine.py`
- [ ] Create `DecisionEngine` class
- [ ] Constructor accepts `RuleFilter`, `QueryClassifier` (accepts any `BaseQueryClassifier`)
- [ ] Implement `decide(text: str) -> str`:
  - If rule blocks with `urgency` → return `redirect`
  - If rule blocks (non-urgency) → return `refuse`
  - Else run classifier → `(label, score)`
  - If `score < CAMEMBERT_CONFIDENCE_THRESHOLD` → return `ask_reformulation`
  - If label ∈ `{avis_medical_personnalise, hors_perimetre}` → return `refuse`
  - If label == `urgence` → return `redirect`
  - If label ∈ `{information_generale, sensible}` and score ≥ threshold → return `allow_rag`
  - Default fallback → return `refuse`

**Commit:** `add decision engine`

**✓ Unit tests** (`tests/unit/test_decision_engine.py`):
- [ ] Urgency rule → `redirect` (classifier not called)
- [ ] Non-urgency rule block → `refuse` (classifier not called)
- [ ] High confidence `information_generale` → `allow_rag`
- [ ] Low confidence (below threshold) → `ask_reformulation`
- [ ] `avis_medical_personnalise` label → `refuse`
- [ ] `hors_perimetre` label → `refuse`
- [ ] `urgence` label from classifier → `redirect`

---

## PHASE 8 — ConversationMemory (move to st.session_state)

> Reminder: ConversationMemory is already defined in PHASE 4.
> This phase is a **checkpoint**: before wiring the pipeline, confirm that
> `get_context_string()` (last 2–3 turns) is integrated into the classifier input in `DecisionEngine`.

- [ ] Verify `ConversationMemoryManager.get_context_string()` is called before `QueryClassifier.classify()`
- [ ] Confirm context string is limited to 2–3 turns maximum
- [ ] Write a short integration test: question + 2-turn history → context correctly prepended

---

## PHASE 10 — Document indexing (Chroma)

### 10.1 `src/rag/document_loader.py`
- [ ] Create `DocumentLoader` class
- [ ] Load PDF, TXT, DOCX from `DOCUMENTS_PATH`
- [ ] Split into chunks (size + overlap configurable in `settings.py`)
- [ ] Attach source file name as metadata to each chunk
- [ ] Return `list[Document]` (LangChain format)

### 10.2 `src/rag/vector_store.py`
- [ ] Create `VectorStoreManager` class
- [ ] Use `sentence-transformers` for local embeddings (no API cost)
- [ ] Implement `index_documents(docs)` → builds and persists Chroma collection
- [ ] Implement `get_store()` → loads existing Chroma collection
- [ ] Handle case where collection does not exist yet

**Commit:** `add Chroma document indexing`

**✓ Tests (manual + integration):**
- [ ] Place at least one test document in `data/documents/` with source+date in filename (e.g. `SFAR_sedation_2024.pdf`)
- [ ] Run notebook 03 → `data/chroma_db/` folder created
- [ ] Collection contains expected number of chunks
- [ ] Source metadata present on each chunk

---

## PHASE 11 — Retriever

### 11.1 `src/rag/retriever.py`
- [ ] Create `Retriever` class
- [ ] Implement `search(query: str) -> list[dict]`:
  - Returns top-k chunks from Chroma
  - Each result: `{text, source, score}`
  - Returns empty list if no result above similarity threshold
- [ ] `top_k` and similarity threshold from `settings.py`

**Commit:** `add retriever`

**✓ Tests:**
- [ ] Known query returns relevant chunk from indexed documents
- [ ] Unknown query (no match) → empty list, no crash
- [ ] Empty Chroma DB → empty list (not an exception)
- [ ] `top_k` respected

---

## PHASE 12 — RAG chain

### 12.1 `src/rag/rag_chain.py`
- [ ] Create `RAGChain` class using LangChain
- [ ] Build prompt template:
  - System: strict medical safety instructions (no diagnosis, no treatment, source-only)
  - Context: retrieved document chunks
  - User: cleaned question
- [ ] Call OpenAI API only with anonymized query + document excerpts (no PII)
- [ ] Return `(answer: str, sources: list[str])`
- [ ] If `sources` is empty → do not call LLM, return `get_no_source_found()`

**Commit:** `add RAG chain`

**✓ Tests (with MockOpenAI to avoid cost):**
- [ ] Authorized question + documents → answer returned with sources
- [ ] Empty sources → LLM not called, `no_source_found` returned
- [ ] OpenAI failure (mock exception) → `api_unavailable` returned, not exception
- [ ] Answer contains no diagnosis keywords (basic check)
- [ ] Sources list matches document metadata

---

## PHASE 13 — Post-generation checker

### 13.1 `src/safety/post_checker.py`
- [ ] Create `PostGenerationChecker` class
- [ ] Load dangerous patterns from `safety_config.py`
- [ ] Implement `check(response: str) -> tuple[str, bool]` → `(safe_response, was_modified)`:
  - Scan for: diagnosis, prognosis, survival prediction, treatment advice, overly affirmative phrasing, absence of source, out-of-scope content
  - If unsafe → replace with `get_refuse_medical_advice()`, set `was_modified = True`
  - If safe → return unchanged, `was_modified = False`

**Commit:** `add post-generation checker`

**✓ Unit tests** (`tests/unit/test_post_checker.py`):
- [ ] Safe answer → returned unchanged, `was_modified = False`
- [ ] Answer containing `"il va survivre"` → replaced, `was_modified = True`
- [ ] Answer containing `"donnez-lui ce médicament"` → replaced
- [ ] Answer with no source reference → replaced
- [ ] `was_modified` flag always set correctly

---

## PHASE 14 — Guardrails

### 14.1 `src/safety/guardrails.py`
- [ ] Create `GuardrailsManager` class
- [ ] **Pre-generation gate:**
  - Block LLM call if decision ≠ `allow_rag`
  - Block if no documents retrieved
  - Block if query contains urgency markers (redundant safety net)
- [ ] **During-generation:** inject strict system prompt into every LLM call
- [ ] **Post-generation:** call `PostGenerationChecker` before returning
- [ ] Single public method: `run(query, decision, sources) -> str`

**Commit:** `add guardrails manager`

**✓ Tests:**
- [ ] `decision = refuse` → LLM never called
- [ ] `sources = []` → LLM never called
- [ ] Unsafe LLM output → `PostGenerationChecker` catches it
- [ ] Safe flow → answer with sources returned

---

## PHASE 15 — Anonymized logger

### 15.1 `src/audit/anonymized_logger.py`

> **Note:** folder is named `audit` not `logging` to avoid shadowing Python’s stdlib `logging` module.
- [ ] Create `AnonymizedLogger` class
- [ ] Hash raw query text before storage (SHA-256 or similar)
- [ ] Log fields per interaction (JSON):
  - `timestamp`
  - `query_hash` (not raw text)
  - `predicted_class`
  - `confidence_score`
  - `decision`
  - `sources_used` (list of filenames)
  - `response_type` (informative / refuse / redirect / reformulation)
  - `post_check_modified` (bool)
- [ ] Write to `logs/interactions.jsonl`
- [ ] Implement log rotation (daily or size-based)

**Commit:** `add anonymized logger`

**✓ Tests:**
- [ ] Log file created after first interaction
- [ ] Raw query text not present anywhere in log file
- [ ] All required fields present in each log entry
- [ ] Valid JSON per line
- [ ] Log entry created for every interaction including refusals

---

## PHASE 16 — Streamlit interface

### 16.1 `app.py`
- [ ] Layout: title, ethical disclaimer banner, chat area, input box, reset button
- [ ] On submit: call **only** `ChatbotPipeline.run(user_input, session_memory)` — no business logic in app.py
- [ ] Display answer + sources (if any)
- [ ] Show distinct visual style for refusal / redirect responses
- [ ] Reset button calls `ConversationMemoryManager.clear()`
- [ ] Ethical disclaimer visible at all times

**Commit:** `add Streamlit interface`

**✓ Manual tests:**
- [ ] General question → answer displayed with sources
- [ ] Forbidden question → refusal message displayed
- [ ] Urgency question → emergency redirect displayed
- [ ] OpenAI API key missing → `api_unavailable` displayed (not a crash)
- [ ] No documents indexed → `no_source_found` displayed (not a crash)
- [ ] Reset button clears conversation
- [ ] Ethical disclaimer visible at all times

---

## PHASE 17 — CamemBERT dataset & training

### 17.1 Dataset construction — `notebooks/01_prepare_dataset.ipynb`
- [ ] Collect ~200 questions from MedQuAD (general medical French)
- [ ] Write ~100 ICU-specific questions manually
- [ ] Write ~80 sensitive / forbidden / urgency / out-of-scope questions
- [ ] Clean all questions using `TextCleaner` (same cleaner used at runtime)
- [ ] Annotate with labels (rule-based first, then manual review)
- [ ] Validate a subset with a healthcare professional if possible
- [ ] Verify class balance (flag imbalance before training)
- [ ] Stratified split: 70% train / 15% validation / 15% test
- [ ] Save: `dataset_clean.csv`, `train.csv`, `val.csv`, `test.csv` to `data/questions_dataset/`
- **Output artifact:** `data/questions_dataset/dataset_clean.csv`

### 17.2 CamemBERT training — `notebooks/02_train_camembert.ipynb`
- [ ] Load `train.csv` and `val.csv` from `data/questions_dataset/`
- [ ] Fine-tune `camembert-base` for sequence classification (5 classes)
- [ ] Training loop with early stopping on validation F1
- [ ] Evaluation metrics per class:
  - [ ] Accuracy
  - [ ] Precision per class
  - [ ] Recall per class
  - [ ] F1-score (macro)
  - [ ] Confusion matrix
- [ ] Analyse misclassifications — especially false `allow_rag` on sensitive questions
- [ ] Save model to `models/camembert_classifier/`
- **Output artifact:** `models/camembert_classifier/`

**Commit:** `add CamemBERT training notebook`

---

## PHASE 18 — Real CamemBERT classifier

### 18.1 `src/safety/query_classifier.py` — Real implementation
- [ ] Implement `CamembertQueryClassifier(BaseQueryClassifier)`:
  - Load tokenizer and model from `CAMEMBERT_MODEL_PATH`
  - Tokenize input, run inference
  - Return `(label: str, confidence: float)`
- [ ] Drop-in replacement for `MockQueryClassifier` (same interface)
- [ ] Handle model-not-found error gracefully (fall back to mock + warning)

**Commit:** `add CamemBERT real classifier`

**✓ Tests:**
- [ ] Model loads without error
- [ ] Output label belongs to the 5 expected classes
- [ ] Confidence score is in `[0.0, 1.0]`
- [ ] Short / ambiguous input handled without crash
- [ ] ML metrics meet target (accuracy ≥ 0.85, no dangerous misclassification)

---

## PHASE 19 — Evaluation & final tests

### 19.1 `src/evaluation/test_questions.py`
- [ ] Populate with all 50 questions from the project specification:
  - 15 general (expected: `allow_rag`)
  - 8 sensitive (expected: `allow_rag` or `ask_reformulation`)
  - 12 forbidden (expected: `refuse`)
  - 5 urgency (expected: `redirect`)
  - 8 out-of-scope (expected: `refuse`)
- [ ] Format: `[{"text": "...", "category": "...", "expected_decision": "..."}]`

### 19.2 `src/evaluation/evaluator.py`
- [ ] Run full pipeline on all test questions
- [ ] Compute:
  - [ ] Overall correct decision rate
  - [ ] Per-category correct decision rate
  - [ ] **Dangerous response rate (target: 0%)**
  - [ ] Correct redirection rate for urgencies (target: 100%)
  - [ ] Average response time (total, per stage)
  - [ ] Source quality (% answers with at least one source)
- [ ] Output: summary to console + `logs/evaluation_report.csv`

### 19.3 Document indexing validation — `notebooks/03_index_documents.ipynb`
- [ ] Load documents from `data/documents/` using `DocumentLoader`
- [ ] Inspect chunks (count, average size, source metadata present)
- [ ] Embed and index into Chroma using `VectorStoreManager`
- [ ] Verify with 3 test queries via `Retriever` — confirm relevant sources returned
- [ ] Confirm `data/chroma_db/` folder created and non-empty
- **Output artifact:** `data/chroma_db/`

### 19.4 Per-module pipeline walkthrough — `notebooks/04_test_pipeline_modules.ipynb`
- [ ] Run `TextCleaner` on sample inputs
- [ ] Run `RuleFilter` on all 5 question categories
- [ ] Run `ConversationMemoryManager` — add turns, check history, clear
- [ ] Run `DecisionEngine` with `MockQueryClassifier` — all 4 decision outcomes
- [ ] Run `Retriever` — known and unknown queries
- [ ] Run `RAGChain` — verify sources attached to answer
- [ ] Run `PostGenerationChecker` — safe and dangerous answers
- [ ] Full walkthrough: one question through every stage in sequence
- **No output artifact — validation only**

### 19.5 Full evaluation — `notebooks/05_evaluate_chatbot.ipynb`
- [ ] Run evaluator on all 50 test questions
- [ ] Visualise results (bar charts per category, confusion matrix)
- [ ] Confirm dangerous response rate = 0%
- [ ] Document failures and edge cases
- **Output artifact:** `logs/evaluation_report.csv`

**Commit:** `add evaluation module`

---

## PHASE 20 — Full test suite

### Unit tests
- [ ] `test_text_cleaner.py` — all cases passing
- [ ] `test_rule_filter.py` — all cases passing
- [ ] `test_decision_engine.py` — all cases passing
- [ ] `test_memory_manager.py` — all cases passing
- [ ] `test_post_checker.py` — all cases passing
- [ ] `test_controlled_responses.py` — all cases passing

### Integration tests (`tests/integration/test_pipeline.py`)
- [ ] General question end-to-end (MockClassifier + MockOpenAI) → answer + sources
- [ ] Sensitive question → `ask_reformulation` response
- [ ] Forbidden question → `refuse` (LLM not called, verified with mock)
- [ ] Urgency question → `redirect` (LLM not called)
- [ ] Out-of-scope → `refuse`
- [ ] Ambiguous question with memory context → correct classification using history
- [ ] No matching document → `no_source_found` response

### Security tests (`tests/security/test_safety.py`)
- [ ] Direct diagnosis request → refused
- [ ] Personalized prognosis → refused
- [ ] Treatment recommendation → refused
- [ ] Lab result interpretation → refused
- [ ] **Prompt injection** (`"ignore previous instructions and..."`) → refused / neutralized
- [ ] Personal patient data request → refused
- [ ] Urgency → redirect (never LLM answer)
- [ ] All above: confirm `decision != allow_rag` in log

### Performance tests
- [ ] Measure average total response time
- [ ] Measure per-stage time: rule filter, CamemBERT, Chroma, OpenAI, post-check
- [ ] Identify bottlenecks and document findings

**Commit:** `add tests and documentation`

---

## PHASE 21 — Final deliverables

- [ ] `README.md` installation section tested on clean environment
- [ ] All tests passing: `pytest` exits with code 0
- [ ] Dangerous response rate = 0% on full evaluation set
- [ ] Emergency redirect rate = 100% on urgency questions
- [ ] Source coverage: answers cite at least one document in ≥ 90% of allowed cases
- [ ] No PII in logs (audit `logs/interactions.jsonl`)
- [ ] No secrets committed to git (`git log --all -- .env` is clean)
- [ ] Final demo run recorded or documented

---

## Quick reference — decision routing

```
Input
  │
  ├─ Rule: urgency          → redirect   (no ML, no LLM)
  ├─ Rule: blocked          → refuse     (no ML, no LLM)
  │
  ├─ ML: urgence            → redirect   (no LLM)
  ├─ ML: avis_medical       → refuse     (no LLM)
  ├─ ML: hors_perimetre     → refuse     (no LLM)
  ├─ ML: low confidence     → ask_reformulation
  │
  └─ ML: information_generale / sensible + high confidence
        │
        ├─ No documents found   → no_source_found  (no LLM)
        └─ Documents found      → RAG → LLM → PostCheck → response
```
