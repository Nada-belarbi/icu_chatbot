# Troubleshooting

| Symptom | Probable cause | Solution |
|---|---|---|
| `EnvironmentError: Required environment variable 'OPENAI_API_KEY' is not set` | `.env` missing or key empty | `copy .env.example .env` and fill in a real key |
| OpenAI API call times out / errors | Network issue, invalid key, quota exceeded | Check the key and account quota; the pipeline returns a controlled `api_unavailable` message rather than crashing |
| Classifier missing / fails to load, or errors mentioning a tiny/invalid `model.safetensors` | Repo cloned without Git LFS installed, so `models/camembert_classifier/model.safetensors` is a small text pointer instead of the real ~420MB weights | Install [Git LFS](https://git-lfs.com/), then run `git lfs pull` in the repo |
| Chroma collection missing | `data/rag_documents/chroma_db/` not built yet | `python scripts/index_rag_documents.py` |
| Every answer is "no source found" | Chroma index empty, or unreadable by the installed `chromadb` version (a real, previously-observed failure mode: a `chromadb`/`langchain-chroma` version mismatch causes a silent panic during load) | Re-run `python scripts/index_rag_documents.py`; confirm `chromadb==1.0.15` / `langchain-chroma==0.2.5` (pinned in `requirements.txt`) are actually installed — `pip show chromadb langchain-chroma` |
| Empty retrieval results for a clearly in-corpus topic | Query phrasing mismatch, or the index needs rebuilding after a corpus change | Re-run `python scripts/index_rag_documents.py`; check `data/rag_documents/sources.csv` for the topic's approval status |
| Document parsing failed during indexing | Corrupt/empty source file, or unsupported format | `python scripts/validate_corpus.py` reports near-empty/missing files by `source_id` |
| Login / database error | `data/auth/` missing or corrupted | Check `data/auth/users.json` exists and is valid JSON; see `src/auth/local_auth.py` |
| Streamlit port conflict | Another process already listening on 8501 | `Get-NetTCPConnection -LocalPort 8501 -State Listen` (Windows) then stop the conflicting process, or run Streamlit with `--server.port` |
| Scripts hang for 30+ seconds with no internet | `sentence-transformers`/`transformers` trying a HuggingFace Hub connectivity check | Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` (already done in `scripts/run_smoke_tests.py`, `scripts/pre_deploy_check.py`, and the `Dockerfile`) |
| Slow first response (~60-90s) after starting the app | CamemBERT + embedding model cold start (loaded once per process) | Expected; subsequent requests are fast |
