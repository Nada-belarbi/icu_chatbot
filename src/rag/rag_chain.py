# RAGChain
#
# Responsibilities:
#   - Build the LangChain chain: retrieved docs → prompt → OpenAI LLM → response
#   - Inject guardrail system prompt (no diagnosis, no treatment, source-only answers)
#   - Attach source references to the returned response
#   - Only called after DecisionEngine returns allow_rag
#   - Output: (answer: str, sources: list[str])
#
# Security note: only send anonymized query + document excerpts to OpenAI — never PII
