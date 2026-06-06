# Integration tests — full pipeline end-to-end
# Uses MockClassifier and MockOpenAI to avoid real API calls
#
# Scenarios to cover:
#   - General question → allow_rag → answer with sources
#   - Sensitive question → ask_reformulation
#   - Forbidden question → refuse (no LLM call)
#   - Urgency → redirect (no LLM call)
#   - Out-of-scope → refuse (no LLM call)
#   - Ambiguous question + conversation memory context
#   - Question with no matching document → controlled response
