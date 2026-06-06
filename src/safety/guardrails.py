# GuardrailsManager  (LangChain-based)
#
# Three levels:
#   1. PRE-generation  — block LLM call if decision != allow_rag or no sources found
#   2. DURING generation — inject strict system prompt:
#       - answer only from retrieved documents
#       - no individualized medical advice
#       - no diagnosis / treatment / prognosis
#       - redirect to medical team when uncertain
#   3. POST-generation — delegate to PostGenerationChecker before returning
