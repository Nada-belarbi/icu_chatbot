# ChatbotPipeline — central orchestrator
#
# Single entry point called by app.py for every user message.
# Calls each module in the correct order and handles all fallback cases.
#
# Call order:
#   1. TextCleaner.clean()
#   2. ConversationMemoryManager.get_context_string()   (last 2-3 turns only)
#   3. RuleFilter.filter()
#   4. QueryClassifier.classify()                       (mock or real CamemBERT)
#   5. DecisionEngine.decide()
#   6. Route by decision:
#        allow_rag → Retriever.search() → RAGChain.generate() → PostGenerationChecker.check()
#        refuse / redirect / ask_reformulation → ControlledResponseManager
#   7. AnonymizedLogger.log()
#   8. ConversationMemoryManager.add_*()
#
# Fallback cases (must never raise an exception to the UI):
#   - TextCleaner raises ValueError (empty input)  → return ask_reformulation response
#   - Retriever returns empty list (no docs found) → return no_source_found response
#   - OpenAI API call fails                        → return api_unavailable response
#   - PostGenerationChecker rejects the answer     → return refuse_medical_advice response
#   - Any unexpected exception                     → return generic_error response + log
#
# Interface:
#   pipeline = ChatbotPipeline()
#   response, sources, decision = pipeline.run(user_input, session_memory)
