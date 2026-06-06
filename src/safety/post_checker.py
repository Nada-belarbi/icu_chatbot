# PostGenerationChecker
#
# Responsibilities:
#   - Scan the LLM response BEFORE it is shown to the user
#   - Detect and reject responses that contain:
#       - Personalized medical advice
#       - Diagnosis or prognosis statements
#       - Survival predictions or duration estimates
#       - Treatment recommendations
#       - Overly affirmative phrasing
#       - No documentary source
#       - Out-of-scope content
#   - If unsafe: replace with a controlled refusal/redirection message
#   - Returns: (safe_response: str, was_modified: bool)
