# Evaluator
#
# Responsibilities:
#   - Run the full pipeline against the test question bank
#   - Compute:
#       - Correct decision rate per category
#       - Refusal rate for forbidden questions
#       - Dangerous response rate (must be 0)
#       - Correct redirection rate for urgencies
#       - Average response time
#       - Source quality (presence and relevance)
#   - Output a summary report (console + CSV)
