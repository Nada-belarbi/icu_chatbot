# ConversationMemoryManager
#
# Responsibilities:
#   - Store recent user/assistant exchanges for the current session only
#   - Limit history to the last N turns (configurable in settings.py)
#   - Expose method to clear memory at session end
#   - Never persist data between sessions
#   - Never store personally identifiable information
