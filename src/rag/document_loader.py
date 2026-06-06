# DocumentLoader
#
# Responsibilities:
#   - Load documents from data/documents/ (PDF, TXT, DOCX)
#   - Split into chunks (size and overlap configurable in settings.py)
#   - Return list of LangChain Document objects with source metadata
#   - Must be run separately from the runtime (index once, reuse Chroma)
