# Retriever
#
# Responsibilities:
#   - Query the Chroma store and return the top-k most relevant document chunks
#   - top_k read from config/settings.py
#   - Return list of (document_chunk: str, source: str, score: float)
#   - If no relevant document found above a similarity threshold → signal absence
