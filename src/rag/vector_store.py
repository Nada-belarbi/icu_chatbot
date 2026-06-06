# VectorStoreManager
#
# Responsibilities:
#   - Create or load a persistent Chroma collection from data/chroma_db/
#   - Embed documents using a sentence-transformers model (local, no API cost)
#   - Expose methods: index_documents(), get_store()
