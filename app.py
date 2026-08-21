"""Application composition root for the ICU information assistant.

The Streamlit presentation layer is in ``src.frontend``. Authentication and
encrypted conversation persistence are deliberately separate from the RAG
pipeline so changes to the interface cannot change medical safety behavior.
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from src.auth.local_auth import LocalAuthService
from src.frontend.ui import initialise_session, render_authentication, render_chat
from src.memory.conversation_repository import ConversationRepository
from src.pipeline.chatbot_pipeline import ChatbotPipeline

_log = logging.getLogger(__name__)
_PROJECT_ROOT = Path(__file__).resolve().parent


@st.cache_resource(show_spinner="Chargement sécurisé de l'assistant…")
def load_pipeline() -> ChatbotPipeline:
    """Wire the existing safety/RAG pipeline without modifying its behavior."""
    from src.audit.anonymized_logger import AnonymizedLogger
    from src.safety.post_checker import PostGenerationChecker
    from src.safety.query_classifier import HybridQueryClassifier

    retriever = None
    rag_chain = None
    try:
        from src.rag.retriever import Retriever
        from src.rag.vector_store import VectorStoreManager

        retriever = Retriever(vector_store_manager=VectorStoreManager())
        _log.info("Retriever wired successfully.")
    except FileNotFoundError:
        _log.warning("Chroma DB not found — retriever not wired.")
    except BaseException as exc:
        _log.error("Failed to wire retriever: %s", exc)

    try:
        from src.rag.rag_chain import RAGChain

        rag_chain = RAGChain()
        _log.info("RAGChain wired successfully.")
    except Exception as exc:
        _log.error("Failed to wire RAGChain: %s", exc)

    return ChatbotPipeline(
        classifier=HybridQueryClassifier(),
        retriever=retriever,
        rag_chain=rag_chain,
        post_checker=PostGenerationChecker(),
        logger=AnonymizedLogger(),
    )


@st.cache_resource
def services() -> tuple[LocalAuthService, ConversationRepository]:
    """Create long-lived local services; conversation records are encrypted at rest."""
    auth = LocalAuthService(_PROJECT_ROOT / "data" / "auth" / "users.json")
    repository = ConversationRepository(
        root_path=_PROJECT_ROOT / "data" / "private_conversations",
        key_path=_PROJECT_ROOT / "data" / ".chat_storage.key",
    )
    return auth, repository


def main() -> None:
    st.set_page_config(
        page_title="Assistant ICU — Réanimation",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialise_session()
    auth_service, conversation_repository = services()

    if st.session_state.authenticated:
        render_chat(load_pipeline(), conversation_repository)
    else:
        render_authentication(auth_service, conversation_repository)


if __name__ == "__main__":
    main()
