# Application entry point — Streamlit UI
# Run: streamlit run app.py

from __future__ import annotations

import logging

import streamlit as st

from config.settings import MEMORY_MAX_TURNS
from src.memory.conversation_memory import ConversationMemoryManager
from src.pipeline.chatbot_pipeline import (
    ChatbotPipeline,
    RTYPE_REFUSE,
    RTYPE_REDIRECT,
    RTYPE_ERROR,
)

_log = logging.getLogger(__name__)

_DISCLAIMER = (
    "⚠️ Cet outil fournit uniquement des informations générales sur la réanimation. "
    "Il ne remplace en aucun cas l'avis de l'équipe médicale. "
    "En cas d'urgence, appelez le **15 (SAMU)**."
)


# ---------------------------------------------------------------------------
# Pipeline — loaded once, cached for the Streamlit session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Chargement du chatbot…")
def _load_pipeline() -> ChatbotPipeline:
    """
    Wire all real modules and return a ready-to-use ``ChatbotPipeline``.

    Gracefully degrades if Chroma or OpenAI are not yet available:
    - No documents indexed → ``retriever = None`` → ``no_source_found`` response
    - OpenAI not configured → ``rag_chain = None`` → ``api_unavailable`` response
    """
    from src.safety.post_checker import PostGenerationChecker
    from src.audit.anonymized_logger import AnonymizedLogger

    retriever = None
    rag_chain = None

    try:
        from src.rag.vector_store import VectorStoreManager
        from src.rag.retriever import Retriever

        vsm = VectorStoreManager()
        retriever = Retriever(vector_store_manager=vsm)
        _log.info("Retriever wired successfully.")
    except FileNotFoundError:
        _log.warning("Chroma DB not found — retriever not wired. Run the indexing notebook.")
    except Exception as exc:
        _log.error("Failed to wire retriever: %s", exc)

    try:
        from src.rag.rag_chain import RAGChain

        rag_chain = RAGChain()
        _log.info("RAGChain wired successfully.")
    except Exception as exc:
        _log.error("Failed to wire RAGChain: %s", exc)

    post_checker = PostGenerationChecker()
    logger = AnonymizedLogger()

    return ChatbotPipeline(
        retriever=retriever,
        rag_chain=rag_chain,
        post_checker=post_checker,
        logger=logger,
    )


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Assistant ICU",
        page_icon="🏥",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    st.title("🏥 Assistant d'information — Réanimation (ICU)")

    # Permanent ethical disclaimer
    st.warning(_DISCLAIMER)
    st.divider()

    # --- Session-state initialisation ---
    if "history" not in st.session_state:
        st.session_state.history: list[dict] = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemoryManager(
            history=st.session_state.history,
            max_turns=MEMORY_MAX_TURNS,
        )

    pipeline = _load_pipeline()

    # --- Replay conversation history ---
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- User input ---
    user_input: str | None = st.chat_input("Posez votre question ici…")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        result = pipeline.run(user_input, memory=st.session_state.memory)

        with st.chat_message("assistant"):
            if result.response_type in (RTYPE_REFUSE, RTYPE_ERROR):
                st.error(result.answer)
            elif result.response_type == RTYPE_REDIRECT:
                st.warning(result.answer)
            else:
                st.markdown(result.answer)
                if result.sources:
                    st.caption("📄 Sources : " + " · ".join(result.sources))

    # --- Reset ---
    st.divider()
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Réinitialiser", type="secondary", use_container_width=True):
            st.session_state.memory.clear()
            st.session_state.history.clear()
            st.rerun()

    # Footer disclaimer
    st.caption(_DISCLAIMER)


if __name__ == "__main__":
    main()
