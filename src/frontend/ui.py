"""Streamlit views and state orchestration for the chat product.

Business logic stays in ``src.pipeline``. This module only renders the UI,
manages session-local view state, and delegates encrypted persistence to the
conversation repository. Visual assets (CSS, SVG icons) and text formatting
helpers are intentionally kept in separate modules of this package.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import streamlit as st

from config.settings import MEMORY_MAX_TURNS
from src.memory.conversation_memory import ConversationMemoryManager
from src.pipeline.chatbot_pipeline import (
    RTYPE_ERROR,
    RTYPE_INFORMATIVE,
    RTYPE_REDIRECT,
    RTYPE_REFORMULATION,
    RTYPE_REFUSE,
)

from src.frontend.formatting import content_to_html, humanize_source, title_from_question
from src.frontend.icons import (
    ASSISTANT_AVATAR_SVG,
    BRAND_MARK_SVG,
    EMERGENCY_ICON_SVG,
    INFO_ICON_SVG,
    LOCK_ICON_SVG,
    WARNING_ICON_SVG,
)
from src.frontend.styles import apply_styles

if TYPE_CHECKING:
    from src.auth.local_auth import LocalAuthService
    from src.memory.conversation_repository import ConversationRepository

_NEUTRAL_RESPONSE_TYPES = (RTYPE_REFUSE, RTYPE_ERROR, RTYPE_REFORMULATION)

DISCLAIMER = (
    "Cet outil fournit uniquement des informations générales sur la réanimation. "
    "Il ne remplace jamais l'avis de l'équipe médicale. En cas d'urgence, prévenez "
    "immédiatement l'équipe soignante ou composez le 15."
)

PRIVACY_NOTICE = (
    "Ne saisissez ni nom, ni date de naissance, ni numéro de dossier, ni information "
    "permettant d'identifier un patient. Les conversations sont chiffrées sur cet appareil "
    "et vous pouvez les supprimer à tout moment."
)

# (title, actual question sent to the pipeline) — two-line suggestion cards.
QUICK_SUGGESTIONS = (
    ("Comprendre l'intubation", "À quoi sert un respirateur ?"),
    ("Comprendre le séjour", "Comment se déroule une journée en réanimation ?"),
    ("Après la réanimation", "Que se passe-t-il après la sortie ?"),
    ("Comprendre les équipements", "Pourquoi les machines de surveillance font-elles des alarmes ?"),
)


def initialise_session() -> None:
    """Set up session-local state; no chat data is kept globally in Streamlit."""
    defaults: dict[str, Any] = {
        "authenticated": False,
        "username": None,
        "display_name": None,
        "active_conversation": None,
        "conversation_list": [],
        "quick_question": None,
        "confirm_delete_conversation": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("memory_history", [])
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemoryManager(
            history=st.session_state.memory_history,
            max_turns=MEMORY_MAX_TURNS,
        )


def render_authentication(auth_service: "LocalAuthService", repository: "ConversationRepository") -> None:
    """Render account access and explicit privacy consent before the chat UI."""
    apply_styles()
    _, column, _ = st.columns((1, 1.3, 1))
    with column:
        with st.container(border=True):
            st.markdown(
                f"<div class='auth-brand'><div class='app-header-mark' style='margin:0 auto;'>{BRAND_MARK_SVG}</div>"
                "<h1>Bienvenue</h1><p>Connectez-vous pour retrouver vos conversations.</p></div>",
                unsafe_allow_html=True,
            )
            login_tab, register_tab = st.tabs(("Se connecter", "Créer un compte"))

            with login_tab:
                with st.form("login"):
                    identifier = st.text_input("Adresse e-mail", autocomplete="username")
                    password = st.text_input("Mot de passe", type="password", autocomplete="current-password")
                    submitted = st.form_submit_button("Se connecter", type="primary", use_container_width=True)
                st.caption("Mot de passe oublié ? Fonctionnalité à venir.")
                if submitted:
                    result = auth_service.authenticate(identifier, password)
                    if result.success:
                        display_name = auth_service.get_display_name(identifier.strip())
                        _open_user_session(identifier.strip(), display_name, repository)
                        st.rerun()
                    st.error(result.message)

            with register_tab:
                with st.form("register"):
                    first_name = st.text_input("Prénom")
                    identifier = st.text_input("Adresse e-mail", autocomplete="username", key="register_identifier")
                    password = st.text_input(
                        "Mot de passe", type="password", autocomplete="new-password", key="register_password"
                    )
                    confirmation = st.text_input(
                        "Confirmer le mot de passe", type="password", autocomplete="new-password"
                    )
                    consent = st.checkbox("J'accepte la politique de confidentialité ci-dessous.")
                    st.caption("Minimum 12 caractères. Ne réutilisez pas un mot de passe sensible.")
                    submitted = st.form_submit_button("Créer un compte", type="primary", use_container_width=True)
                if submitted:
                    result = auth_service.register(identifier, password, confirmation, consent, first_name)
                    if result.success:
                        st.success("Compte créé. Vous pouvez maintenant vous connecter.")
                    else:
                        st.error(result.message)

            st.markdown(
                f"<div class='disclaimer-card'>{WARNING_ICON_SVG}<div><strong>Confidentialité.</strong> "
                f"{PRIVACY_NOTICE}</div></div>",
                unsafe_allow_html=True,
            )


def render_chat(pipeline: Any, repository: "ConversationRepository") -> None:
    """Render the full authenticated application and leave pipeline semantics unchanged."""
    apply_styles()
    username = st.session_state.username
    if not username:
        return

    conversation = st.session_state.active_conversation
    if conversation is None:
        _create_conversation(repository, username)
        conversation = st.session_state.active_conversation

    _render_sidebar(repository, username)
    _render_header()

    st.markdown(
        f"<div class='disclaimer-card'>{WARNING_ICON_SVG}<div>{DISCLAIMER}</div></div>",
        unsafe_allow_html=True,
    )

    messages = conversation["messages"]
    if not messages:
        _render_empty_state()

    for message in messages:
        _render_message(message)

    st.markdown(
        f"<div class='privacy-note'>{LOCK_ICON_SVG}<span>Ne partagez pas de données permettant d'identifier un patient.</span></div>",
        unsafe_allow_html=True,
    )

    question_from_button = st.session_state.quick_question
    st.session_state.quick_question = None
    question = st.chat_input("Posez votre question en langage clair…") or question_from_button
    if question:
        _answer(question, pipeline, repository, username)


def _render_header() -> None:
    identity, account = st.columns((10, 1))
    with identity:
        st.markdown(
            f"""
            <div class="app-header">
              <div class="app-header-mark">{BRAND_MARK_SVG}</div>
              <div>
                <p class="app-header-title">Assistant d'information — Réanimation</p>
                <p class="app-header-subtitle">Des informations générales pour mieux comprendre la réanimation.</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    display_name = st.session_state.display_name or st.session_state.username or ""
    initial = display_name.strip()[:1].upper() or "?"
    with account:
        with st.popover(initial, help="Mon compte"):
            st.markdown(f"Connecté en tant que **{display_name}**")
            if st.button("Se déconnecter", key="logout_popover", use_container_width=True):
                _logout()
                st.rerun()


def _render_empty_state() -> None:
    st.markdown(
        "<div class='empty-state'><h2>Bonjour 👋</h2>"
        "<p>Comment puis-je vous aider aujourd'hui ?</p></div>",
        unsafe_allow_html=True,
    )
    top_row = st.columns(2)
    bottom_row = st.columns(2)
    for index, (title, question) in enumerate(QUICK_SUGGESTIONS):
        column = top_row[index] if index < 2 else bottom_row[index - 2]
        with column:
            if st.button(f"{title}\n{question}", key=f"suggestion_{index}", use_container_width=True):
                st.session_state.quick_question = question
                st.rerun()


def _render_sidebar(repository: "ConversationRepository", username: str) -> None:
    active = st.session_state.active_conversation
    active_id = active["id"] if active else None
    if active_id:
        st.markdown(
            f"<style>[data-testid='stSidebar'] [data-testid='stButton-conversation_{active_id}'] > button {{"
            "background: #DFF2F1 !important; border-left: 3px solid #29B6A6 !important; "
            "color: #123B5D !important; font-weight: 600 !important;}}</style>",
            unsafe_allow_html=True,
        )

    with st.sidebar:
        st.markdown(
            f"<div class='brand-block'>{BRAND_MARK_SVG}<div class='brand-text'>"
            "<div class='brand-title'>Assistant d'information</div>"
            "<div class='brand-sub'>RÉANIMATION</div></div></div>",
            unsafe_allow_html=True,
        )
        if st.button("＋  Nouvelle conversation", key="new_conversation", use_container_width=True):
            _create_conversation(repository, username)
            st.rerun()

        st.markdown("<div class='sidebar-section-label'>Conversations récentes</div>", unsafe_allow_html=True)
        for item in st.session_state.conversation_list:
            if st.button(
                item["title"] or "Nouvelle conversation",
                key=f"conversation_{item['id']}",
                use_container_width=True,
            ):
                _select_conversation(repository, username, item["id"])
                st.rerun()
            st.markdown(
                f"<div class='conversation-date'>{_format_relative_date(item['updated_at'])}</div>",
                unsafe_allow_html=True,
            )

        if st.session_state.conversation_list:
            with st.expander("Gérer les conversations"):
                for item in st.session_state.conversation_list:
                    new_title = st.text_input(
                        "Renommer la conversation",
                        value=item["title"],
                        key=f"rename_input_{item['id']}",
                    )
                    actions = st.columns(2)
                    with actions[0]:
                        if st.button("Renommer", key=f"rename_btn_{item['id']}", use_container_width=True):
                            _rename_conversation(repository, username, item["id"], new_title)
                            st.rerun()
                    with actions[1]:
                        if st.button("Supprimer", key=f"delete_btn_{item['id']}", use_container_width=True):
                            repository.delete(username, item["id"])
                            if item["id"] == active_id:
                                _create_conversation(repository, username)
                            else:
                                st.session_state.conversation_list = repository.list_for_user(username)
                            st.rerun()

        st.markdown(
            "<div class='sidebar-footer'>"
            f"<div class='sidebar-footer-item'>{INFO_ICON_SVG}<span>À propos de l'assistant</span></div>"
            "<div class='sidebar-footer-item'>Ressources utiles</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("Se déconnecter", key="logout", use_container_width=True):
            _logout()
            st.rerun()


def _answer(question: str, pipeline: Any, repository: "ConversationRepository", username: str) -> None:
    """Call the unchanged pipeline, then persist both turns only after it returns."""
    now = datetime.now().strftime("%H:%M")
    _render_message({"role": "user", "content": question, "timestamp": now})

    # This is the original pipeline call and keeps its existing guardrails/RAG behavior.
    with st.spinner("Recherche dans les sources…"):
        result = pipeline.run(question, memory=st.session_state.memory)

    _render_message(
        {
            "role": "assistant",
            "content": result.answer,
            "timestamp": now,
            "sources": result.sources,
            "response_type": result.response_type,
        }
    )

    conversation = st.session_state.active_conversation
    conversation["messages"].extend((
        {"role": "user", "content": question, "timestamp": now},
        {
            "role": "assistant", "content": result.answer, "timestamp": now,
            "sources": result.sources, "response_type": result.response_type,
            "decision": result.decision,
        },
    ))
    if len(conversation["messages"]) == 2:
        conversation["title"] = title_from_question(question)
    conversation["updated_at"] = datetime.now().isoformat()
    repository.save(username, conversation)
    st.session_state.conversation_list = repository.list_for_user(username)
    st.rerun()


def _render_message(message: dict) -> None:
    timestamp = message.get("timestamp", "")
    if message["role"] == "user":
        content_html = content_to_html(message["content"])
        st.markdown(
            f"<div class='msg-row msg-row-user'><div class='bubble bubble-user'>{content_html}</div></div>"
            f"<div class='msg-meta msg-meta-user'>{timestamp}</div>",
            unsafe_allow_html=True,
        )
        return

    response_type = message.get("response_type", RTYPE_INFORMATIVE)
    content_html = content_to_html(message["content"])
    if response_type == RTYPE_REDIRECT:
        body = f"<div class='card-emergency'>{EMERGENCY_ICON_SVG}<div>{content_html}</div></div>"
    elif response_type in _NEUTRAL_RESPONSE_TYPES:
        body = f"<div class='card-neutral'>{INFO_ICON_SVG}<div>{content_html}</div></div>"
    else:
        body = f"<div class='bubble bubble-assistant'>{content_html}</div>"
        body += _sources_html(message.get("sources", []))

    st.markdown(
        "<div class='msg-row msg-row-assistant'>"
        f"<div class='msg-avatar'>{ASSISTANT_AVATAR_SVG}</div>"
        f"<div class='bubble-col'>{body}<div class='msg-meta'>{timestamp}</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _sources_html(sources: list[str]) -> str:
    if not sources:
        return ""
    seen: set[str] = set()
    cards = []
    for source in sources:
        org_label, title = humanize_source(source)
        dedupe_key = f"{org_label}:{title}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        cards.append(f"<span class='source-card'><span class='source-tag'>{org_label}</span>{title}</span>")
    if not cards:
        return ""
    visible, extra = cards[:3], cards[3:]
    html_out = f"<div class='sources-block'><div class='sources-label'>Sources utilisées</div>{''.join(visible)}"
    if extra:
        html_out += (
            "<details class='sources-more'><summary>Voir toutes les sources</summary>"
            f"<div class='sources-more-list'>{''.join(extra)}</div></details>"
        )
    html_out += "</div>"
    return html_out


def _open_user_session(username: str, display_name: str, repository: "ConversationRepository") -> None:
    st.session_state.authenticated = True
    st.session_state.username = username
    st.session_state.display_name = display_name
    st.session_state.conversation_list = repository.list_for_user(username)
    if st.session_state.conversation_list:
        _select_conversation(repository, username, st.session_state.conversation_list[0]["id"])
    else:
        _create_conversation(repository, username)


def _create_conversation(repository: "ConversationRepository", username: str) -> None:
    st.session_state.active_conversation = repository.create()
    st.session_state.memory.clear()
    st.session_state.conversation_list = repository.list_for_user(username)


def _select_conversation(repository: "ConversationRepository", username: str, conversation_id: str) -> None:
    conversation = repository.load(username, conversation_id)
    if conversation is None:
        return
    st.session_state.active_conversation = conversation
    _restore_memory_context(conversation["messages"])


def _rename_conversation(
    repository: "ConversationRepository", username: str, conversation_id: str, new_title: str
) -> None:
    new_title = new_title.strip()
    if not new_title:
        return
    conversation = repository.load(username, conversation_id)
    if conversation is None:
        return
    conversation["title"] = new_title[:80]
    repository.save(username, conversation)
    if st.session_state.active_conversation and st.session_state.active_conversation["id"] == conversation_id:
        st.session_state.active_conversation["title"] = conversation["title"]
    st.session_state.conversation_list = repository.list_for_user(username)


def _restore_memory_context(messages: list[dict]) -> None:
    """Restore only the bounded context the pipeline used originally, not all UI history."""
    context: list[dict] = []
    for message in messages:
        if message["role"] == "user":
            context.append({"role": "user", "content": message["content"]})
        elif message.get("decision") == "allow_rag":
            context.append({"role": "assistant", "content": message["content"]})
    st.session_state.memory_history[:] = context[-(MEMORY_MAX_TURNS * 2):]


def _logout() -> None:
    """Clear authenticated state and the bounded memory context from this browser session."""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.display_name = None
    st.session_state.active_conversation = None
    st.session_state.conversation_list = []
    st.session_state.memory.clear()


def _format_relative_date(iso_timestamp: str) -> str:
    if not iso_timestamp:
        return ""
    try:
        moment = datetime.fromisoformat(iso_timestamp)
    except ValueError:
        return ""
    today = datetime.now().date()
    when = moment.date()
    if when == today:
        return "Aujourd'hui"
    if (today - when).days == 1:
        return "Hier"
    return moment.strftime("%d/%m/%Y")

