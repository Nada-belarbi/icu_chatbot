# ConversationMemoryManager
#
# Responsibilities:
#   - Store recent user/assistant exchanges for the current session only
#   - Limit history to the last N turns (configurable in settings.py)
#
# Design note:
#   The class accepts an external list as its backing store.
#   In production (app.py) that list is st.session_state.history, which is
#   scoped per browser tab and wiped on refresh.
#   In unit tests a plain [] is passed instead — no Streamlit dependency.

from __future__ import annotations


class ConversationMemoryManager:
    """
    Session-scoped conversation history.

    Parameters
    ----------
    history:
        External list used as the backing store.  In production pass
        ``st.session_state.history``; in tests pass an ordinary ``[]``.
    max_turns:
        Maximum number of (user, assistant) exchange pairs to keep.
        Older turns are silently dropped.  Default: 3.
    """

    def __init__(self, history: list | None = None, max_turns: int = 3) -> None:
        self._history: list[dict] = history if history is not None else []
        self._max_turns = max_turns

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_user_message(self, text: str) -> None:
        """Append a user turn and enforce the turn cap."""
        self._history.append({"role": "user", "content": text})
        self._enforce_limit()

    def add_assistant_message(self, text: str) -> None:
        """Append an assistant turn and enforce the turn cap."""
        self._history.append({"role": "assistant", "content": text})
        self._enforce_limit()

    def clear(self) -> None:
        """Wipe the entire history (called on session reset)."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_history(self) -> list[dict]:
        """
        Return the current message list (capped at max_turns * 2 messages).
        Each item: ``{"role": "user" | "assistant", "content": str}``.
        """
        return list(self._history)

    def get_context_string(self) -> str:
        """
        Format the last ``max_turns`` exchanges as a short text prefix
        fed to the query classifier for context.

        Example output::

            Utilisateur: mon p\u00e8re a \u00e9t\u00e9 admis hier
            Assistant: Je comprends votre inqui\u00e9tude...
            Utilisateur: est-ce grave ?
        """
        if not self._history:
            return ""

        lines = []
        for msg in self._history:
            role_label = "Utilisateur" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role_label}: {msg['content']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enforce_limit(self) -> None:
        """
        Keep at most ``max_turns * 2`` individual messages.
        A "turn" is one user message + one assistant message = 2 items.
        Excess messages are removed from the front (oldest first).
        """
        max_messages = self._max_turns * 2
        if len(self._history) > max_messages:
            del self._history[: len(self._history) - max_messages]

#   - Expose method to clear memory at session end
#   - Never persist data between sessions
#   - Never store personally identifiable information
