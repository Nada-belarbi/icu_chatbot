# Unit tests — ConversationMemoryManager
# Cases:
#   - add messages → appear in get_history()
#   - history capped at max_turns (default 3 turns = 6 messages)
#   - clear() → empty history
#   - get_context_string() → correct formatted text
#   - two instances with separate lists share no state

import pytest
from src.memory.conversation_memory import ConversationMemoryManager


@pytest.fixture
def mem():
    """Fresh manager with an independent list and max_turns=3."""
    return ConversationMemoryManager(history=[], max_turns=3)


class TestAddMessages:
    def test_add_user_message(self, mem):
        mem.add_user_message("bonjour")
        history = mem.get_history()
        assert len(history) == 1
        assert history[0] == {"role": "user", "content": "bonjour"}

    def test_add_assistant_message(self, mem):
        mem.add_assistant_message("Je peux vous aider.")
        history = mem.get_history()
        assert len(history) == 1
        assert history[0] == {"role": "assistant", "content": "Je peux vous aider."}

    def test_add_both_roles(self, mem):
        mem.add_user_message("Q1")
        mem.add_assistant_message("R1")
        history = mem.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"


class TestTurnLimit:
    def test_history_capped_at_max_turns(self):
        mem = ConversationMemoryManager(history=[], max_turns=2)
        # Add 3 full turns (6 messages)
        for i in range(3):
            mem.add_user_message(f"Q{i}")
            mem.add_assistant_message(f"R{i}")
        # Only last 2 turns (4 messages) should remain
        history = mem.get_history()
        assert len(history) == 4
        assert history[0]["content"] == "Q1"
        assert history[-1]["content"] == "R2"

    def test_oldest_messages_dropped_first(self):
        mem = ConversationMemoryManager(history=[], max_turns=1)
        mem.add_user_message("old question")
        mem.add_assistant_message("old answer")
        mem.add_user_message("new question")
        mem.add_assistant_message("new answer")
        history = mem.get_history()
        assert len(history) == 2
        assert history[0]["content"] == "new question"
        assert history[1]["content"] == "new answer"

    def test_default_max_turns_is_3(self):
        mem = ConversationMemoryManager(history=[])
        for i in range(4):
            mem.add_user_message(f"Q{i}")
            mem.add_assistant_message(f"R{i}")
        assert len(mem.get_history()) == 6  # 3 turns × 2 messages


class TestClear:
    def test_clear_empties_history(self, mem):
        mem.add_user_message("bonjour")
        mem.add_assistant_message("réponse")
        mem.clear()
        assert mem.get_history() == []

    def test_clear_then_add(self, mem):
        mem.add_user_message("avant clear")
        mem.clear()
        mem.add_user_message("après clear")
        history = mem.get_history()
        assert len(history) == 1
        assert history[0]["content"] == "après clear"


class TestContextString:
    def test_empty_history_returns_empty_string(self, mem):
        assert mem.get_context_string() == ""

    def test_single_user_message(self, mem):
        mem.add_user_message("qu'est-ce qu'un respirateur ?")
        ctx = mem.get_context_string()
        assert ctx == "Utilisateur: qu'est-ce qu'un respirateur ?"

    def test_full_turn_format(self, mem):
        mem.add_user_message("mon père est en réa")
        mem.add_assistant_message("Je comprends votre inquiétude.")
        ctx = mem.get_context_string()
        assert "Utilisateur: mon père est en réa" in ctx
        assert "Assistant: Je comprends votre inquiétude." in ctx

    def test_context_respects_turn_limit(self):
        mem = ConversationMemoryManager(history=[], max_turns=1)
        mem.add_user_message("Q0")
        mem.add_assistant_message("R0")
        mem.add_user_message("Q1")
        mem.add_assistant_message("R1")
        ctx = mem.get_context_string()
        assert "Q0" not in ctx
        assert "Q1" in ctx


class TestIsolation:
    def test_two_instances_share_no_state(self):
        mem1 = ConversationMemoryManager(history=[], max_turns=3)
        mem2 = ConversationMemoryManager(history=[], max_turns=3)
        mem1.add_user_message("message pour mem1")
        assert mem2.get_history() == []

    def test_get_history_returns_copy(self, mem):
        mem.add_user_message("original")
        snapshot = mem.get_history()
        snapshot.append({"role": "user", "content": "injected"})
        # Internal state must not be mutated
        assert len(mem.get_history()) == 1
