from pathlib import Path

from src.memory.conversation_repository import ConversationRepository


def test_conversations_are_encrypted_scoped_and_recoverable(tmp_path: Path) -> None:
    repository = ConversationRepository(tmp_path / "conversations", tmp_path / "storage.key")
    conversation = repository.create()
    conversation["title"] = "Question sur le respirateur"
    conversation["messages"].append({"role": "user", "content": "Contenu privé"})
    repository.save("famille.dupont", conversation)

    stored_files = list((tmp_path / "conversations").rglob("*.bin"))
    assert len(stored_files) == 1
    assert b"Contenu priv" not in stored_files[0].read_bytes()
    assert repository.list_for_user("autre.utilisateur") == []
    assert repository.load("famille.dupont", conversation["id"]) == conversation


def test_delete_only_removes_the_requested_conversation(tmp_path: Path) -> None:
    repository = ConversationRepository(tmp_path / "conversations", tmp_path / "storage.key")
    first = repository.create()
    second = repository.create()
    repository.save("famille.dupont", first)
    repository.save("famille.dupont", second)

    repository.delete("famille.dupont", first["id"])

    assert repository.load("famille.dupont", first["id"]) is None
    assert repository.load("famille.dupont", second["id"]) == second
