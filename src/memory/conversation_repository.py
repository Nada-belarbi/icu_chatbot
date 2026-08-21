"""Encrypted persistence for complete user conversation histories.

Conversation bodies and titles are encrypted at rest with Fernet before being
written to disk. The encryption key must be provided as ``CHAT_STORAGE_KEY``
in production; a local development key is generated only when no key exists.
Do not use the development-key fallback in a shared or clinical deployment.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Final

from cryptography.fernet import Fernet, InvalidToken

_SAFE_CONVERSATION_ID: Final = re.compile(r"^[0-9a-f-]{36}$")


class ConversationRepository:
    """Read and write encrypted conversations scoped to one authenticated user."""

    def __init__(self, root_path: Path, key_path: Path) -> None:
        self._root_path = root_path
        self._cipher = Fernet(self._load_key(key_path))

    def list_for_user(self, username: str) -> list[dict[str, str]]:
        """Return non-sensitive metadata, newest first, after decrypting each record."""
        conversations: list[dict[str, str]] = []
        for path in self._user_directory(username).glob("*.bin"):
            conversation = self._read_file(path)
            conversations.append(
                {
                    "id": conversation["id"],
                    "title": conversation.get("title", "Nouvelle conversation"),
                    "updated_at": conversation.get("updated_at", ""),
                }
            )
        return sorted(conversations, key=lambda item: item["updated_at"], reverse=True)

    def load(self, username: str, conversation_id: str) -> dict | None:
        if not _SAFE_CONVERSATION_ID.fullmatch(conversation_id):
            return None
        path = self._user_directory(username) / f"{conversation_id}.bin"
        if not path.exists():
            return None
        return self._read_file(path)

    def create(self) -> dict:
        now = datetime.now().isoformat()
        return {
            "id": str(uuid.uuid4()),
            "title": "Nouvelle conversation",
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }

    def save(self, username: str, conversation: dict) -> None:
        conversation_id = str(conversation.get("id", ""))
        if not _SAFE_CONVERSATION_ID.fullmatch(conversation_id):
            raise ValueError("Identifiant de conversation invalide.")
        payload = json.dumps(conversation, ensure_ascii=False).encode("utf-8")
        encrypted = self._cipher.encrypt(payload)
        directory = self._user_directory(username)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{conversation_id}.bin"
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(encrypted)
        os.replace(temporary, destination)
        try:
            os.chmod(destination, 0o600)
        except OSError:
            pass

    def delete(self, username: str, conversation_id: str) -> None:
        """Permanently remove one encrypted conversation owned by ``username``."""
        if not _SAFE_CONVERSATION_ID.fullmatch(conversation_id):
            return
        path = self._user_directory(username) / f"{conversation_id}.bin"
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            raise RuntimeError("La conversation ne peut pas être supprimée.") from exc

    def _read_file(self, path: Path) -> dict:
        try:
            decrypted = self._cipher.decrypt(path.read_bytes())
            data = json.loads(decrypted.decode("utf-8"))
        except (OSError, InvalidToken, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Une conversation enregistrée ne peut pas être lue de manière sûre.") from exc
        if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
            raise RuntimeError("Le format d'une conversation enregistrée est invalide.")
        return data

    def _user_directory(self, username: str) -> Path:
        # Keep identifiers out of paths and enforce per-user data separation.
        user_key = hashlib.sha256(username.casefold().encode("utf-8")).hexdigest()
        return self._root_path / user_key

    @staticmethod
    def _load_key(key_path: Path) -> bytes:
        key_from_environment = os.environ.get("CHAT_STORAGE_KEY", "").strip()
        if key_from_environment:
            try:
                Fernet(key_from_environment.encode("ascii"))
            except (ValueError, UnicodeEncodeError) as exc:
                raise RuntimeError("CHAT_STORAGE_KEY doit être une clé Fernet valide.") from exc
            return key_from_environment.encode("ascii")

        # Local-development fallback. Production must inject CHAT_STORAGE_KEY
        # through the deployment secret store, never commit this generated file.
        if key_path.exists():
            key = key_path.read_bytes().strip()
        else:
            key_path.parent.mkdir(parents=True, exist_ok=True)
            key = Fernet.generate_key()
            temporary = key_path.with_suffix(".tmp")
            temporary.write_bytes(key)
            os.replace(temporary, key_path)
            try:
                os.chmod(key_path, 0o600)
            except OSError:
                pass
        try:
            Fernet(key)
        except ValueError as exc:
            raise RuntimeError("La clé locale de stockage est invalide.") from exc
        return key
