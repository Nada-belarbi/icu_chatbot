"""Privacy-conscious local authentication for a single Streamlit deployment.

This module intentionally stores only a username and a slow password verifier.
It never stores a password, email address, patient identifier, or chat content.
For a public/clinical deployment, replace this local implementation with the
organisation's identity provider (OIDC/SAML) and server-side session controls.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_USERNAME_PATTERN: Final = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{2,31}$"  # simple local identifiers
    r"|^[^\s@]{1,64}@[^\s@]{1,190}\.[^\s@]{2,24}$"  # email-shaped identifiers
)
_SCRYPT_N: Final = 2**14
_SCRYPT_R: Final = 8
_SCRYPT_P: Final = 1


@dataclass(frozen=True)
class AuthResult:
    """Outcome returned to the UI without disclosing account-existence details."""

    success: bool
    message: str = ""


class LocalAuthService:
    """Manage local account credentials using salted ``hashlib.scrypt`` hashes."""

    def __init__(self, users_path: Path) -> None:
        self._users_path = users_path

    def register(
        self,
        username: str,
        password: str,
        password_confirmation: str,
        consent: bool,
        display_name: str = "",
    ) -> AuthResult:
        username = username.strip()
        display_name = display_name.strip()
        validation_message = self._validate_registration(username, password, password_confirmation, consent)
        if validation_message:
            return AuthResult(False, validation_message)

        users = self._read_users()
        if username.casefold() in (existing.casefold() for existing in users):
            return AuthResult(False, "Cette adresse est déjà associée à un compte.")

        salt = secrets.token_bytes(16)
        password_hash = self._derive_hash(password, salt)
        users[username] = {
            "password_hash": base64.b64encode(password_hash).decode("ascii"),
            "salt": base64.b64encode(salt).decode("ascii"),
            "display_name": display_name,
        }
        self._write_users(users)
        return AuthResult(True)

    def authenticate(self, username: str, password: str) -> AuthResult:
        username = username.strip()
        users = self._read_users()
        record = users.get(username)
        if record is None:
            # Execute one equivalent derivation to reduce account-enumeration timing differences.
            self._derive_hash(password, b"\0" * 16)
            return AuthResult(False, "Identifiant ou mot de passe incorrect.")

        try:
            salt = base64.b64decode(record["salt"], validate=True)
            expected_hash = base64.b64decode(record["password_hash"], validate=True)
        except (KeyError, ValueError, TypeError):
            return AuthResult(False, "Identifiant ou mot de passe incorrect.")

        actual_hash = self._derive_hash(password, salt)
        if not hmac.compare_digest(actual_hash, expected_hash):
            return AuthResult(False, "Identifiant ou mot de passe incorrect.")
        return AuthResult(True)

    def get_display_name(self, username: str) -> str:
        """Return the friendly name shown in the UI, falling back to the identifier."""
        record = self._read_users().get(username.strip())
        if record:
            display_name = record.get("display_name", "").strip()
            if display_name:
                return display_name
        return username.strip()

    @staticmethod
    def _derive_hash(password: str, salt: bytes) -> bytes:
        return hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=_SCRYPT_N,
            r=_SCRYPT_R,
            p=_SCRYPT_P,
            dklen=32,
        )

    @staticmethod
    def _validate_registration(username: str, password: str, password_confirmation: str, consent: bool) -> str:
        if not _USERNAME_PATTERN.fullmatch(username):
            return "Merci de saisir une adresse e-mail valide ou un identifiant de 3 à 32 caractères."
        if len(password) < 12:
            return "Le mot de passe doit contenir au moins 12 caractères."
        if password != password_confirmation:
            return "Les mots de passe ne correspondent pas."
        if not consent:
            return "Votre accord est nécessaire pour créer un compte et conserver les conversations chiffrées."
        return ""

    def _read_users(self) -> dict[str, dict[str, str]]:
        if not self._users_path.exists():
            return {}
        try:
            data = json.loads(self._users_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # Never replace a malformed credential store; fail closed instead.
            raise RuntimeError("Le registre local des comptes est indisponible.")
        if not isinstance(data, dict):
            raise RuntimeError("Le registre local des comptes est invalide.")
        return data

    def _write_users(self, users: dict[str, dict[str, str]]) -> None:
        self._users_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._users_path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary_path, self._users_path)
        try:
            os.chmod(self._users_path, 0o600)
        except OSError:
            # Windows ACL configuration belongs to deployment provisioning.
            pass
