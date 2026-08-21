from pathlib import Path

from src.auth.local_auth import LocalAuthService


def test_register_hashes_password_and_authenticates(tmp_path: Path) -> None:
    service = LocalAuthService(tmp_path / "users.json")

    created = service.register("famille.dupont", "mot-de-passe-solide", "mot-de-passe-solide", True)

    assert created.success is True
    record = (tmp_path / "users.json").read_text(encoding="utf-8")
    assert "mot-de-passe-solide" not in record
    assert service.authenticate("famille.dupont", "mot-de-passe-solide").success is True
    assert service.authenticate("famille.dupont", "incorrect").success is False


def test_register_requires_confirmation_consent_and_strong_password(tmp_path: Path) -> None:
    service = LocalAuthService(tmp_path / "users.json")

    assert service.register("ab", "court", "court", True).success is False
    assert service.register("famille.dupont", "mot-de-passe-solide", "autre-mot-de-passe", True).success is False
    assert service.register("famille.dupont", "mot-de-passe-solide", "mot-de-passe-solide", False).success is False
