"""Configuration safety and input validation (no database needed)."""
import pytest

from website.auth import validate_password, validate_registration
from website.config import ConfigError, db_settings, flask_config


def test_defaults_point_to_v2_and_never_the_legacy_database(monkeypatch):
    monkeypatch.delenv("SAFEDRIVE_DB_NAME", raising=False)
    monkeypatch.delenv("SAFEDRIVE_DB_PORT", raising=False)
    s = db_settings()
    assert s["database"] == "safedrive_ai_v2" and s["port"] == 3306


def test_no_hardcoded_secret_and_debug_off_by_default(monkeypatch):
    monkeypatch.delenv("SAFEDRIVE_SECRET_KEY", raising=False)
    monkeypatch.delenv("SAFEDRIVE_DEBUG", raising=False)
    a, b = flask_config(), flask_config()
    assert a["SECRET_KEY"] != b["SECRET_KEY"] and len(a["SECRET_KEY"]) >= 32   # random per run
    assert a["DEBUG"] is False
    assert a["SESSION_COOKIE_HTTPONLY"] and a["SESSION_COOKIE_SAMESITE"] == "Lax"


def test_production_refuses_unsafe_settings(monkeypatch):
    monkeypatch.setenv("SAFEDRIVE_ENV", "production")
    monkeypatch.delenv("SAFEDRIVE_SECRET_KEY", raising=False)
    monkeypatch.setenv("SAFEDRIVE_DEBUG", "1")
    monkeypatch.setenv("SAFEDRIVE_DB_PASSWORD", "")
    with pytest.raises(ConfigError) as err:
        flask_config()
    assert "SECRET_KEY" in str(err.value) and "DEBUG" in str(err.value)


def test_production_accepts_safe_settings(monkeypatch):
    monkeypatch.setenv("SAFEDRIVE_ENV", "production")
    monkeypatch.setenv("SAFEDRIVE_SECRET_KEY", "x" * 40)
    monkeypatch.setenv("SAFEDRIVE_DB_PASSWORD", "strong-db-password")
    monkeypatch.delenv("SAFEDRIVE_DEBUG", raising=False)
    cfg = flask_config()
    assert cfg["SESSION_COOKIE_SECURE"] is True and cfg["DEBUG"] is False


@pytest.mark.parametrize("pw, ok", [("short1", False), ("12345678", False), ("abcdefgh", False),
                                    ("Passw0rd!", True), ("é" * 40, False)])
def test_password_policy(pw, ok):
    assert (validate_password(pw) == []) is ok


def test_registration_validation():
    assert validate_registration("Fabrice Ndayisaba", "f@example.com", "Passw0rd!", "Passw0rd!") == []
    errors = validate_registration("x", "not-an-email", "Passw0rd!", "different1")
    assert len(errors) == 3


# ---------------------------------------------------------------- database safety
def test_legacy_database_is_refused_everywhere(monkeypatch):
    from website import get_connection, migrate
    monkeypatch.setenv("SAFEDRIVE_DB_NAME", "safedrive_ai")
    for action in (db_settings, get_connection, migrate.target_database, migrate.create_database, migrate.migrate):
        with pytest.raises(ConfigError):
            action()


@pytest.mark.parametrize("name, allowed", [("safedrive_ai_v2", True), ("safedrive_ai_v2_test", True),
                                           ("other_db", False), ("safedrive_ai_v3", False)])
def test_migrations_only_target_v2_databases(monkeypatch, name, allowed):
    from website import migrate
    monkeypatch.setenv("SAFEDRIVE_DB_NAME", name)
    if allowed:
        assert migrate.target_database() == name
    else:
        with pytest.raises(ValueError):
            migrate.target_database()
