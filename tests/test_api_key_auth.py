"""BRA-881 — DB-authoritative API-key validation (hash + status + expiry).

Unit-tests the validator and the flag-gated auth path: with `auth_db_enabled` a
revoked/expired key in the DB is rejected at runtime (no redeploy); without it,
the env allowlist stays authoritative (offline path unchanged).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_db_pool, get_settings
from app.config import Settings
from app.main import app
from app.repositories.api_keys import hash_key, key_is_active


class _FakeConn:
    """Returns a fixed row from fetchrow (None = key absent)."""

    def __init__(self, row: dict[str, Any] | None) -> None:
        self._row = row
        self.queried_hash: str | None = None

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.queried_hash = args[0]
        return self._row


def test_hash_key_is_deterministic_prefixed_and_non_reversible() -> None:
    assert hash_key("dev-key") == hash_key("dev-key")
    assert hash_key("dev-key").startswith("sha256:")
    assert hash_key("dev-key") != hash_key("other-key")
    assert "dev-key" not in hash_key("dev-key")


async def test_active_key_without_expiry_is_valid() -> None:
    conn = _FakeConn({"status": "active", "expires_at": None})
    assert await key_is_active(conn, "dev-key") is True
    assert conn.queried_hash == hash_key("dev-key")  # looked up by hash, not plaintext


async def test_future_expiry_is_valid() -> None:
    conn = _FakeConn({"status": "active", "expires_at": datetime.now(tz=UTC) + timedelta(days=1)})
    assert await key_is_active(conn, "k") is True


@pytest.mark.parametrize("status", ["revoked", "suspended"])
async def test_non_active_status_is_rejected(status: str) -> None:
    conn = _FakeConn({"status": status, "expires_at": None})
    assert await key_is_active(conn, "k") is False


async def test_expired_key_is_rejected() -> None:
    conn = _FakeConn({"status": "active", "expires_at": datetime.now(tz=UTC) - timedelta(days=1)})
    assert await key_is_active(conn, "k") is False


async def test_unknown_key_is_rejected() -> None:
    assert await key_is_active(_FakeConn(None), "k") is False


# --- flag-gated auth path (end-to-end through the dependency) ----------------


def _settings_db_auth() -> Settings:
    return Settings(
        db_url="postgresql://fake", auth_db_enabled=True, api_keys=("ignored-in-db-mode",)
    )


def test_db_auth_rejects_revoked_key() -> None:
    app.dependency_overrides[get_settings] = _settings_db_auth
    app.dependency_overrides[get_db_pool] = lambda: _FakeConn(
        {"status": "revoked", "expires_at": None}
    )
    try:
        with TestClient(app) as client:
            resp = client.get("/v1/models", headers={"Authorization": "Bearer some-key"})
        assert resp.status_code == 401
    finally:
        app.dependency_overrides.clear()


def test_db_auth_accepts_active_key() -> None:
    app.dependency_overrides[get_settings] = _settings_db_auth
    app.dependency_overrides[get_db_pool] = lambda: _FakeConn(
        {"status": "active", "expires_at": None}
    )
    try:
        with TestClient(app) as client:
            resp = client.get("/v1/models", headers={"Authorization": "Bearer some-key"})
        assert resp.status_code == 200
    finally:
        app.dependency_overrides.clear()
