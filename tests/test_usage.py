"""FE-6 — Tests for GET /v1/usage.

The asyncpg pool is mocked so no live DB is needed.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_db_pool, get_settings
from app.config import Settings
from app.main import app


def _make_row(app_name: str, model: str, inp: int, out: int, cost: str) -> MagicMock:
    row = MagicMock()
    row.__getitem__ = lambda self, key: {
        "app": app_name,
        "model": model,
        "input_tokens": inp,
        "output_tokens": out,
        "total_cost_usd": Decimal(cost),
    }[key]
    return row


def _settings_with_db() -> Settings:
    return Settings(db_url="postgresql://fake", api_keys=("test-key",))


def _settings_no_db() -> Settings:
    return Settings(db_url=None, api_keys=("test-key",))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_usage_returns_503_when_no_db_configured() -> None:
    app.dependency_overrides[get_settings] = _settings_no_db
    app.dependency_overrides[get_db_pool] = lambda: None
    try:
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/usage", headers={"Authorization": "Bearer test-key"})
        assert resp.status_code == 503
    finally:
        app.dependency_overrides.clear()


def test_usage_returns_rows_from_db() -> None:
    fake_pool = AsyncMock()
    fake_pool.fetch.return_value = [
        _make_row("atlas-frontend", "gpt-4o", 1000, 500, "0.01234567"),
        _make_row("atlas-frontend", "claude-sonnet-4-6", 2000, 800, "0.02345678"),
    ]

    app.dependency_overrides[get_settings] = _settings_with_db
    app.dependency_overrides[get_db_pool] = lambda: fake_pool
    try:
        client = TestClient(app)
        resp = client.get("/v1/usage", headers={"Authorization": "Bearer test-key"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["rows"]) == 2
        assert data["rows"][0]["model"] == "gpt-4o"
    finally:
        app.dependency_overrides.clear()


def test_usage_defaults_since_to_first_of_month() -> None:
    fake_pool = AsyncMock()
    fake_pool.fetch.return_value = []

    app.dependency_overrides[get_settings] = _settings_with_db
    app.dependency_overrides[get_db_pool] = lambda: fake_pool
    try:
        client = TestClient(app)
        resp = client.get("/v1/usage", headers={"Authorization": "Bearer test-key"})
        assert resp.status_code == 200
        since = date.fromisoformat(resp.json()["since"])
        assert since.day == 1
    finally:
        app.dependency_overrides.clear()


def test_usage_accepts_custom_since_param() -> None:
    fake_pool = AsyncMock()
    fake_pool.fetch.return_value = []

    app.dependency_overrides[get_settings] = _settings_with_db
    app.dependency_overrides[get_db_pool] = lambda: fake_pool
    try:
        client = TestClient(app)
        resp = client.get(
            "/v1/usage?since=2026-01-01", headers={"Authorization": "Bearer test-key"}
        )
        assert resp.status_code == 200
        assert resp.json()["since"] == "2026-01-01"
    finally:
        app.dependency_overrides.clear()


def test_usage_requires_auth() -> None:
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/v1/usage")
    assert resp.status_code == 401
