"""CORS is config-gated and default OFF (see app/config.py / app/main.py).

The middleware is wired at import time from ATLAS_CORS_ALLOW_ORIGINS, so the
configured-on case is exercised by reloading app.main with the env set, then
restoring the default app object for the rest of the suite.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import app.main as main_mod


def test_no_cors_headers_by_default() -> None:
    """With no ATLAS_CORS_ALLOW_ORIGINS, no CORS middleware is added."""
    client = TestClient(main_mod.app)
    resp = client.get("/healthz", headers={"Origin": "http://localhost:8080"})
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers


@pytest.fixture
def cors_app(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Reload app.main with CORS configured; restore the default app after."""
    monkeypatch.setenv("ATLAS_CORS_ALLOW_ORIGINS", '["http://localhost:8080"]')
    importlib.reload(main_mod)
    try:
        yield TestClient(main_mod.app)
    finally:
        monkeypatch.delenv("ATLAS_CORS_ALLOW_ORIGINS", raising=False)
        importlib.reload(main_mod)  # restore the default (no-CORS) app object


def test_preflight_allows_configured_origin(cors_app: TestClient) -> None:
    resp = cors_app.options(
        "/v1/chat/completions",
        headers={
            "Origin": "http://localhost:8080",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:8080"


def test_disallowed_origin_gets_no_cors_header(cors_app: TestClient) -> None:
    resp = cors_app.get("/healthz", headers={"Origin": "http://evil.example"})
    # The request still succeeds, but the disallowed origin is not echoed back.
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") != "http://evil.example"
