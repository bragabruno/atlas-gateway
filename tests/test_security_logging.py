"""BRA-879 — security-event logging: auth failures, 429s, request-id access log.

Asserts the detection signal exists (so brute-force / hammering is visible) and,
critically, that the full API key never reaches the logs — only a short prefix.
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.observability.security import (
    record_auth_failure,
    record_rate_limit_rejection,
)


def test_record_auth_failure_logs_prefix_not_full_key(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="atlas.security"):
        record_auth_failure("invalid", key="sk-supersecret-abc123")
    text = caplog.text
    assert "auth_failure" in text
    assert "reason=invalid" in text
    assert "sk-sup" in text  # prefix hint is logged
    assert "sk-supersecret-abc123" not in text  # full key never logged


def test_record_rate_limit_rejection_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="atlas.security"):
        record_rate_limit_rejection("budget", api_key_id="tenant-9")
    assert "rate_limit_rejection" in caplog.text
    assert "kind=budget" in caplog.text


def test_missing_auth_is_logged_via_api(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="atlas.security"), TestClient(app) as client:
        resp = client.get("/v1/models")  # no Authorization header
    assert resp.status_code == 401
    assert "auth_failure" in caplog.text
    assert "reason=missing" in caplog.text


def test_invalid_key_logged_without_leaking_it(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="atlas.security"), TestClient(app) as client:
        resp = client.get("/v1/models", headers={"Authorization": "Bearer totally-wrong-key"})
    assert resp.status_code == 401
    assert "auth_failure" in caplog.text
    assert "totally-wrong-key" not in caplog.text  # the rejected key is not leaked


def test_request_id_header_and_access_log() -> None:
    with TestClient(app) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-Id")  # correlation id is set on every response
