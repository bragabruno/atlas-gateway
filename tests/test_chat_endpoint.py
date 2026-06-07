from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_AUTH = {"Authorization": "Bearer dev-key"}
_BODY = {"model": "mock", "messages": [{"role": "user", "content": "hi there"}]}


def test_healthz() -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_requires_bearer_token() -> None:
    resp = client.post("/v1/chat/completions", json=_BODY)
    assert resp.status_code == 401


def test_rejects_unknown_key() -> None:
    resp = client.post("/v1/chat/completions", headers={"Authorization": "Bearer nope"}, json=_BODY)
    assert resp.status_code == 401


def test_chat_completion_openai_shape() -> None:
    resp = client.post("/v1/chat/completions", headers=_AUTH, json=_BODY)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "mock"
    assert data["id"].startswith("chatcmpl-")
    choice = data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert "hi there" in choice["message"]["content"]
    assert choice["finish_reason"] == "stop"
    assert set(data["usage"]) == {"prompt_tokens", "completion_tokens", "total_tokens"}
    assert data["usage"]["total_tokens"] == (
        data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
    )


def test_unknown_model_returns_404() -> None:
    resp = client.post(
        "/v1/chat/completions",
        headers=_AUTH,
        json={"model": "does-not-exist", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 404
