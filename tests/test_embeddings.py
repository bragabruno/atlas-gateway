from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_AUTH = {"Authorization": "Bearer dev-key"}


def test_embeddings_requires_bearer_token() -> None:
    resp = client.post("/v1/embeddings", json={"model": "mock", "input": "hello"})
    assert resp.status_code == 401


def test_embeddings_rejects_unknown_key() -> None:
    resp = client.post(
        "/v1/embeddings",
        headers={"Authorization": "Bearer nope"},
        json={"model": "mock", "input": "hello"},
    )
    assert resp.status_code == 401


def test_embeddings_single_input_openai_shape() -> None:
    resp = client.post("/v1/embeddings", headers=_AUTH, json={"model": "mock", "input": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert data["model"] == "mock"
    assert len(data["data"]) == 1
    item = data["data"][0]
    assert item["object"] == "embedding"
    assert item["index"] == 0
    assert len(item["embedding"]) == 8
    assert all(isinstance(value, float) for value in item["embedding"])
    assert set(data["usage"]) == {"prompt_tokens", "total_tokens"}
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"]


def test_embeddings_batch_input_preserves_order() -> None:
    resp = client.post(
        "/v1/embeddings",
        headers=_AUTH,
        json={"model": "mock", "input": ["alpha", "beta", "gamma"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert [item["index"] for item in data["data"]] == [0, 1, 2]
    assert all(len(item["embedding"]) == 8 for item in data["data"])


def test_embeddings_is_deterministic() -> None:
    body = {"model": "mock", "input": "repeatable"}
    first = client.post("/v1/embeddings", headers=_AUTH, json=body).json()
    second = client.post("/v1/embeddings", headers=_AUTH, json=body).json()
    assert first["data"][0]["embedding"] == second["data"][0]["embedding"]


def test_embeddings_unknown_model_returns_404() -> None:
    resp = client.post(
        "/v1/embeddings",
        headers=_AUTH,
        json={"model": "does-not-exist", "input": "hello"},
    )
    assert resp.status_code == 404
