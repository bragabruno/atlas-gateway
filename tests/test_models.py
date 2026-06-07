from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_AUTH = {"Authorization": "Bearer dev-key"}
_ALIASES = {"smart", "deep", "fast", "balanced", "embed"}


def test_models_requires_bearer_token() -> None:
    resp = client.get("/v1/models")
    assert resp.status_code == 401


def test_models_rejects_unknown_key() -> None:
    resp = client.get("/v1/models", headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401


def test_models_openai_shape() -> None:
    resp = client.get("/v1/models", headers=_AUTH)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    ids = {entry["id"] for entry in data["data"]}
    assert "mock" in ids
    assert _ALIASES <= ids
    for entry in data["data"]:
        assert entry["object"] == "model"
        assert set(entry) == {"id", "object", "owned_by"}
    by_id = {entry["id"]: entry for entry in data["data"]}
    assert by_id["mock"]["owned_by"] == "mock"
    assert by_id["smart"]["owned_by"] == "atlas"
