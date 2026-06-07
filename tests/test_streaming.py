"""GW-7 — streaming SSE contract for /v1/chat/completions.

Pins the OpenAI-compatible `text/event-stream` shape: a leading role delta,
content fragments that reconstruct the full message, a terminal frame with
`finish_reason` + `usage`, and the `[DONE]` sentinel.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_AUTH = {"Authorization": "Bearer dev-key"}
_BODY = {
    "model": "mock",
    "messages": [{"role": "user", "content": "hi there"}],
    "stream": True,
}


def _parse_sse(text: str) -> tuple[list[dict[str, Any]], bool]:
    """Return (chunk payloads, saw_done) from a raw SSE body."""
    chunks: list[dict[str, Any]] = []
    saw_done = False
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ")
        if payload == "[DONE]":
            saw_done = True
            continue
        chunks.append(json.loads(payload))
    return chunks, saw_done


def test_stream_content_type_and_done_sentinel() -> None:
    resp = client.post("/v1/chat/completions", headers=_AUTH, json=_BODY)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    _, saw_done = _parse_sse(resp.text)
    assert saw_done is True


def test_stream_first_chunk_is_role_delta() -> None:
    resp = client.post("/v1/chat/completions", headers=_AUTH, json=_BODY)
    chunks, _ = _parse_sse(resp.text)
    assert chunks, "expected at least one chunk"
    first = chunks[0]
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"]["role"] == "assistant"


def test_stream_reconstructs_message() -> None:
    resp = client.post("/v1/chat/completions", headers=_AUTH, json=_BODY)
    chunks, _ = _parse_sse(resp.text)
    content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert "hi there" in content


def test_stream_final_chunk_has_finish_reason_and_usage() -> None:
    resp = client.post("/v1/chat/completions", headers=_AUTH, json=_BODY)
    chunks, _ = _parse_sse(resp.text)
    final = chunks[-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    usage = final["usage"]
    assert set(usage) == {"prompt_tokens", "completion_tokens", "total_tokens"}
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_stream_all_chunks_share_id_and_model() -> None:
    resp = client.post("/v1/chat/completions", headers=_AUTH, json=_BODY)
    chunks, _ = _parse_sse(resp.text)
    ids = {c["id"] for c in chunks}
    models = {c["model"] for c in chunks}
    assert len(ids) == 1
    assert models == {"mock"}
    assert next(iter(ids)).startswith("chatcmpl-")


def test_stream_requires_bearer_token() -> None:
    resp = client.post("/v1/chat/completions", json=_BODY)
    assert resp.status_code == 401
