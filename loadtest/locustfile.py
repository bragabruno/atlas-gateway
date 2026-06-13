"""Locust suite for the local Atlas stack — zero-spend mock traffic.

Stdlib + locust only, so it runs unmodified in the official locust image
(the compose `loadtest` profile) and from a host venv (`pip install -e
".[loadtest]"`).

User mix (weights):
- ChatMockUser   — non-streaming chat against `model=mock` (fills
  call_records + atlas.calls.v1 + Valkey cache when accounting is wired),
  with an occasional streaming request (UX path — deliberately NOT recorded
  by accounting; see chat_service module docstring).
- UsagePollUser  — GET /v1/usage (the cost-trail read path).
- AgentRunUser   — POST /v1/agent/runs with the mock agent spec (fills
  agent_runs + agent_steps via the runtime's Postgres persistence).

Env:
  ATLAS_BASE   gateway origin     (default http://localhost:8090)
  ATLAS_KEY    bearer key         (default dev-key)
  ATLAS_AGENT  agent-runtime base (default http://localhost:8083)
  ATLAS_AGENT_NAME  agent spec    (default regdoc-qa-mock)

Run (host):       locust -f loadtest/locustfile.py --headless -u 20 -r 5 -t 60s
Run (container):  docker compose -f local/compose.dev.yaml --profile loadtest up locust
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, task

_BASE = os.environ.get("ATLAS_BASE", "http://localhost:8090")
_KEY = os.environ.get("ATLAS_KEY", "dev-key")
_AGENT = os.environ.get("ATLAS_AGENT", "http://localhost:8083")
_AGENT_NAME = os.environ.get("ATLAS_AGENT_NAME", "regdoc-qa-mock")

_HEADERS = {"Authorization": f"Bearer {_KEY}", "Content-Type": "application/json"}

_QUESTIONS = [
    "What does GDPR Article 6 require for consent?",
    "Summarize the REACH Annex XVII restrictions for entry 27.",
    "Which CLP article governs hazard labelling?",
    "What are the record-keeping duties under Article 30?",
    "When is a DPIA mandatory under Article 35?",
    "What does the Biocidal Products Regulation require for treated articles?",
    "Outline the SDS requirements under REACH Annex II.",
    "What is the lawful basis of legitimate interests?",
]


def _chat_body(*, stream: bool) -> dict:
    return {
        "model": "mock",
        "messages": [{"role": "user", "content": random.choice(_QUESTIONS)}],
        "stream": stream,
    }


class ChatMockUser(HttpUser):
    """Chat traffic through the full gateway path (auth → service → mock)."""

    host = _BASE
    weight = 6
    wait_time = between(0.5, 2.0)

    @task(8)
    def chat_non_streaming(self) -> None:
        # 429 is the token-bucket rate limiter (GW-16) doing its job — all
        # users share one dev key (capacity 60, ~1 rps sustained refill), so
        # saturation throttling is the EXPECTED outcome, not a failure. For
        # raw-throughput runs disable ATLAS_RATE_LIMIT_ENABLED or use one key
        # per user.
        with self.client.post(
            "/v1/chat/completions",
            json=_chat_body(stream=False),
            headers=_HEADERS,
            name="POST /v1/chat/completions [mock]",
            catch_response=True,
        ) as resp:
            if resp.status_code in (200, 429):
                resp.success()
            else:
                resp.failure(f"unexpected status {resp.status_code}")

    @task(2)
    def chat_streaming(self) -> None:
        # Streamed responses are consumed fully; accounting deliberately skips
        # this path, so it exercises UX/SSE without touching call_records.
        with self.client.post(
            "/v1/chat/completions",
            json=_chat_body(stream=True),
            headers=_HEADERS,
            stream=True,
            name="POST /v1/chat/completions [mock,stream]",
            catch_response=True,
        ) as resp:
            body = b"".join(resp.iter_content(chunk_size=None) or [])
            if b"[DONE]" in body:
                resp.success()
            else:
                resp.failure("stream ended without [DONE]")

    @task(1)
    def list_models(self) -> None:
        self.client.get("/v1/models", headers=_HEADERS, name="GET /v1/models")


class UsagePollUser(HttpUser):
    """Cost-trail reads — what a dashboard would do."""

    host = _BASE
    weight = 1
    wait_time = between(2.0, 5.0)

    @task
    def usage(self) -> None:
        self.client.get("/v1/usage", headers=_HEADERS, name="GET /v1/usage")


class AgentRunUser(HttpUser):
    """Bounded agent runs against the mock spec (fills agent_runs/steps).

    Runs execute synchronously inside the POST, so latency = one full agent
    loop; keep the user count modest relative to chat users (weight 1).
    """

    host = _AGENT
    weight = 1
    wait_time = between(2.0, 6.0)

    @task
    def run_agent(self) -> None:
        # 429 = the gateway throttled the run's LLM call (propagated by the
        # runtime) — expected under saturation, same as the chat path.
        with self.client.post(
            "/v1/agent/runs",
            json={
                "agent_name": _AGENT_NAME,
                "user_message": random.choice(_QUESTIONS),
            },
            headers={"Content-Type": "application/json"},
            name=f"POST /v1/agent/runs [{_AGENT_NAME}]",
            catch_response=True,
        ) as resp:
            if resp.status_code in (201, 429):
                resp.success()
            else:
                resp.failure(f"unexpected status {resp.status_code}")
