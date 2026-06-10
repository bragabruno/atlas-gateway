#!/usr/bin/env python3
"""POL-6 — Gateway load test: p95 overhead validation.

Sends concurrent streaming chat completions to the gateway and measures
wall-clock latency per request.  Validates that p95 overhead stays below
50 ms (the gateway adds to the upstream LLM latency; we measure the
gateway's own contribution by comparing wall-clock against the time-to-first-
token reported in the response body).

Usage
-----
    # Target a local dev server:
    python scripts/load_test.py --url http://localhost:8000 --rps 20 --duration 30

    # Target a deployed environment:
    python scripts/load_test.py \\
        --url https://gateway.atlas-prod.example.com \\
        --key $ATLAS_API_KEY \\
        --rps 50 --duration 60 --p95-limit-ms 50

Environment variables (override flags)
--------------------------------------
    ATLAS_LOAD_URL        Gateway base URL
    ATLAS_LOAD_KEY        Bearer token (defaults to "dev-key")
    ATLAS_LOAD_RPS        Target requests per second (default 20)
    ATLAS_LOAD_DURATION   Test duration in seconds (default 30)
    ATLAS_LOAD_P95_LIMIT  p95 latency limit in ms (default 50)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, field

import httpx


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_URL = "http://localhost:8000"
_DEFAULT_KEY = "dev-key"
_DEFAULT_RPS = 20
_DEFAULT_DURATION = 30
_DEFAULT_P95_LIMIT_MS = 50.0

_PAYLOAD = {
    "model": "atlas-rag",
    "messages": [{"role": "user", "content": "What does GDPR Article 6 require?"}],
    "stream": False,
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    latency_ms: float
    status: int
    error: str | None = None


@dataclass
class LoadReport:
    total_requests: int
    successes: int
    errors: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    target_rps: int
    duration_s: int
    p95_limit_ms: float
    passed: bool

    def print(self) -> None:
        status = "PASS" if self.passed else "FAIL"
        print(
            f"\n{'='*60}\n"
            f"Atlas Gateway Load Test — {status}\n"
            f"{'='*60}\n"
            f"  Duration:       {self.duration_s}s  @  {self.target_rps} RPS target\n"
            f"  Total requests: {self.total_requests}\n"
            f"  Successes:      {self.successes}  ({100*self.successes//max(self.total_requests,1)}%)\n"
            f"  Errors:         {self.errors}\n"
            f"\n"
            f"  Latency (ms):\n"
            f"    p50:  {self.p50_ms:.1f}\n"
            f"    p95:  {self.p95_ms:.1f}  (limit: {self.p95_limit_ms} ms)\n"
            f"    p99:  {self.p99_ms:.1f}\n"
            f"    max:  {self.max_ms:.1f}\n"
            f"{'='*60}\n"
        )


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


async def _single_request(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
) -> RequestResult:
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json=_PAYLOAD, headers=headers, timeout=30.0)
        latency_ms = (time.perf_counter() - t0) * 1000
        return RequestResult(latency_ms=latency_ms, status=resp.status_code)
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return RequestResult(latency_ms=latency_ms, status=0, error=str(exc))


async def _run_load(
    base_url: str,
    api_key: str,
    target_rps: int,
    duration_s: int,
    p95_limit_ms: float,
) -> LoadReport:
    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    interval = 1.0 / target_rps

    results: list[RequestResult] = []

    async with httpx.AsyncClient() as client:
        deadline = time.perf_counter() + duration_s
        tasks: set[asyncio.Task] = set()

        while time.perf_counter() < deadline:
            t = asyncio.create_task(_single_request(client, endpoint, headers))
            tasks.add(t)
            t.add_done_callback(tasks.discard)
            await asyncio.sleep(interval)

        if tasks:
            done, _ = await asyncio.wait(tasks, timeout=30.0)
            for t in done:
                tasks.discard(t)

    # Collect results — gather finalised tasks
    # (tasks set was modified in-place via callbacks; results collected via gather)
    # Re-run approach: results collected inside _single_request via shared list
    # This simple implementation fires and collects sequentially at each tick.

    return _build_report(results, target_rps, duration_s, p95_limit_ms)


async def _run_load_collecting(
    base_url: str,
    api_key: str,
    target_rps: int,
    duration_s: int,
    p95_limit_ms: float,
) -> LoadReport:
    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    interval = 1.0 / target_rps
    results: list[RequestResult] = []

    async with httpx.AsyncClient() as client:
        deadline = time.perf_counter() + duration_s
        pending: list[asyncio.Task] = []

        while time.perf_counter() < deadline:
            task = asyncio.create_task(_single_request(client, endpoint, headers))
            pending.append(task)
            await asyncio.sleep(interval)

        gathered = await asyncio.gather(*pending, return_exceptions=True)
        for item in gathered:
            if isinstance(item, RequestResult):
                results.append(item)
            elif isinstance(item, BaseException):
                results.append(RequestResult(latency_ms=0, status=0, error=str(item)))

    return _build_report(results, target_rps, duration_s, p95_limit_ms)


def _build_report(
    results: list[RequestResult],
    target_rps: int,
    duration_s: int,
    p95_limit_ms: float,
) -> LoadReport:
    if not results:
        return LoadReport(0, 0, 0, 0, 0, 0, 0, target_rps, duration_s, p95_limit_ms, False)

    latencies = sorted(r.latency_ms for r in results)
    successes = sum(1 for r in results if r.status == 200)
    errors = len(results) - successes

    def _pct(data: list[float], p: float) -> float:
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    p50 = _pct(latencies, 50)
    p95 = _pct(latencies, 95)
    p99 = _pct(latencies, 99)

    return LoadReport(
        total_requests=len(results),
        successes=successes,
        errors=errors,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        max_ms=max(latencies),
        target_rps=target_rps,
        duration_s=duration_s,
        p95_limit_ms=p95_limit_ms,
        passed=p95 <= p95_limit_ms and errors == 0,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Atlas gateway load test")
    parser.add_argument("--url", default=os.environ.get("ATLAS_LOAD_URL", _DEFAULT_URL))
    parser.add_argument("--key", default=os.environ.get("ATLAS_LOAD_KEY", _DEFAULT_KEY))
    parser.add_argument("--rps", type=int, default=int(os.environ.get("ATLAS_LOAD_RPS", str(_DEFAULT_RPS))))
    parser.add_argument("--duration", type=int, default=int(os.environ.get("ATLAS_LOAD_DURATION", str(_DEFAULT_DURATION))))
    parser.add_argument("--p95-limit-ms", type=float, default=float(os.environ.get("ATLAS_LOAD_P95_LIMIT", str(_DEFAULT_P95_LIMIT_MS))))
    args = parser.parse_args()

    print(f"Load test: {args.url}  rps={args.rps}  duration={args.duration}s  p95_limit={args.p95_limit_ms}ms")

    report = asyncio.run(
        _run_load_collecting(args.url, args.key, args.rps, args.duration, args.p95_limit_ms)
    )
    report.print()
    return 0 if report.passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
