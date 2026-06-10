"""POL-6 — Unit tests for the load-test report builder."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from load_test import RequestResult, _build_report  # type: ignore[import-untyped]


def _results(latencies: list[float], status: int = 200) -> list[RequestResult]:
    return [RequestResult(latency_ms=l, status=status) for l in latencies]


def test_report_passes_when_p95_under_limit() -> None:
    # 20 results all well under 50ms
    results = _results([10.0] * 20)
    report = _build_report(results, target_rps=20, duration_s=1, p95_limit_ms=50.0)
    assert report.passed
    assert report.successes == 20
    assert report.errors == 0
    assert report.p95_ms <= 50.0


def test_report_fails_when_p95_exceeds_limit() -> None:
    # 19 fast + 1 slow → p95 slow
    results = _results([10.0] * 19 + [200.0])
    report = _build_report(results, target_rps=20, duration_s=1, p95_limit_ms=50.0)
    assert not report.passed


def test_report_fails_on_any_error() -> None:
    results = _results([10.0] * 19) + [RequestResult(latency_ms=5.0, status=500)]
    report = _build_report(results, target_rps=20, duration_s=1, p95_limit_ms=50.0)
    assert not report.passed
    assert report.errors == 1


def test_report_empty_results() -> None:
    report = _build_report([], target_rps=20, duration_s=1, p95_limit_ms=50.0)
    assert not report.passed
    assert report.total_requests == 0


def test_percentile_calculations() -> None:
    latencies = list(range(1, 101))  # 1..100 ms
    results = _results(latencies)
    report = _build_report(results, target_rps=100, duration_s=1, p95_limit_ms=200.0)
    assert 49 <= report.p50_ms <= 51
    assert 94 <= report.p95_ms <= 96
    assert 98 <= report.p99_ms <= 100
    assert report.max_ms == 100
