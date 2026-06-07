"""GW-12 — Redis-backed per-provider circuit breaker + failover (ADR-011).

Pins the breaker's state machine and failover with ``fakeredis`` (zero network):

- repeated failures trip CLOSED → OPEN at the threshold; an open breaker rejects
  fast with the spec ``CircuitOpenError`` and a positive ``retry_after``;
- a success resets the consecutive-failure counter (a near-threshold run does not
  trip once interrupted by a success);
- after the cooldown elapses the breaker reports HALF-OPEN; a successful probe
  closes it (recovery), a failed probe re-opens it for a fresh cooldown;
- ``choose`` selects the primary when healthy, fails over to the fallback when
  the primary is OPEN, and raises ``AllProvidersUnavailable`` when neither can
  serve;
- state is shared across breaker instances over the same Redis (the replica
  model) and tolerates ``bytes`` responses;
- tuning parameters are validated.

The cooldown is advanced deterministically by patching the module clock — no real
sleeping. See ADR-011 + atlas-docs/02 §ADR-011.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest import mock

import pytest
import pytest_asyncio
from fakeredis.aioredis import FakeRedis

import app.resilience.circuit_breaker as cb_mod
from app.resilience.circuit_breaker import (
    AllProvidersUnavailable,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
)


@pytest_asyncio.fixture
async def client() -> AsyncIterator[FakeRedis]:
    fake = FakeRedis(decode_responses=True)
    try:
        yield fake
    finally:
        await fake.aclose()


async def _fail_n(breaker: CircuitBreaker, provider: str, n: int) -> None:
    for _ in range(n):
        await breaker.record_failure(provider)


# ── opening on repeated failures ─────────────────────────────────────────────


async def test_starts_closed(client: FakeRedis) -> None:
    breaker = CircuitBreaker(client, failure_threshold=3, cooldown_seconds=30)
    assert await breaker.state("openai") is CircuitState.CLOSED
    # A fresh breaker admits.
    await breaker.allow("openai")


async def test_repeated_failures_open_the_breaker(client: FakeRedis) -> None:
    breaker = CircuitBreaker(client, failure_threshold=3, cooldown_seconds=30)
    # Below threshold: still closed and admitting.
    await _fail_n(breaker, "openai", 2)
    assert await breaker.state("openai") is CircuitState.CLOSED
    await breaker.allow("openai")

    # The threshold-th failure trips it open.
    await breaker.record_failure("openai")
    assert await breaker.state("openai") is CircuitState.OPEN


async def test_open_breaker_rejects_fast(client: FakeRedis) -> None:
    breaker = CircuitBreaker(client, failure_threshold=2, cooldown_seconds=30)
    await _fail_n(breaker, "openai", 2)
    with pytest.raises(CircuitOpenError) as exc_info:
        await breaker.allow("openai")
    err = exc_info.value
    assert err.provider == "openai"
    assert err.retry_after >= 1


async def test_success_resets_failure_count(client: FakeRedis) -> None:
    breaker = CircuitBreaker(client, failure_threshold=3, cooldown_seconds=30)
    await _fail_n(breaker, "openai", 2)
    # A success clears the run; the breaker must not trip on the next single fail.
    await breaker.record_success("openai")
    await breaker.record_failure("openai")
    assert await breaker.state("openai") is CircuitState.CLOSED
    await breaker.allow("openai")


# ── half-open recovery ───────────────────────────────────────────────────────


async def test_cooldown_elapsed_reports_half_open(client: FakeRedis) -> None:
    breaker = CircuitBreaker(client, failure_threshold=1, cooldown_seconds=10)
    await breaker.record_failure("openai")
    assert await breaker.state("openai") is CircuitState.OPEN

    real_time = cb_mod.time.time
    with mock.patch.object(cb_mod.time, "time", lambda: real_time() + 11):
        assert await breaker.state("openai") is CircuitState.HALF_OPEN
        # The probe is admitted (no CircuitOpenError) once cooling has elapsed.
        await breaker.allow("openai")


async def test_half_open_success_closes_breaker(client: FakeRedis) -> None:
    breaker = CircuitBreaker(client, failure_threshold=1, cooldown_seconds=10)
    await breaker.record_failure("openai")

    real_time = cb_mod.time.time
    with mock.patch.object(cb_mod.time, "time", lambda: real_time() + 11):
        await breaker.allow("openai")  # admit the probe
        await breaker.record_success("openai")  # probe succeeds → recover

    # Back to CLOSED with a clean count, at the real clock.
    assert await breaker.state("openai") is CircuitState.CLOSED
    await breaker.allow("openai")


async def test_half_open_failure_reopens_breaker(client: FakeRedis) -> None:
    breaker = CircuitBreaker(client, failure_threshold=1, cooldown_seconds=10)
    await breaker.record_failure("openai")

    real_time = cb_mod.time.time
    with mock.patch.object(cb_mod.time, "time", lambda: real_time() + 11):
        await breaker.allow("openai")  # admit the probe
        await breaker.record_failure("openai")  # probe fails → re-open
        # Re-opened for a fresh cooldown at the advanced clock; rejects fast.
        with pytest.raises(CircuitOpenError):
            await breaker.allow("openai")


async def test_open_is_still_open_before_cooldown(client: FakeRedis) -> None:
    breaker = CircuitBreaker(client, failure_threshold=1, cooldown_seconds=30)
    await breaker.record_failure("openai")
    real_time = cb_mod.time.time
    # Only 5s elapsed of a 30s cooldown → still OPEN, still rejecting.
    with mock.patch.object(cb_mod.time, "time", lambda: real_time() + 5):
        assert await breaker.state("openai") is CircuitState.OPEN
        with pytest.raises(CircuitOpenError):
            await breaker.allow("openai")


# ── failover selection (choose) ──────────────────────────────────────────────


async def test_choose_picks_primary_when_healthy(client: FakeRedis) -> None:
    breaker = CircuitBreaker(client, failure_threshold=2, cooldown_seconds=30)
    assert await breaker.choose("anthropic", "openai") == "anthropic"


async def test_choose_fails_over_when_primary_open(client: FakeRedis) -> None:
    breaker = CircuitBreaker(client, failure_threshold=2, cooldown_seconds=30)
    await _fail_n(breaker, "anthropic", 2)  # primary open
    assert await breaker.choose("anthropic", "openai") == "openai"


async def test_choose_raises_when_all_open(client: FakeRedis) -> None:
    breaker = CircuitBreaker(client, failure_threshold=1, cooldown_seconds=30)
    await breaker.record_failure("anthropic")
    await breaker.record_failure("openai")
    with pytest.raises(AllProvidersUnavailable) as exc_info:
        await breaker.choose("anthropic", "openai")
    err = exc_info.value
    assert err.providers == ("anthropic", "openai")
    assert err.retry_after >= 1


async def test_choose_single_candidate_raises_when_open(client: FakeRedis) -> None:
    # An alias with no separate fallback (primary == fallback): one candidate.
    breaker = CircuitBreaker(client, failure_threshold=1, cooldown_seconds=30)
    await breaker.record_failure("anthropic")
    with pytest.raises(AllProvidersUnavailable) as exc_info:
        await breaker.choose("anthropic", "anthropic")
    assert exc_info.value.providers == ("anthropic",)


async def test_choose_fails_over_then_recovers_to_primary(client: FakeRedis) -> None:
    breaker = CircuitBreaker(client, failure_threshold=1, cooldown_seconds=10)
    await breaker.record_failure("anthropic")  # primary open → fail over
    assert await breaker.choose("anthropic", "openai") == "openai"

    real_time = cb_mod.time.time
    with mock.patch.object(cb_mod.time, "time", lambda: real_time() + 11):
        # Cooldown elapsed → primary is HALF-OPEN (not OPEN) → chosen again.
        assert await breaker.choose("anthropic", "openai") == "anthropic"


# ── replica model + encoding tolerance ───────────────────────────────────────


async def test_state_is_shared_across_instances(client: FakeRedis) -> None:
    # Two breaker objects over one Redis = two replicas sharing the state.
    replica_a = CircuitBreaker(client, failure_threshold=2, cooldown_seconds=30)
    replica_b = CircuitBreaker(client, failure_threshold=2, cooldown_seconds=30)
    await _fail_n(replica_a, "openai", 2)
    # The other replica sees the OPEN breaker without observing the failures.
    assert await replica_b.state("openai") is CircuitState.OPEN
    with pytest.raises(CircuitOpenError):
        await replica_b.allow("openai")


async def test_providers_are_isolated(client: FakeRedis) -> None:
    breaker = CircuitBreaker(client, failure_threshold=1, cooldown_seconds=30)
    await breaker.record_failure("openai")
    assert await breaker.state("openai") is CircuitState.OPEN
    # A different provider has its own breaker, untouched.
    assert await breaker.state("anthropic") is CircuitState.CLOSED
    await breaker.allow("anthropic")


async def test_works_with_bytes_responses_too() -> None:
    # decode_responses=False → hash fields come back as bytes; parse them too.
    fake = FakeRedis()
    try:
        breaker = CircuitBreaker(fake, failure_threshold=2, cooldown_seconds=30)
        await _fail_n(breaker, "openai", 2)
        assert await breaker.state("openai") is CircuitState.OPEN
        with pytest.raises(CircuitOpenError):
            await breaker.allow("openai")
    finally:
        await fake.aclose()


# ── parameter validation ─────────────────────────────────────────────────────


def test_rejects_non_positive_threshold(client: FakeRedis) -> None:
    with pytest.raises(ValueError, match="failure_threshold must be >= 1"):
        CircuitBreaker(client, failure_threshold=0, cooldown_seconds=30)


def test_rejects_non_positive_cooldown(client: FakeRedis) -> None:
    with pytest.raises(ValueError, match="cooldown_seconds must be >= 1"):
        CircuitBreaker(client, failure_threshold=1, cooldown_seconds=0)


def test_exposes_tuning_parameters(client: FakeRedis) -> None:
    breaker = CircuitBreaker(client, failure_threshold=7, cooldown_seconds=42)
    assert breaker.failure_threshold == 7
    assert breaker.cooldown_seconds == 42
