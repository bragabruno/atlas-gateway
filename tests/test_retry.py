"""GW-11 — per-call retry policy (tenacity).

Pins the retry *classification* and termination, not wall-clock timing:

- an injected transient failure (:class:`TransientProviderError`) is retried and
  then succeeds, with exactly the expected number of attempts;
- a non-retryable error surfaces immediately on the first attempt (no retries);
- after ``max_attempts`` the last transient error is re-raised unchanged (not a
  tenacity ``RetryError`` wrapper);
- the tuning parameters are validated.

Backoff is configured tiny (sub-millisecond) so the suite stays fast and offline
— the policy's *behaviour* is what is asserted, not the jitter durations.
"""

from __future__ import annotations

import pytest

from app.resilience.retry import (
    TransientProviderError,
    build_async_retrying,
    build_retrying,
)

# Sub-millisecond backoff: exercises the policy without real waiting. Kept as a
# typed tuple (not a splatted dict) so the call sites stay precisely typed.
_FAST_BASE_DELAY = 0.001
_FAST_MAX_DELAY = 0.002


class _Counter:
    """Counts invocations so attempt counts can be asserted exactly."""

    def __init__(self) -> None:
        self.n = 0

    def bump(self) -> int:
        self.n += 1
        return self.n


# ── async path: the one provider calls use ──────────────────────────────────


async def test_transient_failure_retries_then_succeeds() -> None:
    counter = _Counter()

    async def flaky() -> str:
        if counter.bump() < 3:
            raise TransientProviderError("upstream 503")
        return "ok"

    result: str = await build_async_retrying(
        max_attempts=5, base_delay=_FAST_BASE_DELAY, max_delay=_FAST_MAX_DELAY
    )(flaky)
    assert result == "ok"
    assert counter.n == 3  # two transient failures, then success


async def test_non_retryable_surfaces_immediately() -> None:
    counter = _Counter()

    async def permanent() -> str:
        counter.bump()
        raise ValueError("bad request")

    with pytest.raises(ValueError, match="bad request"):
        await build_async_retrying(
            max_attempts=5, base_delay=_FAST_BASE_DELAY, max_delay=_FAST_MAX_DELAY
        )(permanent)
    assert counter.n == 1  # not retried — surfaced on the first attempt


async def test_last_transient_error_is_reraised_after_max_attempts() -> None:
    counter = _Counter()

    async def always_transient() -> str:
        n = counter.bump()
        raise TransientProviderError(f"attempt {n}")

    # reraise=True → the real error, not a tenacity RetryError wrapper.
    with pytest.raises(TransientProviderError, match="attempt 3"):
        await build_async_retrying(
            max_attempts=3, base_delay=_FAST_BASE_DELAY, max_delay=_FAST_MAX_DELAY
        )(always_transient)
    assert counter.n == 3


async def test_succeeds_on_first_attempt_without_retrying() -> None:
    counter = _Counter()

    async def healthy() -> int:
        counter.bump()
        return 42

    assert (
        await build_async_retrying(base_delay=_FAST_BASE_DELAY, max_delay=_FAST_MAX_DELAY)(healthy)
        == 42
    )
    assert counter.n == 1


async def test_single_attempt_policy_does_not_retry() -> None:
    counter = _Counter()

    async def flaky() -> str:
        counter.bump()
        raise TransientProviderError("once")

    with pytest.raises(TransientProviderError, match="once"):
        await build_async_retrying(
            max_attempts=1, base_delay=_FAST_BASE_DELAY, max_delay=_FAST_MAX_DELAY
        )(flaky)
    assert counter.n == 1


async def test_subclass_of_transient_is_retried() -> None:
    class ProviderTimeout(TransientProviderError):
        pass

    counter = _Counter()

    async def flaky() -> str:
        if counter.bump() < 2:
            raise ProviderTimeout("read timeout")
        return "ok"

    assert (
        await build_async_retrying(
            max_attempts=3, base_delay=_FAST_BASE_DELAY, max_delay=_FAST_MAX_DELAY
        )(flaky)
        == "ok"
    )
    assert counter.n == 2


# ── sync path: same classification, shared policy ───────────────────────────


def test_sync_transient_failure_retries_then_succeeds() -> None:
    counter = _Counter()

    def flaky() -> str:
        if counter.bump() < 2:
            raise TransientProviderError("blip")
        return "ok"

    assert (
        build_retrying(max_attempts=3, base_delay=_FAST_BASE_DELAY, max_delay=_FAST_MAX_DELAY)(
            flaky
        )
        == "ok"
    )
    assert counter.n == 2


def test_sync_non_retryable_surfaces_immediately() -> None:
    counter = _Counter()

    def permanent() -> str:
        counter.bump()
        raise KeyError("missing")

    with pytest.raises(KeyError):
        build_retrying(max_attempts=5, base_delay=_FAST_BASE_DELAY, max_delay=_FAST_MAX_DELAY)(
            permanent
        )
    assert counter.n == 1


# ── parameter validation ────────────────────────────────────────────────────


def test_rejects_zero_max_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        build_async_retrying(max_attempts=0)


def test_rejects_non_positive_base_delay() -> None:
    with pytest.raises(ValueError, match="base_delay must be positive"):
        build_async_retrying(base_delay=0)


def test_rejects_max_delay_below_base_delay() -> None:
    with pytest.raises(ValueError, match="max_delay .* must be >= base_delay"):
        build_async_retrying(base_delay=2.0, max_delay=1.0)


def test_sync_builder_validates_parameters_too() -> None:
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        build_retrying(max_attempts=0)
