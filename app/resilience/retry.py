"""GW-11 — Per-call retry with exponential backoff + jitter (tenacity).

A thin capability adapter (ADR-016): it builds a ``tenacity`` retry policy that
the service/provider layer wraps around a single upstream call. Per ADR-011,
``tenacity`` owns *per-call* retry (short-lived, same request) with exponential
backoff + jitter; sustained failure is the circuit breaker's job, not this
module's.

Retry classification is explicit (fail-fast on anything non-transient):

- **Retryable** — only :class:`TransientProviderError`. Provider adapters raise
  it for transient upstream conditions (HTTP 429/5xx, connection resets, read
  timeouts) where re-issuing the *same* request can plausibly succeed.
- **Non-retryable** — every other exception (bad request, auth failure,
  :class:`~app.domain.errors.UnknownModelError`, programming bugs) surfaces
  immediately on the first attempt; retrying cannot help and would only add
  latency and spend.

Backoff is ``base_delay * 2**(attempt-1)`` capped at ``max_delay``, with full
jitter (``tenacity.wait_random_exponential``) to de-correlate retries across the
fleet and avoid a synchronized thundering herd. After ``max_attempts`` the last
:class:`TransientProviderError` is re-raised unchanged (``reraise=True``), so the
caller sees the real provider error rather than a ``RetryError`` wrapper.

Pinned dep: tenacity 9.1.4. See GW-11 + ADR-011 + atlas-docs/02 §ADR-011.
"""

from __future__ import annotations

from typing import TypedDict

from tenacity import (
    AsyncRetrying,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tenacity.retry import retry_base
from tenacity.stop import stop_base
from tenacity.wait import wait_base

#: Default number of attempts (1 initial + retries) before the last transient
#: error is re-raised. Kept small: per-call retry is for blips, not outages.
DEFAULT_MAX_ATTEMPTS = 3

#: Default first-retry backoff ceiling in seconds (the exponential's unit).
DEFAULT_BASE_DELAY = 0.5

#: Default upper bound on any single backoff wait in seconds.
DEFAULT_MAX_DELAY = 8.0


class TransientProviderError(Exception):
    """A transient upstream failure that a retry of the same request may fix.

    Provider adapters raise this (wrapping the underlying cause) for conditions
    that are safe and worthwhile to retry — upstream 429/5xx, connection resets,
    read timeouts. Any error that is *not* this type is treated as permanent and
    surfaces on the first attempt. The originating exception is preserved via
    ``raise TransientProviderError(...) from cause`` at the raising site.
    """


class _RetryKwargs(TypedDict):
    """The shared, precisely-typed kwargs both retrying objects accept."""

    retry: retry_base
    stop: stop_base
    wait: wait_base
    reraise: bool


def _build_kwargs(
    *,
    max_attempts: int,
    base_delay: float,
    max_delay: float,
) -> _RetryKwargs:
    """Shared (a)sync retrying kwargs; validates the tuning parameters."""
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    if base_delay <= 0:
        raise ValueError(f"base_delay must be positive, got {base_delay}")
    if max_delay < base_delay:
        raise ValueError(f"max_delay ({max_delay}) must be >= base_delay ({base_delay})")
    return _RetryKwargs(
        # Only TransientProviderError is retried; everything else propagates now.
        retry=retry_if_exception_type(TransientProviderError),
        stop=stop_after_attempt(max_attempts),
        # Full-jitter exponential backoff: wait ~U(0, min(max_delay, base*2**n)).
        wait=wait_random_exponential(multiplier=base_delay, max=max_delay),
        # Re-raise the last transient error itself, not a RetryError wrapper.
        reraise=True,
    )


def build_async_retrying(
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
) -> AsyncRetrying:
    """Build an :class:`~tenacity.AsyncRetrying` for async provider calls.

    Use as ``async for attempt in build_async_retrying(): with attempt: ...`` or
    via ``await build_async_retrying()(coro_fn, *args)``. Only
    :class:`TransientProviderError` is retried; the last one is re-raised after
    ``max_attempts``.
    """
    return AsyncRetrying(
        **_build_kwargs(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
        )
    )


def build_retrying(
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
) -> Retrying:
    """Build a synchronous :class:`~tenacity.Retrying` with the same policy.

    Provided for completeness so non-async call sites (if any) share the exact
    retry classification and backoff. Async provider calls should use
    :func:`build_async_retrying`.
    """
    return Retrying(
        **_build_kwargs(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
        )
    )
