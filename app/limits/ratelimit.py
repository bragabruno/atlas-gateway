"""GW-16 — Per-key token-bucket rate limiting over Redis → 429.

A thin capability adapter (ADR-016): it maintains one token bucket per API key in
Redis and decides, atomically, whether a request may proceed. On exhaustion it
raises :class:`RateLimitExceeded`, whose body is the exact ``429`` shape the
gateway returns (atlas-docs/03 §5.2 *429 — Rate Limit Exceeded*); the controller
maps the error to the HTTP response with a ``Retry-After`` header (wiring later).

**Algorithm — token bucket.** A bucket holds up to ``capacity`` tokens and
refills continuously at ``refill_per_sec`` tokens/second. Each admitted request
costs one token. A request is allowed iff the bucket (after lazy refill for the
elapsed time) holds ≥ 1 token; otherwise it is denied and ``retry_after`` is the
time until the next whole token accrues. This smooths bursts up to ``capacity``
while bounding the sustained rate to ``refill_per_sec``.

**Atomicity (ADR-011).** ADR-011 specifies Redis Lua for atomic counter updates.
``lupa`` (the engine ``fakeredis`` needs to execute ``EVAL`` offline) is *not* in
the project's pinned deps, so this adapter uses Redis' other first-class atomic
primitive — an optimistic ``WATCH``/``MULTI``/``EXEC`` transaction — to get the
identical read-modify-write atomicity across replicas without a new dependency.
The check-and-decrement (read tokens+timestamp, refill, conditionally consume,
write back) executes as one transaction; a concurrent mutation aborts and the
client retries. State is stored as a Redis hash ``{tokens, ts}`` under a
key-namespaced bucket, with a TTL so idle buckets self-evict.

Pinned deps: redis 7.4.0, fakeredis 2.35.1 (dev/tests). See GW-16 + ADR-011 +
atlas-docs/03 §5.2.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import redis.asyncio as redis_async

from app.limits._redis_typing import TxnPipe, run_transaction

#: Key namespace prefix; keeps rate-limit buckets distinct from the cache and
#: circuit-breaker keyspaces sharing the Redis instance (see app/cache/exact.py).
_KEY_PREFIX = "atlas:ratelimit"

#: Hash fields holding the bucket's current level and last-refill epoch seconds.
_FIELD_TOKENS = "tokens"
_FIELD_TS = "ts"


@dataclass(frozen=True, slots=True)
class RateLimitError:
    """The ``error`` object of the ``429`` rate-limit body (atlas-docs/03 §5.2)."""

    code: str
    message: str
    type: str
    param: None = None


class RateLimitExceeded(Exception):
    """Raised when a key's bucket is empty — carries the spec ``429`` body.

    ``body`` is the exact JSON the gateway returns (``{"error": {...}}`` with
    ``code="rate_limit_exceeded"``, ``type="rate_limit_error"``); ``retry_after``
    is the integer seconds for the ``Retry-After`` header. The controller maps
    this to an HTTP 429 (wiring later); business logic stays HTTP-free (ADR-016).
    """

    def __init__(self, *, alias: str, retry_after: int) -> None:
        self.alias = alias
        self.retry_after = retry_after
        self.error = RateLimitError(
            code="rate_limit_exceeded",
            message=(
                f"Request rate limit reached for alias {alias!r}. "
                f"Retry after {retry_after} seconds."
            ),
            type="rate_limit_error",
        )
        super().__init__(self.error.message)

    @property
    def body(self) -> dict[str, object]:
        """The exact ``429`` response body (atlas-docs/03 §5.2)."""
        return {
            "error": {
                "code": self.error.code,
                "message": self.error.message,
                "type": self.error.type,
                "param": self.error.param,
            }
        }


@dataclass(slots=True)
class _Decision:
    """Mutable carrier for one admission decision out of the transaction closure.

    A typed (not ``list[object]``) holder threads the allow/deny result and the
    ``Retry-After`` seconds out of the transaction for the type checker.
    """

    allowed: bool
    retry_after: int


class TokenBucketRateLimiter:
    """Per-key token-bucket limiter backed by an injected ``redis.asyncio`` client.

    The client is injected (composition root supplies a real client; tests supply
    ``fakeredis``), so this adapter does no connection management. ``capacity`` is
    the burst size and ``refill_per_sec`` the sustained admit rate. The client may
    be configured with ``decode_responses=True`` (matching the rest of the
    gateway); this adapter reads hash fields tolerantly of ``str``/``bytes``.
    """

    def __init__(
        self,
        client: redis_async.Redis,
        *,
        capacity: int,
        refill_per_sec: float,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if refill_per_sec <= 0:
            raise ValueError(f"refill_per_sec must be positive, got {refill_per_sec}")
        self._client = client
        self._capacity = capacity
        self._refill_per_sec = refill_per_sec

    def _key(self, api_key_id: str) -> str:
        return f"{_KEY_PREFIX}:{api_key_id}"

    def _ttl_seconds(self) -> int:
        """TTL so an untouched bucket self-evicts once it would be full again."""
        return max(1, math.ceil(self._capacity / self._refill_per_sec))

    @staticmethod
    def _as_float(raw: str | bytes | None, default: float) -> float:
        """Read a hash field that may be ``str``, ``bytes``, or absent."""
        if raw is None:
            return default
        if isinstance(raw, bytes):
            return float(raw.decode())
        return float(raw)

    async def check(self, *, api_key_id: str, alias: str) -> None:
        """Admit one request, or raise :class:`RateLimitExceeded`.

        Atomically refills the bucket for the elapsed time and consumes one token
        if available. On exhaustion, raises with ``retry_after`` = whole seconds
        until the next token accrues (≥ 1) and the spec ``429`` body.
        """
        key = self._key(api_key_id)
        # Typed carrier threaded out of the transaction closure (no shared state).
        result = _Decision(allowed=True, retry_after=1)

        async def _consume(pipe: TxnPipe) -> None:
            # WATCH is implied by client.transaction(); read the current state,
            # then queue the conditional write inside MULTI for an atomic CAS.
            raw = await pipe.hmget(key, [_FIELD_TOKENS, _FIELD_TS])
            now = time.time()
            tokens = self._as_float(raw[0], float(self._capacity))
            last_ts = self._as_float(raw[1], now)

            # Lazy continuous refill, clamped to capacity.
            tokens = min(
                float(self._capacity),
                tokens + max(0.0, now - last_ts) * self._refill_per_sec,
            )

            if tokens < 1.0:
                # Denied: nothing consumed. Retry-After is the whole seconds until
                # the next token accrues. The refreshed timestamp is still written
                # so refill accounting stays exact across denied requests.
                deficit = 1.0 - tokens
                result.allowed = False
                result.retry_after = max(1, math.ceil(deficit / self._refill_per_sec))
            else:
                tokens -= 1.0
                result.allowed = True

            pipe.multi()
            pipe.hset(key, mapping={_FIELD_TOKENS: str(tokens), _FIELD_TS: str(now)})
            pipe.expire(key, self._ttl_seconds())

        # Optimistic-transaction loop: a concurrent write aborts EXEC and redis-py
        # re-runs the closure until it commits cleanly.
        await run_transaction(self._client, _consume, key)

        if not result.allowed:
            raise RateLimitExceeded(alias=alias, retry_after=result.retry_after)
