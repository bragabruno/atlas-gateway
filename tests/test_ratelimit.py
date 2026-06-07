"""GW-16 — per-key token-bucket rate limiter (Redis → 429).

Pins the adapter contract with ``fakeredis`` (zero network):

- requests under the bucket capacity are admitted;
- once the bucket is empty a request is denied with the exact ``429`` body
  (atlas-docs/03 §5.2) and an integer ``Retry-After``;
- the bucket refills over time, so a denied key is admitted again after enough
  seconds elapse — exercised deterministically by advancing the limiter's clock
  (no real sleeping);
- buckets are isolated per API key, and bursts up to capacity are allowed;
- the tuning parameters are validated.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest import mock

import pytest
import pytest_asyncio
from fakeredis.aioredis import FakeRedis

from app.limits.ratelimit import RateLimitExceeded, TokenBucketRateLimiter


@pytest_asyncio.fixture
async def client() -> AsyncIterator[FakeRedis]:
    fake = FakeRedis(decode_responses=True)
    try:
        yield fake
    finally:
        await fake.aclose()


async def test_requests_under_capacity_are_allowed(client: FakeRedis) -> None:
    limiter = TokenBucketRateLimiter(client, capacity=3, refill_per_sec=1.0)
    # A fresh bucket starts full; capacity admits without raising.
    for _ in range(3):
        await limiter.check(api_key_id="k1", alias="smart")


async def test_over_limit_raises_spec_429_body(client: FakeRedis) -> None:
    limiter = TokenBucketRateLimiter(client, capacity=2, refill_per_sec=1.0)
    await limiter.check(api_key_id="k1", alias="smart")
    await limiter.check(api_key_id="k1", alias="smart")

    with pytest.raises(RateLimitExceeded) as exc_info:
        await limiter.check(api_key_id="k1", alias="smart")

    err = exc_info.value
    assert err.retry_after >= 1
    assert err.body == {
        "error": {
            "code": "rate_limit_exceeded",
            "message": (
                "Request rate limit reached for alias 'smart'. "
                f"Retry after {err.retry_after} seconds."
            ),
            "type": "rate_limit_error",
            "param": None,
        }
    }


async def test_retry_after_reflects_refill_rate(client: FakeRedis) -> None:
    # 0.5 tokens/sec → ~2s to accrue the next whole token after exhaustion.
    limiter = TokenBucketRateLimiter(client, capacity=1, refill_per_sec=0.5)
    await limiter.check(api_key_id="k1", alias="smart")
    with pytest.raises(RateLimitExceeded) as exc_info:
        await limiter.check(api_key_id="k1", alias="smart")
    assert exc_info.value.retry_after == 2


async def test_bucket_refills_over_time(client: FakeRedis) -> None:
    limiter = TokenBucketRateLimiter(client, capacity=1, refill_per_sec=1.0)
    await limiter.check(api_key_id="k1", alias="smart")

    # Immediately exhausted.
    with pytest.raises(RateLimitExceeded):
        await limiter.check(api_key_id="k1", alias="smart")

    # Advance the limiter's clock past one refill period; the lazy refill now
    # grants a whole token and the next request is admitted (no real sleeping).
    import app.limits.ratelimit as ratelimit_mod

    real_time = ratelimit_mod.time.time
    with mock.patch.object(ratelimit_mod.time, "time", lambda: real_time() + 1.5):
        await limiter.check(api_key_id="k1", alias="smart")


async def test_refill_is_capped_at_capacity(client: FakeRedis) -> None:
    limiter = TokenBucketRateLimiter(client, capacity=2, refill_per_sec=1.0)
    await limiter.check(api_key_id="k1", alias="smart")

    import app.limits.ratelimit as ratelimit_mod

    real_time = ratelimit_mod.time.time
    # Wait far longer than needed to refill — tokens must clamp at capacity (2),
    # so exactly two requests are admitted, not the elapsed-seconds' worth.
    with mock.patch.object(ratelimit_mod.time, "time", lambda: real_time() + 100):
        await limiter.check(api_key_id="k1", alias="smart")
        await limiter.check(api_key_id="k1", alias="smart")
        with pytest.raises(RateLimitExceeded):
            await limiter.check(api_key_id="k1", alias="smart")


async def test_buckets_are_isolated_per_key(client: FakeRedis) -> None:
    limiter = TokenBucketRateLimiter(client, capacity=1, refill_per_sec=1.0)
    await limiter.check(api_key_id="k1", alias="smart")
    # k1 is exhausted, but k2 has its own full bucket.
    await limiter.check(api_key_id="k2", alias="smart")
    with pytest.raises(RateLimitExceeded):
        await limiter.check(api_key_id="k1", alias="smart")


async def test_alias_is_echoed_in_the_error(client: FakeRedis) -> None:
    limiter = TokenBucketRateLimiter(client, capacity=1, refill_per_sec=1.0)
    await limiter.check(api_key_id="k1", alias="deep")
    with pytest.raises(RateLimitExceeded) as exc_info:
        await limiter.check(api_key_id="k1", alias="deep")
    assert exc_info.value.alias == "deep"
    assert "'deep'" in exc_info.value.error.message


async def test_works_with_bytes_responses_too() -> None:
    # decode_responses=False → hash fields come back as bytes; the adapter must
    # parse them just the same.
    fake = FakeRedis()
    try:
        limiter = TokenBucketRateLimiter(fake, capacity=1, refill_per_sec=1.0)
        await limiter.check(api_key_id="k1", alias="smart")
        with pytest.raises(RateLimitExceeded):
            await limiter.check(api_key_id="k1", alias="smart")
    finally:
        await fake.aclose()


def test_rejects_non_positive_capacity(client: FakeRedis) -> None:
    with pytest.raises(ValueError, match="capacity must be >= 1"):
        TokenBucketRateLimiter(client, capacity=0, refill_per_sec=1.0)


def test_rejects_non_positive_refill(client: FakeRedis) -> None:
    with pytest.raises(ValueError, match="refill_per_sec must be positive"):
        TokenBucketRateLimiter(client, capacity=1, refill_per_sec=0.0)
