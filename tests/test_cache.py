"""GW-13 — exact-match Redis cache.

Pins the adapter contract with `fakeredis` (zero network): set then get is a
hit, an unseen request is a miss, an entry past its TTL is gone, and any
difference in tenant / prompt-version / model yields a distinct key (no
cross-tenant or cross-version collisions). TTL expiry is exercised
deterministically by advancing fakeredis's clock — no real sleeping.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from unittest import mock

import pytest
import pytest_asyncio
from fakeredis.aioredis import FakeRedis

from app.cache.exact import NO_PROMPT_VERSION, ExactCache, build_key
from app.domain.messages import Message

_MESSAGES = [Message(role="user", content="what is the capital of France?")]
_VALUE = '{"content": "Paris"}'


@pytest_asyncio.fixture
async def client() -> AsyncIterator[FakeRedis]:
    # decode_responses=True so values come back as str, matching the cache's
    # str contract (the composition root configures the real client the same way).
    fake = FakeRedis(decode_responses=True)
    try:
        yield fake
    finally:
        await fake.aclose()


async def test_set_then_get_is_a_hit(client: FakeRedis) -> None:
    cache = ExactCache(client)
    await cache.set(tenant_id="t1", model="smart", messages=_MESSAGES, value=_VALUE)
    hit = await cache.get(tenant_id="t1", model="smart", messages=_MESSAGES)
    assert hit == _VALUE


async def test_unseen_request_is_a_miss(client: FakeRedis) -> None:
    cache = ExactCache(client)
    miss = await cache.get(tenant_id="t1", model="smart", messages=_MESSAGES)
    assert miss is None


async def test_entry_expires_after_ttl(client: FakeRedis) -> None:
    cache = ExactCache(client, ttl_seconds=10)
    await cache.set(tenant_id="t1", model="smart", messages=_MESSAGES, value=_VALUE)
    assert await cache.get(tenant_id="t1", model="smart", messages=_MESSAGES) == _VALUE

    # fakeredis derives key expiry from time.time(); advancing it past the TTL
    # makes the entry expire deterministically with no real sleeping.
    real_time = time.time
    with mock.patch("time.time", lambda: real_time() + 11):
        expired = await cache.get(tenant_id="t1", model="smart", messages=_MESSAGES)
    assert expired is None


async def test_different_tenant_yields_different_key() -> None:
    a = build_key(tenant_id="t1", model="smart", messages=_MESSAGES)
    b = build_key(tenant_id="t2", model="smart", messages=_MESSAGES)
    assert a != b


async def test_different_model_yields_different_key() -> None:
    a = build_key(tenant_id="t1", model="smart", messages=_MESSAGES)
    b = build_key(tenant_id="t1", model="deep", messages=_MESSAGES)
    assert a != b


async def test_different_prompt_version_yields_different_key() -> None:
    a = build_key(tenant_id="t1", model="smart", messages=_MESSAGES, prompt_version="v1")
    b = build_key(tenant_id="t1", model="smart", messages=_MESSAGES, prompt_version="v2")
    assert a != b
    assert a != build_key(tenant_id="t1", model="smart", messages=_MESSAGES)


async def test_different_tenant_does_not_read_anothers_entry(client: FakeRedis) -> None:
    cache = ExactCache(client)
    await cache.set(tenant_id="t1", model="smart", messages=_MESSAGES, value=_VALUE)
    cross = await cache.get(tenant_id="t2", model="smart", messages=_MESSAGES)
    assert cross is None


async def test_same_request_same_key_is_stable() -> None:
    a = build_key(tenant_id="t1", model="smart", messages=_MESSAGES)
    b = build_key(
        tenant_id="t1",
        model="smart",
        messages=[Message(role="user", content="what is the capital of France?")],
    )
    assert a == b


async def test_default_prompt_version_segment_is_the_sentinel() -> None:
    key = build_key(tenant_id="t1", model="smart", messages=_MESSAGES)
    assert f":{NO_PROMPT_VERSION}:" in key


def test_non_positive_ttl_is_rejected(client: FakeRedis) -> None:
    with pytest.raises(ValueError, match="ttl_seconds must be positive"):
        ExactCache(client, ttl_seconds=0)
