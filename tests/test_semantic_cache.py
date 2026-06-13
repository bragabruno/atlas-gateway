"""POL-1 — SemanticCache tests.

Qdrant client is mocked; no external service needed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from app.cache.semantic import SemanticCache, _collection, _normalize, _point_id
from app.domain.messages import Message

_MESSAGES = [Message(role="user", content="What does Article 6 require?")]
_TENANT = "tenant-abc"
_MODEL = "gpt-4o"
_RESPONSE = "Article 6 requires explicit consent."
_VECTOR = [0.1] * 1536


async def _embed(text: str) -> list[float]:
    return _VECTOR


def _make_cache(client: MagicMock) -> SemanticCache:
    reader = InMemoryMetricReader()
    mp = MeterProvider(metric_readers=[reader])
    return SemanticCache(client, vector_size=1536, meter_provider=mp)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_collection_name_is_tenant_scoped() -> None:
    assert _collection("t1") == "atlas_cache_t1"
    assert _collection("t1") != _collection("t2")


def test_normalize_is_stable() -> None:
    a = _normalize(_MESSAGES)
    b = _normalize(_MESSAGES)
    assert a == b


def test_point_id_differs_by_tenant() -> None:
    id1 = _point_id("t1", _MODEL, _MESSAGES)
    id2 = _point_id("t2", _MODEL, _MESSAGES)
    assert id1 != id2


def test_point_id_differs_by_model() -> None:
    id1 = _point_id(_TENANT, "gpt-4o", _MESSAGES)
    id2 = _point_id(_TENANT, "claude-sonnet-4-6", _MESSAGES)
    assert id1 != id2


# ---------------------------------------------------------------------------
# get() — cache hit
# ---------------------------------------------------------------------------

def _qdrant_modules() -> dict[str, MagicMock]:
    mock_models = MagicMock()
    mock_models.Distance.COSINE = "Cosine"
    return {
        "qdrant_client": MagicMock(),
        "qdrant_client.models": mock_models,
    }


@pytest.mark.asyncio
async def test_get_returns_cached_response_on_hit() -> None:
    client = AsyncMock()
    client.get_collections.return_value.collections = []
    hit = MagicMock()
    hit.payload = {"response": _RESPONSE}
    client.search.return_value = [hit]

    with patch.dict("sys.modules", _qdrant_modules()):
        cache = _make_cache(client)
        result = await cache.get(
            tenant_id=_TENANT, model=_MODEL, messages=_MESSAGES, embed_fn=_embed
        )

    assert result == _RESPONSE


@pytest.mark.asyncio
async def test_get_returns_none_on_miss() -> None:
    client = AsyncMock()
    client.get_collections.return_value.collections = []
    client.search.return_value = []

    with patch.dict("sys.modules", _qdrant_modules()):
        cache = _make_cache(client)
        result = await cache.get(
            tenant_id=_TENANT, model=_MODEL, messages=_MESSAGES, embed_fn=_embed
        )

    assert result is None


@pytest.mark.asyncio
async def test_get_bypassed_when_use_semantic_cache_false() -> None:
    client = AsyncMock()
    cache = _make_cache(client)
    result = await cache.get(
        tenant_id=_TENANT, model=_MODEL, messages=_MESSAGES,
        embed_fn=_embed, use_semantic_cache=False
    )
    assert result is None
    client.search.assert_not_called()


# ---------------------------------------------------------------------------
# set() — store and skip-cited
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_set_upserts_to_qdrant() -> None:
    client = AsyncMock()
    client.get_collections.return_value.collections = []

    with patch.dict("sys.modules", _qdrant_modules()):
        cache = _make_cache(client)
        await cache.set(
            tenant_id=_TENANT, model=_MODEL, messages=_MESSAGES,
            response=_RESPONSE, embed_fn=_embed, is_cited=False
        )

    client.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_set_skips_cited_answers() -> None:
    client = AsyncMock()
    cache = _make_cache(client)
    await cache.set(
        tenant_id=_TENANT, model=_MODEL, messages=_MESSAGES,
        response=_RESPONSE, embed_fn=_embed, is_cited=True
    )
    client.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_set_never_shares_across_tenants() -> None:
    """Two different tenants write to different collections."""
    client = AsyncMock()
    client.get_collections.return_value.collections = []

    collections_used: list[str] = []

    async def _mock_upsert(*, collection_name: str, **_: object) -> None:
        collections_used.append(collection_name)

    client.upsert.side_effect = _mock_upsert

    with patch.dict("sys.modules", _qdrant_modules()):
        cache = _make_cache(client)
        await cache.set(
            tenant_id="t1", model=_MODEL, messages=_MESSAGES,
            response=_RESPONSE, embed_fn=_embed
        )
        await cache.set(
            tenant_id="t2", model=_MODEL, messages=_MESSAGES,
            response=_RESPONSE, embed_fn=_embed
        )

    assert len(collections_used) == 2
    assert collections_used[0] != collections_used[1]
