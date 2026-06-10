"""POL-1 — Semantic response cache backed by Qdrant.

Uses cosine-similarity search (threshold 0.97) to serve cached responses to
paraphrase requests that are semantically equivalent.

Design invariants
-----------------
- Tenant-scoped: each tenant gets its own Qdrant collection
  (``atlas_cache_<tenant_id>``).  A cached response can never cross tenants.
- Cited-answer exclusion: responses that contain source citations are never
  stored in the semantic cache (they carry doc-specific provenance that must
  not be served to different inputs).
- Opt-in per route: the caller passes ``use_semantic_cache=True``; routes that
  must not cache (e.g. streaming with citations) omit it.

Metrics (OTel)
--------------
- ``atlas.cache.semantic.hits``  — counter, attr ``tenant_id``
- ``atlas.cache.semantic.misses`` — counter, attr ``tenant_id``

Qdrant payload schema per point
--------------------------------
    {
      "model":    "<model alias>",
      "response": "<serialised response string>",
      "is_cited": false
    }

The ``id`` of each point is a deterministic UUID-v5 derived from
``sha256(tenant_id + model + normalised_messages)`` so identical requests
write to the same point (idempotent upserts).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from opentelemetry import metrics

from app.domain.messages import Message

_COLLECTION_PREFIX = "atlas_cache"
_SIMILARITY_THRESHOLD = 0.97
_METRIC_HITS = "atlas.cache.semantic.hits"
_METRIC_MISSES = "atlas.cache.semantic.misses"
_SCOPE = "atlas.gateway.cache.semantic"

# Namespace UUID for deterministic point IDs (atlas-specific).
_NS = uuid.UUID("d0c5cc1a-b44b-4b6d-9f3a-7a2e3f5c8d9e")

EmbedFn = Callable[[str], Awaitable[list[float]]]


def _normalize(messages: Sequence[Message]) -> str:
    return json.dumps(
        [{"role": m.role, "content": m.content} for m in messages],
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _collection(tenant_id: str) -> str:
    safe = tenant_id.replace("-", "_").replace(":", "_")
    return f"{_COLLECTION_PREFIX}_{safe}"


def _point_id(tenant_id: str, model: str, messages: Sequence[Message]) -> str:
    raw = f"{tenant_id}:{model}:{_normalize(messages)}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return str(uuid.uuid5(_NS, digest))


class SemanticCache:
    """Semantic response cache backed by Qdrant async client.

    Parameters
    ----------
    client:
        An ``qdrant_client.AsyncQdrantClient`` instance.
    vector_size:
        Dimensionality of the embedding vectors.  Must match the model used
        by the ``embed_fn`` at call sites.
    similarity_threshold:
        Minimum cosine similarity for a cache hit (default 0.97).
    ttl_seconds:
        Not enforced by Qdrant itself (no TTL support on free tier); callers
        that need TTL should use a periodic cleanup job.  Stored in payload
        for auditing.
    meter_provider:
        Optional OTel MeterProvider; falls back to the global provider.
    """

    def __init__(
        self,
        client: Any,  # qdrant_client.AsyncQdrantClient — avoid hard import at module level
        *,
        vector_size: int = 1536,
        similarity_threshold: float = _SIMILARITY_THRESHOLD,
        meter_provider: Any | None = None,
    ) -> None:
        self._client = client
        self._vector_size = vector_size
        self._threshold = similarity_threshold

        mp = meter_provider or metrics.get_meter_provider()
        meter = mp.get_meter(_SCOPE)
        self._hits = meter.create_counter(_METRIC_HITS, unit="1")
        self._misses = meter.create_counter(_METRIC_MISSES, unit="1")

    async def _ensure_collection(self, collection: str) -> None:
        """Create the Qdrant collection if it does not yet exist."""
        from qdrant_client.models import Distance, VectorParams  # type: ignore[import-untyped]

        existing = {c.name for c in (await self._client.get_collections()).collections}
        if collection not in existing:
            await self._client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
            )

    async def get(
        self,
        *,
        tenant_id: str,
        model: str,
        messages: Sequence[Message],
        embed_fn: EmbedFn,
        use_semantic_cache: bool = True,
    ) -> str | None:
        """Return a cached response string on a semantic hit, or None on miss.

        Parameters
        ----------
        use_semantic_cache:
            Set to False to bypass the cache entirely for this request.
        embed_fn:
            Async callable that takes a string and returns an embedding vector.
        """
        if not use_semantic_cache:
            return None

        collection = _collection(tenant_id)
        query_text = _normalize(messages)
        vector = await embed_fn(query_text)

        await self._ensure_collection(collection)

        results = await self._client.search(
            collection_name=collection,
            query_vector=vector,
            limit=1,
            score_threshold=self._threshold,
            with_payload=True,
        )

        if results:
            hit = results[0]
            response: str = hit.payload["response"]  # type: ignore[index]
            self._hits.add(1, {"tenant_id": tenant_id})
            return response

        self._misses.add(1, {"tenant_id": tenant_id})
        return None

    async def set(
        self,
        *,
        tenant_id: str,
        model: str,
        messages: Sequence[Message],
        response: str,
        embed_fn: EmbedFn,
        is_cited: bool = False,
    ) -> None:
        """Store a response in the semantic cache.

        Cited answers (``is_cited=True``) are skipped — they carry
        document-specific provenance that must not be served to other inputs.
        """
        if is_cited:
            return

        collection = _collection(tenant_id)
        vector = await embed_fn(_normalize(messages))

        await self._ensure_collection(collection)

        from qdrant_client.models import PointStruct  # type: ignore[import-untyped]

        point_id = _point_id(tenant_id, model, messages)
        await self._client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"model": model, "response": response, "is_cited": False},
                )
            ],
        )
