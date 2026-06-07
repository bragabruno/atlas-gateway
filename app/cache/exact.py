"""GW-13 — Exact-match response cache over Redis.

A thin capability adapter (ADR-016): it composes a collision-proof cache key
from the request and offers `get`/`set` with a TTL. No business logic — the
service layer decides when to consult the cache and what to store.

Key composition (atlas-docs/03 §3.2 tenant-isolation rule) namespaces every
entry by:

- `tenant_id` — the `api_keys.id` of the caller; a hit can never cross tenants.
- `prompt_version` — the `prompt_versions.id` (or a sentinel when a bare alias
  is used); a prompt promotion must not serve a stale response.
- `model` — the resolved model id; different models give different answers.

plus a SHA-256 digest of the *normalized* request messages, so semantically
identical requests collapse to one key while any of the three namespace fields
differing yields a different key. Pinned deps: redis 7.4.0, fakeredis 2.35.1
(dev/tests).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

import redis.asyncio as redis_async

from app.domain.messages import Message

#: Sentinel prompt-version segment used when the request is a bare alias (no
#: `prompt_ref`), so the key shape is uniform whether or not a prompt is bound.
NO_PROMPT_VERSION = "none"

#: Key namespace prefix; keeps gateway cache entries distinct from any other
#: keyspace sharing the Redis instance (rate-limit buckets, circuit breakers).
_KEY_PREFIX = "atlas:cache:exact"


def _normalize(messages: Sequence[Message]) -> str:
    """Canonical JSON for the message list — stable across dict ordering."""
    return json.dumps(
        [{"role": m.role, "content": m.content} for m in messages],
        separators=(",", ":"),
        ensure_ascii=False,
    )


def build_key(
    *,
    tenant_id: str,
    model: str,
    messages: Sequence[Message],
    prompt_version: str = NO_PROMPT_VERSION,
) -> str:
    """Compose the namespaced, collision-proof cache key for a request.

    The tenant, prompt-version, and model are kept as readable key segments
    (cheap to scan/debug); the request body is hashed. Any difference in
    tenant, version, model, or message content produces a different key.
    """
    digest = hashlib.sha256(_normalize(messages).encode("utf-8")).hexdigest()
    return f"{_KEY_PREFIX}:{tenant_id}:{prompt_version}:{model}:{digest}"


class ExactCache:
    """Exact-match cache backed by a `redis.asyncio` client.

    The client is injected (composition root / tests supply a real client or a
    `fakeredis` fake), so this adapter does no connection management beyond the
    calls it makes. The client MUST be configured with ``decode_responses=True``
    so reads come back as ``str`` (matching `get`'s return type). Values are
    opaque serialized response strings; the caller owns serialization.
    """

    def __init__(self, client: redis_async.Redis, *, ttl_seconds: int = 3600) -> None:
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds}")
        self._client = client
        self._ttl_seconds = ttl_seconds

    async def get(
        self,
        *,
        tenant_id: str,
        model: str,
        messages: Sequence[Message],
        prompt_version: str = NO_PROMPT_VERSION,
    ) -> str | None:
        """Return the cached value for this request, or `None` on a miss."""
        key = build_key(
            tenant_id=tenant_id,
            model=model,
            messages=messages,
            prompt_version=prompt_version,
        )
        value: str | None = await self._client.get(key)
        return value

    async def set(
        self,
        *,
        tenant_id: str,
        model: str,
        messages: Sequence[Message],
        value: str,
        prompt_version: str = NO_PROMPT_VERSION,
    ) -> None:
        """Store `value` for this request under the configured TTL."""
        key = build_key(
            tenant_id=tenant_id,
            model=model,
            messages=messages,
            prompt_version=prompt_version,
        )
        await self._client.set(key, value, ex=self._ttl_seconds)
