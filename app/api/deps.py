"""FastAPI dependency providers — the DI wiring for the API layer.

Controllers declare what they need via `Depends(...)` instead of constructing
collaborators, so auth, settings, the provider registry, and services are wired
in one place (the composition root for the request scope). See ADR-016.

Capability collaborators are **config-gated and default OFF** (`app.config`):
each is constructed only when its backing config is present. With the defaults
(no `ATLAS_REDIS_URL`, every `*_enabled` flag off) `get_chat_service` returns a
Mock-only `ChatService(registry)` with no external dependencies — exactly the
pre-wiring behaviour the default and test environments rely on. There is no
Redis/DB/Kafka in those environments, so nothing here may require one to build a
working service.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from functools import lru_cache
from typing import Annotated

import redis.asyncio as redis_async
from fastapi import Depends, Header, HTTPException

from app.cache.exact import ExactCache
from app.config import Settings, get_settings
from app.guardrails.chain import GuardrailChain
from app.guardrails.injection import InjectionGuardrail
from app.guardrails.pii import PiiGuardrail
from app.guardrails.size import SizeGuardrail
from app.limits._redis_typing import create_redis_client
from app.limits.budget import MonthlyBudgetEnforcer, monthly_period
from app.limits.ratelimit import TokenBucketRateLimiter
from app.providers.registry import ProviderRegistry
from app.services.chat_service import (
    BudgetEnforcer,
    ChatService,
    GuardrailRunner,
    RateLimiter,
    ResponseCache,
)
from app.services.embeddings_service import EmbeddingsService
from app.services.models_service import ModelsService

_registry = ProviderRegistry()

#: Default rate-limit bucket when the limiter is enabled (burst / sustained).
_RATE_CAPACITY = 60
_RATE_REFILL_PER_SEC = 1.0

#: Default monthly cap (USD) when budget enforcement is enabled.
_DEFAULT_MONTHLY_CAP_USD = "100.00"


def get_provider_registry() -> ProviderRegistry:
    """Return the process-wide provider registry singleton."""
    return _registry


@lru_cache(maxsize=1)
def _redis_client(url: str) -> redis_async.Redis:
    """Build one process-wide Redis client for the Redis-backed collaborators.

    Cached so the cache, rate limiter, and budget enforcer share a single
    connection pool. Only ever called when `Settings.redis_url` is set, so the
    default/test env (no Redis) never constructs a client. ``decode_responses``
    matches the cache's `str` contract.

    The concrete construction is confined to `app.limits._redis_typing`
    (`create_redis_client`), which pins the loosely-typed `from_url` back to a
    `Redis`, so this builder stays fully typed.
    """
    return create_redis_client(url)


def _build_cache(settings: Settings) -> ResponseCache | None:
    """Construct the exact-match cache, or `None` when not configured (default)."""
    if not settings.cache_enabled or settings.redis_url is None:
        return None
    return ExactCache(_redis_client(settings.redis_url))


def _build_rate_limiter(settings: Settings) -> RateLimiter | None:
    """Construct the token-bucket limiter, or `None` when not configured (default)."""
    if not settings.rate_limit_enabled or settings.redis_url is None:
        return None
    return TokenBucketRateLimiter(
        _redis_client(settings.redis_url),
        capacity=_RATE_CAPACITY,
        refill_per_sec=_RATE_REFILL_PER_SEC,
    )


def _build_budget(settings: Settings) -> BudgetEnforcer | None:
    """Construct the monthly budget enforcer, or `None` when not configured (default)."""
    if not settings.budget_enabled or settings.redis_url is None:
        return None
    today = date.today()
    period_start = today.replace(day=1)
    return MonthlyBudgetEnforcer(
        _redis_client(settings.redis_url),
        cap_usd=Decimal(_DEFAULT_MONTHLY_CAP_USD),
        period=monthly_period(period_start),
        resets_on=period_start,
    )


def _build_guardrails(settings: Settings) -> GuardrailRunner | None:
    """Construct the pre-guardrail chain, or `None` when not configured (default).

    Pure-Python (no Redis/DB), so it gates on its flag alone. Only the pre-phase
    request guardrails (size, injection, PII redaction) are wired here; the
    post-phase guardrails (schema/content/citation) need per-route schema and a
    citation verifier and are composed by their own wiring tickets.
    """
    if not settings.guardrails_enabled:
        return None
    return GuardrailChain(
        pre=(SizeGuardrail(), InjectionGuardrail(), PiiGuardrail()),
    )


def get_chat_service(
    registry: Annotated[ProviderRegistry, Depends(get_provider_registry)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatService:
    """Build the chat service, wiring only the collaborators that are configured.

    In the default/test env (no Redis URL, all flags off) every builder returns
    `None`, so this is `ChatService(registry)` — a working Mock-only service with
    zero external dependencies, identical to the pre-wiring behaviour.
    """
    return ChatService(
        registry,
        cache=_build_cache(settings),
        rate_limiter=_build_rate_limiter(settings),
        budget=_build_budget(settings),
        guardrails=_build_guardrails(settings),
    )


def get_models_service(
    registry: Annotated[ProviderRegistry, Depends(get_provider_registry)],
) -> ModelsService:
    return ModelsService(registry)


def get_embeddings_service(
    registry: Annotated[ProviderRegistry, Depends(get_provider_registry)],
) -> EmbeddingsService:
    return EmbeddingsService(registry)


def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    key = authorization.removeprefix("Bearer ").strip()
    if key not in settings.api_keys:
        raise HTTPException(status_code=401, detail="invalid api key")
    return key
