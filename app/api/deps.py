"""FastAPI dependency providers — the DI wiring for the API layer.

Controllers declare what they need via `Depends(...)` instead of constructing
collaborators, so auth, settings, the provider registry, and services are wired
in one place (the composition root for the request scope). See ADR-016.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException

from app.config import Settings, get_settings
from app.providers.registry import ProviderRegistry
from app.services.chat_service import ChatService
from app.services.embeddings_service import EmbeddingsService
from app.services.models_service import ModelsService

_registry = ProviderRegistry()


def get_provider_registry() -> ProviderRegistry:
    """Return the process-wide provider registry singleton."""
    return _registry


def get_chat_service(
    registry: Annotated[ProviderRegistry, Depends(get_provider_registry)],
) -> ChatService:
    return ChatService(registry)


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
