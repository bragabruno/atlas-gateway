"""GW-8 — GET /v1/models controller (thin HTTP layer).

Enforces per-key bearer auth and delegates to `ModelsService` (injected via
`app.api.deps`) to assemble the OpenAI-compatible model list. All listing logic
lives in the service layer; this controller only maps service → HTTP. See
ADR-016.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_models_service, require_api_key
from app.domain.openai import ErrorEnvelope, ModelList
from app.services.models_service import ModelsService

router = APIRouter()


@router.get(
    "/v1/models",
    tags=["models"],
    summary="List available models",
    description=(
        "OpenAI-compatible model list. Aggregates every model id/alias advertised "
        "by the provider registry (OpenAI + Anthropic + Google + Atlas Mock)."
    ),
    responses={401: {"model": ErrorEnvelope, "description": "Missing or invalid Bearer key."}},
)
async def list_models(
    service: Annotated[ModelsService, Depends(get_models_service)],
    _key: Annotated[str, Depends(require_api_key)],
) -> ModelList:
    return service.list_models()
