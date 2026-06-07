"""GW-8 — POST /v1/embeddings controller (thin HTTP layer).

Parses the request, enforces per-key bearer auth, and delegates to
`EmbeddingsService` (injected via `app.api.deps`). All embedding orchestration
lives in the service layer; this controller only maps between HTTP and the
service, including the domain `UnknownModelError` → 404 mapping. See ADR-016.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_embeddings_service, require_api_key
from app.domain.errors import UnknownModelError
from app.domain.openai import EmbeddingRequest, EmbeddingResponse
from app.services.embeddings_service import EmbeddingsService

router = APIRouter()


@router.post("/v1/embeddings")
async def embeddings(
    req: EmbeddingRequest,
    service: Annotated[EmbeddingsService, Depends(get_embeddings_service)],
    _key: Annotated[str, Depends(require_api_key)],
) -> EmbeddingResponse:
    try:
        return await service.embed(req)
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
