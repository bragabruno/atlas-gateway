"""GW-6/GW-7 — /v1/chat/completions controller (thin HTTP layer).

Parses the request, enforces per-key bearer auth, and delegates to `ChatService`
(injected via `app.api.deps`). All chat orchestration lives in the service
layer; this controller only maps between HTTP and the service, including the
domain `UnknownModelError` → 404 mapping. When `stream=true` the service yields
an OpenAI-compatible `text/event-stream` terminated by `data: [DONE]`. See
ADR-016.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import get_chat_service, require_api_key
from app.domain.errors import UnknownModelError
from app.domain.openai import ChatCompletionRequest, ChatCompletionResponse
from app.services.chat_service import ChatService

router = APIRouter()


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    req: ChatCompletionRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
    _key: Annotated[str, Depends(require_api_key)],
) -> ChatCompletionResponse | StreamingResponse:
    try:
        if req.stream:
            return StreamingResponse(service.stream(req), media_type="text/event-stream")
        return await service.complete(req)
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
