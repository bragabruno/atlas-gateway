"""GW-6 — /v1/chat/completions (non-streaming).

Minimal slice: per-key bearer auth, a tiny provider registry (only the Mock
provider is wired here), and translation between the OpenAI-compatible schema
and the provider-internal types. Alias routing (GW-10), streaming (GW-7),
caching (GW-13), accounting (GW-14) and real providers (GW-3..5) layer on later.
"""

from __future__ import annotations

import time
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException

from app.config import Settings, get_settings
from app.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    CompletionUsage,
    ResponseMessage,
)
from app.providers.base import Message, Provider
from app.providers.mock import MockProvider

router = APIRouter()

_PROVIDERS: dict[str, Provider] = {"mock": MockProvider()}


def _resolve_provider(model: str) -> Provider:
    provider = _PROVIDERS.get(model)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"unknown model: {model}")
    return provider


def _require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
) -> str:
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    key = authorization.removeprefix("Bearer ").strip()
    if key not in settings.api_keys:
        raise HTTPException(status_code=401, detail="invalid api key")
    return key


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    req: ChatCompletionRequest,
    _key: Annotated[str, Depends(_require_api_key)],
) -> ChatCompletionResponse:
    provider = _resolve_provider(req.model)
    result = await provider.chat(
        model=req.model,
        messages=[Message(role=m.role, content=m.content) for m in req.messages],
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )
    usage = CompletionUsage(
        prompt_tokens=result.usage.input_tokens,
        completion_tokens=result.usage.output_tokens,
        total_tokens=result.usage.input_tokens + result.usage.output_tokens,
    )
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=result.model,
        choices=[
            Choice(
                index=0,
                message=ResponseMessage(content=result.content),
                finish_reason=result.finish_reason,
            )
        ],
        usage=usage,
    )
