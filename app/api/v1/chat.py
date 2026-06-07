"""GW-6/GW-7 — /v1/chat/completions (non-streaming + streaming SSE).

Minimal slice: per-key bearer auth, a tiny provider registry (only the Mock
provider is wired here), and translation between the OpenAI-compatible schema
and the provider-internal types. When `stream=true` the response is an
OpenAI-compatible `text/event-stream` of `chat.completion.chunk` events,
terminated by the `data: [DONE]` sentinel. Alias routing (GW-10), caching
(GW-13), accounting (GW-14) and real providers (GW-3..5) layer on later.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceDelta,
    ChunkChoice,
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


def _provider_messages(req: ChatCompletionRequest) -> list[Message]:
    return [Message(role=m.role, content=m.content) for m in req.messages]


async def _stream_sse(req: ChatCompletionRequest, provider: Provider) -> AsyncIterator[str]:
    """Yield OpenAI-compatible SSE frames for a streaming completion.

    Emits a leading role delta, then one content frame per provider delta, then
    a terminal frame carrying `finish_reason` (and `usage` if the provider
    reported it), and finally the `[DONE]` sentinel.
    """
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    role_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=req.model,
        choices=[ChunkChoice(delta=ChoiceDelta(role="assistant"))],
    )
    yield f"data: {role_chunk.model_dump_json()}\n\n"

    async for delta in provider.chat_stream(
        model=req.model,
        messages=_provider_messages(req),
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    ):
        is_terminal = delta.finish_reason is not None or delta.usage is not None
        if is_terminal:
            usage = (
                CompletionUsage(
                    prompt_tokens=delta.usage.input_tokens,
                    completion_tokens=delta.usage.output_tokens,
                    total_tokens=delta.usage.input_tokens + delta.usage.output_tokens,
                )
                if delta.usage is not None
                else None
            )
            final_chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=req.model,
                choices=[
                    ChunkChoice(delta=ChoiceDelta(), finish_reason=delta.finish_reason or "stop")
                ],
                usage=usage,
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"
        else:
            chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=req.model,
                choices=[ChunkChoice(delta=ChoiceDelta(content=delta.content))],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    req: ChatCompletionRequest,
    _key: Annotated[str, Depends(_require_api_key)],
) -> ChatCompletionResponse | StreamingResponse:
    provider = _resolve_provider(req.model)

    if req.stream:
        return StreamingResponse(
            _stream_sse(req, provider),
            media_type="text/event-stream",
        )

    result = await provider.chat(
        model=req.model,
        messages=_provider_messages(req),
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
