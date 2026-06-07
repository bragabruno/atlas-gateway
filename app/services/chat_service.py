"""Chat use-case orchestration (service layer).

Resolves the provider for a request, drives the provider call, and maps
provider-internal results to the OpenAI-compatible wire schema. The controller
(`app.api.v1.chat`) stays thin — all chat business logic lives here. Caching
(GW-13), accounting (GW-14), guardrails (GRD-1) and alias routing (GW-10) layer
in here as collaborators, not in the controller. See ADR-016.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator

from app.domain.errors import UnknownModelError
from app.domain.messages import Message
from app.domain.openai import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceDelta,
    ChunkChoice,
    CompletionUsage,
    ResponseMessage,
)
from app.providers.base import Provider
from app.providers.registry import ProviderRegistry


class ChatService:
    """Orchestrates chat completions over the provider registry."""

    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    def _resolve(self, model: str) -> Provider:
        provider = self._registry.resolve(model)
        if provider is None:
            raise UnknownModelError(model)
        return provider

    @staticmethod
    def _provider_messages(req: ChatCompletionRequest) -> list[Message]:
        return [Message(role=m.role, content=m.content) for m in req.messages]

    async def complete(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        """Run a non-streaming completion and return the OpenAI-shaped response."""
        provider = self._resolve(req.model)
        result = await provider.chat(
            model=req.model,
            messages=self._provider_messages(req),
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

    def stream(self, req: ChatCompletionRequest) -> AsyncIterator[str]:
        """Return an SSE frame iterator for a streaming completion.

        The provider is resolved eagerly (raising `UnknownModelError` before the
        response body starts) so the controller can map it to a 404; the
        returned async generator yields the OpenAI-compatible `data:` frames.
        """
        provider = self._resolve(req.model)
        return self._stream_frames(req, provider)

    async def _stream_frames(
        self, req: ChatCompletionRequest, provider: Provider
    ) -> AsyncIterator[str]:
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
            messages=self._provider_messages(req),
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        ):
            if delta.finish_reason is not None or delta.usage is not None:
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
                        ChunkChoice(
                            delta=ChoiceDelta(), finish_reason=delta.finish_reason or "stop"
                        )
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
