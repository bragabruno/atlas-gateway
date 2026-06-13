"""GW-3 — AnthropicProvider.

Wraps the `anthropic.AsyncAnthropic` SDK to satisfy the `Provider` protocol.
Retryable errors (rate limit, 5xx) are re-raised as `TransientProviderError`
so the retry + circuit-breaker stack above handles them uniformly. Non-retryable
errors (auth, bad request) propagate as-is. The `embed` method raises
`NotImplementedError` — Anthropic has no public embeddings endpoint; alias
routing should not resolve embedding requests here.

See ADR-012 for the provider protocol and ADR-016 for the adapter pattern.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import cast

import anthropic
from anthropic import omit
from anthropic.types import MessageParam, TextBlock

from app.domain.messages import ChatResult, EmbeddingResult, Message, StreamDelta, Usage
from app.resilience.retry import TransientProviderError

#: Models surfaced by this provider (sorted by capability, newest first).
_MODELS = [
    "claude-opus-4-5",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
]


def _to_anthropic_messages(messages: list[Message]) -> list[MessageParam]:
    """Adapt domain Messages to the SDK's typed `MessageParam` list.

    Anthropic accepts only `user`/`assistant` roles; the runtime shape matches
    `MessageParam`, and the cast pins that at the SDK boundary.
    """
    return [cast(MessageParam, {"role": m.role, "content": m.content}) for m in messages]


def _is_transient(exc: anthropic.APIStatusError) -> bool:
    return exc.status_code in {429, 500, 502, 503, 504}


class AnthropicProvider:
    """Adapter from the Anthropic SDK to the `Provider` protocol."""

    name = "anthropic"

    def __init__(self, *, api_key: str) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def chat(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        try:
            resp = await self._client.messages.create(
                model=model,
                messages=_to_anthropic_messages(messages),
                max_tokens=max_tokens or 1024,
                temperature=temperature if temperature is not None else omit,
            )
        except anthropic.APIStatusError as exc:
            if _is_transient(exc):
                raise TransientProviderError(str(exc)) from exc
            raise
        # resp.content is a union of block types (text/thinking/tool_use/…);
        # only TextBlock carries `.text`, so narrow before reading it.
        first = resp.content[0] if resp.content else None
        content = first.text if isinstance(first, TextBlock) else ""
        return ChatResult(
            model=resp.model,
            content=content,
            finish_reason=resp.stop_reason or "stop",
            usage=Usage(
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
            ),
        )

    def chat_stream(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamDelta]:
        # NOT `async def` — see OpenAIProvider.chat_stream: calling it must
        # return the async iterator directly, not a coroutine.
        return self._stream(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def _stream(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamDelta]:
        try:
            async with self._client.messages.stream(
                model=model,
                messages=_to_anthropic_messages(messages),
                max_tokens=max_tokens or 1024,
                temperature=temperature if temperature is not None else omit,
            ) as stream:
                async for text in stream.text_stream:
                    yield StreamDelta(content=text)
                final = await stream.get_final_message()
                yield StreamDelta(
                    finish_reason=final.stop_reason or "stop",
                    usage=Usage(
                        input_tokens=final.usage.input_tokens,
                        output_tokens=final.usage.output_tokens,
                    ),
                )
        except anthropic.APIStatusError as exc:
            if _is_transient(exc):
                raise TransientProviderError(str(exc)) from exc
            raise

    async def embed(self, *, model: str, inputs: list[str]) -> EmbeddingResult:
        raise NotImplementedError("Anthropic has no public embeddings endpoint")

    async def models(self) -> list[str]:
        return list(_MODELS)
