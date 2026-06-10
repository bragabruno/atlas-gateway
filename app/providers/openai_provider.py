"""GW-4 — OpenAIProvider.

Wraps `openai.AsyncOpenAI` to satisfy the `Provider` protocol. Retryable errors
(rate limit, 5xx) become `TransientProviderError`; auth/validation errors
propagate as-is. The `embed` path is fully supported — the gateway's
`/v1/embeddings` passthrough routes here for OpenAI embedding models.

See ADR-012 for the provider protocol and ADR-016 for the adapter pattern.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import openai

from app.domain.messages import ChatResult, EmbeddingResult, Message, StreamDelta, Usage
from app.resilience.retry import TransientProviderError


def _is_transient(exc: openai.APIStatusError) -> bool:
    return exc.status_code in {429, 500, 502, 503, 504}


class OpenAIProvider:
    """Adapter from the OpenAI SDK to the `Provider` protocol."""

    name = "openai"

    def __init__(self, *, api_key: str, base_url: str | None = None) -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def chat(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        kwargs: dict = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except openai.APIStatusError as exc:
            if _is_transient(exc):
                raise TransientProviderError(str(exc)) from exc
            raise
        choice = resp.choices[0]
        usage = resp.usage
        return ChatResult(
            model=resp.model,
            content=choice.message.content or "",
            finish_reason=choice.finish_reason or "stop",
            usage=Usage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
        )

    async def chat_stream(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamDelta]:
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
        kwargs: dict = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        try:
            async with await self._client.chat.completions.create(**kwargs) as stream:
                async for chunk in stream:
                    choice = chunk.choices[0] if chunk.choices else None
                    if choice and choice.delta.content:
                        yield StreamDelta(content=choice.delta.content)
                    if choice and choice.finish_reason:
                        usage = chunk.usage
                        yield StreamDelta(
                            finish_reason=choice.finish_reason,
                            usage=Usage(
                                input_tokens=usage.prompt_tokens if usage else 0,
                                output_tokens=usage.completion_tokens if usage else 0,
                            ),
                        )
        except openai.APIStatusError as exc:
            if _is_transient(exc):
                raise TransientProviderError(str(exc)) from exc
            raise

    async def embed(self, *, model: str, inputs: list[str]) -> EmbeddingResult:
        try:
            resp = await self._client.embeddings.create(model=model, input=inputs)
        except openai.APIStatusError as exc:
            if _is_transient(exc):
                raise TransientProviderError(str(exc)) from exc
            raise
        embeddings = [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
        usage = resp.usage
        return EmbeddingResult(
            model=resp.model,
            embeddings=embeddings,
            usage=Usage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=0,
            ),
        )

    async def models(self) -> list[str]:
        try:
            page = await self._client.models.list()
            return [m.id for m in page.data]
        except openai.APIStatusError as exc:
            if _is_transient(exc):
                raise TransientProviderError(str(exc)) from exc
            raise
