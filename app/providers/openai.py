"""GW-4 — OpenAIProvider.

Adapter over the official ``openai`` SDK that satisfies the `Provider` protocol.
Maps OpenAI's `CompletionUsage` onto our four-field `Usage`: ``prompt_tokens`` →
``input_tokens``, ``completion_tokens`` → ``output_tokens``, and
``prompt_tokens_details.cached_tokens`` → ``cache_read_input_tokens`` (OpenAI has
no separate cache-creation charge, so ``cache_creation_input_tokens`` is 0).
Token counting uses the provider's own ``responses.input_tokens.count`` endpoint
(never tiktoken; see atlas-docs/02 + ADR-012).

The SDK client is **injected** (`client` arg), so offline unit tests pass a fake
that returns canned SDK objects (no network, no key) and assert the
response/usage mapping. `from_api_key` builds a real `AsyncOpenAI`. Authoritative
model id: ``gpt-4.1`` (atlas-docs/02).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI, AsyncStream, omit
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam
from openai.types.completion_usage import CompletionUsage

from app.domain.messages import ChatResult, EmbeddingResult, Message, StreamDelta, Usage

#: Authoritative OpenAI model ids served by this provider (atlas-docs/02).
OPENAI_MODELS: tuple[str, ...] = ("gpt-4.1",)

#: Default embedding model used when a caller does not pin one explicitly.
_DEFAULT_EMBED_MODEL = "text-embedding-3-small"


def _to_openai_messages(messages: list[Message]) -> list[ChatCompletionMessageParam]:
    """Map internal `Message`s to OpenAI chat message params."""
    params: list[ChatCompletionMessageParam] = []
    for m in messages:
        if m.role == "system":
            params.append({"role": "system", "content": m.content})
        elif m.role == "assistant":
            params.append({"role": "assistant", "content": m.content})
        else:
            params.append({"role": "user", "content": m.content})
    return params


def _map_usage(usage: CompletionUsage | None) -> Usage:
    """Map OpenAI `CompletionUsage` onto our four-field `Usage`.

    OpenAI bills cached input at a discount but has no cache-creation charge, so
    ``cache_creation_input_tokens`` is always 0 and the reported cached prompt
    tokens map to ``cache_read_input_tokens``.
    """
    if usage is None:
        return Usage()
    details = usage.prompt_tokens_details
    cache_read = details.cached_tokens if details and details.cached_tokens else 0
    return Usage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=cache_read,
    )


class OpenAIProvider:
    """OpenAI adapter that structurally satisfies `Provider`."""

    name = "openai"

    def __init__(self, client: AsyncOpenAI) -> None:
        """Hold an injected `AsyncOpenAI` client (real or fake for tests)."""
        self._client = client

    @classmethod
    def from_api_key(cls, api_key: str) -> OpenAIProvider:
        """Build a provider backed by a real `AsyncOpenAI` client."""
        return cls(AsyncOpenAI(api_key=api_key))

    async def chat(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        response = await self._client.chat.completions.create(
            model=model,
            messages=_to_openai_messages(messages),
            max_tokens=max_tokens if max_tokens is not None else omit,
            temperature=temperature if temperature is not None else omit,
        )
        choice = response.choices[0]
        return ChatResult(
            model=response.model,
            content=choice.message.content or "",
            finish_reason=choice.finish_reason or "stop",
            usage=_map_usage(response.usage),
        )

    async def chat_stream(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamDelta]:
        finish_reason = "stop"
        usage: CompletionUsage | None = None

        # `include_usage` makes the API emit a final usage-bearing chunk after the
        # content chunks; that chunk has an empty `choices` list.
        stream: AsyncStream[ChatCompletionChunk] = await self._client.chat.completions.create(
            model=model,
            messages=_to_openai_messages(messages),
            max_tokens=max_tokens if max_tokens is not None else omit,
            temperature=temperature if temperature is not None else omit,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.delta.content:
                yield StreamDelta(content=choice.delta.content)
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

        yield StreamDelta(finish_reason=finish_reason, usage=_map_usage(usage))

    async def embed(self, *, model: str, inputs: list[str]) -> EmbeddingResult:
        # `gpt-4.1` is a chat model, not an embedding model; fall back to the
        # default embedding model unless the caller pins a real embedding model.
        embed_model = _DEFAULT_EMBED_MODEL if model in OPENAI_MODELS else model
        response = await self._client.embeddings.create(model=embed_model, input=inputs)
        return EmbeddingResult(
            model=response.model,
            embeddings=[item.embedding for item in response.data],
            usage=Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=0,
            ),
        )

    async def count_tokens(self, *, model: str, messages: list[Message]) -> int:
        """Count input tokens via the provider's own ``responses.input_tokens.count``."""
        text = "\n".join(f"{m.role}: {m.content}" for m in messages)
        result = await self._client.responses.input_tokens.count(model=model, input=text)
        return result.input_tokens

    async def models(self) -> list[str]:
        return list(OPENAI_MODELS)
