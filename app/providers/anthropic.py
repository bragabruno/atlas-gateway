"""GW-3 — AnthropicProvider.

Adapter over the official ``anthropic`` SDK that satisfies the `Provider`
protocol. Maps Anthropic's `Usage` onto our four-field `Usage` — crucially
``cache_creation_input_tokens`` and ``cache_read_input_tokens`` so the cost
recorder (GW-14) can price cached input correctly — and counts tokens via the
provider's own ``beta.messages.count_tokens`` endpoint (never tiktoken; see
atlas-docs/02 + ADR-012).

The SDK client is **injected** (`client` arg), so offline unit tests pass a
fake that returns canned SDK objects (no network, no key) and assert the
response/usage mapping. `from_api_key` builds a real `AsyncAnthropic`.
Authoritative model ids: ``claude-opus-4-8``, ``claude-sonnet-4-6``,
``claude-haiku-4-5`` (atlas-docs/02).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic, AsyncStream, omit
from anthropic.types import Message as AnthropicMessage
from anthropic.types import MessageParam, RawMessageStreamEvent
from anthropic.types import Usage as AnthropicUsage
from anthropic.types.beta import BetaMessageParam

from app.domain.messages import ChatResult, EmbeddingResult, Message, StreamDelta, Usage

#: Authoritative Anthropic model ids served by this provider (atlas-docs/02).
ANTHROPIC_MODELS: tuple[str, ...] = (
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
)

#: Default cap when a caller omits ``max_tokens``; the Anthropic API requires it.
_DEFAULT_MAX_TOKENS = 1024


def _to_anthropic_messages(messages: list[Message]) -> list[MessageParam]:
    """Map internal `Message`s to Anthropic `MessageParam`s (system handled apart)."""
    return [
        MessageParam(role="assistant" if m.role == "assistant" else "user", content=m.content)
        for m in messages
        if m.role != "system"
    ]


def _to_beta_messages(messages: list[Message]) -> list[BetaMessageParam]:
    """Map internal `Message`s to `BetaMessageParam`s for ``beta.count_tokens``."""
    return [
        BetaMessageParam(role="assistant" if m.role == "assistant" else "user", content=m.content)
        for m in messages
        if m.role != "system"
    ]


def _system_prompt(messages: list[Message]) -> str | None:
    """Concatenate any system messages into the Anthropic top-level system field."""
    systems = [m.content for m in messages if m.role == "system"]
    return "\n\n".join(systems) if systems else None


def _map_usage(usage: AnthropicUsage) -> Usage:
    """Map Anthropic `Usage` onto our four-field `Usage` (cache fields nullable)."""
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
        cache_read_input_tokens=usage.cache_read_input_tokens or 0,
    )


def _text_of(message: AnthropicMessage) -> str:
    """Concatenate the text of every text block in the message content."""
    return "".join(block.text for block in message.content if block.type == "text")


class AnthropicProvider:
    """Anthropic adapter that structurally satisfies `Provider`."""

    name = "anthropic"

    def __init__(self, client: AsyncAnthropic) -> None:
        """Hold an injected `AsyncAnthropic` client (real or fake for tests)."""
        self._client = client

    @classmethod
    def from_api_key(cls, api_key: str) -> AnthropicProvider:
        """Build a provider backed by a real `AsyncAnthropic` client."""
        return cls(AsyncAnthropic(api_key=api_key))

    async def chat(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        response = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens or _DEFAULT_MAX_TOKENS,
            messages=_to_anthropic_messages(messages),
            system=_system_prompt(messages) or omit,
            temperature=temperature if temperature is not None else omit,
        )
        return ChatResult(
            model=response.model,
            content=_text_of(response),
            finish_reason=response.stop_reason or "stop",
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
        # Accumulate usage across events: `message_start` carries input + cache
        # tokens, `message_delta` carries the final output_tokens + stop_reason.
        input_tokens = 0
        output_tokens = 0
        cache_creation = 0
        cache_read = 0
        finish_reason = "stop"

        stream: AsyncStream[RawMessageStreamEvent] = await self._client.messages.create(
            model=model,
            max_tokens=max_tokens or _DEFAULT_MAX_TOKENS,
            messages=_to_anthropic_messages(messages),
            system=_system_prompt(messages) or omit,
            temperature=temperature if temperature is not None else omit,
            stream=True,
        )
        async for event in stream:
            if event.type == "message_start":
                usage = event.message.usage
                input_tokens = usage.input_tokens
                cache_creation = usage.cache_creation_input_tokens or 0
                cache_read = usage.cache_read_input_tokens or 0
            elif event.type == "content_block_delta" and event.delta.type == "text_delta":
                yield StreamDelta(content=event.delta.text)
            elif event.type == "message_delta":
                output_tokens = event.usage.output_tokens
                if event.delta.stop_reason is not None:
                    finish_reason = event.delta.stop_reason

        yield StreamDelta(
            finish_reason=finish_reason,
            usage=Usage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=cache_creation,
                cache_read_input_tokens=cache_read,
            ),
        )

    async def embed(self, *, model: str, inputs: list[str]) -> EmbeddingResult:
        """Anthropic exposes no embeddings API; embeddings route to other providers."""
        raise NotImplementedError("AnthropicProvider does not support embeddings")

    async def count_tokens(self, *, model: str, messages: list[Message]) -> int:
        """Count input tokens via the provider's own ``beta.messages.count_tokens``."""
        result = await self._client.beta.messages.count_tokens(
            model=model,
            messages=_to_beta_messages(messages),
            system=_system_prompt(messages) or omit,
        )
        return result.input_tokens

    async def models(self) -> list[str]:
        return list(ANTHROPIC_MODELS)
