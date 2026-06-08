"""GW-3 — AnthropicProvider tests.

Offline unit tests drive the provider with a **fake** `AsyncAnthropic` client
(no network, no key) built from real SDK objects, asserting the response/usage
mapping — especially all four token fields, including
``cache_creation_input_tokens`` and ``cache_read_input_tokens``. The real
integration test is **key-gated**: it skips (never fails) when
``ANTHROPIC_API_KEY`` is absent, so CI stays offline and zero-cost.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import pytest
from anthropic.types import (
    Message as AnthropicMessage,
)
from anthropic.types import (
    RawContentBlockDeltaEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    TextBlock,
    TextDelta,
)
from anthropic.types import (
    Usage as AnthropicUsage,
)
from anthropic.types.beta.beta_message_tokens_count import BetaMessageTokensCount
from anthropic.types.message_delta_usage import MessageDeltaUsage
from anthropic.types.raw_message_delta_event import Delta as MessageDelta

from app.domain.messages import Message
from app.providers.anthropic import AnthropicProvider
from app.providers.base import Provider


class _FakeMessages:
    """Stands in for ``client.messages``: canned create (sync + streaming)."""

    def __init__(self, response: AnthropicMessage, events: list[object]) -> None:
        self._response = response
        self._events = events
        self.create_kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> object:
        self.create_kwargs = kwargs
        if kwargs.get("stream"):
            return self._aiter()
        return self._response

    async def _aiter(self) -> AsyncIterator[object]:
        for event in self._events:
            yield event


class _FakeBetaMessages:
    """Stands in for ``client.beta.messages``: canned count_tokens."""

    def __init__(self, input_tokens: int) -> None:
        self._input_tokens = input_tokens
        self.count_kwargs: dict[str, Any] = {}

    async def count_tokens(self, **kwargs: Any) -> BetaMessageTokensCount:
        self.count_kwargs = kwargs
        return BetaMessageTokensCount(input_tokens=self._input_tokens)


class _FakeBeta:
    def __init__(self, messages: _FakeBetaMessages) -> None:
        self.messages = messages


class _FakeAnthropic:
    """Minimal fake of `AsyncAnthropic` exposing only what the adapter touches."""

    def __init__(
        self,
        response: AnthropicMessage,
        events: list[object],
        count: int,
    ) -> None:
        self.messages = _FakeMessages(response, events)
        self.beta = _FakeBeta(_FakeBetaMessages(count))


def _response() -> AnthropicMessage:
    return AnthropicMessage(
        id="msg_1",
        type="message",
        role="assistant",
        model="claude-opus-4-8",
        content=[TextBlock(type="text", text="hello world")],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=AnthropicUsage(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=3,
            cache_read_input_tokens=2,
        ),
    )


def _stream_events() -> list[object]:
    return [
        RawMessageStartEvent(
            type="message_start",
            message=AnthropicMessage(
                id="msg_1",
                type="message",
                role="assistant",
                model="claude-opus-4-8",
                content=[],
                stop_reason=None,
                stop_sequence=None,
                usage=AnthropicUsage(
                    input_tokens=10,
                    output_tokens=0,
                    cache_creation_input_tokens=3,
                    cache_read_input_tokens=2,
                ),
            ),
        ),
        RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=TextDelta(type="text_delta", text="hello "),
        ),
        RawContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=TextDelta(type="text_delta", text="world"),
        ),
        RawMessageDeltaEvent(
            type="message_delta",
            delta=MessageDelta(stop_reason="end_turn", stop_sequence=None),
            usage=MessageDeltaUsage(output_tokens=7),
        ),
    ]


def _provider(count: int = 42) -> tuple[AnthropicProvider, _FakeAnthropic]:
    fake = _FakeAnthropic(_response(), _stream_events(), count)
    return AnthropicProvider(fake), fake  # type: ignore[arg-type]


def test_satisfies_provider_protocol() -> None:
    provider, _ = _provider()
    assert isinstance(provider, Provider)


async def test_chat_maps_content_and_finish_reason() -> None:
    provider, _ = _provider()
    result = await provider.chat(
        model="claude-opus-4-8", messages=[Message(role="user", content="hi")]
    )
    assert result.content == "hello world"
    assert result.model == "claude-opus-4-8"
    assert result.finish_reason == "end_turn"


async def test_chat_maps_all_four_token_fields() -> None:
    provider, _ = _provider()
    result = await provider.chat(
        model="claude-opus-4-8", messages=[Message(role="user", content="hi")]
    )
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5
    assert result.usage.cache_creation_input_tokens == 3
    assert result.usage.cache_read_input_tokens == 2


async def test_chat_passes_system_and_sampling_params() -> None:
    provider, fake = _provider()
    await provider.chat(
        model="claude-opus-4-8",
        messages=[
            Message(role="system", content="be terse"),
            Message(role="user", content="hi"),
        ],
        max_tokens=256,
        temperature=0.2,
    )
    kwargs = fake.messages.create_kwargs
    assert kwargs["system"] == "be terse"
    assert kwargs["max_tokens"] == 256
    assert kwargs["temperature"] == 0.2
    # system messages are not forwarded as chat turns
    assert all(m["role"] != "system" for m in kwargs["messages"])


async def test_chat_stream_yields_content_then_terminal_usage() -> None:
    provider, _ = _provider()
    deltas = [
        d
        async for d in provider.chat_stream(
            model="claude-opus-4-8", messages=[Message(role="user", content="hi")]
        )
    ]
    content = "".join(d.content for d in deltas if d.usage is None)
    assert content == "hello world"

    terminal = deltas[-1]
    assert terminal.finish_reason == "end_turn"
    assert terminal.usage is not None
    assert terminal.usage.input_tokens == 10
    assert terminal.usage.output_tokens == 7
    assert terminal.usage.cache_creation_input_tokens == 3
    assert terminal.usage.cache_read_input_tokens == 2


async def test_count_tokens_uses_beta_endpoint() -> None:
    provider, fake = _provider(count=123)
    count = await provider.count_tokens(
        model="claude-opus-4-8", messages=[Message(role="user", content="hi")]
    )
    assert count == 123
    assert fake.beta.messages.count_kwargs["model"] == "claude-opus-4-8"


async def test_embed_is_not_supported() -> None:
    provider, _ = _provider()
    with pytest.raises(NotImplementedError):
        await provider.embed(model="claude-opus-4-8", inputs=["x"])


async def test_models_lists_authoritative_ids() -> None:
    provider, _ = _provider()
    assert await provider.models() == [
        "claude-opus-4-8",
        "claude-sonnet-4-6",
        "claude-haiku-4-5",
    ]


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set — real Anthropic integration test skipped (offline/CI)",
)
async def test_real_anthropic_chat_maps_usage() -> None:
    provider = AnthropicProvider.from_api_key(os.environ["ANTHROPIC_API_KEY"])
    result = await provider.chat(
        model="claude-haiku-4-5",
        messages=[Message(role="user", content="Reply with exactly: pong")],
        max_tokens=16,
    )
    assert result.content
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
