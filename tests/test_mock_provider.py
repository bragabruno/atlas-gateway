from __future__ import annotations

from app.domain.messages import Message
from app.providers.base import Provider
from app.providers.mock import MockProvider


async def test_mock_provider_satisfies_protocol() -> None:
    assert isinstance(MockProvider(), Provider)


async def test_mock_chat_is_deterministic() -> None:
    provider = MockProvider()
    messages = [Message(role="user", content="hello world")]
    first = await provider.chat(model="mock", messages=messages)
    second = await provider.chat(model="mock", messages=messages)
    assert first.content == second.content
    assert "hello world" in first.content
    assert first.usage.input_tokens > 0
    assert first.usage.output_tokens > 0


async def test_mock_models() -> None:
    assert await MockProvider().models() == ["mock"]


async def test_mock_embed_is_deterministic() -> None:
    provider = MockProvider()
    first = await provider.embed(model="mock", inputs=["hello world"])
    second = await provider.embed(model="mock", inputs=["hello world"])
    assert first.embeddings == second.embeddings
    assert len(first.embeddings) == 1
    assert len(first.embeddings[0]) == 8
    assert first.usage.input_tokens > 0


async def test_mock_embed_preserves_input_order() -> None:
    provider = MockProvider()
    result = await provider.embed(model="mock", inputs=["a", "b", "c"])
    assert len(result.embeddings) == 3
    assert result.embeddings[0] != result.embeddings[1]
