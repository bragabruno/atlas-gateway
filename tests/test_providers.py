"""GW-3/4/5 — AnthropicProvider, OpenAIProvider, GoogleProvider.

Tests use unittest.mock to replace the underlying SDK clients so no real API
calls or keys are needed. Each test pins the mock return values to match the
SDK's real response shapes; the assertion is on the normalized ChatResult /
EmbeddingResult that the provider returns.
"""

# Tests replace each provider's protected `_client` with a mocked SDK client.
# pyright: reportPrivateUsage=false
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from anthropic.types import TextBlock

from app.domain.messages import Message
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.registry import ProviderRegistry

_MESSAGES = [Message(role="user", content="hi")]


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    def _provider(self) -> AnthropicProvider:
        return AnthropicProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_chat_returns_normalized_result(self) -> None:
        provider = self._provider()
        # Use a real TextBlock: the provider narrows resp.content[0] with
        # isinstance(TextBlock) before reading .text, so a duck-typed stub
        # would (correctly) be treated as a non-text block and dropped.
        mock_content = TextBlock(type="text", text="hello")
        mock_resp = SimpleNamespace(
            model="claude-3-5-sonnet-20241022",
            content=[mock_content],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        provider._client.messages.create = AsyncMock(return_value=mock_resp)
        result = await provider.chat(model="claude-3-5-sonnet-20241022", messages=_MESSAGES)
        assert result.content == "hello"
        assert result.model == "claude-3-5-sonnet-20241022"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    @pytest.mark.asyncio
    async def test_embed_raises_not_implemented(self) -> None:
        provider = self._provider()
        with pytest.raises(NotImplementedError):
            await provider.embed(model="any", inputs=["text"])

    @pytest.mark.asyncio
    async def test_models_returns_list(self) -> None:
        provider = self._provider()
        models = await provider.models()
        assert isinstance(models, list)
        assert len(models) > 0


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    def _provider(self) -> OpenAIProvider:
        return OpenAIProvider(api_key="test-key")

    @pytest.mark.asyncio
    async def test_chat_returns_normalized_result(self) -> None:
        provider = self._provider()
        mock_message = SimpleNamespace(content="world", role="assistant")
        mock_choice = SimpleNamespace(message=mock_message, finish_reason="stop")
        mock_usage = SimpleNamespace(prompt_tokens=8, completion_tokens=3)
        mock_resp = SimpleNamespace(
            model="gpt-4o",
            choices=[mock_choice],
            usage=mock_usage,
        )
        provider._client.chat.completions.create = AsyncMock(return_value=mock_resp)
        result = await provider.chat(model="gpt-4o", messages=_MESSAGES)
        assert result.content == "world"
        assert result.model == "gpt-4o"
        assert result.usage.input_tokens == 8
        assert result.usage.output_tokens == 3

    @pytest.mark.asyncio
    async def test_embed_returns_vectors(self) -> None:
        provider = self._provider()
        vec = [0.1, 0.2, 0.3]
        mock_item = SimpleNamespace(embedding=vec, index=0)
        mock_usage = SimpleNamespace(prompt_tokens=4, total_tokens=4)
        mock_resp = SimpleNamespace(
            model="text-embedding-3-small", data=[mock_item], usage=mock_usage
        )
        provider._client.embeddings.create = AsyncMock(return_value=mock_resp)
        result = await provider.embed(model="text-embedding-3-small", inputs=["hello"])
        assert result.embeddings == [vec]
        assert result.usage.input_tokens == 4

    @pytest.mark.asyncio
    async def test_transient_error_raises_transient_provider_error(self) -> None:
        import openai

        from app.resilience.retry import TransientProviderError

        provider = self._provider()
        exc = openai.APIStatusError(
            message="rate limit",
            response=MagicMock(status_code=429),
            body=None,
        )
        provider._client.chat.completions.create = AsyncMock(side_effect=exc)
        with pytest.raises(TransientProviderError):
            await provider.chat(model="gpt-4o", messages=_MESSAGES)

    def test_chat_stream_returns_async_iterator_not_coroutine(self) -> None:
        """Calling chat_stream must return an async iterator directly.

        The service does ``async for delta in provider.chat_stream(...)`` without
        awaiting, so an ``async def`` that *returns* the generator (yielding a
        coroutine) breaks streaming with 'async for requires __aiter__'.
        """
        import asyncio

        stream = self._provider().chat_stream(model="gpt-4o", messages=_MESSAGES)
        assert not asyncio.iscoroutine(stream)
        assert hasattr(stream, "__aiter__")


# ---------------------------------------------------------------------------
# ProviderRegistry.from_settings
# ---------------------------------------------------------------------------


def test_registry_from_settings_no_keys_has_only_mock() -> None:
    settings = SimpleNamespace(anthropic_api_key=None, openai_api_key=None, google_api_key=None)
    registry = ProviderRegistry.from_settings(settings)
    assert registry.resolve("mock") is not None
    assert registry.resolve("gpt-4o") is None
    assert registry.resolve("claude-3-5-sonnet-20241022") is None


def test_registry_from_settings_wires_anthropic_models() -> None:
    settings = SimpleNamespace(
        anthropic_api_key="sk-ant-test",
        openai_api_key=None,
        google_api_key=None,
    )
    registry = ProviderRegistry.from_settings(settings)
    provider = registry.resolve("claude-3-5-sonnet-20241022")
    assert provider is not None
    assert provider.name == "anthropic"


def test_registry_from_settings_wires_openai_models() -> None:
    settings = SimpleNamespace(
        anthropic_api_key=None,
        openai_api_key="sk-test",
        google_api_key=None,
    )
    registry = ProviderRegistry.from_settings(settings)
    provider = registry.resolve("gpt-4o")
    assert provider is not None
    assert provider.name == "openai"
    embed_provider = registry.resolve("text-embedding-3-small")
    assert embed_provider is not None
    assert embed_provider.name == "openai"


def test_registry_from_settings_wires_ollama_models() -> None:
    settings = SimpleNamespace(
        anthropic_api_key=None,
        openai_api_key=None,
        google_api_key=None,
        ollama_base_url="http://host.docker.internal:11434/v1",
        ollama_models=("gpt-oss:120b-cloud", "kimi-k2.5:cloud"),
    )
    registry = ProviderRegistry.from_settings(settings)
    # Each configured Ollama model id resolves to an OpenAI-protocol provider...
    gpt_oss = registry.resolve("gpt-oss:120b-cloud")
    assert gpt_oss is not None
    assert gpt_oss.name == "openai"
    assert registry.resolve("kimi-k2.5:cloud") is not None
    # ...and the mock fallback is still present.
    assert registry.resolve("mock") is not None
