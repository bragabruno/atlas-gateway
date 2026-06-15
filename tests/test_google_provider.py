"""GW-5 — GoogleProvider (Gemini) tests.

Offline unit tests drive the provider with a **fake** `genai.Client` (no network,
no key) built from real SDK objects, asserting the response/usage mapping —
including the four token fields: ``prompt_token_count`` → ``input_tokens``,
``candidates_token_count`` → ``output_tokens``, ``cached_content_token_count`` →
``cache_read_input_tokens`` (and ``cache_creation_input_tokens`` == 0, since
Gemini has no cache-creation charge). The real integration test is **key-gated**:
it skips (never fails) when ``GOOGLE_API_KEY`` is absent.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import pytest
from google.genai.types import (
    Candidate,
    Content,
    ContentEmbedding,
    CountTokensResponse,
    EmbedContentResponse,
    FinishReason,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    Part,
)

from app.domain.messages import Message
from app.providers.base import Provider
from app.providers.google import GoogleProvider


class _FakeAioModels:
    """Stands in for ``client.aio.models``: canned generate/stream/count/embed."""

    def __init__(
        self,
        response: GenerateContentResponse,
        chunks: list[GenerateContentResponse],
        count: int,
        embeddings: EmbedContentResponse,
    ) -> None:
        self._response = response
        self._chunks = chunks
        self._count = count
        self._embeddings = embeddings
        self.generate_kwargs: dict[str, Any] = {}
        self.count_kwargs: dict[str, Any] = {}
        self.embed_kwargs: dict[str, Any] = {}

    async def generate_content(self, **kwargs: Any) -> GenerateContentResponse:
        self.generate_kwargs = kwargs
        return self._response

    async def generate_content_stream(
        self, **kwargs: Any
    ) -> AsyncIterator[GenerateContentResponse]:
        self.generate_kwargs = kwargs
        return self._aiter()

    async def _aiter(self) -> AsyncIterator[GenerateContentResponse]:
        for chunk in self._chunks:
            yield chunk

    async def count_tokens(self, **kwargs: Any) -> CountTokensResponse:
        self.count_kwargs = kwargs
        return CountTokensResponse(total_tokens=self._count)

    async def embed_content(self, **kwargs: Any) -> EmbedContentResponse:
        self.embed_kwargs = kwargs
        return self._embeddings


class _FakeAio:
    def __init__(self, models: _FakeAioModels) -> None:
        self.models = models


class _FakeGenaiClient:
    """Minimal fake of `genai.Client` exposing only what the adapter touches."""

    def __init__(
        self,
        response: GenerateContentResponse,
        chunks: list[GenerateContentResponse],
        count: int,
        embeddings: EmbedContentResponse,
    ) -> None:
        self.aio = _FakeAio(_FakeAioModels(response, chunks, count, embeddings))


def _usage(out: int) -> GenerateContentResponseUsageMetadata:
    return GenerateContentResponseUsageMetadata(
        prompt_token_count=11,
        candidates_token_count=out,
        cached_content_token_count=3,
    )


def _response() -> GenerateContentResponse:
    return GenerateContentResponse(
        model_version="gemini-2.5-pro",
        candidates=[
            Candidate(
                content=Content(role="model", parts=[Part.from_text(text="hello world")]),
                finish_reason=FinishReason.STOP,
            )
        ],
        usage_metadata=_usage(out=6),
    )


def _chunk(
    text: str, finish: FinishReason | None, usage_out: int | None
) -> GenerateContentResponse:
    return GenerateContentResponse(
        model_version="gemini-2.5-pro",
        candidates=[
            Candidate(
                content=Content(role="model", parts=[Part.from_text(text=text)]),
                finish_reason=finish,
            )
        ],
        usage_metadata=_usage(out=usage_out) if usage_out is not None else None,
    )


def _embeddings() -> EmbedContentResponse:
    return EmbedContentResponse(
        embeddings=[
            ContentEmbedding(values=[0.1, 0.2]),
            ContentEmbedding(values=[0.3, 0.4]),
        ]
    )


def _provider(count: int = 42) -> tuple[GoogleProvider, _FakeGenaiClient]:
    chunks = [
        _chunk("hello ", None, None),
        _chunk("world", FinishReason.STOP, 6),
    ]
    fake = _FakeGenaiClient(_response(), chunks, count, _embeddings())
    return GoogleProvider(fake), fake  # type: ignore[arg-type]


def test_satisfies_provider_protocol() -> None:
    provider, _ = _provider()
    assert isinstance(provider, Provider)


async def test_chat_maps_content_and_finish_reason() -> None:
    provider, _ = _provider()
    result = await provider.chat(
        model="gemini-2.5-pro", messages=[Message(role="user", content="hi")]
    )
    assert result.content == "hello world"
    assert result.model == "gemini-2.5-pro"
    assert result.finish_reason == "stop"


async def test_chat_maps_all_four_token_fields() -> None:
    provider, _ = _provider()
    result = await provider.chat(
        model="gemini-2.5-pro", messages=[Message(role="user", content="hi")]
    )
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 6
    # Gemini has no cache-creation charge; cached tokens are cache reads.
    assert result.usage.cache_creation_input_tokens == 0
    assert result.usage.cache_read_input_tokens == 3


async def test_chat_passes_system_instruction() -> None:
    provider, fake = _provider()
    await provider.chat(
        model="gemini-2.5-pro",
        messages=[
            Message(role="system", content="be terse"),
            Message(role="user", content="hi"),
        ],
        max_tokens=128,
        temperature=0.5,
    )
    config = fake.aio.models.generate_kwargs["config"]
    assert config.system_instruction == "be terse"
    assert config.max_output_tokens == 128
    assert config.temperature == 0.5
    # system messages are not forwarded as conversation contents
    contents = fake.aio.models.generate_kwargs["contents"]
    assert all(c.role != "system" for c in contents)


async def test_chat_stream_yields_content_then_terminal_usage() -> None:
    provider, _ = _provider()
    deltas = [
        d
        async for d in provider.chat_stream(
            model="gemini-2.5-pro", messages=[Message(role="user", content="hi")]
        )
    ]
    content = "".join(d.content for d in deltas if d.usage is None)
    assert content == "hello world"

    terminal = deltas[-1]
    assert terminal.finish_reason == "stop"
    assert terminal.usage is not None
    assert terminal.usage.input_tokens == 11
    assert terminal.usage.output_tokens == 6
    assert terminal.usage.cache_creation_input_tokens == 0
    assert terminal.usage.cache_read_input_tokens == 3


async def test_embed_maps_vectors_and_falls_back_for_chat_model() -> None:
    provider, fake = _provider()
    result = await provider.embed(model="gemini-2.5-pro", inputs=["a", "b"])
    assert result.embeddings == [[0.1, 0.2], [0.3, 0.4]]
    # a chat model id → adapter swaps in the default embedding model
    assert fake.aio.models.embed_kwargs["model"] == "text-embedding-004"


async def test_count_tokens_uses_count_endpoint() -> None:
    provider, fake = _provider(count=256)
    count = await provider.count_tokens(
        model="gemini-2.5-pro", messages=[Message(role="user", content="hi")]
    )
    assert count == 256
    assert fake.aio.models.count_kwargs["model"] == "gemini-2.5-pro"


async def test_models_lists_authoritative_ids() -> None:
    provider, _ = _provider()
    assert await provider.models() == ["gemini-2.5-pro", "gemini-2.5-flash"]


@pytest.mark.skipif(
    not os.getenv("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set — real Gemini integration test skipped (offline/CI)",
)
async def test_real_google_chat_maps_usage() -> None:
    provider = GoogleProvider.from_api_key(os.environ["GOOGLE_API_KEY"])
    result = await provider.chat(
        model="gemini-2.5-flash",
        messages=[Message(role="user", content="Reply with exactly: pong")],
        max_tokens=16,
    )
    assert result.content
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
