"""GW-4 — OpenAIProvider tests.

Offline unit tests drive the provider with a **fake** `AsyncOpenAI` client (no
network, no key) built from real SDK objects, asserting the response/usage
mapping — including the four token fields: ``prompt_tokens`` →
``input_tokens``, ``completion_tokens`` → ``output_tokens``, cached prompt
tokens → ``cache_read_input_tokens`` (and ``cache_creation_input_tokens`` == 0,
since OpenAI has no cache-creation charge). The real integration test is
**key-gated**: it skips (never fails) when ``OPENAI_API_KEY`` is absent.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any, Literal

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.create_embedding_response import Usage as EmbeddingUsage
from openai.types.embedding import Embedding
from openai.types.responses.input_token_count_response import InputTokenCountResponse

from app.domain.messages import Message
from app.providers.base import Provider
from app.providers.openai import OpenAIProvider


class _FakeCompletions:
    def __init__(self, completion: ChatCompletion, chunks: list[ChatCompletionChunk]) -> None:
        self._completion = completion
        self._chunks = chunks
        self.create_kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> object:
        self.create_kwargs = kwargs
        if kwargs.get("stream"):
            return self._aiter()
        return self._completion

    async def _aiter(self) -> AsyncIterator[ChatCompletionChunk]:
        for chunk in self._chunks:
            yield chunk


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeEmbeddings:
    def __init__(self, response: CreateEmbeddingResponse) -> None:
        self._response = response
        self.create_kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> CreateEmbeddingResponse:
        self.create_kwargs = kwargs
        return self._response


class _FakeInputTokens:
    def __init__(self, count: int) -> None:
        self._count = count
        self.count_kwargs: dict[str, Any] = {}

    async def count(self, **kwargs: Any) -> InputTokenCountResponse:
        self.count_kwargs = kwargs
        return InputTokenCountResponse(input_tokens=self._count, object="response.input_tokens")


class _FakeResponses:
    def __init__(self, input_tokens: _FakeInputTokens) -> None:
        self.input_tokens = input_tokens


class _FakeOpenAI:
    """Minimal fake of `AsyncOpenAI` exposing only what the adapter touches."""

    def __init__(
        self,
        completion: ChatCompletion,
        chunks: list[ChatCompletionChunk],
        embeddings: CreateEmbeddingResponse,
        count: int,
    ) -> None:
        self.chat = _FakeChat(_FakeCompletions(completion, chunks))
        self.embeddings = _FakeEmbeddings(embeddings)
        self.responses = _FakeResponses(_FakeInputTokens(count))


def _usage() -> CompletionUsage:
    return CompletionUsage(
        prompt_tokens=12,
        completion_tokens=8,
        total_tokens=20,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=4),
    )


def _completion() -> ChatCompletion:
    return ChatCompletion(
        id="cmpl_1",
        object="chat.completion",
        created=1,
        model="gpt-4.1",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content="hello world"),
            )
        ],
        usage=_usage(),
    )


def _chunks() -> list[ChatCompletionChunk]:
    def content_chunk(text: str, finish: Literal["stop"] | None) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id="cmpl_1",
            object="chat.completion.chunk",
            created=1,
            model="gpt-4.1",
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(content=text), finish_reason=finish)],
            usage=None,
        )

    usage_chunk = ChatCompletionChunk(
        id="cmpl_1",
        object="chat.completion.chunk",
        created=1,
        model="gpt-4.1",
        choices=[],
        usage=_usage(),
    )
    return [content_chunk("hello ", None), content_chunk("world", "stop"), usage_chunk]


def _embedding_response() -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse(
        object="list",
        model="text-embedding-3-small",
        data=[
            Embedding(object="embedding", index=0, embedding=[0.1, 0.2]),
            Embedding(object="embedding", index=1, embedding=[0.3, 0.4]),
        ],
        usage=EmbeddingUsage(prompt_tokens=5, total_tokens=5),
    )


def _provider(count: int = 42) -> tuple[OpenAIProvider, _FakeOpenAI]:
    fake = _FakeOpenAI(_completion(), _chunks(), _embedding_response(), count)
    return OpenAIProvider(fake), fake  # type: ignore[arg-type]


def test_satisfies_provider_protocol() -> None:
    provider, _ = _provider()
    assert isinstance(provider, Provider)


async def test_chat_maps_content_and_finish_reason() -> None:
    provider, _ = _provider()
    result = await provider.chat(model="gpt-4.1", messages=[Message(role="user", content="hi")])
    assert result.content == "hello world"
    assert result.model == "gpt-4.1"
    assert result.finish_reason == "stop"


async def test_chat_maps_all_four_token_fields() -> None:
    provider, _ = _provider()
    result = await provider.chat(model="gpt-4.1", messages=[Message(role="user", content="hi")])
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 8
    # OpenAI has no cache-creation charge; cached prompt tokens are cache reads.
    assert result.usage.cache_creation_input_tokens == 0
    assert result.usage.cache_read_input_tokens == 4


async def test_chat_stream_yields_content_then_terminal_usage() -> None:
    provider, fake = _provider()
    deltas = [
        d
        async for d in provider.chat_stream(
            model="gpt-4.1", messages=[Message(role="user", content="hi")]
        )
    ]
    content = "".join(d.content for d in deltas if d.usage is None)
    assert content == "hello world"

    terminal = deltas[-1]
    assert terminal.finish_reason == "stop"
    assert terminal.usage is not None
    assert terminal.usage.input_tokens == 12
    assert terminal.usage.output_tokens == 8
    assert terminal.usage.cache_creation_input_tokens == 0
    assert terminal.usage.cache_read_input_tokens == 4
    # usage must be requested explicitly so the final chunk carries it
    assert fake.chat.completions.create_kwargs["stream_options"] == {"include_usage": True}


async def test_embed_maps_vectors_and_falls_back_for_chat_model() -> None:
    provider, fake = _provider()
    result = await provider.embed(model="gpt-4.1", inputs=["a", "b"])
    assert result.embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert result.usage.input_tokens == 5
    # gpt-4.1 is a chat model → adapter swaps in the default embedding model
    assert fake.embeddings.create_kwargs["model"] == "text-embedding-3-small"


async def test_count_tokens_uses_responses_endpoint() -> None:
    provider, fake = _provider(count=321)
    count = await provider.count_tokens(
        model="gpt-4.1", messages=[Message(role="user", content="hi")]
    )
    assert count == 321
    assert fake.responses.input_tokens.count_kwargs["model"] == "gpt-4.1"


async def test_models_lists_authoritative_ids() -> None:
    provider, _ = _provider()
    assert await provider.models() == ["gpt-4.1"]


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — real OpenAI integration test skipped (offline/CI)",
)
async def test_real_openai_chat_maps_usage() -> None:
    provider = OpenAIProvider.from_api_key(os.environ["OPENAI_API_KEY"])
    result = await provider.chat(
        model="gpt-4.1",
        messages=[Message(role="user", content="Reply with exactly: pong")],
        max_tokens=16,
    )
    assert result.content
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
