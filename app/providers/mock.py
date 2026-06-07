"""GW-2 — MockProvider.

Deterministic, offline provider used by the test suite and CI (zero API spend)
and as a failover-of-last-resort in dev. Output is a pure function of the
input, so tests are reproducible. Implements the non-streaming `chat`, the
streaming `chat_stream`, and the `embed` (GW-8) paths.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator

from app.domain.messages import ChatResult, EmbeddingResult, Message, StreamDelta, Usage

_EMBED_DIM = 8


def _count_tokens(text: str) -> int:
    """Cheap deterministic token estimate (whitespace words, min 1)."""
    return max(1, len(text.split()))


def _reply_for(model: str, messages: list[Message]) -> str:
    last = messages[-1].content if messages else ""
    return f"[mock:{model}] echo: {last}"


def _pseudo_embedding(text: str) -> list[float]:
    """Deterministic unit-scaled pseudo-embedding from a SHA-256 digest.

    Each of the first `_EMBED_DIM` digest bytes maps to a float in [0, 1), so
    the vector is a pure function of the input text (reproducible tests, no API
    spend). This is a stand-in shape, not a semantic embedding.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[i] / 255.0 for i in range(_EMBED_DIM)]


class MockProvider:
    """A deterministic echo provider that structurally satisfies `Provider`."""

    name = "mock"

    async def chat(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        reply = _reply_for(model, messages)
        return ChatResult(
            model=model,
            content=reply,
            finish_reason="stop",
            usage=Usage(
                input_tokens=sum(_count_tokens(m.content) for m in messages),
                output_tokens=_count_tokens(reply),
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
        reply = _reply_for(model, messages)
        for index, word in enumerate(reply.split()):
            yield StreamDelta(content=word if index == 0 else f" {word}")
        yield StreamDelta(
            finish_reason="stop",
            usage=Usage(
                input_tokens=sum(_count_tokens(m.content) for m in messages),
                output_tokens=_count_tokens(reply),
            ),
        )

    async def embed(self, *, model: str, inputs: list[str]) -> EmbeddingResult:
        return EmbeddingResult(
            model=model,
            embeddings=[_pseudo_embedding(text) for text in inputs],
            usage=Usage(input_tokens=sum(_count_tokens(text) for text in inputs)),
        )

    async def models(self) -> list[str]:
        return ["mock"]
