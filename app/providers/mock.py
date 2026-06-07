"""GW-2 — MockProvider.

Deterministic, offline provider used by the test suite and CI (zero API spend)
and as a failover-of-last-resort in dev. Output is a pure function of the
input, so tests are reproducible. Implements both the non-streaming `chat` and
the streaming `chat_stream` paths.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.providers.base import ChatResult, Message, StreamDelta, Usage


def _count_tokens(text: str) -> int:
    """Cheap deterministic token estimate (whitespace words, min 1)."""
    return max(1, len(text.split()))


def _reply_for(model: str, messages: list[Message]) -> str:
    last = messages[-1].content if messages else ""
    return f"[mock:{model}] echo: {last}"


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

    async def models(self) -> list[str]:
        return ["mock"]
