"""GW-2 — MockProvider.

Deterministic, offline provider used by the test suite and CI (zero API spend)
and as a failover-of-last-resort in dev. Output is a pure function of the
input, so tests are reproducible.
"""

from __future__ import annotations

from app.providers.base import ChatResult, Message, Usage


def _count_tokens(text: str) -> int:
    """Cheap deterministic token estimate (whitespace words, min 1)."""
    return max(1, len(text.split()))


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
        last = messages[-1].content if messages else ""
        reply = f"[mock:{model}] echo: {last}"
        input_tokens = sum(_count_tokens(m.content) for m in messages)
        output_tokens = _count_tokens(reply)
        return ChatResult(
            model=model,
            content=reply,
            finish_reason="stop",
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        )

    async def models(self) -> list[str]:
        return ["mock"]
