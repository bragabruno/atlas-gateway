"""Provider port — the behavioural interface every adapter implements.

The data contracts (`Message`, `Usage`, `ChatResult`, `StreamDelta`) live in
`app.domain.messages`; this module defines only the `Provider` protocol so the
service layer depends on an interface, not a concrete provider. Concrete
adapters (OpenAI, Anthropic, Google, Mock) satisfy it structurally. See
atlas-docs/03 + ADR-012.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from app.domain.messages import ChatResult, EmbeddingResult, Message, StreamDelta


@runtime_checkable
class Provider(Protocol):
    """Uniform async interface implemented by every provider."""

    name: str

    async def chat(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        """Run a chat completion and return a normalized `ChatResult`."""
        ...

    def chat_stream(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """Stream a chat completion as incremental `StreamDelta`s."""
        ...

    async def embed(self, *, model: str, inputs: list[str]) -> EmbeddingResult:
        """Embed each input string and return a normalized `EmbeddingResult`."""
        ...

    async def models(self) -> list[str]:
        """Return the model identifiers this provider can serve."""
        ...
