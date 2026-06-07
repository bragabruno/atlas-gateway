"""GW-1 — Provider Protocol + result/usage types.

Every concrete provider (OpenAI, Anthropic, Google, Mock) implements the
`Provider` protocol so the gateway can route, fail over, and account for calls
uniformly. `Usage` carries all four token fields so the cost recorder (GW-14)
can price cached input correctly — see atlas-docs/03.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class Message(BaseModel):
    """A single chat message in the provider-internal representation."""

    role: str
    content: str


class Usage(BaseModel):
    """Token accounting for one provider call.

    The two cache fields are billed differently from `input_tokens`
    (cache-creation ~1.25x, cache-read ~0.1x); they are recorded separately
    rather than folded into the input total.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class ChatResult(BaseModel):
    """Normalized result of a provider chat call."""

    model: str
    content: str
    finish_reason: str = "stop"
    usage: Usage


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

    async def models(self) -> list[str]:
        """Return the model identifiers this provider can serve."""
        ...
