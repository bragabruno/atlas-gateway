"""Provider-internal domain types: messages, token usage, results, deltas.

These are the contracts the service and provider layers exchange. They carry no
framework dependency. The external OpenAI-compatible wire schema lives in
`app.domain.openai`; these are the internal representation. `Usage` keeps all
four token fields so the cost recorder (GW-14) can price cached input
correctly. See atlas-docs/03 + ADR-012.
"""

from __future__ import annotations

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
    """Normalized result of a non-streaming provider chat call."""

    model: str
    content: str
    finish_reason: str = "stop"
    usage: Usage


class StreamDelta(BaseModel):
    """One increment of a streaming provider response.

    Content deltas carry `content`; the terminal delta carries `finish_reason`
    and the final `usage` (and no content).
    """

    content: str = ""
    finish_reason: str | None = None
    usage: Usage | None = None


class EmbeddingResult(BaseModel):
    """Normalized result of a provider embedding call (GW-8).

    Carries one vector per input string (order-preserving) plus token `usage`
    so the cost recorder (GW-14) can price embedding calls uniformly.
    """

    model: str
    embeddings: list[list[float]]
    usage: Usage
