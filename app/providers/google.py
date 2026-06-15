"""GW-5 — GoogleProvider (Gemini).

Adapter over the official ``google-genai`` SDK that satisfies the `Provider`
protocol. Maps Gemini's `GenerateContentResponseUsageMetadata` onto our
four-field `Usage`: ``prompt_token_count`` → ``input_tokens``,
``candidates_token_count`` → ``output_tokens``, ``cached_content_token_count`` →
``cache_read_input_tokens`` (Gemini has no separate cache-creation charge, so
``cache_creation_input_tokens`` is 0). Token counting uses the provider's own
``count_tokens`` endpoint (never tiktoken; see atlas-docs/02 + ADR-012).

The SDK client is **injected** (`client` arg), so offline unit tests pass a fake
that returns canned SDK objects (no network, no key) and assert the
response/usage mapping. `from_api_key` builds a real `genai.Client`.
Authoritative model ids: ``gemini-*`` (atlas-docs/02).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from google.genai import Client
from google.genai.types import (
    Content,
    CountTokensResponse,
    EmbedContentResponse,
    GenerateContentConfig,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    Part,
)

from app.domain.messages import ChatResult, EmbeddingResult, Message, StreamDelta, Usage

#: Authoritative Gemini model ids served by this provider (atlas-docs/02).
GOOGLE_MODELS: tuple[str, ...] = (
    "gemini-2.5-pro",
    "gemini-2.5-flash",
)

#: Default embedding model used when a caller does not pin one explicitly.
_DEFAULT_EMBED_MODEL = "text-embedding-004"


def _to_contents(messages: list[Message]) -> list[Content]:
    """Map non-system `Message`s to Gemini `Content`s (Gemini role is user/model)."""
    contents: list[Content] = []
    for m in messages:
        if m.role == "system":
            continue
        role = "model" if m.role == "assistant" else "user"
        contents.append(Content(role=role, parts=[Part.from_text(text=m.content)]))
    return contents


def _system_instruction(messages: list[Message]) -> str | None:
    """Concatenate any system messages into the Gemini system_instruction field."""
    systems = [m.content for m in messages if m.role == "system"]
    return "\n\n".join(systems) if systems else None


def _config(
    messages: list[Message], max_tokens: int | None, temperature: float | None
) -> GenerateContentConfig:
    """Build the Gemini generation config (system instruction + sampling caps)."""
    return GenerateContentConfig(
        system_instruction=_system_instruction(messages),
        max_output_tokens=max_tokens,
        temperature=temperature,
    )


def _map_usage(usage: GenerateContentResponseUsageMetadata | None) -> Usage:
    """Map Gemini usage metadata onto our four-field `Usage`.

    Gemini bills cached input at a discount but has no cache-creation charge, so
    ``cache_creation_input_tokens`` is always 0 and the cached prompt tokens map
    to ``cache_read_input_tokens``.
    """
    if usage is None:
        return Usage()
    return Usage(
        input_tokens=usage.prompt_token_count or 0,
        output_tokens=usage.candidates_token_count or 0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=usage.cached_content_token_count or 0,
    )


def _finish_reason(response: GenerateContentResponse) -> str:
    """Lowercase the candidate finish reason, defaulting to ``stop``."""
    if response.candidates and response.candidates[0].finish_reason is not None:
        return response.candidates[0].finish_reason.name.lower()
    return "stop"


class GoogleProvider:
    """Google Gemini adapter that structurally satisfies `Provider`."""

    name = "google"

    def __init__(self, client: Client) -> None:
        """Hold an injected `genai.Client` (real or fake for tests)."""
        self._client = client

    @classmethod
    def from_api_key(cls, api_key: str) -> GoogleProvider:
        """Build a provider backed by a real `genai.Client`."""
        return cls(Client(api_key=api_key))

    async def chat(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        # The SDK's `contents` param union includes an untyped (optional PIL)
        # member, so pyright marks the bound method partially-unknown; the return
        # type is fully typed, so we pin it with an explicit annotation.
        response: GenerateContentResponse = await self._client.aio.models.generate_content(  # type: ignore[reportUnknownMemberType]
            model=model,
            contents=_to_contents(messages),
            config=_config(messages, max_tokens, temperature),
        )
        return ChatResult(
            model=response.model_version or model,
            content=response.text or "",
            finish_reason=_finish_reason(response),
            usage=_map_usage(response.usage_metadata),
        )

    async def chat_stream(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamDelta]:
        finish_reason = "stop"
        usage: GenerateContentResponseUsageMetadata | None = None

        stream: AsyncIterator[
            GenerateContentResponse
        ] = await self._client.aio.models.generate_content_stream(  # type: ignore[reportUnknownMemberType]
            model=model,
            contents=_to_contents(messages),
            config=_config(messages, max_tokens, temperature),
        )
        async for chunk in stream:
            if chunk.usage_metadata is not None:
                usage = chunk.usage_metadata
            if chunk.text:
                yield StreamDelta(content=chunk.text)
            if chunk.candidates and chunk.candidates[0].finish_reason is not None:
                finish_reason = chunk.candidates[0].finish_reason.name.lower()

        yield StreamDelta(finish_reason=finish_reason, usage=_map_usage(usage))

    async def embed(self, *, model: str, inputs: list[str]) -> EmbeddingResult:
        # Chat model ids are not embedding models; fall back to the default
        # embedding model unless the caller pins a real embedding model.
        embed_model = _DEFAULT_EMBED_MODEL if model in GOOGLE_MODELS else model
        response: EmbedContentResponse = await self._client.aio.models.embed_content(  # type: ignore[reportUnknownMemberType]
            model=embed_model, contents=inputs
        )
        embeddings = [list(e.values or []) for e in (response.embeddings or [])]
        return EmbeddingResult(
            model=embed_model,
            embeddings=embeddings,
            usage=Usage(),
        )

    async def count_tokens(self, *, model: str, messages: list[Message]) -> int:
        """Count input tokens via the provider's own ``count_tokens`` endpoint."""
        result: CountTokensResponse = await self._client.aio.models.count_tokens(  # type: ignore[reportUnknownMemberType]
            model=model, contents=_to_contents(messages)
        )
        return result.total_tokens or 0

    async def models(self) -> list[str]:
        return list(GOOGLE_MODELS)
