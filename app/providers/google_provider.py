"""GW-5 — GoogleProvider (Gemini).

Wraps the `google-genai` SDK (`google.genai.Client`) to satisfy the `Provider`
protocol. HTTP errors (429/5xx) are re-raised as `TransientProviderError`.
The `embed` path is supported via Gemini's text-embedding models.

See ADR-012 for the provider protocol and ADR-016 for the adapter pattern.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

# `google` is a PEP-420 namespace package with no top-level py.typed marker, so
# pyright reports a missing stub for it even though `google.genai` itself ships
# py.typed and is fully typed. Pin the one unavoidable namespace-level miss here.
from google import genai  # pyright: ignore[reportMissingTypeStubs]
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from app.domain.messages import ChatResult, EmbeddingResult, Message, StreamDelta, Usage
from app.resilience.retry import TransientProviderError

#: Models surfaced by this provider.
_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]

_EMBED_MODELS = [
    "text-embedding-004",
    "gemini-embedding-exp-03-07",
]

_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_transient(exc: genai_errors.APIError) -> bool:
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return code in _TRANSIENT_STATUS_CODES


def _messages_to_contents(messages: list[Message]) -> list[genai_types.Content]:
    """Convert domain Messages to the google-genai typed `contents` list."""
    role_map = {"user": "user", "assistant": "model", "system": "user"}
    return [
        genai_types.Content(
            role=role_map.get(m.role, "user"),
            parts=[genai_types.Part(text=m.content)],
        )
        for m in messages
    ]


def _config(max_tokens: int | None, temperature: float | None) -> genai_types.GenerateContentConfig:
    """Build the typed generation config (both fields are Optional in the SDK)."""
    return genai_types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )


class GoogleProvider:
    """Adapter from the google-genai SDK to the `Provider` protocol."""

    name = "google"

    def __init__(self, *, api_key: str) -> None:
        self._client = genai.Client(api_key=api_key)

    async def chat(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        try:
            # The SDK types `contents` with PIL.Image (an optional image dep we
            # don't install); pyright renders it Unknown → partially-unknown
            # member. The return type stays fully known, so pin just this access.
            resp = await self._client.aio.models.generate_content(  # pyright: ignore[reportUnknownMemberType]
                model=model,
                contents=_messages_to_contents(messages),
                config=_config(max_tokens, temperature),
            )
        except genai_errors.APIError as exc:
            if _is_transient(exc):
                raise TransientProviderError(str(exc)) from exc
            raise
        text = resp.text or ""
        meta = resp.usage_metadata
        candidates = resp.candidates
        finish = str(candidates[0].finish_reason).lower() if candidates else "stop"
        return ChatResult(
            model=model,
            content=text,
            finish_reason=finish,
            usage=Usage(
                input_tokens=(meta.prompt_token_count or 0) if meta else 0,
                output_tokens=(meta.candidates_token_count or 0) if meta else 0,
            ),
        )

    def chat_stream(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamDelta]:
        # NOT `async def` — see OpenAIProvider.chat_stream: calling it must
        # return the async iterator directly, not a coroutine.
        return self._stream(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def _stream(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamDelta]:
        try:
            # See chat(): SDK's PIL-typed `contents` → ignore only this access.
            stream = await self._client.aio.models.generate_content_stream(  # pyright: ignore[reportUnknownMemberType]
                model=model,
                contents=_messages_to_contents(messages),
                config=_config(max_tokens, temperature),
            )
            async for chunk in stream:
                if chunk.text:
                    yield StreamDelta(content=chunk.text)
                meta = chunk.usage_metadata
                if meta and meta.candidates_token_count:
                    candidates = chunk.candidates
                    finish = str(candidates[0].finish_reason).lower() if candidates else "stop"
                    yield StreamDelta(
                        finish_reason=finish,
                        usage=Usage(
                            input_tokens=meta.prompt_token_count or 0,
                            output_tokens=meta.candidates_token_count or 0,
                        ),
                    )
        except genai_errors.APIError as exc:
            if _is_transient(exc):
                raise TransientProviderError(str(exc)) from exc
            raise

    async def embed(self, *, model: str, inputs: list[str]) -> EmbeddingResult:
        try:
            # See chat(): SDK's PIL-typed `contents` → ignore only this access.
            resp = await self._client.aio.models.embed_content(  # pyright: ignore[reportUnknownMemberType]
                model=model,
                contents=inputs,
            )
        except genai_errors.APIError as exc:
            if _is_transient(exc):
                raise TransientProviderError(str(exc)) from exc
            raise
        embeddings = [e.values or [] for e in (resp.embeddings or [])]
        return EmbeddingResult(model=model, embeddings=embeddings, usage=Usage())

    async def models(self) -> list[str]:
        return list(_MODELS)
