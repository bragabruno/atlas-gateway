"""GW-5 — GoogleProvider (Gemini).

Wraps the `google-genai` SDK (`google.genai.Client`) to satisfy the `Provider`
protocol. HTTP errors (429/5xx) are re-raised as `TransientProviderError`.
The `embed` path is supported via Gemini's text-embedding models.

See ADR-012 for the provider protocol and ADR-016 for the adapter pattern.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from google import genai
from google.genai import errors as genai_errors

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


def _messages_to_contents(messages: list[Message]) -> list[dict]:
    """Convert domain Messages to the google-genai `contents` format."""
    role_map = {"user": "user", "assistant": "model", "system": "user"}
    return [{"role": role_map.get(m.role, "user"), "parts": [{"text": m.content}]} for m in messages]


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
        config: dict = {}
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        if temperature is not None:
            config["temperature"] = temperature
        try:
            resp = await self._client.aio.models.generate_content(
                model=model,
                contents=_messages_to_contents(messages),
                config=config or None,
            )
        except genai_errors.APIError as exc:
            if _is_transient(exc):
                raise TransientProviderError(str(exc)) from exc
            raise
        text = resp.text or ""
        meta = resp.usage_metadata
        return ChatResult(
            model=model,
            content=text,
            finish_reason=str(resp.candidates[0].finish_reason).lower() if resp.candidates else "stop",
            usage=Usage(
                input_tokens=meta.prompt_token_count if meta else 0,
                output_tokens=meta.candidates_token_count if meta else 0,
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
        config: dict = {}
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        if temperature is not None:
            config["temperature"] = temperature
        try:
            async for chunk in await self._client.aio.models.generate_content_stream(
                model=model,
                contents=_messages_to_contents(messages),
                config=config or None,
            ):
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
            resp = await self._client.aio.models.embed_content(
                model=model,
                contents=inputs,
            )
        except genai_errors.APIError as exc:
            if _is_transient(exc):
                raise TransientProviderError(str(exc)) from exc
            raise
        embeddings = [e.values for e in resp.embeddings]
        return EmbeddingResult(model=model, embeddings=embeddings, usage=Usage())

    async def models(self) -> list[str]:
        return list(_MODELS)
