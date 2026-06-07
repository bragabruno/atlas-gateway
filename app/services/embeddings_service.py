"""GW-8 — embeddings use-case orchestration (service layer).

Resolves the provider for an embeddings request, drives the provider `embed`
call, and maps the provider-internal `EmbeddingResult` to the OpenAI-compatible
wire schema. The controller (`app.api.v1.embeddings`) stays thin — all
embedding business logic lives here. Caching (GW-13) and accounting (GW-14)
layer in as collaborators, not in the controller. See ADR-016.
"""

from __future__ import annotations

from app.domain.errors import UnknownModelError
from app.domain.openai import (
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
)
from app.providers.base import Provider
from app.providers.registry import ProviderRegistry


class EmbeddingsService:
    """Orchestrates embeddings over the provider registry."""

    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    def _resolve(self, model: str) -> Provider:
        provider = self._registry.resolve(model)
        if provider is None:
            raise UnknownModelError(model)
        return provider

    @staticmethod
    def _inputs(req: EmbeddingRequest) -> list[str]:
        return [req.input] if isinstance(req.input, str) else list(req.input)

    async def embed(self, req: EmbeddingRequest) -> EmbeddingResponse:
        """Run an embedding call and return the OpenAI-shaped response."""
        provider = self._resolve(req.model)
        result = await provider.embed(model=req.model, inputs=self._inputs(req))
        return EmbeddingResponse(
            data=[
                EmbeddingData(index=index, embedding=vector)
                for index, vector in enumerate(result.embeddings)
            ],
            model=result.model,
            usage=EmbeddingUsage(
                prompt_tokens=result.usage.input_tokens,
                total_tokens=result.usage.input_tokens,
            ),
        )
