"""Provider registry — the composition point mapping model/alias → adapter.

Analogous to a bean registry: concrete providers register here and the service
layer resolves them through `ProviderRegistry`, so adding a real provider
(GW-3..5) touches neither the service nor the controller. The Mock provider is
always present; the real providers (Anthropic/OpenAI/Google) are wired in,
keyed by their authoritative model ids, **only when their API key is configured**
(`app.config`). With no keys (the default and test environments) the registry
stays Mock-only, byte-for-byte the pre-wiring behaviour. Alias routing (GW-10)
resolves aliases to entries here. See ADR-012 + ADR-016.
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.providers.anthropic import ANTHROPIC_MODELS, AnthropicProvider
from app.providers.base import Provider
from app.providers.google import GOOGLE_MODELS, GoogleProvider
from app.providers.mock import MockProvider
from app.providers.openai import OPENAI_MODELS, OpenAIProvider


def default_providers(settings: Settings | None = None) -> dict[str, Provider]:
    """Build the default model → adapter map.

    Mock is always present. Each real provider is added — keyed by every model id
    it serves — only when its API key is configured, so the default/test env (no
    keys) yields exactly ``{"mock": MockProvider()}``. The provider's SDK client
    is constructed from the key here; no network call is made at registration.
    """
    cfg = settings if settings is not None else get_settings()
    providers: dict[str, Provider] = {"mock": MockProvider()}

    if cfg.anthropic_api_key:
        anthropic = AnthropicProvider.from_api_key(cfg.anthropic_api_key)
        for model in ANTHROPIC_MODELS:
            providers[model] = anthropic

    if cfg.openai_api_key:
        openai = OpenAIProvider.from_api_key(cfg.openai_api_key)
        for model in OPENAI_MODELS:
            providers[model] = openai

    if cfg.google_api_key:
        google = GoogleProvider.from_api_key(cfg.google_api_key)
        for model in GOOGLE_MODELS:
            providers[model] = google

    return providers


class ProviderRegistry:
    """Holds the model → `Provider` adapter map and resolves by model id."""

    def __init__(self, providers: dict[str, Provider] | None = None) -> None:
        self._providers: dict[str, Provider] = (
            providers if providers is not None else default_providers()
        )

    def resolve(self, model: str) -> Provider | None:
        return self._providers.get(model)

    def names(self) -> list[str]:
        return list(self._providers)
