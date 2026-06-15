"""Provider registry — the composition point mapping model/alias → adapter.

Analogous to a bean registry: concrete providers register here and the service
layer resolves them through `ProviderRegistry`, so adding a real provider
(GW-3..5) touches neither the service nor the controller. The Mock provider is
always present; the real providers (Anthropic/OpenAI/Google) are wired in,
keyed by their authoritative model ids, **only when their API key is configured**
(`app.config`). With no keys (the default and test environments) the registry
stays Mock-only, byte-for-byte the pre-wiring behaviour. Alias routing (GW-10)
resolves aliases to entries here. See ADR-012 + ADR-016.

Two construction paths coexist: the module-level `default_providers` (used by
``ProviderRegistry()``) wires the `anthropic`/`openai`/`google` adapters, while
`ProviderRegistry.from_settings` wires the `*_provider` adapters and the local
Ollama endpoint from runtime config — each provider added only when its API key
env var is present; mock is always available as the offline fallback.
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

    @classmethod
    def from_settings(cls, settings: object) -> ProviderRegistry:
        """Build a registry from runtime `Settings` — wire only present providers."""
        from app.providers.anthropic_provider import AnthropicProvider
        from app.providers.openai_provider import OpenAIProvider

        providers: dict[str, Provider] = {"mock": MockProvider()}
        if key := getattr(settings, "anthropic_api_key", None):
            p = AnthropicProvider(api_key=key)
            for m in [
                "claude-opus-4-5",
                "claude-sonnet-4-5",
                "claude-haiku-4-5",
                "claude-3-5-sonnet-20241022",
                "claude-3-5-haiku-20241022",
                "claude-3-opus-20240229",
            ]:
                providers[m] = p
        if key := getattr(settings, "openai_api_key", None):
            p = OpenAIProvider(api_key=key)
            for m in [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ]:
                providers[m] = p
        # Local Ollama (OpenAI-compatible): one provider at ollama_base_url, keyed
        # by each served model id. No API key is required — Ollama ignores it, but
        # the OpenAI SDK demands a non-empty string, so a placeholder is passed.
        if base_url := getattr(settings, "ollama_base_url", None):
            ollama = OpenAIProvider(api_key="ollama", base_url=base_url)
            for m in getattr(settings, "ollama_models", ()) or ():
                providers[m] = ollama
        if key := getattr(settings, "google_api_key", None):
            try:
                from app.providers.google_provider import GoogleProvider as _GP

                p = _GP(api_key=key)
                for m in [
                    "gemini-2.5-pro",
                    "gemini-2.5-flash",
                    "gemini-2.0-flash",
                    "gemini-1.5-pro",
                    "gemini-1.5-flash",
                    "text-embedding-004",
                    "gemini-embedding-exp-03-07",
                ]:
                    providers[m] = p
            except ImportError:
                pass  # google-genai not installed in this environment
        return cls(providers)

    def resolve(self, model: str) -> Provider | None:
        return self._providers.get(model)

    def names(self) -> list[str]:
        return list(self._providers)
