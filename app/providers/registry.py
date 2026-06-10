"""Provider registry — the composition point mapping model/alias → adapter.

Analogous to a bean registry: concrete providers register here and the service
layer resolves them through `ProviderRegistry`, so adding a real provider
touches neither the service nor the controller. Alias routing (GW-10) resolves
aliases to entries here. See ADR-012 + ADR-016.

GW-3/4/5: `from_settings` constructs the registry from runtime config — a
provider is wired only when its API key env var is present; mock is always
available as the offline fallback.
"""

from __future__ import annotations

from app.providers.base import Provider
from app.providers.mock import MockProvider


class ProviderRegistry:
    """Holds the model → `Provider` adapter map and resolves by model id."""

    def __init__(self, providers: dict[str, Provider] | None = None) -> None:
        self._providers: dict[str, Provider] = (
            providers if providers is not None else {"mock": MockProvider()}
        )

    @classmethod
    def from_settings(cls, settings: object) -> "ProviderRegistry":
        """Build a registry from runtime `Settings` — wire only present providers."""
        from app.providers.anthropic_provider import AnthropicProvider
        from app.providers.openai_provider import OpenAIProvider

        providers: dict[str, Provider] = {"mock": MockProvider()}
        if key := getattr(settings, "anthropic_api_key", None):
            p = AnthropicProvider(api_key=key)
            for m in ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5",
                      "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
                      "claude-3-opus-20240229"]:
                providers[m] = p
        if key := getattr(settings, "openai_api_key", None):
            p = OpenAIProvider(api_key=key)
            for m in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",
                      "text-embedding-3-small", "text-embedding-3-large",
                      "text-embedding-ada-002"]:
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
                for m in ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash",
                          "gemini-1.5-pro", "gemini-1.5-flash", "text-embedding-004",
                          "gemini-embedding-exp-03-07"]:
                    providers[m] = p
            except ImportError:
                pass  # google-genai not installed in this environment
        return cls(providers)

    def resolve(self, model: str) -> Provider | None:
        return self._providers.get(model)

    def names(self) -> list[str]:
        return list(self._providers)
