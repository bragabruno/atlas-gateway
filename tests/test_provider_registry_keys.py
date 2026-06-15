"""GW-3..5 — key-gated provider registration in `ProviderRegistry`.

Confirms the registry's default map is Mock-only when no provider keys are
configured (the default and test environments), and that each real provider is
registered — keyed by every authoritative model id it serves — only when its key
is present. Uses ``Settings`` overrides (no real keys leak into CI); building a
real SDK client from a fake key makes no network call.
"""

from __future__ import annotations

from app.config import Settings
from app.providers.anthropic import ANTHROPIC_MODELS, AnthropicProvider
from app.providers.google import GOOGLE_MODELS, GoogleProvider
from app.providers.openai import OPENAI_MODELS, OpenAIProvider
from app.providers.registry import ProviderRegistry, default_providers


def test_default_registry_is_mock_only_without_keys() -> None:
    """No keys → registry is exactly the pre-wiring Mock-only map."""
    providers = default_providers(Settings(api_keys=("dev-key",)))
    assert list(providers) == ["mock"]


def test_anthropic_models_register_when_key_present() -> None:
    providers = default_providers(Settings(api_keys=("dev-key",), anthropic_api_key="sk-ant-test"))
    for model in ANTHROPIC_MODELS:
        assert isinstance(providers[model], AnthropicProvider)
    # the same adapter instance is shared across that provider's model ids
    assert providers[ANTHROPIC_MODELS[0]] is providers[ANTHROPIC_MODELS[1]]
    # other providers stay absent
    assert OPENAI_MODELS[0] not in providers


def test_openai_models_register_when_key_present() -> None:
    providers = default_providers(Settings(api_keys=("dev-key",), openai_api_key="sk-test"))
    for model in OPENAI_MODELS:
        assert isinstance(providers[model], OpenAIProvider)


def test_google_models_register_when_key_present() -> None:
    providers = default_providers(Settings(api_keys=("dev-key",), google_api_key="g-test"))
    for model in GOOGLE_MODELS:
        assert isinstance(providers[model], GoogleProvider)


def test_all_providers_register_together() -> None:
    providers = default_providers(
        Settings(
            api_keys=("dev-key",),
            anthropic_api_key="sk-ant-test",
            openai_api_key="sk-test",
            google_api_key="g-test",
        )
    )
    expected = {"mock", *ANTHROPIC_MODELS, *OPENAI_MODELS, *GOOGLE_MODELS}
    assert set(providers) == expected


def test_registry_resolve_routes_to_real_provider() -> None:
    registry = ProviderRegistry(
        default_providers(Settings(api_keys=("dev-key",), openai_api_key="sk-test"))
    )
    assert isinstance(registry.resolve(OPENAI_MODELS[0]), OpenAIProvider)
    assert registry.resolve("mock") is not None
    assert registry.resolve("nonexistent-model") is None
