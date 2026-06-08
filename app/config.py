"""Runtime configuration (env-driven, no secrets in code).

Per-key auth uses an allowlist; override via the ATLAS_API_KEYS env var
(JSON list) in real deployments — secrets come from Key Vault via the CSI
driver, never from the image. The default dev key exists only for local tests.

Capability wiring is **config-gated and default OFF**: the request-path
collaborators (cache, rate-limit, budget, accounting, guardrails) are only
constructed in `app.api.deps` when their backing config is present. With the
defaults below (no Redis URL, flags off) `get_chat_service` builds a Mock-only
`ChatService` with zero external dependencies, identical to the pre-wiring
behaviour — which is exactly what the default and test environments run.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATLAS_", env_file=".env", extra="ignore")

    api_keys: tuple[str, ...] = ("dev-key",)

    #: Redis connection URL for the cache, rate limiter, budget enforcer, and
    #: circuit breaker. ``None`` (default) means no Redis is available, so every
    #: Redis-backed collaborator stays unconstructed/inert (the test env has no
    #: Redis). A real deployment sets ATLAS_REDIS_URL from Key Vault.
    redis_url: str | None = None

    #: Per-feature gates, all default OFF. Even with a Redis URL present a
    #: collaborator is only wired when its flag is on, so each capability can be
    #: rolled out independently. The default path (all off) is byte-for-byte the
    #: pre-wiring gateway.
    cache_enabled: bool = False
    rate_limit_enabled: bool = False
    budget_enabled: bool = False
    guardrails_enabled: bool = False

    #: Provider API keys, all default ``None`` (Key-Vault-injected in real
    #: deployments). A real provider (GW-3..5) is registered only when its key is
    #: present; with the defaults (no keys) the registry stays Mock-only, exactly
    #: the pre-wiring behaviour the default and test environments rely on.
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None

    #: Streamable-HTTP endpoint of the ``mcp-citations`` server (AGT-12). ``None``
    #: (default) means no live citation verifier is built, so the GRD-9 citation
    #: guardrail keeps its stub/unwired default and the request path is unchanged.
    #: A real deployment sets ATLAS_CITATION_MCP_URL so the composition root can
    #: build an `McpToolClient` and inject the live `McpCitationVerifier`.
    citation_mcp_url: str | None = None


def get_settings() -> Settings:
    return Settings()
