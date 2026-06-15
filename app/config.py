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

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ATLAS_", env_file=".env", extra="ignore")

    #: Deployment environment. Drives env-conditional defaults (today: gates
    #: ``/docs`` + ``/openapi.json`` in prod; future: tighter CORS, debug
    #: logs off). ``dev`` is the default so local runs / tests keep working
    #: without env-var changes; Helm values set ``ATLAS_ENVIRONMENT`` per env.
    environment: Literal["dev", "stage", "prod"] = "dev"

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

    #: Auth source (BRA-881): when on AND db_url is set, bearer keys are validated
    #: against the `api_keys` table (hash + status + expiry), so revocation/expiry
    #: take effect at runtime. Default OFF → the env allowlist (`api_keys`) is
    #: authoritative, which keeps the offline / test path unchanged. Enabling it
    #: requires the keys to be seeded with `app.repositories.api_keys.hash_key`.
    auth_db_enabled: bool = False

    #: Accounting (GW-14/15): when on AND db_url is set, every non-streaming
    #: completion writes a priced `call_records` row; with Kafka also configured
    #: each row fans out as an `atlas.calls.v1` event. Default OFF.
    accounting_enabled: bool = False

    #: Kafka bootstrap for the accounting event stream (e.g.
    #: ``redpanda:9092`` in the local compose loop). ``None`` (default) means
    #: records persist to Postgres only — no producer is ever constructed.
    kafka_bootstrap_servers: str | None = None

    #: PostgreSQL DSN for the gateway DB (call_records etc.).  ``None`` disables
    #: the DB-backed usage endpoint; a real deployment sets ATLAS_DB_URL from
    #: Key Vault.
    db_url: str | None = None

    #: Browser origins allowed to call the gateway via CORS (e.g. the Atlas
    #: frontend SPA). Empty (default) adds **no** CORS middleware — byte-for-byte
    #: the pre-CORS behaviour, same config-gated/default-OFF philosophy as above.
    #: Set ATLAS_CORS_ALLOW_ORIGINS (JSON list) to enable, e.g.
    #: ``["http://localhost:8080"]`` for the local compose frontend.
    cors_allow_origins: tuple[str, ...] = ()

    #: Provider API keys — sourced from Key Vault via CSI in real deployments,
    #: set as env vars locally. A real provider (GW-3..5) is registered only when
    #: its key is present; with the defaults (no keys) the registry stays
    #: Mock-only, exactly the pre-wiring behaviour the default and test
    #: environments rely on.
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None

    #: Local Ollama (OpenAI-compatible) endpoint. When set, the registry wires an
    #: OpenAI-protocol provider at this base URL — no API key needed (Ollama
    #: ignores it). Example (gateway-in-Docker → host Ollama):
    #: ``http://host.docker.internal:11434/v1``. ``ollama_models`` are the model
    #: ids served (e.g. ``["gpt-oss:120b-cloud"]``); each is registered so a
    #: chat request for that id routes to Ollama instead of the mock provider.
    ollama_base_url: str | None = None
    ollama_models: tuple[str, ...] = ()

    #: Streamable-HTTP endpoint of the ``mcp-citations`` server (AGT-12). ``None``
    #: (default) means no live citation verifier is built, so the GRD-9 citation
    #: guardrail keeps its stub/unwired default and the request path is unchanged.
    #: A real deployment sets ATLAS_CITATION_MCP_URL so the composition root can
    #: build an `McpToolClient` and inject the live `McpCitationVerifier`.
    citation_mcp_url: str | None = None


def get_settings() -> Settings:
    return Settings()
