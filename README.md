# atlas-gateway

OpenAI-compatible LLM Gateway for the Atlas platform. Routes chat, embedding, and model-list requests across multiple AI providers with per-key auth, rate limiting, budget enforcement, caching, circuit breaking, and full OpenTelemetry instrumentation.

## Requirements

- Python 3.12
- PostgreSQL, Redis, Qdrant, Kafka (see `atlas-docs` for infrastructure context)
- All pinned dependencies ≥ 14 days old; no secrets in code or images (env-var placeholders only)

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Non-streaming and SSE streaming (chat.completion.chunk + `data: [DONE]`) |
| GET | `/v1/models` | List available model aliases and provider models |
| POST | `/v1/embeddings` | Generate text embeddings |

Authentication: Bearer token per-key in `Authorization` header.

## Architecture: layered + DI (ADR-016)

The service follows a layered spine — **controllers → services → repositories → domain** — with the capability modules (providers, cache, limits, …) as adapters the service layer composes. FastAPI `Depends` is the DI container; `app/api/deps.py` is the composition root. This is "Spring-style" separation without a heavyweight framework: HTTP shape lives only in `api/`, business logic only in `services/`, DB access only in `repositories/`, and the framework-free contracts in `domain/`. See [`../atlas-docs/02-tech-stack-and-adrs.md`](../atlas-docs/02-tech-stack-and-adrs.md) (ADR-016) and [`../atlas-docs/research/framework-evaluation.md`](../atlas-docs/research/framework-evaluation.md) §5.2.

## Module Map (`app/`)

```
app/
├── main.py              # FastAPI app factory + /healthz (composition root)
├── config.py            # Settings (env/Key Vault, no secrets in code)
│
├── api/                 # ── Controllers: HTTP only (parse, auth, serialize) ──
│   ├── deps.py          # DI providers: settings, auth, registry, services
│   └── v1/
│       ├── chat.py      # POST /v1/chat/completions (stream + non-stream) → ChatService
│       ├── models.py    # GET /v1/models
│       └── embeddings.py# POST /v1/embeddings
│
├── services/            # ── Service layer: the only home of business logic ──
│   └── chat_service.py  # resolve provider → call → map usage → response/SSE frames
│
├── repositories/        # ── Persistence: asyncpg (hot path) / SQLAlchemy (ADR-010) ──
│   └── (GW-9 schema, GW-14 accounting)
│
├── domain/              # ── Contracts (no framework deps) ──
│   ├── messages.py      # Message, Usage (4 token fields), ChatResult, StreamDelta
│   ├── openai.py        # OpenAI-compatible request/response/chunk wire schema
│   └── errors.py        # UnknownModelError (→ 404 in the controller)
│
└── (capability adapters the services compose)
    ├── providers/
    │   ├── base.py      # Provider port (Protocol)
    │   ├── registry.py  # ProviderRegistry: model/alias → adapter
    │   ├── openai.py · anthropic.py · google.py  # real adapters (GW-3..5)
    │   └── mock.py      # deterministic offline adapter (testing)
    ├── routing/         # aliases.py — AliasResolver: alias→primary+fallback (GW-10)
    ├── resilience/      # retry.py (tenacity) · circuit_breaker.py (Redis, per-provider)
    ├── cache/           # exact.py (Redis) · semantic.py (Qdrant 0.97, opt-in)
    ├── accounting/      # recorder.py — call_records cost formula → Kafka atlas.calls.v1
    ├── limits/          # ratelimit.py (token-bucket→429) · budget.py (monthly→429 +80%)
    ├── guardrails/      # chain.py — pre/post middleware chain
    ├── telemetry/       # otel.py — GenAI semconv → OTel Collector → Splunk
    └── prompts/         # registry.py — resolve prompt_ref → rendered config
```

### Model Aliases

| Alias | Primary | Fallback |
|-------|---------|----------|
| `smart` | claude-sonnet-4-6 | gpt-4.1 |
| `deep` | claude-opus-4-8 | — |
| `fast` | claude-haiku-4-5 | — |
| `balanced` | gemini-* | — |
| `embed` | embeddings | — |

### Cost Accounting

Token costs are computed from `Usage`:
- `input_tokens` × base rate
- `output_tokens` × base rate
- `cache_creation_input_tokens` × 1.25×
- `cache_read_input_tokens` × 0.1×

Records written to PostgreSQL `call_records`; events published to Kafka topic `atlas.calls.v1`.

## Diagrams

| Diagram | Description |
|---------|-------------|
| [Component C4 L3](docs/diagrams/component-c4.md) | Internal modules and external dependencies |
| [Provider Class](docs/diagrams/provider-class.puml) | Provider protocol, ChatResult, resilience, cache classes |
| [Circuit Breaker States](docs/diagrams/circuit-breaker-state.md) | Per-provider CB state machine |
| [Seq: Non-stream Chat](docs/diagrams/seq-chat-nonstream.md) | Full non-streaming request flow |
| [Seq: Stream Chat](docs/diagrams/seq-chat-stream.md) | SSE streaming path with chunk deltas |
| [Seq: Failover](docs/diagrams/seq-failover.md) | Primary failure → retry → CB open → fallback |
| [Seq: Cache Hit](docs/diagrams/seq-cache-hit.md) | Exact and semantic cache fast paths |

## System Context

For infrastructure provisioning, deployment topology, and cross-service architecture, see the `atlas-docs` repository.
