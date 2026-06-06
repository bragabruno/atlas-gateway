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

## Module Map (`app/`)

```
app/
├── api/v1/
│   ├── chat.py          # POST /v1/chat/completions (stream + non-stream)
│   ├── models.py        # GET /v1/models
│   └── embeddings.py    # POST /v1/embeddings
├── providers/
│   ├── base.py          # Provider Protocol: async chat(...)->ChatResult, async models()
│   ├── openai.py        # OpenAI provider
│   ├── anthropic.py     # Anthropic provider
│   ├── google.py        # Google (Gemini) provider
│   └── mock.py          # Mock provider (testing)
├── routing/
│   └── aliases.py       # AliasResolver: alias→primary+fallback, per-key overrides
├── resilience/
│   ├── retry.py         # tenacity retry policies
│   └── circuit_breaker.py  # Redis-backed per-provider CB (closed/open/half-open)
├── cache/
│   ├── exact.py         # Redis exact cache (key: prompt_version + tenant)
│   └── semantic.py      # Qdrant semantic cache (threshold 0.97, opt-in, tenant-scoped)
├── accounting/
│   └── recorder.py      # asyncpg insert to call_records; cost formula; emits to Kafka atlas.calls.v1
├── limits/
│   ├── ratelimit.py     # Redis token-bucket → 429
│   └── budget.py        # Monthly budget → 429 + 80% alert
├── guardrails/
│   └── chain.py         # Pre/post middleware chain
├── telemetry/
│   └── otel.py          # GenAI semconv → OTel Collector → Splunk
└── prompts/
    └── registry.py      # Registry client: resolve prompt_ref → rendered config
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
