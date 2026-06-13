"""FastAPI application entrypoint for the Atlas gateway."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.chat import router as chat_router
from app.api.v1.embeddings import router as embeddings_router
from app.api.v1.models import router as models_router
from app.api.v1.usage import router as usage_router
from app.config import get_settings


def _resolve_app_version() -> str:
    # `pip install -e .` not run yet (e.g. `python -m app.main` from a clean
    # checkout); fall through to the pyproject pin so the spec still renders.
    try:
        return version("atlas-gateway")
    except PackageNotFoundError:
        return "0.1.0"


_APP_VERSION = _resolve_app_version()

_API_DESCRIPTION = """\
The **Atlas Gateway** is the single OpenAI-compatible facade in front of
OpenAI, Anthropic, and Google. Every LLM call from the frontend and the agent
runtime flows through here so prompts are versionable, guardrails enforceable,
and every call cost-traceable (ADR-014).

### Authentication
All `/v1/*` endpoints require `Authorization: Bearer <key>`. Local dev keys
come from `ATLAS_API_KEYS` (default `["dev-key"]`); real deployments source
keys from Key Vault.

### Streaming
`POST /v1/chat/completions` with `"stream": true` returns
`text/event-stream` carrying OpenAI `chat.completion.chunk` deltas terminated
by `data: [DONE]`.

### Errors
4xx/5xx responses use the `ErrorEnvelope` schema (`{detail, code?}`).
Per-endpoint codes are documented under each operation's responses.
"""

_OPENAPI_TAGS = [
    {
        "name": "chat",
        "description": (
            "OpenAI-compatible chat completions (non-streaming JSON + SSE stream). "
            "Honors guardrails, rate limits, and budget; emits cost events to "
            "Kafka via the accounting recorder."
        ),
    },
    {
        "name": "embeddings",
        "description": (
            "OpenAI-compatible embeddings. Returns a list of 1+ embedding vectors "
            "matching the order of the input."
        ),
    },
    {
        "name": "models",
        "description": "OpenAI-compatible model list. Aggregated from the provider registry.",
    },
    {
        "name": "usage",
        "description": (
            "Per-(app, model) token + cost aggregates over a date window. Requires "
            "`ATLAS_DB_URL`; returns 503 when the accounting DB is not configured."
        ),
    },
    {
        "name": "health",
        "description": "Liveness probe. Used by Kubernetes and the local-compose stack.",
    },
]

app = FastAPI(
    title="Atlas Gateway",
    version=_APP_VERSION,
    description=_API_DESCRIPTION,
    contact={"name": "Bruno Braga", "email": "contact@bragdev.com"},
    license_info={"name": "Internal — Atlas (Enhesa mirror)"},
    servers=[
        {"url": "http://localhost:8090", "description": "Local debug (VS Code launcher)"},
        {"url": "http://atlas-gateway.atlas-platform.svc:8000", "description": "In-cluster (AKS)"},
    ],
    openapi_tags=_OPENAPI_TAGS,
    swagger_ui_parameters={
        "persistAuthorization": True,
        "displayRequestDuration": True,
        "tryItOutEnabled": True,
        "docExpansion": "list",
    },
)

# CORS is config-gated and default OFF: with no ATLAS_CORS_ALLOW_ORIGINS set the
# middleware is not added, so behaviour is identical to the pre-CORS gateway.
# Browser SPAs (e.g. the local-compose frontend on http://localhost:8080) call
# the gateway cross-origin and need their origin allowlisted here.
_cors_origins = get_settings().cors_allow_origins
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(_cors_origins),
        allow_credentials=False,  # auth is via the Authorization header, not cookies
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(chat_router)
app.include_router(models_router)
app.include_router(embeddings_router)
app.include_router(usage_router)


@app.get(
    "/healthz",
    tags=["health"],
    summary="Liveness probe",
    description='Returns `{"status": "ok"}`. No dependencies; safe to poll.',
    response_model=dict[str, str],
)
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
