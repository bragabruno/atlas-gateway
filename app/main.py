"""FastAPI application entrypoint for the Atlas gateway."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.chat import router as chat_router
from app.api.v1.embeddings import router as embeddings_router
from app.api.v1.models import router as models_router
from app.api.v1.usage import router as usage_router
from app.config import Settings, get_settings

_access_log = logging.getLogger("atlas.access")


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


async def healthz() -> dict[str, str]:
    """Liveness probe — `{"status": "ok"}`. No dependencies; safe to poll."""
    return {"status": "ok"}


def create_app(settings: Settings) -> FastAPI:
    """Construct the FastAPI app for ``settings``.

    Extracted from module scope so the docs gate (``environment != "prod"``)
    and the CORS allowlist are unit-testable without monkey-patching. The
    module-level ``app = create_app(get_settings())`` below preserves the
    runtime wiring unchanged.

    persistAuthorization is hard-False (was True): the Swagger UI no longer
    persists bearer tokens to browser localStorage, shrinking the token-theft
    surface via XSS or compromised browser extensions in environments where
    docs remain reachable.
    """
    docs_enabled = settings.environment != "prod"
    application = FastAPI(
        title="Atlas Gateway",
        version=_APP_VERSION,
        description=_API_DESCRIPTION,
        contact={"name": "Bruno Braga", "email": "contact@bragdev.com"},
        license_info={"name": "Internal — Atlas (Enhesa mirror)"},
        servers=[
            {"url": "http://localhost:8090", "description": "Local debug (VS Code launcher)"},
            {
                "url": "http://atlas-gateway.atlas-platform.svc:8000",
                "description": "In-cluster (AKS)",
            },
        ],
        openapi_tags=_OPENAPI_TAGS,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
        swagger_ui_parameters=(
            {
                "persistAuthorization": False,
                "displayRequestDuration": True,
                "tryItOutEnabled": True,
                "docExpansion": "list",
            }
            if docs_enabled
            else None
        ),
    )

    # CORS is config-gated and default OFF: with no ATLAS_CORS_ALLOW_ORIGINS set
    # the middleware is not added, so behaviour is identical to the pre-CORS
    # gateway. Browser SPAs (e.g. the local-compose frontend on
    # http://localhost:8080) call the gateway cross-origin and need their origin
    # allowlisted here.
    if settings.cors_allow_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_allow_origins),
            allow_credentials=False,  # auth is via the Authorization header, not cookies
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Access log + request-id correlation: one structured line per request
    # (method, path, status, duration) with a generated request id echoed in the
    # X-Request-Id header. Path only — never query strings or bodies — so no PII
    # leaks; this gives the audit trail something to correlate auth/429 events to.
    @application.middleware("http")
    async def _access_log_middleware(  # pyright: ignore[reportUnusedFunction]  — registered via the decorator
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        # If call_next raises (unhandled 5xx), the framework still returns a 500 —
        # so log in `finally` to guarantee one correlatable line per request, with
        # the request id, even on failures. Default status reflects that 5xx.
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-Id"] = request_id
            return response
        finally:
            _access_log.info(
                "request method=%s path=%s status=%d dur_ms=%.1f request_id=%s",
                request.method,
                request.url.path,
                status,
                (time.perf_counter() - start) * 1000,
                request_id,
            )

    application.include_router(chat_router)
    application.include_router(models_router)
    application.include_router(embeddings_router)
    application.include_router(usage_router)

    application.get(
        "/healthz",
        tags=["health"],
        summary="Liveness probe",
        description='Returns `{"status": "ok"}`. No dependencies; safe to poll.',
        response_model=dict[str, str],
    )(healthz)

    return application


app = create_app(get_settings())
