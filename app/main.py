"""FastAPI application entrypoint for the Atlas gateway."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.chat import router as chat_router
from app.api.v1.embeddings import router as embeddings_router
from app.api.v1.models import router as models_router
from app.config import get_settings

app = FastAPI(title="Atlas Gateway", version="0.1.0")

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


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
