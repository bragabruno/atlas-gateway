"""FastAPI application entrypoint for the Atlas gateway."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.v1.chat import router as chat_router
from app.api.v1.embeddings import router as embeddings_router
from app.api.v1.models import router as models_router

app = FastAPI(title="Atlas Gateway", version="0.1.0")
app.include_router(chat_router)
app.include_router(models_router)
app.include_router(embeddings_router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}
