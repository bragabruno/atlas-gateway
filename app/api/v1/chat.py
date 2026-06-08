"""GW-6/GW-7 — /v1/chat/completions controller (thin HTTP layer).

Parses the request, enforces per-key bearer auth, and delegates to `ChatService`
(injected via `app.api.deps`). All chat orchestration lives in the service
layer; this controller only maps between HTTP and the service. The domain →
HTTP error map is:

- `UnknownModelError`  → 404 (no provider/alias for the requested model).
- `RateLimitExceeded`  → 429 with the spec ``{"error": {...}}`` body and a
  ``Retry-After`` header (atlas-docs/03 §5.2 *Rate Limit Exceeded*, GW-16).
- `BudgetExceeded`     → 429 with the spec ``{"error": {...}}`` body and a
  ``Retry-After`` header derived from the reset date (atlas-docs/03 §5.2
  *Budget Exceeded*, GW-17).
- `GuardrailRejection` → 422 with an explicit ``reason`` naming the guardrail
  and phase (GRD-1).

These mappings are inert unless the corresponding collaborator is wired in
`deps.py` (default OFF): with no rate limiter / budget / guardrails configured
the service never raises them, so the default path returns exactly as before.
When `stream=true` the service yields an OpenAI-compatible `text/event-stream`
terminated by `data: [DONE]`. See ADR-016.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.deps import get_chat_service, require_api_key
from app.domain.errors import UnknownModelError
from app.domain.openai import ChatCompletionRequest, ChatCompletionResponse
from app.guardrails.chain import GuardrailRejection
from app.limits.budget import BudgetExceeded
from app.limits.ratelimit import RateLimitExceeded
from app.services.chat_service import ChatService

router = APIRouter()

#: Whole days → seconds, for the budget ``Retry-After`` (the cap resets on a
#: date, not an instant); a same-day reset still yields a positive header.
_SECONDS_PER_DAY = 24 * 60 * 60


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    req: ChatCompletionRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
    key: Annotated[str, Depends(require_api_key)],
) -> ChatCompletionResponse | StreamingResponse | JSONResponse:
    try:
        if req.stream:
            return StreamingResponse(
                service.stream(req, api_key_id=key), media_type="text/event-stream"
            )
        return await service.complete(req, api_key_id=key)
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RateLimitExceeded as exc:
        return JSONResponse(
            status_code=429,
            content=exc.body,
            headers={"Retry-After": str(exc.retry_after)},
        )
    except BudgetExceeded as exc:
        return JSONResponse(
            status_code=429,
            content=exc.body,
            headers={"Retry-After": str(_budget_retry_after(exc))},
        )
    except GuardrailRejection as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "guardrail": exc.guardrail,
                "phase": exc.phase.value,
                "reason": exc.reason,
            },
        ) from exc


def _budget_retry_after(exc: BudgetExceeded) -> int:
    """Whole seconds until the budget cap resets (≥ 1 so the header is positive)."""
    days = (exc.resets_on - date.today()).days
    return max(1, days * _SECONDS_PER_DAY)
