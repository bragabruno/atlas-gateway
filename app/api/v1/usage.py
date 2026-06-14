"""FE-6 — GET /v1/usage: per-key token + cost aggregates from call_records.

Returns usage aggregated over the requested window (default: current calendar
month) grouped by (app, model).  Requires a live DB connection
(ATLAS_DB_URL); when not configured, returns 503.

Query params
------------
since : ISO date string (YYYY-MM-DD), optional — defaults to the 1st of the
        current month.  Rows with `created_at >= since` are included.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.accounting.adapter import api_key_uuid
from app.api.deps import get_db_pool, require_api_key
from app.config import Settings, get_settings
from app.domain.openai import ErrorEnvelope

router = APIRouter()


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class UsageRow(BaseModel):
    app: str = Field(description="Calling app id (per-API-key label).")
    model: str = Field(description="Provider model id or Atlas alias served.")
    input_tokens: int = Field(description="Sum of prompt tokens over the window.")
    output_tokens: int = Field(description="Sum of completion tokens over the window.")
    total_cost_usd: Decimal = Field(description="Sum of `computed_cost_usd` (USD).")


class UsageResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "since": "2026-06-01",
                "rows": [
                    {
                        "app": "regdoc-qa",
                        "model": "gpt-4o-mini",
                        "input_tokens": 12_345,
                        "output_tokens": 4_567,
                        "total_cost_usd": "1.2345",
                    }
                ],
            }
        }
    )

    since: date = Field(description="Inclusive start of the aggregation window (UTC date).")
    rows: list[UsageRow] = Field(description="One row per (app, model). Sorted by cost desc.")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

#: Tenant-scoped aggregate: only rows for the authenticated key's deterministic
#: UUID (api_key_uuid(bearer)) are returned. Uses the composite index
#: ``idx_call_records_api_key_id (api_key_id, created_at DESC)``.
_USAGE_SQL = """
SELECT
    app,
    model,
    SUM(input_tokens)  AS input_tokens,
    SUM(output_tokens) AS output_tokens,
    SUM(computed_cost_usd) AS total_cost_usd
FROM call_records
WHERE api_key_id = $1
  AND created_at >= $2::timestamptz
GROUP BY app, model
ORDER BY total_cost_usd DESC
"""


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.get(
    "/v1/usage",
    response_model=UsageResponse,
    tags=["usage"],
    summary="Per-(app, model) usage aggregates",
    description=(
        "Token + cost aggregates from `call_records`, grouped by (app, model) "
        "and sorted by total cost desc. Default window: current calendar month. "
        "Requires `ATLAS_DB_URL`; returns 503 when accounting DB is not configured."
    ),
    responses={
        401: {"model": ErrorEnvelope, "description": "Missing or invalid Bearer key."},
        503: {
            "model": ErrorEnvelope,
            "description": "Accounting DB not configured (`ATLAS_DB_URL` unset).",
        },
    },
)
async def get_usage(
    key: Annotated[str, Depends(require_api_key)],
    settings: Annotated[Settings, Depends(get_settings)],
    since: date | None = None,
    pool: Any = Depends(get_db_pool),
) -> UsageResponse:
    if pool is None:
        raise HTTPException(status_code=503, detail="usage data unavailable: DB not configured")

    window_start = since or date.today().replace(day=1)
    # api_key_uuid is the same deterministic UUIDv5 the accounting recorder
    # writes into call_records.api_key_id, so this lookup matches what was
    # written without a separate api_keys table join.
    api_key_id = api_key_uuid(key)
    rows = await pool.fetch(_USAGE_SQL, api_key_id, window_start)

    return UsageResponse(
        since=window_start,
        rows=[
            UsageRow(
                app=r["app"],
                model=r["model"],
                input_tokens=r["input_tokens"],
                output_tokens=r["output_tokens"],
                total_cost_usd=r["total_cost_usd"],
            )
            for r in rows
        ],
    )
