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
from pydantic import BaseModel

from app.api.deps import get_db_pool, require_api_key
from app.config import Settings, get_settings

router = APIRouter()


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class UsageRow(BaseModel):
    app: str
    model: str
    input_tokens: int
    output_tokens: int
    total_cost_usd: Decimal


class UsageResponse(BaseModel):
    since: date
    rows: list[UsageRow]


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

_USAGE_SQL = """
SELECT
    app,
    model,
    SUM(input_tokens)  AS input_tokens,
    SUM(output_tokens) AS output_tokens,
    SUM(computed_cost_usd) AS total_cost_usd
FROM call_records
WHERE created_at >= $1::timestamptz
GROUP BY app, model
ORDER BY total_cost_usd DESC
"""


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.get("/v1/usage", response_model=UsageResponse)
async def get_usage(
    _key: Annotated[str, Depends(require_api_key)],
    settings: Annotated[Settings, Depends(get_settings)],
    since: date | None = None,
    pool: Any = Depends(get_db_pool),
) -> UsageResponse:
    if pool is None:
        raise HTTPException(status_code=503, detail="usage data unavailable: DB not configured")

    window_start = since or date.today().replace(day=1)
    rows = await pool.fetch(_USAGE_SQL, window_start)

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
