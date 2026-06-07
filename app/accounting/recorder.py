"""GW-14 — Call accounting: cost formula + `call_records` recorder.

Two pieces, both thin (ADR-016):

`compute_cost(usage, rates)` is a pure function over the four `Usage` token
fields, priced at the alias row's per-1M rates exactly as atlas-docs/03 §2:

    cost = input_tokens                * in / 1e6
         + output_tokens               * out / 1e6
         + cache_creation_input_tokens * in / 1e6 * 1.25   (cache write premium)
         + cache_read_input_tokens     * in / 1e6 * 0.10   (cache read discount)

All arithmetic is `Decimal` so money is never floated; the rates passed in are
the ones pinned on the `call_records` row, making cost reproducible even if
alias prices change later (atlas-docs/03 §2).

`CallRecorder` persists one `call_records` row (atlas-docs/03 §1.4) via an
injected asyncpg connection/pool — the connection is never opened here, so it
unit-tests with a fake. The insert is idempotent on the call `id`
(`ON CONFLICT (id) DO NOTHING`): a retried record never double-bills. Publishing
the matching Kafka `atlas.calls.v1` event and wiring this into the request path
are separate tickets. See GW-14 + ADR-016 + atlas-docs/03 §1.4, §2.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from app.domain.messages import Usage
from app.repositories.tables import ProviderEnum

#: Per-million-token divisor (rates are quoted per 1M tokens, atlas-docs §2).
_PER_MILLION = Decimal(1_000_000)

#: Billing multipliers for the two cache token classes (atlas-docs §2).
_CACHE_CREATION_MULTIPLIER = Decimal("1.25")  # tokens written to the prompt cache
_CACHE_READ_MULTIPLIER = Decimal("0.10")  # tokens served from the prompt cache


@dataclass(frozen=True, slots=True)
class Rates:
    """The per-1M token prices applied to one call (from the alias row/overrides).

    `in_per_1m` prices `input_tokens` and both cache token classes (at their
    multipliers); `out_per_1m` prices `output_tokens`. These are the rates
    pinned on the `call_records` row, so cost stays auditable (atlas-docs §2).
    """

    in_per_1m: Decimal
    out_per_1m: Decimal


def compute_cost(usage: Usage, rates: Rates) -> Decimal:
    """Compute `computed_cost_usd` for one call over the four token fields.

    Pure and `Decimal`-exact (atlas-docs/03 §2). Cache-creation tokens bill at
    1.25x and cache-read tokens at 0.10x the input rate; input and output bill
    at 1.0x their respective rates.
    """
    in_unit = rates.in_per_1m / _PER_MILLION
    out_unit = rates.out_per_1m / _PER_MILLION
    return (
        Decimal(usage.input_tokens) * in_unit
        + Decimal(usage.output_tokens) * out_unit
        + Decimal(usage.cache_creation_input_tokens) * in_unit * _CACHE_CREATION_MULTIPLIER
        + Decimal(usage.cache_read_input_tokens) * in_unit * _CACHE_READ_MULTIPLIER
    )


#: Idempotent insert: a retried record (same `id`) is a no-op, so a call is
#: never billed twice (atlas-docs/03 §1.4 — one priced record per call). Column
#: order matches `call_records` (atlas-docs/03 §1.4 DDL); `created_at` defaults
#: to now() in the schema and is intentionally not supplied here.
_INSERT_SQL = """
INSERT INTO call_records (
    id, api_key_id, app, prompt_version_id, alias, model, provider,
    input_tokens, output_tokens, cache_creation_input_tokens,
    cache_read_input_tokens, computed_cost_usd, latency_ms, status
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
)
ON CONFLICT (id) DO NOTHING
"""


class _Connection(Protocol):
    """Structural view of the asyncpg connection/pool the recorder needs.

    `asyncpg.Connection` and `asyncpg.Pool` both expose this `execute`
    signature, so either can be injected; tests inject a fake. Importing the
    Protocol (not a concrete class) keeps the adapter decoupled from how the
    connection is created/pooled.
    """

    async def execute(self, query: str, *args: object) -> str: ...


@dataclass(frozen=True, slots=True)
class CallRecord:
    """The values written to one `call_records` row (atlas-docs/03 §1.4).

    `cost` and the four token fields are stored verbatim alongside the resolved
    `model`/`provider` so the record stays reproducible. `id` keys idempotency:
    the caller supplies a stable call id so a retry collapses onto the same row.
    """

    id: uuid.UUID
    api_key_id: uuid.UUID
    app: str
    model: str
    provider: ProviderEnum
    usage: Usage
    cost: Decimal
    latency_ms: int
    status: int
    alias: str | None = None
    prompt_version_id: uuid.UUID | None = None


class CallRecorder:
    """Inserts `call_records` rows via an injected asyncpg connection/pool.

    The connection is injected (composition root supplies a real pool; tests
    supply a fake), so this adapter does no connection management. Inserts are
    idempotent on the call `id` (see `_INSERT_SQL`).
    """

    def __init__(self, conn: _Connection) -> None:
        self._conn = conn

    async def record(self, record: CallRecord) -> None:
        """Persist one priced call record (idempotent on `record.id`)."""
        await self._conn.execute(
            _INSERT_SQL,
            record.id,
            record.api_key_id,
            record.app,
            record.prompt_version_id,
            record.alias,
            record.model,
            record.provider.value,
            record.usage.input_tokens,
            record.usage.output_tokens,
            record.usage.cache_creation_input_tokens,
            record.usage.cache_read_input_tokens,
            record.cost,
            record.latency_ms,
            record.status,
        )
