"""GW-14 — call accounting: cost formula + `call_records` recorder.

Two halves, both offline (no live DB):

- `compute_cost` is pinned purely against atlas-docs/03 §2 — the worked `smart`
  example, the cache-read 0.10x discount and cache-creation 1.25x premium in
  isolation, output pricing, and `Decimal` exactness.
- `CallRecorder` is exercised with a fake asyncpg connection that records the
  SQL and bound params, proving the insert param construction (column→value
  mapping, provider rendered as its enum value) and idempotency (the statement
  carries `ON CONFLICT (id) DO NOTHING`, and the same call id collapses onto one
  row in the fake).
"""

from __future__ import annotations

import uuid
from decimal import Decimal

from app.accounting.recorder import (
    CallRecord,
    CallRecorder,
    Rates,
    compute_cost,
)
from app.domain.messages import Usage
from app.repositories.tables import ProviderEnum

# `smart` alias rates from atlas-docs/03 §2 / seed (claude-sonnet-4-6, $3/$15 per 1M).
_SMART_RATES = Rates(in_per_1m=Decimal("3.00"), out_per_1m=Decimal("15.00"))


# ── compute_cost: pure formula (atlas-docs/03 §2) ───────────────────────────


def test_worked_example_matches_docs() -> None:
    """The atlas-docs/03 §2 worked example resolves to exactly $0.007365."""
    usage = Usage(
        input_tokens=800,
        output_tokens=200,
        cache_creation_input_tokens=500,
        cache_read_input_tokens=300,
    )
    assert compute_cost(usage, _SMART_RATES) == Decimal("0.007365")


def test_input_only_priced_at_input_rate() -> None:
    usage = Usage(input_tokens=1_000_000)
    assert compute_cost(usage, _SMART_RATES) == Decimal("3.00")


def test_output_only_priced_at_output_rate() -> None:
    usage = Usage(output_tokens=1_000_000)
    assert compute_cost(usage, _SMART_RATES) == Decimal("15.00")


def test_cache_read_billed_at_one_tenth_input_rate() -> None:
    """cache_read tokens cost 0.10x the input rate (atlas-docs §2)."""
    cache_read = compute_cost(Usage(cache_read_input_tokens=1_000_000), _SMART_RATES)
    plain_input = compute_cost(Usage(input_tokens=1_000_000), _SMART_RATES)
    assert cache_read == Decimal("0.30")
    assert cache_read == plain_input * Decimal("0.10")


def test_cache_creation_billed_at_1_25x_input_rate() -> None:
    """cache_creation tokens cost 1.25x the input rate (atlas-docs §2)."""
    cache_creation = compute_cost(Usage(cache_creation_input_tokens=1_000_000), _SMART_RATES)
    plain_input = compute_cost(Usage(input_tokens=1_000_000), _SMART_RATES)
    assert cache_creation == Decimal("3.75")
    assert cache_creation == plain_input * Decimal("1.25")


def test_zero_usage_costs_zero() -> None:
    assert compute_cost(Usage(), _SMART_RATES) == Decimal("0")


def test_cost_is_decimal_not_float() -> None:
    """Money stays Decimal-exact end to end (no float contamination)."""
    cost = compute_cost(Usage(input_tokens=1, cache_read_input_tokens=1), _SMART_RATES)
    assert isinstance(cost, Decimal)
    # 1 * 3/1e6 + 1 * 3/1e6 * 0.10 — exact, no binary-float rounding.
    assert cost == Decimal("0.0000033")


def test_cost_is_additive_over_token_fields() -> None:
    """Total equals the sum of each field priced alone (linearity of the formula)."""
    usage = Usage(
        input_tokens=800,
        output_tokens=200,
        cache_creation_input_tokens=500,
        cache_read_input_tokens=300,
    )
    parts = (
        compute_cost(Usage(input_tokens=usage.input_tokens), _SMART_RATES)
        + compute_cost(Usage(output_tokens=usage.output_tokens), _SMART_RATES)
        + compute_cost(
            Usage(cache_creation_input_tokens=usage.cache_creation_input_tokens), _SMART_RATES
        )
        + compute_cost(Usage(cache_read_input_tokens=usage.cache_read_input_tokens), _SMART_RATES)
    )
    assert compute_cost(usage, _SMART_RATES) == parts


# ── CallRecorder: insert param construction + idempotency (fake conn) ────────


class _FakeConn:
    """Fake asyncpg connection: records calls and emulates `ON CONFLICT DO NOTHING`.

    Captures (query, args) per `execute` and tracks inserted call ids so a
    repeated id is a no-op — exactly the idempotency the real PK conflict gives.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.inserted_ids: set[object] = set()

    async def execute(self, query: str, *args: object) -> str:
        self.calls.append((query, args))
        call_id = args[0]
        if "ON CONFLICT" in query and call_id in self.inserted_ids:
            return "INSERT 0 0"  # conflict: row already present, nothing written
        self.inserted_ids.add(call_id)
        return "INSERT 0 1"


def _record(call_id: uuid.UUID | None = None) -> CallRecord:
    return CallRecord(
        id=call_id or uuid.uuid4(),
        api_key_id=uuid.uuid4(),
        app="research-bot",
        model="claude-sonnet-4-6",
        provider=ProviderEnum.anthropic,
        usage=Usage(
            input_tokens=800,
            output_tokens=200,
            cache_creation_input_tokens=500,
            cache_read_input_tokens=300,
        ),
        cost=Decimal("0.007365"),
        latency_ms=412,
        status=200,
        alias="smart",
    )


async def test_record_issues_single_insert() -> None:
    conn = _FakeConn()
    await CallRecorder(conn).record(_record())
    assert len(conn.calls) == 1
    query, _ = conn.calls[0]
    assert query.lstrip().startswith("INSERT INTO call_records")


async def test_insert_param_order_matches_columns() -> None:
    """Bound params map positionally to the documented column order (atlas-docs §1.4)."""
    conn = _FakeConn()
    rec = _record()
    await CallRecorder(conn).record(rec)
    _, args = conn.calls[0]
    assert args == (
        rec.id,
        rec.api_key_id,
        rec.app,
        rec.prompt_version_id,  # None — prompt_ref not used
        rec.alias,
        rec.model,
        "anthropic",  # provider rendered as its enum value, not the enum
        800,
        200,
        500,
        300,
        Decimal("0.007365"),
        412,
        200,
    )


async def test_provider_bound_as_enum_value() -> None:
    conn = _FakeConn()
    await CallRecorder(conn).record(_record())
    _, args = conn.calls[0]
    assert args[6] == "anthropic"
    assert not isinstance(args[6], ProviderEnum)


async def test_optional_fields_default_to_none() -> None:
    conn = _FakeConn()
    rec = CallRecord(
        id=uuid.uuid4(),
        api_key_id=uuid.uuid4(),
        app="bare",
        model="gpt-4.1",
        provider=ProviderEnum.openai,
        usage=Usage(input_tokens=10),
        cost=Decimal("0.00003"),
        latency_ms=5,
        status=200,
    )
    await CallRecorder(conn).record(rec)
    _, args = conn.calls[0]
    assert args[3] is None  # prompt_version_id
    assert args[4] is None  # alias


async def test_insert_is_idempotent_on_call_id() -> None:
    """A retried record (same id) hits ON CONFLICT and does not double-insert."""
    conn = _FakeConn()
    recorder = CallRecorder(conn)
    rec = _record(call_id=uuid.uuid4())

    await recorder.record(rec)
    await recorder.record(rec)  # retry with the same id

    # The statement is sent twice (the recorder is not stateful) ...
    assert len(conn.calls) == 2
    # ... but every insert carries the idempotency guard, so only one row lands.
    assert all("ON CONFLICT (id) DO NOTHING" in q for q, _ in conn.calls)
    assert len(conn.inserted_ids) == 1


async def test_distinct_ids_insert_distinct_rows() -> None:
    conn = _FakeConn()
    recorder = CallRecorder(conn)
    await recorder.record(_record(call_id=uuid.uuid4()))
    await recorder.record(_record(call_id=uuid.uuid4()))
    assert len(conn.inserted_ids) == 2
