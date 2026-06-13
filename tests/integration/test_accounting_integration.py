"""Integration test — accounting persistence against a real Postgres (GW-14/15).

Spins an ephemeral Postgres via testcontainers, creates the real schema, seeds
the `api_keys` row the FK needs, then drives the REAL AccountingRecorder →
CallRecorder path and asserts a correctly-priced row lands in `call_records`.
This validates the seam the offline unit tests can only fake (they inject a
`_FakeConn`): the actual SQL, the asyncpg type round-trip, the api-key UUIDv5
FK, and the Decimal cost — end to end.

Requires Docker. Marked `integration`, so it is excluded from the default
offline suite (`pytest`); run explicitly with `pytest -m integration`.
"""

# This opt-in suite drives deliberately-untyped infra libraries (asyncpg has no
# stubs; testcontainers is a namespace package), so relax the "unknown from an
# untyped import" diagnostics here — the value is real-service behaviour.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal

import asyncpg
import pytest
from sqlalchemy import create_engine, text

pytest.importorskip("testcontainers.postgres")
from testcontainers.postgres import PostgresContainer  # noqa: E402

from app.accounting.adapter import (  # noqa: E402
    AccountingRecorder,
    api_key_uuid,
    rates_from_seed,
)
from app.accounting.recorder import CallRecorder  # noqa: E402
from app.domain.messages import Usage  # noqa: E402
from app.repositories.tables import Base  # noqa: E402
from app.services.chat_service import CallContext  # noqa: E402

# Match the local stack's pinned Postgres (atlas-infra/local/compose.dev.yaml).
_PG_IMAGE = "postgres:16.8"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def asyncpg_dsn() -> Iterator[str]:
    """Bring up Postgres, create the schema + the FK api_keys row, yield an asyncpg DSN."""
    with PostgresContainer(_PG_IMAGE, driver="psycopg") as pg:
        sync_url = pg.get_connection_url()  # postgresql+psycopg://...
        engine = create_engine(sync_url)
        Base.metadata.create_all(engine)
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO api_keys (id, hashed_secret, app, owner, status) "
                    "VALUES (:id, :secret, :app, :owner, 'active')"
                ),
                {
                    "id": api_key_uuid("dev-key"),
                    "secret": "integration-test-placeholder",
                    "app": "integration",
                    "owner": "integration",
                },
            )
        engine.dispose()
        # asyncpg uses a bare postgresql:// DSN (no +driver suffix).
        yield sync_url.replace("postgresql+psycopg://", "postgresql://")


async def test_recorder_persists_priced_row(asyncpg_dsn: str) -> None:
    pool = await asyncpg.create_pool(asyncpg_dsn)
    assert pool is not None
    try:
        recorder = AccountingRecorder(CallRecorder(pool), rates=rates_from_seed())
        # `smart` alias prices at in=3.00 / out=15.00 per 1M tokens, so
        # cost = 1000*3/1e6 + 500*15/1e6 = 0.003 + 0.0075 = 0.0105.
        usage = Usage(input_tokens=1000, output_tokens=500)
        await recorder.record(CallContext(api_key_id="dev-key", model="smart", usage=usage))
        row = await pool.fetchrow(
            "SELECT model, provider, input_tokens, output_tokens, computed_cost_usd "
            "FROM call_records"
        )
    finally:
        await pool.close()

    assert row is not None
    assert row["model"] == "smart"
    assert row["provider"] == "openai"
    assert row["input_tokens"] == 1000
    assert row["output_tokens"] == 500
    assert row["computed_cost_usd"] == Decimal("0.0105")
