"""Integration test — BRA-887 row-level security: tenant isolation on call_records.

Spins an ephemeral Postgres, builds the schema, applies the **exact** RLS DDL the
migration ships, seeds `call_records` for two tenants, then connects as a
*non-superuser* login role — the only way RLS engages, since superusers (and the
table owner, absent `FORCE`) bypass it — and asserts:

- with the tenant GUC set, a caller sees only their own rows;
- a second tenant sees only theirs (never the first's — the IDOR the WHERE
  clause also blocks, but here enforced by the database with no WHERE at all);
- with no GUC set, **zero** rows come back (fail closed).

This validates what the offline unit tests can only fake: real Postgres RLS under
a non-superuser role. Requires Docker. Marked `integration`, so it is excluded
from the default offline suite; run explicitly with `pytest -m integration`.
"""

# Drives deliberately-untyped infra libraries (asyncpg has no stubs;
# testcontainers is a namespace package), so relax "unknown from an untyped
# import" diagnostics here — the value is real-service behaviour.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAttributeAccessIssue=false
from __future__ import annotations

import os
from collections.abc import Iterator
from decimal import Decimal

import asyncpg
import pytest
from sqlalchemy import create_engine, insert, text

pytest.importorskip("testcontainers.postgres")
from testcontainers.postgres import PostgresContainer  # noqa: E402

from app.accounting.adapter import api_key_uuid  # noqa: E402
from app.repositories.tables import Base, CallRecord, ProviderEnum  # noqa: E402

# Match the local stack's pinned Postgres (atlas-infra/local/compose.dev.yaml).
_PG_IMAGE = "postgres:16.8"

# Non-superuser login role the gateway connects as in this test. Its password is
# an ephemeral, throwaway container credential — not a secret.
_APP_ROLE = "atlas_app"
_APP_PW = "rls-test-placeholder"  # noqa: S105

pytestmark = pytest.mark.integration

_TENANT_A = api_key_uuid("tenant-a")
_TENANT_B = api_key_uuid("tenant-b")
_A_ROWS = 3
_B_ROWS = 2


def _rls_statements() -> list[str]:
    """The exact RLS DDL the migration applies — loaded from the migration module.

    Re-using the migration's `enable_sql` (rather than re-spelling the DDL here)
    means this test proves the *shipped* policies isolate tenants, not a parallel
    copy that could drift.
    """
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    cfg = Config(os.path.join(root, "alembic.ini"))
    module = ScriptDirectory.from_config(cfg).get_revision("e4b8c2d6f1a9").module
    return [stmt for table in module.RLS_TABLES for stmt in module.enable_sql(table)]


def _seed(sync_url: str) -> None:
    """Create the schema + two tenants' api_keys/call_records (as the superuser)."""
    engine = create_engine(sync_url)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for key, n in (("tenant-a", _A_ROWS), ("tenant-b", _B_ROWS)):
            kid = api_key_uuid(key)
            conn.execute(
                text(
                    "INSERT INTO api_keys (id, hashed_secret, app, owner, status) "
                    "VALUES (:id, :secret, 'rls', 'rls', 'active')"
                ),
                {"id": kid, "secret": f"placeholder-{key}"},
            )
            for _ in range(n):
                conn.execute(
                    insert(CallRecord).values(
                        api_key_id=kid,
                        app="rls",
                        model="mock",
                        provider=ProviderEnum.openai,
                        computed_cost_usd=Decimal("0.0"),
                        latency_ms=1,
                        status=200,
                    )
                )
        # Apply the shipped RLS policies, then a non-superuser role to test under.
        for stmt in _rls_statements():
            conn.execute(text(stmt))
        conn.execute(text(f"CREATE ROLE {_APP_ROLE} LOGIN PASSWORD '{_APP_PW}' NOSUPERUSER"))
        conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {_APP_ROLE}"))
        conn.execute(text(f"GRANT SELECT, INSERT ON call_records, budgets TO {_APP_ROLE}"))
    engine.dispose()


@pytest.fixture(scope="module")
def app_dsn() -> Iterator[str]:
    """Bring up Postgres, seed + lock down, yield an asyncpg DSN for the app role."""
    with PostgresContainer(_PG_IMAGE, driver="psycopg") as pg:
        sync_url = pg.get_connection_url()  # postgresql+psycopg://test:test@host:port/test
        _seed(sync_url)
        # Same host/port/db, but connect as the non-superuser role (no +driver).
        host_port_db = sync_url.replace("postgresql+psycopg://", "postgresql://").split("@", 1)[1]
        yield f"postgresql://{_APP_ROLE}:{_APP_PW}@{host_port_db}"


async def _rows_for(pool: asyncpg.Pool, tenant: object | None) -> list[asyncpg.Record]:
    """Read call_records as the app role, optionally scoped to `tenant` via the GUC.

    No WHERE clause — so whatever comes back is whatever RLS allows. With `tenant`
    set, the policy admits only that tenant's rows; with `None`, the GUC stays
    unset and the policy admits none.
    """
    async with pool.acquire() as conn, conn.transaction():
        if tenant is not None:
            await conn.execute("SELECT set_config('atlas.api_key_id', $1, true)", str(tenant))
        return await conn.fetch("SELECT api_key_id FROM call_records")


async def test_rls_scopes_reads_to_the_tenant_guc(app_dsn: str) -> None:
    pool = await asyncpg.create_pool(app_dsn)
    assert pool is not None
    try:
        rows_a = await _rows_for(pool, _TENANT_A)
        rows_b = await _rows_for(pool, _TENANT_B)
        rows_unset = await _rows_for(pool, None)
    finally:
        await pool.close()

    # Each tenant sees exactly their own rows — never the other's.
    assert len(rows_a) == _A_ROWS
    assert all(r["api_key_id"] == _TENANT_A for r in rows_a)
    assert len(rows_b) == _B_ROWS
    assert all(r["api_key_id"] == _TENANT_B for r in rows_b)

    # No tenant GUC → no rows. A query that forgets to scope leaks nothing.
    assert rows_unset == []
