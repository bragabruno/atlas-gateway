"""BRA-887 — row-level security: tenant isolation on call_records + budgets.

Revision ID: e4b8c2d6f1a9
Revises: d2f1a7c9e3b4
Create Date: 2026-06-15

Defense-in-depth for the data plane. The usage endpoint already filters by
``api_key_id`` in its WHERE clause, but RLS makes the *database itself* refuse to
return another tenant's rows if that clause were ever dropped or a new query
forgot it. Reads are scoped to the per-connection GUC ``atlas.api_key_id`` (set
transaction-locally by the read path); an unset GUC yields **zero** rows — fail
closed. Inserts stay permissive (``WITH CHECK (true)``) because the only writer
is the accounting recorder, which derives ``api_key_id`` server-side from the
authenticated key — isolation is enforced on the read side, so the recorder
needn't set the GUC (its fakes/tests are unchanged).

ENFORCEMENT REQUIRES A NON-SUPERUSER ROLE. Postgres superusers bypass RLS
entirely, and the table owner bypasses it unless ``FORCE`` is set. ``FORCE ROW
LEVEL SECURITY`` (below) subjects the owner too, but the gateway must still
connect as a *non-superuser* login role for the policies to apply — see
atlas-infra (the ``atlas_app`` role). The seeder runs as the superuser and so
bypasses RLS by design.
"""

from __future__ import annotations

from alembic import op

revision = "e4b8c2d6f1a9"
down_revision = "d2f1a7c9e3b4"
branch_labels = None
depends_on = None

#: Tables with an ``api_key_id`` tenant column that RLS scopes (atlas-docs/03 §1).
RLS_TABLES = ("call_records", "budgets")

#: The per-connection GUC the read path sets (transaction-local) to name the
#: calling tenant; the policies compare ``api_key_id`` against it.
TENANT_GUC = "atlas.api_key_id"


def enable_sql(table: str) -> list[str]:
    """SQL that turns on RLS + the tenant policies for one table.

    Exposed (not inlined into ``upgrade``) so the integration test applies the
    *exact* production DDL against a real Postgres rather than re-spelling it.
    """
    return [
        f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
        f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY",
        # Reads: only rows whose api_key_id matches the per-connection GUC. An
        # unset GUC → current_setting(..., true) is NULL → api_key_id = NULL is
        # never true → no rows (fail closed).
        f"CREATE POLICY {table}_read_tenant ON {table} "
        f"FOR SELECT USING (api_key_id = current_setting('{TENANT_GUC}', true)::uuid)",
        # Writes: server-controlled (the recorder sets api_key_id from the
        # authenticated key), so the insert path needn't set the GUC.
        f"CREATE POLICY {table}_insert_server ON {table} FOR INSERT WITH CHECK (true)",
    ]


def disable_sql(table: str) -> list[str]:
    """Inverse of :func:`enable_sql` for one table."""
    return [
        f"DROP POLICY IF EXISTS {table}_insert_server ON {table}",
        f"DROP POLICY IF EXISTS {table}_read_tenant ON {table}",
        f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY",
        f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY",
    ]


def upgrade() -> None:
    for table in RLS_TABLES:
        for stmt in enable_sql(table):
            op.execute(stmt)


def downgrade() -> None:
    for table in RLS_TABLES:
        for stmt in disable_sql(table):
            op.execute(stmt)
