"""Initial gateway schema: model_aliases, api_keys, budgets, call_records (GW-9).

Creates the four gateway-owned tables (ADR-015) with the columns, types, and
indexes specified in atlas-docs/03 §1.1–1.4, the two Postgres enum types
(`provider_enum`, `key_status_enum`), and seeds the `model_aliases` rows
(smart/deep/fast/balanced/embed) from atlas-docs/03 §2. Targets Azure Database
for PostgreSQL via the psycopg3 sync driver (ADR-010).

Revision ID: b113ca4237b0
Revises:
Create Date: 2026-06-07

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from app.repositories.seed import ALIAS_SEED

# revision identifiers, used by Alembic.
revision: str = "b113ca4237b0"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Dialect-specific `postgresql.ENUM` with `create_type=False`: the types are
# created once explicitly in upgrade() (they each back two tables), so the
# per-column references must not re-emit `CREATE TYPE`. Alembic's create_table
# honors `create_type=False` only on the dialect-specific ENUM, not generic
# `sa.Enum` — and this migration targets Postgres only (ADR-010).
provider_enum = postgresql.ENUM(
    "anthropic",
    "openai",
    "google",
    "azure_openai",
    name="provider_enum",
    create_type=False,
)
key_status_enum = postgresql.ENUM(
    "active",
    "suspended",
    "revoked",
    name="key_status_enum",
    create_type=False,
)
# JSONB on Postgres (atlas-docs §1.1).
per_key_overrides_type = postgresql.JSONB()


def upgrade() -> None:
    # Create the shared enum types once. `.create(checkfirst=True)` is
    # idempotent online; offline (`--sql`) it emits a single `CREATE TYPE` per
    # enum. The per-column `postgresql.ENUM(..., create_type=False)` references
    # then reuse the type without re-emitting it (each backs two tables).
    bind = op.get_bind()
    provider_enum.create(bind, checkfirst=True)
    key_status_enum.create(bind, checkfirst=True)

    op.create_table(
        "model_aliases",
        sa.Column("alias", sa.Text(), primary_key=True),
        sa.Column("primary_model_id", sa.Text(), nullable=False),
        sa.Column("fallback_model_id", sa.Text(), nullable=False),
        sa.Column("provider", provider_enum, nullable=False),
        sa.Column("in_price_per_1m", sa.Numeric(10, 6), nullable=False),
        sa.Column("out_price_per_1m", sa.Numeric(10, 6), nullable=False),
        sa.Column(
            "per_key_overrides",
            per_key_overrides_type,
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "api_keys",
        sa.Column(
            "id",
            sa.Uuid(),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("hashed_secret", sa.Text(), nullable=False, unique=True),
        sa.Column("app", sa.Text(), nullable=False),
        sa.Column("owner", sa.Text(), nullable=False),
        sa.Column(
            "status",
            key_status_enum,
            nullable=False,
            server_default=sa.text("'active'"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("idx_api_keys_hashed_secret", "api_keys", ["hashed_secret"])
    op.create_index("idx_api_keys_app", "api_keys", ["app"])

    op.create_table(
        "budgets",
        sa.Column(
            "id",
            sa.Uuid(),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "api_key_id",
            sa.Uuid(),
            sa.ForeignKey("api_keys.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("monthly_cap_usd", sa.Numeric(12, 4), nullable=False),
        sa.Column(
            "current_spend",
            sa.Numeric(12, 4),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "alert_at_80pct",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "reset_cycle",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'monthly'"),
        ),
        sa.Column("period_start", sa.Date(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("api_key_id", name="budgets_api_key_unique"),
    )
    op.create_index("idx_budgets_api_key_id", "budgets", ["api_key_id"])

    op.create_table(
        "call_records",
        sa.Column(
            "id",
            sa.Uuid(),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "api_key_id",
            sa.Uuid(),
            sa.ForeignKey("api_keys.id"),
            nullable=False,
        ),
        sa.Column("app", sa.Text(), nullable=False),
        sa.Column("prompt_version_id", sa.Uuid(), nullable=True),
        sa.Column(
            "alias",
            sa.Text(),
            sa.ForeignKey("model_aliases.alias"),
            nullable=True,
        ),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("provider", provider_enum, nullable=False),
        sa.Column(
            "input_tokens", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "output_tokens", sa.Integer(), nullable=False, server_default=sa.text("0")
        ),
        sa.Column(
            "cache_creation_input_tokens",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "cache_read_input_tokens",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("computed_cost_usd", sa.Numeric(12, 8), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("status", sa.SmallInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "idx_call_records_api_key_id",
        "call_records",
        ["api_key_id", sa.text("created_at DESC")],
    )
    op.create_index(
        "idx_call_records_created_at",
        "call_records",
        [sa.text("created_at DESC")],
    )
    op.create_index("idx_call_records_alias", "call_records", ["alias"])

    _seed_aliases()


def _seed_aliases() -> None:
    """Insert the initial model_aliases rows (atlas-docs/03 §2)."""
    model_aliases = sa.table(
        "model_aliases",
        sa.column("alias", sa.Text()),
        sa.column("primary_model_id", sa.Text()),
        sa.column("fallback_model_id", sa.Text()),
        sa.column("provider", provider_enum),
        sa.column("in_price_per_1m", sa.Numeric(10, 6)),
        sa.column("out_price_per_1m", sa.Numeric(10, 6)),
    )
    op.bulk_insert(model_aliases, [dict(row) for row in ALIAS_SEED])


def downgrade() -> None:
    op.drop_index("idx_call_records_alias", table_name="call_records")
    op.drop_index("idx_call_records_created_at", table_name="call_records")
    op.drop_index("idx_call_records_api_key_id", table_name="call_records")
    op.drop_table("call_records")

    op.drop_index("idx_budgets_api_key_id", table_name="budgets")
    op.drop_table("budgets")

    op.drop_index("idx_api_keys_app", table_name="api_keys")
    op.drop_index("idx_api_keys_hashed_secret", table_name="api_keys")
    op.drop_table("api_keys")

    op.drop_table("model_aliases")

    bind = op.get_bind()
    key_status_enum.drop(bind, checkfirst=True)
    provider_enum.drop(bind, checkfirst=True)
