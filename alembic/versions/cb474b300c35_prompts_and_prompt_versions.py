"""Add prompts and prompt_versions to the gateway schema (REG-1).

Creates the two prompt-registry tables this service owns (ADR-015) with the
columns, types, and indexes specified in atlas-docs/03 §1.5: `prompts`
(id, unique name) and `prompt_versions` (semver, Jinja2 template, JSONB
params_schema, model_alias FK, status), plus the `prompt_status_enum` Postgres
type and the `(prompt_id, semver)` unique constraint. Revises the GW-9 head
(b113ca4237b0). Targets Azure Database for PostgreSQL via the psycopg3 sync
driver (ADR-010). The single-`production`-version-per-prompt rule is enforced at
the application layer (`app.registry.promotion`), not by a DB constraint here,
per atlas-docs/03 §1.5.

Revision ID: cb474b300c35
Revises: b113ca4237b0
Create Date: 2026-06-07

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "cb474b300c35"
down_revision: str | None = "b113ca4237b0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Dialect-specific `postgresql.ENUM` with `create_type=False`: the type is
# created once explicitly in upgrade(), so the per-column reference must not
# re-emit `CREATE TYPE`. This migration targets Postgres only (ADR-010).
prompt_status_enum = postgresql.ENUM(
    "draft",
    "candidate",
    "production",
    "retired",
    name="prompt_status_enum",
    create_type=False,
)
# JSONB on Postgres (atlas-docs §1.5).
params_schema_type = postgresql.JSONB()


def upgrade() -> None:
    # Create the prompt status enum once; `.create(checkfirst=True)` is
    # idempotent online and emits a single `CREATE TYPE` offline (`--sql`).
    bind = op.get_bind()
    prompt_status_enum.create(bind, checkfirst=True)

    op.create_table(
        "prompts",
        sa.Column(
            "id",
            sa.Uuid(),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.Text(), nullable=False, unique=True),
    )

    op.create_table(
        "prompt_versions",
        sa.Column(
            "id",
            sa.Uuid(),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "prompt_id",
            sa.Uuid(),
            sa.ForeignKey("prompts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("semver", sa.Text(), nullable=False),
        sa.Column("template", sa.Text(), nullable=False),
        sa.Column(
            "params_schema",
            params_schema_type,
            nullable=False,
            server_default=sa.text("'{}'"),
        ),
        sa.Column(
            "model_alias",
            sa.Text(),
            sa.ForeignKey("model_aliases.alias"),
            nullable=True,
        ),
        sa.Column(
            "status",
            prompt_status_enum,
            nullable=False,
            server_default=sa.text("'draft'"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("prompt_id", "semver", name="prompt_versions_semver_unique"),
    )
    op.create_index("idx_prompt_versions_prompt_id", "prompt_versions", ["prompt_id"])
    op.create_index("idx_prompt_versions_status", "prompt_versions", ["status"])


def downgrade() -> None:
    op.drop_index("idx_prompt_versions_status", table_name="prompt_versions")
    op.drop_index("idx_prompt_versions_prompt_id", table_name="prompt_versions")
    op.drop_table("prompt_versions")
    op.drop_table("prompts")

    bind = op.get_bind()
    prompt_status_enum.drop(bind, checkfirst=True)
