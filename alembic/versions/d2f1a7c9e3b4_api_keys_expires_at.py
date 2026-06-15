"""Add api_keys.expires_at for key lifecycle/expiry (BRA-881).

Adds a nullable `expires_at TIMESTAMPTZ` to `api_keys` so keys can carry an
expiry that the auth path enforces at runtime (alongside the existing `status`
enum), making revocation/expiry take effect without a redeploy. Revises the
prompt-registry head (cb474b300c35).

Revision ID: d2f1a7c9e3b4
Revises: cb474b300c35
Create Date: 2026-06-15

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d2f1a7c9e3b4"
down_revision: str | None = "cb474b300c35"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "api_keys",
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("api_keys", "expires_at")
