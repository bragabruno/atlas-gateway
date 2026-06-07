"""Alembic migration environment for atlas-gateway (GW-9).

Wires Alembic to the SQLAlchemy schema declared in `app.repositories` so
autogenerate diffs against `Base.metadata` (the schema source of truth, ADR-010).
Supports both offline (`--sql`, emits DDL without a DBAPI connection) and online
(connects and applies) runs. The connection uses the psycopg3 SYNC driver per
ADR-010 — never asyncpg in the migration entrypoint. The database URL comes from
the ATLAS_DATABASE_URL environment variable (no connection string is committed);
the alembic.ini value is an inert placeholder. See atlas-docs/03 §1.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.repositories.base import Base
from app.repositories import tables  # noqa: F401  (registers models on Base.metadata)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Schema source of truth for autogenerate.
target_metadata = Base.metadata


def _database_url() -> str:
    """Resolve the migration DB URL from the environment (ADR-010, sync psycopg).

    Falls back to the inert alembic.ini placeholder only when ATLAS_DATABASE_URL
    is unset (e.g. local `--sql` offline rendering); production always sets it.
    """
    env_url = os.environ.get("ATLAS_DATABASE_URL")
    if env_url:
        return env_url
    ini_url = config.get_main_option("sqlalchemy.url")
    if not ini_url:
        raise RuntimeError(
            "ATLAS_DATABASE_URL is not set and no sqlalchemy.url is configured."
        )
    return ini_url


def run_migrations_offline() -> None:
    """Render migrations as SQL without a live DBAPI connection (`--sql`)."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect with the sync psycopg engine and apply migrations."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
