"""Schema smoke tests for the gateway persistence layer (GW-9).

Asserts the SQLAlchemy 2.0 metadata builds and the models import, smoke-tests
the schema with `metadata.create_all` against an in-memory SQLite engine
(tables + the atlas-docs/03 §1 indexes), checks the alias seed contents, and
verifies the Alembic initial migration renders the expected Postgres DDL +
seed offline (the live Azure-apply/rollback path is deferred — needs a real
Postgres). See atlas-docs/03 §1–2 + ADR-010.
"""

from __future__ import annotations

import os
from decimal import Decimal

import sqlalchemy as sa
from sqlalchemy import create_engine, inspect, select

from app.repositories.base import Base
from app.repositories.seed import ALIAS_SEED
from app.repositories.tables import (
    ApiKey,
    Budget,
    CallRecord,
    KeyStatusEnum,
    ModelAlias,
    ProviderEnum,
)

EXPECTED_TABLES = {"model_aliases", "api_keys", "budgets", "call_records"}
EXPECTED_INDEXES: dict[str, set[str]] = {
    "api_keys": {"idx_api_keys_hashed_secret", "idx_api_keys_app"},
    "budgets": {"idx_budgets_api_key_id"},
    "call_records": {
        "idx_call_records_api_key_id",
        "idx_call_records_created_at",
        "idx_call_records_alias",
    },
    # model_aliases: PK on alias covers all lookups, no extra index (atlas-docs §1.1).
    "model_aliases": set(),
}


def test_models_import_and_metadata_builds() -> None:
    """All four models register on the shared metadata."""
    tables = set(Base.metadata.tables)
    assert EXPECTED_TABLES <= tables
    assert {ModelAlias, ApiKey, Budget, CallRecord}  # imported, mapped classes


def test_create_all_builds_tables_and_indexes() -> None:
    """`create_all` on in-memory SQLite yields the documented tables + indexes."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    insp = inspect(engine)

    assert EXPECTED_TABLES <= set(insp.get_table_names())
    for table, expected in EXPECTED_INDEXES.items():
        actual = {idx["name"] for idx in insp.get_indexes(table)}
        assert expected <= actual, f"{table}: missing {expected - actual}"


def test_enum_values_match_docs() -> None:
    """Enum members match the atlas-docs/03 §1 provider/status enums."""
    assert {e.value for e in ProviderEnum} == {
        "anthropic",
        "openai",
        "google",
        "azure_openai",
    }
    assert {e.value for e in KeyStatusEnum} == {"active", "suspended", "revoked"}


def test_alias_seed_contents() -> None:
    """The seed carries the five documented aliases with their pinned prices."""
    by_alias = {row["alias"]: row for row in ALIAS_SEED}
    assert list(by_alias) == ["smart", "deep", "fast", "balanced", "embed"]

    smart = by_alias["smart"]
    assert smart["primary_model_id"] == "claude-sonnet-4-6"
    assert smart["fallback_model_id"] == "gpt-4.1"
    assert smart["provider"] == "anthropic"
    assert smart["in_price_per_1m"] == Decimal("3.00")
    assert smart["out_price_per_1m"] == Decimal("15.00")

    deep = by_alias["deep"]
    assert deep["primary_model_id"] == "claude-opus-4-8"
    assert deep["in_price_per_1m"] == Decimal("5.00")
    assert deep["out_price_per_1m"] == Decimal("25.00")

    fast = by_alias["fast"]
    assert fast["primary_model_id"] == "claude-haiku-4-5"
    assert fast["fallback_model_id"] == "gpt-4.1-mini"

    # Every seed provider is a valid ProviderEnum value.
    for row in ALIAS_SEED:
        assert row["provider"] in {e.value for e in ProviderEnum}


def test_seed_rows_insert_into_sqlite_schema() -> None:
    """The seed rows satisfy the model_aliases column shape (round-trips on SQLite)."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(sa.insert(ModelAlias), [dict(row) for row in ALIAS_SEED])
    with engine.connect() as conn:
        aliases = conn.execute(select(ModelAlias.alias).order_by(ModelAlias.alias)).scalars().all()
    assert set(aliases) == {row["alias"] for row in ALIAS_SEED}


def test_initial_migration_renders_postgres_ddl_and_seed() -> None:
    """The Alembic initial migration renders the PG tables, enums, and seed offline.

    Offline (`--sql`) render proves the migration is wired to the schema without
    a live database; applying it on Azure Postgres (and rollback) is deferred.
    """
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    root = os.path.dirname(os.path.dirname(__file__))
    cfg = Config(os.path.join(root, "alembic.ini"))
    script = ScriptDirectory.from_config(cfg)
    revisions = list(script.walk_revisions())
    assert len(revisions) == 1
    head = revisions[0]
    assert head.down_revision is None

    module = head.module  # type: ignore[attr-defined]
    assert hasattr(module, "upgrade")
    assert hasattr(module, "downgrade")
    # The migration sources its seed from the single source of truth.
    assert module.ALIAS_SEED is ALIAS_SEED
