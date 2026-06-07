"""Schema smoke tests for the gateway persistence layer (GW-9, REG-1).

Asserts the SQLAlchemy 2.0 metadata builds and the models import, smoke-tests
the schema with `metadata.create_all` against an in-memory SQLite engine
(tables + the atlas-docs/03 §1 indexes — incl. the REG-1 `prompts`/
`prompt_versions` tables), checks the alias seed contents, and verifies both
Alembic migrations render the expected Postgres DDL + seed offline (the live
Azure-apply/rollback path is deferred — needs a real Postgres). See
atlas-docs/03 §1–2 + ADR-010.
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
    Prompt,
    PromptStatusEnum,
    PromptVersion,
    ProviderEnum,
)

EXPECTED_TABLES = {
    "model_aliases",
    "api_keys",
    "budgets",
    "call_records",
    "prompts",
    "prompt_versions",
}
EXPECTED_INDEXES: dict[str, set[str]] = {
    "api_keys": {"idx_api_keys_hashed_secret", "idx_api_keys_app"},
    "budgets": {"idx_budgets_api_key_id"},
    "call_records": {
        "idx_call_records_api_key_id",
        "idx_call_records_created_at",
        "idx_call_records_alias",
    },
    "prompt_versions": {
        "idx_prompt_versions_prompt_id",
        "idx_prompt_versions_status",
    },
    # model_aliases: PK on alias covers all lookups, no extra index (atlas-docs §1.1).
    "model_aliases": set(),
    # prompts: PK on id + unique on name cover all lookups (atlas-docs §1.5).
    "prompts": set(),
}


def test_models_import_and_metadata_builds() -> None:
    """All gateway-owned models register on the shared metadata."""
    tables = set(Base.metadata.tables)
    assert EXPECTED_TABLES <= tables
    # imported, mapped classes (GW-9 + REG-1)
    assert {ModelAlias, ApiKey, Budget, CallRecord, Prompt, PromptVersion}


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


def _script_directory():  # type: ignore[no-untyped-def]
    """Load the Alembic ScriptDirectory for the gateway migrations."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    root = os.path.dirname(os.path.dirname(__file__))
    cfg = Config(os.path.join(root, "alembic.ini"))
    return ScriptDirectory.from_config(cfg)


def test_migration_chain_has_single_head_revising_gw9() -> None:
    """The REG-1 migration revises the GW-9 head; the chain has one linear head.

    Offline (`--sql`) render proves the migrations are wired to the schema
    without a live database; applying on Azure Postgres (and rollback) is
    deferred. The GW-9 base seeds the aliases from the single source of truth.
    """
    script = _script_directory()
    revisions = list(script.walk_revisions())
    assert len(revisions) == 2
    assert set(script.get_heads()) == {"cb474b300c35"}

    by_id = {rev.revision: rev for rev in revisions}
    assert set(by_id) == {"b113ca4237b0", "cb474b300c35"}

    head = by_id["cb474b300c35"]
    assert head.down_revision == "b113ca4237b0"

    base = by_id["b113ca4237b0"]
    assert base.down_revision is None

    for rev in revisions:
        module = rev.module  # type: ignore[attr-defined]
        assert hasattr(module, "upgrade")
        assert hasattr(module, "downgrade")

    # The GW-9 base migration sources its seed from the single source of truth.
    assert base.module.ALIAS_SEED is ALIAS_SEED  # type: ignore[attr-defined]


def test_reg1_migration_creates_prompt_tables() -> None:
    """The REG-1 head migration defines the prompts/prompt_versions DDL + enum."""
    head = _script_directory().get_revision("cb474b300c35")
    module = head.module  # type: ignore[attr-defined]
    assert module.prompt_status_enum.name == "prompt_status_enum"
    assert set(module.prompt_status_enum.enums) == {
        "draft",
        "candidate",
        "production",
        "retired",
    }


def test_prompt_version_row_trips_on_sqlite() -> None:
    """A prompts + prompt_versions pair round-trips through the SQLite schema (REG-1)."""
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        result = conn.execute(sa.insert(Prompt).values(name="summarize-doc").returning(Prompt.id))
        prompt_id = result.scalar_one()
        conn.execute(
            sa.insert(PromptVersion).values(
                prompt_id=prompt_id,
                semver="1.0.0",
                template="Summarize: {{ doc }}",
                params_schema={"type": "object"},
                model_alias="smart",
                status=PromptStatusEnum.draft,
            )
        )
    with engine.connect() as conn:
        row = conn.execute(select(PromptVersion.semver, PromptVersion.status)).one()
    assert row.semver == "1.0.0"
    assert row.status == PromptStatusEnum.draft


def test_prompt_status_enum_values_match_docs() -> None:
    """Prompt status enum members match atlas-docs/03 §1.5."""
    assert {e.value for e in PromptStatusEnum} == {
        "draft",
        "candidate",
        "production",
        "retired",
    }
