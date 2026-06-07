"""SQLAlchemy 2.0 typed models for the gateway-owned tables (GW-9).

Mirrors the DDL in atlas-docs/03 §1 for the four tables this service owns
(ADR-015): `model_aliases`, `api_keys`, `budgets`, `call_records`. Models use
SQLAlchemy 2.0 typed `Mapped[...]` columns and form the schema source of truth
Alembic diffs against (ADR-010). Column types are chosen to compile to the
atlas-docs Postgres DDL in production while remaining valid under SQLite for the
offline schema smoke test (the `Enum` types render as native PG enums and as
VARCHAR+CHECK on SQLite). Per ADR-016 nothing outside `app.repositories`
imports these. See atlas-docs/03 §1.1–1.4.
"""

from __future__ import annotations

import enum
import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.repositories.base import Base


class ProviderEnum(enum.Enum):
    """Upstream LLM provider — the `provider_enum` Postgres type."""

    anthropic = "anthropic"
    openai = "openai"
    google = "google"
    azure_openai = "azure_openai"


class KeyStatusEnum(enum.Enum):
    """API-key lifecycle state — the `key_status_enum` Postgres type."""

    active = "active"
    suspended = "suspended"
    revoked = "revoked"


class ModelAlias(Base):
    """A routable model alias with pinned per-1M token prices (atlas-docs §1.1).

    Prices are pinned at alias-creation time so historical `call_records` cost is
    reproducible; `per_key_overrides` carries negotiated per-tenant rates.
    """

    __tablename__ = "model_aliases"

    alias: Mapped[str] = mapped_column(Text, primary_key=True)
    primary_model_id: Mapped[str] = mapped_column(Text, nullable=False)
    fallback_model_id: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[ProviderEnum] = mapped_column(
        Enum(ProviderEnum, name="provider_enum"), nullable=False
    )
    in_price_per_1m: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    out_price_per_1m: Mapped[Decimal] = mapped_column(Numeric(10, 6), nullable=False)
    per_key_overrides: Mapped[dict[str, object]] = mapped_column(
        JSON, nullable=False, default=dict, server_default=text("'{}'")
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ApiKey(Base):
    """A hashed-secret API key with owner/app and lifecycle status (atlas-docs §1.2)."""

    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    hashed_secret: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    app: Mapped[str] = mapped_column(Text, nullable=False)
    owner: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[KeyStatusEnum] = mapped_column(
        Enum(KeyStatusEnum, name="key_status_enum"),
        nullable=False,
        default=KeyStatusEnum.active,
        server_default=text("'active'"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("idx_api_keys_hashed_secret", "hashed_secret"),
        Index("idx_api_keys_app", "app"),
    )


class Budget(Base):
    """Per-key spend cap and running spend in the current period (atlas-docs §1.3)."""

    __tablename__ = "budgets"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    api_key_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("api_keys.id", ondelete="CASCADE"), nullable=False
    )
    monthly_cap_usd: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    current_spend: Mapped[Decimal] = mapped_column(
        Numeric(12, 4), nullable=False, default=Decimal("0"), server_default=text("0")
    )
    alert_at_80pct: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=text("true")
    )
    reset_cycle: Mapped[str] = mapped_column(
        Text, nullable=False, default="monthly", server_default=text("'monthly'")
    )
    period_start: Mapped[date] = mapped_column(Date, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("api_key_id", name="budgets_api_key_unique"),
        Index("idx_budgets_api_key_id", "api_key_id"),
    )


class CallRecord(Base):
    """One priced, audited record per completed LLM call (atlas-docs §1.4).

    Token fields and `computed_cost_usd` are stored verbatim so cost stays
    reproducible even if alias prices change later (atlas-docs §2).
    """

    __tablename__ = "call_records"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    api_key_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("api_keys.id"), nullable=False)
    app: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_version_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    alias: Mapped[str | None] = mapped_column(
        Text, ForeignKey("model_aliases.alias"), nullable=True
    )
    model: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[ProviderEnum] = mapped_column(
        Enum(ProviderEnum, name="provider_enum"), nullable=False
    )
    input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    output_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    cache_creation_input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    cache_read_input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    computed_cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 8), nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index(
            "idx_call_records_api_key_id",
            "api_key_id",
            text("created_at DESC"),
        ),
        Index("idx_call_records_created_at", text("created_at DESC")),
        Index("idx_call_records_alias", "alias"),
    )
