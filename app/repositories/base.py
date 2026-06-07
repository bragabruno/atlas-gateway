"""SQLAlchemy 2.0 declarative base for the gateway persistence schema (GW-9).

`Base` is the single declarative registry every ORM model inherits from, so
`Base.metadata` is the schema source of truth Alembic diffs against (ADR-010).
Per ADR-016 all DB types live in `app.repositories`; the service layer never
imports SQLAlchemy. Hot-path access stays on asyncpg (ADR-010) — these mapped
classes back the registry/accounting paths and own the table definitions.
See atlas-docs/03 §1.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base; `Base.metadata` is the authoritative schema."""
