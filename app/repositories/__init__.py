"""Persistence layer (repository pattern).

DB access lives here behind repository interfaces so the service layer depends
on an abstraction, not raw SQL. Per ADR-010, hot-path repositories use asyncpg
and the rest use SQLAlchemy 2.0; Alembic owns migrations. Populated by GW-9
(schema/migrations), GW-14 (accounting recorder) and the registry tables. See
ADR-016.
"""
