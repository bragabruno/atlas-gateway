"""GW-10 — Alias routing resolver.

Turns a routable alias (`smart`, `deep`, …) into a concrete `RouteTarget`
(provider + primary model + fallback model) the service layer can dispatch to.
A thin capability adapter (ADR-016): alias rows are read through an injected
`Mapping[str, AliasRow]`, so the resolver is exercised offline straight from
`ALIAS_SEED` (`rows_from_seed`) with no live DB, and a real repository can feed
the same shape later.

Per-key overrides (atlas-docs/03 §1.1 `model_aliases.per_key_overrides`) let a
tenant pin a different primary/fallback/provider for an alias; supplied
overrides win field-by-field over the alias row, anything they omit falls back
to the row. An unknown alias is an explicit `UnknownAliasError` (never a silent
default) so the controller can map it to a 404 — fail fast, no guessing.

This module decides routing *targets* only; it does not decide *when* to fall
back from primary to fallback (that is the resilience layer's call). See GW-10 +
ADR-016 + atlas-docs/03 §1.1, §2.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from app.repositories.seed import ALIAS_SEED

#: Keys in `per_key_overrides` that override a `RouteTarget` field. Only these
#: are honoured; any other key in the override blob is ignored (it may carry
#: rate overrides consumed elsewhere — see GW-14 cost pricing).
_OVERRIDE_PROVIDER = "provider"
_OVERRIDE_PRIMARY = "primary_model_id"
_OVERRIDE_FALLBACK = "fallback_model_id"


class AliasRow(Protocol):
    """Structural view of a `model_aliases` row the resolver needs.

    A real asyncpg/ORM row (or the frozen `_SeedRow` adapter `rows_from_seed`
    builds from the `AliasSeed` TypedDict) satisfies this, so the resolver never
    imports a concrete persistence type. The fields are read-only so a frozen
    row adapter qualifies. Only the routing fields are required here; prices are
    the accounting layer's concern (GW-14).
    """

    @property
    def provider(self) -> str: ...

    @property
    def primary_model_id(self) -> str: ...

    @property
    def fallback_model_id(self) -> str: ...


@dataclass(frozen=True, slots=True)
class _SeedRow:
    """Immutable `AliasRow` adapter over an `ALIAS_SEED` entry."""

    provider: str
    primary_model_id: str
    fallback_model_id: str


@dataclass(frozen=True, slots=True)
class RouteTarget:
    """The resolved routing target for an alias.

    `provider` is the upstream provider id (a `provider_enum` value);
    `primary_model` is the model to call first and `fallback_model` the model to
    fail over to (equal to `primary_model` when the alias has no separate
    fallback — see `seed` §1.1).
    """

    provider: str
    primary_model: str
    fallback_model: str


class UnknownAliasError(Exception):
    """Raised when no alias row exists for the requested alias."""

    def __init__(self, alias: str) -> None:
        self.alias = alias
        super().__init__(f"unknown alias: {alias}")


def rows_from_seed() -> dict[str, AliasRow]:
    """Build the injected alias mapping from `ALIAS_SEED` (offline source).

    Lets the resolver be constructed and tested with no live DB; a real
    repository can build the same `Mapping[str, AliasRow]` shape from the
    `model_aliases` table later.
    """
    return {
        row["alias"]: _SeedRow(
            provider=row["provider"],
            primary_model_id=row["primary_model_id"],
            fallback_model_id=row["fallback_model_id"],
        )
        for row in ALIAS_SEED
    }


def _override_str(overrides: Mapping[str, object], key: str, default: str) -> str:
    """Return a string override for `key`, or `default` when absent/empty.

    Non-string and empty override values are ignored (the row value stands)
    rather than silently coercing a malformed override into the route.
    """
    value = overrides.get(key)
    if isinstance(value, str) and value:
        return value
    return default


class AliasResolver:
    """Resolves an alias to a `RouteTarget`, honouring per-key overrides.

    The alias mapping is injected (composition root / tests supply it; pass
    `rows_from_seed()` for an offline resolver), so this adapter does no DB
    access of its own.
    """

    def __init__(self, rows: Mapping[str, AliasRow]) -> None:
        self._rows = rows

    def resolve(
        self,
        alias: str,
        per_key_overrides: Mapping[str, object] | None = None,
    ) -> RouteTarget:
        """Resolve `alias` to a `RouteTarget`.

        `per_key_overrides` (a tenant's `model_aliases.per_key_overrides` blob)
        overrides provider/primary/fallback field-by-field; omitted fields fall
        back to the alias row. An unknown alias raises `UnknownAliasError`.
        """
        row = self._rows.get(alias)
        if row is None:
            raise UnknownAliasError(alias)

        overrides: Mapping[str, object] = per_key_overrides or {}
        return RouteTarget(
            provider=_override_str(overrides, _OVERRIDE_PROVIDER, row.provider),
            primary_model=_override_str(overrides, _OVERRIDE_PRIMARY, row.primary_model_id),
            fallback_model=_override_str(overrides, _OVERRIDE_FALLBACK, row.fallback_model_id),
        )
