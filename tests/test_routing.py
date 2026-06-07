"""GW-10 — alias routing resolver.

Pins the resolver contract fully offline, straight from `ALIAS_SEED` (no live
DB): a known alias resolves to its primary/fallback/provider; a per-key
override wins field-by-field while omitted fields fall back to the row; and an
unknown alias raises an explicit `UnknownAliasError` (never a silent default).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.repositories.seed import ALIAS_SEED
from app.routing.aliases import (
    AliasResolver,
    RouteTarget,
    UnknownAliasError,
    rows_from_seed,
)


@dataclass(frozen=True)
class _Row:
    """A minimal structural `AliasRow` for injecting custom rows in tests."""

    provider: str
    primary_model_id: str
    fallback_model_id: str


def _resolver() -> AliasResolver:
    """An offline resolver backed by the alias seed."""
    return AliasResolver(rows_from_seed())


def test_primary_and_fallback_chosen_from_seed() -> None:
    target = _resolver().resolve("smart")
    assert target == RouteTarget(
        provider="anthropic",
        primary_model="claude-sonnet-4-6",
        fallback_model="gpt-4.1",
    )


def test_every_seed_alias_resolves() -> None:
    """Each seeded alias resolves to its row's provider/primary/fallback."""
    resolver = _resolver()
    for row in ALIAS_SEED:
        target = resolver.resolve(row["alias"])
        assert target.provider == row["provider"]
        assert target.primary_model == row["primary_model_id"]
        assert target.fallback_model == row["fallback_model_id"]


def test_alias_with_no_separate_fallback_uses_its_primary() -> None:
    """`deep`/`balanced`/`embed` document no separate fallback (== primary in seed)."""
    target = _resolver().resolve("balanced")
    assert target.primary_model == "gemini-2.5-pro"
    assert target.fallback_model == "gemini-2.5-pro"


def test_per_key_override_replaces_primary_model() -> None:
    target = _resolver().resolve("smart", {"primary_model_id": "claude-opus-4-8"})
    assert target.primary_model == "claude-opus-4-8"
    # Untouched fields still come from the seed row.
    assert target.provider == "anthropic"
    assert target.fallback_model == "gpt-4.1"


def test_per_key_override_replaces_provider_and_fallback() -> None:
    target = _resolver().resolve(
        "smart",
        {"provider": "azure_openai", "fallback_model_id": "gpt-4.1-mini"},
    )
    assert target.provider == "azure_openai"
    assert target.fallback_model == "gpt-4.1-mini"
    assert target.primary_model == "claude-sonnet-4-6"


def test_per_key_override_can_replace_all_three_fields() -> None:
    target = _resolver().resolve(
        "fast",
        {
            "provider": "openai",
            "primary_model_id": "gpt-4.1",
            "fallback_model_id": "gpt-4.1-mini",
        },
    )
    assert target == RouteTarget(
        provider="openai",
        primary_model="gpt-4.1",
        fallback_model="gpt-4.1-mini",
    )


def test_none_overrides_resolves_to_seed_row() -> None:
    assert _resolver().resolve("deep", None) == _resolver().resolve("deep")


def test_empty_overrides_blob_resolves_to_seed_row() -> None:
    assert _resolver().resolve("deep", {}) == _resolver().resolve("deep")


def test_unrelated_override_keys_are_ignored() -> None:
    """Keys that aren't routing fields (e.g. rate overrides) don't affect the route."""
    target = _resolver().resolve("smart", {"in_price_per_1m": "1.50", "unknown": "x"})
    assert target == _resolver().resolve("smart")


def test_blank_override_value_falls_back_to_row() -> None:
    """An empty/non-string override is ignored rather than poisoning the route."""
    target = _resolver().resolve("smart", {"primary_model_id": ""})
    assert target.primary_model == "claude-sonnet-4-6"


def test_unknown_alias_raises_explicit_error() -> None:
    with pytest.raises(UnknownAliasError) as exc:
        _resolver().resolve("does-not-exist")
    assert exc.value.alias == "does-not-exist"
    assert "does-not-exist" in str(exc.value)


def test_resolver_reads_injected_mapping_not_a_global() -> None:
    """The mapping is injected: a custom row is resolvable without touching the seed."""
    resolver = AliasResolver(
        {"custom": _Row(provider="google", primary_model_id="m1", fallback_model_id="m2")}
    )
    assert resolver.resolve("custom") == RouteTarget(
        provider="google", primary_model="m1", fallback_model="m2"
    )
    with pytest.raises(UnknownAliasError):
        resolver.resolve("smart")
