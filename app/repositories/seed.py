"""Seed data for the `model_aliases` table (GW-9).

Single source of truth for the initial alias rows, shared by the Alembic data
migration (which inserts them) and the schema test (which asserts them). Values
mirror the README alias table and atlas-docs/03 §2 "Seed Alias Prices". Prices
for `gpt-4.1`/Gemini/embeddings are deploy-time placeholders managed via env
vars (atlas-docs §2) and overwritten by the data migration when present; the
literals here are the documented defaults. `fallback_model_id` is NOT NULL in
the schema, so aliases documented with no separate fallback ("—") fall back to
their own primary model (i.e. no cross-model failover). See atlas-docs/03 §1.1,
§2.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TypedDict


class AliasSeed(TypedDict):
    """One `model_aliases` seed row."""

    alias: str
    primary_model_id: str
    fallback_model_id: str
    provider: str
    in_price_per_1m: Decimal
    out_price_per_1m: Decimal


ALIAS_SEED: tuple[AliasSeed, ...] = (
    {
        "alias": "smart",
        "primary_model_id": "claude-sonnet-4-6",
        "fallback_model_id": "gpt-4.1",
        "provider": "anthropic",
        "in_price_per_1m": Decimal("3.00"),
        "out_price_per_1m": Decimal("15.00"),
    },
    {
        "alias": "deep",
        "primary_model_id": "claude-opus-4-8",
        "fallback_model_id": "gpt-4.1",
        "provider": "anthropic",
        "in_price_per_1m": Decimal("5.00"),
        "out_price_per_1m": Decimal("25.00"),
    },
    {
        "alias": "fast",
        "primary_model_id": "claude-haiku-4-5",
        "fallback_model_id": "gpt-4.1-mini",
        "provider": "anthropic",
        "in_price_per_1m": Decimal("1.00"),
        "out_price_per_1m": Decimal("5.00"),
    },
    {
        "alias": "balanced",
        "primary_model_id": "gemini-2.5-pro",
        "fallback_model_id": "gemini-2.5-pro",
        "provider": "google",
        "in_price_per_1m": Decimal("0.00"),
        "out_price_per_1m": Decimal("0.00"),
    },
    {
        "alias": "embed",
        "primary_model_id": "text-embedding-3-large",
        "fallback_model_id": "text-embedding-3-large",
        "provider": "openai",
        "in_price_per_1m": Decimal("0.00"),
        "out_price_per_1m": Decimal("0.00"),
    },
)
"""Initial alias rows, in stable insertion order (smart/deep/fast/balanced/embed)."""
