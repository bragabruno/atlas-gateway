"""Accounting adapter — wires GW-14/15 into the chat request path.

The composition-root implementation of the service-layer `Recorder` port
(`app.services.chat_service.Recorder`): it prices a realized `CallContext`
into a `CallRecord`, persists it via `CallRecorder` (GW-14), and fans the
matching `atlas.calls.v1` event out via `EventPublisher` (GW-15) when one is
wired.

Contract notes (per the Recorder port docstring):
- **Never raises on the request path** — any persistence/publish failure is
  logged and swallowed; a billing problem must not fail a user's completion.
- `api_key_id` arrives as the bearer-key *string*; `call_records.api_key_id`
  is a UUID FK onto `api_keys.id`, so the key is mapped through a deterministic
  UUIDv5 (same namespace the seeder uses when inserting `api_keys` rows).
- Rates come from an injected `{alias_or_model: Rates}` mapping (built from
  `ALIAS_SEED` by the composition root). Models with no row — `mock` and the
  locally wired Ollama ids — price at zero, which is exactly right for the
  free local loop.
- `provider` is inferred from the model id (claude→anthropic, gemini→google,
  otherwise openai — also correct for Ollama's OpenAI-protocol models and the
  placeholder for mock). Latency is not measured at this seam yet; recorded as
  0 until timing is plumbed through `CallContext`.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from decimal import Decimal

from app.accounting.events import EventPublisher
from app.accounting.recorder import CallRecord, CallRecorder, Rates, compute_cost
from app.repositories.tables import ProviderEnum
from app.services.chat_service import CallContext

log = logging.getLogger(__name__)

#: Namespace for deterministic api-key UUIDs (must match scripts/seed_db.py).
API_KEY_NS = uuid.UUID("3f1c8d5a-9b2e-4f7c-8a1d-6e5b4c3a2f10")

_ZERO_RATES = Rates(in_per_1m=Decimal("0"), out_per_1m=Decimal("0"))


def api_key_uuid(api_key: str) -> uuid.UUID:
    """Deterministic UUID for a bearer-key string (UUIDv5 in API_KEY_NS)."""
    return uuid.uuid5(API_KEY_NS, api_key)


def provider_for_model(model: str) -> ProviderEnum:
    """Best-effort provider classification from a model id."""
    lowered = model.lower()
    if lowered.startswith("claude"):
        return ProviderEnum.anthropic
    if lowered.startswith(("gemini", "text-embedding-004")):
        return ProviderEnum.google
    return ProviderEnum.openai


class AccountingRecorder:
    """`Recorder`-port adapter: price → persist (GW-14) → publish (GW-15)."""

    def __init__(
        self,
        recorder: CallRecorder,
        *,
        rates: Mapping[str, Rates],
        publisher: EventPublisher | None = None,
        app_name: str = "atlas-gateway",
    ) -> None:
        self._recorder = recorder
        self._rates = rates
        self._publisher = publisher
        self._app_name = app_name

    async def record(self, call: CallContext) -> None:
        try:
            record = self._to_record(call)
            await self._recorder.record(record)
            if self._publisher is not None:
                await self._publisher.publish_record(record)
        except Exception:
            log.warning(
                "accounting record failed (model=%s) — swallowed per GW-15",
                call.model,
                exc_info=True,
            )

    def _to_record(self, call: CallContext) -> CallRecord:
        rates = self._rates.get(call.model, _ZERO_RATES)
        return CallRecord(
            id=uuid.uuid4(),
            api_key_id=api_key_uuid(call.api_key_id),
            app=self._app_name,
            model=call.model,
            provider=provider_for_model(call.model),
            usage=call.usage,
            cost=compute_cost(call.usage, rates),
            latency_ms=0,
            status=200,
        )


def rates_from_seed() -> dict[str, Rates]:
    """Build the alias→Rates mapping from `ALIAS_SEED` (offline source).

    Keyed by alias (`smart`, `deep`, …) — the names requests actually carry.
    Unlisted models (mock, Ollama ids) fall back to zero rates in the adapter.
    """
    from app.repositories.seed import ALIAS_SEED

    return {
        row["alias"]: Rates(
            in_per_1m=row["in_price_per_1m"],
            out_per_1m=row["out_price_per_1m"],
        )
        for row in ALIAS_SEED
    }
