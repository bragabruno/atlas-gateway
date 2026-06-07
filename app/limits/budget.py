"""GW-17 — Monthly per-key budget enforcement over Redis → 429 + 80% alert.

A thin capability adapter (ADR-016): it tracks a key's running spend against its
monthly cap and decides whether a request may proceed. On a reached cap it raises
:class:`BudgetExceeded`, whose body is the exact ``429`` shape the gateway returns
(atlas-docs/03 §5.2 *429 — Budget Exceeded*); the controller maps it to the HTTP
response (wiring later). As spend crosses 80% of the cap it emits a one-shot alert
signal (atlas-docs/03 §1.3 ``alert_at_80pct``) for the alerting consumer.

**Model.** Mirrors the ``budgets`` row (atlas-docs/03 §1.3): a ``monthly_cap_usd``
hard cap, a ``current_spend`` running total, ``period_start`` for the active
window, and ``alert_at_80pct``. The source of truth is Postgres + the Kafka
``atlas.calls.v1`` aggregation (atlas-docs/03 §3); this adapter keeps the *hot
read/write* in Redis so the request path never blocks on the DB. The cap is
enforced **pre-call** against the spend recorded so far — a single call cannot be
split mid-flight, so admission uses the already-accrued spend and ``charge``
records the realized cost afterward (post-call, from `Usage` via GW-14).

**Cycle reset.** Spend is namespaced by billing period (``YYYY-MM`` for the
default monthly cycle), so a new month reads a fresh zero bucket automatically and
the previous period's key self-evicts via TTL. ``period`` is injected (computed
from ``period_start``/``reset_cycle`` upstream) so the adapter stays a pure
function of its inputs and is deterministic in tests.

**Atomicity (ADR-011).** As with the rate limiter, ``lupa`` is not in the pinned
deps, so the read-decide-write uses Redis' optimistic ``WATCH``/``MULTI``/``EXEC``
transaction rather than ``EVAL`` — identical atomicity, no new dependency. The
80% crossing is detected inside the transaction (was-below vs now-at-or-above) so
the alert fires exactly once across replicas.

Pinned deps: redis 7.4.0, fakeredis 2.35.1 (dev/tests). All money is ``Decimal``
(never floated), matching GW-14. See GW-17 + ADR-011 + atlas-docs/03 §1.3, §5.2.
"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import cast

import redis.asyncio as redis_async

from app.limits._redis_typing import TxnPipe, run_transaction

#: Key namespace prefix; keeps budget counters distinct from the cache,
#: rate-limit, and circuit-breaker keyspaces on the shared Redis instance.
_KEY_PREFIX = "atlas:budget"

#: Hash field holding the period's running spend as a Decimal-parsable string.
_FIELD_SPEND = "spend"

#: Hash field marking that the 80% alert already fired this period (one-shot).
_FIELD_ALERTED = "alerted"

#: Fraction of the cap at which the soft alert fires (atlas-docs/03 §1.3).
_ALERT_FRACTION = Decimal("0.80")

#: Spend keys live a little over a 31-day month so a finished period self-evicts
#: while an active one is always refreshed on write.
_PERIOD_TTL_SECONDS = 35 * 24 * 60 * 60


def monthly_period(period_start: date) -> str:
    """Render the ``YYYY-MM`` period segment for a monthly cycle.

    The spend bucket is namespaced by this segment, so rolling into a new month
    yields a fresh zero bucket and the prior month's key expires on its own.
    """
    return f"{period_start.year:04d}-{period_start.month:02d}"


@dataclass(frozen=True, slots=True)
class BudgetError:
    """The ``error`` object of the ``429`` budget body (atlas-docs/03 §5.2)."""

    code: str
    message: str
    type: str
    param: None = None


class BudgetExceeded(Exception):
    """Raised when a key's monthly cap is reached — carries the spec ``429`` body.

    ``body`` is the exact JSON the gateway returns (``{"error": {...}}`` with
    ``code="budget_exceeded"``, ``type="budget_error"``). The controller maps this
    to an HTTP 429 (wiring later); business logic stays HTTP-free (ADR-016).
    """

    def __init__(self, *, cap_usd: Decimal, resets_on: date) -> None:
        self.cap_usd = cap_usd
        self.resets_on = resets_on
        self.error = BudgetError(
            code="budget_exceeded",
            message=(
                f"Monthly spend cap of ${cap_usd:.2f} has been reached for this "
                f"API key. Cap resets on {resets_on.isoformat()}."
            ),
            type="budget_error",
        )
        super().__init__(self.error.message)

    @property
    def body(self) -> dict[str, object]:
        """The exact ``429`` response body (atlas-docs/03 §5.2)."""
        return {
            "error": {
                "code": self.error.code,
                "message": self.error.message,
                "type": self.error.type,
                "param": self.error.param,
            }
        }


@dataclass(frozen=True, slots=True)
class BudgetState:
    """Snapshot of a key's budget after an admission check (for telemetry/tests).

    ``alert_fired`` is ``True`` only on the single transition across 80% of the
    cap (edge-triggered); ``spend`` is the running total in the current period.
    """

    spend: Decimal
    cap_usd: Decimal
    alert_fired: bool


@dataclass(slots=True)
class _Accrual:
    """Mutable carrier for the outcome of one atomic accrual transaction.

    A typed (not ``list[object]``) holder threads the read-modify-write result
    out of the transaction closure for the type checker, with no narrowing casts.
    """

    spend: Decimal
    exceeded: bool
    alert_fired: bool


class MonthlyBudgetEnforcer:
    """Per-key monthly budget enforcer backed by an injected ``redis.asyncio`` client.

    The client is injected (composition root supplies a real client; tests supply
    ``fakeredis``), so this adapter does no connection management. ``cap_usd``,
    ``period``, ``resets_on``, and ``alert_at_80pct`` come from the key's
    ``budgets`` row (atlas-docs/03 §1.3); the adapter is a pure function of them
    plus the Redis-tracked running spend. The client may be configured with
    ``decode_responses=True`` (matching the rest of the gateway).
    """

    def __init__(
        self,
        client: redis_async.Redis,
        *,
        cap_usd: Decimal,
        period: str,
        resets_on: date,
        alert_at_80pct: bool = True,
    ) -> None:
        if cap_usd <= 0:
            raise ValueError(f"cap_usd must be positive, got {cap_usd}")
        if not period:
            raise ValueError("period must be a non-empty period segment")
        self._client = client
        self._cap_usd = cap_usd
        self._period = period
        self._resets_on = resets_on
        self._alert_at_80pct = alert_at_80pct

    def _key(self, api_key_id: str) -> str:
        return f"{_KEY_PREFIX}:{api_key_id}:{self._period}"

    @staticmethod
    def _as_decimal(raw: str | bytes | None, default: Decimal) -> Decimal:
        """Read a hash field that may be ``str``, ``bytes``, or absent."""
        if raw is None:
            return default
        if isinstance(raw, bytes):
            return Decimal(raw.decode())
        return Decimal(raw)

    @staticmethod
    def _as_flag(raw: str | bytes | None) -> bool:
        """Read the one-shot alert flag (absent/``"0"``/``b"0"`` → False)."""
        if raw is None:
            return False
        if isinstance(raw, bytes):
            raw = raw.decode()
        return raw == "1"

    async def _accrue(self, *, api_key_id: str, cost: Decimal, enforce: bool) -> _Accrual:
        """Atomically add ``cost`` to the period spend (the shared read-decide-write).

        Runs as one optimistic transaction: read spend + alert flag, optionally
        deny when ``enforce`` and already at/over the cap, else accrue ``cost``,
        detect the one-shot 80% crossing, and write back with a refreshed TTL.
        """
        if cost < 0:
            raise ValueError(f"cost must be non-negative, got {cost}")
        key = self._key(api_key_id)
        result = _Accrual(spend=Decimal("0"), exceeded=False, alert_fired=False)

        async def _txn(pipe: TxnPipe) -> None:
            raw = await pipe.hmget(key, [_FIELD_SPEND, _FIELD_ALERTED])
            spend = self._as_decimal(raw[0], Decimal("0"))
            already_alerted = self._as_flag(raw[1])

            if enforce and spend >= self._cap_usd:
                # Hard cap already reached: deny, accrue nothing, no write.
                result.exceeded = True
                result.spend = spend
                pipe.multi()  # empty MULTI/EXEC keeps the WATCH cleanly released
                return

            new_spend = spend + cost
            threshold = self._cap_usd * _ALERT_FRACTION
            crossed = (
                self._alert_at_80pct
                and not already_alerted
                and spend < threshold
                and new_spend >= threshold
            )
            result.spend = new_spend
            result.alert_fired = crossed

            pipe.multi()
            mapping: dict[str, str] = {_FIELD_SPEND: str(new_spend)}
            if crossed:
                mapping[_FIELD_ALERTED] = "1"
            pipe.hset(key, mapping=mapping)
            pipe.expire(key, _PERIOD_TTL_SECONDS)

        await run_transaction(self._client, _txn, key)
        return result

    async def check(self, *, api_key_id: str, cost: Decimal) -> BudgetState:
        """Admit a call expected to add ``cost``, or raise :class:`BudgetExceeded`.

        Admission is pre-call: it raises if spend is **already** at or over the
        cap. Otherwise it accrues ``cost`` atomically and returns the new
        :class:`BudgetState`, with ``alert_fired`` set on the single 80% crossing.
        Use :meth:`charge` instead when reconciling realized post-call cost without
        re-checking the cap. ``cost`` must be non-negative.
        """
        result = await self._accrue(api_key_id=api_key_id, cost=cost, enforce=True)
        if result.exceeded:
            raise BudgetExceeded(cap_usd=self._cap_usd, resets_on=self._resets_on)
        return BudgetState(
            spend=result.spend,
            cap_usd=self._cap_usd,
            alert_fired=result.alert_fired,
        )

    async def charge(self, *, api_key_id: str, cost: Decimal) -> BudgetState:
        """Record realized ``cost`` post-call without enforcing the cap.

        Used to reconcile the actual billed cost (from `Usage` via GW-14) after a
        call completes; it never raises :class:`BudgetExceeded` (the call already
        happened) but still fires the one-shot 80% alert on crossing.
        """
        result = await self._accrue(api_key_id=api_key_id, cost=cost, enforce=False)
        return BudgetState(
            spend=result.spend,
            cap_usd=self._cap_usd,
            alert_fired=result.alert_fired,
        )

    async def current_spend(self, *, api_key_id: str) -> Decimal:
        """Return the running spend for this key in the current period."""
        raw = await cast(
            "Awaitable[str | bytes | None]",
            self._client.hget(self._key(api_key_id), _FIELD_SPEND),
        )
        return self._as_decimal(raw, Decimal("0"))
