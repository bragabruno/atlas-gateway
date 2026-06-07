"""GW-17 — per-key monthly budget enforcer (Redis → 429 + 80% alert).

Pins the adapter contract with ``fakeredis`` (zero network), all money
``Decimal``-exact:

- spend under the cap is admitted;
- a call once spend is at/over the cap is denied with the exact ``429`` body
  (atlas-docs/03 §5.2);
- the 80% alert fires exactly once on the crossing (edge-triggered, one-shot);
- spend resets when the billing period rolls over (a new ``period`` segment reads
  a fresh zero bucket);
- ``charge`` reconciles post-call cost without enforcing the cap;
- ``alert_at_80pct=False`` suppresses the alert; parameters are validated.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date
from decimal import Decimal

import pytest
import pytest_asyncio
from fakeredis.aioredis import FakeRedis

from app.limits.budget import (
    BudgetExceeded,
    MonthlyBudgetEnforcer,
    monthly_period,
)

_CAP = Decimal("50.00")
_JUNE = monthly_period(date(2026, 6, 1))
_RESETS_ON = date(2026, 7, 1)


@pytest_asyncio.fixture
async def client() -> AsyncIterator[FakeRedis]:
    fake = FakeRedis(decode_responses=True)
    try:
        yield fake
    finally:
        await fake.aclose()


def _enforcer(client: FakeRedis, *, alert: bool = True) -> MonthlyBudgetEnforcer:
    return MonthlyBudgetEnforcer(
        client,
        cap_usd=_CAP,
        period=_JUNE,
        resets_on=_RESETS_ON,
        alert_at_80pct=alert,
    )


async def test_under_cap_is_admitted(client: FakeRedis) -> None:
    budget = _enforcer(client)
    state = await budget.check(api_key_id="k1", cost=Decimal("10.00"))
    assert state.spend == Decimal("10.00")
    assert state.alert_fired is False
    assert await budget.current_spend(api_key_id="k1") == Decimal("10.00")


async def test_over_cap_raises_spec_429_body(client: FakeRedis) -> None:
    budget = _enforcer(client)
    # Drive spend to/over the cap, then the next call is denied.
    await budget.check(api_key_id="k1", cost=Decimal("50.00"))  # spend == cap
    with pytest.raises(BudgetExceeded) as exc_info:
        await budget.check(api_key_id="k1", cost=Decimal("0.01"))

    assert exc_info.value.body == {
        "error": {
            "code": "budget_exceeded",
            "message": (
                "Monthly spend cap of $50.00 has been reached for this API key. "
                "Cap resets on 2026-07-01."
            ),
            "type": "budget_error",
            "param": None,
        }
    }


async def test_denied_call_does_not_change_spend(client: FakeRedis) -> None:
    budget = _enforcer(client)
    await budget.check(api_key_id="k1", cost=_CAP)  # at cap
    with pytest.raises(BudgetExceeded):
        await budget.check(api_key_id="k1", cost=Decimal("5.00"))
    # The denied call accrued nothing — spend stays exactly at the cap.
    assert await budget.current_spend(api_key_id="k1") == _CAP


async def test_80pct_alert_fires_once_on_crossing(client: FakeRedis) -> None:
    budget = _enforcer(client)
    # 80% of 50 == 40. Cross it in one step (35 → 42).
    first = await budget.check(api_key_id="k1", cost=Decimal("35.00"))
    assert first.alert_fired is False
    crossing = await budget.check(api_key_id="k1", cost=Decimal("7.00"))  # 42 >= 40
    assert crossing.alert_fired is True
    # Spend keeps climbing but the one-shot alert does not re-fire.
    after = await budget.check(api_key_id="k1", cost=Decimal("3.00"))  # 45
    assert after.alert_fired is False


async def test_alert_fires_when_landing_exactly_on_threshold(client: FakeRedis) -> None:
    budget = _enforcer(client)
    state = await budget.check(api_key_id="k1", cost=Decimal("40.00"))  # exactly 80%
    assert state.alert_fired is True


async def test_alert_suppressed_when_disabled(client: FakeRedis) -> None:
    budget = _enforcer(client, alert=False)
    state = await budget.check(api_key_id="k1", cost=Decimal("45.00"))  # past 80%
    assert state.alert_fired is False


async def test_spend_resets_on_new_period(client: FakeRedis) -> None:
    june = _enforcer(client)
    await june.check(api_key_id="k1", cost=_CAP)  # June at cap
    with pytest.raises(BudgetExceeded):
        await june.check(api_key_id="k1", cost=Decimal("1.00"))

    # A new billing period reads a fresh, zero bucket for the same key.
    july = MonthlyBudgetEnforcer(
        client,
        cap_usd=_CAP,
        period=monthly_period(date(2026, 7, 1)),
        resets_on=date(2026, 8, 1),
    )
    assert await july.current_spend(api_key_id="k1") == Decimal("0")
    state = await july.check(api_key_id="k1", cost=Decimal("10.00"))
    assert state.spend == Decimal("10.00")
    assert state.alert_fired is False


async def test_budgets_are_isolated_per_key(client: FakeRedis) -> None:
    budget = _enforcer(client)
    await budget.check(api_key_id="k1", cost=_CAP)  # k1 at cap
    # k2 has its own fresh budget.
    state = await budget.check(api_key_id="k2", cost=Decimal("5.00"))
    assert state.spend == Decimal("5.00")
    with pytest.raises(BudgetExceeded):
        await budget.check(api_key_id="k1", cost=Decimal("1.00"))


async def test_charge_accrues_without_enforcing_cap(client: FakeRedis) -> None:
    budget = _enforcer(client)
    # charge() never raises even past the cap (the call already happened).
    state = await budget.charge(api_key_id="k1", cost=Decimal("60.00"))
    assert state.spend == Decimal("60.00")
    # ... but a subsequent pre-call check now denies.
    with pytest.raises(BudgetExceeded):
        await budget.check(api_key_id="k1", cost=Decimal("0.01"))


async def test_charge_fires_the_one_shot_alert(client: FakeRedis) -> None:
    budget = _enforcer(client)
    crossing = await budget.charge(api_key_id="k1", cost=Decimal("45.00"))  # past 80%
    assert crossing.alert_fired is True
    again = await budget.charge(api_key_id="k1", cost=Decimal("1.00"))
    assert again.alert_fired is False


async def test_spend_is_decimal_exact(client: FakeRedis) -> None:
    budget = _enforcer(client)
    await budget.check(api_key_id="k1", cost=Decimal("0.0000033"))
    await budget.check(api_key_id="k1", cost=Decimal("0.0000067"))
    spend = await budget.current_spend(api_key_id="k1")
    assert isinstance(spend, Decimal)
    assert spend == Decimal("0.0000100")


async def test_works_with_bytes_responses_too() -> None:
    fake = FakeRedis()  # decode_responses=False → bytes hash fields
    try:
        budget = MonthlyBudgetEnforcer(fake, cap_usd=_CAP, period=_JUNE, resets_on=_RESETS_ON)
        await budget.check(api_key_id="k1", cost=Decimal("10.00"))
        assert await budget.current_spend(api_key_id="k1") == Decimal("10.00")
    finally:
        await fake.aclose()


async def test_monthly_period_segment_shape() -> None:
    assert monthly_period(date(2026, 6, 1)) == "2026-06"
    assert monthly_period(date(2026, 12, 15)) == "2026-12"
    assert monthly_period(date(2027, 1, 31)) == "2027-01"


def test_rejects_non_positive_cap(client: FakeRedis) -> None:
    with pytest.raises(ValueError, match="cap_usd must be positive"):
        MonthlyBudgetEnforcer(client, cap_usd=Decimal("0"), period=_JUNE, resets_on=_RESETS_ON)


def test_rejects_empty_period(client: FakeRedis) -> None:
    with pytest.raises(ValueError, match="period must be a non-empty"):
        MonthlyBudgetEnforcer(client, cap_usd=_CAP, period="", resets_on=_RESETS_ON)


async def test_negative_cost_is_rejected(client: FakeRedis) -> None:
    budget = _enforcer(client)
    with pytest.raises(ValueError, match="cost must be non-negative"):
        await budget.check(api_key_id="k1", cost=Decimal("-1.00"))
