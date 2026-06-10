"""GRD-11 — per-check OTel counters on the guardrail chain.

Each check increments `atlas.guardrail.checks` with `guardrail.name` and
`guardrail.outcome` (pass | block). Tests use an InMemoryMetricReader so no
collector or real MeterProvider is needed.
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from app.domain.messages import Message
from app.guardrails.chain import (
    GuardrailChain,
    GuardrailContext,
    GuardrailPhase,
    GuardrailRejection,
)

_MESSAGES = [Message(role="user", content="ping")]


def _ctx() -> GuardrailContext:
    return GuardrailContext(tenant_id="t1", model="mock", messages=_MESSAGES)


def _make_chain(*guardrails: object, reader: InMemoryMetricReader) -> GuardrailChain:
    mp = MeterProvider(metric_readers=[reader])
    return GuardrailChain(pre=list(guardrails), meter_provider=mp)  # type: ignore[arg-type]


def _data_points(reader: InMemoryMetricReader) -> dict[tuple[str, str], int]:
    """Return {(name, outcome): value} from recorded data points."""
    result: dict[tuple[str, str], int] = {}
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "atlas.guardrail.checks":
                    for dp in metric.data.data_points:
                        key = (dp.attributes["guardrail.name"], dp.attributes["guardrail.outcome"])
                        result[key] = int(dp.value)
    return result


class _PassGuardrail:
    name = "always_pass"

    async def check(self, ctx: GuardrailContext) -> None:
        pass


class _BlockGuardrail:
    name = "always_block"

    async def check(self, ctx: GuardrailContext) -> None:
        raise GuardrailRejection(guardrail=self.name, phase=GuardrailPhase.PRE, reason="blocked")


@pytest.mark.asyncio
async def test_passing_check_increments_pass_counter() -> None:
    reader = InMemoryMetricReader()
    chain = _make_chain(_PassGuardrail(), reader=reader)
    await chain.run_pre(_ctx())
    pts = _data_points(reader)
    assert pts[("always_pass", "pass")] == 1
    assert ("always_pass", "block") not in pts


@pytest.mark.asyncio
async def test_blocking_check_increments_block_counter() -> None:
    reader = InMemoryMetricReader()
    chain = _make_chain(_BlockGuardrail(), reader=reader)
    with pytest.raises(GuardrailRejection):
        await chain.run_pre(_ctx())
    pts = _data_points(reader)
    assert pts[("always_block", "block")] == 1
    assert ("always_block", "pass") not in pts


@pytest.mark.asyncio
async def test_multiple_checks_accumulate_independently() -> None:
    reader = InMemoryMetricReader()
    chain = _make_chain(_PassGuardrail(), _PassGuardrail(), reader=reader)
    await chain.run_pre(_ctx())
    await chain.run_pre(_ctx())
    pts = _data_points(reader)
    assert pts[("always_pass", "pass")] == 4  # 2 checks × 2 runs


@pytest.mark.asyncio
async def test_block_stops_chain_and_second_check_not_recorded() -> None:
    reader = InMemoryMetricReader()
    chain = _make_chain(_BlockGuardrail(), _PassGuardrail(), reader=reader)
    with pytest.raises(GuardrailRejection):
        await chain.run_pre(_ctx())
    pts = _data_points(reader)
    assert pts[("always_block", "block")] == 1
    assert ("always_pass", "pass") not in pts
