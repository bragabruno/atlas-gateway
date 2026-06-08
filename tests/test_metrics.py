"""GRD-11 — Per-check guardrail OTel metrics tests, offline.

A `MeterProvider` wired to an `InMemoryMetricReader` captures counters without a
collector or network, proving each check increments exactly one of its
``atlas.guardrail.<name>.pass`` / ``atlas.guardrail.<name>.block`` counters per
run, that a `MeteredGuardrailChain` records the right outcome per check while
preserving fail-fast ordering, and that a non-`GuardrailRejection` error is not
miscounted as a block. No raw content is attached to any metric. See GRD-11.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader, Sum

from app.domain.messages import Message
from app.guardrails.chain import (
    GuardrailChain,
    GuardrailContext,
    GuardrailPhase,
    GuardrailRejection,
)
from app.guardrails.metrics import GuardrailMetrics, MeteredGuardrailChain


@pytest.fixture
def reader() -> InMemoryMetricReader:
    return InMemoryMetricReader()


@pytest.fixture
def provider(reader: InMemoryMetricReader) -> MeterProvider:
    return MeterProvider(metric_readers=[reader])


def _counter_values(reader: InMemoryMetricReader) -> dict[str, int]:
    """Collapse the in-memory reader's data into ``{metric_name: total}``."""
    out: dict[str, int] = {}
    data = reader.get_metrics_data()
    if data is None:
        return out
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                aggregation = metric.data
                # Counters export a Sum of integer NumberDataPoints; narrow so the
                # value type is known (histogram points carry no `.value`).
                assert isinstance(aggregation, Sum)
                total = sum(int(point.value) for point in aggregation.data_points)
                out[metric.name] = out.get(metric.name, 0) + total
    return out


def _ctx() -> GuardrailContext:
    return GuardrailContext(
        tenant_id="tenant-a",
        model="mock",
        messages=[Message(role="user", content="hello")],
    )


class _PassingGuardrail:
    def __init__(self, name: str) -> None:
        self.name = name

    async def check(self, ctx: GuardrailContext) -> None:
        return None


class _BlockingGuardrail:
    def __init__(self, name: str) -> None:
        self.name = name

    async def check(self, ctx: GuardrailContext) -> None:
        raise GuardrailRejection(
            guardrail=self.name,
            phase=GuardrailPhase.PRE,
            reason="blocked",
        )


class _ExplodingGuardrail:
    name = "exploder"

    async def check(self, ctx: GuardrailContext) -> None:
        raise RuntimeError("infra failure, not a policy decision")


# ── direct counter API ──────────────────────────────────────────────────────


def test_record_pass_increments_pass_counter(
    provider: MeterProvider, reader: InMemoryMetricReader
) -> None:
    metrics = GuardrailMetrics(provider)
    metrics.record_pass("pii")
    values = _counter_values(reader)
    assert values["atlas.guardrail.pii.pass"] == 1
    assert "atlas.guardrail.pii.block" not in values


def test_record_block_increments_block_counter(
    provider: MeterProvider, reader: InMemoryMetricReader
) -> None:
    metrics = GuardrailMetrics(provider)
    metrics.record_block("injection")
    values = _counter_values(reader)
    assert values["atlas.guardrail.injection.block"] == 1


def test_repeated_record_accumulates(provider: MeterProvider, reader: InMemoryMetricReader) -> None:
    metrics = GuardrailMetrics(provider)
    metrics.record_pass("size")
    metrics.record_pass("size")
    metrics.record_pass("size")
    assert _counter_values(reader)["atlas.guardrail.size.pass"] == 3


# ── metered chain: one counter per check per run ────────────────────────────


async def test_metered_chain_records_pass_per_check(
    provider: MeterProvider, reader: InMemoryMetricReader
) -> None:
    chain = GuardrailChain(pre=[_PassingGuardrail("size"), _PassingGuardrail("pii")])
    metered = MeteredGuardrailChain(chain, GuardrailMetrics(provider))
    await metered.run_pre(_ctx())
    values = _counter_values(reader)
    assert values["atlas.guardrail.size.pass"] == 1
    assert values["atlas.guardrail.pii.pass"] == 1


async def test_metered_chain_records_block_and_is_fail_fast(
    provider: MeterProvider, reader: InMemoryMetricReader
) -> None:
    # size passes, injection blocks → pii must never run (fail-fast).
    chain = GuardrailChain(
        pre=[
            _PassingGuardrail("size"),
            _BlockingGuardrail("injection"),
            _PassingGuardrail("pii"),
        ]
    )
    metered = MeteredGuardrailChain(chain, GuardrailMetrics(provider))
    with pytest.raises(GuardrailRejection):
        await metered.run_pre(_ctx())
    values = _counter_values(reader)
    assert values["atlas.guardrail.size.pass"] == 1
    assert values["atlas.guardrail.injection.block"] == 1
    # pii never executed → no counter for it at all.
    assert "atlas.guardrail.pii.pass" not in values
    assert "atlas.guardrail.pii.block" not in values


async def test_metered_chain_post_phase_records(
    provider: MeterProvider, reader: InMemoryMetricReader
) -> None:
    chain = GuardrailChain(post=[_PassingGuardrail("content_policy")])
    metered = MeteredGuardrailChain(chain, GuardrailMetrics(provider))
    await metered.run_post(_ctx())
    assert _counter_values(reader)["atlas.guardrail.content_policy.pass"] == 1


async def test_non_rejection_error_not_counted_as_block(
    provider: MeterProvider, reader: InMemoryMetricReader
) -> None:
    chain = GuardrailChain(pre=[_ExplodingGuardrail()])
    metered = MeteredGuardrailChain(chain, GuardrailMetrics(provider))
    with pytest.raises(RuntimeError, match="infra failure"):
        await metered.run_pre(_ctx())
    values = _counter_values(reader)
    # An infra failure is neither a pass nor a policy block.
    assert "atlas.guardrail.exploder.block" not in values
    assert "atlas.guardrail.exploder.pass" not in values


def test_no_raw_content_in_metric_attributes(
    provider: MeterProvider, reader: InMemoryMetricReader
) -> None:
    metrics = GuardrailMetrics(provider)
    metrics.record_pass("pii")
    metrics.record_block("pii")
    data = reader.get_metrics_data()
    assert data is not None
    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                for point in metric.data.data_points:
                    attrs: Mapping[str, object] = point.attributes or {}
                    assert attrs == {}  # only the check name is encoded, in the metric name
