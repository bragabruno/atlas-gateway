"""GRD-11 — Per-check OTel guardrail metrics.

One OpenTelemetry counter pair per guardrail check —
``atlas.guardrail.<name>.pass`` and ``atlas.guardrail.<name>.block`` — so every
check in the chain is independently observable in Splunk (atlas-docs/05 §1.1,
§1.2: each check stanza maps 1-to-1 to a metric namespace). A check that passes
increments its `.pass` counter; a check that rejects (raises
`GuardrailRejection`) increments its `.block` counter.

Design (ADR-016, additive)
--------------------------
This is a thin capability adapter that *wraps* guardrail execution; it does not
modify the GRD-1 chain contracts. `GuardrailMetrics` lazily creates and caches
the counter pair per check name from an injected `MeterProvider` (defaults to the
global provider configured once at the composition root → OTel Collector →
Splunk, INF-13; tests pass a provider wired to an `InMemoryMetricReader` and
assert offline). `MeteredGuardrailChain` composes a `GuardrailChain` and records
the pass/block outcome of each check while preserving the chain's fail-fast
semantics — the first `GuardrailRejection` is recorded as a `.block` and then
re-raised, exactly as the bare chain would propagate it.

Following the PII/trace policy (atlas-docs/05 §6.4) the counters carry only the
check name in the metric *name*; no message text, raw PII, or rejection reason is
attached as a metric attribute. See GRD-11 + INF-13 + ADR-016.
"""

from __future__ import annotations

from opentelemetry import metrics as otel_metrics
from opentelemetry.metrics import Counter, Meter, MeterProvider

from app.guardrails.chain import (
    Guardrail,
    GuardrailChain,
    GuardrailContext,
    GuardrailRejection,
)

#: Instrumentation scope name for the meter this module acquires — identifies
#: gateway-emitted guardrail metrics in Splunk.
INSTRUMENTATION_NAME = "atlas.gateway.guardrails"

#: Counter-name template per outcome. Splits to ``atlas.guardrail.<name>.pass``
#: and ``atlas.guardrail.<name>.block`` (atlas-docs/05 §1.2).
_PASS_SUFFIX = "pass"
_BLOCK_SUFFIX = "block"
_NAME_PREFIX = "atlas.guardrail"


def _counter_name(check: str, outcome: str) -> str:
    """Render the counter name for one check + outcome (no raw content)."""
    return f"{_NAME_PREFIX}.{check}.{outcome}"


def get_meter(meter_provider: MeterProvider | None = None) -> Meter:
    """Return the guardrail meter from the given provider (global if omitted).

    The composition root configures the global provider once (SDK → Collector →
    Splunk, INF-13); tests pass a provider backed by an `InMemoryMetricReader`.
    """
    if meter_provider is None:
        return otel_metrics.get_meter(INSTRUMENTATION_NAME)
    return meter_provider.get_meter(INSTRUMENTATION_NAME)


class GuardrailMetrics:
    """Lazily creates and caches the per-check pass/block counter pair.

    Counters are created on first use for a given check name and cached, so a
    repeated check reuses the same instrument (the SDK also dedupes by name).
    Only the check name appears in the metric name; no content is recorded.
    """

    def __init__(self, meter_provider: MeterProvider | None = None) -> None:
        self._meter = get_meter(meter_provider)
        self._pass: dict[str, Counter] = {}
        self._block: dict[str, Counter] = {}

    def record_pass(self, check: str) -> None:
        """Increment ``atlas.guardrail.<check>.pass``."""
        self._pass_counter(check).add(1)

    def record_block(self, check: str) -> None:
        """Increment ``atlas.guardrail.<check>.block``."""
        self._block_counter(check).add(1)

    def _pass_counter(self, check: str) -> Counter:
        counter = self._pass.get(check)
        if counter is None:
            counter = self._meter.create_counter(_counter_name(check, _PASS_SUFFIX))
            self._pass[check] = counter
        return counter

    def _block_counter(self, check: str) -> Counter:
        counter = self._block.get(check)
        if counter is None:
            counter = self._meter.create_counter(_counter_name(check, _BLOCK_SUFFIX))
            self._block[check] = counter
        return counter


class MeteredGuardrailChain:
    """Runs a `GuardrailChain`, recording each check's pass/block outcome.

    Composes (does not subclass) a `GuardrailChain`: `run_pre`/`run_post` execute
    the same ordered, fail-fast pipeline, but each check is wrapped so a clean
    return records a `.pass` and a `GuardrailRejection` records a `.block` before
    being re-raised. Non-`GuardrailRejection` errors are *not* recorded as a
    block (they are infrastructure failures, not policy decisions) and propagate
    unchanged.
    """

    def __init__(self, chain: GuardrailChain, metrics: GuardrailMetrics) -> None:
        self._chain = chain
        self._metrics: GuardrailMetrics = metrics

    async def run_pre(self, ctx: GuardrailContext) -> None:
        """Run pre-phase checks, recording pass/block per check."""
        await self._run(self._chain.pre, ctx)

    async def run_post(self, ctx: GuardrailContext) -> None:
        """Run post-phase checks, recording pass/block per check."""
        await self._run(self._chain.post, ctx)

    async def _run(self, guardrails: tuple[Guardrail, ...], ctx: GuardrailContext) -> None:
        for guardrail in guardrails:
            try:
                await guardrail.check(ctx)
            except GuardrailRejection:
                self._metrics.record_block(guardrail.name)
                raise
            self._metrics.record_pass(guardrail.name)
