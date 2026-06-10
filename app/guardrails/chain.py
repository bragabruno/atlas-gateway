"""GRD-1 — Guardrail chain framework.

Defines the contracts and runner for pre/post guardrails:

- `GuardrailContext` — the request/response payload a check inspects.
- `Guardrail` — the port (Protocol) every check implements: a `name` plus a
  `check` over a `GuardrailContext`.
- `GuardrailChain` — an ordered runner that executes pre checks before the
  provider call and post checks after, in configured order.
- `GuardrailRejection` — the explicit, documented failure raised when a check
  rejects; the chain is fail-fast, so the first rejection stops the chain.

GRD-11: `GuardrailChain` records one OTel counter increment per check
(`atlas.guardrail.checks`) with attributes `guardrail.name` and
`guardrail.outcome` (pass | block). The `MeterProvider` is injected (defaults
to the global provider) so tests can pass an `InMemoryMetricReader` and assert
on the recorded data points entirely offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Protocol, runtime_checkable

from opentelemetry import metrics
from opentelemetry.metrics import Counter, MeterProvider

from app.domain.messages import ChatResult, Message

#: OTel instrumentation scope name for guardrail metrics.
_SCOPE = "atlas.gateway.guardrails"

#: Single counter for all guardrail outcomes; use attributes to fan out.
_METRIC_CHECKS = "atlas.guardrail.checks"


class GuardrailPhase(str, Enum):
    """Which side of the provider call a check runs on."""

    PRE = "pre"
    POST = "post"


class GuardrailRejection(Exception):
    """Raised when a guardrail rejects a request or response.

    Carries the rejecting guardrail's `name`, the `phase` it ran in, and an
    explicit human-readable `reason`. The API layer maps this to an HTTP
    response (the wiring ticket decides the status code); business logic stays
    framework-neutral. See ADR-016.
    """

    def __init__(self, *, guardrail: str, phase: GuardrailPhase, reason: str) -> None:
        self.guardrail = guardrail
        self.phase = phase
        self.reason = reason
        super().__init__(f"guardrail '{guardrail}' rejected during {phase.value}: {reason}")


class GuardrailContext:
    """The mutable payload a guardrail inspects.

    `messages` is the inbound request (always present). `result` is the provider
    response and is `None` during the pre phase, populated for the post phase.
    `tenant_id` and `model` identify the caller and target so checks can apply
    per-tenant or per-model policy.
    """

    def __init__(
        self,
        *,
        tenant_id: str,
        model: str,
        messages: Sequence[Message],
        result: ChatResult | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.model = model
        self.messages = messages
        self.result = result


@runtime_checkable
class Guardrail(Protocol):
    """A single safety check over a `GuardrailContext`.

    Implementations are fail-fast: `check` returns `None` to pass and raises
    `GuardrailRejection` to reject. A check must never swallow a policy
    violation silently — if it cannot decide, it raises.
    """

    name: str

    async def check(self, ctx: GuardrailContext) -> None:
        """Inspect `ctx`; return on pass, raise `GuardrailRejection` on reject."""
        ...


class GuardrailChain:
    """Runs an ordered set of guardrails for one route.

    Construction order is execution order — per-route config decides which
    guardrails are included and in what sequence. `run_pre` executes the chain
    before the provider call (over the request); `run_post` executes it after
    (over the request + result). The chain is fail-fast: the first
    `GuardrailRejection` propagates and stops the remaining checks.

    GRD-11: every check increments `atlas.guardrail.checks` with
    `guardrail.name=<name>` and `guardrail.outcome=pass|block`.
    """

    def __init__(
        self,
        *,
        pre: Sequence[Guardrail] = (),
        post: Sequence[Guardrail] = (),
        meter_provider: MeterProvider | None = None,
    ) -> None:
        self._pre: tuple[Guardrail, ...] = tuple(pre)
        self._post: tuple[Guardrail, ...] = tuple(post)
        mp = meter_provider or metrics.get_meter_provider()
        meter = mp.get_meter(_SCOPE)
        self._counter: Counter = meter.create_counter(
            _METRIC_CHECKS,
            unit="1",
            description="Number of guardrail check executions by name and outcome.",
        )

    @property
    def pre(self) -> tuple[Guardrail, ...]:
        """The ordered pre-phase guardrails."""
        return self._pre

    @property
    def post(self) -> tuple[Guardrail, ...]:
        """The ordered post-phase guardrails."""
        return self._post

    async def run_pre(self, ctx: GuardrailContext) -> None:
        """Run the pre-phase checks in order over the inbound request."""
        await self._run(self._pre, GuardrailPhase.PRE, ctx, self._counter)

    async def run_post(self, ctx: GuardrailContext) -> None:
        """Run the post-phase checks in order over the provider result."""
        await self._run(self._post, GuardrailPhase.POST, ctx, self._counter)

    @staticmethod
    async def _run(
        guardrails: Sequence[Guardrail],
        phase: GuardrailPhase,
        ctx: GuardrailContext,
        counter: Counter,
    ) -> None:
        for guardrail in guardrails:
            try:
                await guardrail.check(ctx)
            except GuardrailRejection:
                counter.add(1, {"guardrail.name": guardrail.name, "guardrail.outcome": "block"})
                raise
            counter.add(1, {"guardrail.name": guardrail.name, "guardrail.outcome": "pass"})
