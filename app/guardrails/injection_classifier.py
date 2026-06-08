"""GRD-5 — Cheap-model prompt-injection classifier (PRE).

A `Guardrail` (conforms to `app.guardrails.chain.Guardrail`) that adjudicates the
*ambiguous* inputs the GRD-4 heuristic (`app.guardrails.injection`) cannot decide
cheaply. The GRD-4 screen favours recall over precision and is fail-fast on a
match; this check covers the grey zone — inputs that *look* uncertain but did not
trip a hard heuristic — by asking a cheap classifier model whether the input is a
prompt-injection attempt, and gating the request on its verdict.

Why the classifier is called **through the gateway** (atlas-docs/05 §2.2)
------------------------------------------------------------------------
All model calls — including this safety classifier — route through the gateway so
they inherit rate-limiting, circuit-breaking, cost attribution, and audit
logging. Calling a provider SDK directly here would open an unobserved, unmetered
side-channel that bypasses budget enforcement and breaks the single-pane audit
trail. So this module depends on an **injected** `ClassifierGatewayClient` port
(not an SDK): production wires it to a client that issues a normal gateway chat
call against the cheap classifier alias (`GW-10` resolves the alias → model), so
the classifier call is accounted and traced like any other request. Offline tests
inject a fake/Mock client and never touch the network.

Seams (all injected, ADR-016)
-----------------------------
- `AmbiguityDetector` — decides whether an input is ambiguous enough to warrant
  the classifier. The default delegates to GRD-4: an input the heuristic already
  flags is *not* ambiguous (GRD-4 handles it); everything GRD-4 passes is treated
  as ambiguous and forwarded to the classifier. A route may inject a narrower
  detector so only genuinely uncertain inputs incur the model call.
- `ClassifierGatewayClient` — the gateway-routed call returning a `ClassifierVerdict`.
- `model` / `tenant_id` defaults — the alias the verdict call resolves through and
  the attribution identity; both overridable per construction.

Fail-fast: a verdict of `INJECTION` raises an explicit `GuardrailRejection`
naming the stage; `BENIGN` passes. The reason never echoes the inspected content.
See GRD-5 + GRD-4 + GW-10 + ADR-016 + atlas-docs/05 §2.2.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Protocol, runtime_checkable

from app.domain.messages import Message
from app.guardrails.chain import GuardrailContext, GuardrailPhase, GuardrailRejection
from app.guardrails.injection import InjectionGuardrail

#: Cheap classifier alias the verdict call resolves through (GW-10 maps it to a
#: concrete model). A route may inject a different alias; this is the default.
DEFAULT_CLASSIFIER_MODEL = "fast"

#: Identity the gateway-routed classifier call is attributed to when the caller
#: does not supply one. Keeps safety-classifier spend attributable rather than
#: anonymous (atlas-docs/05 §2.2).
DEFAULT_CLASSIFIER_TENANT = "guardrails"


class ClassifierVerdict(str, Enum):
    """The cheap classifier's binary judgement of an input."""

    INJECTION = "injection"
    BENIGN = "benign"


@runtime_checkable
class AmbiguityDetector(Protocol):
    """Decides whether an input is ambiguous enough to warrant the classifier.

    Returns `True` when the input should be escalated to the cheap model, `False`
    when it can be left to the cheaper heuristic path. Implementations must not
    raise on benign input — ambiguity is a routing signal, not a verdict.
    """

    def is_ambiguous(self, messages: Sequence[Message]) -> bool:
        """Return `True` if `messages` should be escalated to the classifier."""
        ...


@runtime_checkable
class ClassifierGatewayClient(Protocol):
    """Port for the gateway-routed cheap-classifier call.

    Implementations issue a normal gateway chat call (so it is rate-limited,
    circuit-broken, cost-attributed, and traced) against `model` for `tenant_id`
    and return a `ClassifierVerdict`. Tests inject a fake/Mock; production injects
    a client backed by the real gateway chat path. The classifier prompt and the
    inspected content never leave this boundary as a span/log attribute.
    """

    async def classify(
        self,
        *,
        model: str,
        tenant_id: str,
        messages: Sequence[Message],
    ) -> ClassifierVerdict:
        """Return the classifier's verdict for `messages` via a gateway call."""
        ...


class _HeuristicAmbiguityDetector:
    """Default detector: everything GRD-4 *passes* is treated as ambiguous.

    GRD-4 is fail-fast on a hard match, so an input it flags is already handled
    upstream and is *not* ambiguous here. Everything the heuristic lets through
    is the grey zone this classifier adjudicates. A route wanting a tighter
    escalation policy injects its own `AmbiguityDetector`.
    """

    def __init__(self, heuristic: InjectionGuardrail | None = None) -> None:
        self._heuristic = heuristic if heuristic is not None else InjectionGuardrail()

    def is_ambiguous(self, messages: Sequence[Message]) -> bool:
        """Ambiguous iff the GRD-4 heuristic does *not* already flag the input."""
        return self._heuristic.matched_label(messages) is None


class InjectionClassifierGuardrail:
    """Adjudicates ambiguous inputs via a gateway-routed cheap classifier.

    Conforms to the `Guardrail` protocol. For inputs the injected
    `AmbiguityDetector` deems ambiguous, it calls the injected
    `ClassifierGatewayClient` (routed through the gateway → accounted + traced)
    and raises a `GuardrailRejection` on an `INJECTION` verdict; non-ambiguous
    inputs and `BENIGN` verdicts pass. The rejection reason names the stage, never
    the inspected content.
    """

    name = "injection_classifier"

    def __init__(
        self,
        *,
        client: ClassifierGatewayClient,
        detector: AmbiguityDetector | None = None,
        model: str = DEFAULT_CLASSIFIER_MODEL,
        tenant_id: str = DEFAULT_CLASSIFIER_TENANT,
    ) -> None:
        self._client = client
        self._detector: AmbiguityDetector = (
            detector if detector is not None else _HeuristicAmbiguityDetector()
        )
        self._model = model
        self._tenant_id = tenant_id

    @property
    def model(self) -> str:
        """The classifier alias the gateway-routed verdict call resolves through."""
        return self._model

    async def check(self, ctx: GuardrailContext) -> None:
        """Classify ambiguous input via the gateway; reject on an injection verdict."""
        if not self._detector.is_ambiguous(ctx.messages):
            return
        verdict = await self._client.classify(
            model=self._model,
            tenant_id=self._tenant_id,
            messages=ctx.messages,
        )
        if verdict is ClassifierVerdict.BENIGN:
            return
        raise GuardrailRejection(
            guardrail=self.name,
            phase=GuardrailPhase.PRE,
            reason=f"cheap-model classifier verdict: {verdict.value}",
        )
