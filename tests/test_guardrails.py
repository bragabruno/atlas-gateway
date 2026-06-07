"""GRD-1 — guardrail chain framework.

Pins the chain contract: checks run in construction (config) order, a rejecting
check raises `GuardrailRejection` carrying the reason and stops the chain, and
pre/post phases run their own configured ordering. Fully offline.
"""

from __future__ import annotations

import pytest

from app.domain.messages import ChatResult, Message, Usage
from app.guardrails.chain import (
    GuardrailChain,
    GuardrailContext,
    GuardrailPhase,
    GuardrailRejection,
)

_MESSAGES = [Message(role="user", content="hello")]
_RESULT = ChatResult(model="mock", content="hi", usage=Usage())


def _ctx(result: ChatResult | None = None) -> GuardrailContext:
    return GuardrailContext(
        tenant_id="tenant-a",
        model="mock",
        messages=_MESSAGES,
        result=result,
    )


class RecordingGuardrail:
    """Passing guardrail that appends its name to a shared log when it runs."""

    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self._log = log

    async def check(self, ctx: GuardrailContext) -> None:
        self._log.append(self.name)


class RejectingGuardrail:
    """Guardrail that always rejects with a fixed reason."""

    def __init__(self, name: str, reason: str, phase: GuardrailPhase) -> None:
        self.name = name
        self._reason = reason
        self._phase = phase

    async def check(self, ctx: GuardrailContext) -> None:
        raise GuardrailRejection(guardrail=self.name, phase=self._phase, reason=self._reason)


async def test_pre_checks_run_in_configured_order() -> None:
    log: list[str] = []
    chain = GuardrailChain(
        pre=[
            RecordingGuardrail("first", log),
            RecordingGuardrail("second", log),
            RecordingGuardrail("third", log),
        ]
    )
    await chain.run_pre(_ctx())
    assert log == ["first", "second", "third"]


async def test_post_checks_run_in_configured_order() -> None:
    log: list[str] = []
    chain = GuardrailChain(
        post=[
            RecordingGuardrail("alpha", log),
            RecordingGuardrail("beta", log),
        ]
    )
    await chain.run_post(_ctx(result=_RESULT))
    assert log == ["alpha", "beta"]


async def test_reordering_config_changes_execution_order() -> None:
    log: list[str] = []
    chain = GuardrailChain(
        pre=[
            RecordingGuardrail("second", log),
            RecordingGuardrail("first", log),
        ]
    )
    await chain.run_pre(_ctx())
    assert log == ["second", "first"]


async def test_rejecting_check_raises_with_reason() -> None:
    chain = GuardrailChain(
        pre=[RejectingGuardrail("blocklist", "banned term detected", GuardrailPhase.PRE)]
    )
    with pytest.raises(GuardrailRejection) as exc_info:
        await chain.run_pre(_ctx())
    rejection = exc_info.value
    assert rejection.guardrail == "blocklist"
    assert rejection.reason == "banned term detected"
    assert rejection.phase is GuardrailPhase.PRE
    assert "banned term detected" in str(rejection)


async def test_chain_is_fail_fast_and_stops_after_rejection() -> None:
    log: list[str] = []
    chain = GuardrailChain(
        pre=[
            RecordingGuardrail("first", log),
            RejectingGuardrail("gate", "denied", GuardrailPhase.PRE),
            RecordingGuardrail("never", log),
        ]
    )
    with pytest.raises(GuardrailRejection):
        await chain.run_pre(_ctx())
    assert log == ["first"]


async def test_post_rejection_inspects_result() -> None:
    class ResultGuardrail:
        name = "no-empty-output"

        async def check(self, ctx: GuardrailContext) -> None:
            if ctx.result is not None and not ctx.result.content:
                raise GuardrailRejection(
                    guardrail=self.name,
                    phase=GuardrailPhase.POST,
                    reason="empty completion",
                )

    chain = GuardrailChain(post=[ResultGuardrail()])
    empty = ChatResult(model="mock", content="", usage=Usage())
    with pytest.raises(GuardrailRejection) as exc_info:
        await chain.run_post(_ctx(result=empty))
    assert exc_info.value.reason == "empty completion"


async def test_empty_chain_is_a_noop() -> None:
    chain = GuardrailChain()
    await chain.run_pre(_ctx())
    await chain.run_post(_ctx(result=_RESULT))


async def test_guardrails_expose_configured_order_via_properties() -> None:
    log: list[str] = []
    pre = [RecordingGuardrail("p1", log), RecordingGuardrail("p2", log)]
    post = [RecordingGuardrail("q1", log)]
    chain = GuardrailChain(pre=pre, post=post)
    assert [g.name for g in chain.pre] == ["p1", "p2"]
    assert [g.name for g in chain.post] == ["q1"]
