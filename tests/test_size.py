"""GRD-7 — Input size-cap guardrail tests.

Pins: inputs at or below the configured cap pass; oversized inputs reject
fail-fast with an explicit reason reporting the limit and observed size; the
cap counts total characters across all messages; and a non-positive cap is a
loud construction-time error. Fully offline. See ADR-016.
"""

from __future__ import annotations

import pytest

from app.domain.messages import Message
from app.guardrails.chain import Guardrail, GuardrailContext, GuardrailRejection
from app.guardrails.size import SizeGuardrail


def _ctx(*contents: str) -> GuardrailContext:
    return GuardrailContext(
        tenant_id="tenant-a",
        model="mock",
        messages=[Message(role="user", content=c) for c in contents],
    )


async def test_input_under_cap_passes() -> None:
    await SizeGuardrail(max_chars=10).check(_ctx("hello"))


async def test_input_at_cap_passes() -> None:
    await SizeGuardrail(max_chars=5).check(_ctx("hello"))


async def test_input_over_cap_rejects() -> None:
    with pytest.raises(GuardrailRejection) as exc_info:
        await SizeGuardrail(max_chars=4).check(_ctx("hello"))
    rejection = exc_info.value
    assert rejection.guardrail == "size"
    assert "5 chars" in rejection.reason
    assert "cap of 4" in rejection.reason


async def test_cap_counts_total_across_messages() -> None:
    # "abc" + "def" = 6 chars total, cap is 5 -> reject.
    with pytest.raises(GuardrailRejection) as exc_info:
        await SizeGuardrail(max_chars=5).check(_ctx("abc", "def"))
    assert "6 chars" in exc_info.value.reason


async def test_empty_input_passes() -> None:
    await SizeGuardrail(max_chars=1).check(_ctx())


@pytest.mark.parametrize("bad_cap", [0, -1, -100])
def test_non_positive_cap_fails_loudly(bad_cap: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        SizeGuardrail(max_chars=bad_cap)


async def test_max_chars_property_exposes_cap() -> None:
    assert SizeGuardrail(max_chars=42).max_chars == 42


async def test_conforms_to_guardrail_protocol() -> None:
    assert isinstance(SizeGuardrail(max_chars=10), Guardrail)
