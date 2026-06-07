"""GRD-2 — PII fast-path guardrail tests.

Pins: common PII categories are detected; matches are redacted in place with
placeholders; the redacted messages are written back onto the context; reject
mode raises with category counts only; and no raw PII ever appears in the
rejection reason. Fully offline. See ADR-016.
"""

from __future__ import annotations

import pytest

from app.domain.messages import Message
from app.guardrails.chain import GuardrailContext, GuardrailRejection
from app.guardrails.pii import PiiGuardrail, PiiMode


def _ctx(*contents: str) -> GuardrailContext:
    return GuardrailContext(
        tenant_id="tenant-a",
        model="mock",
        messages=[Message(role="user", content=c) for c in contents],
    )


async def test_redacts_email() -> None:
    ctx = _ctx("contact me at alice@example.com please")
    await PiiGuardrail().check(ctx)
    assert ctx.messages[0].content == "contact me at [REDACTED_EMAIL] please"


async def test_redacts_ssn_like() -> None:
    ctx = _ctx("my ssn is 123-45-6789")
    await PiiGuardrail().check(ctx)
    assert ctx.messages[0].content == "my ssn is [REDACTED_SSN]"


async def test_redacts_credit_card_like() -> None:
    ctx = _ctx("card 4111 1111 1111 1111 expires soon")
    await PiiGuardrail().check(ctx)
    assert ctx.messages[0].content == "card [REDACTED_CREDIT_CARD] expires soon"


async def test_redacts_phone_like() -> None:
    ctx = _ctx("call +1 (415) 555-2671 tomorrow")
    await PiiGuardrail().check(ctx)
    assert ctx.messages[0].content == "call [REDACTED_PHONE] tomorrow"


async def test_clean_input_passes_unchanged() -> None:
    ctx = _ctx("just a normal harmless sentence")
    await PiiGuardrail().check(ctx)
    assert ctx.messages[0].content == "just a normal harmless sentence"


async def test_redacts_across_multiple_messages() -> None:
    ctx = _ctx("email a@b.co", "and a@b.co again")
    await PiiGuardrail().check(ctx)
    assert ctx.messages[0].content == "email [REDACTED_EMAIL]"
    assert ctx.messages[1].content == "and [REDACTED_EMAIL] again"


async def test_reject_mode_raises_with_counts_only() -> None:
    ctx = _ctx("alice@example.com and bob@example.com and 123-45-6789")
    with pytest.raises(GuardrailRejection) as exc_info:
        await PiiGuardrail(mode=PiiMode.REJECT).check(ctx)
    rejection = exc_info.value
    assert rejection.guardrail == "pii"
    assert "EMAIL=2" in rejection.reason
    assert "SSN=1" in rejection.reason


async def test_rejection_reason_never_leaks_raw_pii() -> None:
    secret_email = "topsecret@private.example"
    ctx = _ctx(f"reach me at {secret_email}")
    with pytest.raises(GuardrailRejection) as exc_info:
        await PiiGuardrail(mode=PiiMode.REJECT).check(ctx)
    assert secret_email not in str(exc_info.value)
    assert "topsecret" not in str(exc_info.value)


async def test_reject_mode_does_not_mutate_messages() -> None:
    ctx = _ctx("alice@example.com")
    with pytest.raises(GuardrailRejection):
        await PiiGuardrail(mode=PiiMode.REJECT).check(ctx)
    # Reject mode must not have rewritten the context before raising.
    assert ctx.messages[0].content == "alice@example.com"


async def test_default_mode_is_redact() -> None:
    assert PiiGuardrail().mode is PiiMode.REDACT


async def test_conforms_to_guardrail_protocol() -> None:
    from app.guardrails.chain import Guardrail

    assert isinstance(PiiGuardrail(), Guardrail)
