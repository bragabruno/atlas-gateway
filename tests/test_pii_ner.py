"""GRD-3 — PII NER (off inline path) tests.

Pins: the NER stand-in catches novel free-text PII formats the GRD-2 regex
fast-path misses (person names, street addresses, IBAN, IP); the analyzer is
non-mutating (off-inline-path posture, atlas-docs/05 §2.1) so the live context is
never rewritten; reject mode raises with category counts only and never leaks raw
PII; the detector seam is injectable so a pinned Presidio/GLiNER backend can drop
in later. Fully offline (dependency-free stand-in). See GRD-3 + GRD-2 + ADR-016.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.domain.messages import Message
from app.guardrails.chain import Guardrail, GuardrailContext, GuardrailRejection
from app.guardrails.pii import PiiGuardrail, PiiMode
from app.guardrails.pii_ner import (
    ExtendedPatternNer,
    NerFindings,
    PiiNerDetector,
    PiiNerGuardrail,
)


def _ctx(*contents: str) -> GuardrailContext:
    return GuardrailContext(
        tenant_id="tenant-a",
        model="mock",
        messages=[Message(role="user", content=c) for c in contents],
    )


# ── catches formats GRD-2 regex misses ──────────────────────────────────────


@pytest.mark.parametrize(
    ("content", "category"),
    [
        ("please contact Dr. Jane Smith about it", "PERSON"),
        ("ship it to 1600 Pennsylvania Avenue", "ADDRESS"),
        ("transfer to GB82WEST12345698765432 today", "IBAN"),
        ("the server is at 192.168.10.254 right now", "IP_ADDRESS"),
    ],
)
async def test_ner_catches_novel_formats(content: str, category: str) -> None:
    findings = ExtendedPatternNer().analyze([Message(role="user", content=content)])
    assert findings.detected
    assert category in findings.counts


async def test_grd2_regex_misses_what_ner_catches() -> None:
    # The discriminating case: a person name carries no email/phone/SSN/card, so
    # the GRD-2 regex fast-path leaves it untouched — proving NER adds recall.
    text = "please contact Dr. Jane Smith about it"
    ctx = _ctx(text)
    await PiiGuardrail().check(ctx)
    assert ctx.messages[0].content == text  # GRD-2 found nothing to redact

    findings = ExtendedPatternNer().analyze([Message(role="user", content=text)])
    assert findings.counts["PERSON"] == 1  # GRD-3 NER catches it


# ── off-inline-path design: analyzer is non-mutating ────────────────────────


async def test_analyze_does_not_mutate_context() -> None:
    ctx = _ctx("ship it to 1600 Pennsylvania Avenue")
    PiiNerGuardrail().analyze(ctx.messages)
    # Off-path posture: the live inline context is never rewritten.
    assert ctx.messages[0].content == "ship it to 1600 Pennsylvania Avenue"


async def test_redact_mode_check_is_non_mutating() -> None:
    ctx = _ctx("ship it to 1600 Pennsylvania Avenue")
    await PiiNerGuardrail(mode=PiiMode.REDACT).check(ctx)
    # REDACT mode passes without rewriting the inline context (off-path).
    assert ctx.messages[0].content == "ship it to 1600 Pennsylvania Avenue"


async def test_findings_carry_redacted_copies() -> None:
    findings = ExtendedPatternNer().analyze([Message(role="user", content="ping Dr. Jane Smith")])
    assert findings.redacted_messages[0].content == "ping [PII:PERSON]"


async def test_clean_input_yields_no_findings() -> None:
    findings = ExtendedPatternNer().analyze(
        [Message(role="user", content="just a normal harmless sentence")]
    )
    assert not findings.detected
    assert findings.counts == {}


# ── reject mode: counts only, never raw PII ─────────────────────────────────


async def test_reject_mode_raises_with_counts_only() -> None:
    ctx = _ctx("contact Dr. Jane Smith at 1600 Pennsylvania Avenue")
    with pytest.raises(GuardrailRejection) as exc_info:
        await PiiNerGuardrail(mode=PiiMode.REJECT).check(ctx)
    rejection = exc_info.value
    assert rejection.guardrail == "pii_ner"
    assert "PERSON=1" in rejection.reason
    assert "ADDRESS=1" in rejection.reason


async def test_reject_reason_never_leaks_raw_pii() -> None:
    ctx = _ctx("the account is GB82WEST12345698765432 confidential")
    with pytest.raises(GuardrailRejection) as exc_info:
        await PiiNerGuardrail(mode=PiiMode.REJECT).check(ctx)
    assert "GB82WEST12345698765432" not in str(exc_info.value)


async def test_clean_input_passes_in_reject_mode() -> None:
    await PiiNerGuardrail(mode=PiiMode.REJECT).check(_ctx("a perfectly clean message"))


async def test_default_mode_is_redact() -> None:
    assert PiiNerGuardrail().mode is PiiMode.REDACT


# ── injectable seam: a pinned Presidio/GLiNER backend can drop in later ──────


async def test_detector_seam_is_injectable() -> None:
    class _FakeNer:
        def analyze(self, messages: Sequence[Message]) -> NerFindings:
            return NerFindings(counts={"PERSON": 1}, redacted_messages=tuple(messages))

    guard = PiiNerGuardrail(mode=PiiMode.REJECT, detector=_FakeNer())
    with pytest.raises(GuardrailRejection) as exc_info:
        await guard.check(_ctx("anything"))
    assert "PERSON=1" in exc_info.value.reason


async def test_extended_pattern_ner_conforms_to_detector_protocol() -> None:
    assert isinstance(ExtendedPatternNer(), PiiNerDetector)


async def test_conforms_to_guardrail_protocol() -> None:
    assert isinstance(PiiNerGuardrail(), Guardrail)
