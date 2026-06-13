"""GRD-3 — PII NER guardrail (Presidio, off inline path).

Tests mock out the Presidio AnalyzerEngine so no model download or presidio
install is required locally. The mock injects fake analysis results to exercise
the guardrail logic. Skip the whole module if presidio is not installed.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

if importlib.util.find_spec("presidio_analyzer") is None:
    pytest.skip("presidio-analyzer not installed", allow_module_level=True)

from app.domain.messages import Message
from app.guardrails.chain import GuardrailContext, GuardrailRejection
from app.guardrails.pii_ner import PiiNerGuardrail

_MESSAGES = [Message(role="user", content="My name is John Smith")]


def _ctx(messages: list[Message] | None = None) -> GuardrailContext:
    return GuardrailContext(
        tenant_id="t1",
        model="mock",
        messages=messages or _MESSAGES,
    )


def _fake_result(entity_type: str) -> SimpleNamespace:
    return SimpleNamespace(entity_type=entity_type)


@pytest.mark.asyncio
async def test_pii_ner_passes_when_no_entities_detected() -> None:
    guardrail = PiiNerGuardrail()
    mock_analyzer = MagicMock()
    mock_analyzer.analyze.return_value = []
    guardrail._analyzer = mock_analyzer
    await guardrail.check(_ctx())  # should not raise


@pytest.mark.asyncio
async def test_pii_ner_rejects_when_person_entity_detected() -> None:
    guardrail = PiiNerGuardrail()
    mock_analyzer = MagicMock()
    mock_analyzer.analyze.return_value = [_fake_result("PERSON")]
    guardrail._analyzer = mock_analyzer
    with pytest.raises(GuardrailRejection) as exc_info:
        await guardrail.check(_ctx())
    assert "PERSON" in exc_info.value.reason
    assert exc_info.value.guardrail == "pii_ner"


@pytest.mark.asyncio
async def test_pii_ner_rejects_with_all_detected_types_in_reason() -> None:
    guardrail = PiiNerGuardrail()
    mock_analyzer = MagicMock()
    mock_analyzer.analyze.return_value = [
        _fake_result("PERSON"),
        _fake_result("LOCATION"),
    ]
    guardrail._analyzer = mock_analyzer
    with pytest.raises(GuardrailRejection) as exc_info:
        await guardrail.check(_ctx())
    assert "PERSON" in exc_info.value.reason
    assert "LOCATION" in exc_info.value.reason


@pytest.mark.asyncio
async def test_pii_ner_checks_all_messages() -> None:
    clean = Message(role="user", content="Tell me about regulations")
    pii = Message(role="assistant", content="Response from Jane Doe")
    guardrail = PiiNerGuardrail()
    call_count = 0

    def analyze_side_effect(text: str, **_: object) -> list:
        nonlocal call_count
        call_count += 1
        return [_fake_result("PERSON")] if "Jane" in text else []

    mock_analyzer = MagicMock()
    mock_analyzer.analyze.side_effect = analyze_side_effect
    guardrail._analyzer = mock_analyzer
    with pytest.raises(GuardrailRejection):
        await guardrail.check(_ctx(messages=[clean, pii]))
    assert call_count >= 1
