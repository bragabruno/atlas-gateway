"""GRD-12 — no-raw-PII-in-logs/traces enforcement tests.

Pins the privacy invariant (atlas-docs/05 §6.4): a PII-bearing message routed
through the logging filter OR the OTel span processor leaves **no raw PII** in the
captured output — the raw value is absent and the redaction marker is present.
Both sinks reuse the GRD-2 regex + GRD-3 pattern detectors, so the categories
covered here track those modules. Fully offline (in-memory log capture +
in-memory span exporter; no network). See GRD-12 + GRD-2 + GRD-3.
"""

from __future__ import annotations

import logging

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.observability.log_redaction import (
    REDACTION_MARKER,
    PiiRedactingLogFilter,
    PiiRedactingSpanProcessor,
    redact_text,
)

# Realistic PII samples (matching the GRD-2 fast-path and GRD-3 NER patterns).
_EMAIL = "alice@example.com"
_SSN = "123-45-6789"
_CARD = "4111 1111 1111 1111"
_PHONE = "+1 415 555 0142"
_IP = "192.168.1.42"
_PERSON = "Dr. Jane Smith"
_ADDRESS = "742 Evergreen Terrace Road"
_IBAN = "DE89370400440532013000"

_ALL_PII = (_EMAIL, _SSN, _CARD, _PHONE, _IP, _PERSON, _ADDRESS, _IBAN)


# ── the shared primitive ────────────────────────────────────────────────────


@pytest.mark.parametrize("raw", _ALL_PII)
def test_redact_text_removes_each_pii_category(raw: str) -> None:
    out = redact_text(f"value is {raw} end")
    assert raw not in out
    assert REDACTION_MARKER in out


def test_redact_text_passes_clean_text_through_unchanged() -> None:
    clean = "the quarterly report is ready for review"
    assert redact_text(clean) == clean


# ── logging filter: no raw PII reaches the formatted line ────────────────────


@pytest.fixture
def captured() -> tuple[logging.Logger, list[logging.LogRecord]]:
    """A logger with the redacting filter installed and an in-memory sink."""
    logger = logging.getLogger("atlas.test.redaction")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Capture()
    handler.addFilter(PiiRedactingLogFilter())
    logger.addHandler(handler)
    return logger, records


def test_log_message_with_pii_is_redacted(
    captured: tuple[logging.Logger, list[logging.LogRecord]],
) -> None:
    logger, records = captured
    logger.info("user email was %s and ssn %s", _EMAIL, _SSN)
    (record,) = records
    line = record.getMessage()
    assert _EMAIL not in line
    assert _SSN not in line
    assert REDACTION_MARKER in line


def test_log_preformatted_msg_with_pii_is_redacted(
    captured: tuple[logging.Logger, list[logging.LogRecord]],
) -> None:
    logger, records = captured
    logger.info(f"contact {_PERSON} at {_ADDRESS}")
    (record,) = records
    line = record.getMessage()
    assert _PERSON not in line
    assert _ADDRESS not in line
    assert REDACTION_MARKER in line


def test_log_mapping_args_with_pii_are_redacted(
    captured: tuple[logging.Logger, list[logging.LogRecord]],
) -> None:
    logger, records = captured
    logger.info("card=%(card)s", {"card": _CARD})
    (record,) = records
    line = record.getMessage()
    assert _CARD not in line
    assert REDACTION_MARKER in line


def test_log_filter_always_allows_the_record_through() -> None:
    record = logging.LogRecord(
        name="x", level=logging.INFO, pathname="", lineno=0, msg=_EMAIL, args=(), exc_info=None
    )
    assert PiiRedactingLogFilter().filter(record) is True


def test_log_clean_message_is_unchanged(
    captured: tuple[logging.Logger, list[logging.LogRecord]],
) -> None:
    logger, records = captured
    logger.info("processed %d records", 12)
    (record,) = records
    assert record.getMessage() == "processed 12 records"


# ── span processor: no raw PII reaches the exporter ──────────────────────────


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter: InMemorySpanExporter) -> TracerProvider:
    """Redacting processor registered BEFORE the exporter (order = redact then export)."""
    tp = TracerProvider()
    tp.add_span_processor(PiiRedactingSpanProcessor())
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    return tp


def test_span_string_attribute_with_pii_is_redacted(
    provider: TracerProvider, exporter: InMemorySpanExporter
) -> None:
    tracer = provider.get_tracer("atlas.test")
    with tracer.start_as_current_span("op") as span:
        span.set_attribute("note", f"reached {_EMAIL} about case")
    (finished,) = exporter.get_finished_spans()
    attrs = finished.attributes or {}
    assert _EMAIL not in str(attrs["note"])
    assert REDACTION_MARKER in str(attrs["note"])


def test_span_sequence_attribute_with_pii_is_redacted(
    provider: TracerProvider, exporter: InMemorySpanExporter
) -> None:
    tracer = provider.get_tracer("atlas.test")
    with tracer.start_as_current_span("op") as span:
        span.set_attribute("contacts", [_EMAIL, "clean text", _PHONE])
    (finished,) = exporter.get_finished_spans()
    contacts = (finished.attributes or {})["contacts"]
    joined = " ".join(str(v) for v in contacts)  # type: ignore[union-attr]
    assert _EMAIL not in joined
    assert _PHONE not in joined
    assert REDACTION_MARKER in joined


def test_span_non_string_attributes_survive(
    provider: TracerProvider, exporter: InMemorySpanExporter
) -> None:
    tracer = provider.get_tracer("atlas.test")
    with tracer.start_as_current_span("op") as span:
        span.set_attribute("count", 7)
        span.set_attribute("ratio", 0.5)
    (finished,) = exporter.get_finished_spans()
    attrs = finished.attributes or {}
    assert attrs["count"] == 7
    assert attrs["ratio"] == 0.5


def test_span_without_attributes_is_handled(
    provider: TracerProvider, exporter: InMemorySpanExporter
) -> None:
    tracer = provider.get_tracer("atlas.test")
    with tracer.start_as_current_span("op"):
        pass
    (finished,) = exporter.get_finished_spans()
    assert not (finished.attributes or {})
