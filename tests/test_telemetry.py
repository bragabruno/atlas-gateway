"""GW-18 — OTel GenAI span helpers, exercised offline with an in-memory exporter.

A `TracerProvider` wired to an `InMemorySpanExporter` (`SimpleSpanProcessor`)
captures spans without a collector or network, proving a chat call produces one
span carrying the GenAI semconv attributes (atlas-docs/04 §6.2): system, request
model, parameters, response model, finish reasons, and the four `gen_ai.usage.*`
token attributes mapped from `Usage`. Error handling (span marked ERROR, exception
recorded, re-raised) and the PII policy (no message text on the span) are pinned
too. Span "lands in Splunk" pends the SDK→Collector export wiring (INF-13).
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from app.domain.messages import Usage
from app.telemetry.otel import (
    GenAIRequest,
    GenAIResponse,
    chat_span,
    get_tracer,
    set_response_attributes,
)

_USAGE = Usage(
    input_tokens=512,
    output_tokens=128,
    cache_creation_input_tokens=64,
    cache_read_input_tokens=32,
)


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter: InMemorySpanExporter) -> TracerProvider:
    """A TracerProvider that records spans into the in-memory exporter."""
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    return tp


# ── a chat call produces one GenAI span with gen_ai.* attributes ────────────


def test_chat_span_produces_single_span(
    provider: TracerProvider, exporter: InMemorySpanExporter
) -> None:
    tracer = get_tracer(provider)
    with chat_span(
        tracer,
        GenAIRequest(system="anthropic", model="claude-sonnet-4-6"),
        response=GenAIResponse(usage=_USAGE),
    ):
        pass
    spans = exporter.get_finished_spans()
    assert len(spans) == 1


def test_span_named_chat_and_client_kind(
    provider: TracerProvider, exporter: InMemorySpanExporter
) -> None:
    tracer = get_tracer(provider)
    with chat_span(
        tracer,
        GenAIRequest(system="anthropic", model="claude-sonnet-4-6"),
        response=GenAIResponse(usage=_USAGE),
    ):
        pass
    (span,) = exporter.get_finished_spans()
    assert span.name == "chat claude-sonnet-4-6"
    assert span.kind is SpanKind.CLIENT


def test_request_attributes_set(provider: TracerProvider, exporter: InMemorySpanExporter) -> None:
    tracer = get_tracer(provider)
    with chat_span(
        tracer,
        GenAIRequest(system="openai", model="gpt-4.1", max_tokens=4096, temperature=0.7),
        response=GenAIResponse(usage=_USAGE),
    ):
        pass
    (span,) = exporter.get_finished_spans()
    attrs = span.attributes or {}
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.system"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-4.1"
    assert attrs["gen_ai.request.max_tokens"] == 4096
    assert attrs["gen_ai.request.temperature"] == 0.7


def test_optional_request_params_omitted_when_none(
    provider: TracerProvider, exporter: InMemorySpanExporter
) -> None:
    tracer = get_tracer(provider)
    with chat_span(
        tracer,
        GenAIRequest(system="google", model="gemini-2.5"),
        response=GenAIResponse(usage=Usage()),
    ):
        pass
    (span,) = exporter.get_finished_spans()
    attrs = span.attributes or {}
    assert "gen_ai.request.max_tokens" not in attrs
    assert "gen_ai.request.temperature" not in attrs


def test_response_usage_attributes_mapped_from_usage(
    provider: TracerProvider, exporter: InMemorySpanExporter
) -> None:
    """The four Usage token fields map onto the distinct gen_ai.usage.* attrs."""
    tracer = get_tracer(provider)
    with chat_span(
        tracer,
        GenAIRequest(system="anthropic", model="claude-sonnet-4-6"),
        response=GenAIResponse(
            usage=_USAGE,
            model="claude-sonnet-4-6-20260101",
            finish_reasons=["stop"],
        ),
    ):
        pass
    (span,) = exporter.get_finished_spans()
    attrs = span.attributes or {}
    assert attrs["gen_ai.response.model"] == "claude-sonnet-4-6-20260101"
    assert attrs["gen_ai.response.finish_reasons"] == ("stop",)
    assert attrs["gen_ai.usage.input_tokens"] == 512
    assert attrs["gen_ai.usage.output_tokens"] == 128
    assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 64
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 32


def test_span_ok_on_success(provider: TracerProvider, exporter: InMemorySpanExporter) -> None:
    tracer = get_tracer(provider)
    with chat_span(
        tracer,
        GenAIRequest(system="anthropic", model="claude-sonnet-4-6"),
        response=GenAIResponse(usage=_USAGE),
    ):
        pass
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.OK


# ── streaming path: usage known only at the terminal delta ──────────────────


def test_set_response_attributes_after_span_start(
    provider: TracerProvider, exporter: InMemorySpanExporter
) -> None:
    """Streaming sets usage on the live span once the terminal delta arrives."""
    tracer = get_tracer(provider)
    with chat_span(tracer, GenAIRequest(system="anthropic", model="claude-sonnet-4-6")) as span:
        # ... stream content deltas ... then the terminal usage lands:
        set_response_attributes(span, GenAIResponse(usage=_USAGE, finish_reasons=["stop"]))
    (finished,) = exporter.get_finished_spans()
    attrs = finished.attributes or {}
    assert attrs["gen_ai.usage.output_tokens"] == 128
    assert attrs["gen_ai.response.finish_reasons"] == ("stop",)


# ── error path: span ERROR + exception recorded, then re-raised ─────────────


def test_exception_marks_span_error_and_reraises(
    provider: TracerProvider, exporter: InMemorySpanExporter
) -> None:
    tracer = get_tracer(provider)
    with pytest.raises(RuntimeError, match="provider exploded"):
        with chat_span(tracer, GenAIRequest(system="openai", model="gpt-4.1")):
            raise RuntimeError("provider exploded")
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.events  # exception recorded as a span event
    assert any(e.name == "exception" for e in span.events)


def test_no_message_text_on_span(provider: TracerProvider, exporter: InMemorySpanExporter) -> None:
    """PII policy (atlas-docs/04 §6.2): no prompt/completion text on the span."""
    tracer = get_tracer(provider)
    with chat_span(
        tracer,
        GenAIRequest(system="anthropic", model="claude-sonnet-4-6"),
        response=GenAIResponse(usage=_USAGE),
    ):
        pass
    (span,) = exporter.get_finished_spans()
    attrs = span.attributes or {}
    for key in attrs:
        assert "prompt" not in key
        assert "completion" not in key
        assert "messages" not in key


def test_get_tracer_defaults_to_global_provider() -> None:
    """Omitting the provider returns a tracer from the global provider."""
    tracer = get_tracer()
    assert tracer is not None
