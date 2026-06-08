"""GW-18 — OpenTelemetry GenAI span helpers for the chat request path.

A thin capability adapter (ADR-016): it opens one span per chat call and sets
the OpenTelemetry GenAI semantic-convention attributes (`gen_ai.*`) the docs
mandate (atlas-docs/04 §6.2), so per-model latency, token usage, and error
classification surface in Splunk without bespoke attribute names (ADR-009). The
attribute *keys* are bound to the semconv constants (re-exported, typed, via
`app.telemetry._genai_semconv`), not hand-typed strings, so they track the spec.

The tracer is injected. `tracer_provider` defaults to the global provider
(configured once at the composition root → OTel Collector → Splunk, INF-13),
but tests pass a provider wired to an `InMemorySpanExporter` and assert on the
recorded spans entirely offline — no collector, no network.

`GenAIRequest`/`GenAIResponse` map the gateway's own request/result shape onto
the semconv attributes, including the four `Usage` token fields
(input/output + the two cache classes, atlas-docs/03 §1.4) which OTel models as
distinct `gen_ai.usage.*` attributes. Per the PII policy (atlas-docs/04 §6.2)
**no raw prompt or completion text is placed on a span** — only model ids,
parameters, token counts, and finish reasons. See GW-18 + ADR-009.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer, TracerProvider

from app.domain.messages import Usage
from app.telemetry import _genai_semconv as genai

#: OTel GenAI operation name for a chat completion (`gen_ai.operation.name`),
#: per the semconv spec; names the span (`{operation} {model}`) too.
_OPERATION_CHAT = "chat"

#: Instrumentation scope name for the tracer this module acquires — identifies
#: gateway-emitted GenAI spans in Splunk.
INSTRUMENTATION_NAME = "atlas.gateway.genai"


@dataclass(frozen=True, slots=True)
class GenAIRequest:
    """The request-side facts that become `gen_ai.request.*` span attributes.

    Carries no message text (PII policy, atlas-docs/04 §6.2) — only the system
    (provider) name, requested model id, and sampling parameters. `system` is
    the `gen_ai.system` / provider name (e.g. ``"anthropic"``); `model` is the
    requested model id before any provider-side resolution.
    """

    system: str
    model: str
    max_tokens: int | None = None
    temperature: float | None = None


@dataclass(frozen=True, slots=True)
class GenAIResponse:
    """The response-side facts that become `gen_ai.response.*`/`usage.*` attrs.

    `model` is the model the provider actually served (may differ from the
    requested id); `finish_reasons` maps to `gen_ai.response.finish_reasons`;
    `usage` supplies the four token counts. No completion text is carried.
    """

    usage: Usage
    model: str | None = None
    finish_reasons: Sequence[str] | None = None


def get_tracer(tracer_provider: TracerProvider | None = None) -> Tracer:
    """Return the GenAI tracer from the given provider (global if omitted).

    The composition root configures the global provider once (SDK → Collector →
    Splunk, INF-13); tests pass a provider backed by an in-memory exporter.
    """
    if tracer_provider is None:
        return trace.get_tracer(INSTRUMENTATION_NAME)
    return tracer_provider.get_tracer(INSTRUMENTATION_NAME)


def _set_request_attributes(span: trace.Span, request: GenAIRequest) -> None:
    """Set the `gen_ai.*` request attributes (no message text — PII policy)."""
    span.set_attribute(genai.GEN_AI_OPERATION_NAME, _OPERATION_CHAT)
    span.set_attribute(genai.GEN_AI_SYSTEM, request.system)
    span.set_attribute(genai.GEN_AI_REQUEST_MODEL, request.model)
    if request.max_tokens is not None:
        span.set_attribute(genai.GEN_AI_REQUEST_MAX_TOKENS, request.max_tokens)
    if request.temperature is not None:
        span.set_attribute(genai.GEN_AI_REQUEST_TEMPERATURE, request.temperature)


def set_response_attributes(span: trace.Span, response: GenAIResponse) -> None:
    """Set the `gen_ai.response.*` and `gen_ai.usage.*` attributes for a result.

    Exposed for the streaming path, where the final usage/finish-reason is known
    only once the terminal delta arrives (the `chat_span` context manager covers
    the non-streaming path). Maps the gateway's four `Usage` token fields onto
    the distinct semconv usage attributes (atlas-docs/03 §1.4, §6.2):

    - ``input_tokens``                → ``gen_ai.usage.input_tokens``
    - ``output_tokens``               → ``gen_ai.usage.output_tokens``
    - ``cache_creation_input_tokens`` → ``gen_ai.usage.cache_creation.input_tokens``
    - ``cache_read_input_tokens``     → ``gen_ai.usage.cache_read.input_tokens``
    """
    if response.model is not None:
        span.set_attribute(genai.GEN_AI_RESPONSE_MODEL, response.model)
    if response.finish_reasons is not None:
        span.set_attribute(genai.GEN_AI_RESPONSE_FINISH_REASONS, list(response.finish_reasons))
    usage = response.usage
    span.set_attribute(genai.GEN_AI_USAGE_INPUT_TOKENS, usage.input_tokens)
    span.set_attribute(genai.GEN_AI_USAGE_OUTPUT_TOKENS, usage.output_tokens)
    span.set_attribute(
        genai.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
        usage.cache_creation_input_tokens,
    )
    span.set_attribute(
        genai.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        usage.cache_read_input_tokens,
    )


@contextmanager
def chat_span(
    tracer: Tracer,
    request: GenAIRequest,
    *,
    response: GenAIResponse | None = None,
) -> Iterator[trace.Span]:
    """Open a GenAI `chat` span around one completion, setting `gen_ai.*` attrs.

    The span is named ``"chat {model}"`` (semconv convention) and kind
    ``CLIENT`` (the gateway is the GenAI client). Request attributes are set on
    entry. For the non-streaming path, pass `response` and the usage/finish-
    reason attributes are set on exit; for streaming, omit it and call
    `set_response_attributes` on the yielded span once the terminal delta lands.

    Any exception raised inside the block records the exception on the span and
    marks it ``ERROR`` before re-raising — the caller's error handling is
    unchanged, only observed.
    """
    span_name = f"{_OPERATION_CHAT} {request.model}"
    with tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
        _set_request_attributes(span, request)
        try:
            yield span
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.record_exception(exc)
            raise
        else:
            if response is not None:
                set_response_attributes(span, response)
            span.set_status(Status(StatusCode.OK))
