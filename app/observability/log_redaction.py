"""GRD-12 — strip raw PII from logs and traces before they are emitted.

Privacy invariant (atlas-docs/05 §6.4): **raw PII is never written to any log
line, OTel span attribute, or metric label.** GRD-2 (`app.guardrails.pii`) and
GRD-3 (`app.guardrails.pii_ner`) redact PII on the *request* path, but a stray
``logger.info("user said %s", text)`` or a hand-set span attribute could still
carry a raw value into a sink. This module is the **last-line enforcement** at the
telemetry boundary: a logging `Filter` and an OTel `SpanProcessor` that run the
exact same GRD-2 regex + GRD-3 pattern detectors over every outbound log record
and span attribute, replacing any match in place with a redaction marker.

Reuse, not reinvention
----------------------
The detectors are the **same compiled patterns** GRD-2 and GRD-3 already own
(`app.guardrails.pii._PATTERNS`, `app.guardrails.pii_ner._NER_PATTERNS`) — imported
and applied here, never re-declared, so the request-path and telemetry-path
definitions of "what is PII" cannot drift. `redact_text` is the single shared
primitive both sinks call.

Defence in depth, not a silencer
--------------------------------
This filter is additive and orthogonal to the request path (it is a no-op when no
PII is present), and it never raises on user content: its job is to *redact*, not
to reject — a sink that dropped a log line on detection would itself become a
failure mode. The fail-fast rejection posture lives in the request-path
guardrails; here the contract is "nothing raw escapes". See GRD-12 + GRD-2 +
GRD-3 + atlas-docs/05 §6.4.

Span-attribute redaction mechanics
----------------------------------
An OTel `SpanProcessor.on_end` receives the span as a `ReadableSpan` whose
``attributes`` are an immutable view. To redact before export, this processor must
run **before** the exporting processor (processors fire in registration order over
the same span object) and replaces the span's backing ``_attributes`` mapping with
a redacted plain ``dict``. Touching that one private field is confined to this
module (the same "confine the awkward boundary to one file" idiom as
`app.telemetry._genai_semconv` and `app.limits._redis_typing`); the public
`ReadableSpan.attributes` view then reflects the redacted values for every
downstream processor.
"""

from __future__ import annotations

import logging
from typing import TypeVar, cast

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor
from opentelemetry.util import types

from app.guardrails.pii import PATTERNS as _REGEX_PATTERNS
from app.guardrails.pii_ner import NER_PATTERNS as _NER_PATTERNS

#: A value flowing through a log record or span attribute. `_redact_value` returns
#: the same type it is given, so callers (span attrs typed `AttributeValue`, log
#: args typed `object`) keep their static types across redaction.
_V = TypeVar("_V")

#: Marker substituted for any detected PII in a log/trace value. A single,
#: category-agnostic token (the sinks only need to prove *something* was redacted;
#: the per-category placeholders the request path uses carry no extra value at the
#: telemetry boundary and would leak the detector's category taxonomy into logs).
REDACTION_MARKER = "[REDACTED]"


def redact_text(text: str) -> str:
    """Return `text` with every GRD-2/GRD-3 PII match replaced by the marker.

    Applies the regex fast-path patterns (GRD-2) and the extended NER-style
    patterns (GRD-3) in order. Pure and dependency-free: the raw matched
    substrings are discarded, only the marker survives. A string with no PII is
    returned unchanged (object identity preserved), so the sinks stay zero-cost on
    clean content.
    """
    redacted = text
    for _category, pattern in (*_REGEX_PATTERNS, *_NER_PATTERNS):
        redacted = pattern.sub(REDACTION_MARKER, redacted)
    return redacted


def _redact_value(value: _V) -> _V:
    """Redact PII in a single value, recursing into the containers logs/spans use.

    Strings are redacted; tuples/lists are redacted element-wise (log ``record.args``
    and OTel sequence attributes); everything else (ints, floats, bools, ``None``)
    is returned unchanged. Non-string scalars cannot carry textual PII, so they are
    left untouched to keep the hot path cheap. Returns the same type it receives so
    static types survive redaction at both sinks.
    """
    if isinstance(value, str):
        return cast("_V", redact_text(value))
    if isinstance(value, tuple):
        items: tuple[object, ...] = value
        return cast("_V", tuple(_redact_value(item) for item in items))
    if isinstance(value, list):
        elements: list[object] = value
        return cast("_V", [_redact_value(item) for item in elements])
    return value


class PiiRedactingLogFilter(logging.Filter):
    """A `logging.Filter` that strips raw PII from every record before emission.

    Attached to a handler/logger, it rewrites the record's message and positional
    ``args`` in place so the formatted line — however the record was built
    (``"%s"`` lazy args or a pre-formatted ``msg``) — carries only the redaction
    marker, never the raw value. Returns ``True`` always: it redacts, it never
    drops a log line (dropping would itself be a silent failure mode).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact `record.msg` and `record.args`; always allow the record through."""
        if isinstance(record.msg, str):
            record.msg = redact_text(record.msg)
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(_redact_value(arg) for arg in record.args)
            elif isinstance(record.args, dict):
                # ``logger.info("%(x)s", {"x": ...})`` mapping-style args.
                record.args = {key: _redact_value(val) for key, val in record.args.items()}
        return True


class PiiRedactingSpanProcessor(SpanProcessor):
    """An OTel `SpanProcessor` that strips raw PII from span attributes on end.

    Register this **before** the exporting processor: processors fire in
    registration order over the same span object, so redacting here makes the
    redacted attributes visible to the exporter (and any later processor). The
    span's backing ``_attributes`` mapping is replaced with a redacted plain
    ``dict`` (the public ``attributes`` view is read-only); a span with no string
    attributes is left untouched.

    Defence in depth: span helpers (`app.telemetry.otel`) already keep raw
    message text off spans by policy — this processor guarantees the invariant
    holds for *any* attribute set anywhere on the span.
    """

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        """No-op: attributes are not yet known at span start."""

    def on_end(self, span: ReadableSpan) -> None:
        """Replace the span's attributes with a PII-redacted copy before export."""
        attributes = span.attributes
        if not attributes:
            return
        redacted: dict[str, types.AttributeValue] = {
            key: _redact_value(value) for key, value in attributes.items()
        }
        # The `attributes` property exposes a read-only view over `_attributes`;
        # replacing the backing mapping is the supported way to mutate a span that
        # has already ended. Confined to this module (see module docstring).
        span._attributes = redacted  # pyright: ignore[reportPrivateUsage]

    def shutdown(self) -> None:
        """No resources to release."""

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Nothing is buffered; always reports success."""
        return True
