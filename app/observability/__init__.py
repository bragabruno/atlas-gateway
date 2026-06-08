"""Observability hardening that is orthogonal to the request path.

GRD-12 lives here: the PII redaction enforcement applied to *outbound telemetry*
(log records and OTel span attributes) so raw PII can never leak through the
logging/tracing sinks even if an upstream stage forgot to redact it. See
`app.observability.log_redaction` and atlas-docs/05 §6.4.
"""

from __future__ import annotations
