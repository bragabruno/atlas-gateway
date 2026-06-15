"""Observability hardening that is orthogonal to the request path.

Two concerns live here:

- Security event logging + metrics (BRA-* track) — see
  `app.observability.security`.
- GRD-12: the PII redaction enforcement applied to *outbound telemetry* (log
  records and OTel span attributes) so raw PII can never leak through the
  logging/tracing sinks even if an upstream stage forgot to redact it. See
  `app.observability.log_redaction` and atlas-docs/05 §6.4.
"""

from __future__ import annotations
