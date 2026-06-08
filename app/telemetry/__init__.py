"""Telemetry layer — OpenTelemetry GenAI instrumentation (GW-18).

Holds the OTel span helpers (`otel`): a thin capability adapter (ADR-016) that
sets the GenAI semantic-convention span attributes (`gen_ai.*`) for one chat
call. The tracer is injected (composition root supplies the real SDK tracer;
tests supply one wired to an in-memory exporter), so this adapter does no SDK
bootstrapping and unit-tests offline. Per the PII policy (atlas-docs/04 §6.2)
no raw prompt or completion text is ever placed on a span. Wiring this into the
chat request path — and the OTel SDK → Collector → Splunk export (INF-13) — are
separate tickets. See GW-18 + ADR-009 + atlas-docs/04 §6.1, §6.2.
"""
