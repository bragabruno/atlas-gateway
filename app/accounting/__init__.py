"""Accounting layer — per-call cost pricing, recording, and event publish.

Holds the cost recorder (`recorder`, GW-14): a pure `compute_cost` over the four
`Usage` token fields at the alias row's rates (cache-creation ~1.25x,
cache-read ~0.1x per atlas-docs/03 §2) and a `CallRecorder` that persists one
`call_records` row via asyncpg; plus the accounting-event publisher (`events`,
GW-15): one `atlas.calls.v1` event per completed call, keyed by `api_key_id`
(atlas-docs/03 §4.1, ADR-007). Both are thin capability adapters (ADR-016): the
formula is pure, the asyncpg connection and the Kafka producer are injected (so
each unit-tests with a fake), the insert is idempotent on the call id so a retry
never double-bills, and the event publish is non-blocking and backpressure-
tolerant so accounting never stalls the request path. Wiring them into the chat
request path is a separate ticket. See GW-14 + GW-15 + ADR-016 + ADR-007 +
atlas-docs/03 §1.4, §2, §4.1.
"""
