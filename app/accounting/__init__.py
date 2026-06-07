"""Accounting layer — per-call cost pricing and recording (GW-14).

Holds the cost recorder (`recorder`): a pure `compute_cost` over the four
`Usage` token fields at the alias row's rates (cache-creation ~1.25x,
cache-read ~0.1x per atlas-docs/03 §2) and a `CallRecorder` that persists one
`call_records` row via asyncpg. A thin capability adapter (ADR-016): the
formula is pure, the connection is injected (so it unit-tests with a fake), and
the insert is idempotent on the call id so a retry never double-bills. Wiring it
(and the Kafka `atlas.calls.v1` publish) into the chat request path is a
separate ticket. See GW-14 + ADR-016 + atlas-docs/03 §1.4, §2.
"""
