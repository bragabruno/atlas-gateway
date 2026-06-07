"""Guardrail layer — pre/post safety checks composed as an ordered chain.

A capability adapter (ADR-016) the service layer composes around a provider
call: a `GuardrailChain` runs an ordered list of `Guardrail`s before the
request reaches the provider (pre checks) and after the response returns (post
checks). Checks are fail-fast — a rejecting check raises `GuardrailRejection`
with an explicit reason and the chain stops; nothing passes silently. Per-route
configuration selects which guardrails run and in what order. See GRD-1 +
ADR-016.
"""
