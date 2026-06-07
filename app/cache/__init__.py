"""Cache layer — response caches the service composes around a provider call.

Holds the exact-match cache (`exact`, Redis) and, later, the semantic cache
(Qdrant, opt-in). These are thin capability adapters (ADR-016): key
composition, get/set with TTL, no business logic. Cache keys are namespaced by
tenant + prompt-version + model so a hit can never cross a tenant or version
boundary (atlas-docs/03 §3.2 tenant-isolation rule). Wiring the cache into the
chat request path is a separate ticket. See GW-13 + ADR-016.
"""
