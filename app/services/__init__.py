"""Service layer — use-case orchestration.

The only layer that holds business logic. Services depend on domain types,
repositories, and capability adapters (providers, cache, …); controllers depend
on services. See ADR-016.
"""
