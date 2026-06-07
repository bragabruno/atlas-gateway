"""Domain layer — entities, DTOs, and protocols shared across layers.

Holds the framework-free contracts (provider-internal types in `messages`, the
OpenAI-compatible wire schema in `openai`, domain errors in `errors`). The
service, controller, and repository layers depend on these types, not on each
other's internals. See ADR-016.
"""
