"""Routing layer — alias resolution the service composes ahead of a call.

Holds the alias resolver (`aliases`, GW-10): it turns a routable alias
(`smart`, `deep`, …) into a concrete `RouteTarget` (provider + primary +
fallback model), honouring negotiated per-key overrides. A thin capability
adapter (ADR-016): it reads alias rows through an injected mapping so it is
testable offline from `ALIAS_SEED` with no live DB, holds no business logic,
and decides nothing about *when* to fail over. Wiring it into the chat request
path is a separate ticket. See GW-10 + ADR-016 + atlas-docs/03 §1.1, §2.
"""
