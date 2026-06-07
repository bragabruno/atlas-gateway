"""Provider registry — the composition point mapping model/alias → adapter.

Analogous to a bean registry: concrete providers register here and the service
layer resolves them through `ProviderRegistry`, so adding a real provider
(GW-3..5) touches neither the service nor the controller. Today only the Mock
provider is wired. Alias routing (GW-10) will resolve aliases to entries here.
See ADR-012 + ADR-016.
"""

from __future__ import annotations

from app.providers.base import Provider
from app.providers.mock import MockProvider


class ProviderRegistry:
    """Holds the model → `Provider` adapter map and resolves by model id."""

    def __init__(self, providers: dict[str, Provider] | None = None) -> None:
        self._providers: dict[str, Provider] = (
            providers if providers is not None else {"mock": MockProvider()}
        )

    def resolve(self, model: str) -> Provider | None:
        return self._providers.get(model)

    def names(self) -> list[str]:
        return list(self._providers)
