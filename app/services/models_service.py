"""GW-8 — model-listing use-case orchestration (service layer).

Builds the OpenAI-compatible `/v1/models` list from two sources: the concrete
provider models registered in `ProviderRegistry` (owned by their provider) and
the Atlas alias table (the README aliases, owned by Atlas). The controller
(`app.api.v1.models`) stays thin — all listing logic lives here. See ADR-016.
"""

from __future__ import annotations

from app.domain.openai import Model, ModelList
from app.providers.registry import ProviderRegistry

# Atlas-defined aliases (README "Model Aliases"); routing to a primary/fallback
# provider lands in GW-10. Listed here so clients can discover them.
_ALIASES: tuple[str, ...] = ("smart", "deep", "fast", "balanced", "embed")
_ALIAS_OWNER = "atlas"


class ModelsService:
    """Assembles the `/v1/models` list from the registry and alias table."""

    def __init__(self, registry: ProviderRegistry) -> None:
        self._registry = registry

    def list_models(self) -> ModelList:
        """Return the OpenAI-shaped list of provider models plus Atlas aliases."""
        data = [Model(id=name, owned_by=name) for name in self._registry.names()]
        data.extend(Model(id=alias, owned_by=_ALIAS_OWNER) for alias in _ALIASES)
        return ModelList(data=data)
