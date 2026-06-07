"""Domain-level errors.

Raised by the service layer in framework-neutral terms; the API layer maps them
to HTTP responses (see `app.api.v1.chat`). Keeps business logic free of
HTTP/FastAPI concerns. See ADR-016.
"""

from __future__ import annotations


class UnknownModelError(Exception):
    """Raised when no provider is registered for the requested model/alias."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"unknown model: {model}")
