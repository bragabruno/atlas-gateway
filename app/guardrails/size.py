"""GRD-7 — Input size-cap guardrail.

A `Guardrail` (conforms to `app.guardrails.chain.Guardrail`) that rejects
oversized inputs before they reach a provider. It enforces a configurable
maximum total character count across all inbound message content (a cheap
proxy for token cost and a DoS / cost-blowout guard). No external deps, no
network.

Fail-fast and explicit: when the total exceeds the cap, the check raises a
`GuardrailRejection` whose reason reports the limit and the observed size
(counts only — never the message content). The cap must be a positive integer;
a non-positive cap is a configuration error and fails loudly at construction.
Wiring into the request path is a later ticket. See ADR-016.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.domain.messages import Message
from app.guardrails.chain import GuardrailContext, GuardrailPhase, GuardrailRejection

_DEFAULT_MAX_CHARS = 100_000


class SizeGuardrail:
    """Rejects requests whose total content length exceeds `max_chars`.

    Conforms to the `Guardrail` protocol. `max_chars` is the inclusive upper
    bound on the summed character length of every message's content; inputs at
    or below the cap pass, anything above is rejected.
    """

    name = "size"

    def __init__(self, *, max_chars: int = _DEFAULT_MAX_CHARS) -> None:
        if max_chars <= 0:
            raise ValueError(f"max_chars must be a positive integer, got {max_chars}")
        self._max_chars = max_chars

    @property
    def max_chars(self) -> int:
        """The configured inclusive character cap."""
        return self._max_chars

    async def check(self, ctx: GuardrailContext) -> None:
        """Reject if total content length exceeds the cap; otherwise pass."""
        total = self._total_chars(ctx.messages)
        if total > self._max_chars:
            raise GuardrailRejection(
                guardrail=self.name,
                phase=GuardrailPhase.PRE,
                reason=f"input too large: {total} chars exceeds cap of {self._max_chars}",
            )

    @staticmethod
    def _total_chars(messages: Sequence[Message]) -> int:
        """Sum the character length of every message's content."""
        return sum(len(message.content) for message in messages)
