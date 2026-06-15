"""GRD-4 — Prompt-injection heuristic guardrail.

A `Guardrail` (conforms to `app.guardrails.chain.Guardrail`) that flags known
prompt-injection patterns in inbound message content using stdlib `re` only —
no external deps, no network. Detection is heuristic and fail-fast: on a match
it raises an explicit `GuardrailRejection` naming which pattern fired; it never
passes a flagged request silently.

False-positive note
-------------------
These are deliberately broad surface-pattern heuristics, not a classifier, so
they WILL fire on legitimate content that merely discusses prompt injection —
e.g. a security researcher pasting "ignore previous instructions" as the text
they are analyzing, or documentation that quotes a known jailbreak. This is an
accepted trade-off for a cheap pre-filter: it favors recall over precision.
Mitigations for a later ticket: scope matching to the latest user turn,
allow-list trusted tenants, or downgrade to a soft signal that a heavier
classifier adjudicates. This module is intentionally conservative and standalone
until that wiring lands. See ADR-016.
"""

from __future__ import annotations

from collections.abc import Sequence
from re import IGNORECASE, Pattern, compile

from app.domain.messages import Message
from app.guardrails.chain import GuardrailContext, GuardrailPhase, GuardrailRejection

# label -> compiled pattern. Labels are safe to surface in a rejection reason;
# matched text is never included. Patterns are case-insensitive.
_PATTERNS: tuple[tuple[str, Pattern[str]], ...] = (
    (
        "ignore-previous-instructions",
        compile(r"ignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\s+", IGNORECASE),
    ),
    (
        "disregard-instructions",
        compile(
            r"disregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|earlier)\s+", IGNORECASE
        ),
    ),
    (
        "override-system-prompt",
        compile(
            r"(?:override|bypass|forget)\s+(?:your\s+|the\s+)?(?:system\s+)?(?:prompt|instructions|rules)",
            IGNORECASE,
        ),
    ),
    (
        "reveal-system-prompt",
        compile(
            r"(?:reveal|show|print|repeat|leak)\s+(?:me\s+)?(?:your\s+|the\s+)?(?:system\s+)?prompt",
            IGNORECASE,
        ),
    ),
    (
        "developer-mode",
        compile(r"\b(?:developer|dev|debug)\s+mode\b", IGNORECASE),
    ),
    (
        "dan-jailbreak",
        compile(r"\bdo\s+anything\s+now\b|\bDAN\s+mode\b", IGNORECASE),
    ),
    (
        "fake-role-injection",
        compile(r"(?m)^\s*(?:system|assistant)\s*:", IGNORECASE),
    ),
)


class InjectionGuardrail:
    """Flags known prompt-injection patterns; rejects fail-fast on first match.

    Conforms to the `Guardrail` protocol. Scans each message's content in order;
    the first matching pattern raises a `GuardrailRejection` whose reason names
    the pattern label (never the matched text). See the module false-positive
    note for the recall-over-precision trade-off.
    """

    name = "injection"

    async def check(self, ctx: GuardrailContext) -> None:
        """Scan `ctx.messages`; raise on the first injection pattern matched."""
        label = self.matched_label(ctx.messages)
        if label is None:
            return
        raise GuardrailRejection(
            guardrail=self.name,
            phase=GuardrailPhase.PRE,
            reason=f"prompt-injection pattern matched: {label}",
        )

    @staticmethod
    def matched_label(messages: Sequence[Message]) -> str | None:
        """Return the label of the first heuristic pattern that matches, else `None`.

        The public, side-effect-free form of the screen `check` raises on. GRD-5
        (`injection_classifier`) reuses it to decide ambiguity — an input the
        heuristic already flags is handled here and is *not* escalated to the
        cheap-model classifier. Returns the label, never the matched text.
        """
        for message in messages:
            for label, pattern in _PATTERNS:
                if pattern.search(message.content):
                    return label
        return None
