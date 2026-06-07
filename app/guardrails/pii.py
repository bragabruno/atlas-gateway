"""GRD-2 — PII regex fast-path guardrail.

A `Guardrail` (conforms to `app.guardrails.chain.Guardrail`) that detects and
redacts common PII in message content using stdlib `re` only — no external
deps, no network. It is a *fast path*: cheap regex catches the obvious cases
(email, phone, SSN-like, credit-card-like) before any heavier downstream check.

Privacy invariant: this check NEVER logs, stores, or embeds raw PII. It reports
only category counts and placeholder tokens; the matched substrings are
replaced in place with `[REDACTED_<CATEGORY>]`. The redacted, PII-free messages
are written back onto the `GuardrailContext` so later stages and the provider
never see the raw values.

Mode is configurable. In `redact` mode (default) the check mutates the context
and passes; in `reject` mode it raises an explicit `GuardrailRejection` carrying
only category counts (never the raw match). Fail-fast: it never silently drops a
detection. Wiring into the request path is a later ticket. See ADR-016.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import Enum

from app.domain.messages import Message
from app.guardrails.chain import GuardrailContext, GuardrailPhase, GuardrailRejection


class PiiMode(str, Enum):
    """What the check does when it finds PII."""

    REDACT = "redact"
    REJECT = "reject"


# Ordered category -> compiled pattern. Order matters: credit cards are matched
# before phone numbers so a 16-digit card is not partially consumed as a phone.
# These are deliberately conservative fast-path heuristics, not exhaustive
# validators (e.g. no Luhn check); a later ticket can layer stricter validation.
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        # 13-16 digit card-like sequences, optional space/hyphen separators
        # strictly *between* digits (so a trailing space is never consumed).
        "CREDIT_CARD",
        re.compile(r"\b\d(?:[ -]?\d){12,15}\b"),
    ),
    (
        # US SSN-like: 3-2-4 grouped digits with hyphen or space separators.
        "SSN",
        re.compile(r"\b\d{3}[ -]\d{2}[ -]\d{4}\b"),
    ),
    (
        # Phone-like: optional +country, separators, 9-14 digits total.
        "PHONE",
        re.compile(r"(?<!\w)\+?\d[\d \-().]{7,}\d(?!\w)"),
    ),
)


def _redact_text(text: str) -> tuple[str, dict[str, int]]:
    """Redact all known PII in `text`.

    Returns the redacted text and a per-category match count. The raw matched
    substrings are discarded — only counts and placeholders survive.
    """
    counts: dict[str, int] = {}
    redacted = text
    for category, pattern in _PATTERNS:
        placeholder = f"[REDACTED_{category}]"
        redacted, n = pattern.subn(placeholder, redacted)
        if n:
            counts[category] = counts.get(category, 0) + n
    return redacted, counts


class PiiGuardrail:
    """Regex PII fast-path: redact (default) or reject on detection.

    Conforms to the `Guardrail` protocol. In `redact` mode the context's
    messages are replaced with PII-free copies and the check passes. In
    `reject` mode a `GuardrailRejection` is raised whose reason carries only
    category counts — never the raw matched PII.
    """

    name = "pii"

    def __init__(self, *, mode: PiiMode = PiiMode.REDACT) -> None:
        self._mode = mode

    @property
    def mode(self) -> PiiMode:
        """The configured detection mode."""
        return self._mode

    async def check(self, ctx: GuardrailContext) -> None:
        """Scan `ctx.messages`; redact in place or reject. Never logs raw PII."""
        redacted_messages, counts = self._scan(ctx.messages)
        if not counts:
            return
        if self._mode is PiiMode.REJECT:
            raise GuardrailRejection(
                guardrail=self.name,
                phase=GuardrailPhase.PRE,
                reason=f"PII detected: {self._summary(counts)}",
            )
        ctx.messages = redacted_messages

    @staticmethod
    def _scan(messages: Sequence[Message]) -> tuple[list[Message], dict[str, int]]:
        """Return redacted copies of `messages` plus aggregate category counts."""
        out: list[Message] = []
        totals: dict[str, int] = {}
        for message in messages:
            redacted, counts = _redact_text(message.content)
            for category, n in counts.items():
                totals[category] = totals.get(category, 0) + n
            out.append(message.model_copy(update={"content": redacted}))
        return out, totals

    @staticmethod
    def _summary(counts: dict[str, int]) -> str:
        """Render counts deterministically, e.g. `EMAIL=1, SSN=2`. No raw PII."""
        return ", ".join(f"{category}={counts[category]}" for category in sorted(counts))
