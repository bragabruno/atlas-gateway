"""GRD-10 — Output content-policy guardrail (POST).

A POST `Guardrail` (conforms to `app.guardrails.chain.Guardrail`) that screens a
provider's *output* against a configurable content policy and rejects violations
with an explicit, named reason. Stdlib `re` only — no external deps, no network.

A content policy is a list of named :class:`ContentRule`s, each a labelled
case-insensitive regex over the response text. The policy is *configurable*: the
caller (or a later wiring ticket / per-tenant config) supplies the rules, so this
adapter ships a small conservative default set but never hard-codes the only
policy. On the first matching rule the check raises a `GuardrailRejection` whose
reason names the rule that fired — never the matched substring, so a policy
violation is reported without echoing the offending content into logs or errors.

Fail-fast, no silent pass: a response that matches any rule is rejected; one that
matches none passes unchanged. An empty policy is a configuration error and fails
loudly at construction (an output guardrail that screens nothing is almost always
a mistake). This is a standalone capability adapter; wiring into the chat path is
a later ticket. See GRD-10 + ADR-016.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.guardrails.chain import GuardrailContext, GuardrailPhase, GuardrailRejection


@dataclass(frozen=True, slots=True)
class ContentRule:
    """One named content-policy rule: a label plus a compiled matcher.

    ``label`` is safe to surface in a rejection reason; the matched text never
    is. ``pattern`` is matched against the full response content with
    :meth:`re.Pattern.search` (a hit anywhere is a violation).
    """

    label: str
    pattern: re.Pattern[str]


def _rule(label: str, regex: str) -> ContentRule:
    """Build a case-insensitive :class:`ContentRule` from a raw pattern."""
    return ContentRule(label=label, pattern=re.compile(regex, re.IGNORECASE))


#: Conservative default output policy. Deliberately small and surface-level — a
#: cheap backstop, not a classifier — and fully overridable by the caller. These
#: catch the obvious cases where an upstream emits a refusal-leak, embedded
#: credentials, or an obvious self-harm/violence instruction in the *output*.
DEFAULT_CONTENT_POLICY: tuple[ContentRule, ...] = (
    _rule(
        "leaked-system-prompt",
        r"\b(?:my\s+system\s+prompt|the\s+system\s+prompt\s+is)\b",
    ),
    _rule(
        "leaked-credentials",
        r"\b(?:api[_-]?key|secret[_-]?key|password|bearer\s+token)\s*[:=]\s*\S+",
    ),
    _rule(
        "self-harm-instructions",
        r"\bhow\s+to\s+(?:kill|harm|hurt)\s+(?:yourself|myself)\b",
    ),
    _rule(
        "weapon-instructions",
        r"\bhow\s+to\s+(?:build|make|construct)\s+a\s+(?:bomb|explosive|weapon)\b",
    ),
)


class ContentPolicyGuardrail:
    """POST guardrail: reject outputs that match a configurable content policy.

    Conforms to the `Guardrail` protocol. ``policy`` is the ordered list of
    :class:`ContentRule`s applied to `ctx.result.content`; it defaults to
    :data:`DEFAULT_CONTENT_POLICY` but is fully replaceable. The first rule that
    matches raises a `GuardrailRejection` whose reason names the rule label
    (never the matched text). An empty ``policy`` fails loudly at construction.
    """

    name = "content_policy"

    def __init__(self, *, policy: Sequence[ContentRule] = DEFAULT_CONTENT_POLICY) -> None:
        if not policy:
            raise ValueError("content policy must contain at least one rule")
        self._policy: tuple[ContentRule, ...] = tuple(policy)

    @property
    def policy(self) -> tuple[ContentRule, ...]:
        """The ordered content-policy rules applied to outputs."""
        return self._policy

    async def check(self, ctx: GuardrailContext) -> None:
        """Screen `ctx.result.content`; raise on the first rule matched."""
        if ctx.result is None:
            raise GuardrailRejection(
                guardrail=self.name,
                phase=GuardrailPhase.POST,
                reason="no provider result to screen (post-phase guardrail)",
            )

        label = self._first_match(ctx.result.content)
        if label is None:
            return
        raise GuardrailRejection(
            guardrail=self.name,
            phase=GuardrailPhase.POST,
            reason=f"output content-policy violation: {label}",
        )

    def _first_match(self, content: str) -> str | None:
        """Return the label of the first rule that matches, else ``None``."""
        for rule in self._policy:
            if rule.pattern.search(content):
                return rule.label
        return None
