"""GRD-3 — PII NER detection for novel formats, run OFF the inline path.

The GRD-2 regex fast-path (`app.guardrails.pii`) catches high-confidence,
structurally-regular PII (email, phone, SSN-like, card-like) sub-millisecond on
the inline PRE path. This module is the *slow-path* complement: named-entity-style
detection of the free-text PII regex misses — person names, street/postal
addresses, IBAN-style account numbers, IP addresses — that have higher recall but
higher cost.

Latency strategy (atlas-docs/05 §2.1) — why this runs OFF the inline path
-------------------------------------------------------------------------
The PRE chain has a ≤50ms p95 inline budget. Regex (GRD-2) runs inline always and
redacts immediately; this NER pass is **offloaded off the inline path** — the
gateway enqueues it asynchronously / fans it to a sidecar so the request is not
blocked on it (`ner_offload: true` per route). To make "off-path" a property of
the *design*, not just the wiring, this analyzer is **non-mutating**: `analyze`
returns a `NerFindings` report (per-category counts + a redacted copy of each
message) and never writes back onto the live `GuardrailContext` the inline path
forwards. The off-path lane consumes the report (audit log, async redaction of a
stored transcript, alerting); the inline request proceeds on the regex-redacted
text. A route that explicitly opts into synchronous NER (high-sensitivity
data-room queries) can still drive this analyzer inline and apply the report
itself, accepting the latency trade-off.

DEP DECISION — extended-pattern stand-in; full NER model deferred
-----------------------------------------------------------------
The ticket prefers Presidio (presidio-analyzer + a pinned spaCy model) or GLiNER.
Those cannot all be pinned to versions published on or before the 2026-05-24
supply-chain floor *through PyPI*:

- The spaCy model wheel Presidio's NlpEngine needs (``en_core_web_*``) is **not
  distributed on PyPI** — it is a GitHub release asset, so it cannot be pinned as
  a PyPI dependency nor its publish date verified through the required PyPI
  mechanism. Presidio NER is non-functional without it.
- Presidio's required transitive ``tldextract`` performs a **network fetch** of
  the public-suffix list on first use, which breaks the offline-clean test
  posture every other guardrail holds.

Per the ticket's fallback clause, this module ships a lightweight
extended-pattern NER stand-in behind the **same `PiiNerDetector` Protocol/seam**.
Swapping in a fully-pinned Presidio/GLiNER backend later is a drop-in: implement
`PiiNerDetector` and inject it — no caller change. **The full NER model is
deferred** (tracked by GRD-3); the stand-in raises recall over GRD-2's regex
without adding any dependency.

Privacy invariant (atlas-docs/05 §6.4): like GRD-2, this never logs/stores/embeds
raw PII — `NerFindings` carries only per-category counts and placeholder-redacted
copies. See GRD-3 + GRD-2 + ADR-016 + atlas-docs/05 §2.1, §6.4.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.domain.messages import Message
from app.guardrails.chain import GuardrailContext, GuardrailPhase, GuardrailRejection
from app.guardrails.pii import PiiMode


@dataclass(frozen=True, slots=True)
class NerFindings:
    """The non-mutating result of an off-path NER pass over a request.

    `counts` is the per-category match total (e.g. ``{"PERSON": 1, "ADDRESS":
    1}``); empty when nothing was found. `redacted_messages` are placeholder-
    redacted copies the off-path lane may persist instead of the raw transcript —
    the live inline context is never mutated. Carries only counts and
    placeholders; never raw PII (atlas-docs/05 §6.4).
    """

    counts: dict[str, int]
    redacted_messages: tuple[Message, ...]

    @property
    def detected(self) -> bool:
        """`True` if any NER entity was found."""
        return bool(self.counts)


@runtime_checkable
class PiiNerDetector(Protocol):
    """Seam for a named-entity PII detector applied off the inline path.

    A fully-pinned Presidio/GLiNER backend implements this later with no caller
    change; `ExtendedPatternNer` is the dependency-free stand-in shipped now.
    `analyze` is pure: it returns a `NerFindings` report and mutates nothing.
    """

    def analyze(self, messages: Sequence[Message]) -> NerFindings:
        """Return the NER findings for `messages` without mutating them."""
        ...


def _ner_rule(label: str, regex: str) -> tuple[str, re.Pattern[str]]:
    """Build a labelled compiled pattern (case-sensitive where casing matters)."""
    return (label, re.compile(regex))


#: Extended NER-style patterns targeting free-text PII the GRD-2 regex fast-path
#: does NOT cover. Deliberately conservative surface heuristics (a stand-in for a
#: real NER model, not an exhaustive validator): they raise recall on novel
#: formats while staying dependency-free and offline. Order is stable so counts
#: are deterministic.
_NER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Titled person name: an honorific followed by 1-2 capitalised tokens.
    _ner_rule(
        "PERSON",
        r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b",
    ),
    # US-style street address: number + street words + street-type suffix.
    _ner_rule(
        "ADDRESS",
        r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}"
        r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|Way|Court|Ct)\b\.?",
    ),
    # IBAN-style account number: 2 country letters, 2 check digits, 11-30 alnum.
    _ner_rule(
        "IBAN",
        r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
    ),
    # IPv4 address (each octet 0-255 is over-precise for a stand-in; the simple
    # dotted-quad shape is enough to raise recall over GRD-2 which has no IP rule).
    _ner_rule(
        "IP_ADDRESS",
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    ),
)


#: Public, read-only view of the extended NER-style patterns for reuse by the
#: telemetry redaction layer (GRD-12), so the request-path and log/trace
#: definitions of "what is PII" share one source and cannot drift.
NER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = _NER_PATTERNS


def _redact_text(text: str) -> tuple[str, dict[str, int]]:
    """Redact all extended-NER PII in `text`; return redacted text + counts.

    Placeholders use the ``[PII:<CATEGORY>]`` form (atlas-docs/05 §2.1). Raw
    matched substrings are discarded — only counts and placeholders survive.
    """
    counts: dict[str, int] = {}
    redacted = text
    for category, pattern in _NER_PATTERNS:
        placeholder = f"[PII:{category}]"
        redacted, n = pattern.subn(placeholder, redacted)
        if n:
            counts[category] = counts.get(category, 0) + n
    return redacted, counts


class ExtendedPatternNer:
    """Dependency-free NER stand-in over the extended PII pattern set.

    Conforms to `PiiNerDetector`. Pure and non-mutating: `analyze` returns a
    `NerFindings` report with per-category counts and placeholder-redacted copies,
    leaving the inputs untouched. This is the deferred-model fallback; a pinned
    Presidio/GLiNER backend can replace it behind the same Protocol.
    """

    def analyze(self, messages: Sequence[Message]) -> NerFindings:
        """Scan `messages` for extended-NER PII; return findings, mutate nothing."""
        totals: dict[str, int] = {}
        redacted: list[Message] = []
        for message in messages:
            redacted_text, counts = _redact_text(message.content)
            for category, n in counts.items():
                totals[category] = totals.get(category, 0) + n
            redacted.append(message.model_copy(update={"content": redacted_text}))
        return NerFindings(counts=totals, redacted_messages=tuple(redacted))


def _summary(counts: dict[str, int]) -> str:
    """Render counts deterministically, e.g. ``ADDRESS=1, PERSON=2``. No raw PII."""
    return ", ".join(f"{category}={counts[category]}" for category in sorted(counts))


class PiiNerGuardrail:
    """Off-path NER PII guardrail: report (default) or reject on detection.

    Conforms to the `Guardrail` protocol so it can sit in a `GuardrailChain`, but
    its intended home is the **off-inline-path lane** (atlas-docs/05 §2.1): the
    detector is non-mutating, so in the default `REDACT` mode `check` passes
    without rewriting the live inline context — the `NerFindings` are meant to be
    consumed asynchronously off-path. In `REJECT` mode it raises a
    `GuardrailRejection` carrying only category counts (never raw PII), for routes
    that opt into synchronous high-sensitivity NER.

    The `detector` is injected (defaults to `ExtendedPatternNer`); a fully-pinned
    Presidio/GLiNER backend drops in here later with no other change.
    """

    name = "pii_ner"

    def __init__(
        self,
        *,
        mode: PiiMode = PiiMode.REDACT,
        detector: PiiNerDetector | None = None,
    ) -> None:
        self._mode = mode
        self._detector: PiiNerDetector = detector if detector is not None else ExtendedPatternNer()

    @property
    def mode(self) -> PiiMode:
        """The configured detection mode."""
        return self._mode

    def analyze(self, messages: Sequence[Message]) -> NerFindings:
        """Run the injected detector off-path; pure, returns findings only."""
        return self._detector.analyze(messages)

    async def check(self, ctx: GuardrailContext) -> None:
        """Detect novel-format PII. Reject mode raises with counts only; never logs raw PII.

        `REDACT` mode is non-mutating by design (off-path posture, §2.1): it does
        not rewrite `ctx`. Callers wanting the redacted copies read `analyze`.
        """
        findings = self._detector.analyze(ctx.messages)
        if not findings.detected:
            return
        if self._mode is PiiMode.REJECT:
            raise GuardrailRejection(
                guardrail=self.name,
                phase=GuardrailPhase.PRE,
                reason=f"NER PII detected: {_summary(findings.counts)}",
            )
