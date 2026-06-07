"""GRD-9 — Citation-enforcement guardrail (POST).

A POST `Guardrail` (conforms to `app.guardrails.chain.Guardrail`) that enforces
the answer-grounding rule for cited responses: **every factual claim in the
output must reference a doc-search ``source_id``, and that citation must be
verifiable**. An uncited claim, or a cited claim the source does not actually
support, is rejected fail-fast with an explicit reason. A response whose claims
all carry a verified ``[source: <source_id>]`` citation passes unchanged.

Contract, not wiring (atlas-docs/03 §6.2)
-----------------------------------------
Verification goes through a :class:`CitationVerifier` Protocol shaped exactly
like the ``verify_citation`` MCP tool: ``verify(source_id, claim) ->
CitationCheck`` mirrors the tool's ``{source_id, claim} -> {exists, snippet}``
JSON-Schema I/O. The ``source_id`` is the value from a ``doc_search`` result
(§6.1). This module is built and tested against a **stub** verifier (p3); the
live wiring to the citations MCP is ``AGT-12`` and deliberately not done here —
no network, no MCP client import. Because the Protocol matches the tool's
contract, the live adapter drops in without changing this check.

How claims are identified
-------------------------
A "factual claim" is a sentence of the answer. Each sentence must either carry an
inline ``[source: <source_id>]`` marker (the citation convention from
atlas-docs/03 §3.2's "citation markers") or be exempt as non-factual
boilerplate (greetings, "I don't know", a pure question). The split is a
deliberately simple sentence tokenizer — stdlib ``re`` only, no NLP dependency —
because the goal is an enforceable gate, not perfect linguistics; the
false-negative/positive trade-off is documented below.

Fail-fast, no silent pass
-------------------------
The first violation raises a `GuardrailRejection` whose reason names the failure
class (``uncited-claim`` or ``unsupported-citation``) and never echoes the source
snippet. A missing ``ctx.result`` (post-phase contract breach) raises too. A
verifier that errors is allowed to propagate — this check never swallows a
verification failure into a silent pass. This is a standalone capability adapter;
request-path/MCP wiring is a later ticket. See GRD-9 + ADR-016.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.guardrails.chain import GuardrailContext, GuardrailPhase, GuardrailRejection

#: Inline citation marker: ``[source: <source_id>]`` (atlas-docs/03 §3.2). The
#: ``source_id`` is captured; it is the opaque id from a ``doc_search`` result
#: (§6.1). Case-insensitive on the ``source`` keyword; the id itself is taken
#: verbatim (trimmed) since source ids are opaque.
_CITATION_RE = re.compile(r"\[source:\s*(?P<source_id>[^\]]+?)\s*\]", re.IGNORECASE)

#: Sentence boundary: split on terminal punctuation followed by whitespace. A
#: pragmatic tokenizer, not a linguistic one — it over-splits on abbreviations and
#: under-splits on missing punctuation. That trade-off favors enforceability: a
#: claim that dodges the splitter still needs *a* citation in its sentence to
#: pass, and a research-grade splitter is out of scope (no NLP dep). A citation
#: marker the convention places *after* a claim's period (``... Paris. [source:
#: doc-1]``) lands at the head of the next fragment; `_split_sentences`
#: re-attaches such an orphaned leading marker to the claim it cites.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

#: A fragment that *begins* with one or more citation markers (after the split) —
#: it is the trailing citation of the preceding claim, not a sentence of its own.
_LEADING_CITATION_RE = re.compile(r"^\s*(?:\[source:[^\]]*\]\s*)+", re.IGNORECASE)

#: Sentences that are not factual claims and therefore need no citation. Matched
#: against the citation-stripped, lowercased sentence. Conservative on purpose:
#: only obvious non-factual boilerplate is exempt, so the gate stays strict.
_NON_FACTUAL_RES: tuple[re.Pattern[str], ...] = (
    # A pure question carries no asserted fact.
    re.compile(r"^[^.!]*\?$"),
    # Explicit non-answers / refusals.
    re.compile(r"^(i\s+(?:do\s+not|don'?t)\s+know|i\s+cannot\s+(?:answer|help))\b"),
    # Greetings / closings.
    re.compile(r"^(hello|hi|thanks|thank\s+you|you'?re\s+welcome|sure|certainly)\b"),
)


@dataclass(frozen=True, slots=True)
class CitationCheck:
    """Result of verifying one claim against one source (mirrors §6.2 output).

    ``exists`` is ``True`` when the source supports the claim; ``snippet`` is the
    supporting excerpt (or ``None`` when ``exists`` is ``False``), matching the
    ``verify_citation`` ``{exists, snippet}`` output schema. Kept as a value
    object so a real MCP response maps onto it directly (AGT-12).
    """

    exists: bool
    snippet: str | None = None


@runtime_checkable
class CitationVerifier(Protocol):
    """Port shaped like the ``verify_citation`` MCP tool (atlas-docs/03 §6.2).

    ``verify(source_id, claim)`` answers whether `claim` is grounded in the
    document identified by `source_id` (a ``doc_search`` result id), returning a
    :class:`CitationCheck`. The async signature matches an MCP call; the stub in
    tests and the live AGT-12 adapter both satisfy it. Implementations raise on a
    verification *error* (the check must not be able to mistake an error for a
    pass).
    """

    async def verify(self, source_id: str, claim: str) -> CitationCheck:
        """Return whether `claim` is supported by source `source_id`."""
        ...


def _split_sentences(text: str) -> list[str]:
    """Split `text` into trimmed, non-empty sentences (pragmatic tokenizer).

    After splitting, a fragment that *starts* with a citation marker is the
    trailing citation of the previous claim (the period fell before the marker):
    its leading marker(s) are re-attached to the preceding sentence so the claim
    keeps its citation, and the remainder (if any) continues as its own sentence.
    """
    raw = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    sentences: list[str] = []
    for fragment in raw:
        match = _LEADING_CITATION_RE.match(fragment)
        if match and sentences:
            sentences[-1] = f"{sentences[-1]} {match.group(0).strip()}"
            remainder = fragment[match.end() :].strip()
            if remainder:
                sentences.append(remainder)
        else:
            sentences.append(fragment)
    return sentences


def _strip_citations(sentence: str) -> str:
    """Return `sentence` with its citation markers removed, for non-factual matching."""
    return _CITATION_RE.sub("", sentence).strip()


def _is_non_factual(sentence: str) -> bool:
    """Whether `sentence` (citations stripped) is exempt boilerplate, not a claim."""
    bare = _strip_citations(sentence).lower()
    if not bare:
        # A sentence that is only a citation marker asserts nothing on its own.
        return True
    return any(pattern.match(bare) for pattern in _NON_FACTUAL_RES)


class CitationGuardrail:
    """POST guardrail: reject answers with uncited or unsupported claims (GRD-9).

    Conforms to the `Guardrail` protocol. The :class:`CitationVerifier` is
    injected (the stub in tests, the AGT-12 MCP adapter in production), so this
    check does no I/O of its own. Each factual sentence of ``ctx.result.content``
    must carry at least one ``[source: <id>]`` marker AND verify against its
    source; the first sentence that fails either gate raises a
    `GuardrailRejection`. ``require_citation`` may be relaxed by a caller, but it
    defaults to ``True`` because an ungrounded answer is the failure mode this
    check exists to stop.
    """

    name = "citation"

    def __init__(self, verifier: CitationVerifier, *, require_citation: bool = True) -> None:
        self._verifier = verifier
        self._require_citation = require_citation

    async def check(self, ctx: GuardrailContext) -> None:
        """Enforce that every factual claim in the output is cited and supported.

        Raises `GuardrailRejection` on the first uncited or unsupported claim,
        and also when there is no provider result to screen (post-phase
        contract). Passes silently only when every factual sentence is grounded.
        """
        if ctx.result is None:
            raise self._reject("no provider result to screen (post-phase guardrail)")

        for sentence in _split_sentences(ctx.result.content):
            if _is_non_factual(sentence):
                continue
            source_ids = self._citations(sentence)
            if not source_ids:
                if self._require_citation:
                    raise self._reject("uncited-claim")
                continue
            await self._require_supported(sentence, source_ids)

    async def _require_supported(self, sentence: str, source_ids: Sequence[str]) -> None:
        """Pass if any cited source supports the claim; else raise unsupported-citation.

        A claim may cite several sources; it is supported if *any* of them backs
        it. The verifier is awaited per source until one returns ``exists`` —
        fail-fast on the claim, lazy on the sources.
        """
        claim = _strip_citations(sentence)
        for source_id in source_ids:
            result = await self._verifier.verify(source_id, claim)
            if result.exists:
                return
        raise self._reject("unsupported-citation")

    @staticmethod
    def _citations(sentence: str) -> list[str]:
        """Return the (trimmed, non-empty) ``source_id``s cited in `sentence`."""
        return [m.group("source_id").strip() for m in _CITATION_RE.finditer(sentence)]

    def _reject(self, reason: str) -> GuardrailRejection:
        """Build a POST-phase `GuardrailRejection` for this guardrail."""
        return GuardrailRejection(
            guardrail=self.name,
            phase=GuardrailPhase.POST,
            reason=reason,
        )
