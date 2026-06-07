"""GRD-9 — citation-enforcement guardrail tests (POST).

Pins (fully offline) the answer-grounding rule against a **stub** verifier shaped
like the ``verify_citation`` MCP tool (atlas-docs/03 §6.2):

- an uncited factual claim is rejected fail-fast (``uncited-claim``);
- a cited claim the source does not support is rejected (``unsupported-citation``);
- a properly-cited, supported answer passes;
- non-factual boilerplate (questions, refusals, greetings) needs no citation;
- multi-source claims pass if any cited source supports them;
- the stub matches the §6.2 contract shape (CitationVerifier / CitationCheck);
- the post-phase contract (no result → reject) holds and the protocol conforms.

The live wiring to the citations MCP is AGT-12 — not exercised here. See GRD-9 +
ADR-016.
"""

from __future__ import annotations

import pytest

from app.domain.messages import ChatResult, Usage
from app.guardrails.chain import Guardrail, GuardrailContext, GuardrailRejection
from app.guardrails.citation import (
    CitationCheck,
    CitationGuardrail,
    CitationVerifier,
)


class _StubVerifier:
    """Stub `CitationVerifier` (mirrors ``verify_citation`` §6.2).

    Backed by a ``{source_id: supporting_text}`` map. A claim "exists" when the
    mapped source text contains every significant word of the claim — a crude but
    deterministic stand-in for the real grounding check, enough to prove the gate.
    Records calls so tests can assert the verifier was actually consulted.
    """

    def __init__(self, sources: dict[str, str]) -> None:
        self._sources = sources
        self.calls: list[tuple[str, str]] = []

    async def verify(self, source_id: str, claim: str) -> CitationCheck:
        self.calls.append((source_id, claim))
        text = self._sources.get(source_id)
        if text is None:
            return CitationCheck(exists=False, snippet=None)
        words = [w for w in claim.lower().split() if len(w) > 3]
        if words and all(w.strip(".,") in text.lower() for w in words):
            return CitationCheck(exists=True, snippet=text[:40])
        return CitationCheck(exists=False, snippet=None)


def _ctx(content: str) -> GuardrailContext:
    return GuardrailContext(
        tenant_id="tenant-a",
        model="mock",
        messages=[],
        result=ChatResult(model="mock", content=content, usage=Usage()),
    )


# ── happy path ─────────────────────────────────────────────────────────────────


async def test_properly_cited_supported_answer_passes() -> None:
    verifier = _StubVerifier({"doc-1": "The Eiffel Tower is located in Paris France."})
    guard = CitationGuardrail(verifier)
    await guard.check(_ctx("The Eiffel Tower is in Paris. [source: doc-1]"))
    assert verifier.calls  # the verifier was actually consulted


# ── rejection: uncited ─────────────────────────────────────────────────────────


async def test_uncited_claim_is_rejected() -> None:
    verifier = _StubVerifier({})
    guard = CitationGuardrail(verifier)
    with pytest.raises(GuardrailRejection) as exc:
        await guard.check(_ctx("The market grew 12 percent last quarter."))
    assert exc.value.guardrail == "citation"
    assert "uncited-claim" in exc.value.reason
    # An uncited claim never reaches the verifier.
    assert verifier.calls == []


async def test_one_uncited_claim_among_cited_ones_is_rejected() -> None:
    verifier = _StubVerifier({"doc-1": "Revenue rose sharply in 2025 across regions."})
    guard = CitationGuardrail(verifier)
    answer = (
        "Revenue rose sharply in 2025. [source: doc-1] "
        "Headcount also doubled over the same period."  # uncited
    )
    with pytest.raises(GuardrailRejection) as exc:
        await guard.check(_ctx(answer))
    assert "uncited-claim" in exc.value.reason


# ── rejection: unsupported ─────────────────────────────────────────────────────


async def test_cited_but_unsupported_claim_is_rejected() -> None:
    verifier = _StubVerifier({"doc-1": "The report covers fiscal year 2024 only."})
    guard = CitationGuardrail(verifier)
    with pytest.raises(GuardrailRejection) as exc:
        await guard.check(_ctx("Profits tripled in 2030. [source: doc-1]"))
    assert "unsupported-citation" in exc.value.reason


async def test_citation_to_unknown_source_is_rejected() -> None:
    verifier = _StubVerifier({"doc-1": "known content"})
    guard = CitationGuardrail(verifier)
    with pytest.raises(GuardrailRejection) as exc:
        await guard.check(_ctx("Some asserted fact here. [source: ghost-doc]"))
    assert "unsupported-citation" in exc.value.reason


async def test_reason_never_echoes_source_snippet() -> None:
    secret_snippet = "CONFIDENTIAL internal figure 42"
    verifier = _StubVerifier({"doc-1": secret_snippet})
    guard = CitationGuardrail(verifier)
    with pytest.raises(GuardrailRejection) as exc:
        await guard.check(_ctx("An entirely different unsupported claim. [source: doc-1]"))
    assert secret_snippet not in exc.value.reason


# ── non-factual sentences are exempt ───────────────────────────────────────────


async def test_question_needs_no_citation() -> None:
    guard = CitationGuardrail(_StubVerifier({}))
    await guard.check(_ctx("Would you like a more detailed breakdown?"))


async def test_refusal_needs_no_citation() -> None:
    guard = CitationGuardrail(_StubVerifier({}))
    await guard.check(_ctx("I don't know the answer to that."))


async def test_greeting_needs_no_citation() -> None:
    guard = CitationGuardrail(_StubVerifier({}))
    await guard.check(_ctx("Hello! Thanks for your question."))


# ── multi-source claims ────────────────────────────────────────────────────────


async def test_claim_passes_if_any_cited_source_supports_it() -> None:
    verifier = _StubVerifier(
        {
            "doc-1": "unrelated content about weather patterns",
            "doc-2": "The new policy takes effect in January 2026.",
        }
    )
    guard = CitationGuardrail(verifier)
    await guard.check(
        _ctx("The new policy takes effect January 2026. [source: doc-1][source: doc-2]")
    )


# ── configurability ────────────────────────────────────────────────────────────


async def test_require_citation_false_allows_uncited_but_still_verifies_cited() -> None:
    verifier = _StubVerifier({"doc-1": "supported claim about quarterly growth metrics"})
    guard = CitationGuardrail(verifier, require_citation=False)
    # Uncited claim is tolerated when require_citation is off ...
    await guard.check(_ctx("An uncited assertion."))
    # ... but a cited-yet-unsupported claim is still rejected.
    with pytest.raises(GuardrailRejection) as exc:
        await guard.check(_ctx("Completely different unsupported text. [source: doc-1]"))
    assert "unsupported-citation" in exc.value.reason


# ── contract + post-phase ──────────────────────────────────────────────────────


async def test_no_result_is_rejected() -> None:
    guard = CitationGuardrail(_StubVerifier({}))
    ctx = GuardrailContext(tenant_id="t", model="mock", messages=[], result=None)
    with pytest.raises(GuardrailRejection, match="no provider result"):
        await guard.check(ctx)


async def test_conforms_to_guardrail_protocol() -> None:
    assert isinstance(CitationGuardrail(_StubVerifier({})), Guardrail)


def test_stub_conforms_to_verifier_protocol() -> None:
    assert isinstance(_StubVerifier({}), CitationVerifier)


def test_citation_check_mirrors_contract_shape() -> None:
    # §6.2 output: {exists: bool, snippet: string|null}
    check = CitationCheck(exists=True, snippet="excerpt")
    assert check.exists is True
    assert check.snippet == "excerpt"
    assert CitationCheck(exists=False).snippet is None
