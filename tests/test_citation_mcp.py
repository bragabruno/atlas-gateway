"""AGT-12 — live citations-MCP verifier tests (offline, fake client).

Pins the live `McpCitationVerifier` against a **fake** `McpToolClient` (no live
server, no network):

- it satisfies the GRD-9 `CitationVerifier` Protocol and drops into `CitationGuardrail`;
- one ``verify`` call issues exactly one ``verify_citation`` tool call with the §6.2
  ``{source_id, claim}`` arguments;
- the tool's ``{exists, snippet}`` result is mapped onto a `CitationCheck`;
- a malformed result (missing/non-boolean ``exists``) raises — never a silent pass;
- a transport error from the client propagates (fail-closed, atlas-docs/01 §1.6);
- the config gate is default OFF (``None`` client → ``None`` verifier; request path unchanged);
- end-to-end: GRD-9's guardrail using the live verifier rejects an unsupported claim.

Live verification against the ingested corpus needs the deployed ``mcp-citations``
server + corpus (cluster) and is deferred. See AGT-12 + GRD-9.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.domain.messages import ChatResult, Usage
from app.guardrails.chain import GuardrailContext, GuardrailRejection
from app.guardrails.citation import CitationCheck, CitationGuardrail, CitationVerifier
from app.guardrails.citation_mcp import (
    McpCitationVerifier,
    McpToolClient,
    make_citation_verifier,
)


class _FakeMcpClient:
    """Fake `McpToolClient`: a ``{source_id: {exists, snippet}}`` lookup, no network.

    Records every call so tests can assert the tool name and argument shape. An
    unknown ``source_id`` returns the fail-closed ``{exists: false}`` the real
    server would (atlas-docs/01 §1.6).
    """

    def __init__(self, results: dict[str, dict[str, Any]]) -> None:
        self._results = results
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        source_id = arguments["source_id"]
        return self._results.get(source_id, {"exists": False, "snippet": None})


class _RaisingMcpClient:
    """Fake `McpToolClient` whose transport always errors (propagation test)."""

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        raise ConnectionError("mcp-citations unreachable")


def _ctx(content: str) -> GuardrailContext:
    return GuardrailContext(
        tenant_id="tenant-a",
        model="mock",
        messages=[],
        result=ChatResult(model="mock", content=content, usage=Usage()),
    )


# ── result mapping ───────────────────────────────────────────────────────────


async def test_maps_exists_true_with_snippet() -> None:
    client = _FakeMcpClient({"doc-1": {"exists": True, "snippet": "the supporting text"}})
    check = await McpCitationVerifier(client).verify("doc-1", "a claim")
    assert check == CitationCheck(exists=True, snippet="the supporting text")


async def test_maps_exists_false_to_no_snippet() -> None:
    client = _FakeMcpClient({"doc-1": {"exists": False, "snippet": None}})
    check = await McpCitationVerifier(client).verify("doc-1", "a claim")
    assert check.exists is False
    assert check.snippet is None


async def test_unknown_source_is_fail_closed_false() -> None:
    client = _FakeMcpClient({})
    check = await McpCitationVerifier(client).verify("ghost-doc", "a claim")
    assert check.exists is False


async def test_non_string_snippet_is_normalised_to_none() -> None:
    client = _FakeMcpClient({"doc-1": {"exists": True, "snippet": 123}})
    check = await McpCitationVerifier(client).verify("doc-1", "a claim")
    assert check.exists is True
    assert check.snippet is None


# ── tool call shape (matches §6.2) ───────────────────────────────────────────


async def test_issues_one_verify_citation_call_with_expected_args() -> None:
    client = _FakeMcpClient({"doc-1": {"exists": True, "snippet": "x"}})
    await McpCitationVerifier(client).verify("doc-1", "the market grew")
    assert client.calls == [("verify_citation", {"source_id": "doc-1", "claim": "the market grew"})]


# ── fail-closed: malformed result + transport error ──────────────────────────


@pytest.mark.parametrize("bad", [{}, {"snippet": "x"}, {"exists": "yes"}, {"exists": None}])
async def test_malformed_result_raises_never_silent_pass(bad: dict[str, Any]) -> None:
    client = _FakeMcpClient({"doc-1": bad})
    with pytest.raises(ValueError, match="exists"):
        await McpCitationVerifier(client).verify("doc-1", "a claim")


async def test_transport_error_propagates() -> None:
    with pytest.raises(ConnectionError, match="unreachable"):
        await McpCitationVerifier(_RaisingMcpClient()).verify("doc-1", "a claim")


# ── protocol conformance ─────────────────────────────────────────────────────


def test_verifier_conforms_to_citation_verifier_protocol() -> None:
    assert isinstance(McpCitationVerifier(_FakeMcpClient({})), CitationVerifier)


def test_fake_client_conforms_to_mcp_tool_client_protocol() -> None:
    assert isinstance(_FakeMcpClient({}), McpToolClient)


# ── config gate: default OFF ─────────────────────────────────────────────────


def test_make_verifier_returns_none_without_client() -> None:
    assert make_citation_verifier(None) is None


def test_make_verifier_builds_live_verifier_with_client() -> None:
    verifier = make_citation_verifier(_FakeMcpClient({}))
    assert isinstance(verifier, McpCitationVerifier)


# ── end-to-end through the GRD-9 guardrail (live verifier, fake client) ───────


async def test_guardrail_passes_supported_claim_via_live_verifier() -> None:
    client = _FakeMcpClient({"doc-1": {"exists": True, "snippet": "grounded"}})
    guard = CitationGuardrail(McpCitationVerifier(client))
    await guard.check(_ctx("Revenue rose in 2025. [source: doc-1]"))
    assert client.calls  # the live verifier was consulted


async def test_guardrail_rejects_unsupported_claim_via_live_verifier() -> None:
    client = _FakeMcpClient({"doc-1": {"exists": False, "snippet": None}})
    guard = CitationGuardrail(McpCitationVerifier(client))
    with pytest.raises(GuardrailRejection, match="unsupported-citation"):
        await guard.check(_ctx("Profits tripled in 2030. [source: doc-1]"))
