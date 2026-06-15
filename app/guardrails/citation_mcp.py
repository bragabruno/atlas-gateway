"""AGT-12 — live citations-MCP adapter for the GRD-9 citation guardrail.

GRD-9 (`app.guardrails.citation`) enforces answer-grounding against a
`CitationVerifier` Protocol shaped exactly like the ``verify_citation`` MCP tool
(atlas-docs/03 §6.2: ``{source_id, claim} -> {exists, snippet}``). GRD-9 ships and
is tested against a **stub** verifier; this module supplies the **live** adapter
that fulfils that same Protocol by calling the real ``mcp-citations`` server, so
the guardrail drops in unchanged (`CitationGuardrail(make_live_or_stub(...))`).

Injected client, no transport in the verifier
---------------------------------------------
`McpCitationVerifier` does no I/O of its own — it holds an injected `McpToolClient`
(the seam) and translates one ``verify(source_id, claim)`` call into one
``call_tool("verify_citation", {...})`` round-trip, then maps the tool's
``{exists, snippet}`` result onto a `CitationCheck`. This mirrors the GRD-9 stub
seam and the GRD-3 detector seam: the unit under test is the *mapping +
fail-closed* logic, exercised offline with a **fake** client (no live server, no
network). The concrete transport is the official ``mcp`` SDK / FastMCP over
Streamable HTTP (atlas-docs/02 tech-stack table; atlas-docs/03 §6); that SDK is
**not** vendored here — adding it pre-pins a heavy dependency for a wiring seam
that the injected `McpToolClient` Protocol already abstracts. A thin
Streamable-HTTP `McpToolClient` (httpx) or an ``mcp``-SDK-backed one can be
supplied at the composition root without touching this verifier.

Fail-closed, never a silent pass (atlas-docs/01 §1.6)
----------------------------------------------------
``mcp-citations`` is fail-closed by contract: a chunk it cannot find yields
``exists: false`` and the guardrail blocks the response. This adapter preserves
that: a malformed tool result (missing/!bool ``exists``) raises rather than being
coerced to a pass, and a transport error from the injected client **propagates** —
GRD-9 must never mistake a verification *failure* for a verified claim.

Config-gated, default OFF (request path unchanged)
--------------------------------------------------
`make_citation_verifier` returns the live verifier only when a client is supplied
(driven by `Settings.citation_mcp_url`, default ``None``); otherwise it returns
``None``. With the default config no live verifier is built and the request path
is byte-for-byte the pre-AGT-12 gateway — exactly the config-gated/default-OFF
posture of every other capability (`app.config`, `app.api.deps`). Live
verification against the ingested corpus needs the deployed ``mcp-citations``
server + corpus (cluster) and is deferred (AGT-12 live step). See AGT-12 + GRD-9 +
ADR-016.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.guardrails.citation import CitationCheck, CitationVerifier

#: The MCP tool name this adapter invokes (atlas-docs/03 §6.2, atlas-docs/01 §1.6).
_VERIFY_CITATION_TOOL = "verify_citation"


@runtime_checkable
class McpToolClient(Protocol):
    """Minimal seam over an MCP client's tool-call surface.

    One method — ``call_tool(name, arguments) -> result-mapping`` — matching the
    official ``mcp`` SDK client shape, so a real Streamable-HTTP client satisfies
    it structurally and the offline fake does too. Kept this narrow on purpose:
    the verifier needs nothing else, and a wider port would couple it to the SDK's
    session lifecycle. Implementations raise on a transport/protocol error (the
    adapter must be able to let that propagate, never swallow it into a pass).
    """

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Invoke MCP tool `name` with `arguments`; return its result mapping."""
        ...


class McpCitationVerifier:
    """Live `CitationVerifier` (GRD-9) backed by the ``mcp-citations`` server.

    Satisfies the GRD-9 `CitationVerifier` Protocol by delegating each
    ``verify(source_id, claim)`` to one ``verify_citation`` MCP tool call via the
    injected `McpToolClient`, then mapping the tool's ``{exists, snippet}`` output
    onto a `CitationCheck`. Does no transport itself — the client is the seam, so
    this is unit-tested offline with a fake client.

    Fail-closed: a result whose ``exists`` is missing or non-boolean raises (it is
    never read as a pass), and a client error propagates unchanged.
    """

    def __init__(self, client: McpToolClient) -> None:
        self._client = client

    async def verify(self, source_id: str, claim: str) -> CitationCheck:
        """Verify `claim` against `source_id` via the ``verify_citation`` MCP tool.

        Returns the tool's `{exists, snippet}` as a `CitationCheck`. Raises on a
        malformed tool result; lets any transport error from the injected client
        propagate (fail-closed — a verification failure must never look like a
        verified claim to GRD-9).
        """
        result = await self._client.call_tool(
            _VERIFY_CITATION_TOOL,
            {"source_id": source_id, "claim": claim},
        )
        return _to_citation_check(result)


def _to_citation_check(result: dict[str, Any]) -> CitationCheck:
    """Map a ``verify_citation`` result mapping onto a `CitationCheck` (§6.2).

    The §6.2 output schema requires a boolean ``exists`` and an optional
    ``snippet`` (string or null). A missing or non-boolean ``exists`` is a contract
    breach and raises — coercing it would risk turning an unverifiable result into
    a false pass. ``snippet`` is normalised to ``None`` unless it is a string.
    """
    exists = result.get("exists")
    if not isinstance(exists, bool):
        raise ValueError("verify_citation result missing a boolean 'exists' field")
    snippet = result.get("snippet")
    return CitationCheck(exists=exists, snippet=snippet if isinstance(snippet, str) else None)


def make_citation_verifier(client: McpToolClient | None) -> CitationVerifier | None:
    """Return the live verifier when a client is configured, else ``None``.

    Config-gated, default OFF: the composition root passes the MCP client only when
    `Settings.citation_mcp_url` is set; with the default (``None``) this returns
    ``None`` and the citation guardrail is left to its stub/unwired default — the
    request path is unchanged. When a client is present, the returned
    `McpCitationVerifier` drops straight into `CitationGuardrail(verifier)`.
    """
    if client is None:
        return None
    return McpCitationVerifier(client)
