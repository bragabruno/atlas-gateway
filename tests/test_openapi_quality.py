"""GW-21 — OpenAPI quality invariants.

The drift guard in `test_openapi_contract.py` keeps the committed spec in sync
with the live `app.openapi()` — but a perfectly drift-free spec can still be
useless documentation (no descriptions, no tags, no security scheme). This
file asserts the structural invariants that make `/docs` documentation-grade
and the downstream codegen (FE-3, agent-runtime client) emit useful types.

Fully offline. See ADR-014 + ADR-016.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.main import app


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    return app.openapi()


def test_app_metadata_is_populated(spec: dict[str, Any]) -> None:
    info = spec["info"]
    assert info["title"] == "Atlas Gateway"
    assert info["version"], "version must be non-empty"
    assert len(info.get("description") or "") > 100, "description should be markdown-rich"
    assert info.get("contact"), "contact block missing"
    assert info.get("license"), "license_info missing"
    assert len(spec.get("servers", [])) >= 1, "at least one server URL must be advertised"


def test_bearer_security_scheme_present(spec: dict[str, Any]) -> None:
    schemes = spec.get("components", {}).get("securitySchemes", {})
    assert "BearerAuth" in schemes, "BearerAuth scheme missing (Authorize button won't render)"
    bearer = schemes["BearerAuth"]
    assert bearer.get("type") == "http"
    assert bearer.get("scheme") == "bearer"


def test_error_envelope_schema_present(spec: dict[str, Any]) -> None:
    schemas = spec.get("components", {}).get("schemas", {})
    assert "ErrorEnvelope" in schemas, "ErrorEnvelope schema missing (typed errors absent)"


def test_every_operation_has_summary_description_and_tag(spec: dict[str, Any]) -> None:
    missing: list[str] = []
    for path, methods in spec["paths"].items():
        for verb, op in methods.items():
            if verb in ("parameters", "summary", "description"):
                continue
            op_id = f"{verb.upper()} {path}"
            if not op.get("summary"):
                missing.append(f"{op_id}: summary")
            if not op.get("tags"):
                missing.append(f"{op_id}: tags")
    assert not missing, "operations missing metadata:\n  " + "\n  ".join(missing)


def test_protected_endpoints_advertise_bearer_security(spec: dict[str, Any]) -> None:
    """Every `/v1/*` operation must require BearerAuth so Swagger UI prompts for it."""
    missing: list[str] = []
    paths: dict[str, dict[str, dict[str, Any]]] = spec["paths"]
    for path, methods in paths.items():
        if not path.startswith("/v1/"):
            continue
        for verb, op in methods.items():
            if verb in ("parameters", "summary", "description"):
                continue
            sec: list[dict[str, list[str]]] = op.get("security") or []
            if not any("BearerAuth" in s for s in sec):
                missing.append(f"{verb.upper()} {path}")
    assert not missing, "v1 operations without BearerAuth:\n  " + "\n  ".join(missing)


def test_chat_completions_documents_sse_response(spec: dict[str, Any]) -> None:
    """The streaming contract is the trickiest one — assert both content types appear."""
    op = spec["paths"]["/v1/chat/completions"]["post"]
    content = op["responses"]["200"]["content"]
    assert "application/json" in content, "non-streaming JSON body missing from 200"
    assert "text/event-stream" in content, "SSE body missing from 200 (streaming undocumented)"


def test_error_responses_reference_error_envelope(spec: dict[str, Any]) -> None:
    """4xx/5xx responses we author should point at ErrorEnvelope so codegen types them.

    FastAPI auto-generates a 422 with `HTTPValidationError` for any operation that
    has a request body or query param — that's framework-owned and not part of our
    contract, so we exclude 422 from this check.
    """
    missing: list[str] = []
    for path, methods in spec["paths"].items():
        for verb, op in methods.items():
            if verb in ("parameters", "summary", "description"):
                continue
            for status, resp in op.get("responses", {}).items():
                if not status.startswith(("4", "5")) or status == "422":
                    continue
                schema_ref = resp.get("content", {}).get("application/json", {}).get("schema", {})
                ref = schema_ref.get("$ref", "")
                if "ErrorEnvelope" not in ref:
                    missing.append(f"{verb.upper()} {path} {status}: {ref or '<no schema>'}")
    assert not missing, "error responses not using ErrorEnvelope:\n  " + "\n  ".join(missing)
