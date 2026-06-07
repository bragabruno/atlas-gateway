"""GRD-8 — structured-output JSON-schema validation + bounded repair (POST).

Pins the validate → bounded-repair → re-validate loop and its hard cap (fully
offline, stdlib only):

- a valid output passes untouched;
- a malformed output that the injected repairer fixes within the cap passes, and
  the repaired JSON is written back onto the result;
- a malformed output that cannot be fixed within the cap is rejected fail-fast
  with an explicit reason carrying the cap and the schema errors;
- the cap is enforced exactly (the repairer is called at most ``max_attempts``
  times);
- the validator catches the common keyword violations (type/required/enum);
- ``max_attempts=0`` is validate-only; a negative cap and a missing repairer fail
  loudly at construction;
- the post-phase contract (no result → reject) holds.

See GRD-8 + ADR-016.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from app.domain.messages import ChatResult, Usage
from app.guardrails.chain import Guardrail, GuardrailContext, GuardrailRejection
from app.guardrails.schema_repair import SchemaRepairGuardrail

# A representative object schema exercising type/required/enum/additionalProperties.
_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": ["name", "status"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "status": {"type": "string", "enum": ["ok", "error"]},
        "score": {"type": "integer", "minimum": 0, "maximum": 100},
    },
}


def _ctx(content: str) -> GuardrailContext:
    return GuardrailContext(
        tenant_id="tenant-a",
        model="mock",
        messages=[],
        result=ChatResult(model="mock", content=content, usage=Usage()),
    )


class _Repairer:
    """Records calls and returns a scripted sequence of repaired candidates."""

    def __init__(self, *responses: str | None) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def __call__(self, candidate: str, errors: Sequence[str]) -> str | None:
        self.calls.append((candidate, tuple(errors)))
        if not self._responses:
            return None
        return self._responses.pop(0)


# ── happy path ───────────────────────────────────────────────────────────────


async def test_valid_output_passes_untouched() -> None:
    valid = json.dumps({"name": "atlas", "status": "ok"})
    repairer = _Repairer()
    guard = SchemaRepairGuardrail(schema=_SCHEMA, repair=repairer, max_attempts=1)
    ctx = _ctx(valid)
    await guard.check(ctx)
    assert repairer.calls == []  # never needed
    assert ctx.result is not None
    assert ctx.result.content == valid


# ── repaired within the cap ──────────────────────────────────────────────────


async def test_malformed_repaired_within_cap_passes() -> None:
    fixed = json.dumps({"name": "atlas", "status": "ok"})
    repairer = _Repairer(fixed)  # one repair attempt succeeds
    guard = SchemaRepairGuardrail(schema=_SCHEMA, repair=repairer, max_attempts=2)
    ctx = _ctx("{not valid json")
    await guard.check(ctx)
    assert len(repairer.calls) == 1
    assert ctx.result is not None
    assert ctx.result.content == fixed  # repaired JSON written back


async def test_repair_receives_validation_errors() -> None:
    # First candidate is valid JSON but violates the schema (missing 'status').
    bad = json.dumps({"name": "atlas"})
    fixed = json.dumps({"name": "atlas", "status": "ok"})
    repairer = _Repairer(fixed)
    guard = SchemaRepairGuardrail(schema=_SCHEMA, repair=repairer, max_attempts=1)
    await guard.check(_ctx(bad))
    assert len(repairer.calls) == 1
    _, errors = repairer.calls[0]
    assert any("status" in e for e in errors)


# ── cap enforcement / rejection ──────────────────────────────────────────────


async def test_unfixable_output_rejected_after_cap() -> None:
    # Repairer keeps returning malformed candidates → cap exhausts → reject.
    repairer = _Repairer("still bad", "still bad", "still bad")
    guard = SchemaRepairGuardrail(schema=_SCHEMA, repair=repairer, max_attempts=2)
    with pytest.raises(GuardrailRejection) as exc_info:
        await guard.check(_ctx("nope"))
    rejection = exc_info.value
    assert rejection.guardrail == "schema_repair"
    assert "2/2 repair attempts" in rejection.reason
    # Cap is hard: exactly max_attempts repair calls, never more.
    assert len(repairer.calls) == 2


async def test_repair_returning_none_ends_loop_and_rejects() -> None:
    repairer = _Repairer(None)  # repairer gives up immediately
    guard = SchemaRepairGuardrail(schema=_SCHEMA, repair=repairer, max_attempts=3)
    with pytest.raises(GuardrailRejection):
        await guard.check(_ctx("bad"))
    assert len(repairer.calls) == 1  # stopped at the None, did not exhaust the cap


async def test_validate_only_rejects_without_repair() -> None:
    # max_attempts=0 → no repairer needed; an invalid output is rejected outright.
    guard = SchemaRepairGuardrail(schema=_SCHEMA, max_attempts=0)
    with pytest.raises(GuardrailRejection) as exc_info:
        await guard.check(_ctx("bad"))
    assert "0/0 repair attempts" in exc_info.value.reason


async def test_validate_only_passes_valid_output() -> None:
    guard = SchemaRepairGuardrail(schema=_SCHEMA, max_attempts=0)
    await guard.check(_ctx(json.dumps({"name": "atlas", "status": "ok"})))


# ── validator coverage ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "atlas"},  # missing required 'status'
        {"name": "atlas", "status": "maybe"},  # enum violation
        {"name": "atlas", "status": "ok", "extra": 1},  # additionalProperties
        {"name": "atlas", "status": "ok", "score": 200},  # maximum violation
        {"name": "", "status": "ok"},  # minLength violation
        {"name": 5, "status": "ok"},  # type violation
    ],
)
async def test_schema_violations_are_caught(payload: dict[str, object]) -> None:
    guard = SchemaRepairGuardrail(schema=_SCHEMA, max_attempts=0)
    with pytest.raises(GuardrailRejection):
        await guard.check(_ctx(json.dumps(payload)))


async def test_integer_field_rejects_bool() -> None:
    # A Python bool must not satisfy an integer field.
    guard = SchemaRepairGuardrail(schema=_SCHEMA, max_attempts=0)
    with pytest.raises(GuardrailRejection):
        await guard.check(_ctx(json.dumps({"name": "atlas", "status": "ok", "score": True})))


async def test_array_schema_validates_items() -> None:
    schema: dict[str, object] = {
        "type": "array",
        "minItems": 1,
        "items": {"type": "integer"},
    }
    guard = SchemaRepairGuardrail(schema=schema, max_attempts=0)
    await guard.check(_ctx("[1, 2, 3]"))
    with pytest.raises(GuardrailRejection):
        await guard.check(_ctx('[1, "two"]'))


# ── construction + contract ──────────────────────────────────────────────────


def test_negative_cap_fails_loudly() -> None:
    with pytest.raises(ValueError, match="max_attempts must be >= 0"):
        SchemaRepairGuardrail(schema=_SCHEMA, max_attempts=-1)


def test_repair_required_when_cap_positive() -> None:
    with pytest.raises(ValueError, match="repair callback is required"):
        SchemaRepairGuardrail(schema=_SCHEMA, max_attempts=1)


def test_malformed_schema_type_fails_loudly() -> None:
    with pytest.raises(ValueError, match="schema 'type' must be"):
        SchemaRepairGuardrail(schema={"type": 123}, max_attempts=0)


async def test_no_result_is_rejected() -> None:
    guard = SchemaRepairGuardrail(schema=_SCHEMA, max_attempts=0)
    ctx = GuardrailContext(tenant_id="t", model="mock", messages=[], result=None)
    with pytest.raises(GuardrailRejection, match="no provider result"):
        await guard.check(ctx)


def test_max_attempts_property() -> None:
    repairer = _Repairer()
    assert SchemaRepairGuardrail(schema=_SCHEMA, repair=repairer, max_attempts=3).max_attempts == 3


async def test_conforms_to_guardrail_protocol() -> None:
    assert isinstance(SchemaRepairGuardrail(schema=_SCHEMA, max_attempts=0), Guardrail)
