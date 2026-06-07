"""GRD-8 — Structured-output JSON-schema validation + bounded repair (POST).

A POST `Guardrail` (conforms to `app.guardrails.chain.Guardrail`) that validates
a provider's structured output against a JSON schema and, when it does not
conform, attempts a *bounded* auto-repair before giving up. Stdlib only (`json`
+ `re`) — no `jsonschema` dependency, no network: the schema check is an explicit
walk over the common JSON Schema keywords the gateway needs.

**Why bounded.** Auto-repair re-asks an upstream to fix its own malformed output.
Left unbounded that is an unbounded cost/latency loop, so the cap is hard:
`max_attempts` repair tries (default 1). The repair *itself* is injected as a
`RepairFn` callback — this module owns the validate→repair→re-validate loop and
the cap, not how a fix is produced (a later wiring ticket supplies a real
re-prompt; tests supply a deterministic fixer). The loop is:

1. validate the current candidate against the schema;
2. if valid, write the (possibly repaired) JSON back onto `ctx.result.content`
   and pass;
3. if invalid and attempts remain, call `repair(candidate, errors)` to get a new
   candidate and loop;
4. if invalid with no attempts left, raise `GuardrailRejection`.

**Fail-fast, no silent pass.** A response that never validates within the cap is
rejected with an explicit reason carrying the cap and the schema errors — it is
never forwarded unvalidated. A `None` from the repair callback ends the loop and
rejects (the repairer could not produce a candidate). The schema is validated for
structural sanity at construction so a malformed schema fails loudly, not at
request time. This is a standalone capability adapter; wiring into the chat path
is a later ticket. See GRD-8 + ADR-016.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Protocol, cast

from app.guardrails.chain import GuardrailContext, GuardrailPhase, GuardrailRejection

#: Default hard cap on repair attempts. One re-ask is the pragmatic default: an
#: output that two upstream tries cannot make conform is rejected, not retried
#: into an unbounded cost loop.
_DEFAULT_MAX_ATTEMPTS = 1

#: JSON Schema ``type`` names mapped to the Python types they accept. ``integer``
#: excludes ``bool`` (a Python ``bool`` is an ``int`` subclass) so a boolean is
#: never silently accepted where an integer is required.
_JSON_TYPES: Mapping[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "boolean": (bool,),
    "null": (type(None),),
}


class RepairFn(Protocol):
    """The injected bounded-repair callback.

    Given the current candidate string and the validation ``errors`` it failed
    with, returns a new candidate to re-validate, or ``None`` when it cannot
    produce one (which ends the loop and rejects). Async so a real implementation
    can re-prompt an upstream; tests supply a deterministic in-memory fixer.
    """

    async def __call__(self, candidate: str, errors: Sequence[str]) -> str | None:
        """Return a repaired candidate, or ``None`` if none can be produced."""
        ...


def _as_mapping(value: object) -> Mapping[str, object] | None:
    """Return ``value`` as a ``str``-keyed mapping, or ``None`` if it is not one.

    A JSON object — whether a sub-schema or a decoded response object — is a
    ``dict`` with ``str`` keys, so a mapping value *is* a ``Mapping[str, object]``.
    ``isinstance`` alone narrows only to ``Mapping[Unknown, Unknown]`` under
    pyright strict, so the one unavoidable ``cast`` is confined here (the same
    philosophy as ``app.limits._redis_typing``) instead of leaking ``Unknown``
    through the validator.
    """
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value)
    return None


def _type_matches(value: object, type_name: str) -> bool:
    """Return whether ``value`` matches a JSON Schema ``type`` name."""
    if type_name == "number":
        # JSON ``number`` accepts int or float, but not bool.
        return isinstance(value, int | float) and not isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    expected = _JSON_TYPES.get(type_name)
    if expected is None:
        return False
    if bool in expected:
        return isinstance(value, expected)
    # Exclude bool from int matches even when int is among the expected types.
    if int in expected:
        return isinstance(value, expected) and not isinstance(value, bool)
    return isinstance(value, expected)


def _validate(value: object, schema: Mapping[str, object], path: str = "$") -> list[str]:
    """Validate ``value`` against ``schema``; return a list of error strings.

    A supporting subset of JSON Schema: ``type``, ``enum``, ``required``,
    ``properties``, ``additionalProperties``, ``items``, ``minimum``/``maximum``,
    ``minLength``/``maxLength``, ``minItems``/``maxItems``. Unknown keywords are
    ignored (a superset schema still validates the parts we understand); we never
    silently accept a value that *violates* a keyword we do understand.
    """
    errors: list[str] = []

    type_kw = schema.get("type")
    if isinstance(type_kw, str) and not _type_matches(value, type_kw):
        errors.append(f"{path}: expected type {type_kw}, got {type(value).__name__}")
        # A wrong top-level type makes nested checks meaningless; stop here.
        return errors
    if isinstance(type_kw, list):
        type_names: list[object] = type_kw
        names = [t for t in type_names if isinstance(t, str)]
        if not any(_type_matches(value, name) for name in names):
            errors.append(f"{path}: expected one of types {names}, got {type(value).__name__}")
            return errors

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append(f"{path}: value not in enum")

    if isinstance(value, str):
        errors.extend(_validate_string(value, schema, path))
    elif isinstance(value, int | float) and not isinstance(value, bool):
        errors.extend(_validate_number(value, schema, path))
    else:
        as_object = _as_mapping(value)
        if as_object is not None:
            errors.extend(_validate_object(as_object, schema, path))
        elif isinstance(value, list):
            arr_value: list[object] = cast("list[object]", value)
            errors.extend(_validate_array(arr_value, schema, path))

    return errors


def _validate_string(value: str, schema: Mapping[str, object], path: str) -> list[str]:
    errors: list[str] = []
    min_length = schema.get("minLength")
    if isinstance(min_length, int) and len(value) < min_length:
        errors.append(f"{path}: string shorter than minLength {min_length}")
    max_length = schema.get("maxLength")
    if isinstance(max_length, int) and len(value) > max_length:
        errors.append(f"{path}: string longer than maxLength {max_length}")
    return errors


def _validate_number(value: float, schema: Mapping[str, object], path: str) -> list[str]:
    errors: list[str] = []
    minimum = schema.get("minimum")
    if isinstance(minimum, int | float) and not isinstance(minimum, bool) and value < minimum:
        errors.append(f"{path}: value below minimum {minimum}")
    maximum = schema.get("maximum")
    if isinstance(maximum, int | float) and not isinstance(maximum, bool) and value > maximum:
        errors.append(f"{path}: value above maximum {maximum}")
    return errors


def _validate_object(
    value: Mapping[str, object], schema: Mapping[str, object], path: str
) -> list[str]:
    errors: list[str] = []

    required = schema.get("required")
    if isinstance(required, list):
        required_fields: list[object] = required
        for field in required_fields:
            if isinstance(field, str) and field not in value:
                errors.append(f"{path}: missing required property '{field}'")

    props = _as_mapping(schema.get("properties")) or {}
    for key, sub_value in value.items():
        sub_schema = _as_mapping(props.get(key))
        if sub_schema is not None:
            errors.extend(_validate(sub_value, sub_schema, f"{path}.{key}"))

    additional = schema.get("additionalProperties")
    if additional is False:
        for key in value:
            if key not in props:
                errors.append(f"{path}: additional property '{key}' not allowed")

    return errors


def _validate_array(value: Sequence[object], schema: Mapping[str, object], path: str) -> list[str]:
    errors: list[str] = []
    min_items = schema.get("minItems")
    if isinstance(min_items, int) and len(value) < min_items:
        errors.append(f"{path}: array shorter than minItems {min_items}")
    max_items = schema.get("maxItems")
    if isinstance(max_items, int) and len(value) > max_items:
        errors.append(f"{path}: array longer than maxItems {max_items}")
    items = _as_mapping(schema.get("items"))
    if items is not None:
        for index, element in enumerate(value):
            errors.extend(_validate(element, items, f"{path}[{index}]"))
    return errors


def _parse_and_validate(candidate: str, schema: Mapping[str, object]) -> list[str]:
    """Parse ``candidate`` as JSON then validate it; both failures are errors."""
    try:
        parsed: object = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return [f"$: invalid JSON: {exc.msg} at line {exc.lineno} col {exc.colno}"]
    return _validate(parsed, schema)


class SchemaRepairGuardrail:
    """POST guardrail: validate structured output, repair within a hard cap.

    Conforms to the `Guardrail` protocol. The output is `ctx.result.content`
    (the provider's response body, expected to be a JSON document). On a
    validation failure the injected ``repair`` callback is invoked up to
    ``max_attempts`` times; the first candidate that validates is written back
    onto `ctx.result.content` and the check passes. If the cap is exhausted (or
    repair returns ``None``) the check raises `GuardrailRejection` — the
    unvalidated output is never forwarded.

    ``max_attempts`` is the number of *repair* tries (the initial validation does
    not count against it); ``0`` means validate-only (no repair). It must be
    non-negative — a negative cap is a configuration error and fails loudly.
    """

    name = "schema_repair"

    def __init__(
        self,
        *,
        schema: Mapping[str, object],
        repair: RepairFn | None = None,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        if max_attempts < 0:
            raise ValueError(f"max_attempts must be >= 0, got {max_attempts}")
        if max_attempts > 0 and repair is None:
            raise ValueError("a repair callback is required when max_attempts > 0")
        _assert_schema_shape(schema)
        self._schema = schema
        self._repair = repair
        self._max_attempts = max_attempts

    @property
    def max_attempts(self) -> int:
        """The hard cap on repair attempts."""
        return self._max_attempts

    async def check(self, ctx: GuardrailContext) -> None:
        """Validate (and bounded-repair) `ctx.result.content` against the schema."""
        if ctx.result is None:
            raise GuardrailRejection(
                guardrail=self.name,
                phase=GuardrailPhase.POST,
                reason="no provider result to validate (post-phase guardrail)",
            )

        candidate = ctx.result.content
        errors = _parse_and_validate(candidate, self._schema)
        if not errors:
            return

        attempts_used = 0
        while attempts_used < self._max_attempts:
            # _max_attempts > 0 guarantees a repair callback (enforced in __init__).
            repair = self._repair
            assert repair is not None  # noqa: S101 — invariant held by constructor
            repaired = await repair(candidate, tuple(errors))
            attempts_used += 1
            if repaired is None:
                break
            candidate = repaired
            errors = _parse_and_validate(candidate, self._schema)
            if not errors:
                # Persist the repaired output so downstream sees valid JSON only.
                ctx.result = ctx.result.model_copy(update={"content": candidate})
                return

        raise GuardrailRejection(
            guardrail=self.name,
            phase=GuardrailPhase.POST,
            reason=(
                f"structured output failed schema validation after "
                f"{attempts_used}/{self._max_attempts} repair attempts: "
                f"{'; '.join(errors)}"
            ),
        )


def _assert_schema_shape(schema: Mapping[str, object]) -> None:
    """Fail loudly at construction if the schema is structurally unusable.

    Not a full meta-schema check — just enough to catch an obviously malformed
    schema (a non-string/-list ``type``) at wiring time rather than silently
    passing every response at request time.
    """
    type_kw = schema.get("type")
    if type_kw is not None and not isinstance(type_kw, str | list):
        raise ValueError(f"schema 'type' must be a string or list, got {type(type_kw).__name__}")
