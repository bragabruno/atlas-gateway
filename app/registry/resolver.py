"""REG-3 — Prompt registry resolver: ref → version → validated render.

Implements the `resolve(prompt_ref) -> ResolvedPrompt` contract from
atlas-docs/03 §7.1. Given a `prompt_ref` of the form ``<name>@<semver>`` or
``<name>@production``, the resolver:

1. parses the ref and looks the version up THROUGH AN INJECTED REPOSITORY —
   ``@production`` resolves the version currently holding ``production`` status
   (the production pointer), any other ``@<semver>`` resolves that exact version
   by SemVer (atlas-docs/03 §7.1 contract rule 3);
2. validates the caller-supplied params against the version's JSON-Schema
   ``params_schema`` and FAILS FAST with an explicit error on any mismatch
   (missing required key, unknown key, or wrong primitive type) — never renders
   with bad input;
3. renders the version's Jinja2 ``template`` with the validated params and
   returns a `ResolvedPrompt`.

A capability adapter per ADR-016: the repository is a `Protocol` injected at
construction, so the resolver is exercised fully offline with an in-memory fake
(`tests/test_registry_resolver.py`) and a real asyncpg/ORM-backed repo can feed
the same shape later. Bare-alias resolution (no prompt, caller builds messages —
§7.1 contract rule 1) and the 60-second in-process cache (§7.1 rule 4) are the
request-path wiring's concern (REG-4) and are intentionally out of scope here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import jinja2
import jinja2.sandbox

#: A `prompt_ref` is ``<name>@<label>`` where label is a SemVer or ``production``
#: (atlas-docs/03 §7.1). The name is a slug; the label half is parsed separately.
_REF_RE = re.compile(r"^(?P<name>[^@]+)@(?P<label>.+)$")

#: The label selecting the version that currently holds ``production`` status.
PRODUCTION_LABEL = "production"

#: JSON-Schema primitive ``type`` names → the Python types we accept for them.
#: ``int`` is rejected for ``number`` only when it is a ``bool`` (a bool is an
#: ``int`` subclass in Python); ``integer`` likewise excludes ``bool``.
_JSON_TYPE_CHECKS: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


class PromptVersionRow(Protocol):
    """Structural view of a `prompt_versions` row the resolver needs.

    A real asyncpg/ORM row (or the in-memory fake the tests inject) satisfies
    this, so the resolver never imports a concrete persistence type. Fields are
    read-only so a frozen row adapter qualifies.
    """

    @property
    def id(self) -> str: ...

    @property
    def semver(self) -> str: ...

    @property
    def template(self) -> str: ...

    @property
    def params_schema(self) -> dict[str, object]: ...

    @property
    def model_alias(self) -> str | None: ...


class PromptRepository(Protocol):
    """Read port over `prompts`/`prompt_versions` the resolver depends on.

    Two lookups cover the §7.1 contract: the production pointer for
    ``@production`` and an exact-version lookup for ``@<semver>``. Both return
    ``None`` when nothing matches so the resolver can raise an explicit,
    typed error rather than guessing.
    """

    def get_production_version(self, name: str) -> PromptVersionRow | None:
        """Return the version of prompt `name` holding ``production`` status."""
        ...

    def get_version_by_semver(self, name: str, semver: str) -> PromptVersionRow | None:
        """Return the exact `semver` version of prompt `name`."""
        ...


@dataclass(frozen=True, slots=True)
class ResolvedPrompt:
    """The outcome of resolving a `prompt_ref` (atlas-docs/03 §7.1).

    `prompt_version_id` is the resolved `prompt_versions.id`; `rendered` is the
    template rendered with the validated params (ready to inject as the system
    message); `model_alias` is the alias the rendered prompt should run against
    (may be ``None`` when the version pins no alias). `params_schema` is carried
    through for callers that re-validate or log it.
    """

    prompt_version_id: str
    rendered: str
    model_alias: str | None
    params_schema: dict[str, object]


class PromptNotFoundError(Exception):
    """Raised when a `prompt_ref` resolves to no `prompt_versions` row."""

    def __init__(self, ref: str) -> None:
        self.ref = ref
        super().__init__(f"prompt ref not found: {ref}")


class InvalidPromptRefError(Exception):
    """Raised when a `prompt_ref` is not of the form ``<name>@<label>``."""

    def __init__(self, ref: str) -> None:
        self.ref = ref
        super().__init__(f"invalid prompt ref (expected '<name>@<semver|production>'): {ref}")


class ParamsValidationError(Exception):
    """Raised when caller params do not satisfy a version's `params_schema`.

    Fail-fast: the template is never rendered with invalid input. The message
    names exactly what was wrong (missing/unknown key or wrong type).
    """

    def __init__(self, ref: str, reason: str) -> None:
        self.ref = ref
        self.reason = reason
        super().__init__(f"params do not satisfy schema for {ref}: {reason}")


def _validate_params(
    ref: str,
    params_schema: dict[str, object],
    params: dict[str, object],
) -> None:
    """Validate `params` against the JSON-Schema-shaped `params_schema`.

    Supports the subset atlas-docs/03 §1.5 prompt templates use — an ``object``
    schema with ``properties`` (each carrying a primitive ``type``) and
    ``required`` — without pulling in a JSON-Schema dependency. An empty schema
    (``{}``) accepts any params. Raises `ParamsValidationError` on the first
    violation rather than rendering with bad input. ``additionalProperties:
    false`` (the default here) rejects any param not declared in ``properties``.
    """
    if not params_schema:
        return

    # No declared properties → accept nothing extra; only an empty schema
    # (handled above) is permissive. `object` values are narrowed explicitly so
    # the validator stays fully typed under pyright strict.
    raw_properties = params_schema.get("properties")
    properties: dict[str, object] = raw_properties if isinstance(raw_properties, dict) else {}

    raw_required = params_schema.get("required")
    required_list: list[object] = raw_required if isinstance(raw_required, list) else []
    required_keys: set[str] = {str(k) for k in required_list}

    missing: set[str] = required_keys - params.keys()
    if missing:
        raise ParamsValidationError(ref, f"missing required params: {sorted(missing)}")

    allow_extra = params_schema.get("additionalProperties") is True
    if not allow_extra:
        unknown: set[str] = params.keys() - properties.keys()
        if unknown:
            raise ParamsValidationError(ref, f"unknown params: {sorted(unknown)}")

    for key, value in params.items():
        spec = properties.get(key)
        if not isinstance(spec, dict):
            continue
        spec_typed: dict[str, object] = spec
        declared = spec_typed.get("type")
        if not isinstance(declared, str):
            continue
        accepted = _JSON_TYPE_CHECKS.get(declared)
        if accepted is None:
            continue
        # A bool is an int subclass; only ``boolean`` may accept a bool.
        if isinstance(value, bool) and declared != "boolean":
            raise ParamsValidationError(ref, f"param '{key}' must be {declared}")
        if not isinstance(value, accepted):
            raise ParamsValidationError(ref, f"param '{key}' must be {declared}")


def _parse_ref(ref: str) -> tuple[str, str]:
    """Split a `prompt_ref` into ``(name, label)`` or raise `InvalidPromptRefError`."""
    match = _REF_RE.match(ref)
    if match is None:
        raise InvalidPromptRefError(ref)
    name = match.group("name")
    label = match.group("label")
    if not name or not label:
        raise InvalidPromptRefError(ref)
    return name, label


class PromptResolver:
    """Resolves a `prompt_ref` to a rendered `ResolvedPrompt` (atlas-docs §7.1).

    The repository is injected (composition root / tests supply it), so this
    adapter does no DB access of its own. A sandboxed Jinja2 environment with
    ``StrictUndefined`` is used so an undefined template variable is a render
    error (fail fast) rather than a silently empty substitution.
    """

    def __init__(self, repository: PromptRepository) -> None:
        self._repository = repository
        # Sandboxed + StrictUndefined: templates come from the registry (trusted
        # authors) but rendering untrusted params, an undefined var must error
        # rather than render blank, and template features are sandboxed.
        self._env = jinja2.sandbox.SandboxedEnvironment(
            undefined=jinja2.StrictUndefined,
            autoescape=False,
        )

    def resolve(
        self,
        ref: str,
        params: dict[str, object] | None = None,
    ) -> ResolvedPrompt:
        """Resolve `ref` to a rendered `ResolvedPrompt`.

        `ref` is ``<name>@<semver>`` or ``<name>@production``. `params` are the
        caller-supplied template variables; they are validated against the
        version's `params_schema` BEFORE rendering and an invalid set raises
        `ParamsValidationError` (fail fast). A ref that matches no version raises
        `PromptNotFoundError`; a malformed ref raises `InvalidPromptRefError`.
        """
        params = params or {}
        name, label = _parse_ref(ref)

        if label == PRODUCTION_LABEL:
            version = self._repository.get_production_version(name)
        else:
            version = self._repository.get_version_by_semver(name, label)
        if version is None:
            raise PromptNotFoundError(ref)

        _validate_params(ref, version.params_schema, params)

        try:
            rendered = self._env.from_string(version.template).render(**params)
        except jinja2.UndefinedError as exc:
            # An undefined variable slipped past schema validation (e.g. the
            # schema declared no properties): surface it as a params error.
            raise ParamsValidationError(ref, str(exc)) from exc

        return ResolvedPrompt(
            prompt_version_id=version.id,
            rendered=rendered,
            model_alias=version.model_alias,
            params_schema=version.params_schema,
        )
