"""REG-3 — prompt registry resolver: ref → version → validated render.

Pins the `resolve` contract (atlas-docs/03 §7.1) fully offline against an
in-memory `PromptRepository` fake (no live DB): ``@production`` resolves the
version holding production status (the pointer), ``@<semver>`` resolves that
exact version, invalid params fail fast with an explicit `ParamsValidationError`
(never rendering with bad input), the Jinja2 template renders correctly with
valid params, and malformed/unknown refs raise explicit typed errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from app.registry.resolver import (
    InvalidPromptRefError,
    ParamsValidationError,
    PromptNotFoundError,
    PromptResolver,
    PromptVersionRow,
    ResolvedPrompt,
)


@dataclass(frozen=True)
class _Version:
    """A minimal structural `PromptVersionRow` for tests."""

    id: str
    semver: str
    template: str
    params_schema: dict[str, object] = field(default_factory=dict)
    model_alias: str | None = None


@dataclass
class _FakeRepo:
    """In-memory `PromptRepository`: name → list of versions."""

    versions: dict[str, list[_Version]]

    def get_production_version(self, name: str) -> PromptVersionRow | None:
        # The store's production pointer is modelled by an explicit override so a
        # test can flip it; default is "no production version".
        prod_id = self.production_pointer.get(name)
        if prod_id is None:
            return None
        for v in self.versions.get(name, []):
            if v.id == prod_id:
                return v
        return None

    def get_version_by_semver(self, name: str, semver: str) -> PromptVersionRow | None:
        for v in self.versions.get(name, []):
            if v.semver == semver:
                return v
        return None

    production_pointer: dict[str, str] = field(default_factory=dict)


_OBJECT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"doc": {"type": "string"}},
    "required": ["doc"],
}


def _repo() -> _FakeRepo:
    repo = _FakeRepo(
        versions={
            "summarize-doc": [
                _Version(
                    id="v1",
                    semver="1.0.0",
                    template="Summarize: {{ doc }}",
                    params_schema=_OBJECT_SCHEMA,
                    model_alias="smart",
                ),
                _Version(
                    id="v2",
                    semver="2.0.0",
                    template="TL;DR of {{ doc }}",
                    params_schema=_OBJECT_SCHEMA,
                    model_alias="deep",
                ),
            ],
        },
        production_pointer={"summarize-doc": "v1"},
    )
    return repo


def test_production_label_resolves_the_production_version() -> None:
    resolver = PromptResolver(_repo())
    resolved = resolver.resolve("summarize-doc@production", {"doc": "hi"})
    assert isinstance(resolved, ResolvedPrompt)
    assert resolved.prompt_version_id == "v1"
    assert resolved.model_alias == "smart"
    assert resolved.rendered == "Summarize: hi"


def test_semver_label_resolves_that_exact_version() -> None:
    resolved = PromptResolver(_repo()).resolve("summarize-doc@2.0.0", {"doc": "x"})
    assert resolved.prompt_version_id == "v2"
    assert resolved.model_alias == "deep"
    assert resolved.rendered == "TL;DR of x"


def test_production_picks_pointer_not_highest_semver() -> None:
    """The production version is the one the pointer names, not the newest semver."""
    repo = _repo()
    repo.production_pointer["summarize-doc"] = "v2"
    resolved = PromptResolver(repo).resolve("summarize-doc@production", {"doc": "y"})
    assert resolved.prompt_version_id == "v2"


def test_template_renders_correctly_with_valid_params() -> None:
    repo = _FakeRepo(
        versions={
            "greet": [
                _Version(
                    id="g1",
                    semver="1.0.0",
                    template="Hello {{ name }}, you have {{ count }} messages.",
                    params_schema={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "count": {"type": "integer"},
                        },
                        "required": ["name", "count"],
                    },
                )
            ]
        },
        production_pointer={"greet": "g1"},
    )
    resolved = PromptResolver(repo).resolve("greet@production", {"name": "Ada", "count": 3})
    assert resolved.rendered == "Hello Ada, you have 3 messages."


def test_missing_required_param_fails_fast() -> None:
    with pytest.raises(ParamsValidationError) as exc:
        PromptResolver(_repo()).resolve("summarize-doc@1.0.0", {})
    assert "missing required params" in str(exc.value)
    assert "doc" in str(exc.value)


def test_unknown_param_fails_fast() -> None:
    with pytest.raises(ParamsValidationError) as exc:
        PromptResolver(_repo()).resolve("summarize-doc@1.0.0", {"doc": "x", "stray": "nope"})
    assert "unknown params" in str(exc.value)
    assert "stray" in str(exc.value)


def test_wrong_param_type_fails_fast() -> None:
    with pytest.raises(ParamsValidationError) as exc:
        PromptResolver(_repo()).resolve("summarize-doc@1.0.0", {"doc": 123})
    assert "must be string" in str(exc.value)


def test_bool_rejected_for_non_boolean_type() -> None:
    """A bool (an int subclass) is not accepted where a string/number is declared."""
    with pytest.raises(ParamsValidationError):
        PromptResolver(_repo()).resolve("summarize-doc@1.0.0", {"doc": True})


def test_empty_schema_accepts_any_declared_params() -> None:
    """An empty params_schema (`{}`) renders with whatever the template uses."""
    repo = _FakeRepo(
        versions={"free": [_Version(id="f1", semver="1.0.0", template="hi {{ who }}")]},
        production_pointer={"free": "f1"},
    )
    resolved = PromptResolver(repo).resolve("free@production", {"who": "there"})
    assert resolved.rendered == "hi there"


def test_undefined_template_var_fails_fast() -> None:
    """StrictUndefined: a var the template uses but params omit is a render error."""
    repo = _FakeRepo(
        versions={"free": [_Version(id="f1", semver="1.0.0", template="hi {{ who }}")]},
        production_pointer={"free": "f1"},
    )
    with pytest.raises(ParamsValidationError):
        PromptResolver(repo).resolve("free@production", {})


def test_no_production_version_raises_not_found() -> None:
    repo = _repo()
    repo.production_pointer.clear()
    with pytest.raises(PromptNotFoundError) as exc:
        PromptResolver(repo).resolve("summarize-doc@production", {"doc": "x"})
    assert exc.value.ref == "summarize-doc@production"


def test_unknown_semver_raises_not_found() -> None:
    with pytest.raises(PromptNotFoundError):
        PromptResolver(_repo()).resolve("summarize-doc@9.9.9", {"doc": "x"})


def test_unknown_prompt_name_raises_not_found() -> None:
    with pytest.raises(PromptNotFoundError):
        PromptResolver(_repo()).resolve("nope@production", {})


def test_malformed_ref_without_at_raises_invalid() -> None:
    with pytest.raises(InvalidPromptRefError) as exc:
        PromptResolver(_repo()).resolve("summarize-doc", {})
    assert exc.value.ref == "summarize-doc"


def test_malformed_ref_empty_label_raises_invalid() -> None:
    with pytest.raises(InvalidPromptRefError):
        PromptResolver(_repo()).resolve("summarize-doc@", {})


def test_resolver_reads_injected_repo_not_a_global() -> None:
    """The repository is injected: a custom prompt resolves without the seed."""
    repo = _FakeRepo(
        versions={"custom": [_Version(id="c1", semver="1.0.0", template="X")]},
        production_pointer={"custom": "c1"},
    )
    resolved = PromptResolver(repo).resolve("custom@production")
    assert resolved.prompt_version_id == "c1"
    assert resolved.rendered == "X"
    assert resolved.model_alias is None
