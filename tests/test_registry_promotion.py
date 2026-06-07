"""REG-5 — prompt-version promotion state machine + instant rollback.

Pins the lifecycle (atlas-docs/03 §1.5: draft → candidate → production; any →
retired) and the production pointer fully offline against an in-memory
`PromotionStore` fake (no live DB): legal transitions advance, illegal ones
raise `IllegalTransitionError`, promoting to production demotes the prior
production version (single-production invariant), and rollback flips the pointer
back so the next resolve uses the prior version.

These REG-5 lifecycle tests inject an always-green eval gate so they exercise the
state machine in isolation; the REG-13 eval-gate enforcement (production
promotion blocked unless the gate is green) is pinned in
`tests/test_promotion_eval_gate.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from app.registry.promotion import (
    IllegalTransitionError,
    PromotionService,
    UnknownVersionError,
)
from app.registry.resolver import PromptResolver, PromptVersionRow
from app.repositories.tables import PromptStatusEnum


class _GreenGate:
    """Always-green eval gate so REG-5 tests focus on lifecycle, not REG-13."""

    def is_green(self, version_id: str) -> bool:  # noqa: ARG002 — always green for REG-5
        return True


def _svc(store: _FakeStore) -> PromotionService:
    """A `PromotionService` over `store` with the always-green REG-5 gate."""
    return PromotionService(store, eval_gate=_GreenGate())


@dataclass
class _StoreRow:
    """A prompt_versions row in the fake store/repo."""

    id: str
    prompt_id: str
    semver: str
    template: str
    status: PromptStatusEnum


@dataclass
class _FakeStore:
    """In-memory `PromotionStore` (and `PromptRepository`) over `_StoreRow`s.

    Doubles as a resolver repository so a single fake proves the end-to-end
    rollback property (promotion flips the pointer; the next resolve follows it).
    """

    rows: dict[str, _StoreRow] = field(default_factory=dict)

    # --- PromotionStore ---
    def get_status(self, version_id: str) -> PromptStatusEnum | None:
        row = self.rows.get(version_id)
        return row.status if row else None

    def get_prompt_id(self, version_id: str) -> str | None:
        row = self.rows.get(version_id)
        return row.prompt_id if row else None

    def get_production_version_id(self, prompt_id: str) -> str | None:
        for row in self.rows.values():
            if row.prompt_id == prompt_id and row.status == PromptStatusEnum.production:
                return row.id
        return None

    def set_status(self, version_id: str, status: PromptStatusEnum) -> None:
        self.rows[version_id].status = status

    # --- PromptRepository (for the end-to-end rollback assertion) ---
    def _by_name(self, name: str) -> list[_StoreRow]:
        return [r for r in self.rows.values() if r.prompt_id == name]

    def get_production_version(self, name: str) -> PromptVersionRow | None:
        for r in self._by_name(name):
            if r.status == PromptStatusEnum.production:
                return _ResolverRow(r)
        return None

    def get_version_by_semver(self, name: str, semver: str) -> PromptVersionRow | None:
        for r in self._by_name(name):
            if r.semver == semver:
                return _ResolverRow(r)
        return None


@dataclass(frozen=True)
class _ResolverRow:
    """Adapts a `_StoreRow` to the resolver's `PromptVersionRow` shape."""

    _row: _StoreRow

    @property
    def id(self) -> str:
        return self._row.id

    @property
    def semver(self) -> str:
        return self._row.semver

    @property
    def template(self) -> str:
        return self._row.template

    @property
    def params_schema(self) -> dict[str, object]:
        return {}

    @property
    def model_alias(self) -> str | None:
        return None


def _store(*rows: _StoreRow) -> _FakeStore:
    return _FakeStore(rows={r.id: r for r in rows})


def _row(
    id: str, status: PromptStatusEnum, *, prompt_id: str = "p", semver: str = "1.0.0"
) -> _StoreRow:
    return _StoreRow(id=id, prompt_id=prompt_id, semver=semver, template=f"t-{id}", status=status)


# --- valid transitions ---


def test_draft_to_candidate_is_allowed() -> None:
    store = _store(_row("v1", PromptStatusEnum.draft))
    svc = _svc(store)
    assert svc.transition("v1", PromptStatusEnum.candidate) == PromptStatusEnum.candidate
    assert store.rows["v1"].status == PromptStatusEnum.candidate


def test_candidate_to_production_is_allowed() -> None:
    store = _store(_row("v1", PromptStatusEnum.candidate))
    _svc(store).transition("v1", PromptStatusEnum.production)
    assert store.rows["v1"].status == PromptStatusEnum.production


def test_promote_walks_one_step_at_a_time() -> None:
    store = _store(_row("v1", PromptStatusEnum.draft))
    svc = _svc(store)
    assert svc.promote("v1") == PromptStatusEnum.candidate
    assert svc.promote("v1") == PromptStatusEnum.production


def test_any_state_can_retire() -> None:
    store = _store(
        _row("d", PromptStatusEnum.draft, semver="1.0.0"),
        _row("c", PromptStatusEnum.candidate, semver="2.0.0"),
        _row("p", PromptStatusEnum.production, semver="3.0.0"),
    )
    svc = _svc(store)
    assert svc.retire("d") == PromptStatusEnum.retired
    assert svc.retire("c") == PromptStatusEnum.retired
    assert svc.retire("p") == PromptStatusEnum.retired


# --- invalid transitions ---


def test_draft_to_production_is_rejected() -> None:
    store = _store(_row("v1", PromptStatusEnum.draft))
    with pytest.raises(IllegalTransitionError) as exc:
        _svc(store).transition("v1", PromptStatusEnum.production)
    assert exc.value.current == PromptStatusEnum.draft
    assert exc.value.target == PromptStatusEnum.production
    assert store.rows["v1"].status == PromptStatusEnum.draft  # unchanged


def test_candidate_to_draft_is_rejected() -> None:
    store = _store(_row("v1", PromptStatusEnum.candidate))
    with pytest.raises(IllegalTransitionError):
        _svc(store).transition("v1", PromptStatusEnum.draft)


def test_retired_is_terminal() -> None:
    store = _store(_row("v1", PromptStatusEnum.retired))
    svc = _svc(store)
    for target in (
        PromptStatusEnum.draft,
        PromptStatusEnum.candidate,
        PromptStatusEnum.production,
        PromptStatusEnum.retired,
    ):
        with pytest.raises(IllegalTransitionError):
            svc.transition("v1", target)


def test_noop_transition_is_rejected() -> None:
    store = _store(_row("v1", PromptStatusEnum.candidate))
    with pytest.raises(IllegalTransitionError):
        _svc(store).transition("v1", PromptStatusEnum.candidate)


def test_promote_from_production_has_no_forward_edge() -> None:
    store = _store(_row("v1", PromptStatusEnum.production))
    with pytest.raises(IllegalTransitionError):
        _svc(store).promote("v1")


def test_unknown_version_fails_fast() -> None:
    svc = _svc(_store())
    with pytest.raises(UnknownVersionError) as exc:
        svc.transition("ghost", PromptStatusEnum.candidate)
    assert exc.value.version_id == "ghost"


# --- production pointer + rollback ---


def test_promoting_to_production_demotes_prior_production() -> None:
    store = _store(
        _row("old", PromptStatusEnum.production, semver="1.0.0"),
        _row("new", PromptStatusEnum.candidate, semver="2.0.0"),
    )
    _svc(store).transition("new", PromptStatusEnum.production)
    assert store.rows["new"].status == PromptStatusEnum.production
    assert store.rows["old"].status == PromptStatusEnum.candidate
    # At most one production version per prompt.
    prod = [r for r in store.rows.values() if r.status == PromptStatusEnum.production]
    assert [r.id for r in prod] == ["new"]


def test_other_prompts_production_is_untouched() -> None:
    """Promoting one prompt's version does not demote another prompt's production."""
    store = _store(
        _row("a-prod", PromptStatusEnum.production, prompt_id="A", semver="1.0.0"),
        _row("b-cand", PromptStatusEnum.candidate, prompt_id="B", semver="1.0.0"),
    )
    _svc(store).transition("b-cand", PromptStatusEnum.production)
    assert store.rows["a-prod"].status == PromptStatusEnum.production


def test_rollback_flips_pointer_and_demotes_current() -> None:
    store = _store(
        _row("prior", PromptStatusEnum.candidate, semver="1.0.0"),
        _row("current", PromptStatusEnum.production, semver="2.0.0"),
    )
    svc = _svc(store)
    assert svc.rollback("p", "prior") == PromptStatusEnum.production
    assert store.rows["prior"].status == PromptStatusEnum.production
    assert store.rows["current"].status == PromptStatusEnum.candidate


def test_rollback_then_resolve_uses_prior_version() -> None:
    """End-to-end: rollback flips the pointer; the next resolve follows it."""
    store = _store(
        _row("prior", PromptStatusEnum.candidate, semver="1.0.0"),
        _row("current", PromptStatusEnum.production, semver="2.0.0"),
    )
    resolver = PromptResolver(store)

    # Before rollback, production resolves to the current version.
    assert resolver.resolve("p@production").prompt_version_id == "current"

    _svc(store).rollback("p", "prior")

    # After the pointer flip, the same resolve returns the prior version.
    assert resolver.resolve("p@production").prompt_version_id == "prior"


def test_rollback_to_current_production_is_noop() -> None:
    store = _store(_row("v1", PromptStatusEnum.production))
    assert _svc(store).rollback("p", "v1") == PromptStatusEnum.production
    assert store.rows["v1"].status == PromptStatusEnum.production


def test_rollback_to_retired_version_is_rejected() -> None:
    store = _store(
        _row("retired", PromptStatusEnum.retired, semver="1.0.0"),
        _row("current", PromptStatusEnum.production, semver="2.0.0"),
    )
    with pytest.raises(IllegalTransitionError):
        _svc(store).rollback("p", "retired")
    assert store.rows["current"].status == PromptStatusEnum.production


def test_rollback_unknown_version_fails_fast() -> None:
    store = _store(_row("current", PromptStatusEnum.production))
    with pytest.raises(UnknownVersionError):
        _svc(store).rollback("p", "ghost")
