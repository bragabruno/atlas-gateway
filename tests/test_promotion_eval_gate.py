"""REG-13 — promotion gated on evals (server-side eval-gate enforcement).

Pins (fully offline, no live DB / CI) that ``candidate → production`` is allowed
only when an eval-gate-green signal exists for that exact version, enforced
server-side in the gateway registry:

- promotion to production is REJECTED when the gate is not green (fail-closed),
  and nothing is written;
- promotion to production is ALLOWED when the gate is green;
- enforcement is server-side — it lives in `PromotionService`, queries the
  injected `EvalGateChecker`, and blocks even though the lifecycle transition
  itself is legal;
- with no gate injected the fail-closed default blocks all production promotions;
- non-production moves (draft → candidate, retire) are NOT gated;
- rollback to a version is also gated (flipping the pointer is a promotion).

See REG-13 + ADR-015 (eval gate) + ADR-016.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from app.registry.promotion import (
    EvalGateNotGreenError,
    PromotionService,
)
from app.repositories.tables import PromptStatusEnum


@dataclass
class _Row:
    """A prompt_versions row in the fake store."""

    id: str
    prompt_id: str
    status: PromptStatusEnum


@dataclass
class _FakeStore:
    """In-memory `PromotionStore` over `_Row`s (no live DB)."""

    rows: dict[str, _Row] = field(default_factory=dict)

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


@dataclass
class _FakeGate:
    """In-memory `EvalGateChecker`: green only for explicitly green version ids."""

    green: set[str] = field(default_factory=set)
    calls: list[str] = field(default_factory=list)

    def is_green(self, version_id: str) -> bool:
        self.calls.append(version_id)
        return version_id in self.green


def _store(*rows: _Row) -> _FakeStore:
    return _FakeStore(rows={r.id: r for r in rows})


def _row(id: str, status: PromptStatusEnum, *, prompt_id: str = "p") -> _Row:
    return _Row(id=id, prompt_id=prompt_id, status=status)


# ── rejected when evals fail ───────────────────────────────────────────────────


def test_promotion_rejected_when_gate_not_green() -> None:
    store = _store(_row("v1", PromptStatusEnum.candidate))
    gate = _FakeGate(green=set())  # v1 is NOT green
    svc = PromotionService(store, eval_gate=gate)
    with pytest.raises(EvalGateNotGreenError) as exc:
        svc.transition("v1", PromptStatusEnum.production)
    assert exc.value.version_id == "v1"
    # Nothing was written — the version stays candidate.
    assert store.rows["v1"].status == PromptStatusEnum.candidate


def test_promote_helper_rejected_when_gate_not_green() -> None:
    store = _store(_row("v1", PromptStatusEnum.candidate))
    svc = PromotionService(store, eval_gate=_FakeGate(green=set()))
    with pytest.raises(EvalGateNotGreenError):
        svc.promote("v1")
    assert store.rows["v1"].status == PromptStatusEnum.candidate


def test_failed_gate_does_not_demote_prior_production() -> None:
    """A blocked promotion leaves the existing production pointer intact."""
    store = _store(
        _row("old", PromptStatusEnum.production),
        _row("new", PromptStatusEnum.candidate),
    )
    svc = PromotionService(store, eval_gate=_FakeGate(green=set()))
    with pytest.raises(EvalGateNotGreenError):
        svc.transition("new", PromptStatusEnum.production)
    assert store.rows["old"].status == PromptStatusEnum.production
    assert store.rows["new"].status == PromptStatusEnum.candidate


# ── allowed when green ─────────────────────────────────────────────────────────


def test_promotion_allowed_when_gate_green() -> None:
    store = _store(_row("v1", PromptStatusEnum.candidate))
    gate = _FakeGate(green={"v1"})
    svc = PromotionService(store, eval_gate=gate)
    assert svc.transition("v1", PromptStatusEnum.production) == PromptStatusEnum.production
    assert store.rows["v1"].status == PromptStatusEnum.production


def test_promote_helper_allowed_when_gate_green() -> None:
    store = _store(_row("v1", PromptStatusEnum.candidate))
    svc = PromotionService(store, eval_gate=_FakeGate(green={"v1"}))
    assert svc.promote("v1") == PromptStatusEnum.production
    assert store.rows["v1"].status == PromptStatusEnum.production


# ── enforcement is server-side ─────────────────────────────────────────────────


def test_gate_is_consulted_server_side_for_the_exact_version() -> None:
    store = _store(_row("v1", PromptStatusEnum.candidate))
    gate = _FakeGate(green={"v1"})
    PromotionService(store, eval_gate=gate).transition("v1", PromptStatusEnum.production)
    # The service itself queried the gate for the promoted version id.
    assert "v1" in gate.calls


def test_green_for_a_different_version_does_not_authorize_this_one() -> None:
    """The gate must be green for the *exact* version, not merely some version."""
    store = _store(_row("v1", PromptStatusEnum.candidate))
    svc = PromotionService(store, eval_gate=_FakeGate(green={"other"}))
    with pytest.raises(EvalGateNotGreenError):
        svc.transition("v1", PromptStatusEnum.production)


def test_default_gate_is_fail_closed() -> None:
    """No injected gate → production promotion is impossible (fail-closed default)."""
    store = _store(_row("v1", PromptStatusEnum.candidate))
    svc = PromotionService(store)  # no eval_gate
    with pytest.raises(EvalGateNotGreenError):
        svc.transition("v1", PromptStatusEnum.production)
    assert store.rows["v1"].status == PromptStatusEnum.candidate


# ── non-production moves are not gated ──────────────────────────────────────────


def test_draft_to_candidate_is_not_gated() -> None:
    store = _store(_row("v1", PromptStatusEnum.draft))
    gate = _FakeGate(green=set())  # not green anywhere
    svc = PromotionService(store, eval_gate=gate)
    assert svc.transition("v1", PromptStatusEnum.candidate) == PromptStatusEnum.candidate
    # The gate is never consulted for a non-production move.
    assert gate.calls == []


def test_retire_is_not_gated() -> None:
    store = _store(_row("v1", PromptStatusEnum.production))
    svc = PromotionService(store, eval_gate=_FakeGate(green=set()))
    assert svc.retire("v1") == PromptStatusEnum.retired


# ── rollback is gated too ──────────────────────────────────────────────────────


def test_rollback_rejected_when_target_gate_not_green() -> None:
    store = _store(
        _row("prior", PromptStatusEnum.candidate),
        _row("current", PromptStatusEnum.production),
    )
    svc = PromotionService(store, eval_gate=_FakeGate(green=set()))
    with pytest.raises(EvalGateNotGreenError):
        svc.rollback("p", "prior")
    # Pointer untouched on a blocked rollback.
    assert store.rows["current"].status == PromptStatusEnum.production
    assert store.rows["prior"].status == PromptStatusEnum.candidate


def test_rollback_allowed_when_target_gate_green() -> None:
    store = _store(
        _row("prior", PromptStatusEnum.candidate),
        _row("current", PromptStatusEnum.production),
    )
    svc = PromotionService(store, eval_gate=_FakeGate(green={"prior"}))
    assert svc.rollback("p", "prior") == PromptStatusEnum.production
    assert store.rows["prior"].status == PromptStatusEnum.production
    assert store.rows["current"].status == PromptStatusEnum.candidate


def test_rollback_to_current_production_is_noop_and_ungated() -> None:
    """A no-op rollback to the live version does not need to re-check the gate."""
    store = _store(_row("v1", PromptStatusEnum.production))
    gate = _FakeGate(green=set())
    svc = PromotionService(store, eval_gate=gate)
    assert svc.rollback("p", "v1") == PromptStatusEnum.production
    assert gate.calls == []
