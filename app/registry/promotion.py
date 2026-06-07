"""REG-5 — Prompt-version promotion state machine + instant rollback.
REG-13 — Promotion gated on evals (server-side eval-gate enforcement).

Walks a `prompt_versions` row through the lifecycle from atlas-docs/03 §1.5 —
``draft`` → ``candidate`` → ``production``, and any state → ``retired`` — and
maintains the single-``production``-version-per-prompt invariant ("the
production pointer"): promoting a version to ``production`` demotes whatever
version was production before it. Rollback is an instant pointer flip — promote a
prior (still ``candidate``-eligible) version straight back to ``production`` —
so the next `app.registry.resolver` resolve of ``<name>@production`` returns the
rolled-back version with no redeploy.

REG-13 adds the **eval gate**: ``candidate → production`` is allowed only when an
eval-gate-green signal exists for that exact version. The gate is the regression
gate from atlas-docs/03 §1.6 / ADR-015 (``eval_runs`` / ``eval_results.passed``,
the REG-11 ``gate.py`` verdict), surfaced here through an injected
:class:`EvalGateChecker`. Enforcement is **server-side** — it lives in this
gateway-owned service, not in CI or a client — so a green CI check is necessary
but the production pointer never flips without the gateway itself re-confirming
the gate. A version with no green eval result, or a failing one, cannot be
promoted to production through any path (``transition``, ``promote``, or
``rollback``). Retirement and the ``draft → candidate`` step are not gated — only
the move that puts a version in front of live traffic.

A capability module per ADR-016: the status store and the eval-gate checker are
`Protocol`s injected at construction, so the state machine is exercised fully
offline with in-memory fakes (`tests/test_registry_promotion.py`,
`tests/test_promotion_eval_gate.py`) and real asyncpg/ORM-backed implementations
can satisfy the same shapes later. Transition rules are enforced here (the DB
holds the status column but does not police legal transitions), failing fast on
any illegal move rather than silently writing an invalid state.
"""

from __future__ import annotations

from typing import Protocol

from app.repositories.tables import PromptStatusEnum

#: Legal forward transitions (atlas-docs/03 §1.5). ``retired`` is reachable from
#: ANY non-retired state and is terminal, so it is handled separately rather
#: than listed per source. Promoting to a state a version already holds is not a
#: transition (rejected) — callers must move forward or retire.
_ALLOWED_TRANSITIONS: dict[PromptStatusEnum, frozenset[PromptStatusEnum]] = {
    PromptStatusEnum.draft: frozenset({PromptStatusEnum.candidate}),
    PromptStatusEnum.candidate: frozenset({PromptStatusEnum.production}),
    PromptStatusEnum.production: frozenset(),
    PromptStatusEnum.retired: frozenset(),
}

#: Production rollback demotes the outgoing production version to this state, so
#: it stays promotable again (a later flip back) without going through ``draft``.
_DEMOTED_STATUS = PromptStatusEnum.candidate


class PromotionStore(Protocol):
    """Write port over `prompt_versions` status the state machine depends on.

    Kept minimal and framework-neutral: read a version's status and owning
    prompt, find the prompt's current production version (the pointer), and set
    a version's status. The in-memory fake the tests inject and a real
    asyncpg/ORM-backed store both satisfy this. ``None`` returns mean "no such
    row", so the state machine raises an explicit error rather than guessing.
    """

    def get_status(self, version_id: str) -> PromptStatusEnum | None:
        """Return the current status of `version_id`, or ``None`` if unknown."""
        ...

    def get_prompt_id(self, version_id: str) -> str | None:
        """Return the `prompt_id` owning `version_id`, or ``None`` if unknown."""
        ...

    def get_production_version_id(self, prompt_id: str) -> str | None:
        """Return the prompt's current ``production`` version id, or ``None``."""
        ...

    def set_status(self, version_id: str, status: PromptStatusEnum) -> None:
        """Persist `status` for `version_id`."""
        ...


class EvalGateChecker(Protocol):
    """Read port over the eval-gate verdict for a version (REG-13).

    ``is_green(version_id)`` returns ``True`` only when the eval gate has passed
    for that exact version — i.e. there is a green eval run (atlas-docs/03 §1.6
    ``eval_results.passed``, the REG-11 ``gate.py`` verdict). The state machine
    calls this server-side before any promotion to ``production``. The in-memory
    fake the tests inject and a real ``eval_runs``/``eval_results``-backed checker
    both satisfy this. Implementations must default to *not* green for an unknown
    version (fail-closed): the absence of a passing eval is not a pass.
    """

    def is_green(self, version_id: str) -> bool:
        """Return ``True`` iff `version_id` has a passing eval-gate result."""
        ...


class _NoEvalGate:
    """Default checker used when REG-5 callers inject no gate (fail-closed).

    Promotion to ``production`` is impossible without an explicit
    :class:`EvalGateChecker` — an un-gated `PromotionService` may walk
    ``draft → candidate``, retire, and roll back to an *already-eligible* prior
    version, but it can never put a fresh version in front of live traffic. This
    keeps REG-13 enforcement on by default rather than opt-in.
    """

    def is_green(self, version_id: str) -> bool:  # noqa: ARG002 — fixed False by contract
        return False


class UnknownVersionError(Exception):
    """Raised when a version id is not present in the store."""

    def __init__(self, version_id: str) -> None:
        self.version_id = version_id
        super().__init__(f"unknown prompt version: {version_id}")


class IllegalTransitionError(Exception):
    """Raised when a requested status transition is not allowed (§1.5)."""

    def __init__(
        self, version_id: str, current: PromptStatusEnum, target: PromptStatusEnum
    ) -> None:
        self.version_id = version_id
        self.current = current
        self.target = target
        super().__init__(f"illegal transition for {version_id}: {current.value} -> {target.value}")


class EvalGateNotGreenError(Exception):
    """Raised when a version is promoted to ``production`` without a green eval gate (REG-13).

    Distinct from `IllegalTransitionError`: the transition itself is *legal*
    (``candidate → production``, or a rollback target), but the server-side eval
    gate has not passed for `version_id`, so the production pointer must not flip.
    """

    def __init__(self, version_id: str) -> None:
        self.version_id = version_id
        super().__init__(f"eval gate is not green for version: {version_id}")


def _is_allowed(current: PromptStatusEnum, target: PromptStatusEnum) -> bool:
    """Whether `current` → `target` is a legal lifecycle move (§1.5).

    ``retired`` is reachable from any non-retired state; otherwise only the
    forward edges in `_ALLOWED_TRANSITIONS` are legal. A no-op (target == current)
    is not a transition.
    """
    if target == current:
        return False
    if target == PromptStatusEnum.retired:
        return current != PromptStatusEnum.retired
    return target in _ALLOWED_TRANSITIONS[current]


class PromotionService:
    """Enforces the prompt-version lifecycle and the production pointer (REG-5/REG-13).

    The status store and the eval-gate checker are injected (composition root /
    tests supply them), so this service does no DB access of its own. Every state
    change is validated before it is written; promoting to ``production``
    atomically demotes the previous production version so at most one version is
    ever production. Per REG-13 any move *into* ``production`` is additionally
    gated server-side on `eval_gate.is_green(version_id)`: with no gate injected a
    fail-closed default (`_NoEvalGate`) blocks all production promotions.
    """

    def __init__(
        self,
        store: PromotionStore,
        *,
        eval_gate: EvalGateChecker | None = None,
    ) -> None:
        self._store = store
        self._eval_gate: EvalGateChecker = eval_gate if eval_gate is not None else _NoEvalGate()

    def _require_status(self, version_id: str) -> PromptStatusEnum:
        """Return `version_id`'s status or raise `UnknownVersionError` (fail fast)."""
        current = self._store.get_status(version_id)
        if current is None:
            raise UnknownVersionError(version_id)
        return current

    def transition(self, version_id: str, target: PromptStatusEnum) -> PromptStatusEnum:
        """Move `version_id` to `target`, enforcing the §1.5 lifecycle rules.

        Returns the new status. Raises `UnknownVersionError` for an unknown
        version and `IllegalTransitionError` for a disallowed move. Promoting to
        ``production`` is additionally gated server-side on the eval gate
        (REG-13): an un-green version raises `EvalGateNotGreenError` and nothing
        is written. A successful production promotion demotes the prompt's prior
        production version (the pointer flip); use `rollback` to flip the pointer
        back to a specific version.
        """
        current = self._require_status(version_id)
        if not _is_allowed(current, target):
            raise IllegalTransitionError(version_id, current, target)

        if target == PromptStatusEnum.production:
            self._require_eval_green(version_id)
            self._demote_current_production(version_id)

        self._store.set_status(version_id, target)
        return target

    def promote(self, version_id: str) -> PromptStatusEnum:
        """Advance `version_id` one step along ``draft → candidate → production``.

        Convenience over `transition`: picks the single legal forward target for
        the version's current status. Raises `IllegalTransitionError` from a
        terminal/already-production state (no forward edge).
        """
        current = self._require_status(version_id)
        forward = _ALLOWED_TRANSITIONS[current]
        if not forward:
            raise IllegalTransitionError(version_id, current, current)
        # Each non-terminal state has exactly one forward edge (§1.5).
        (target,) = tuple(forward)
        return self.transition(version_id, target)

    def retire(self, version_id: str) -> PromptStatusEnum:
        """Retire `version_id` (any non-retired state → ``retired``)."""
        return self.transition(version_id, PromptStatusEnum.retired)

    def rollback(self, prompt_id: str, to_version_id: str) -> PromptStatusEnum:
        """Instantly flip the production pointer to `to_version_id` (REG-5).

        The current production version (if any) is demoted to ``candidate`` and
        `to_version_id` is set ``production``, so the next resolve of
        ``<name>@production`` returns `to_version_id` with no redeploy. Raises
        `UnknownVersionError` if `to_version_id` is unknown, and
        `IllegalTransitionError` if it is ``retired`` (a retired version cannot
        be made production). The rolled-back version must still pass the eval gate
        (REG-13) — flipping the pointer is a production promotion, so a version
        whose gate is not green raises `EvalGateNotGreenError`. Rolling back to
        the version already in production is a no-op that returns ``production``.
        """
        current = self._require_status(to_version_id)
        if current == PromptStatusEnum.production:
            return PromptStatusEnum.production
        if current == PromptStatusEnum.retired:
            raise IllegalTransitionError(to_version_id, current, PromptStatusEnum.production)

        self._require_eval_green(to_version_id)
        self._demote_current_production(to_version_id)
        self._store.set_status(to_version_id, PromptStatusEnum.production)
        return PromptStatusEnum.production

    def _require_eval_green(self, version_id: str) -> None:
        """Block a production promotion unless the eval gate is green (REG-13).

        Server-side enforcement: queries the injected `EvalGateChecker` and
        raises `EvalGateNotGreenError` if `version_id` has no passing eval result,
        before any status is written. Called on every path that flips the
        production pointer (`transition` to ``production`` and `rollback`).
        """
        if not self._eval_gate.is_green(version_id):
            raise EvalGateNotGreenError(version_id)

    def _demote_current_production(self, incoming_version_id: str) -> None:
        """Demote the prompt's current production version (≠ incoming) to candidate.

        Keeps the single-production invariant: there is at most one production
        version per prompt at any time (atlas-docs/03 §1.5).
        """
        prompt_id = self._store.get_prompt_id(incoming_version_id)
        if prompt_id is None:
            raise UnknownVersionError(incoming_version_id)
        existing = self._store.get_production_version_id(prompt_id)
        if existing is not None and existing != incoming_version_id:
            self._store.set_status(existing, _DEMOTED_STATUS)
