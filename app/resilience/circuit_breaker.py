"""GW-12 — Redis-backed per-provider circuit breaker + failover (ADR-011).

A thin capability adapter (ADR-016): a per-provider circuit breaker whose state
(CLOSED / OPEN / HALF-OPEN) lives in Redis so every AKS replica shares one view
of a provider's health — a provider that fails on one replica is quarantined
fleet-wide within seconds, not re-discovered independently by each replica
(ADR-011). The service layer consults the breaker before each upstream call and
reports the outcome back; this adapter owns *sustained-failure* shedding, while
`app.resilience.retry` (GW-11) owns *per-call* retry of the same request.

**State machine** (see docs/diagrams/circuit-breaker-state.md):

- **CLOSED** — calls flow. A success resets the consecutive-failure counter; a
  failure increments it. Reaching ``failure_threshold`` opens the breaker.
- **OPEN** — calls are rejected fast (the controller maps this to a 503 and the
  service fails over to the fallback provider). After ``cooldown_seconds`` the
  next admission probe transitions to HALF-OPEN.
- **HALF-OPEN** — a single probe call is admitted. Its success closes the
  breaker (counter cleared); its failure re-opens it for a fresh cooldown.

**Atomicity (ADR-011).** ADR-011 specifies Redis for atomic state. ``lupa`` (the
engine ``fakeredis`` needs to run ``EVAL`` offline) is not in the project's
pinned deps, so — exactly as the GW-16/GW-17 limit adapters do — this breaker
uses Redis' other first-class atomic primitive, an optimistic
``WATCH``/``MULTI``/``EXEC`` transaction via the shared
:func:`app.limits._redis_typing.run_transaction` seam. Each admission decision
and each outcome record is one transaction (read state, decide, queue the write),
so concurrent replicas never corrupt the counter or race a transition. State is a
Redis hash ``{state, failures, opened_at}`` under a per-provider key with a TTL so
an idle breaker self-evicts to a clean CLOSED default.

**Failover.** :meth:`CircuitBreaker.choose` reads two providers' states and
returns the first whose breaker would admit a call (primary unless its breaker is
OPEN and still cooling, else the fallback), or raises
:class:`AllProvidersUnavailable` when neither can serve — fail fast, never a
silent drop. Recovery is automatic via the HALF-OPEN probe; no manual reset.

Pinned deps: redis 7.4.0, fakeredis 2.35.1 (dev/tests). See GW-12 + ADR-011 +
atlas-docs/02 §ADR-011.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import redis.asyncio as redis_async

from app.limits._redis_typing import TxnPipe, run_transaction

#: Key namespace prefix; keeps breaker state distinct from the cache and
#: rate-limit keyspaces sharing the Redis instance (see app/cache/exact.py).
_KEY_PREFIX = "atlas:circuit"

#: Hash fields: the textual state, the consecutive-failure count, and the epoch
#: seconds the breaker last opened (only meaningful while OPEN/HALF-OPEN).
_FIELD_STATE = "state"
_FIELD_FAILURES = "failures"
_FIELD_OPENED_AT = "opened_at"


class CircuitState(str, Enum):
    """The three breaker states shared across replicas (ADR-011)."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised by :meth:`CircuitBreaker.allow` when a provider is shedding load.

    The breaker for ``provider`` is OPEN and still within its cooldown, so the
    call is rejected fast rather than sent to a known-unhealthy upstream. The
    controller maps this to a 503; the service layer catches it to fail over to
    the fallback provider (wiring later). Business logic stays HTTP-free
    (ADR-016).
    """

    def __init__(self, *, provider: str, retry_after: int) -> None:
        self.provider = provider
        self.retry_after = retry_after
        super().__init__(
            f"circuit breaker open for provider {provider!r}; retry after {retry_after} seconds"
        )


class AllProvidersUnavailable(Exception):
    """Raised by :meth:`CircuitBreaker.choose` when no candidate can serve.

    Both the primary and fallback breakers are OPEN and cooling, so there is no
    healthy provider to route to. The controller maps this to a 503; failing
    fast here is deliberate — the gateway never silently serves from a provider
    it knows is unhealthy (ADR-016).
    """

    def __init__(self, *, providers: tuple[str, ...], retry_after: int) -> None:
        self.providers = providers
        self.retry_after = retry_after
        super().__init__(
            f"all providers unavailable: {', '.join(providers)}; retry after {retry_after} seconds"
        )


@dataclass(slots=True)
class _Snapshot:
    """A breaker's current state as read inside a transaction.

    ``effective_state`` is the state *after* applying the cooldown rule: an OPEN
    breaker whose cooldown has elapsed is reported as HALF_OPEN (the next call is
    the probe), so callers act on the live state, not the stale stored one.
    """

    state: CircuitState
    failures: int
    opened_at: float
    effective_state: CircuitState
    retry_after: int


class CircuitBreaker:
    """Per-provider circuit breaker backed by an injected ``redis.asyncio`` client.

    The client is injected (composition root supplies a real client; tests supply
    ``fakeredis``), so this adapter does no connection management. ``failure_threshold``
    is the number of *consecutive* failures that trips the breaker from CLOSED to
    OPEN; ``cooldown_seconds`` is how long it stays OPEN before admitting a
    HALF-OPEN probe. The client may be configured with ``decode_responses=True``
    (matching the rest of the gateway); this adapter reads hash fields tolerantly
    of ``str``/``bytes``.

    Usage per call (wiring is a later ticket)::

        await breaker.allow(provider)          # raises CircuitOpenError if OPEN
        try:
            result = await provider_call()
        except ProviderFailure:
            await breaker.record_failure(provider)
            raise
        else:
            await breaker.record_success(provider)
    """

    def __init__(
        self,
        client: redis_async.Redis,
        *,
        failure_threshold: int = 5,
        cooldown_seconds: int = 30,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError(f"failure_threshold must be >= 1, got {failure_threshold}")
        if cooldown_seconds < 1:
            raise ValueError(f"cooldown_seconds must be >= 1, got {cooldown_seconds}")
        self._client = client
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds

    @property
    def failure_threshold(self) -> int:
        """Consecutive failures that trip CLOSED → OPEN."""
        return self._failure_threshold

    @property
    def cooldown_seconds(self) -> int:
        """Seconds an OPEN breaker waits before admitting a HALF-OPEN probe."""
        return self._cooldown_seconds

    def _key(self, provider: str) -> str:
        return f"{_KEY_PREFIX}:{provider}"

    def _ttl_seconds(self) -> int:
        """TTL so an idle breaker self-evicts to a clean CLOSED default.

        Generous relative to the cooldown so OPEN/HALF-OPEN state is never lost
        out from under a cooling breaker, while a long-quiet provider's key
        still expires.
        """
        return max(self._cooldown_seconds * 4, 60)

    @staticmethod
    def _as_int(raw: str | bytes | None, default: int) -> int:
        """Read a hash field that may be ``str``, ``bytes``, or absent."""
        if raw is None:
            return default
        if isinstance(raw, bytes):
            return int(raw.decode())
        return int(raw)

    @staticmethod
    def _as_float(raw: str | bytes | None, default: float) -> float:
        """Read a hash field that may be ``str``, ``bytes``, or absent."""
        if raw is None:
            return default
        if isinstance(raw, bytes):
            return float(raw.decode())
        return float(raw)

    @staticmethod
    def _as_state(raw: str | bytes | None) -> CircuitState:
        """Read the state field, defaulting to CLOSED when absent/unknown."""
        if raw is None:
            return CircuitState.CLOSED
        text = raw.decode() if isinstance(raw, bytes) else raw
        try:
            return CircuitState(text)
        except ValueError:
            # An unrecognized value is a corrupt/foreign key, not a silent pass:
            # treat as CLOSED so the breaker self-heals to a known-good default.
            return CircuitState.CLOSED

    def _read(self, raw: list[str | bytes | None], now: float) -> _Snapshot:
        """Interpret raw hash fields into a cooldown-aware snapshot."""
        state = self._as_state(raw[0])
        failures = self._as_int(raw[1], 0)
        opened_at = self._as_float(raw[2], 0.0)

        effective = state
        retry_after = 0
        if state is CircuitState.OPEN:
            elapsed = now - opened_at
            remaining = self._cooldown_seconds - elapsed
            if remaining <= 0:
                # Cooldown elapsed: the next admission is the HALF-OPEN probe.
                effective = CircuitState.HALF_OPEN
            else:
                retry_after = max(1, int(remaining) + (1 if remaining % 1 else 0))
        return _Snapshot(
            state=state,
            failures=failures,
            opened_at=opened_at,
            effective_state=effective,
            retry_after=retry_after,
        )

    async def _load(self, provider: str) -> _Snapshot:
        """Read the current snapshot for ``provider`` (no mutation).

        Routed through the typed :class:`TxnPipe` seam (:func:`run_transaction`
        with a read-only closure) so the ``hmget`` result is fully typed —
        ``redis.asyncio.Redis.hmget`` is otherwise partially unknown under pyright
        strict (the same reason ``app.limits._redis_typing`` exists). A
        no-``multi()`` closure runs the read in immediate mode without queuing a
        write, so this stays a pure read.
        """
        key = self._key(provider)
        raw: list[str | bytes | None] = []

        async def _read_only(pipe: TxnPipe) -> None:
            raw.extend(await pipe.hmget(key, [_FIELD_STATE, _FIELD_FAILURES, _FIELD_OPENED_AT]))

        await run_transaction(self._client, _read_only, key)
        return self._read(raw, time.time())

    async def state(self, provider: str) -> CircuitState:
        """Return the cooldown-aware effective state for ``provider``.

        An OPEN breaker past its cooldown reports HALF_OPEN (the next call is the
        probe); otherwise the stored state is returned. Pure read, no transition.
        """
        snapshot = await self._load(provider)
        return snapshot.effective_state

    async def allow(self, provider: str) -> None:
        """Admit one call for ``provider``, or raise :class:`CircuitOpenError`.

        CLOSED and HALF-OPEN admit (HALF-OPEN admits exactly the next probe — the
        outcome of which :meth:`record_success`/:meth:`record_failure` resolves).
        OPEN-and-still-cooling rejects fast with the remaining cooldown as
        ``retry_after``. When the cooldown has elapsed the stored state is
        transitioned to HALF-OPEN in the same transaction so concurrent replicas
        agree the probe is now in flight.
        """
        key = self._key(provider)
        decision = _Snapshot(
            state=CircuitState.CLOSED,
            failures=0,
            opened_at=0.0,
            effective_state=CircuitState.CLOSED,
            retry_after=0,
        )

        async def _admit(pipe: TxnPipe) -> None:
            raw = await pipe.hmget(key, [_FIELD_STATE, _FIELD_FAILURES, _FIELD_OPENED_AT])
            snapshot = self._read(raw, time.time())
            decision.effective_state = snapshot.effective_state
            decision.retry_after = snapshot.retry_after

            pipe.multi()
            if (
                snapshot.state is CircuitState.OPEN
                and snapshot.effective_state is CircuitState.HALF_OPEN
            ):
                # Cooldown elapsed: persist the HALF-OPEN transition so the probe
                # is admitted exactly once across the fleet.
                pipe.hset(
                    key,
                    mapping={
                        _FIELD_STATE: CircuitState.HALF_OPEN.value,
                        _FIELD_FAILURES: str(snapshot.failures),
                        _FIELD_OPENED_AT: str(snapshot.opened_at),
                    },
                )
                pipe.expire(key, self._ttl_seconds())

        await run_transaction(self._client, _admit, key)

        if decision.effective_state is CircuitState.OPEN:
            raise CircuitOpenError(provider=provider, retry_after=decision.retry_after)

    async def record_success(self, provider: str) -> None:
        """Record a successful call: reset failures and close the breaker.

        A success from any state (CLOSED, or a HALF-OPEN probe) returns the
        breaker to CLOSED with a zero failure count — the provider is healthy.
        """
        key = self._key(provider)

        async def _record(pipe: TxnPipe) -> None:
            await pipe.hmget(key, [_FIELD_STATE])
            pipe.multi()
            pipe.hset(
                key,
                mapping={
                    _FIELD_STATE: CircuitState.CLOSED.value,
                    _FIELD_FAILURES: "0",
                    _FIELD_OPENED_AT: "0",
                },
            )
            pipe.expire(key, self._ttl_seconds())

        await run_transaction(self._client, _record, key)

    async def record_failure(self, provider: str) -> None:
        """Record a failed call: increment failures and open if tripped.

        From CLOSED, the consecutive-failure counter increments and the breaker
        opens once it reaches ``failure_threshold``. From HALF-OPEN, a failed
        probe re-opens the breaker immediately for a fresh cooldown. ``opened_at``
        is stamped on the open transition so cooldown is measured from then.
        """
        key = self._key(provider)

        async def _record(pipe: TxnPipe) -> None:
            raw = await pipe.hmget(key, [_FIELD_STATE, _FIELD_FAILURES, _FIELD_OPENED_AT])
            now = time.time()
            snapshot = self._read(raw, now)

            if snapshot.effective_state is CircuitState.HALF_OPEN:
                # A failed probe re-opens immediately for a fresh cooldown.
                new_state = CircuitState.OPEN
                new_failures = max(snapshot.failures, self._failure_threshold)
                new_opened_at = now
            else:
                new_failures = snapshot.failures + 1
                if new_failures >= self._failure_threshold:
                    new_state = CircuitState.OPEN
                    new_opened_at = now
                else:
                    new_state = CircuitState.CLOSED
                    new_opened_at = snapshot.opened_at

            pipe.multi()
            pipe.hset(
                key,
                mapping={
                    _FIELD_STATE: new_state.value,
                    _FIELD_FAILURES: str(new_failures),
                    _FIELD_OPENED_AT: str(new_opened_at),
                },
            )
            pipe.expire(key, self._ttl_seconds())

        await run_transaction(self._client, _record, key)

    async def choose(self, primary: str, fallback: str) -> str:
        """Select the provider to call, honouring both breakers' states.

        Returns ``primary`` unless its breaker is OPEN-and-cooling, in which case
        it returns ``fallback`` if that breaker would admit a call. If both are
        OPEN-and-cooling, raises :class:`AllProvidersUnavailable` with the
        smaller remaining cooldown as ``retry_after`` — fail fast, never serve
        from a provider known to be unhealthy. When ``primary == fallback`` (an
        alias with no separate fallback), there is only one candidate.

        This decides the *target* only; the caller still calls :meth:`allow`
        before dispatching, which performs the HALF-OPEN transition atomically.
        """
        primary_snapshot = await self._load(primary)
        if primary_snapshot.effective_state is not CircuitState.OPEN:
            return primary

        if fallback == primary:
            raise AllProvidersUnavailable(
                providers=(primary,),
                retry_after=primary_snapshot.retry_after,
            )

        fallback_snapshot = await self._load(fallback)
        if fallback_snapshot.effective_state is not CircuitState.OPEN:
            return fallback

        raise AllProvidersUnavailable(
            providers=(primary, fallback),
            retry_after=min(primary_snapshot.retry_after, fallback_snapshot.retry_after),
        )
