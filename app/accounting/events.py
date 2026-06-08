"""GW-15 — Publish per-call accounting events to Kafka `atlas.calls.v1`.

A thin capability adapter (ADR-016) sitting beside the `call_records` recorder
(GW-14): for each completed call it emits one event on the `atlas.calls.v1`
topic, **keyed by `api_key_id`** so every event for a tenant lands on the same
partition for ordered budget aggregation (atlas-docs/03 §4.1, ADR-007). The
payload is exactly the §4.1 JSON-Schema shape.

The producer is hidden behind a `KafkaProducer` Protocol so this module never
imports `aiokafka` at module load: the composition root injects an
`AIOKafkaProducerAdapter` (real broker), while offline tests inject a fake that
captures messages — no live Kafka. `AIOKafkaProducerAdapter` imports `aiokafka`
lazily inside `__init__`, so the import cost is paid only when a real producer
is actually constructed.

Publishing is **non-blocking on the request path** and **backpressure-
tolerant**: `EventPublisher.publish` enqueues into the producer's send buffer
and returns without awaiting the broker ack, and any producer error (buffer
full, broker unavailable) is swallowed after invoking an injectable error sink —
a billing event must never fail or stall a user's completion. The durable
record of truth is the Postgres `call_records` row (GW-14); this stream powers
real-time budget/dashboards and is replayable from the log. Wiring this into the
chat request path is a separate ticket. See GW-15 + ADR-007 + atlas-docs/03 §4.1.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

from app.accounting._aiokafka_typing import create_producer
from app.accounting.recorder import CallRecord

#: Accounting events topic (atlas-docs/03 §4.1, ADR-007). `.v1` suffix encodes
#: the schema version; evolution is a new topic, not a breaking change.
CALLS_TOPIC = "atlas.calls.v1"


@dataclass(frozen=True, slots=True)
class CallEvent:
    """One `atlas.calls.v1` accounting event (atlas-docs/03 §4.1 schema).

    `event_id` is unique per emission (a replay/retry of the same call gets a
    new event id); `call_record_id` is the stable id of the priced
    `call_records` row, so consumers can de-duplicate on it. All fields mirror
    the §4.1 JSON-Schema payload one-to-one.
    """

    event_id: uuid.UUID
    api_key_id: uuid.UUID
    call_record_id: uuid.UUID
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    computed_cost_usd: Decimal
    latency_ms: int
    status: int
    created_at: datetime
    alias: str | None = None

    @classmethod
    def from_record(
        cls,
        record: CallRecord,
        *,
        event_id: uuid.UUID | None = None,
        created_at: datetime | None = None,
    ) -> CallEvent:
        """Build an event from a priced `CallRecord` (GW-14).

        `event_id` defaults to a fresh uuid (unique per emission);
        `created_at` defaults to now (UTC). The record's `provider` enum is
        rendered as its string value to match the §4.1 schema.
        """
        return cls(
            event_id=event_id or uuid.uuid4(),
            api_key_id=record.api_key_id,
            call_record_id=record.id,
            model=record.model,
            provider=record.provider.value,
            input_tokens=record.usage.input_tokens,
            output_tokens=record.usage.output_tokens,
            cache_creation_input_tokens=record.usage.cache_creation_input_tokens,
            cache_read_input_tokens=record.usage.cache_read_input_tokens,
            computed_cost_usd=record.cost,
            latency_ms=record.latency_ms,
            status=record.status,
            created_at=created_at or datetime.now(UTC),
            alias=record.alias,
        )

    def to_payload(self) -> dict[str, Any]:
        """Render the §4.1 JSON-Schema payload as a JSON-serializable dict.

        UUIDs become strings; `created_at` is ISO-8601; `computed_cost_usd` is a
        float (the schema types it `number`) — the Decimal-exact value of record
        lives in Postgres, the event is for real-time aggregation only.
        """
        return {
            "event_id": str(self.event_id),
            "api_key_id": str(self.api_key_id),
            "call_record_id": str(self.call_record_id),
            "alias": self.alias,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "computed_cost_usd": float(self.computed_cost_usd),
            "latency_ms": self.latency_ms,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
        }

    def serialize(self) -> bytes:
        """Serialize the payload to UTF-8 JSON bytes (the Kafka message value)."""
        return json.dumps(self.to_payload(), separators=(",", ":")).encode("utf-8")

    def partition_key(self) -> bytes:
        """The Kafka message key — `api_key_id` bytes (atlas-docs/03 §4.1).

        Keying by tenant collocates a tenant's events on one partition so budget
        aggregation sees them in order.
        """
        return str(self.api_key_id).encode("utf-8")


class KafkaProducer(Protocol):
    """Structural view of the async Kafka producer the publisher needs.

    `send` enqueues a message into the producer's send buffer and returns once
    buffered (it does *not* block on the broker ack). `AIOKafkaProducer.send`
    satisfies this; tests inject a fake. Keeping the dependency a Protocol means
    this module never imports `aiokafka` to be type-checked or unit-tested.
    """

    async def send(
        self,
        topic: str,
        value: bytes | None = None,
        key: bytes | None = None,
    ) -> object: ...


class AIOKafkaProducerAdapter:
    """Adapts a real `aiokafka.AIOKafkaProducer` to the `KafkaProducer` port.

    The underlying producer is built by `_aiokafka_typing.create_producer`,
    which imports `aiokafka` lazily — so the dependency is only loaded when a
    live producer is actually constructed (the composition root), never at
    module import or in offline tests — and returns it typed against a narrow
    structural contract (no scattered `# type: ignore`). `start`/`stop` manage
    the underlying producer's connection lifecycle and are called by the
    composition root.
    """

    def __init__(self, *, bootstrap_servers: str, **producer_kwargs: Any) -> None:
        self._producer = create_producer(bootstrap_servers=bootstrap_servers, **producer_kwargs)

    async def start(self) -> None:
        """Connect the underlying producer (composition-root lifecycle)."""
        await self._producer.start()

    async def stop(self) -> None:
        """Flush and disconnect the underlying producer (graceful shutdown)."""
        await self._producer.stop()

    async def send(
        self,
        topic: str,
        value: bytes | None = None,
        key: bytes | None = None,
    ) -> object:
        """Enqueue a message; returns the broker-ack future (intentionally not
        awaited by `EventPublisher`, keeping the request path non-blocking)."""
        return await self._producer.send(topic, value=value, key=key)


#: An error sink invoked (with the failed event and the exception) when a
#: publish cannot be enqueued. Defaults to a no-op so accounting failures never
#: surface on the request path; the composition root can pass one that logs.
ErrorSink = Callable[[CallEvent, Exception], Awaitable[None] | None]


async def _noop_error_sink(event: CallEvent, exc: Exception) -> None:
    """Default error sink — swallows the failure (request path must not break)."""
    return None


class EventPublisher:
    """Publishes `atlas.calls.v1` accounting events, keyed by `api_key_id`.

    The producer is injected (real adapter or fake), so this composes the §4.1
    payload + tenant key and is decoupled from Kafka. `publish` is non-blocking
    and backpressure-tolerant: it enqueues via `producer.send` and does not
    await the broker ack, and any producer exception is routed to the injected
    `error_sink` and then swallowed — a billing event must never fail a user's
    completion (the durable record is the Postgres row, GW-14).
    """

    def __init__(
        self,
        producer: KafkaProducer,
        *,
        topic: str = CALLS_TOPIC,
        error_sink: ErrorSink = _noop_error_sink,
    ) -> None:
        self._producer = producer
        self._topic = topic
        self._error_sink = error_sink

    async def publish(self, event: CallEvent) -> bool:
        """Enqueue one accounting event; return whether it was enqueued.

        Returns `True` once the message is buffered by the producer, `False` if
        the producer raised (buffer full, broker down) — in which case the
        `error_sink` is invoked and the exception is suppressed so the request
        path is never blocked or failed by accounting.
        """
        try:
            await self._producer.send(
                self._topic,
                value=event.serialize(),
                key=event.partition_key(),
            )
        except Exception as exc:
            result = self._error_sink(event, exc)
            if result is not None:
                await result
            return False
        return True

    async def publish_record(self, record: CallRecord) -> bool:
        """Convenience: build an event from a `CallRecord` and publish it."""
        return await self.publish(CallEvent.from_record(record))
