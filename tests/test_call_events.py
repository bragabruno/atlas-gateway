"""GW-15 — `atlas.calls.v1` accounting-event publish, offline (FAKE producer).

A fake `KafkaProducer` captures the (topic, value, key) of every enqueued
message — no live Kafka. A fake consumer then deserializes the captured bytes
and asserts the events are well-formed against the atlas-docs/03 §4.1 schema,
keyed by `api_key_id`. Backpressure tolerance is pinned by injecting a producer
that raises: `publish` must route to the error sink and return `False` without
propagating — accounting never fails the request path. The end-to-end "a live
consumer reads the events" check pends a real broker.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from decimal import Decimal

from app.accounting.events import (
    CALLS_TOPIC,
    CallEvent,
    EventPublisher,
)
from app.accounting.recorder import CallRecord
from app.domain.messages import Usage
from app.repositories.tables import ProviderEnum

# atlas-docs/03 §4.1 required payload fields.
_REQUIRED_FIELDS = {
    "event_id",
    "api_key_id",
    "call_record_id",
    "alias",
    "model",
    "provider",
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "computed_cost_usd",
    "latency_ms",
    "status",
    "created_at",
}


def _record(api_key_id: uuid.UUID | None = None) -> CallRecord:
    return CallRecord(
        id=uuid.uuid4(),
        api_key_id=api_key_id or uuid.uuid4(),
        app="research-bot",
        model="claude-sonnet-4-6",
        provider=ProviderEnum.anthropic,
        usage=Usage(
            input_tokens=800,
            output_tokens=200,
            cache_creation_input_tokens=500,
            cache_read_input_tokens=300,
        ),
        cost=Decimal("0.007365"),
        latency_ms=412,
        status=200,
        alias="smart",
    )


class _FakeProducer:
    """Fake Kafka producer: captures every enqueued message (no broker)."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, bytes | None, bytes | None]] = []

    async def send(
        self,
        topic: str,
        value: bytes | None = None,
        key: bytes | None = None,
    ) -> object:
        self.sent.append((topic, value, key))
        return object()  # stand-in for the broker-ack future (never awaited)


class _ExplodingProducer:
    """Fake producer that always raises — models buffer-full / broker-down."""

    async def send(
        self,
        topic: str,
        value: bytes | None = None,
        key: bytes | None = None,
    ) -> object:
        raise RuntimeError("kafka buffer full")


class _FakeConsumer:
    """Reads the bytes a `_FakeProducer` captured and decodes them as events."""

    def __init__(self, producer: _FakeProducer) -> None:
        self._producer = producer

    def poll(self) -> list[tuple[str, dict[str, object], str | None]]:
        out: list[tuple[str, dict[str, object], str | None]] = []
        for topic, value, key in self._producer.sent:
            assert value is not None
            payload = json.loads(value.decode("utf-8"))
            decoded_key = key.decode("utf-8") if key is not None else None
            out.append((topic, payload, decoded_key))
        return out


# ── CallEvent payload shape (atlas-docs/03 §4.1) ────────────────────────────


def test_event_payload_has_all_required_fields() -> None:
    event = CallEvent.from_record(_record())
    payload = event.to_payload()
    assert set(payload) == _REQUIRED_FIELDS


def test_payload_maps_record_fields() -> None:
    rec = _record()
    payload = CallEvent.from_record(rec).to_payload()
    assert payload["call_record_id"] == str(rec.id)
    assert payload["api_key_id"] == str(rec.api_key_id)
    assert payload["model"] == "claude-sonnet-4-6"
    assert payload["provider"] == "anthropic"  # enum rendered as value
    assert payload["alias"] == "smart"
    assert payload["input_tokens"] == 800
    assert payload["output_tokens"] == 200
    assert payload["cache_creation_input_tokens"] == 500
    assert payload["cache_read_input_tokens"] == 300
    assert payload["latency_ms"] == 412
    assert payload["status"] == 200


def test_cost_is_number_in_payload() -> None:
    """The §4.1 schema types computed_cost_usd as `number` (float on the wire)."""
    payload = CallEvent.from_record(_record()).to_payload()
    assert isinstance(payload["computed_cost_usd"], float)
    assert payload["computed_cost_usd"] == 0.007365


def test_created_at_is_iso8601() -> None:
    fixed = datetime(2026, 6, 7, 12, 0, 0, tzinfo=UTC)
    payload = CallEvent.from_record(_record(), created_at=fixed).to_payload()
    assert payload["created_at"] == "2026-06-07T12:00:00+00:00"


def test_alias_nullable() -> None:
    rec = CallRecord(
        id=uuid.uuid4(),
        api_key_id=uuid.uuid4(),
        app="bare",
        model="gpt-4.1",
        provider=ProviderEnum.openai,
        usage=Usage(input_tokens=10),
        cost=Decimal("0.00003"),
        latency_ms=5,
        status=200,
    )
    payload = CallEvent.from_record(rec).to_payload()
    assert payload["alias"] is None


def test_distinct_event_id_per_emission() -> None:
    """A replay of the same call gets a new event_id but the same record id."""
    rec = _record()
    e1 = CallEvent.from_record(rec)
    e2 = CallEvent.from_record(rec)
    assert e1.event_id != e2.event_id
    assert e1.call_record_id == e2.call_record_id == rec.id


# ── publish: topic + tenant key, then a fake consumer reads events ──────────


async def test_publish_sends_to_calls_topic_keyed_by_api_key() -> None:
    producer = _FakeProducer()
    rec = _record()
    ok = await EventPublisher(producer).publish_record(rec)
    assert ok is True
    (topic, value, key) = producer.sent[0]
    assert topic == CALLS_TOPIC
    assert value is not None
    assert key == str(rec.api_key_id).encode("utf-8")  # partitioned by tenant


async def test_fake_consumer_reads_well_formed_events() -> None:
    producer = _FakeProducer()
    publisher = EventPublisher(producer)
    api_key = uuid.uuid4()
    await publisher.publish_record(_record(api_key_id=api_key))
    await publisher.publish_record(_record(api_key_id=api_key))

    messages = _FakeConsumer(producer).poll()
    assert len(messages) == 2
    for topic, payload, key in messages:
        assert topic == CALLS_TOPIC
        assert set(payload) == _REQUIRED_FIELDS
        assert key == str(api_key)  # all this tenant's events share a partition key
        # ids round-trip as valid uuids; provider is in the allowed enum
        uuid.UUID(str(payload["event_id"]))
        uuid.UUID(str(payload["call_record_id"]))
        assert payload["provider"] in {"anthropic", "openai", "google", "azure_openai"}


async def test_same_tenant_events_share_partition_key() -> None:
    producer = _FakeProducer()
    publisher = EventPublisher(producer)
    api_key = uuid.uuid4()
    await publisher.publish_record(_record(api_key_id=api_key))
    await publisher.publish_record(_record(api_key_id=api_key))
    keys = {key for _, _, key in producer.sent}
    assert keys == {str(api_key).encode("utf-8")}


async def test_different_tenant_events_have_distinct_keys() -> None:
    producer = _FakeProducer()
    publisher = EventPublisher(producer)
    await publisher.publish_record(_record(api_key_id=uuid.uuid4()))
    await publisher.publish_record(_record(api_key_id=uuid.uuid4()))
    keys = {key for _, _, key in producer.sent}
    assert len(keys) == 2


# ── backpressure tolerance: a failing producer never breaks the path ────────


async def test_publish_swallows_producer_error_and_returns_false() -> None:
    publisher = EventPublisher(_ExplodingProducer())
    ok = await publisher.publish_record(_record())  # must not raise
    assert ok is False


async def test_error_sink_invoked_on_failure() -> None:
    captured: list[tuple[CallEvent, Exception]] = []

    async def sink(event: CallEvent, exc: Exception) -> None:
        captured.append((event, exc))

    publisher = EventPublisher(_ExplodingProducer(), error_sink=sink)
    rec = _record()
    ok = await publisher.publish_record(rec)
    assert ok is False
    assert len(captured) == 1
    event, exc = captured[0]
    assert event.call_record_id == rec.id
    assert isinstance(exc, RuntimeError)


async def test_sync_error_sink_supported() -> None:
    """A non-async error sink is accepted too (return value is not awaited)."""
    captured: list[Exception] = []

    def sink(event: CallEvent, exc: Exception) -> None:
        captured.append(exc)

    publisher = EventPublisher(_ExplodingProducer(), error_sink=sink)
    ok = await publisher.publish_record(_record())
    assert ok is False
    assert len(captured) == 1
