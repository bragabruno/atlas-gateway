"""AccountingRecorder adapter — prices CallContext, persists, publishes.

Fakes mirror tests/test_accounting.py (_FakeConn) and tests/test_call_events.py
(_FakeProducer); no DB or Kafka.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest

from app.accounting.adapter import (
    AccountingRecorder,
    api_key_uuid,
    provider_for_model,
    rates_from_seed,
)
from app.accounting.events import EventPublisher
from app.accounting.recorder import CallRecorder, Rates
from app.domain.messages import Usage
from app.repositories.tables import ProviderEnum
from app.services.chat_service import CallContext


class _FakeConn:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    async def execute(self, query: str, *args: object) -> str:
        self.executed.append((query, args))
        return "INSERT 0 1"


class _FakeProducer:
    def __init__(self, *, fail: bool = False) -> None:
        self.sent: list[tuple[str, bytes | None, bytes | None]] = []
        self._fail = fail

    async def send(
        self, topic: str, value: bytes | None = None, key: bytes | None = None
    ) -> object:
        if self._fail:
            raise RuntimeError("broker down")
        self.sent.append((topic, value, key))
        return object()


_CALL = CallContext(
    api_key_id="dev-key",
    model="smart",
    usage=Usage(input_tokens=1000, output_tokens=500),
)


def test_api_key_uuid_is_deterministic() -> None:
    assert api_key_uuid("dev-key") == api_key_uuid("dev-key")
    assert api_key_uuid("dev-key") != api_key_uuid("other-key")
    assert isinstance(api_key_uuid("dev-key"), uuid.UUID)


def test_provider_classification() -> None:
    assert provider_for_model("claude-sonnet-4-6") is ProviderEnum.anthropic
    assert provider_for_model("gemini-2.5-pro") is ProviderEnum.google
    assert provider_for_model("gpt-4o") is ProviderEnum.openai
    assert provider_for_model("mock") is ProviderEnum.openai
    assert provider_for_model("gpt-oss:120b-cloud") is ProviderEnum.openai


def test_rates_from_seed_covers_aliases() -> None:
    rates = rates_from_seed()
    assert rates["smart"].in_per_1m == Decimal("3.00")
    assert "mock" not in rates  # falls back to zero in the adapter


@pytest.mark.asyncio
async def test_record_persists_priced_row() -> None:
    conn = _FakeConn()
    adapter = AccountingRecorder(
        CallRecorder(conn),
        rates={"smart": Rates(in_per_1m=Decimal("3.00"), out_per_1m=Decimal("15.00"))},
    )

    await adapter.record(_CALL)

    assert len(conn.executed) == 1
    _, args = conn.executed[0]
    # column order: id, api_key_id, app, prompt_version_id, alias, model,
    # provider, in, out, cache_creation, cache_read, cost, latency, status
    assert args[1] == api_key_uuid("dev-key")
    assert args[5] == "smart"
    assert args[6] == ProviderEnum.openai.value or args[6] == "openai"
    # cost = 1000*3/1e6 + 500*15/1e6 = 0.003 + 0.0075
    assert args[11] == Decimal("0.0105")
    assert args[13] == 200


@pytest.mark.asyncio
async def test_unknown_model_prices_at_zero() -> None:
    conn = _FakeConn()
    adapter = AccountingRecorder(CallRecorder(conn), rates={})

    await adapter.record(
        CallContext(api_key_id="dev-key", model="mock", usage=Usage(input_tokens=99))
    )

    _, args = conn.executed[0]
    assert args[11] == Decimal("0")


@pytest.mark.asyncio
async def test_record_publishes_event_when_publisher_wired() -> None:
    conn = _FakeConn()
    producer = _FakeProducer()
    adapter = AccountingRecorder(CallRecorder(conn), rates={}, publisher=EventPublisher(producer))

    await adapter.record(_CALL)

    assert len(producer.sent) == 1
    topic, value, key = producer.sent[0]
    assert topic == "atlas.calls.v1"
    assert value is not None and b"smart" in value
    assert key == str(api_key_uuid("dev-key")).encode()


@pytest.mark.asyncio
async def test_record_never_raises_on_db_failure() -> None:
    class _BoomConn:
        async def execute(self, query: str, *args: object) -> str:
            raise RuntimeError("db down")

    adapter = AccountingRecorder(CallRecorder(_BoomConn()), rates={})
    await adapter.record(_CALL)  # must not raise


@pytest.mark.asyncio
async def test_record_never_raises_on_publish_failure() -> None:
    conn = _FakeConn()
    adapter = AccountingRecorder(
        CallRecorder(conn),
        rates={},
        publisher=EventPublisher(_FakeProducer(fail=True)),
    )
    await adapter.record(_CALL)  # must not raise
    assert len(conn.executed) == 1  # DB write still happened
