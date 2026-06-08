"""Typed boundary over the subset of `aiokafka` the event publisher uses.

`aiokafka` ships no ``py.typed`` marker, so pyright strict flags its import as a
missing stub and types `AIOKafkaProducer.send` as partially unknown. Rather than
scatter ``# type: ignore`` across the publisher, this module pins a precise
structural view of exactly the producer operations the adapter calls
(`start`/`stop`/`send`) and a single factory that constructs the real producer,
confining the one suppression here — the same idiom as
`app.limits._redis_typing`. The real `AIOKafkaProducer` satisfies `_RawProducer`
structurally at runtime; this is a typing-only contract. The import is performed
inside the factory so `aiokafka` is loaded only when a live producer is built
(the composition root), never at module import or in offline tests.
"""

from __future__ import annotations

from typing import Any, Protocol


class RawProducer(Protocol):
    """Structural view of the `aiokafka.AIOKafkaProducer` methods used here."""

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def send(
        self,
        topic: str,
        value: bytes | None = ...,
        key: bytes | None = ...,
    ) -> object: ...


def create_producer(*, bootstrap_servers: str, **producer_kwargs: Any) -> RawProducer:
    """Construct a real `aiokafka.AIOKafkaProducer` typed as `RawProducer`.

    Imports `aiokafka` lazily so the dependency is only loaded when a live
    producer is actually constructed; the returned object is typed against the
    narrow structural contract above, keeping the adapter fully typed.
    """
    from aiokafka import AIOKafkaProducer  # type: ignore[reportMissingTypeStubs]

    producer: RawProducer = AIOKafkaProducer(bootstrap_servers=bootstrap_servers, **producer_kwargs)
    return producer
