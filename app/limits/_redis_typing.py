"""Typed views over the subset of the Redis pipeline the limit adapters use.

``redis.asyncio`` types its command methods with bare, un-parameterized returns
(e.g. ``Union[Awaitable[List], List]``), which surface as *partially unknown* under
pyright strict. The limit adapters (GW-16/GW-17) run their read-modify-write inside
``redis.asyncio.Redis.transaction``, whose callback is handed a ``Pipeline``. This
module pins a precise structural view of exactly the pipeline operations those
adapters call, so the call sites stay fully typed without scattering ``cast`` /
``# type: ignore`` over business logic. The real ``Pipeline`` satisfies it
structurally (it is passed at runtime; this is a typing-only contract).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Protocol, cast

import redis.asyncio as redis_async
from redis.asyncio.client import Pipeline


class TxnPipe(Protocol):
    """Structural view of the transaction pipeline used by the limit adapters.

    Before ``multi()`` the pipeline runs in immediate mode, so reads
    (:meth:`hmget`) are awaited; after ``multi()`` writes (:meth:`hset`,
    :meth:`expire`) are queued and committed by ``transaction``'s ``EXEC``.
    """

    def hmget(self, name: str, keys: Sequence[str]) -> Awaitable[list[str | bytes | None]]:
        """Immediate-mode multi-field hash read (awaited)."""
        ...

    def multi(self) -> None:
        """Switch the pipeline from immediate mode to buffered MULTI mode."""
        ...

    def hset(self, name: str, *, mapping: Mapping[str, str]) -> object:
        """Queue a hash write (committed by the surrounding transaction)."""
        ...

    def expire(self, name: str, time: int) -> object:
        """Queue a TTL refresh (committed by the surrounding transaction)."""
        ...


def create_redis_client(url: str, *, decode_responses: bool = True) -> redis_async.Redis:
    """Construct a `redis.asyncio.Redis` from a URL, typed concretely.

    ``redis.asyncio.Redis.from_url`` is typed with ``**kwargs: Unknown``, which
    pyright strict reports as a partially unknown member. Confining that one
    suppression here (the same idiom as the rest of this module) keeps the
    composition root fully typed; the returned client is the real `Redis`.
    """
    client: redis_async.Redis = redis_async.Redis.from_url(  # type: ignore[reportUnknownMemberType]
        url, decode_responses=decode_responses
    )
    return client


async def run_transaction(
    client: redis_async.Redis,
    func: Callable[[TxnPipe], Awaitable[None]],
    *watches: str,
) -> None:
    """Run ``func`` as a watched optimistic transaction over ``client``.

    A single typed seam: ``redis.asyncio.Redis.transaction`` types its callback as
    ``Callable[[Pipeline], ...]`` with ``Pipeline`` un-parameterized, so command
    reads come back partially unknown. This helper adapts the precisely-typed
    :class:`TxnPipe` callback the adapters write to that loose signature, confining
    the one unavoidable ``cast`` here instead of leaking it into business logic.
    The real ``Pipeline`` satisfies :class:`TxnPipe` structurally at runtime.
    """
    pipe_func = cast("Callable[[Pipeline], Awaitable[None]]", func)
    await client.transaction(pipe_func, *watches)
