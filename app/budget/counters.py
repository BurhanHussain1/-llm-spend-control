"""Spend counters -- the fast, atomic view of what a team has spent this period.

Budget checks sit in the hot path of every request, so they cannot be a `SUM()`
over the usage table. These counters hold the same number in a place that can be
incremented atomically.

Two backends behind one interface:

* **Redis** -- ``INCRBYFLOAT`` is atomic, so concurrent requests cannot both pass
  a check that only one of them should. Survives restarts and is shared across
  workers. This is the production shape.
* **In-memory** -- a dict behind an ``asyncio.Lock``. Correct for a single
  process, which is all a local run needs, and it means the gateway starts with
  no Redis.

The interface is deliberately increment-first: callers add their estimate, read
back the new total, and roll the increment back if it broke a limit. Reading and
then writing would leave a race in which two requests each see room for one.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class SpendCounters(Protocol):
    """The counter operations the budget enforcer needs."""

    async def add(
        self, keys: Sequence[tuple[str, int]], amount: float
    ) -> dict[str, float]:
        """Add `amount` to each ``(key, ttl_seconds)`` and return the new totals.

        A single call for all keys so a request's daily and monthly counters move
        together.
        """
        ...

    async def get(self, keys: Sequence[str]) -> dict[str, float]:
        """Read current totals without modifying them. For reporting only."""
        ...

    async def set(self, key: str, value: float, ttl_seconds: int) -> None:
        """Overwrite a counter. Used to seed counters from the usage log on startup."""
        ...

    async def close(self) -> None: ...


class InMemoryCounters:
    """Single-process counters. Correct, not durable."""

    def __init__(self) -> None:
        self._values: dict[str, float] = {}
        self._expires_at: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def add(
        self, keys: Sequence[tuple[str, int]], amount: float
    ) -> dict[str, float]:
        async with self._lock:
            self._prune()
            totals = {}
            for key, ttl_seconds in keys:
                self._values[key] = self._values.get(key, 0.0) + amount
                self._expires_at[key] = time.monotonic() + ttl_seconds
                totals[key] = self._values[key]
            return totals

    async def get(self, keys: Sequence[str]) -> dict[str, float]:
        async with self._lock:
            self._prune()
            return {key: self._values.get(key, 0.0) for key in keys}

    async def set(self, key: str, value: float, ttl_seconds: int) -> None:
        async with self._lock:
            self._values[key] = value
            self._expires_at[key] = time.monotonic() + ttl_seconds

    async def close(self) -> None:
        return None

    def _prune(self) -> None:
        """Drop expired keys so a long-lived process doesn't grow forever."""
        now = time.monotonic()
        expired = [key for key, expiry in self._expires_at.items() if expiry <= now]
        for key in expired:
            self._values.pop(key, None)
            self._expires_at.pop(key, None)


class RedisCounters:
    """Redis-backed counters. Atomic across workers, survives restarts."""

    def __init__(self, redis_url: str) -> None:
        # Imported lazily so `redis` is only needed when it is actually used.
        from redis.asyncio import from_url

        self._client = from_url(redis_url, decode_responses=True)

    async def add(
        self, keys: Sequence[tuple[str, int]], amount: float
    ) -> dict[str, float]:
        async with self._client.pipeline(transaction=True) as pipe:
            for key, ttl_seconds in keys:
                pipe.incrbyfloat(key, amount)
                pipe.expire(key, ttl_seconds)
            results = await pipe.execute()

        # The pipeline returns one result per queued command, so the increments
        # are every other entry.
        return {
            key: float(results[index * 2])
            for index, (key, _ttl) in enumerate(keys)
        }

    async def get(self, keys: Sequence[str]) -> dict[str, float]:
        if not keys:
            return {}
        values = await self._client.mget(list(keys))
        return {
            key: float(value) if value is not None else 0.0
            for key, value in zip(keys, values)
        }

    async def set(self, key: str, value: float, ttl_seconds: int) -> None:
        await self._client.set(key, value, ex=ttl_seconds)

    async def close(self) -> None:
        await self._client.close()


def build_counters(redis_url: str | None) -> SpendCounters:
    """Return the Redis backend when a URL is configured, otherwise in-memory."""
    if redis_url:
        return RedisCounters(redis_url)
    return InMemoryCounters()
