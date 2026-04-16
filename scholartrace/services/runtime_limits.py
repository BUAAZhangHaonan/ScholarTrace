from __future__ import annotations

import asyncio
import math
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetPolicy:
    name: str
    limit: int
    window_seconds: int
    concurrency: int | None = None


class RateLimitExceeded(Exception):
    def __init__(self, policy_name: str, retry_after_seconds: int):
        self.policy_name = policy_name
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"{policy_name} rate limit exceeded; retry in {retry_after_seconds}s")


class RuntimeBudgetManager:
    def __init__(self) -> None:
        self._events: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._inflight: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    async def _acquire(self, policy: BudgetPolicy, client_key: str) -> None:
        async with self._lock:
            now = time.monotonic()
            event_key = (policy.name, client_key)
            events = self._events[event_key]
            while events and now - events[0] >= policy.window_seconds:
                events.popleft()

            if len(events) >= policy.limit:
                retry_after = max(1, math.ceil(policy.window_seconds - (now - events[0])))
                raise RateLimitExceeded(policy.name, retry_after)

            if policy.concurrency is not None and self._inflight[policy.name] >= policy.concurrency:
                raise RateLimitExceeded(policy.name, 1)

            events.append(now)
            if policy.concurrency is not None:
                self._inflight[policy.name] += 1

    async def _release(self, policy: BudgetPolicy) -> None:
        if policy.concurrency is None:
            return
        async with self._lock:
            if self._inflight[policy.name] > 0:
                self._inflight[policy.name] -= 1

    @asynccontextmanager
    async def enforce(self, policy: BudgetPolicy, client_key: str):
        await self._acquire(policy, client_key)
        try:
            yield
        finally:
            await self._release(policy)

    async def reset(self) -> None:
        async with self._lock:
            self._events.clear()
            self._inflight.clear()


RETRIEVAL_JOB_POLICY = BudgetPolicy("retrieval_job", limit=5, window_seconds=60, concurrency=2)
FULLTEXT_ACQUIRE_POLICY = BudgetPolicy("fulltext_acquire", limit=20, window_seconds=600, concurrency=4)
DEEPXIV_SEARCH_POLICY = BudgetPolicy("deepxiv_search", limit=30, window_seconds=60, concurrency=4)
AGENT_FILTER_POLICY = BudgetPolicy("agent_filter", limit=10, window_seconds=60, concurrency=2)
AUTO_REGISTER_POLICY = BudgetPolicy("deepxiv_auto_register", limit=2, window_seconds=3600, concurrency=1)

budget_manager = RuntimeBudgetManager()
