"""Per-tenant request rate limit (F5c.5).

A sliding-window log per tenant, kept in memory: the limit is per process, so more than
one worker would need a shared store. Tenant ids come from signed tokens, and memory is
bounded both in hits per tenant (at most ``limit``) and in tracked tenants.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict, deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_MAX_TENANTS = 10_000


class TenantRateLimiter:
    """Allow at most ``limit`` requests per tenant in any ``window_seconds`` window."""

    def __init__(
        self,
        limit: int,
        window_seconds: float = 60.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_tenants: int = DEFAULT_MAX_TENANTS,
    ) -> None:
        if limit < 1:
            msg = "limit must be at least 1"
            raise ValueError(msg)
        self.limit = limit
        self._window = window_seconds
        self._clock = clock
        self._max_tenants = max_tenants
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()

    def check(self, tenant_id: str) -> int | None:
        """Count one request; return ``None`` if allowed, else seconds until it would be.

        Rejected requests are not recorded, so hammering a blocked tenant does not
        extend its block.
        """
        now = self._clock()
        self._forget_idle(now)
        hits = self._hits.setdefault(tenant_id, deque())
        self._hits.move_to_end(tenant_id)
        while hits and hits[0] <= now - self._window:
            hits.popleft()
        if len(hits) >= self.limit:
            return max(1, math.ceil(hits[0] + self._window - now))
        hits.append(now)
        while len(self._hits) > self._max_tenants:
            self._hits.popitem(last=False)
        return None

    def _forget_idle(self, now: float) -> None:
        # Least recently used tenants sit at the front: drop those with no hit in the window.
        while self._hits:
            tenant, hits = next(iter(self._hits.items()))
            if hits and hits[-1] > now - self._window:
                break
            del self._hits[tenant]

    def tracked_tenants(self) -> int:
        return len(self._hits)

    def hits(self, tenant_id: str) -> int:
        return len(self._hits.get(tenant_id, ()))
