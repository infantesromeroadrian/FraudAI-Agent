"""Cached dependency health for /readyz and /health (F5c.1).

A background monitor runs the dependency checks on an interval and keeps the last result.
The probe routes only read that snapshot, so an unauthenticated probe can never trigger an
outbound call, and a dependency that hangs cannot hang the probe.

Readiness is honest in both directions: nothing is ready before the first snapshot exists,
and a snapshot that has stopped refreshing expires instead of staying green forever.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

import httpx
from pydantic import BaseModel

from fraudai.api.schemas import HealthResponse

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from fraudai.core.config import Settings

logger = logging.getLogger(__name__)

_BUILD_SHA = re.compile(r"[0-9a-f]{7,64}")
# A snapshot older than this many intervals means the refresher is stuck or dead.
STALE_AFTER_INTERVALS = 3
DEFAULT_CHECK_TIMEOUT_SECONDS = 5.0


def served_build_sha(cfg: Settings) -> str | None:
    """The image's build SHA exactly as set in ``BUILD_SHA``, or ``None`` if absent/malformed.

    Never computed or guessed: a host run without the image metadata reports ``None``.
    """
    value = cfg.build_sha.strip()
    return value if _BUILD_SHA.fullmatch(value) else None


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    """Result of one round of checks."""

    checks: Mapping[str, bool]
    corpus_version: str | None
    checked_at: datetime
    taken_at: float


class HealthMonitor:
    """Runs the checks periodically and serves the last result."""

    def __init__(
        self,
        checks: Mapping[str, Callable[[], Awaitable[bool]]],
        required: frozenset[str],
        interval_seconds: float = 15.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        check_timeout_seconds: float = DEFAULT_CHECK_TIMEOUT_SECONDS,
        corpus_version_probe: Callable[[], Awaitable[str | None]] | None = None,
    ) -> None:
        self._checks = dict(checks)
        self._required = required
        self._interval = interval_seconds
        self._clock = clock
        self._now = now
        self._timeout = check_timeout_seconds
        self._corpus_probe = corpus_version_probe
        self.snapshot: HealthSnapshot | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def task(self) -> asyncio.Task[None] | None:
        """The background loop task (kept referenced so it cannot be collected)."""
        return self._task

    async def _run_check(self, name: str) -> bool:
        try:
            return bool(await asyncio.wait_for(self._checks[name](), timeout=self._timeout))
        except Exception as exc:
            # TimeoutError is an Exception; cancellation (BaseException) still propagates.
            logger.warning("health_check_failed: check=%s, error=%s", name, type(exc).__name__)
            return False

    async def _corpus_version(self) -> str | None:
        if self._corpus_probe is None:
            return None
        try:
            return await asyncio.wait_for(self._corpus_probe(), timeout=self._timeout)
        except Exception as exc:
            logger.warning("health_corpus_version_failed: error=%s", type(exc).__name__)
            return None

    async def refresh(self) -> HealthSnapshot:
        """Run every check once (concurrently) and publish the new snapshot."""
        names = list(self._checks)
        outcomes = await asyncio.gather(*(self._run_check(name) for name in names))
        corpus = await self._corpus_version()
        snapshot = HealthSnapshot(
            checks=dict(zip(names, outcomes, strict=True)),
            corpus_version=corpus,
            checked_at=self._now(),
            taken_at=self._clock(),
        )
        self.snapshot = snapshot
        return snapshot

    def is_ready(self) -> bool:
        """Fresh snapshot with every required check passing."""
        snapshot = self.snapshot
        if snapshot is None:
            return False
        if self._clock() - snapshot.taken_at > STALE_AFTER_INTERVALS * self._interval:
            return False
        return all(snapshot.checks.get(name, False) for name in self._required)

    def status(self) -> Literal["starting", "ready", "not_ready"]:
        if self.snapshot is None:
            return "starting"
        return "ready" if self.is_ready() else "not_ready"

    def start(self) -> None:
        """Start the background loop (no-op if it is already running)."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.get_running_loop().create_task(self._loop(), name="health-monitor")

    async def stop(self) -> None:
        """Cancel the loop and wait for it to finish (safe to call repeatedly)."""
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _loop(self) -> None:
        while True:
            try:
                await self.refresh()
            except Exception:
                logger.exception("health_refresh_failed")
            await asyncio.sleep(self._interval)


# ---------------------------------------------------------------------------
# Real checks
# ---------------------------------------------------------------------------

_HTTP_TIMEOUT_SECONDS = 5.0


def build_checks(
    *,
    store: Any,
    db: Any | None,
    cfg: Settings,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[dict[str, Callable[[], Awaitable[bool]]], frozenset[str]]:
    """The dependency checks for this configuration and which of them gate readiness.

    Only the configured LLM provider is contacted, and a provider's key is sent only to
    that provider. The checks run in the monitor, never inside a probe request.
    """

    def client() -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=_HTTP_TIMEOUT_SECONDS, transport=transport)

    async def qdrant() -> bool:
        return bool(await store.health_check())

    async def database() -> bool:
        try:
            await db.check_writable()  # type: ignore[union-attr]
        except Exception:
            return False
        return True

    async def ollama() -> bool:
        async with client() as http:
            response = await http.get(f"{cfg.ollama_host.rstrip('/')}/api/tags")
        return response.status_code == 200

    async def groq() -> bool:
        if not cfg.groq_api_key:
            return False
        async with client() as http:
            response = await http.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {cfg.groq_api_key}"},
            )
        return response.status_code == 200

    async def openai() -> bool:
        if not cfg.openai_api_key:
            return False
        async with client() as http:
            response = await http.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {cfg.openai_api_key}"},
            )
        return response.status_code == 200

    async def anthropic() -> bool:
        if not cfg.anthropic_api_key:
            return False
        async with client() as http:
            response = await http.get(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": cfg.anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
        return response.status_code in (200, 405)  # GET on /messages: reachable + authenticated

    providers = {"ollama": ollama, "groq": groq, "openai": openai, "anthropic": anthropic}
    checks: dict[str, Callable[[], Awaitable[bool]]] = {
        "qdrant": qdrant,
        "ollama": ollama,  # Donna's local classifier; informational unless it is the provider
        "llm": providers[cfg.llm_provider],
    }
    required = {"qdrant", "llm"}
    if db is not None:
        checks["database"] = database
        required.add("database")
    return checks, frozenset(required)


# ---------------------------------------------------------------------------
# Response models and views
# ---------------------------------------------------------------------------


class HealthDetailResponse(HealthResponse):
    """``/health``: the original fields plus readiness, snapshot time and build SHA."""

    ready: bool
    checked_at: str | None = None
    build_sha: str | None = None


class ReadinessResponse(BaseModel):
    """``/readyz`` body (HTTP 200 when ready, 503 otherwise)."""

    status: Literal["starting", "ready", "not_ready"]
    checks: dict[str, bool]
    checked_at: str | None = None
    build_sha: str | None = None


def readiness_view(monitor: HealthMonitor | None, cfg: Settings) -> ReadinessResponse:
    snapshot = monitor.snapshot if monitor is not None else None
    return ReadinessResponse(
        status=monitor.status() if monitor is not None else "starting",
        checks=dict(snapshot.checks) if snapshot is not None else {},
        checked_at=snapshot.checked_at.isoformat() if snapshot is not None else None,
        build_sha=served_build_sha(cfg),
    )


def health_view(monitor: HealthMonitor | None, cfg: Settings) -> HealthDetailResponse:
    snapshot = monitor.snapshot if monitor is not None else None
    checks: Mapping[str, bool] = snapshot.checks if snapshot is not None else {}
    qdrant = checks.get("qdrant", False)
    ollama = checks.get("ollama", False)
    llm = checks.get("llm", False)
    components = [qdrant, ollama, llm]
    if all(components):
        overall = "healthy"
    elif any(components):
        overall = "degraded"
    else:
        overall = "unhealthy"
    return HealthDetailResponse(
        status=overall,
        qdrant=qdrant,
        ollama=ollama,
        claude_api=llm,
        corpus_version=snapshot.corpus_version if snapshot is not None else None,
        ready=monitor.is_ready() if monitor is not None else False,
        checked_at=snapshot.checked_at.isoformat() if snapshot is not None else None,
        build_sha=served_build_sha(cfg),
    )
