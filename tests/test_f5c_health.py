"""F5c.1: /livez, /readyz and /health served from a cached snapshot.

No probe may reach a dependency: a background monitor refreshes the snapshot and the
routes only read it. ``build_sha`` comes from the image metadata (``BUILD_SHA``) and is
``None`` when absent, never invented.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI

from fraudai.api.health import HealthMonitor, build_checks, served_build_sha
from fraudai.api.routes import router
from fraudai.core.config import Settings

if TYPE_CHECKING:
    from collections.abc import Callable

VALID_SHA = "3795f363e6bcebe3838b319cfe1a03d923f46f8f"
INTERVAL = 15.0


class _Probe:
    """A dependency check that counts its calls and can be switched on and off."""

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.calls = 0

    async def __call__(self) -> bool:
        self.calls += 1
        return self.ok


class _Clock:
    def __init__(self) -> None:
        self.value = 1_000.0

    def __call__(self) -> float:
        return self.value


def _monitor(
    checks: dict[str, Callable[[], Any]],
    required: set[str],
    clock: _Clock | None = None,
    **kwargs: Any,
) -> HealthMonitor:
    return HealthMonitor(
        checks=checks,
        required=frozenset(required),
        interval_seconds=INTERVAL,
        clock=clock or _Clock(),
        now=lambda: datetime(2026, 11, 10, 12, 0, tzinfo=UTC),
        **kwargs,
    )


def _cfg(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)


def _app(monitor: HealthMonitor | None, cfg: Settings | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    if monitor is not None:
        app.state.health_monitor = monitor
    app.state.settings = cfg or _cfg()
    return app


@pytest.fixture()
def build_sha_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fraudai.api.routes.settings", _cfg(build_sha=VALID_SHA))


async def _get(app: FastAPI, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        return await http.get(f"/api/v1{path}")


# ---------------------------------------------------------------------------
# /livez
# ---------------------------------------------------------------------------


async def test_livez_is_200_without_any_dependency_or_monitor() -> None:
    response = await _get(_app(None), "/livez")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


async def test_livez_stays_200_while_every_dependency_is_down() -> None:
    probes = {"qdrant": _Probe(False), "llm": _Probe(False)}
    monitor = _monitor(dict(probes), {"qdrant", "llm"})
    await monitor.refresh()

    response = await _get(_app(monitor), "/livez")

    assert response.status_code == 200
    assert probes["qdrant"].calls == 1  # only the refresh touched it


# ---------------------------------------------------------------------------
# /readyz
# ---------------------------------------------------------------------------


async def test_readyz_is_503_starting_before_the_first_check_finishes() -> None:
    monitor = _monitor({"qdrant": _Probe()}, {"qdrant"})

    response = await _get(_app(monitor), "/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "starting"
    assert body["checked_at"] is None
    assert body["checks"] == {}


async def test_readyz_is_503_when_there_is_no_monitor_at_all() -> None:
    response = await _get(_app(None), "/readyz")

    assert response.status_code == 503
    assert response.json()["status"] == "starting"


async def test_readyz_is_200_when_every_required_check_passes() -> None:
    monitor = _monitor({"qdrant": _Probe(), "llm": _Probe()}, {"qdrant", "llm"})
    await monitor.refresh()

    response = await _get(_app(monitor), "/readyz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"] == {"qdrant": True, "llm": True}
    assert body["checked_at"] == "2026-11-10T12:00:00+00:00"


async def test_readyz_is_503_and_names_the_failed_required_check() -> None:
    monitor = _monitor({"qdrant": _Probe(), "llm": _Probe(False)}, {"qdrant", "llm"})
    await monitor.refresh()

    response = await _get(_app(monitor), "/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"] == {"qdrant": True, "llm": False}


async def test_a_failing_optional_check_does_not_block_readiness() -> None:
    monitor = _monitor({"qdrant": _Probe(), "ollama": _Probe(False)}, {"qdrant"})
    await monitor.refresh()

    response = await _get(_app(monitor), "/readyz")

    assert response.status_code == 200
    assert response.json()["checks"]["ollama"] is False


async def test_initial_failure_then_recovery() -> None:
    qdrant = _Probe(False)
    monitor = _monitor({"qdrant": qdrant}, {"qdrant"})
    app = _app(monitor)
    await monitor.refresh()
    assert (await _get(app, "/readyz")).status_code == 503

    qdrant.ok = True
    await monitor.refresh()

    assert (await _get(app, "/readyz")).status_code == 200


async def test_probes_are_served_from_cache_and_never_call_dependencies() -> None:
    qdrant, llm = _Probe(), _Probe()
    monitor = _monitor({"qdrant": qdrant, "llm": llm}, {"qdrant", "llm"})
    await monitor.refresh()
    app = _app(monitor)

    for path in ("/readyz", "/health", "/readyz", "/health", "/livez"):
        for _ in range(5):
            await _get(app, path)

    assert (qdrant.calls, llm.calls) == (1, 1)


async def test_a_raising_check_counts_as_failed_and_does_not_break_the_others() -> None:
    async def boom() -> bool:
        raise ConnectionError("synthetic outage")

    ok = _Probe()
    monitor = _monitor({"qdrant": boom, "llm": ok}, {"qdrant", "llm"})

    snapshot = await monitor.refresh()

    assert snapshot.checks == {"qdrant": False, "llm": True}
    assert monitor.is_ready() is False


async def test_a_hanging_check_times_out_as_failed() -> None:
    async def hang() -> bool:
        await asyncio.sleep(30)
        return True

    monitor = _monitor({"qdrant": hang}, {"qdrant"}, check_timeout_seconds=0.05)

    snapshot = await asyncio.wait_for(monitor.refresh(), timeout=5)

    assert snapshot.checks == {"qdrant": False}


async def test_a_stale_snapshot_is_not_ready() -> None:
    clock = _Clock()
    monitor = _monitor({"qdrant": _Probe()}, {"qdrant"}, clock=clock)
    await monitor.refresh()
    assert monitor.is_ready() is True

    clock.value += 3 * INTERVAL + 1

    assert monitor.is_ready() is False
    response = await _get(_app(monitor), "/readyz")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


async def test_readyz_reports_the_build_sha(build_sha_env: None) -> None:
    monitor = _monitor({"qdrant": _Probe()}, {"qdrant"})
    await monitor.refresh()

    response = await _get(_app(monitor), "/readyz")

    assert response.json()["build_sha"] == VALID_SHA


async def test_readyz_build_sha_is_null_when_the_image_carries_none() -> None:
    monitor = _monitor({"qdrant": _Probe()}, {"qdrant"})
    await monitor.refresh()

    response = await _get(_app(monitor), "/readyz")

    assert response.json()["build_sha"] is None


# ---------------------------------------------------------------------------
# build_sha contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (VALID_SHA, VALID_SHA),
        ("3795f36", "3795f36"),
        ("", None),
        ("   ", None),
        ("unknown", None),
        ("3795F363E6BCEBE3838B319CFE1A03D923F46F8F", None),
        ("3795f3", None),
        (VALID_SHA + "0" * 30, None),
        ("3795f363; rm -rf /", None),
        (f"{VALID_SHA}\n", VALID_SHA),
    ],
    ids=[
        "full",
        "short",
        "empty",
        "blank",
        "word",
        "uppercase",
        "too_short",
        "too_long",
        "inject",
        "newline",
    ],
)
def test_served_build_sha_is_the_env_value_or_none(raw: str, expected: str | None) -> None:
    assert served_build_sha(_cfg(build_sha=raw)) == expected


# ---------------------------------------------------------------------------
# /health (compatible, cached, honest)
# ---------------------------------------------------------------------------


async def test_health_keeps_the_old_fields_and_adds_the_new_ones(build_sha_env: None) -> None:
    monitor = _monitor(
        {"qdrant": _Probe(), "ollama": _Probe(), "llm": _Probe()},
        {"qdrant", "llm"},
        corpus_version_probe=AsyncMock(return_value="v2026.11"),
    )
    await monitor.refresh()

    response = await _get(_app(monitor), "/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "qdrant": True,
        "ollama": True,
        "claude_api": True,
        "corpus_version": "v2026.11",
        "ready": True,
        "checked_at": "2026-11-10T12:00:00+00:00",
        "build_sha": VALID_SHA,
    }


@pytest.mark.parametrize(
    ("qdrant", "ollama", "llm", "expected"),
    [
        (True, True, True, "healthy"),
        (False, True, True, "degraded"),
        (True, False, False, "degraded"),
        (False, False, False, "unhealthy"),
    ],
)
async def test_health_status_summarises_the_three_components(
    qdrant: bool, ollama: bool, llm: bool, expected: str
) -> None:
    monitor = _monitor(
        {"qdrant": _Probe(qdrant), "ollama": _Probe(ollama), "llm": _Probe(llm)},
        {"qdrant", "llm"},
    )
    await monitor.refresh()

    body = (await _get(_app(monitor), "/health")).json()

    assert body["status"] == expected
    assert body["ready"] is (qdrant and llm)


async def test_health_before_any_check_is_unhealthy_with_no_timestamp() -> None:
    monitor = _monitor({"qdrant": _Probe()}, {"qdrant"})

    response = await _get(_app(monitor), "/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unhealthy"
    assert body["ready"] is False
    assert body["checked_at"] is None
    assert (body["qdrant"], body["ollama"], body["claude_api"]) == (False, False, False)


# ---------------------------------------------------------------------------
# Background monitor lifecycle
# ---------------------------------------------------------------------------


async def test_monitor_refreshes_in_the_background_and_stop_closes_the_task() -> None:
    probe = _Probe()
    monitor = HealthMonitor(
        checks={"qdrant": probe}, required=frozenset({"qdrant"}), interval_seconds=0.01
    )

    monitor.start()
    task = monitor.task
    assert task is not None, "the loop task must be kept referenced"
    await asyncio.sleep(0.15)
    await monitor.stop()

    assert probe.calls >= 3
    assert task.done()
    assert monitor.task is None


async def test_stop_is_idempotent_and_safe_before_start() -> None:
    monitor = _monitor({"qdrant": _Probe()}, {"qdrant"})

    await monitor.stop()
    monitor.start()
    await monitor.stop()
    await monitor.stop()


async def test_start_twice_keeps_a_single_loop() -> None:
    monitor = HealthMonitor(
        checks={"qdrant": _Probe()}, required=frozenset({"qdrant"}), interval_seconds=0.01
    )

    monitor.start()
    first = monitor.task
    monitor.start()

    assert monitor.task is first
    await monitor.stop()


async def test_a_failing_refresh_does_not_kill_the_loop() -> None:
    calls = 0

    async def corpus() -> str | None:
        nonlocal calls
        calls += 1
        raise RuntimeError("synthetic")

    monitor = HealthMonitor(
        checks={"qdrant": _Probe()},
        required=frozenset({"qdrant"}),
        interval_seconds=0.01,
        corpus_version_probe=corpus,
    )

    monitor.start()
    await asyncio.sleep(0.1)
    task = monitor.task
    assert task is not None
    assert not task.done()
    await monitor.stop()

    assert calls >= 2


# ---------------------------------------------------------------------------
# build_checks: what the real checks talk to
# ---------------------------------------------------------------------------


def _recording_transport(
    status_by_host: dict[str, int],
) -> tuple[httpx.MockTransport, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status_by_host.get(request.url.host, 500), json={})

    return httpx.MockTransport(handler), seen


def _store(ok: bool = True) -> MagicMock:
    store = MagicMock()
    store.health_check = AsyncMock(return_value=ok)
    store.get_corpus_version = AsyncMock(return_value="v1")
    return store


async def test_ollama_provider_checks_only_the_configured_daemon() -> None:
    transport, seen = _recording_transport({"daemon.test": 200})
    cfg = _cfg(
        llm_provider="ollama",
        llm_model="gpt-oss:120b-cloud",
        ollama_host="http://daemon.test:11434",
        groq_api_key="synthetic-groq-key",
        anthropic_api_key="synthetic-anthropic-key",
    )
    checks, required = build_checks(store=_store(), db=None, cfg=cfg, transport=transport)

    results = {name: await check() for name, check in checks.items()}

    assert results["ollama"] is True
    assert results["llm"] is True
    assert {r.url.host for r in seen} == {"daemon.test"}
    assert all(r.url.path == "/api/tags" for r in seen)
    assert all("synthetic" not in str(r.headers) for r in seen), "foreign keys must not leave"
    assert required == frozenset({"qdrant", "llm"})


async def test_ollama_daemon_down_fails_the_llm_check() -> None:
    transport, _ = _recording_transport({"daemon.test": 503})
    cfg = _cfg(llm_provider="ollama", ollama_host="http://daemon.test:11434")
    checks, _ = build_checks(store=_store(), db=None, cfg=cfg, transport=transport)

    assert await checks["llm"]() is False


async def test_groq_without_a_key_is_not_ready_and_makes_no_call() -> None:
    transport, seen = _recording_transport({"api.groq.com": 200})
    cfg = _cfg(llm_provider="groq", groq_api_key="")
    checks, _ = build_checks(store=_store(), db=None, cfg=cfg, transport=transport)

    assert await checks["llm"]() is False
    assert [r for r in seen if r.url.host == "api.groq.com"] == []


async def test_groq_with_a_key_sends_it_only_to_groq() -> None:
    transport, seen = _recording_transport({"api.groq.com": 200, "localhost": 200})
    cfg = _cfg(llm_provider="groq", groq_api_key="synthetic-groq-key")
    checks, _ = build_checks(store=_store(), db=None, cfg=cfg, transport=transport)

    assert await checks["llm"]() is True
    groq = [r for r in seen if r.url.host == "api.groq.com"]
    assert len(groq) == 1
    assert groq[0].headers["authorization"] == "Bearer synthetic-groq-key"


async def test_database_is_checked_and_required_only_when_configured() -> None:
    db = MagicMock()
    db.check_writable = AsyncMock(return_value=None)
    cfg = _cfg(llm_provider="ollama")
    transport, _ = _recording_transport({"localhost": 200})

    with_db, required_with = build_checks(store=_store(), db=db, cfg=cfg, transport=transport)
    without_db, required_without = build_checks(
        store=_store(), db=None, cfg=cfg, transport=transport
    )

    assert await with_db["database"]() is True
    db.check_writable.assert_awaited_once()
    assert "database" in required_with
    assert "database" not in without_db
    assert "database" not in required_without


async def test_database_check_fails_when_the_database_is_not_writable() -> None:
    db = MagicMock()
    db.check_writable = AsyncMock(side_effect=OSError("read-only"))
    transport, _ = _recording_transport({"localhost": 200})
    checks, _ = build_checks(
        store=_store(), db=db, cfg=_cfg(llm_provider="ollama"), transport=transport
    )

    assert await checks["database"]() is False


async def test_qdrant_check_delegates_to_the_store() -> None:
    transport, _ = _recording_transport({"localhost": 200})
    checks, _ = build_checks(
        store=_store(ok=False), db=None, cfg=_cfg(llm_provider="ollama"), transport=transport
    )

    assert await checks["qdrant"]() is False
