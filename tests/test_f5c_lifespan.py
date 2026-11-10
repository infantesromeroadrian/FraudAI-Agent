"""F5c wiring in the application factory and lifespan.

Heavy collaborators (Qdrant, embeddings, sandbox, graph compilation) are replaced; what is
checked is that the monitor, limiter, database policy and shutdown order are connected.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from prometheus_client import REGISTRY

from fraudai.api.app import create_app, lifespan
from fraudai.api.limits import TenantRateLimiter
from fraudai.core.config import Settings
from fraudai.core.database import DatabaseUnavailableError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

SYNTHETIC_SECRET = "w" * 48  # noqa: S105 - fixture signing key


def _cfg(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "development",
        "jwt_secret": SYNTHETIC_SECRET,
        "database_path": str(tmp_path / "app.db"),
        "llm_provider": "ollama",
        "ollama_host": "http://127.0.0.1:9",  # closed port: connection refused, no network
        "sandbox_mode": "disabled",
        "readiness_interval_seconds": 0.05,
        "rate_limit_requests_per_minute": 7,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@contextmanager
def _patched(cfg: Settings) -> Iterator[None]:
    with (
        patch("fraudai.api.app.settings", cfg),
        patch("fraudai.agents.graph.build_fraud_ai_graph", return_value=MagicMock()),
        patch("fraudai.rag.qdrant_store.QdrantStore.initialize", new_callable=AsyncMock),
        patch(
            "fraudai.rag.qdrant_store.QdrantStore.health_check",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "fraudai.rag.qdrant_store.QdrantStore.get_corpus_version",
            new_callable=AsyncMock,
            return_value="v-test",
        ),
        patch(
            "fraudai.ingestion.embeddings.EmbeddingGenerator", side_effect=RuntimeError("no model")
        ),
    ):
        yield


async def _wait_for_snapshot(app: FastAPI) -> None:
    for _ in range(100):
        if app.state.health_monitor.snapshot is not None:
            return
        await asyncio.sleep(0.02)
    pytest.fail("the monitor never produced a snapshot")


async def test_lifespan_installs_a_running_monitor_and_stops_it_on_shutdown(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    app = FastAPI()

    with _patched(cfg):
        async with lifespan(app):
            monitor = app.state.health_monitor
            task = monitor.task
            assert task is not None and not task.done()
            await _wait_for_snapshot(app)
            snapshot = monitor.snapshot
            assert snapshot.checks["qdrant"] is True
            assert snapshot.checks["llm"] is False  # nothing listens on the closed port
            assert snapshot.checks["database"] is True
            assert snapshot.corpus_version == "v-test"
            assert monitor.is_ready() is False

    assert task.done()
    assert monitor.task is None


async def test_lifespan_installs_the_rate_limiter_from_settings(tmp_path: Path) -> None:
    app = FastAPI()

    with _patched(_cfg(tmp_path)):
        async with lifespan(app):
            limiter = app.state.rate_limiter

    assert isinstance(limiter, TenantRateLimiter)
    assert limiter.limit == 7


async def test_shutdown_drains_session_writes_before_closing_the_database(tmp_path: Path) -> None:
    app = FastAPI()
    order: list[str] = []

    with _patched(_cfg(tmp_path)):
        async with lifespan(app):
            manager = app.state.session_manager
            db = app.state.db
            real_aclose, real_close = manager.aclose, db.close

            async def aclose() -> None:
                order.append("drain")
                await real_aclose()

            async def close() -> None:
                order.append("close")
                await real_close()

            manager.aclose = aclose
            db.close = close
            manager.create_session("tenant-a", "pro")

    assert order == ["drain", "close"]


async def test_production_stops_startup_when_the_database_is_not_writable(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    cfg = _cfg(
        tmp_path,
        environment="production",
        database_path=str(blocker / "app.db"),
    )

    with _patched(cfg), pytest.raises(DatabaseUnavailableError):
        async with lifespan(FastAPI()):
            pytest.fail("the app must not start without its database")


async def test_development_starts_without_a_database_and_reports_it(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    cfg = _cfg(tmp_path, database_path=str(blocker / "app.db"))
    app = FastAPI()

    with _patched(cfg):
        async with lifespan(app):
            assert app.state.db is None
            await _wait_for_snapshot(app)
            assert "database" not in app.state.health_monitor.snapshot.checks


async def test_lifespan_publishes_the_build_sha_in_system_info(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, build_sha="abcdef0123456")

    with _patched(cfg):
        async with lifespan(FastAPI()):
            pass

    value = REGISTRY.get_sample_value(
        "fraudai_info",
        {"version": "0.1.0", "environment": "development", "build_sha": "abcdef0123456"},
    )
    assert value == 1.0


async def test_missing_build_sha_is_published_as_unknown_not_invented(tmp_path: Path) -> None:
    with _patched(_cfg(tmp_path, build_sha="")):
        async with lifespan(FastAPI()):
            pass

    assert (
        REGISTRY.get_sample_value(
            "fraudai_info",
            {"version": "0.1.0", "environment": "development", "build_sha": "unknown"},
        )
        == 1.0
    )


async def test_non_development_warns_when_login_or_scrape_are_not_configured(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _cfg(tmp_path, environment="production", jwt_secret=SYNTHETIC_SECRET)

    with _patched(cfg), caplog.at_level("WARNING"):
        async with lifespan(FastAPI()):
            pass

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "AUTH_USERNAME" in messages
    assert "METRICS_SCRAPE_TOKEN" in messages


def test_create_app_configures_logging_once_from_settings(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, log_format="json", log_level="DEBUG")

    with patch("fraudai.api.app.settings", cfg), patch("fraudai.api.app.configure_logging") as conf:
        create_app()

    conf.assert_called_once_with(level="DEBUG", fmt="json")
