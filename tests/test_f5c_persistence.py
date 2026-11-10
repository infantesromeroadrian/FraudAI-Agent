"""F5c.4: configurable SQLite, visible failure in production, persisted feedback, tracked tasks.

Restart is simulated by closing the database and opening the same file again.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI

from fraudai.api.app import init_database
from fraudai.api.auth import create_token
from fraudai.api.routes import router
from fraudai.api.session_manager import SessionManager
from fraudai.core.config import Settings
from fraudai.core.database import Database, DatabaseUnavailableError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

SYNTHETIC_SECRET = "p" * 48  # noqa: S105 - fixture signing key
not_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="permission bits do not bind root"
)


def _cfg(db_path: Path | str, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "development",
        "jwt_secret": SYNTHETIC_SECRET,
        "database_path": str(db_path),
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _rows(path: Path, sql: str) -> list[tuple[Any, ...]]:
    with sqlite3.connect(path) as conn:
        return conn.execute(sql).fetchall()  # noqa: S608 - fixed test SQL


# ---------------------------------------------------------------------------
# Configuration and write checks
# ---------------------------------------------------------------------------


def test_database_path_is_configurable_with_a_backwards_compatible_default() -> None:
    assert Settings(_env_file=None).database_path == "data/fraudai.db"
    assert Settings(_env_file=None, database_path="/data/x.db").database_path == "/data/x.db"


async def test_initialize_creates_the_parent_directory_and_is_writable(tmp_path: Path) -> None:
    db = Database(tmp_path / "nested" / "dir" / "app.db")

    await db.initialize()
    await db.check_writable()
    await db.close()

    assert (tmp_path / "nested" / "dir" / "app.db").is_file()


async def test_parent_that_is_a_file_raises_a_visible_error(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    db = Database(blocker / "app.db")

    with pytest.raises(DatabaseUnavailableError, match="app.db"):
        await db.initialize()


@not_root
async def test_unwritable_directory_raises_a_visible_error(tmp_path: Path) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(DatabaseUnavailableError):
            await Database(locked / "app.db").initialize()
    finally:
        locked.chmod(0o700)


@not_root
async def test_read_only_database_file_is_detected_even_if_tables_exist(tmp_path: Path) -> None:
    path = tmp_path / "ro.db"
    db = Database(path)
    await db.initialize()
    await db.close()
    path.chmod(0o444)
    try:
        with pytest.raises(DatabaseUnavailableError):
            await Database(path).initialize()
    finally:
        path.chmod(0o644)


async def test_check_writable_requires_an_initialised_database(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        await Database(tmp_path / "x.db").check_writable()


# ---------------------------------------------------------------------------
# init_database: development degrades loudly, production refuses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("environment", ["production", "staging"])
async def test_non_development_refuses_to_start_without_a_writable_database(
    environment: str, tmp_path: Path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    cfg = _cfg(blocker / "app.db", environment=environment)

    with pytest.raises(DatabaseUnavailableError):
        await init_database(cfg)


async def test_development_degrades_to_memory_but_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    cfg = _cfg(blocker / "app.db", environment="development")

    with caplog.at_level("ERROR"):
        db = await init_database(cfg)

    assert db is None
    assert any("in-memory" in record.getMessage() for record in caplog.records)


async def test_production_with_a_writable_path_returns_a_ready_database(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path / "ok.db", environment="production")

    db = await init_database(cfg)

    assert db is not None
    await db.check_writable()
    await db.close()


# ---------------------------------------------------------------------------
# Sessions survive a restart; background persistence is tracked and drained
# ---------------------------------------------------------------------------


async def test_sessions_survive_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "restart.db"
    db = Database(path)
    await db.initialize()
    manager = SessionManager(db=db)
    sid = manager.create_session("tenant-a", "pro")
    manager.record_turn(sid, "harvey")
    manager.record_turn(sid, "louis")
    manager.record_upload(sid, "file-1")
    await manager.aclose()
    await db.close()

    reopened = Database(path)
    await reopened.initialize()
    restored = SessionManager(db=reopened)
    assert await restored.load_from_db() == 1
    await reopened.close()

    session = restored.get_session(sid)
    assert session is not None
    assert (session.tenant_id, session.tier) == ("tenant-a", "pro")
    assert session.agent_history == ["harvey", "louis"]
    assert session.uploaded_files == ["file-1"]
    assert session.turn_count == 2


async def test_persistence_tasks_are_referenced_until_done_and_drained_on_close(
    tmp_path: Path,
) -> None:
    db = Database(tmp_path / "tasks.db")
    await db.initialize()
    manager = SessionManager(db=db)

    manager.create_session("tenant-a", "pro")
    assert manager.pending_persist_count() == 1, "fire-and-forget tasks must be referenced"
    await manager.aclose()

    assert manager.pending_persist_count() == 0
    assert _rows(tmp_path / "tasks.db", "SELECT COUNT(*) FROM sessions") == [(1,)]
    await db.close()


async def test_aclose_is_idempotent_and_safe_without_a_database() -> None:
    manager = SessionManager()
    manager.create_session("tenant-a", "pro")

    await manager.aclose()
    await manager.aclose()


async def test_a_failing_database_write_does_not_raise_into_the_request_path() -> None:
    db = MagicMock()
    db.create_session = AsyncMock(side_effect=RuntimeError("disk full"))
    manager = SessionManager(db=db)

    sid = manager.create_session("tenant-a", "pro")
    await manager.aclose()

    assert manager.get_session(sid) is not None


# ---------------------------------------------------------------------------
# Feedback persistence
# ---------------------------------------------------------------------------


@pytest.fixture()
async def stack(tmp_path: Path) -> AsyncIterator[dict[str, Any]]:
    path = tmp_path / "feedback.db"
    db = Database(path)
    await db.initialize()
    manager = SessionManager(db=db)
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.session_manager = manager
    app.state.feedback_store = []
    app.state.db = db
    cfg = _cfg(path)
    with (
        patch("fraudai.core.config.settings", cfg),
        patch("fraudai.api.routes.settings", cfg),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as http:
            yield {"app": app, "http": http, "db": db, "manager": manager, "path": path}
    await manager.aclose()
    await db.close()


def _auth(tenant: str, user: str = "alice") -> dict[str, str]:
    return {"Authorization": f"Bearer {create_token(user, tenant, 'pro')}"}


async def test_feedback_is_persisted_and_survives_a_restart(stack: dict[str, Any]) -> None:
    sid = stack["manager"].create_session("tenant-a", "pro")

    response = await stack["http"].post(
        "/api/v1/feedback",
        json={"session_id": sid, "message_id": "m-1", "rating": 4, "comment": "useful"},
        headers=_auth("tenant-a"),
    )
    assert response.status_code == 201
    await stack["manager"].aclose()
    await stack["db"].close()

    rows = _rows(
        stack["path"], "SELECT session_id, message_id, user_id, rating, comment FROM feedback"
    )
    assert rows == [(sid, "m-1", "alice", 4, "useful")]
    assert len(stack["app"].state.feedback_store) == 1


async def test_a_failed_feedback_write_is_a_visible_503_and_stores_nothing(
    stack: dict[str, Any],
) -> None:
    sid = stack["manager"].create_session("tenant-a", "pro")
    stack["db"].save_feedback = AsyncMock(side_effect=sqlite3.OperationalError("readonly"))

    response = await stack["http"].post(
        "/api/v1/feedback",
        json={"session_id": sid, "message_id": "m-1", "rating": 2},
        headers=_auth("tenant-a"),
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Feedback could not be stored."}
    assert len(stack["app"].state.feedback_store) == 0


async def test_feedback_for_a_foreign_session_is_rejected_and_not_persisted(
    stack: dict[str, Any],
) -> None:
    victim = stack["manager"].create_session("tenant-victim", "pro")

    response = await stack["http"].post(
        "/api/v1/feedback",
        json={"session_id": victim, "message_id": "m-1", "rating": 1},
        headers=_auth("tenant-attacker"),
    )

    assert response.status_code == 404
    assert _rows(stack["path"], "SELECT COUNT(*) FROM feedback") == [(0,)]


async def test_feedback_without_a_database_stays_in_memory(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    manager = SessionManager()
    app.state.session_manager = manager
    app.state.feedback_store = []
    sid = manager.create_session("tenant-a", "pro")
    cfg = _cfg(tmp_path / "unused.db")

    with patch("fraudai.core.config.settings", cfg), patch("fraudai.api.routes.settings", cfg):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as http:
            response = await http.post(
                "/api/v1/feedback",
                json={"session_id": sid, "message_id": "m", "rating": 5},
                headers=_auth("tenant-a"),
            )

    assert response.status_code == 201
    assert len(app.state.feedback_store) == 1


async def test_concurrent_persistence_keeps_every_row(tmp_path: Path) -> None:
    db = Database(tmp_path / "burst.db")
    await db.initialize()
    manager = SessionManager(db=db)

    ids = [manager.create_session("tenant-a", "pro") for _ in range(25)]
    await asyncio.gather(*(asyncio.sleep(0) for _ in range(3)))
    await manager.aclose()
    await db.close()

    stored = {row[0] for row in _rows(tmp_path / "burst.db", "SELECT session_id FROM sessions")}
    assert stored == set(ids)
