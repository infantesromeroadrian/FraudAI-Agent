"""Session management for FraudAI Agent API.

Tracks session metadata, turn history, and uploaded file references.
Uses an in-memory cache backed by an optional async SQLite database
for persistence across restarts.

Graceful degradation: if the database is unavailable or any DB
operation fails, the in-memory cache continues to work as before.

Reference: SR-009 (tenant isolation), ADR-003 (LangGraph thread_id mapping).
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from fraudai.core.database import Database

logger = logging.getLogger(__name__)


class SessionData(BaseModel):
    """Metadata for a single chat session."""

    session_id: str
    tenant_id: str
    tier: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )
    agent_history: list[str] = Field(default_factory=list)
    uploaded_files: list[str] = Field(default_factory=list)
    turn_count: int = 0


_MAX_SESSIONS = 1000


class SessionManager:
    """Manages chat sessions with metadata tracking.

    Thread-safe for single-process async usage (no cross-process locking).
    Evicts the oldest session when ``_MAX_SESSIONS`` is exceeded.

    When constructed with a ``Database`` instance, all mutations are
    persisted to SQLite. If the database is ``None`` (or any DB call
    fails), the manager falls back to the in-memory cache only.
    """

    def __init__(self, db: Database | None = None) -> None:
        self._db = db
        self._sessions: dict[str, SessionData] = {}

    def create_session(self, tenant_id: str, tier: str) -> str:
        """Create a new session and return its ID."""
        if len(self._sessions) >= _MAX_SESSIONS:
            oldest_key = next(iter(self._sessions))
            del self._sessions[oldest_key]
            logger.info("Session evicted (max %d reached): %s", _MAX_SESSIONS, oldest_key)

        session_id = str(uuid.uuid4())
        self._sessions[session_id] = SessionData(
            session_id=session_id,
            tenant_id=tenant_id,
            tier=tier,
        )

        # Persist to DB (fire-and-forget via task to keep sync signature)
        if self._db is not None:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._db_create_session(session_id, tenant_id, tier))
            except RuntimeError:
                pass  # No event loop -- skip DB persist

        logger.info(
            "Session created: session_id=%s, tenant_id=%s, tier=%s",
            session_id,
            tenant_id,
            tier,
        )
        return session_id

    def get_session(self, session_id: str) -> SessionData | None:
        """Return session data or None if not found."""
        return self._sessions.get(session_id)

    def ensure_session(self, session_id: str, tenant_id: str, tier: str) -> str:
        """Register a session under a caller-chosen ID (internal use only).

        Never call this with an ID taken from a request: the API routes only
        accept IDs issued by ``create_session`` and owned by the caller's
        tenant. Kept for restore/migration paths that already trust the ID.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionData(
                session_id=session_id,
                tenant_id=tenant_id,
                tier=tier,
            )
            if self._db is not None:
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._db_create_session(session_id, tenant_id, tier))
                except RuntimeError:
                    pass
            logger.info("Session registered via ensure_session: %s", session_id)
        return session_id

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        removed = self._sessions.pop(session_id, None)
        if removed:
            if self._db is not None:
                import asyncio

                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._db_delete_session(session_id))
                except RuntimeError:
                    pass
            logger.info("Session deleted: session_id=%s", session_id)
            return True
        return False

    def record_turn(self, session_id: str, agent: str) -> None:
        """Record an agent turn in the session history."""
        session = self._sessions.get(session_id)
        if session is None:
            logger.warning("record_turn called for unknown session: %s", session_id)
            return
        session.agent_history.append(agent)
        session.turn_count += 1

        if self._db is not None:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._db_record_turn(session_id, agent))
            except RuntimeError:
                pass

    def record_upload(self, session_id: str, filename: str) -> None:
        """Record a file upload in the session."""
        session = self._sessions.get(session_id)
        if session is None:
            logger.warning("record_upload called for unknown session: %s", session_id)
            return
        session.uploaded_files.append(filename)

        if self._db is not None:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._db_record_upload(session_id, filename))
            except RuntimeError:
                pass

    def list_sessions(self, tenant_id: str | None = None) -> list[SessionData]:
        """List all sessions, optionally filtered by tenant."""
        if tenant_id is None:
            return list(self._sessions.values())
        return [s for s in self._sessions.values() if s.tenant_id == tenant_id]

    async def load_from_db(self) -> int:
        """Hydrate the in-memory cache from the database.

        Called during startup to restore sessions that survived a restart.
        Returns the number of sessions loaded.
        """
        if self._db is None:
            return 0
        try:
            db = self._db._ensure_connected()
            cursor = await db.execute("SELECT * FROM sessions ORDER BY created_at")
            rows = await cursor.fetchall()
            count = 0
            for row in rows:
                data = self._db._row_to_session_dict(row)
                self._sessions[data["session_id"]] = SessionData(**data)
                count += 1
            logger.info("Loaded %d sessions from database", count)
            return count
        except Exception:
            logger.exception("Failed to load sessions from database, starting fresh")
            return 0

    # ------------------------------------------------------------------
    # DB helpers (async, called via create_task)
    # ------------------------------------------------------------------

    async def _db_create_session(self, session_id: str, tenant_id: str, tier: str) -> None:
        try:
            assert self._db is not None  # noqa: S101
            await self._db.create_session(session_id, tenant_id, tier)
        except Exception:
            logger.exception("DB: failed to persist session %s", session_id)

    async def _db_delete_session(self, session_id: str) -> None:
        try:
            assert self._db is not None  # noqa: S101
            await self._db.delete_session(session_id)
        except Exception:
            logger.exception("DB: failed to delete session %s from DB", session_id)

    async def _db_record_turn(self, session_id: str, agent: str) -> None:
        try:
            assert self._db is not None  # noqa: S101
            await self._db.record_turn(session_id, agent)
        except Exception:
            logger.exception("DB: failed to record turn for session %s", session_id)

    async def _db_record_upload(self, session_id: str, filename: str) -> None:
        try:
            assert self._db is not None  # noqa: S101
            await self._db.record_upload(session_id, filename)
        except Exception:
            logger.exception("DB: failed to record upload for session %s", session_id)
