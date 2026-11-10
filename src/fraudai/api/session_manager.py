"""Session management for FraudAI Agent API.

Tracks session metadata, turn history, and uploaded file references.
Uses an in-memory cache backed by an optional async SQLite database
for persistence across restarts.

Graceful degradation: if the database is unavailable or any DB
operation fails, the in-memory cache continues to work as before.

Reference: SR-009 (tenant isolation), ADR-003 (LangGraph thread_id mapping).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import Counter
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from fraudai.core.database import MAX_AGENT_HISTORY
from fraudai.core.metrics import ACTIVE_SESSIONS

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine
    from typing import Any

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
_MAX_SESSIONS_PER_TENANT = 200


class SessionManager:
    """Manages chat sessions with metadata tracking.

    Single-process: no cross-process locking, and every limit below is per process.

    Limits: ``max_sessions_per_tenant`` evicts that tenant's own oldest session; the global
    ``max_sessions`` evicts the oldest session of the tenant holding the most, so one tenant
    flooding the service cannot push out the others. ``max_agent_history`` keeps only the
    newest history entries (``turn_count`` keeps the true total).

    When constructed with a ``Database`` instance, all mutations are persisted to SQLite by
    background tasks that this manager keeps referenced; ``aclose`` waits for them. If the
    database is ``None`` (or a write fails), the in-memory cache keeps working.
    """

    def __init__(
        self,
        db: Database | None = None,
        *,
        max_sessions: int = _MAX_SESSIONS,
        max_sessions_per_tenant: int = _MAX_SESSIONS_PER_TENANT,
        max_agent_history: int = MAX_AGENT_HISTORY,
    ) -> None:
        self._db = db
        self._sessions: dict[str, SessionData] = {}
        self._max_sessions = max_sessions
        self._max_per_tenant = max_sessions_per_tenant
        self._max_history = max_agent_history
        # Strong references: the event loop only keeps weak ones, so an unreferenced
        # fire-and-forget task can be collected before it finishes.
        self._tasks: set[asyncio.Task[None]] = set()
        # The DB helpers read-modify-write one row; the lock keeps them in creation order
        # so two quick writes to the same session cannot overwrite each other.
        self._db_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Background persistence
    # ------------------------------------------------------------------

    def _persist(self, operation: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """Run a DB write in a referenced background task (skipped without DB or loop)."""
        if self._db is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # No event loop: nothing can run the write; memory stays authoritative.
        task = loop.create_task(operation())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def pending_persist_count(self) -> int:
        """Number of database writes still running in the background."""
        return len(self._tasks)

    async def aclose(self) -> None:
        """Wait for every pending database write (call before closing the database)."""
        while self._tasks:
            pending = list(self._tasks)
            await asyncio.gather(*pending, return_exceptions=True)
            self._tasks.difference_update(pending)

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def _evict(self, session_id: str, reason: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        self._persist(lambda: self._db_delete_session(session_id))
        logger.info(
            "Session evicted (%s): session_id=%s, tenant_id=%s",
            reason,
            session_id,
            session.tenant_id,
        )

    def _make_room_for(self, tenant_id: str) -> None:
        """Apply the per-tenant quota, then the global cap, before adding a session."""
        own = [sid for sid, s in self._sessions.items() if s.tenant_id == tenant_id]
        if len(own) >= self._max_per_tenant:
            self._evict(own[0], "tenant quota")
        if len(self._sessions) >= self._max_sessions:
            holders = Counter(s.tenant_id for s in self._sessions.values())
            heaviest = max(holders, key=lambda tenant: holders[tenant])
            victim = next(sid for sid, s in self._sessions.items() if s.tenant_id == heaviest)
            self._evict(victim, "global cap")

    def create_session(self, tenant_id: str, tier: str) -> str:
        """Create a new session and return its ID."""
        self._make_room_for(tenant_id)

        session_id = str(uuid.uuid4())
        self._sessions[session_id] = SessionData(
            session_id=session_id,
            tenant_id=tenant_id,
            tier=tier,
        )
        self._persist(lambda: self._db_create_session(session_id, tenant_id, tier))
        ACTIVE_SESSIONS.set(len(self._sessions))

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
            self._make_room_for(tenant_id)
            self._sessions[session_id] = SessionData(
                session_id=session_id,
                tenant_id=tenant_id,
                tier=tier,
            )
            self._persist(lambda: self._db_create_session(session_id, tenant_id, tier))
            ACTIVE_SESSIONS.set(len(self._sessions))
            logger.info("Session registered via ensure_session: %s", session_id)
        return session_id

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        removed = self._sessions.pop(session_id, None)
        if removed:
            self._persist(lambda: self._db_delete_session(session_id))
            ACTIVE_SESSIONS.set(len(self._sessions))
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
        del session.agent_history[: -self._max_history]
        session.turn_count += 1
        self._persist(lambda: self._db_record_turn(session_id, agent))

    def record_upload(self, session_id: str, filename: str) -> None:
        """Record a file upload in the session."""
        session = self._sessions.get(session_id)
        if session is None:
            logger.warning("record_upload called for unknown session: %s", session_id)
            return
        session.uploaded_files.append(filename)
        self._persist(lambda: self._db_record_upload(session_id, filename))

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
            ACTIVE_SESSIONS.set(len(self._sessions))
            logger.info("Loaded %d sessions from database", count)
            return count
        except Exception:
            logger.exception("Failed to load sessions from database, starting fresh")
            return 0

    # ------------------------------------------------------------------
    # DB helpers (async, run as tracked background tasks)
    # ------------------------------------------------------------------

    async def _db_create_session(self, session_id: str, tenant_id: str, tier: str) -> None:
        try:
            assert self._db is not None  # noqa: S101
            async with self._db_lock:
                await self._db.create_session(session_id, tenant_id, tier)
        except Exception:
            logger.exception("DB: failed to persist session %s", session_id)

    async def _db_delete_session(self, session_id: str) -> None:
        try:
            assert self._db is not None  # noqa: S101
            async with self._db_lock:
                await self._db.delete_session(session_id)
        except Exception:
            logger.exception("DB: failed to delete session %s from DB", session_id)

    async def _db_record_turn(self, session_id: str, agent: str) -> None:
        try:
            assert self._db is not None  # noqa: S101
            async with self._db_lock:
                await self._db.record_turn(session_id, agent, max_history=self._max_history)
        except Exception:
            logger.exception("DB: failed to record turn for session %s", session_id)

    async def _db_record_upload(self, session_id: str, filename: str) -> None:
        try:
            assert self._db is not None  # noqa: S101
            async with self._db_lock:
                await self._db.record_upload(session_id, filename)
        except Exception:
            logger.exception("DB: failed to record upload for session %s", session_id)
