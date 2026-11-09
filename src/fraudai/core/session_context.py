"""Trusted session context shared by the API and the tools.

The backend binds the active session only after it has verified that the
caller's tenant owns it. Tools read the binding instead of trusting a session
id or a path supplied by the model, so this module is the single place that
defines what a valid session id is and where a session's uploads live.
"""

from __future__ import annotations

import tempfile
import uuid
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_UPLOADS_DIRNAME = "fraudai_uploads"

# ContextVar values are per asyncio task / thread-context copy, so concurrent
# requests never see each other's binding.
_active_session_id: ContextVar[str | None] = ContextVar("fraudai_active_session_id", default=None)


def is_valid_session_id(value: str) -> bool:
    """Return True only for canonical, server-issued session IDs.

    The server emits ``str(uuid.uuid4())``. Accepting only that exact form
    (lowercase, hyphenated) keeps path separators, ``..`` and alternative UUID
    spellings out of every filesystem and graph-thread key derived from it.
    """
    try:
        return str(uuid.UUID(value)) == value
    except (ValueError, AttributeError, TypeError):
        return False


def uploads_root() -> Path:
    """Base directory that holds one sub-directory per session."""
    return Path(tempfile.gettempdir()) / _UPLOADS_DIRNAME


def upload_dir(session_id: str) -> Path:
    """Directory where the files of *session_id* are stored.

    Raises:
        ValueError: *session_id* is not a canonical server-issued UUID.
    """
    if not is_valid_session_id(session_id):
        msg = "invalid session id"
        raise ValueError(msg)
    return uploads_root() / session_id


def active_session_id() -> str | None:
    """Session bound to the running request, or None outside a bound scope."""
    return _active_session_id.get()


@contextmanager
def bind_session(session_id: str) -> Iterator[None]:
    """Bind *session_id* as the active session for the enclosed scope.

    The previous binding is restored on normal exit, on error and on
    cancellation. Only call this after the ownership check.

    Raises:
        ValueError: *session_id* is not a canonical server-issued UUID.
    """
    if not is_valid_session_id(session_id):
        msg = "invalid session id"
        raise ValueError(msg)
    token = _active_session_id.set(session_id)
    try:
        yield
    finally:
        # An async generator closed from another context cannot reset the
        # token; that context dies with the request, so nothing leaks.
        with suppress(ValueError):
            _active_session_id.reset(token)
