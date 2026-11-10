"""JSON logging with a correlation id (F5c.3).

One JSON object per line on stderr, with the correlation id set by the request middleware.
No remote shipping and no extra service: the container runtime collects stderr.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from fraudai.core.tracing import CorrelationFilter

if TYPE_CHECKING:
    from typing import TextIO

_HANDLER_MARK = "_fraudai_log_handler"
_TEXT_FORMAT = "%(asctime)s [%(levelname)s] [cid=%(correlation_id)s] %(name)s: %(message)s"
# uvicorn installs its own handlers and does not propagate; route them through ours so the
# whole process emits one format.
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


class JsonFormatter(logging.Formatter):
    """Format a record as a single-line JSON object (newlines are escaped by JSON)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", ""),
        }
        if record.exc_info and record.exc_info[0] is not None:
            payload["exc_type"] = record.exc_info[0].__name__
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(
    level: str = "INFO",
    fmt: Literal["json", "text"] = "json",
    *,
    root: logging.Logger | None = None,
    stream: TextIO | None = None,
) -> None:
    """Install (once) a handler with the correlation filter and the chosen format.

    Idempotent: a second call replaces our handler instead of stacking another one.
    """
    target = root if root is not None else logging.getLogger()
    for existing in [h for h in target.handlers if getattr(h, _HANDLER_MARK, False)]:
        target.removeHandler(existing)

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.addFilter(CorrelationFilter())
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter(_TEXT_FORMAT))
    setattr(handler, _HANDLER_MARK, True)
    target.addHandler(handler)
    target.setLevel(level.upper())

    if root is None:
        for name in _UVICORN_LOGGERS:
            uvicorn_logger = logging.getLogger(name)
            uvicorn_logger.handlers = [handler]
            uvicorn_logger.propagate = False
