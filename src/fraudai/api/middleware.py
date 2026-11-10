"""FastAPI middleware for Prometheus metrics and request tracing.

Records per-request latency and counts in Prometheus histograms /
counters, and propagates a correlation ID via ``X-Correlation-ID``
header for log-based distributed tracing.

Reference: F2 spec section 5.1 -- SLA monitoring.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from fraudai.core.metrics import REQUEST_COUNT, REQUEST_LATENCY
from fraudai.core.tracing import correlation_id, new_correlation_id

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response


# An inbound id is only trusted if it is a plain token: it is echoed in a response header
# and written to every log line, so anything else (spaces, CR/LF, huge values) is replaced.
_INBOUND_CORRELATION_ID = re.compile(r"[A-Za-z0-9_-]{8,64}")
UNMATCHED_ENDPOINT = "unmatched"
# The method label comes from the client: anything outside the standard set shares one value.
_KNOWN_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})
UNKNOWN_METHOD = "OTHER"


_PATH_PARAM = re.compile(r"\{([^}:]+)(?::[^}]*)?\}")


def _endpoint_label(request: Request) -> str:
    """Full route template, e.g. ``/api/v1/sessions/{session_id}``; ``unmatched`` if none.

    FastAPI's matched route carries only the path relative to ``include_router(prefix=...)``,
    so the prefix is recovered from the real path by re-expanding the template with the
    path parameters and stripping that suffix.
    """
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    if not template:
        return UNMATCHED_ENDPOINT
    params = request.scope.get("path_params") or {}
    try:
        concrete = _PATH_PARAM.sub(lambda match: str(params[match.group(1)]), template)
    except KeyError:
        return str(template)
    path = str(request.scope.get("path", ""))
    if concrete and path.endswith(concrete):
        return path[: len(path) - len(concrete)] + str(template)
    return str(template)


def _start_correlation(request: Request) -> str:
    inbound = request.headers.get("x-correlation-id", "")
    if _INBOUND_CORRELATION_ID.fullmatch(inbound):
        correlation_id.set(inbound)
        return inbound
    return new_correlation_id()


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware that records Prometheus metrics for every request.

    For each incoming request this middleware:
    1. Adopts a well-formed inbound ``X-Correlation-ID`` or generates one, and stores it
       in the context so every log line of the request carries it.
    2. Measures wall-clock latency.
    3. Records ``fraudai_request_duration_seconds`` (histogram).
    4. Increments ``fraudai_requests_total`` (counter) with status code.
    5. Adds ``X-Correlation-ID`` to the response headers.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        cid = _start_correlation(request)
        method = request.method.upper()
        if method not in _KNOWN_METHODS:
            method = UNKNOWN_METHOD

        start = time.monotonic()
        response: Response = await call_next(request)

        # Label by the matched route template to keep cardinality bounded
        # (e.g. "/api/v1/sessions/{session_id}"). Requests that match no route share one
        # label: the raw path is attacker-controlled and would create a series per probe.
        endpoint = _endpoint_label(request)
        duration = time.monotonic() - start

        status_code = str(response.status_code)

        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code,
        ).inc()

        response.headers["X-Correlation-ID"] = cid

        return response
