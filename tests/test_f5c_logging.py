"""F5c.3: JSON logs with correlation id, generic stream errors, user text kept out of INFO logs.

Sandbox rejections are not exceptions: their explicit payload must survive, while unexpected
internal exceptions never reach the client.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import uuid
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage
from starlette.responses import StreamingResponse

from fraudai.agents.claude_invoker import ClaudeAgentInvoker
from fraudai.api.auth import create_token
from fraudai.api.logging_config import JsonFormatter, configure_logging
from fraudai.api.middleware import MetricsMiddleware
from fraudai.api.routes import router
from fraudai.api.session_manager import SessionManager
from fraudai.core.config import Settings
from fraudai.core.tracing import CorrelationFilter, correlation_id

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

SYNTHETIC_SECRET = "g" * 48  # noqa: S105 - fixture signing key
USER_TEXT = "MARKER-private-fraud-case-4711"


def _record(msg: str = "hello", **kwargs: Any) -> logging.LogRecord:
    return logging.LogRecord("fraudai.test", logging.INFO, __file__, 1, msg, (), None, **kwargs)


def _format(record: logging.LogRecord) -> dict[str, Any]:
    CorrelationFilter().filter(record)
    line = JsonFormatter().format(record)
    assert "\n" not in line, "one record, one line"
    parsed: dict[str, Any] = json.loads(line)
    return parsed


# ---------------------------------------------------------------------------
# JsonFormatter / configure_logging
# ---------------------------------------------------------------------------


def test_record_becomes_one_json_line_with_the_fixed_fields() -> None:
    token = correlation_id.set("abc12345")
    try:
        payload = _format(_record("hello %s"))
    finally:
        correlation_id.reset(token)

    assert payload["level"] == "INFO"
    assert payload["logger"] == "fraudai.test"
    assert payload["message"] == "hello %s"
    assert payload["correlation_id"] == "abc12345"
    assert payload["timestamp"].endswith("+00:00")


def test_message_arguments_are_interpolated_and_newlines_cannot_forge_records() -> None:
    record = logging.LogRecord(
        "fraudai.test", logging.WARNING, __file__, 1, "user=%s", ('a\n{"level": "FAKE"}',), None
    )

    payload = _format(record)

    assert payload["level"] == "WARNING"
    assert payload["message"] == 'user=a\n{"level": "FAKE"}'


def test_exceptions_are_serialised_with_type_and_traceback() -> None:
    try:
        raise ValueError("synthetic failure")
    except ValueError:
        import sys

        record = logging.LogRecord(
            "fraudai.test", logging.ERROR, __file__, 1, "failed", (), sys.exc_info()
        )

    payload = _format(record)

    assert payload["exc_type"] == "ValueError"
    assert "synthetic failure" in payload["exception"]
    assert "Traceback" in payload["exception"]


def test_missing_correlation_id_is_an_empty_string() -> None:
    assert _format(_record())["correlation_id"] == ""


def test_configure_logging_is_idempotent_and_json() -> None:
    root = logging.Logger("isolated-root")
    stream = io.StringIO()

    configure_logging(level="INFO", fmt="json", root=root, stream=stream)
    configure_logging(level="INFO", fmt="json", root=root, stream=stream)
    root.info("one line")

    lines = stream.getvalue().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["message"] == "one line"


def test_configure_logging_text_format_is_available_for_local_reading() -> None:
    root = logging.Logger("isolated-root-text")
    stream = io.StringIO()

    configure_logging(level="INFO", fmt="text", root=root, stream=stream)
    root.info("plain line")

    assert "plain line" in stream.getvalue()
    with pytest.raises(json.JSONDecodeError):
        json.loads(stream.getvalue().strip())


def test_configure_logging_applies_the_level() -> None:
    root = logging.Logger("isolated-root-level")
    stream = io.StringIO()

    configure_logging(level="WARNING", fmt="json", root=root, stream=stream)
    root.info("hidden")
    root.warning("shown")

    assert [json.loads(line)["message"] for line in stream.getvalue().splitlines()] == ["shown"]


# ---------------------------------------------------------------------------
# Correlation id through the middleware
# ---------------------------------------------------------------------------


def _correlated_app(sink: list[dict[str, Any]]) -> FastAPI:
    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            sink.append(_format(record))

    handler = _Capture()
    probe_logger = logging.getLogger("fraudai.cid-probe")
    probe_logger.handlers = [handler]
    probe_logger.setLevel(logging.INFO)
    probe_logger.propagate = False

    app = FastAPI()
    app.add_middleware(MetricsMiddleware)

    @app.get("/probe")
    async def probe(delay: float = 0.0) -> dict[str, str]:
        await asyncio.sleep(delay)
        probe_logger.info("inside handler")
        return {"ok": "yes"}

    @app.get("/stream")
    async def stream() -> StreamingResponse:
        async def gen() -> AsyncIterator[str]:
            probe_logger.info("inside stream")
            yield "data: x\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


async def _http(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_each_request_gets_its_own_correlation_id_on_header_and_in_logs() -> None:
    sink: list[dict[str, Any]] = []
    async with await _http(_correlated_app(sink)) as http:
        first = await http.get("/probe")
        second = await http.get("/probe")

    ids = [first.headers["x-correlation-id"], second.headers["x-correlation-id"]]
    assert ids[0] != ids[1]
    assert [entry["correlation_id"] for entry in sink] == ids


async def test_a_well_formed_inbound_correlation_id_is_kept() -> None:
    sink: list[dict[str, Any]] = []
    async with await _http(_correlated_app(sink)) as http:
        response = await http.get("/probe", headers={"X-Correlation-ID": "req-2026_11-abcDEF12"})

    assert response.headers["x-correlation-id"] == "req-2026_11-abcDEF12"
    assert sink[0]["correlation_id"] == "req-2026_11-abcDEF12"


@pytest.mark.parametrize(
    "inbound",
    [
        "short",
        "x" * 65,
        "has space in it",
        "inject\r\nX-Evil: 1",
        "semi;colon;value",
        "ñandú-12345",
    ],
)
async def test_a_malformed_inbound_correlation_id_is_replaced(inbound: str) -> None:
    sink: list[dict[str, Any]] = []
    async with await _http(_correlated_app(sink)) as http:
        try:
            response = await http.get("/probe", headers={"X-Correlation-ID": inbound})
        except (httpx.InvalidURL, UnicodeEncodeError, httpx.LocalProtocolError):
            pytest.skip("client refuses to send this header at all")

    issued = response.headers["x-correlation-id"]
    assert issued != inbound
    assert len(issued) == 8
    assert sink[0]["correlation_id"] == issued


async def test_concurrent_requests_do_not_share_correlation_ids() -> None:
    sink: list[dict[str, Any]] = []
    async with await _http(_correlated_app(sink)) as http:
        slow, fast = await asyncio.gather(
            http.get(
                "/probe", params={"delay": 0.1}, headers={"X-Correlation-ID": "slow-request-1"}
            ),
            http.get("/probe", headers={"X-Correlation-ID": "fast-request-1"}),
        )

    assert slow.headers["x-correlation-id"] == "slow-request-1"
    assert fast.headers["x-correlation-id"] == "fast-request-1"
    assert sorted(entry["correlation_id"] for entry in sink) == ["fast-request-1", "slow-request-1"]


async def test_the_correlation_id_reaches_the_streaming_generator() -> None:
    sink: list[dict[str, Any]] = []
    async with await _http(_correlated_app(sink)) as http:
        response = await http.get("/stream", headers={"X-Correlation-ID": "stream-req-0001"})

    assert response.status_code == 200
    assert [entry["correlation_id"] for entry in sink] == ["stream-req-0001"]


# ---------------------------------------------------------------------------
# Stream errors are generic; user text stays out of INFO logs
# ---------------------------------------------------------------------------


class _BrokenGraph:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def aget_state(self, config: dict[str, Any]) -> Any:
        return SimpleNamespace(tasks=[], values={})

    async def astream_events(self, payload: Any, config: dict[str, Any], version: str) -> Any:
        raise self.error
        yield {}  # pragma: no cover - makes this an async generator

    async def ainvoke(self, payload: Any, config: dict[str, Any]) -> dict[str, Any]:
        return {"messages": [AIMessage(content="fine")], "current_agent": "harvey"}


def _api(graph: Any, sessions: SessionManager | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.graph = graph
    app.state.session_manager = sessions or SessionManager()
    app.state.feedback_store = []
    app.state.embedder = None
    app.state.store = MagicMock(
        create_session_collection=AsyncMock(return_value="c"),
        upsert_chunks=AsyncMock(return_value=0),
    )
    return app


def _headers() -> dict[str, str]:
    cfg = Settings(_env_file=None, jwt_secret=SYNTHETIC_SECRET)
    with patch("fraudai.core.config.settings", cfg):
        return {"Authorization": f"Bearer {create_token('alice', 'tenant-a', 'pro')}"}


def _events(body: str) -> list[dict[str, Any]]:
    return [json.loads(chunk[6:]) for chunk in body.split("\n\n") if chunk.startswith("data: ")]


async def test_internal_stream_exceptions_are_generic_to_the_client_but_logged_in_full(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_detail = "/srv/fraudai/secrets/db-password.txt not readable"
    app = _api(_BrokenGraph(RuntimeError(secret_detail)))
    cfg = Settings(_env_file=None, jwt_secret=SYNTHETIC_SECRET)

    with (
        caplog.at_level(logging.ERROR),
        patch("fraudai.core.config.settings", cfg),
        patch("fraudai.api.routes.settings", cfg),
    ):
        async with await _http(app) as http:
            response = await http.post(
                "/api/v1/chat/stream", json={"message": "hi"}, headers=_headers()
            )

    events = _events(response.text)
    assert events[-1] == {
        "type": "error",
        "code": "agent_error",
        "message": "Agent processing failed. Please try again.",
    }
    assert secret_detail not in response.text
    assert secret_detail in caplog.text, "the operator still gets the real cause"


async def test_pending_action_required_keeps_its_explicit_error_code() -> None:
    pending_value = {
        "kind": "offensive_tool_approval",
        "action_id": str(uuid.uuid4()),
        "agent": "mike",
        "tool": "adversarial_evasion",
        "arguments": {"target_endpoint": "https://x.example"},
        "target": "https://x.example",
        "warning": "w",
    }
    graph = _BrokenGraph(RuntimeError("unused"))
    interrupt = SimpleNamespace(value=pending_value)
    graph.aget_state = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(tasks=[SimpleNamespace(interrupts=[interrupt])], values={})
    )
    app = _api(graph)
    cfg = Settings(_env_file=None, jwt_secret=SYNTHETIC_SECRET)
    sessions = app.state.session_manager
    sid = sessions.create_session("tenant-a", "pro")

    with patch("fraudai.core.config.settings", cfg), patch("fraudai.api.routes.settings", cfg):
        async with await _http(app) as http:
            response = await http.post(
                "/api/v1/chat/stream",
                json={"message": "hi", "session_id": sid},
                headers=_headers(),
            )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "pending_action_required"


async def test_sandbox_rejection_keeps_its_explicit_payload() -> None:
    tool = MagicMock()
    tool.name = "analyze_transactions"
    tool.ainvoke = AsyncMock(
        return_value={
            "status": "disabled",
            "error": "Sandbox disabled by configuration.",
            "error_code": "sandbox_disabled",
        }
    )

    outcome = await ClaudeAgentInvoker.run_tool_call([tool], "analyze_transactions", {})

    assert outcome["status"] == "error"
    assert outcome["error_code"] == "sandbox_disabled"
    assert outcome["result"] == {
        "status": "disabled",
        "error": "Sandbox disabled by configuration.",
        "error_code": "sandbox_disabled",
    }


async def test_user_text_never_appears_in_info_or_higher_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = _api(_BrokenGraph(RuntimeError("unused")))
    cfg = Settings(_env_file=None, jwt_secret=SYNTHETIC_SECRET)
    sid = app.state.session_manager.create_session("tenant-a", "pro")

    with (
        caplog.at_level(logging.INFO),
        patch("fraudai.core.config.settings", cfg),
        patch("fraudai.api.routes.settings", cfg),
    ):
        async with await _http(app) as http:
            await http.post(
                "/api/v1/chat",
                json={"message": USER_TEXT, "session_id": sid},
                headers=_headers(),
            )
            await http.post("/api/v1/chat/stream", json={"message": USER_TEXT}, headers=_headers())
            await http.post(
                "/api/v1/feedback",
                json={"session_id": sid, "message_id": "m", "rating": 3, "comment": USER_TEXT},
                headers=_headers(),
            )
            await http.post(
                "/api/v1/files/upload",
                params={"session_id": sid},
                files={"file": (f"{USER_TEXT}.csv", io.BytesIO(b"a\n"), "text/csv")},
                headers=_headers(),
            )

    assert USER_TEXT not in caplog.text


# ---------------------------------------------------------------------------
# B6: the search_boe tool logs the user's query only at DEBUG
# ---------------------------------------------------------------------------


async def _run_search_boe(query: str) -> None:
    from fraudai.agents.tools import create_search_boe

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=[])
    retriever.format_citations = MagicMock(return_value=[])
    await create_search_boe(retriever).ainvoke({"query": query})


async def test_search_boe_query_is_not_logged_at_info(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="fraudai"):
        await _run_search_boe(USER_TEXT)

    assert USER_TEXT not in caplog.text
    assert any("search_boe" in r.getMessage() for r in caplog.records), "tool still logs"


async def test_search_boe_query_is_available_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="fraudai"):
        await _run_search_boe(USER_TEXT)

    debug_records = [r for r in caplog.records if USER_TEXT in r.getMessage()]
    assert debug_records
    assert {r.levelno for r in debug_records} == {logging.DEBUG}
