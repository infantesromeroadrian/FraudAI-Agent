"""Tenant isolation matrix (F1 / R1): every session route x {own, foreign, unknown}.

Real pieces: JWT auth (``create_token`` + ``get_current_user``), the routes and
``SessionManager``. The graph, the vector store and the filesystem root are
spies, so each rejection can also prove that nothing behind the gate was
touched: no graph run, no collection, no directory, no state change.
"""

from __future__ import annotations

import io
import re
import uuid
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage

from fraudai.agents import graph as graph_module
from fraudai.api import routes as routes_module
from fraudai.api.auth import create_token
from fraudai.api.routes import router
from fraudai.api.session_manager import SessionManager
from fraudai.core import session_context
from fraudai.core.session_context import active_session_id
from fraudai.tools.implementations import ToolImplementations

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

OWNER_TENANT = "tenant-owner"
OTHER_TENANT = "tenant-other"
OWN, FOREIGN, UNKNOWN = "own", "foreign", "unknown"
MALFORMED_IDS = ["../../etc", "/etc/passwd", "not-a-uuid", "..", "a" * 300]
UUID_CSV_NAME = re.compile(r"^[0-9a-f-]{36}\.csv$")


class _SpyGraph:
    """Records every thread the routes try to run or read state for."""

    def __init__(self) -> None:
        self.threads: list[str] = []
        self.state_reads: list[str] = []
        self._paused: set[str] = set()

    def pause(self, thread_id: str) -> None:
        """Make ``aget_state`` report an approval pending on this thread."""
        self._paused.add(thread_id)

    async def aget_state(self, config: dict[str, Any]) -> Any:
        thread_id = config["configurable"]["thread_id"]
        self.state_reads.append(thread_id)
        pending = {
            "kind": "offensive_tool_approval",
            "action_id": "any",
            "agent": "mike",
            "tool": "adversarial_evasion",
            "arguments": {},
            "target": None,
            "warning": "w",
        }
        interrupts = (SimpleNamespace(value=pending),) if thread_id in self._paused else ()
        return SimpleNamespace(tasks=(SimpleNamespace(interrupts=interrupts),))

    async def ainvoke(self, payload: Any, config: dict[str, Any]) -> dict[str, Any]:
        self.threads.append(config["configurable"]["thread_id"])
        return {"messages": [AIMessage(content="ok")], "current_agent": "harvey"}

    async def astream_events(
        self, payload: Any, config: dict[str, Any], version: str
    ) -> AsyncIterator[dict[str, Any]]:
        self.threads.append(config["configurable"]["thread_id"])
        for _ in ():
            yield {}


@pytest.fixture()
def uploads_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "fraudai_uploads"
    monkeypatch.setattr(session_context, "uploads_root", lambda: root)
    return root


@pytest.fixture()
def spy_graph() -> _SpyGraph:
    return _SpyGraph()


@pytest.fixture()
def store() -> MagicMock:
    mock = MagicMock()
    mock.create_session_collection = AsyncMock(return_value="collection")
    mock.upsert_chunks = AsyncMock(return_value=0)
    mock.delete_session_collection = AsyncMock()
    return mock


@pytest.fixture()
def sessions() -> SessionManager:
    return SessionManager()


@pytest.fixture()
def app(
    spy_graph: _SpyGraph, store: MagicMock, sessions: SessionManager, uploads_root: Path
) -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api/v1")
    test_app.state.graph = spy_graph
    test_app.state.store = store
    test_app.state.session_manager = sessions
    test_app.state.feedback_store = []
    test_app.state.embedder = None
    return test_app


@pytest.fixture()
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


def _bearer(tenant_id: str) -> dict[str, str]:
    token = create_token(user_id=f"user-of-{tenant_id}", tenant_id=tenant_id, tier="pro")
    return {"Authorization": f"Bearer {token}"}


Call = Callable[[httpx.AsyncClient, str, dict[str, str]], Awaitable[httpx.Response]]


async def _chat(client: httpx.AsyncClient, sid: str, headers: dict[str, str]) -> httpx.Response:
    return await client.post(
        "/api/v1/chat", json={"message": "hi", "session_id": sid}, headers=headers
    )


async def _stream(client: httpx.AsyncClient, sid: str, headers: dict[str, str]) -> httpx.Response:
    return await client.post(
        "/api/v1/chat/stream", json={"message": "hi", "session_id": sid}, headers=headers
    )


async def _confirm(client: httpx.AsyncClient, sid: str, headers: dict[str, str]) -> httpx.Response:
    return await client.post(
        "/api/v1/confirm",
        json={"session_id": sid, "action_id": "any", "approved": True},
        headers=headers,
    )


async def _upload(client: httpx.AsyncClient, sid: str, headers: dict[str, str]) -> httpx.Response:
    return await client.post(
        "/api/v1/files/upload",
        params={"session_id": sid},
        files={"file": ("tx.csv", io.BytesIO(b"id\n1\n"), "text/csv")},
        headers=headers,
    )


async def _get(client: httpx.AsyncClient, sid: str, headers: dict[str, str]) -> httpx.Response:
    return await client.get(f"/api/v1/sessions/{sid}", headers=headers)


async def _delete(client: httpx.AsyncClient, sid: str, headers: dict[str, str]) -> httpx.Response:
    return await client.delete(f"/api/v1/sessions/{sid}", headers=headers)


async def _feedback(client: httpx.AsyncClient, sid: str, headers: dict[str, str]) -> httpx.Response:
    return await client.post(
        "/api/v1/feedback",
        json={"session_id": sid, "message_id": "m1", "rating": 4},
        headers=headers,
    )


ROUTES: list[tuple[str, Call, int]] = [
    ("chat", _chat, 200),
    ("chat_stream", _stream, 200),
    ("confirm", _confirm, 200),
    ("upload", _upload, 200),
    ("get_session", _get, 200),
    ("delete_session", _delete, 204),
    ("feedback", _feedback, 201),
]
ROUTE_IDS = [name for name, _, _ in ROUTES]


def _tree(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*")} if root.exists() else set()


@pytest.mark.parametrize(("name", "call", "ok_status"), ROUTES, ids=ROUTE_IDS)
async def test_own_session_is_served(
    name: str,
    call: Call,
    ok_status: int,
    client: httpx.AsyncClient,
    sessions: SessionManager,
    spy_graph: _SpyGraph,
) -> None:
    own_id = sessions.create_session(OWNER_TENANT, "pro")
    if name == "confirm":
        spy_graph.pause(own_id)  # /confirm needs an action awaiting approval

    response = await call(client, own_id, _bearer(OWNER_TENANT))

    assert response.status_code == ok_status


@pytest.mark.parametrize(("name", "call", "ok_status"), ROUTES, ids=ROUTE_IDS)
@pytest.mark.parametrize("target", [FOREIGN, UNKNOWN])
async def test_foreign_and_unknown_sessions_are_rejected_without_side_effects(
    name: str,
    call: Call,
    ok_status: int,
    target: str,
    client: httpx.AsyncClient,
    sessions: SessionManager,
    spy_graph: _SpyGraph,
    store: MagicMock,
    uploads_root: Path,
) -> None:
    victim_id = sessions.create_session(OWNER_TENANT, "pro")
    sessions.record_turn(victim_id, "harvey")
    sessions.record_upload(victim_id, "existing-file")
    session_id = victim_id if target == FOREIGN else str(uuid.uuid4())
    before = sessions.get_session(victim_id)
    assert before is not None
    before_snapshot = before.model_dump()

    response = await call(client, session_id, _bearer(OTHER_TENANT))

    assert response.status_code == 404
    assert spy_graph.threads == [], "graph ran for a session the caller does not own"
    assert spy_graph.state_reads == [], "graph state read for a session the caller does not own"
    store.create_session_collection.assert_not_called()
    store.upsert_chunks.assert_not_called()
    store.delete_session_collection.assert_not_called()
    assert _tree(uploads_root) == set(), "filesystem touched before authorization"
    after = sessions.get_session(victim_id)
    assert after is not None
    assert after.model_dump() == before_snapshot, "victim session was modified"
    assert len(sessions.list_sessions()) == 1, "a session was created as a side effect"


@pytest.mark.parametrize(("name", "call", "ok_status"), ROUTES, ids=ROUTE_IDS)
async def test_foreign_and_unknown_are_indistinguishable(
    name: str,
    call: Call,
    ok_status: int,
    client: httpx.AsyncClient,
    sessions: SessionManager,
) -> None:
    victim_id = sessions.create_session(OWNER_TENANT, "pro")

    foreign = await call(client, victim_id, _bearer(OTHER_TENANT))
    unknown = await call(client, str(uuid.uuid4()), _bearer(OTHER_TENANT))

    assert foreign.status_code == unknown.status_code == 404
    assert foreign.json() == unknown.json()


@pytest.mark.parametrize(("name", "call", "ok_status"), ROUTES, ids=ROUTE_IDS)
@pytest.mark.parametrize("session_id", MALFORMED_IDS)
async def test_client_chosen_or_malformed_ids_are_rejected(
    name: str,
    call: Call,
    ok_status: int,
    session_id: str,
    client: httpx.AsyncClient,
    sessions: SessionManager,
    spy_graph: _SpyGraph,
    uploads_root: Path,
) -> None:
    response = await call(client, session_id, _bearer(OWNER_TENANT))

    assert response.status_code in (404, 405)  # "../.." in a path may not even route
    assert spy_graph.threads == []
    assert sessions.list_sessions() == []
    assert _tree(uploads_root) == set()


@pytest.mark.parametrize("route", ["chat", "chat_stream"])
async def test_client_supplied_id_is_never_adopted_as_a_new_session(
    route: str, client: httpx.AsyncClient, sessions: SessionManager
) -> None:
    chosen = str(uuid.uuid4())
    call = _chat if route == "chat" else _stream

    response = await call(client, chosen, _bearer(OWNER_TENANT))

    assert response.status_code == 404
    assert sessions.get_session(chosen) is None


@pytest.mark.parametrize("route", ["chat", "chat_stream"])
async def test_new_conversation_gets_a_server_issued_uuid_for_the_callers_tenant(
    route: str,
    client: httpx.AsyncClient,
    sessions: SessionManager,
    spy_graph: _SpyGraph,
) -> None:
    if route == "chat":
        response = await client.post(
            "/api/v1/chat", json={"message": "hi"}, headers=_bearer(OWNER_TENANT)
        )
        issued = response.json()["session_id"]
    else:
        response = await client.post(
            "/api/v1/chat/stream", json={"message": "hi"}, headers=_bearer(OWNER_TENANT)
        )
        issued = spy_graph.threads[0]

    assert response.status_code == 200
    assert str(uuid.UUID(issued)) == issued
    session = sessions.get_session(issued)
    assert session is not None
    assert session.tenant_id == OWNER_TENANT
    assert spy_graph.threads == [issued]


async def test_unauthenticated_requests_never_reach_session_lookup(
    client: httpx.AsyncClient, sessions: SessionManager, spy_graph: _SpyGraph
) -> None:
    own_id = sessions.create_session(OWNER_TENANT, "pro")

    for _, call, _ in ROUTES:
        response = await call(client, own_id, {})
        assert response.status_code == 401
    assert spy_graph.threads == []


# ---------------------------------------------------------------------------
# Upload (M1)
# ---------------------------------------------------------------------------


async def test_upload_stores_a_server_named_file_under_the_session_directory(
    client: httpx.AsyncClient, sessions: SessionManager, uploads_root: Path
) -> None:
    sid = sessions.create_session(OWNER_TENANT, "pro")

    response = await client.post(
        "/api/v1/files/upload",
        params={"session_id": sid},
        files={"file": ("../../evil.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")},
        headers=_bearer(OWNER_TENANT),
    )

    assert response.status_code == 200
    written = [p for p in uploads_root.rglob("*") if p.is_file()]
    assert len(written) == 1
    assert written[0].parent == uploads_root / sid
    assert UUID_CSV_NAME.match(written[0].name), "client file name reached the filesystem"
    assert written[0].read_bytes() == b"a,b\n1,2\n"


async def test_upload_larger_than_the_limit_is_rejected_and_not_stored(
    client: httpx.AsyncClient,
    sessions: SessionManager,
    store: MagicMock,
    uploads_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes_module, "MAX_FILE_SIZE", 1_000)
    monkeypatch.setattr(routes_module, "UPLOAD_READ_CHUNK_SIZE", 64)
    sid = sessions.create_session(OWNER_TENANT, "pro")

    response = await client.post(
        "/api/v1/files/upload",
        params={"session_id": sid},
        files={"file": ("big.csv", io.BytesIO(b"x" * 5_000), "text/csv")},
        headers=_bearer(OWNER_TENANT),
    )

    assert response.status_code == 413
    assert _tree(uploads_root) == set()
    store.create_session_collection.assert_not_called()
    session = sessions.get_session(sid)
    assert session is not None
    assert session.uploaded_files == []


async def test_upload_of_exactly_the_limit_is_accepted(
    client: httpx.AsyncClient,
    sessions: SessionManager,
    uploads_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes_module, "MAX_FILE_SIZE", 1_000)
    monkeypatch.setattr(routes_module, "UPLOAD_READ_CHUNK_SIZE", 64)
    sid = sessions.create_session(OWNER_TENANT, "pro")

    response = await client.post(
        "/api/v1/files/upload",
        params={"session_id": sid},
        files={"file": ("edge.csv", io.BytesIO(b"x" * 1_000), "text/csv")},
        headers=_bearer(OWNER_TENANT),
    )

    assert response.status_code == 200
    assert response.json()["size_bytes"] == 1_000


async def test_upload_count_per_session_is_bounded(
    client: httpx.AsyncClient,
    sessions: SessionManager,
    uploads_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes_module, "MAX_UPLOADS_PER_SESSION", 2)
    sid = sessions.create_session(OWNER_TENANT, "pro")

    statuses = [(await _upload(client, sid, _bearer(OWNER_TENANT))).status_code for _ in range(3)]

    assert statuses == [200, 200, 429]
    assert len([p for p in uploads_root.rglob("*") if p.is_file()]) == 2


async def test_foreign_upload_does_not_reach_victim_directory_even_with_a_valid_file(
    client: httpx.AsyncClient, sessions: SessionManager, uploads_root: Path
) -> None:
    victim_id = sessions.create_session(OWNER_TENANT, "pro")

    response = await _upload(client, victim_id, _bearer(OTHER_TENANT))

    assert response.status_code == 404
    assert not (uploads_root / victim_id).exists()


BOUNDARY = "f1boundary"
PART_HEAD = (
    f"--{BOUNDARY}\r\n"
    'Content-Disposition: form-data; name="file"; filename="big.csv"\r\n'
    "Content-Type: text/csv\r\n\r\n"
).encode()
PART_TAIL = f"\r\n--{BOUNDARY}--\r\n".encode()
MULTIPART_HEADERS = {"content-type": f"multipart/form-data; boundary={BOUNDARY}"}


class _ChunkedBody:
    """Async body with no Content-Length that records how much was pulled."""

    def __init__(self, chunk_count: int, chunk_size: int) -> None:
        self.chunk_count = chunk_count
        self.chunk_size = chunk_size
        self.pulled_bytes = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for part in (PART_HEAD, *([b"x" * self.chunk_size] * self.chunk_count), PART_TAIL):
            self.pulled_bytes += len(part)
            yield part


async def test_chunked_body_without_content_length_is_cut_at_the_limit(
    client: httpx.AsyncClient,
    sessions: SessionManager,
    store: MagicMock,
    uploads_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes_module, "MAX_FILE_SIZE", 10_000)
    monkeypatch.setattr(routes_module, "MULTIPART_OVERHEAD_ALLOWANCE", 1_000)
    sid = sessions.create_session(OWNER_TENANT, "pro")
    body = _ChunkedBody(chunk_count=500, chunk_size=1_000)  # 500 kB offered, 10 kB allowed

    response = await client.post(
        "/api/v1/files/upload",
        params={"session_id": sid},
        content=body,
        headers={**MULTIPART_HEADERS, **_bearer(OWNER_TENANT)},
    )

    assert response.status_code == 413
    assert body.pulled_bytes <= 11_000 + body.chunk_size, "body was read past the limit"
    assert _tree(uploads_root) == set(), "bytes were written for an oversized upload"
    store.create_session_collection.assert_not_called()
    session = sessions.get_session(sid)
    assert session is not None
    assert session.uploaded_files == []


async def test_chunked_body_within_the_limit_is_accepted(
    client: httpx.AsyncClient,
    sessions: SessionManager,
    uploads_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes_module, "MAX_FILE_SIZE", 10_000)
    monkeypatch.setattr(routes_module, "MULTIPART_OVERHEAD_ALLOWANCE", 1_000)
    sid = sessions.create_session(OWNER_TENANT, "pro")

    response = await client.post(
        "/api/v1/files/upload",
        params={"session_id": sid},
        content=_ChunkedBody(chunk_count=10, chunk_size=1_000),
        headers={**MULTIPART_HEADERS, **_bearer(OWNER_TENANT)},
    )

    assert response.status_code == 200
    assert response.json()["size_bytes"] == 10_000


async def test_declared_content_length_over_the_budget_is_rejected_before_reading(
    client: httpx.AsyncClient,
    sessions: SessionManager,
    uploads_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes_module, "MAX_FILE_SIZE", 100)
    monkeypatch.setattr(routes_module, "MULTIPART_OVERHEAD_ALLOWANCE", 100)
    sid = sessions.create_session(OWNER_TENANT, "pro")

    response = await client.post(
        "/api/v1/files/upload",
        params={"session_id": sid},
        content=PART_HEAD + b"x" * 5_000 + PART_TAIL,
        headers={**MULTIPART_HEADERS, **_bearer(OWNER_TENANT)},
    )

    assert response.status_code == 413
    assert _tree(uploads_root) == set()


@pytest.mark.parametrize("target", [FOREIGN, UNKNOWN])
async def test_unauthorized_upload_never_reads_the_body(
    target: str,
    client: httpx.AsyncClient,
    sessions: SessionManager,
    store: MagicMock,
    uploads_root: Path,
) -> None:
    victim_id = sessions.create_session(OWNER_TENANT, "pro")
    sid = victim_id if target == FOREIGN else str(uuid.uuid4())
    body = _ChunkedBody(chunk_count=50, chunk_size=1_000)

    response = await client.post(
        "/api/v1/files/upload",
        params={"session_id": sid},
        content=body,
        headers={**MULTIPART_HEADERS, **_bearer(OTHER_TENANT)},
    )

    assert response.status_code == 404
    assert body.pulled_bytes == 0, "payload was consumed before authorization"
    store.create_session_collection.assert_not_called()
    assert _tree(uploads_root) == set()


@pytest.mark.parametrize(
    ("headers", "content", "expected"),
    [
        pytest.param({"content-type": "application/json"}, b"{}", 415, id="not_multipart"),
        pytest.param(MULTIPART_HEADERS, b"garbage", 400, id="malformed_multipart"),
        pytest.param(
            MULTIPART_HEADERS,
            f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="other"\r\n\r\nv'
            f"\r\n--{BOUNDARY}--\r\n".encode(),
            422,
            id="missing_file_part",
        ),
    ],
)
async def test_invalid_upload_bodies_are_rejected_without_storing(
    headers: dict[str, str],
    content: bytes,
    expected: int,
    client: httpx.AsyncClient,
    sessions: SessionManager,
    uploads_root: Path,
) -> None:
    sid = sessions.create_session(OWNER_TENANT, "pro")

    response = await client.post(
        "/api/v1/files/upload",
        params={"session_id": sid},
        content=content,
        headers={**headers, **_bearer(OWNER_TENANT)},
    )

    assert response.status_code == expected
    assert _tree(uploads_root) == set()


# ---------------------------------------------------------------------------
# End to end with the real graph: the session reaches the tools (M6)
# ---------------------------------------------------------------------------


@pytest.fixture()
def tool_probe(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Real graph; the specialist stub reads files as a tool would."""
    probe: dict[str, Any] = {"active": [], "own": [], "foreign": []}

    async def fake_classify(message: str) -> dict[str, Any]:
        return {"agent": "harvey", "language": "es", "confidence": 0.99}

    async def fake_invoke(
        agent_name: str, system_prompt: str, tools: list[Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        probe["active"].append(active_session_id())
        impls = ToolImplementations(sandbox=MagicMock())
        for key, path in (("own", probe.get("own_path")), ("foreign", probe.get("foreign_path"))):
            if path is None:
                continue
            try:
                probe[key].append(impls._read_input_file(path))
            except PermissionError as exc:
                probe[key].append(exc)
        return {
            "message": AIMessage(content="stub reply"),
            "tool_results": [],
            "analysis_summary": None,
            "escalation": None,
        }

    monkeypatch.setattr(graph_module, "classify_intent_local", fake_classify)
    monkeypatch.setattr(graph_module, "invoke_claude_agent", fake_invoke)
    return probe


@pytest.mark.parametrize("route", ["chat", "chat_stream"])
async def test_tools_only_read_the_active_sessions_uploads_through_the_real_graph(
    route: str,
    tool_probe: dict[str, Any],
    sessions: SessionManager,
    uploads_root: Path,
    store: MagicMock,
) -> None:
    own_id = sessions.create_session(OWNER_TENANT, "pro")
    victim_id = sessions.create_session(OTHER_TENANT, "pro")
    for sid, content in ((own_id, b"mine"), (victim_id, b"victim-secret")):
        (uploads_root / sid).mkdir(parents=True)
        (uploads_root / sid / "tx.csv").write_bytes(content)
    tool_probe["own_path"] = str(uploads_root / own_id / "tx.csv")
    tool_probe["foreign_path"] = str(uploads_root / victim_id / "tx.csv")

    real_app = FastAPI()
    real_app.include_router(router, prefix="/api/v1")
    real_app.state.graph = graph_module.build_fraud_ai_graph()
    real_app.state.session_manager = sessions
    real_app.state.store = store
    real_app.state.feedback_store = []
    path = "/api/v1/chat" if route == "chat" else "/api/v1/chat/stream"
    transport = httpx.ASGITransport(app=real_app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        response = await http.post(
            path,
            json={"message": "analyse my file", "session_id": own_id},
            headers=_bearer(OWNER_TENANT),
        )

    assert response.status_code == 200
    assert tool_probe["active"] == [own_id]
    assert tool_probe["own"] == [b"mine"]
    assert len(tool_probe["foreign"]) == 1
    assert isinstance(tool_probe["foreign"][0], PermissionError)
    assert active_session_id() is None, "session binding leaked out of the request"
