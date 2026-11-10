"""Tests for FraudAI Agent API routes.

All external dependencies (LangGraph, Qdrant, Ollama, Claude) are mocked.
Uses httpx.AsyncClient with ASGITransport for async test client.
"""

from __future__ import annotations

import io
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage, HumanMessage

from fraudai.api.auth import User
from fraudai.api.session_manager import SessionManager

# Test placeholder -- not a real credential


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_USER = User(
    user_id="test-user-001",
    tenant_id="test-tenant",
    email="test@example.com",
    tier="pro",
    is_admin=False,
)

ADMIN_USER = User(
    user_id="admin-001",
    tenant_id="test-tenant",
    email="admin@example.com",
    tier="enterprise",
    is_admin=True,
)


def _mock_graph_state(**overrides: Any) -> dict[str, Any]:
    """Build a mock final state as returned by graph.ainvoke."""
    state: dict[str, Any] = {
        "messages": [
            HumanMessage(content="Test message"),
            AIMessage(content="Test response from Harvey."),
        ],
        "current_agent": "harvey",
        "previous_agent": None,
        "session_id": "test-session-123",
        "tenant_id": "test-tenant",
        "user_tier": "pro",
        "shared_context": {
            "routing_confidence": 0.92,
            "tokens_used": 1500,
            "corpus_version": "v2024.01",
        },
        "uploaded_documents": [],
        "tool_results": [],
        "escalation_request": None,
        "needs_human_confirmation": False,
        "language": "es",
        "turn_count": 1,
    }
    state.update(overrides)
    return state


def _snapshot(pending: dict[str, Any] | None = None) -> SimpleNamespace:
    """Checkpoint stand-in: paused on ``pending`` (an interrupt value) or not paused."""
    interrupts = (SimpleNamespace(value=pending),) if pending is not None else ()
    return SimpleNamespace(tasks=(SimpleNamespace(interrupts=interrupts),))


def _pending_payload(action_id: str = "action-001") -> dict[str, Any]:
    return {
        "kind": "offensive_tool_approval",
        "action_id": action_id,
        "agent": "mike",
        "tool": "adversarial_evasion",
        "arguments": {"target_endpoint": "https://models.example.com/predict"},
        "target": "https://models.example.com/predict",
        "warning": "Approve only if you are authorised to test it.",
    }


@pytest.fixture()
def mock_graph() -> MagicMock:
    """Create a mock compiled graph (not paused unless a test says so)."""
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value=_mock_graph_state())
    graph.aget_state = AsyncMock(return_value=_snapshot())
    graph.astream_events = MagicMock()
    return graph


@pytest.fixture()
def mock_store() -> MagicMock:
    """Create a mock QdrantStore."""
    store = MagicMock()
    store.initialize = AsyncMock()
    store.health_check = AsyncMock(return_value=True)
    store.get_corpus_version = AsyncMock(return_value="v2024.01")
    store.create_session_collection = AsyncMock(return_value="session_test-session-123")
    store.upsert_chunks = AsyncMock(return_value=5)
    store.delete_session_collection = AsyncMock()
    return store


@pytest.fixture()
def session_manager() -> SessionManager:
    """Create a real SessionManager for testing."""
    return SessionManager()


@pytest.fixture()
def mock_embedder() -> MagicMock:
    """Create a mock EmbeddingGenerator."""
    embedder = MagicMock()
    embedder.encode.return_value = [
        {
            "dense": [0.1] * 1024,
            "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.5, 0.3]},
        },
    ]
    return embedder


@pytest.fixture()
def app(
    mock_graph: MagicMock,
    mock_store: MagicMock,
    session_manager: SessionManager,
    mock_embedder: MagicMock,
) -> FastAPI:
    """Create a test FastAPI app with mocked dependencies."""
    from fraudai.api.routes import router

    test_app = FastAPI()
    test_app.include_router(router, prefix="/api/v1")

    # Attach mocked state
    test_app.state.graph = mock_graph
    test_app.state.store = mock_store
    test_app.state.session_manager = session_manager
    test_app.state.feedback_store = []
    test_app.state.embedder = mock_embedder

    return test_app


@pytest.fixture()
async def client(app: FastAPI) -> httpx.AsyncClient:
    """Create an async test client."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def _auth_override() -> User:
    """Return test user -- replaces get_current_user dependency."""
    return TEST_USER


def _admin_override() -> User:
    """Return admin user -- replaces get_admin_user dependency."""
    return ADMIN_USER


@pytest.fixture(autouse=True)
def _override_auth(app: FastAPI) -> None:
    """Override auth dependencies for all tests."""
    from fraudai.api.auth import get_admin_user, get_current_user

    app.dependency_overrides[get_current_user] = _auth_override
    app.dependency_overrides[get_admin_user] = _admin_override


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------


async def test_chat_basic(client: httpx.AsyncClient, mock_graph: MagicMock) -> None:
    """Basic chat returns a valid ChatResponse."""
    resp = await client.post(
        "/api/v1/chat",
        json={"message": "Analyze this transaction for fraud"},
    )
    assert resp.status_code == 200

    data = resp.json()
    assert "session_id" in data
    assert data["agent"] == "harvey"
    assert data["message"] == "Test response from Harvey."
    assert "metadata" in data
    assert data["metadata"]["routing_agent"] == "harvey"
    mock_graph.ainvoke.assert_awaited_once()


async def test_chat_creates_new_session(
    client: httpx.AsyncClient, session_manager: SessionManager
) -> None:
    """Chat without session_id creates a new session."""
    resp = await client.post(
        "/api/v1/chat",
        json={"message": "Hello"},
    )
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    assert session_manager.get_session(session_id) is not None


async def test_chat_with_existing_session(
    client: httpx.AsyncClient,
    mock_graph: MagicMock,
    session_manager: SessionManager,
) -> None:
    """Chat with an existing session_id uses that session."""
    sid = session_manager.create_session(tenant_id="test-tenant", tier="pro")

    resp = await client.post(
        "/api/v1/chat",
        json={"message": "Follow-up question", "session_id": sid},
    )
    assert resp.status_code == 200
    assert resp.json()["session_id"] == sid


async def test_chat_with_agent_override(client: httpx.AsyncClient, mock_graph: MagicMock) -> None:
    """Chat with agent_override bypasses Donna."""
    resp = await client.post(
        "/api/v1/chat",
        json={
            "message": "Check compliance for KYC",
            "agent_override": "louis",
        },
    )
    assert resp.status_code == 200

    # Verify graph was called with current_agent set
    call_args = mock_graph.ainvoke.call_args
    input_state = call_args[0][0]
    assert input_state["current_agent"] == "louis"


async def test_chat_graph_failure_returns_500(
    client: httpx.AsyncClient, mock_graph: MagicMock
) -> None:
    """Graph invocation failure returns HTTP 500."""
    mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

    resp = await client.post(
        "/api/v1/chat",
        json={"message": "This will fail"},
    )
    assert resp.status_code == 500
    assert "Agent processing failed" in resp.json()["detail"]


async def test_chat_records_turn(
    client: httpx.AsyncClient, session_manager: SessionManager
) -> None:
    """Chat records the agent turn in the session."""
    resp = await client.post("/api/v1/chat", json={"message": "Hello"})
    assert resp.status_code == 200
    sid = resp.json()["session_id"]
    session = session_manager.get_session(sid)
    assert session is not None
    assert session.turn_count == 1
    assert "harvey" in session.agent_history


# ---------------------------------------------------------------------------
# POST /chat/stream
# ---------------------------------------------------------------------------


async def test_chat_stream_returns_sse(client: httpx.AsyncClient, mock_graph: MagicMock) -> None:
    """Stream endpoint returns text/event-stream with SSE format."""

    # Mock astream_events as an async generator
    async def mock_events(*args: Any, **kwargs: Any):
        mock_chunk = MagicMock()
        mock_chunk.content = "Hello "
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": mock_chunk},
        }
        mock_chunk2 = MagicMock()
        mock_chunk2.content = "world"
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": mock_chunk2},
        }
        yield {
            "event": "on_tool_end",
            "name": "analyze_transactions",
            "data": {"output": "Found 3 anomalies"},
        }

    mock_graph.astream_events = mock_events

    resp = await client.post(
        "/api/v1/chat/stream",
        json={"message": "Stream this response"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"

    # Parse SSE events from response body
    body = resp.text
    lines = [line for line in body.strip().split("\n") if line.startswith("data: ")]
    assert len(lines) >= 3  # 2 tokens + 1 tool_end + 1 done

    import json

    events = [json.loads(line.removeprefix("data: ")) for line in lines]
    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) == 2
    assert token_events[0]["content"] == "Hello "
    assert token_events[1]["content"] == "world"

    tool_events = [e for e in events if e["type"] == "tool_result"]
    assert len(tool_events) == 1
    assert tool_events[0]["tool_result"]["tool_name"] == "analyze_transactions"

    done_events = [e for e in events if e["type"] == "done"]
    assert len(done_events) == 1


async def test_chat_stream_error_yields_error_event(
    client: httpx.AsyncClient, mock_graph: MagicMock
) -> None:
    """Stream errors produce an error SSE event."""

    async def failing_events(*args: Any, **kwargs: Any):
        raise RuntimeError("Stream failure")
        yield  # Make it an async generator  # noqa: RET503

    mock_graph.astream_events = failing_events

    resp = await client.post(
        "/api/v1/chat/stream",
        json={"message": "This will fail"},
    )
    assert resp.status_code == 200  # SSE returns 200, error is in the stream

    import json

    body = resp.text
    lines = [line for line in body.strip().split("\n") if line.startswith("data: ")]
    events = [json.loads(line.removeprefix("data: ")) for line in lines]
    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    # The internal cause stays in the server log; the client gets a generic message.
    assert "Stream failure" not in error_events[0]["message"]
    assert error_events[0]["code"] == "agent_error"


# ---------------------------------------------------------------------------
# POST /files/upload
# ---------------------------------------------------------------------------


async def test_upload_valid_file(
    client: httpx.AsyncClient, mock_store: MagicMock, session_manager: SessionManager
) -> None:
    """Upload a valid CSV file succeeds."""
    sid = session_manager.create_session(tenant_id="test-tenant", tier="pro")

    content = b"id,amount,is_fraud\n1,100.0,false\n2,50000.0,true\n"
    resp = await client.post(
        "/api/v1/files/upload",
        params={"session_id": sid},
        files={"file": ("transactions.csv", io.BytesIO(content), "text/csv")},
    )
    assert resp.status_code == 200

    data = resp.json()
    assert data["filename"] == "transactions.csv"
    assert data["size_bytes"] == len(content)
    assert data["indexed"] is True
    assert data["chunks_generated"] > 0
    mock_store.create_session_collection.assert_awaited_once()


async def test_upload_invalid_extension(
    client: httpx.AsyncClient, session_manager: SessionManager
) -> None:
    """Upload a file with unsupported extension is rejected."""
    sid = session_manager.create_session(tenant_id="test-tenant", tier="pro")

    resp = await client.post(
        "/api/v1/files/upload",
        params={"session_id": sid},
        files={"file": ("malware.exe", io.BytesIO(b"MZ..."), "application/octet-stream")},
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]


async def test_upload_file_too_large(
    client: httpx.AsyncClient, session_manager: SessionManager
) -> None:
    """Upload a file exceeding size limit is rejected."""
    sid = session_manager.create_session(tenant_id="test-tenant", tier="pro")

    # Patch MAX_FILE_SIZE to a small value for fast testing
    with patch("fraudai.api.routes.MAX_FILE_SIZE", 100):
        content = b"x" * 200  # 200 bytes > 100 byte limit
        resp = await client.post(
            "/api/v1/files/upload",
            params={"session_id": sid},
            files={"file": ("big.csv", io.BytesIO(content), "text/csv")},
        )
    assert resp.status_code == 413
    assert "File too large" in resp.json()["detail"]


async def test_upload_records_in_session(
    client: httpx.AsyncClient, session_manager: SessionManager
) -> None:
    """Upload records the file ID in the session."""
    sid = session_manager.create_session(tenant_id="test-tenant", tier="pro")

    resp = await client.post(
        "/api/v1/files/upload",
        params={"session_id": sid},
        files={"file": ("data.json", io.BytesIO(b'{"key": "value"}'), "application/json")},
    )
    assert resp.status_code == 200

    session = session_manager.get_session(sid)
    assert session is not None
    assert len(session.uploaded_files) == 1


# ---------------------------------------------------------------------------
# GET /sessions/{session_id}
# ---------------------------------------------------------------------------


async def test_get_session_valid(
    client: httpx.AsyncClient, session_manager: SessionManager
) -> None:
    """Get session returns correct session info."""
    sid = session_manager.create_session(tenant_id="test-tenant", tier="pro")
    session_manager.record_turn(sid, "harvey")
    session_manager.record_turn(sid, "louis")

    resp = await client.get(f"/api/v1/sessions/{sid}")
    assert resp.status_code == 200

    data = resp.json()
    assert data["session_id"] == sid
    assert data["turn_count"] == 2
    assert data["agent_history"] == ["harvey", "louis"]


async def test_get_session_not_found(client: httpx.AsyncClient) -> None:
    """Get session with invalid ID returns 404."""
    resp = await client.get("/api/v1/sessions/nonexistent-id")
    assert resp.status_code == 404


async def test_get_session_tenant_isolation(
    client: httpx.AsyncClient, session_manager: SessionManager
) -> None:
    """A session of another tenant is indistinguishable from an unknown one (404)."""
    foreign_id = session_manager.create_session(tenant_id="other-tenant", tier="free")

    resp = await client.get(f"/api/v1/sessions/{foreign_id}")
    assert resp.status_code == 404
    unknown = await client.get(f"/api/v1/sessions/{uuid.uuid4()}")
    assert resp.json() == unknown.json()


# ---------------------------------------------------------------------------
# DELETE /sessions/{session_id}
# ---------------------------------------------------------------------------


async def test_delete_session(
    client: httpx.AsyncClient,
    session_manager: SessionManager,
    mock_store: MagicMock,
) -> None:
    """Delete session removes it and cleans up Qdrant."""
    sid = session_manager.create_session(tenant_id="test-tenant", tier="pro")

    resp = await client.delete(f"/api/v1/sessions/{sid}")
    assert resp.status_code == 204

    assert session_manager.get_session(sid) is None
    mock_store.delete_session_collection.assert_awaited_once_with(sid)


async def test_delete_session_not_found(client: httpx.AsyncClient) -> None:
    """Delete nonexistent session returns 404."""
    resp = await client.delete("/api/v1/sessions/nonexistent-id")
    assert resp.status_code == 404


async def test_delete_session_tenant_isolation(
    client: httpx.AsyncClient, session_manager: SessionManager
) -> None:
    """Delete of another tenant's session is rejected and leaves it intact."""
    foreign_id = session_manager.create_session(tenant_id="other-tenant", tier="free")

    resp = await client.delete(f"/api/v1/sessions/{foreign_id}")
    assert resp.status_code == 404
    assert session_manager.get_session(foreign_id) is not None


async def test_chat_starts_each_turn_with_a_fresh_escalation_budget_and_no_old_approvals(
    client: httpx.AsyncClient, mock_graph: MagicMock
) -> None:
    resp = await client.post("/api/v1/chat", json={"message": "Hola"})
    assert resp.status_code == 200

    sent = mock_graph.ainvoke.call_args[0][0]
    assert sent["turn_count"] == 0
    assert sent["proposed_actions"] == []


async def test_chat_returns_409_with_the_pending_action_when_the_session_is_paused(
    client: httpx.AsyncClient, mock_graph: MagicMock, session_manager: SessionManager
) -> None:
    sid = session_manager.create_session(tenant_id="test-tenant", tier="pro")
    mock_graph.aget_state = AsyncMock(return_value=_snapshot(_pending_payload()))

    resp = await client.post("/api/v1/chat", json={"message": "Hola", "session_id": sid})

    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["code"] == "pending_action_required"
    assert detail["pending_action"]["action_id"] == "action-001"
    mock_graph.ainvoke.assert_not_awaited()


# ---------------------------------------------------------------------------
# POST /confirm
# ---------------------------------------------------------------------------


async def test_confirm_approved(
    client: httpx.AsyncClient, mock_graph: MagicMock, session_manager: SessionManager
) -> None:
    """Confirmation with approved=True resumes the graph."""
    mock_graph.aget_state = AsyncMock(side_effect=[_snapshot(_pending_payload()), _snapshot()])
    sid = session_manager.create_session(tenant_id="test-tenant", tier="pro")

    mock_graph.ainvoke = AsyncMock(
        return_value=_mock_graph_state(
            current_agent="mike",
            messages=[
                HumanMessage(content="Run adversarial test"),
                AIMessage(content="Red teaming action executed successfully."),
            ],
        )
    )

    resp = await client.post(
        "/api/v1/confirm",
        json={
            "session_id": sid,
            "action_id": "action-001",
            "approved": True,
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    assert data["agent"] == "mike"
    assert "executed successfully" in data["message"]

    # Verify graph was resumed with Command
    call_args = mock_graph.ainvoke.call_args
    command = call_args[0][0]
    assert hasattr(command, "resume")
    assert command.resume == {"action_id": "action-001", "approved": True}


async def test_confirm_rejected(
    client: httpx.AsyncClient, mock_graph: MagicMock, session_manager: SessionManager
) -> None:
    """Confirmation with approved=False cancels the action."""
    mock_graph.aget_state = AsyncMock(side_effect=[_snapshot(_pending_payload()), _snapshot()])
    sid = session_manager.create_session(tenant_id="test-tenant", tier="pro")

    mock_graph.ainvoke = AsyncMock(
        return_value=_mock_graph_state(
            current_agent="mike",
            messages=[
                HumanMessage(content="Run adversarial test"),
                AIMessage(content="Red teaming action cancelled by user."),
            ],
        )
    )

    resp = await client.post(
        "/api/v1/confirm",
        json={
            "session_id": sid,
            "action_id": "action-001",
            "approved": False,
        },
    )
    assert resp.status_code == 200

    data = resp.json()
    assert "cancelled" in data["message"]

    call_args = mock_graph.ainvoke.call_args
    command = call_args[0][0]
    assert command.resume == {"action_id": "action-001", "approved": False}


async def test_confirm_failure_returns_500(
    client: httpx.AsyncClient, mock_graph: MagicMock, session_manager: SessionManager
) -> None:
    """Confirmation resume failure returns 500."""
    mock_graph.aget_state = AsyncMock(side_effect=[_snapshot(_pending_payload()), _snapshot()])
    sid = session_manager.create_session(tenant_id="test-tenant", tier="pro")
    mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("No pending interrupt"))

    resp = await client.post(
        "/api/v1/confirm",
        json={
            "session_id": sid,
            "action_id": "action-001",
            "approved": True,
        },
    )
    assert resp.status_code == 500
    assert "Failed to process confirmation" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# POST /feedback
# ---------------------------------------------------------------------------


async def test_submit_feedback(
    client: httpx.AsyncClient, app: FastAPI, session_manager: SessionManager
) -> None:
    """Feedback submission returns 201 with feedback_id."""
    sid = session_manager.create_session(tenant_id="test-tenant", tier="pro")
    resp = await client.post(
        "/api/v1/feedback",
        json={
            "session_id": sid,
            "message_id": "msg-001",
            "rating": 5,
            "comment": "Very helpful analysis",
        },
    )
    assert resp.status_code == 201

    data = resp.json()
    assert data["status"] == "accepted"
    assert "feedback_id" in data

    # Verify feedback was stored
    assert len(app.state.feedback_store) == 1
    entry = app.state.feedback_store[0]
    assert entry["rating"] == 5
    assert entry["user_id"] == "test-user-001"


async def test_submit_feedback_minimum_rating(
    client: httpx.AsyncClient, session_manager: SessionManager
) -> None:
    """Feedback with rating=1 is accepted."""
    sid = session_manager.create_session(tenant_id="test-tenant", tier="pro")
    resp = await client.post(
        "/api/v1/feedback",
        json={
            "session_id": sid,
            "message_id": "msg-001",
            "rating": 1,
        },
    )
    assert resp.status_code == 201


async def test_submit_feedback_invalid_rating(client: httpx.AsyncClient) -> None:
    """Feedback with rating out of range is rejected."""
    resp = await client.post(
        "/api/v1/feedback",
        json={
            "session_id": "session-001",
            "message_id": "msg-001",
            "rating": 0,
        },
    )
    assert resp.status_code == 422  # Pydantic validation error


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


async def _monitor_with(
    app: FastAPI, *, qdrant: bool, ollama: bool, llm: bool, corpus: str | None = "v2024.01"
) -> None:
    """Install a refreshed health monitor whose checks return the given outcomes."""
    from fraudai.api.health import HealthMonitor

    def check(ok: bool) -> Any:
        async def run() -> bool:
            return ok

        return run

    async def corpus_probe() -> str | None:
        return corpus

    monitor = HealthMonitor(
        {"qdrant": check(qdrant), "ollama": check(ollama), "llm": check(llm)},
        frozenset({"qdrant", "llm"}),
        corpus_version_probe=corpus_probe,
    )
    await monitor.refresh()
    app.state.health_monitor = monitor


async def test_health_all_healthy(client: httpx.AsyncClient, app: FastAPI) -> None:
    """Health check returns 'healthy' when all services are up."""
    await _monitor_with(app, qdrant=True, ollama=True, llm=True)

    resp = await client.get("/api/v1/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["qdrant"] is True
    assert data["ollama"] is True
    assert data["claude_api"] is True
    assert data["corpus_version"] == "v2024.01"


async def test_health_degraded(client: httpx.AsyncClient, app: FastAPI) -> None:
    """Health check returns 'degraded' when some services are down."""
    await _monitor_with(app, qdrant=True, ollama=False, llm=False)

    resp = await client.get("/api/v1/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["qdrant"] is True


async def test_health_unhealthy(client: httpx.AsyncClient, app: FastAPI) -> None:
    """Health check returns 'unhealthy' when all services are down."""
    await _monitor_with(app, qdrant=False, ollama=False, llm=False, corpus=None)

    resp = await client.get("/api/v1/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "unhealthy"
    assert data["qdrant"] is False
    assert data["ollama"] is False
    assert data["claude_api"] is False


# ---------------------------------------------------------------------------
# Auth required (401 without token)
# ---------------------------------------------------------------------------


async def test_auth_required_without_override(app: FastAPI) -> None:
    """Endpoints return 401 when auth is not overridden and no token provided."""
    from fraudai.api.auth import get_current_user

    # Remove the auth override to test real auth behavior
    app.dependency_overrides.pop(get_current_user, None)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.post("/api/v1/chat", json={"message": "Hello"})
        # Without Bearer token, FastAPI's OAuth2 scheme returns 401
        assert resp.status_code == 401

    # Restore override for other tests
    app.dependency_overrides[get_current_user] = _auth_override


async def test_health_no_auth_required(app: FastAPI) -> None:
    """Health endpoint does not require authentication and calls no dependency."""
    from fraudai.api.auth import get_current_user

    app.dependency_overrides.pop(get_current_user, None)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/v1/health")

    assert resp.status_code == 200  # no monitor snapshot yet: unhealthy, but never an error
    app.dependency_overrides[get_current_user] = _auth_override


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


async def test_chat_with_citations_and_tools(
    client: httpx.AsyncClient, mock_graph: MagicMock
) -> None:
    """Chat with citations and tool results in shared_context."""
    mock_graph.ainvoke = AsyncMock(
        return_value=_mock_graph_state(
            shared_context={
                "routing_confidence": 0.95,
                "tokens_used": 2000,
                "corpus_version": "v2024.02",
                "citations": [
                    {
                        "boe_id": "BOE-A-2010-6737",
                        "norma_titulo": "Ley 10/2010",
                        "articulo": "Art. 18",
                        "texto_relevante": "Due diligence requirements...",
                        "score": 0.91,
                    }
                ],
            },
            tool_results=[
                {
                    "tool_name": "analyze_transactions",
                    "status": "success",
                    "result": {"anomalies": 3},
                    "duration_ms": 450,
                }
            ],
        )
    )

    resp = await client.post(
        "/api/v1/chat",
        json={"message": "Check this transaction"},
    )
    assert resp.status_code == 200

    data = resp.json()
    assert len(data["citations"]) == 1
    assert data["citations"][0]["boe_id"] == "BOE-A-2010-6737"
    assert len(data["tool_results"]) == 1
    assert data["tool_results"][0]["tool_name"] == "analyze_transactions"


async def test_chat_language_parameter(client: httpx.AsyncClient, mock_graph: MagicMock) -> None:
    """Chat passes language parameter to graph."""
    resp = await client.post(
        "/api/v1/chat",
        json={"message": "Analyze fraud", "language": "en"},
    )
    assert resp.status_code == 200

    call_args = mock_graph.ainvoke.call_args
    input_state = call_args[0][0]
    assert input_state["language"] == "en"
