"""F5c.5: per-tenant rate limit, bounded session quotas and a bounded prompt window.

All limits are per process (no shared store). Tenant isolation, ``pending_action`` and the
checkpoint snapshot must survive every limit.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from fraudai.agents import claude_invoker as invoker_module
from fraudai.agents import graph as graph_module
from fraudai.agents.claude_invoker import ClaudeAgentInvoker
from fraudai.api.auth import create_token
from fraudai.api.limits import TenantRateLimiter
from fraudai.api.routes import router
from fraudai.api.session_manager import SessionManager
from fraudai.core import session_context
from fraudai.core.config import Settings
from fraudai.core.database import Database

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

SYNTHETIC_SECRET = "l" * 48  # noqa: S105 - fixture signing key


class _Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


# ---------------------------------------------------------------------------
# TenantRateLimiter
# ---------------------------------------------------------------------------


class TestTenantRateLimiter:
    def test_allows_up_to_the_limit_then_rejects_with_a_retry_hint(self) -> None:
        clock = _Clock()
        limiter = TenantRateLimiter(limit=3, window_seconds=60, clock=clock)

        results = [limiter.check("tenant-a") for _ in range(4)]

        assert results[:3] == [None, None, None]
        assert results[3] == 60

    def test_window_slides(self) -> None:
        clock = _Clock()
        limiter = TenantRateLimiter(limit=2, window_seconds=60, clock=clock)
        limiter.check("t")
        clock.value += 30
        limiter.check("t")
        assert limiter.check("t") == 30  # the oldest hit leaves in 30 s

        clock.value += 31

        assert limiter.check("t") is None

    def test_rejected_calls_do_not_extend_the_block(self) -> None:
        clock = _Clock()
        limiter = TenantRateLimiter(limit=1, window_seconds=60, clock=clock)
        limiter.check("t")
        for _ in range(100):
            limiter.check("t")
        clock.value += 61

        assert limiter.check("t") is None

    def test_tenants_are_independent(self) -> None:
        limiter = TenantRateLimiter(limit=1, window_seconds=60, clock=_Clock())

        assert limiter.check("a") is None
        assert limiter.check("a") is not None
        assert limiter.check("b") is None

    def test_memory_is_bounded_in_tenants_and_hits(self) -> None:
        limiter = TenantRateLimiter(limit=5, window_seconds=60, clock=_Clock(), max_tenants=50)

        for index in range(500):
            limiter.check(f"tenant-{index}")

        assert limiter.tracked_tenants() <= 50
        for _ in range(100):
            limiter.check("hot")
        assert limiter.hits("hot") <= 5

    def test_idle_tenants_are_forgotten(self) -> None:
        clock = _Clock()
        limiter = TenantRateLimiter(limit=5, window_seconds=60, clock=clock)
        limiter.check("old")
        clock.value += 120

        limiter.check("new")

        assert limiter.tracked_tenants() == 1

    @pytest.mark.parametrize("limit", [0, -1])
    def test_a_non_positive_limit_is_a_configuration_error(self, limit: int) -> None:
        with pytest.raises(ValueError, match="limit"):
            TenantRateLimiter(limit=limit)


# ---------------------------------------------------------------------------
# HTTP behaviour
# ---------------------------------------------------------------------------


class _SpyGraph:
    def __init__(self) -> None:
        self.runs = 0

    async def ainvoke(self, payload: Any, config: dict[str, Any]) -> dict[str, Any]:
        self.runs += 1
        return {"messages": [AIMessage(content="ok")], "current_agent": "harvey"}

    async def astream_events(self, payload: Any, config: dict[str, Any], version: str) -> Any:
        self.runs += 1
        for _ in ():
            yield {}

    async def aget_state(self, config: dict[str, Any]) -> Any:
        return SimpleNamespace(tasks=[], values={})


@pytest.fixture()
def spy_graph() -> _SpyGraph:
    return _SpyGraph()


@pytest.fixture()
def sessions() -> SessionManager:
    return SessionManager()


def _app(
    spy_graph: _SpyGraph, sessions: SessionManager, limiter: TenantRateLimiter | None
) -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.graph = spy_graph
    app.state.session_manager = sessions
    app.state.store = MagicMock(
        create_session_collection=AsyncMock(return_value="c"),
        upsert_chunks=AsyncMock(return_value=0),
    )
    app.state.feedback_store = []
    app.state.embedder = None
    if limiter is not None:
        app.state.rate_limiter = limiter
    return app


def _headers(tenant: str, user: str = "alice") -> dict[str, str]:
    with patch(
        "fraudai.core.config.settings", Settings(_env_file=None, jwt_secret=SYNTHETIC_SECRET)
    ):
        return {"Authorization": f"Bearer {create_token(user, tenant, 'pro')}"}


@pytest.fixture()
async def client(
    spy_graph: _SpyGraph, sessions: SessionManager
) -> AsyncIterator[httpx.AsyncClient]:
    cfg = Settings(_env_file=None, jwt_secret=SYNTHETIC_SECRET)
    app = _app(spy_graph, sessions, TenantRateLimiter(limit=3, window_seconds=60, clock=_Clock()))
    with patch("fraudai.core.config.settings", cfg), patch("fraudai.api.routes.settings", cfg):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as http:
            yield http


async def _chat(client: httpx.AsyncClient, tenant: str, user: str = "alice") -> httpx.Response:
    return await client.post("/api/v1/chat", json={"message": "hi"}, headers=_headers(tenant, user))


async def test_chat_over_the_limit_is_429_with_retry_after_and_never_reaches_the_graph(
    client: httpx.AsyncClient, spy_graph: _SpyGraph
) -> None:
    statuses = [(await _chat(client, "tenant-a")).status_code for _ in range(3)]
    blocked = await _chat(client, "tenant-a")

    assert statuses == [200, 200, 200]
    assert blocked.status_code == 429
    assert blocked.json() == {"detail": "Rate limit exceeded."}
    assert int(blocked.headers["retry-after"]) >= 1
    assert spy_graph.runs == 3


async def test_the_limit_is_per_tenant_and_shared_by_its_users(client: httpx.AsyncClient) -> None:
    await _chat(client, "tenant-a", "alice")
    await _chat(client, "tenant-a", "bob")
    await _chat(client, "tenant-a", "carol")

    assert (await _chat(client, "tenant-a", "dave")).status_code == 429
    assert (await _chat(client, "tenant-b", "alice")).status_code == 200


async def test_stream_confirm_and_upload_spend_the_same_budget(
    client: httpx.AsyncClient, sessions: SessionManager
) -> None:
    sid = sessions.create_session("tenant-a", "pro")
    headers = _headers("tenant-a")

    stream = await client.post("/api/v1/chat/stream", json={"message": "hi"}, headers=headers)
    upload = await client.post(
        "/api/v1/files/upload",
        params={"session_id": sid},
        files={"file": ("a.csv", io.BytesIO(b"a\n1\n"), "text/csv")},
        headers=headers,
    )
    confirm = await client.post(
        "/api/v1/confirm",
        json={"session_id": sid, "action_id": "x", "approved": True},
        headers=headers,
    )
    blocked = await _chat(client, "tenant-a")

    assert stream.status_code == 200
    assert upload.status_code == 200
    assert confirm.status_code == 409  # counted, then refused: nothing pending
    assert blocked.status_code == 429


async def test_reads_and_probes_are_not_rate_limited(
    client: httpx.AsyncClient, sessions: SessionManager
) -> None:
    sid = sessions.create_session("tenant-a", "pro")
    for _ in range(3):
        await _chat(client, "tenant-a")

    reads = [
        await client.get(f"/api/v1/sessions/{sid}", headers=_headers("tenant-a")),
        await client.get("/api/v1/livez"),
        await client.get("/api/v1/readyz"),
        await client.get("/api/v1/health"),
    ]

    assert [r.status_code for r in reads] == [200, 200, 503, 200]


async def test_unauthenticated_requests_do_not_consume_a_tenants_budget(
    client: httpx.AsyncClient,
) -> None:
    for _ in range(10):
        assert (await client.post("/api/v1/chat", json={"message": "hi"})).status_code == 401

    assert (await _chat(client, "tenant-a")).status_code == 200


async def test_limited_requests_do_not_leak_whether_a_session_exists(
    client: httpx.AsyncClient,
) -> None:
    for _ in range(3):
        await _chat(client, "tenant-a")

    foreign = await client.post(
        "/api/v1/chat",
        json={"message": "hi", "session_id": "00000000-0000-4000-8000-000000000000"},
        headers=_headers("tenant-a"),
    )

    assert foreign.status_code == 429


async def test_an_app_without_a_limiter_is_not_limited(
    spy_graph: _SpyGraph, sessions: SessionManager
) -> None:
    cfg = Settings(_env_file=None, jwt_secret=SYNTHETIC_SECRET)
    app = _app(spy_graph, sessions, limiter=None)
    with patch("fraudai.core.config.settings", cfg), patch("fraudai.api.routes.settings", cfg):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as http:
            statuses = [(await _chat(http, "tenant-a")).status_code for _ in range(10)]

    assert statuses == [200] * 10


def test_the_limit_is_configurable() -> None:
    assert Settings(_env_file=None).rate_limit_requests_per_minute == 60
    assert (
        Settings(_env_file=None, rate_limit_requests_per_minute=5).rate_limit_requests_per_minute
        == 5
    )
    with pytest.raises(ValueError, match="rate_limit_requests_per_minute"):
        Settings(_env_file=None, rate_limit_requests_per_minute=0)


# ---------------------------------------------------------------------------
# Session quotas
# ---------------------------------------------------------------------------


def test_a_tenant_over_its_quota_evicts_only_its_own_oldest_session() -> None:
    manager = SessionManager(max_sessions=100, max_sessions_per_tenant=3)
    other = manager.create_session("tenant-b", "pro")
    mine = [manager.create_session("tenant-a", "pro") for _ in range(4)]

    assert manager.get_session(mine[0]) is None
    assert all(manager.get_session(sid) is not None for sid in mine[1:])
    assert manager.get_session(other) is not None
    assert len(manager.list_sessions("tenant-a")) == 3


def test_global_eviction_takes_from_the_tenant_with_most_sessions() -> None:
    manager = SessionManager(max_sessions=5, max_sessions_per_tenant=10)
    quiet = manager.create_session("quiet", "pro")
    noisy = [manager.create_session("noisy", "pro") for _ in range(4)]

    manager.create_session("noisy", "pro")

    assert manager.get_session(quiet) is not None, "a flood must not evict other tenants"
    assert manager.get_session(noisy[0]) is None
    assert len(manager.list_sessions()) == 5


def test_quota_defaults_come_from_settings() -> None:
    assert Settings(_env_file=None).max_sessions_per_tenant == 200


async def test_an_evicted_session_is_removed_from_the_database_too(tmp_path: Path) -> None:
    db = Database(tmp_path / "evict.db")
    await db.initialize()
    manager = SessionManager(db=db, max_sessions=100, max_sessions_per_tenant=2)

    first = manager.create_session("tenant-a", "pro")
    manager.create_session("tenant-a", "pro")
    manager.create_session("tenant-a", "pro")
    await manager.aclose()

    assert await db.get_session(first) is None
    await db.close()


async def test_an_evicted_session_is_not_reachable_through_the_api(
    spy_graph: _SpyGraph,
) -> None:
    manager = SessionManager(max_sessions=100, max_sessions_per_tenant=1)
    cfg = Settings(_env_file=None, jwt_secret=SYNTHETIC_SECRET)
    app = _app(spy_graph, manager, limiter=None)
    old = manager.create_session("tenant-a", "pro")
    manager.create_session("tenant-a", "pro")

    with patch("fraudai.core.config.settings", cfg), patch("fraudai.api.routes.settings", cfg):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as http:
            response = await http.get(f"/api/v1/sessions/{old}", headers=_headers("tenant-a"))

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# History bounds
# ---------------------------------------------------------------------------


def test_agent_history_is_bounded_but_turn_count_keeps_the_total() -> None:
    manager = SessionManager(max_agent_history=5)
    sid = manager.create_session("tenant-a", "pro")

    for index in range(12):
        manager.record_turn(sid, f"agent-{index}")

    session = manager.get_session(sid)
    assert session is not None
    assert session.agent_history == [f"agent-{i}" for i in range(7, 12)]
    assert session.turn_count == 12


async def test_persisted_history_is_bounded_too(tmp_path: Path) -> None:
    db = Database(tmp_path / "hist.db")
    await db.initialize()
    manager = SessionManager(db=db, max_agent_history=3)
    sid = manager.create_session("tenant-a", "pro")

    for index in range(8):
        manager.record_turn(sid, f"a{index}")
        await manager.aclose()  # keep DB writes ordered, as sequential requests would be
    stored = await db.get_session(sid)
    await db.close()

    assert stored is not None
    assert stored["agent_history"] == ["a5", "a6", "a7"]
    assert stored["turn_count"] == 8


def _invoker_with(model: MagicMock) -> ClaudeAgentInvoker:
    with patch("fraudai.agents.claude_invoker._build_llm", return_value=model):
        return ClaudeAgentInvoker(api_key="synthetic", provider="groq", model="m")


def _conversation(turns: int) -> list[Any]:
    messages: list[Any] = []
    for index in range(turns):
        messages.append(HumanMessage(content=f"question-{index}"))
        messages.append(AIMessage(content=f"answer-{index}"))
    return messages


def test_prompt_window_keeps_the_latest_messages_and_starts_on_a_user_turn() -> None:
    history = [*_conversation(20), HumanMessage(content="latest")]
    cfg = Settings(_env_file=None, history_window_messages=5)

    with patch("fraudai.agents.claude_invoker.settings", cfg):
        built = ClaudeAgentInvoker._build_messages("system", {"messages": history})  # type: ignore[arg-type]

    body = built[1:]
    assert built[0].content == "system"
    assert len(body) <= 5
    assert isinstance(body[0], HumanMessage), "the window must not open on an orphan reply"
    assert body[-1].content == "latest"
    assert len(history) == 41, "the stored state is never trimmed"


def test_prompt_window_never_drops_the_latest_user_message() -> None:
    history = [HumanMessage(content="old"), *[AIMessage(content=f"a{i}") for i in range(10)]]
    history.append(HumanMessage(content="latest"))
    history.extend(AIMessage(content=f"b{i}") for i in range(10))
    cfg = Settings(_env_file=None, history_window_messages=3)

    with patch("fraudai.agents.claude_invoker.settings", cfg):
        built = ClaudeAgentInvoker._build_messages("system", {"messages": history})  # type: ignore[arg-type]

    assert any(isinstance(m, HumanMessage) and m.content == "latest" for m in built)
    assert not any(getattr(m, "content", "") == "old" for m in built)


def test_short_histories_pass_through_untouched() -> None:
    history = _conversation(2)
    cfg = Settings(_env_file=None, history_window_messages=40)

    with patch("fraudai.agents.claude_invoker.settings", cfg):
        built = ClaudeAgentInvoker._build_messages("system", {"messages": history})  # type: ignore[arg-type]

    assert built[1:] == history


async def test_the_model_only_sees_the_window() -> None:
    model = MagicMock()
    model.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
    model.bind_tools = MagicMock(return_value=model)
    invoker = _invoker_with(model)
    history = [*_conversation(30), HumanMessage(content="latest")]
    cfg = Settings(_env_file=None, history_window_messages=6)

    with patch("fraudai.agents.claude_invoker.settings", cfg):
        await invoker.invoke("harvey", "prompt", [], {"messages": history})  # type: ignore[arg-type]

    sent = model.ainvoke.await_args.args[0]
    assert len(sent) <= 7
    assert sent[-1].content == "latest"


def test_window_size_is_configurable_and_validated() -> None:
    assert Settings(_env_file=None).history_window_messages == 40
    with pytest.raises(ValueError, match="history_window_messages"):
        Settings(_env_file=None, history_window_messages=0)


# ---------------------------------------------------------------------------
# The window must not break the approval flow (real graph, scripted model)
# ---------------------------------------------------------------------------

TARGET_URL = "https://models.example.com/predict"
PLANNED_TOOL = "adversarial_evasion"


class _ScriptedChatModel:
    def __init__(self, executions: list[dict[str, Any]]) -> None:
        self.executions = executions
        self.seen_lengths: list[int] = []

    def bind_tools(self, tools: list[Any]) -> _ScriptedChatModel:
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.seen_lengths.append(len(messages))
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="stub: analysis finished")
        return AIMessage(
            content="",
            tool_calls=[
                {"name": PLANNED_TOOL, "args": {"target_endpoint": TARGET_URL}, "id": "call-1"}
            ],
        )


async def test_a_tiny_prompt_window_keeps_the_pending_action_and_the_single_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executions: list[dict[str, Any]] = []

    @tool(PLANNED_TOOL)
    async def spy_tool(target_endpoint: str, attack_type: str = "sign_noise") -> dict[str, Any]:
        """Harmless spy: records the execution and touches no network."""
        executions.append({"target_endpoint": target_endpoint})
        return {"status": "spy"}

    async def fake_classify(message: str) -> dict[str, Any]:
        return {"agent": "mike", "language": "en", "confidence": 0.99}

    model = _ScriptedChatModel(executions)
    monkeypatch.setattr(graph_module, "classify_intent_local", fake_classify)
    monkeypatch.setattr(
        graph_module, "_tool_registry", SimpleNamespace(get_tools_for_agent=lambda name: [spy_tool])
    )
    monkeypatch.setattr(graph_module, "_claude_invoker", None)
    monkeypatch.setattr(invoker_module, "_build_llm", lambda *a, **k: model)
    cfg = Settings(_env_file=None, jwt_secret=SYNTHETIC_SECRET, history_window_messages=1)
    manager = SessionManager()
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.graph = graph_module.build_fraud_ai_graph()
    app.state.session_manager = manager
    app.state.feedback_store = []

    with (
        patch("fraudai.core.config.settings", cfg),
        patch("fraudai.api.routes.settings", cfg),
        patch("fraudai.agents.claude_invoker.settings", cfg),
    ):
        headers = _headers("tenant-a")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as http:
            first = await http.post(
                "/api/v1/chat", json={"message": f"pentest {TARGET_URL}"}, headers=headers
            )
            pending = first.json()["pending_action"]
            assert pending is not None
            assert executions == []
            sid = first.json()["session_id"]
            confirm = await http.post(
                "/api/v1/confirm",
                json={"session_id": sid, "action_id": pending["action_id"], "approved": True},
                headers=headers,
            )
            repeat = await http.post(
                "/api/v1/confirm",
                json={"session_id": sid, "action_id": pending["action_id"], "approved": True},
                headers=headers,
            )

    assert confirm.status_code == 200
    assert len(executions) == 1
    assert repeat.status_code == 409
    assert max(model.seen_lengths) <= 4, "model input stays bounded"
    assert session_context.active_session_id() is None


# ---------------------------------------------------------------------------
# Restoring from SQLite respects the quotas and the history bound
# ---------------------------------------------------------------------------


async def _seed(db: Database, rows: list[tuple[str, str, int]]) -> list[str]:
    """Insert sessions (tenant, history length, in creation order); return their ids."""
    ids: list[str] = []
    for index, (tenant, history_length, _) in enumerate(rows):
        session_id = f"00000000-0000-4000-8000-{index:012d}"
        await db.create_session(session_id, tenant, "pro")
        for turn in range(history_length):
            await db.record_turn(session_id, f"agent-{turn}", max_history=10_000)
        ids.append(session_id)
    return ids


async def test_restore_applies_tenant_and_global_quotas_and_the_history_bound(
    tmp_path: Path,
) -> None:
    db = Database(tmp_path / "old.db")
    await db.initialize()
    ids = await _seed(db, [("tenant-a", 8, 0)] * 4)
    manager = SessionManager(db=db, max_sessions=3, max_sessions_per_tenant=2, max_agent_history=3)

    restored = await manager.load_from_db()

    kept = manager.list_sessions("tenant-a")
    assert restored == 2 == len(manager.list_sessions())
    assert [s.session_id for s in kept] == ids[2:], "the newest sessions survive"
    assert all(len(s.agent_history) == 3 for s in kept)
    assert all(s.agent_history == ["agent-5", "agent-6", "agent-7"] for s in kept)
    assert all(s.turn_count == 8 for s in kept), "the true total is preserved"
    # Restoring is not destructive: nothing is deleted from the database.
    for session_id in ids:
        assert await db.get_session(session_id) is not None
    await db.close()


async def test_a_new_session_after_restore_stays_within_the_tenant_quota(tmp_path: Path) -> None:
    db = Database(tmp_path / "old.db")
    await db.initialize()
    await _seed(db, [("tenant-a", 1, 0)] * 4)
    manager = SessionManager(db=db, max_sessions=100, max_sessions_per_tenant=2)
    await manager.load_from_db()

    manager.create_session("tenant-a", "pro")

    assert len(manager.list_sessions("tenant-a")) == 2
    await manager.aclose()
    await db.close()


async def test_global_cap_on_restore_evicts_from_the_heaviest_tenant(tmp_path: Path) -> None:
    db = Database(tmp_path / "old.db")
    await db.initialize()
    ids = await _seed(
        db,
        [("tenant-a", 0, 0), ("tenant-b", 0, 0), ("tenant-a", 0, 0), ("tenant-a", 0, 0)],
    )
    manager = SessionManager(db=db, max_sessions=3, max_sessions_per_tenant=10)

    await manager.load_from_db()

    assert manager.get_session(ids[1]) is not None, "the lighter tenant keeps its session"
    assert manager.get_session(ids[0]) is None, "the heaviest tenant pays, oldest first"
    assert len(manager.list_sessions()) == 3
    await db.close()


async def test_lowering_the_limits_between_runs_takes_effect_on_the_next_start(
    tmp_path: Path,
) -> None:
    db = Database(tmp_path / "old.db")
    await db.initialize()
    await _seed(db, [("tenant-a", 5, 0)] * 4)
    first = SessionManager(db=db)
    assert await first.load_from_db() == 4

    second = SessionManager(db=db, max_sessions_per_tenant=1, max_agent_history=2)

    assert await second.load_from_db() == 1
    only = second.list_sessions()[0]
    assert len(only.agent_history) == 2
    assert only.turn_count == 5
    await db.close()


async def test_restore_within_the_limits_is_unchanged(tmp_path: Path) -> None:
    db = Database(tmp_path / "old.db")
    await db.initialize()
    await _seed(db, [("tenant-a", 2, 0), ("tenant-b", 1, 0)])
    manager = SessionManager(db=db)

    assert await manager.load_from_db() == 2
    assert [len(s.agent_history) for s in manager.list_sessions()] == [2, 1]
    await db.close()
