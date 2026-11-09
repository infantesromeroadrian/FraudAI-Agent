"""Regression A1: /chat must not let a tenant drive another tenant's session.

Audit finding: ``routes.chat`` accepts any client-supplied ``session_id`` and
uses it as the LangGraph ``thread_id`` without checking who owns it, so a
second tenant reads and extends the victim's checkpointed conversation.

This test is RED on purpose until ownership is enforced (expected: 403/404).
Real pieces under test: JWT auth, ``SessionManager``, the ``/chat`` route and
the compiled graph with its ``MemorySaver`` checkpointer. Only the Donna
classifier and the LLM invoker are replaced by deterministic stubs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage

from fraudai.agents import graph as graph_module
from fraudai.api.auth import create_token
from fraudai.api.routes import router
from fraudai.api.session_manager import SessionManager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

VICTIM_TENANT = "tenant-victim"
ATTACKER_TENANT = "tenant-attacker"
VICTIM_SECRET = "victim-confidential-case-4711"
REJECTION_CODES = (403, 404)


class _AgentCalls:
    """Records what the stubbed specialist saw, one entry per invocation."""

    def __init__(self) -> None:
        self.visible_texts: list[list[str]] = []


@pytest.fixture()
def agent_calls(monkeypatch: pytest.MonkeyPatch) -> _AgentCalls:
    calls = _AgentCalls()

    async def fake_classify(message: str) -> dict[str, Any]:
        return {"agent": "harvey", "language": "es", "confidence": 0.99}

    async def fake_invoke(
        agent_name: str, system_prompt: str, tools: list[Any], state: dict[str, Any]
    ) -> dict[str, Any]:
        calls.visible_texts.append([str(m.content) for m in state["messages"]])
        return {
            "message": AIMessage(content="stub reply"),
            "tool_results": [],
            "analysis_summary": None,
            "escalation": None,
        }

    monkeypatch.setattr(graph_module, "classify_intent_local", fake_classify)
    monkeypatch.setattr(graph_module, "invoke_claude_agent", fake_invoke)
    return calls


@pytest.fixture()
def session_manager() -> SessionManager:
    return SessionManager()


@pytest.fixture()
def app(session_manager: SessionManager, agent_calls: _AgentCalls) -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api/v1")
    test_app.state.graph = graph_module.build_fraud_ai_graph()
    test_app.state.session_manager = session_manager
    test_app.state.feedback_store = []
    return test_app


@pytest.fixture()
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


def _bearer(tenant_id: str) -> dict[str, str]:
    token = create_token(user_id=f"user-of-{tenant_id}", tenant_id=tenant_id, tier="pro")
    return {"Authorization": f"Bearer {token}"}


async def test_chat_rejects_session_id_owned_by_another_tenant(
    app: FastAPI,
    client: httpx.AsyncClient,
    session_manager: SessionManager,
    agent_calls: _AgentCalls,
) -> None:
    victim_resp = await client.post(
        "/api/v1/chat",
        json={"message": VICTIM_SECRET},
        headers=_bearer(VICTIM_TENANT),
    )
    assert victim_resp.status_code == 200
    victim_session = victim_resp.json()["session_id"]
    thread = {"configurable": {"thread_id": victim_session}}
    victim_history = (await app.state.graph.aget_state(thread)).values["messages"]
    assert len(agent_calls.visible_texts) == 1  # setup sanity: only the victim ran

    attacker_resp = await client.post(
        "/api/v1/chat",
        json={"message": "continue this conversation", "session_id": victim_session},
        headers=_bearer(ATTACKER_TENANT),
    )

    assert attacker_resp.status_code in REJECTION_CODES, (
        f"cross-tenant /chat returned {attacker_resp.status_code}; expected one of "
        f"{REJECTION_CODES}. Specialist invocations: {len(agent_calls.visible_texts)}; "
        f"victim text visible to attacker turn: "
        f"{any(VICTIM_SECRET in t for t in agent_calls.visible_texts[-1])}"
    )
    assert len(agent_calls.visible_texts) == 1, "victim graph thread was invoked by the attacker"
    after = (await app.state.graph.aget_state(thread)).values["messages"]
    assert len(after) == len(victim_history), "attacker wrote into the victim checkpoint"
    stored = session_manager.get_session(victim_session)
    assert stored is not None
    assert stored.tenant_id == VICTIM_TENANT
    assert stored.turn_count == 1
