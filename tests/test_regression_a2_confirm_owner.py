"""Regression A2: /confirm must not resume another tenant's paused session.

Audit finding: ``routes.confirm_action`` resumes the graph for any
``session_id`` with ``Command(resume=...)`` without checking the caller owns
the session (``action_id`` is ignored too; binding it is scheduled for F4 and
is out of scope here). A different tenant can therefore approve, and trigger,
the victim's pending red-teaming action.

This test is RED on purpose until ownership is enforced (expected: 403/404 and
the victim's interrupt stays pending). Real pieces: JWT auth,
``SessionManager``, ``/chat`` + ``/confirm`` routes, compiled graph with its
``MemorySaver`` and the real ``ClaudeAgentInvoker``. Donna and the chat model
at the provider edge are deterministic stubs; the model plans an
``adversarial_evasion`` call and a harmless spy tool counts its *execution*.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from fraudai.agents import claude_invoker as invoker_module
from fraudai.agents import graph as graph_module
from fraudai.api.auth import create_token
from fraudai.api.routes import router
from fraudai.api.session_manager import SessionManager

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

VICTIM_TENANT = "tenant-victim"
ATTACKER_TENANT = "tenant-attacker"
REJECTION_CODES = (403, 404)
TARGET_URL = "https://models.example.com/predict"
PLANNED_TOOL = "adversarial_evasion"


class _Probe:
    """What the stubbed provider and the spy tool observed."""

    def __init__(self) -> None:
        self.llm_calls = 0
        self.executions: list[dict[str, Any]] = []


class _ScriptedChatModel:
    """Provider-edge stand-in: plans one tool call, then answers in plain text."""

    def __init__(self, probe: _Probe) -> None:
        self._probe = probe

    def bind_tools(self, tools: list[Any]) -> _ScriptedChatModel:
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self._probe.llm_calls += 1
        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="stub: analysis finished")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": PLANNED_TOOL,
                    "args": {"target_endpoint": TARGET_URL},
                    "id": "call-planned-1",
                }
            ],
        )


@pytest.fixture()
def probe(monkeypatch: pytest.MonkeyPatch) -> _Probe:
    """Real graph + real ``ClaudeAgentInvoker``; only the model and tool are fakes.

    Executing the spy tool is the observable "offensive action ran"; merely
    invoking the specialist (e.g. to plan before asking approval) is not.
    """
    state = _Probe()

    @tool(PLANNED_TOOL)
    async def spy_tool(target_endpoint: str, attack_type: str = "sign_noise") -> dict[str, Any]:
        """Harmless spy: records the execution and touches no network."""
        state.executions.append({"target_endpoint": target_endpoint, "attack_type": attack_type})
        return {"status": "spy"}

    async def fake_classify(message: str) -> dict[str, Any]:
        return {"agent": "mike", "language": "en", "confidence": 0.99}

    registry = SimpleNamespace(get_tools_for_agent=lambda agent_name: [spy_tool])
    monkeypatch.setattr(graph_module, "classify_intent_local", fake_classify)
    monkeypatch.setattr(graph_module, "_tool_registry", registry)
    monkeypatch.setattr(graph_module, "_claude_invoker", None)  # rebuild with the fake model
    monkeypatch.setattr(
        invoker_module, "_build_llm", lambda *args, **kwargs: _ScriptedChatModel(state)
    )
    return state


@pytest.fixture()
def app(probe: _Probe) -> FastAPI:
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api/v1")
    test_app.state.graph = graph_module.build_fraud_ai_graph()
    test_app.state.session_manager = SessionManager()
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


async def test_confirm_rejects_session_owned_by_another_tenant(
    app: FastAPI,
    client: httpx.AsyncClient,
    probe: _Probe,
) -> None:
    victim_resp = await client.post(
        "/api/v1/chat",
        json={"message": f"Run a pentest against {TARGET_URL}"},
        headers=_bearer(VICTIM_TENANT),
    )
    assert victim_resp.status_code == 200
    victim_session = victim_resp.json()["session_id"]
    thread = {"configurable": {"thread_id": victim_session}}
    paused = await app.state.graph.aget_state(thread)
    assert paused.next, "victim graph is not paused on HITL; fixture is invalid"
    assert probe.executions == [], "tool executed before any approval; fixture is invalid"

    attacker_resp = await client.post(
        "/api/v1/confirm",
        json={"session_id": victim_session, "action_id": "any-action", "approved": True},
        headers=_bearer(ATTACKER_TENANT),
    )

    assert attacker_resp.status_code in REJECTION_CODES, (
        f"cross-tenant /confirm returned {attacker_resp.status_code}; expected one of "
        f"{REJECTION_CODES}. Tool executions triggered by the attacker: {probe.executions}"
    )
    assert probe.executions == [], "attacker approved and executed the victim's action"
    still_paused = await app.state.graph.aget_state(thread)
    assert still_paused.next == paused.next, "victim's pending approval was consumed"
