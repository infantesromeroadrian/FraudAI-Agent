"""Regression A4: /chat must expose a pending HITL action to the client.

Audit finding: when Mike Ross is routed, ``human_confirmation_node`` calls
``interrupt()`` and the graph pauses, but ``routes.chat`` only looks for an
``AIMessage`` and answers "No response generated." The client never learns
that an approval is pending, nor which tool, arguments, target and action id
it would be approving.

This test is RED on purpose until ``ChatResponse`` carries ``pending_action``.
It uses the real compiled graph with its ``MemorySaver`` checkpointer and the
real ``ClaudeAgentInvoker``. Only the Donna classifier and the chat model at
the provider edge (``_build_llm``) are deterministic stubs; the model plans an
``adversarial_evasion`` tool call and a spy tool counts real *executions*.
The precondition is "no tool executed", not "no LLM call": a correct fix may
plan with the model before asking for approval.
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

TARGET_URL = "https://models.example.com/predict"
PLANNED_TOOL = "adversarial_evasion"
USER_REQUEST = f"Run an {PLANNED_TOOL} pentest against {TARGET_URL}"

# The issue fixes the concepts, not the field names; accept the natural
# spellings so the test does not pin internal naming.
TOOL_KEYS = ("tool", "tool_name")
ARGUMENTS_KEYS = ("arguments", "args", "tool_args")
TARGET_KEYS = ("target", "target_endpoint")


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


def _first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    return next((payload[k] for k in keys if payload.get(k)), None)


async def test_chat_exposes_pending_action_when_graph_awaits_approval(
    app: FastAPI,
    client: httpx.AsyncClient,
    probe: _Probe,
) -> None:
    token = create_token(user_id="analyst", tenant_id="tenant-a", tier="pro")

    resp = await client.post(
        "/api/v1/chat",
        json={"message": USER_REQUEST},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert resp.status_code == 200
    data = resp.json()
    # Preconditions: the interruption is real and nothing offensive ran, so a
    # failure below is about the missing client contract, not the fixture.
    snapshot = await app.state.graph.aget_state({"configurable": {"thread_id": data["session_id"]}})
    pending = [i for task in snapshot.tasks for i in task.interrupts]
    assert pending, "graph did not pause on a HITL interrupt; fixture is invalid"
    assert probe.executions == [], "offensive tool executed before approval"

    assert "pending_action" in data, (
        f"/chat hides the pending approval: message={data['message']!r}, keys={sorted(data)}"
    )
    action = data["pending_action"]
    assert isinstance(action, dict)
    assert isinstance(action.get("action_id"), str) and action["action_id"]
    assert _first_present(action, TOOL_KEYS) == PLANNED_TOOL, f"wrong tool: {action}"
    assert _first_present(action, ARGUMENTS_KEYS) is not None, f"no arguments: {action}"
    assert TARGET_URL in str(_first_present(action, TARGET_KEYS)), f"no target: {action}"
