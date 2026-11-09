"""End-to-end tests for the FraudAI Agent system.

Tests the full flow: HTTP request -> API -> Graph -> Agent -> Response.
External services (Groq, Qdrant, Ollama, Docker) are mocked so tests
run without any infrastructure.  The LangGraph graph executes for real;
only the leaf calls (LLM invocations, intent classifier) are stubbed.

Mock strategy:
- ``classify_intent_local`` -> deterministic keyword classifier (no Ollama)
- ``invoke_claude_agent`` -> canned specialist responses (no LLM API)
- ``QdrantStore`` -> in-memory mock (no Docker/Qdrant)
- Graph is compiled and executed for real (``build_fraud_ai_graph``)
- FastAPI app is assembled manually (ASGITransport does not run lifespan)
  with app.state populated from real graph + mocked services
"""

from __future__ import annotations

import io
from collections import deque
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage

import fraudai.agents.graph as graph_mod  # Pre-import so patch targets exist
from fraudai.api.auth import User, get_admin_user, get_current_user
from fraudai.api.session_manager import SessionManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TEST_USER = User(
    user_id="e2e-user",
    tenant_id="e2e-tenant",
    email="e2e@fraudai.local",
    tier="enterprise",
    is_admin=False,
)

_ADMIN_USER = User(
    user_id="e2e-admin",
    tenant_id="e2e-tenant",
    email="admin@fraudai.local",
    tier="enterprise",
    is_admin=True,
)

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_classification(
    agent: str | None,
    language: str = "es",
    confidence: float = 0.85,
) -> dict[str, Any]:
    """Build a fake IntentClassification dict."""
    return {"agent": agent, "language": language, "confidence": confidence}


def _keyword_classifier(message: str) -> dict[str, Any]:
    """Simple keyword-based classifier for deterministic E2E routing.

    Mirrors the real Donna keyword fallback but simplified for test control.
    Checks specialist agents with distinctive keywords first to avoid
    overly broad matches (e.g. "fraude" triggering Harvey when the user
    really wants Jessica's graph analysis or Rachel's pipeline).
    """
    msg = message.lower()
    # Jessica: graph analysis, network investigation -- check before Harvey
    if any(w in msg for w in ("red de fraude", "grafo", "investigaci", "graph")):
        return _make_classification("jessica")
    # Rachel: data engineering, pipelines -- check before Harvey
    if any(w in msg for w in ("pipeline", "etl", "feature", "data engineer", "calidad de datos")):
        return _make_classification("rachel")
    # Mike: red teaming, adversarial
    mike_kw = ("pentest", "red team", "adversar", "prompt injection", "seguridad ia")
    if any(w in msg for w in mike_kw):
        return _make_classification("mike")
    # Louis: compliance, legal
    if any(w in msg for w in ("ley", "blanque", "compliance", "normativ", "legal")):
        return _make_classification("louis")
    # Harvey: fraud, transactions (broadest -- last among specialists)
    if any(w in msg for w in ("estaf", "fraud", "transacci", "anomal")):
        return _make_classification("harvey")
    # No match -> clarification (Donna)
    return _make_classification(None, confidence=0.0)


def _make_invoker_result(agent_name: str) -> dict[str, Any]:
    """Build a fake AgentInvocationResult for a specialist agent."""
    responses: dict[str, str] = {
        "harvey": (
            "He analizado las transacciones proporcionadas. "
            "Detecto patrones sospechosos en las transferencias de alto valor."
        ),
        "louis": (
            "Segun la normativa AML vigente y el SEPBLAC, "
            "las obligaciones de reporte incluyen operaciones superiores a 10.000 EUR."
        ),
        "jessica": (
            "El analisis de grafos revela una red de cuentas interconectadas "
            "con patron de estructuracion financiera."
        ),
        "mike": (
            "He identificado vulnerabilidades en el modelo ML. "
            "Se recomienda aplicar adversarial training y input validation."
        ),
        "rachel": (
            "He disenado un pipeline ETL con las siguientes etapas: "
            "ingesta, validacion, transformacion, feature engineering y carga."
        ),
    }
    text = responses.get(agent_name, f"Response from {agent_name}")
    return {
        "message": AIMessage(content=text),
        "tool_results": [],
        "analysis_summary": None,
        "escalation": None,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_graph_singletons():
    """Reset module-level singletons in graph.py between tests.

    The graph module caches DonnaRouter, ClaudeAgentInvoker, and
    ToolRegistry singletons.  We must clear them so each test gets
    fresh mocks and an isolated graph.
    """
    old_donna = graph_mod._donna_router
    old_invoker = graph_mod._claude_invoker
    old_registry = graph_mod._tool_registry
    graph_mod._donna_router = None
    graph_mod._claude_invoker = None
    graph_mod._tool_registry = None
    yield
    graph_mod._donna_router = old_donna
    graph_mod._claude_invoker = old_invoker
    graph_mod._tool_registry = old_registry


@pytest.fixture()
async def client() -> AsyncClient:
    """Create an async test client with the real graph and mocked externals.

    The app is assembled manually (like production) with:
    - Real compiled LangGraph graph
    - Mocked QdrantStore, SessionManager, feedback store
    - Mocked intent classifier and agent invoker

    Auth is overridden to return a fixed test user.
    """

    # -- Patch intent classifier and agent invoker on graph module --
    async def _classify(message: str) -> dict[str, Any]:
        return _keyword_classifier(message)

    async def _invoke_agent(
        agent_name: str,
        system_prompt: str,
        tools: list[Any],
        state: Any,
    ) -> dict[str, Any]:
        return _make_invoker_result(agent_name)

    with (
        patch.object(graph_mod, "classify_intent_local", side_effect=_classify),
        patch.object(graph_mod, "invoke_claude_agent", side_effect=_invoke_agent),
    ):
        from fraudai.agents.graph import build_fraud_ai_graph
        from fraudai.api.app import create_app

        # Build the real graph (with mocked leaf functions)
        compiled_graph = build_fraud_ai_graph()

        # Create the app (lifespan won't run via ASGITransport)
        app = create_app()

        # Manually populate app.state as the lifespan would
        mock_store = MagicMock()
        mock_store.initialize = AsyncMock()
        mock_store.health_check = AsyncMock(return_value=False)
        mock_store.get_corpus_version = AsyncMock(return_value="e2e-test")
        mock_store.create_session_collection = AsyncMock(return_value="session_e2e")
        mock_store.upsert_chunks = AsyncMock(return_value=3)
        mock_store.delete_session_collection = AsyncMock()

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [
            {
                "dense": [0.1] * 1024,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.5, 0.3]},
            }
        ]

        app.state.graph = compiled_graph
        app.state.store = mock_store
        app.state.session_manager = SessionManager()
        app.state.feedback_store = deque(maxlen=10_000)
        app.state.tool_registry = None
        app.state.retriever = None
        app.state.embedder = mock_embedder

        # Override auth dependencies
        app.dependency_overrides[get_current_user] = lambda: _TEST_USER
        app.dependency_overrides[get_admin_user] = lambda: _ADMIN_USER

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://e2e-test") as ac:
            yield ac  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chat_payload(
    message: str,
    session_id: str | None = None,
    agent_override: str | None = None,
    language: str = "es",
) -> dict[str, Any]:
    payload: dict[str, Any] = {"message": message, "language": language}
    if session_id is not None:
        payload["session_id"] = session_id
    if agent_override is not None:
        payload["agent_override"] = agent_override
    return payload


# =========================================================================
# E2E Test Cases
# =========================================================================


class TestE2EHolaDonnaClarity:
    """1. E2E: Hola -> Donna clarification."""

    async def test_hola_returns_donna_clarification(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/chat",
            json=_chat_payload("Hola"),
        )
        assert resp.status_code == 200
        data = resp.json()
        # Donna cannot classify "Hola" -> clarification response
        assert data["agent"] == "donna"
        assert "Harvey Specter" in data["message"] or "especialista" in data["message"].lower()


class TestE2EEstafaHarveyRouting:
    """2. E2E: Estafa -> Harvey routing."""

    async def test_estafa_routes_to_harvey(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/chat",
            json=_chat_payload("Me han estafado con una transferencia bancaria"),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent"] == "harvey"
        assert data["message"]  # Non-empty response


class TestE2ELeyLouisRouting:
    """3. E2E: Ley -> Louis routing."""

    async def test_ley_routes_to_louis(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/chat",
            json=_chat_payload("Que dice la ley sobre blanqueo de capitales?"),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent"] == "louis"
        assert data["message"]


class TestE2ERedTeamingMikeRouting:
    """4. E2E: Red teaming -> Mike -> per-call human approval (R2).

    Mike plans with the model, then the graph pauses on the concrete call.  The
    old test only checked that ``message`` was non-empty, which the paused graph
    satisfied with "No response generated."; this one checks what the client
    must actually see and that nothing ran before the decision.
    """

    @staticmethod
    def _install_planning_invoker(target: str) -> MagicMock:
        plan = AIMessage(
            content="",
            tool_calls=[
                {"name": "adversarial_evasion", "args": {"target_endpoint": target}, "id": "c1"}
            ],
        )
        final = AIMessage(content="Se recomienda aplicar adversarial training.")
        invoker = MagicMock()
        invoker.plan = AsyncMock(return_value=plan)
        invoker.respond_with_tool_results = AsyncMock(return_value=final)
        invoker.build_result = MagicMock(
            return_value={
                "message": final,
                "tool_results": [],
                "analysis_summary": None,
                "escalation": None,
            }
        )
        graph_mod._claude_invoker = invoker
        return invoker

    async def test_pentest_pauses_on_the_concrete_call_until_confirmed(
        self, client: AsyncClient
    ) -> None:
        target = "https://models.example.com/predict"
        invoker = self._install_planning_invoker(target)

        resp = await client.post(
            "/api/v1/chat",
            json=_chat_payload(
                "Necesito hacer un pentest a un modelo de ML", agent_override="mike"
            ),
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["agent"] == "mike"
        assert "No response generated" not in data["message"]
        action = data["pending_action"]
        assert (action["tool"], action["target"]) == ("adversarial_evasion", target)
        # The effective arguments, default included, are what the user approves.
        assert action["arguments"] == {"target_endpoint": target, "attack_type": "sign_noise"}
        assert data["tool_results"] == []
        invoker.respond_with_tool_results.assert_not_awaited()

        confirmed = await client.post(
            "/api/v1/confirm",
            json={
                "session_id": data["session_id"],
                "action_id": action["action_id"],
                "approved": True,
            },
        )

        assert confirmed.status_code == 200
        done = confirmed.json()
        assert done["pending_action"] is None
        assert "adversarial training" in done["message"]
        invoker.plan.assert_awaited_once()  # the approval resumed; it did not re-plan


class TestE2EPipelineRachelRouting:
    """5. E2E: Pipeline -> Rachel routing."""

    async def test_pipeline_routes_to_rachel(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/chat",
            json=_chat_payload("Necesito un pipeline ETL para deteccion de fraude"),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent"] == "rachel"
        assert data["message"]


class TestE2ESessionPersistence:
    """6. E2E: Session persistence across turns."""

    async def test_session_persists_across_turns(self, client: AsyncClient) -> None:
        # First turn: get session_id
        resp1 = await client.post(
            "/api/v1/chat",
            json=_chat_payload("Me han estafado"),
        )
        assert resp1.status_code == 200
        session_id = resp1.json()["session_id"]
        assert session_id  # Non-empty

        # Second turn: same session_id
        resp2 = await client.post(
            "/api/v1/chat",
            json=_chat_payload("Dame mas detalles del analisis", session_id=session_id),
        )
        assert resp2.status_code == 200
        assert resp2.json()["session_id"] == session_id


class TestE2EAgentOverride:
    """7. E2E: Agent override bypasses Donna routing."""

    async def test_agent_override_forces_louis(self, client: AsyncClient) -> None:
        # Message would normally route to Harvey, but override to Louis
        resp = await client.post(
            "/api/v1/chat",
            json=_chat_payload(
                "Me han estafado con una transferencia",
                agent_override="louis",
            ),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent"] == "louis"


class TestE2EHealthEndpoint:
    """8. E2E: Health endpoint returns correct structure."""

    async def test_health_returns_200_and_structure(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        # Verify all expected fields
        assert "status" in data
        assert "qdrant" in data
        assert "ollama" in data
        assert "claude_api" in data
        assert isinstance(data["qdrant"], bool)
        assert isinstance(data["ollama"], bool)
        assert isinstance(data["claude_api"], bool)
        # With all services mocked/unavailable, expect degraded or unhealthy
        assert data["status"] in ("healthy", "degraded", "unhealthy")


class TestE2EFeedback:
    """9. E2E: Feedback submission after a chat turn."""

    async def test_feedback_accepted(self, client: AsyncClient) -> None:
        # First: get a chat response with session_id
        chat_resp = await client.post(
            "/api/v1/chat",
            json=_chat_payload("Me han estafado"),
        )
        assert chat_resp.status_code == 200
        session_id = chat_resp.json()["session_id"]

        # Submit feedback
        feedback_resp = await client.post(
            "/api/v1/feedback",
            json={
                "session_id": session_id,
                "message_id": "msg-001",
                "rating": 5,
                "comment": "Excellent analysis",
            },
        )
        assert feedback_resp.status_code == 201
        data = feedback_resp.json()
        assert data["status"] == "accepted"
        assert "feedback_id" in data


class TestE2EFileUpload:
    """10. E2E: File upload with CSV."""

    async def test_csv_upload_returns_file_info(self, client: AsyncClient) -> None:
        # Create a session first
        chat_resp = await client.post(
            "/api/v1/chat",
            json=_chat_payload("Hola"),
        )
        session_id = chat_resp.json()["session_id"]

        # Read test CSV fixture
        csv_path = _FIXTURES_DIR / "test_transactions.csv"
        csv_content = csv_path.read_bytes()

        # Upload the file
        resp = await client.post(
            "/api/v1/files/upload",
            params={"session_id": session_id},
            files={"file": ("test_transactions.csv", io.BytesIO(csv_content), "text/csv")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "file_id" in data
        assert data["filename"] == "test_transactions.csv"
        assert data["size_bytes"] > 0
        assert data["chunks_generated"] >= 0


# =========================================================================
# Additional E2E tests
# =========================================================================


class TestE2EJessicaRouting:
    """E2E: Network investigation -> Jessica routing."""

    async def test_network_investigation_routes_to_jessica(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/chat",
            json=_chat_payload("Necesito investigar una red de fraude con analisis de grafos"),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent"] == "jessica"


class TestE2EResponseMetadata:
    """E2E: Response metadata contains expected fields."""

    async def test_metadata_present_in_response(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/chat",
            json=_chat_payload("Me han estafado"),
        )
        assert resp.status_code == 200
        data = resp.json()
        meta = data["metadata"]
        assert "routing_agent" in meta
        assert "latency_ms" in meta
        assert meta["latency_ms"] >= 0
        assert "tokens_used" in meta
        assert "corpus_version" in meta


class TestE2ESessionInfo:
    """E2E: Session info endpoint after chat turns."""

    async def test_session_info_reflects_turns(self, client: AsyncClient) -> None:
        # Chat to create session
        chat_resp = await client.post(
            "/api/v1/chat",
            json=_chat_payload("Me han estafado"),
        )
        session_id = chat_resp.json()["session_id"]

        # Get session info
        resp = await client.get(f"/api/v1/sessions/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert data["turn_count"] >= 1
        assert len(data["agent_history"]) >= 1


class TestE2EAuthToken:
    """E2E: Auth token issuance in dev mode."""

    async def test_token_endpoint_returns_jwt(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/auth/token",
            data={"username": "e2e-test-user", "password": "test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0
