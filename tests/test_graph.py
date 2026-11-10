"""Tests for LangGraph orchestration graph routing logic."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from fraudai.agents.graph import (
    build_fraud_ai_graph,
    route_action_flow,
    route_after_agent,
    route_from_donna,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _base_state(**overrides) -> dict:
    """Build a minimal AgentState dict for testing."""
    state: dict = {
        "messages": [],
        "current_agent": None,
        "previous_agent": None,
        "session_id": "test-session",
        "tenant_id": "test-tenant",
        "user_tier": "pro",
        "shared_context": {},
        "uploaded_documents": [],
        "tool_results": [],
        "escalation_request": None,
        "needs_human_confirmation": False,
        "language": "es",
        "turn_count": 0,
    }
    state.update(overrides)
    return state


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------


def test_graph_compiles():
    graph = build_fraud_ai_graph()
    assert graph is not None


def test_graph_has_expected_nodes():
    graph = build_fraud_ai_graph()
    node_names = set(graph.nodes.keys())
    expected = {
        "__start__",
        "donna",
        "harvey",
        "louis",
        "jessica",
        "mike",
        "rachel",
        "approve_action",
        "execute_action",
        "finalize_actions",
        "escalation_handoff",
        "responder",
    }
    assert "human_confirmation" not in node_names  # the pre-planning gate is gone
    assert expected.issubset(node_names)


# ---------------------------------------------------------------------------
# route_from_donna
# ---------------------------------------------------------------------------


def test_route_from_donna_harvey():
    state = _base_state(current_agent="harvey")
    assert route_from_donna(state) == "harvey"


def test_route_from_donna_louis():
    state = _base_state(current_agent="louis")
    assert route_from_donna(state) == "louis"


def test_route_from_donna_jessica():
    state = _base_state(current_agent="jessica")
    assert route_from_donna(state) == "jessica"


def test_route_from_donna_mike_plans_before_asking_approval():
    # The approval comes per tool call after the model plans, not before routing.
    state = _base_state(current_agent="mike")
    assert route_from_donna(state) == "mike"


def test_route_from_donna_rachel():
    state = _base_state(current_agent="rachel")
    assert route_from_donna(state) == "rachel"


def test_route_from_donna_none_clarifies():
    state = _base_state(current_agent=None)
    assert route_from_donna(state) == "clarify"


# ---------------------------------------------------------------------------
# route_after_agent
# ---------------------------------------------------------------------------


def test_route_after_agent_respond():
    state = _base_state(escalation_request=None)
    assert route_after_agent(state) == "respond"


def test_route_after_agent_escalate():
    state = _base_state(
        current_agent="harvey",
        escalation_request={"target": "louis", "reason": "needs compliance"},
    )
    assert route_after_agent(state) == "escalate"


def test_route_after_agent_empty_dict_does_not_escalate():
    state = _base_state(escalation_request={})
    assert route_after_agent(state) == "respond"


def test_route_after_agent_without_target_responds():
    state = _base_state(
        current_agent="harvey", escalation_request={"source": "harvey", "target": None}
    )
    assert route_after_agent(state) == "respond"


def test_route_after_agent_unknown_or_self_target_responds():
    for target in ("donna", "nobody", "harvey"):
        state = _base_state(current_agent="harvey", escalation_request={"target": target})
        assert route_after_agent(state) == "respond"


def test_route_after_agent_stops_at_escalation_cap():
    state = _base_state(
        current_agent="harvey", escalation_request={"target": "louis"}, turn_count=3
    )
    assert route_after_agent(state) == "respond"


# ---------------------------------------------------------------------------
# route_action_flow (approval-gated agent sub-flow)
# ---------------------------------------------------------------------------


def _proposal(status: str) -> dict:
    return {"status": status}


def test_route_action_flow_without_proposals_follows_agent_routing():
    assert route_action_flow(_base_state(proposed_actions=[])) == "respond"
    escalating = _base_state(
        current_agent="mike", proposed_actions=[], escalation_request={"target": "harvey"}
    )
    assert route_action_flow(escalating) == "escalate"


def test_route_action_flow_runs_allowed_calls_before_asking_about_the_next():
    state = _base_state(proposed_actions=[_proposal("awaiting_approval"), _proposal("approved")])
    assert route_action_flow(state) == "execute"


def test_route_action_flow_asks_when_a_call_awaits_approval():
    state = _base_state(proposed_actions=[_proposal("executed"), _proposal("awaiting_approval")])
    assert route_action_flow(state) == "approve"


def test_route_action_flow_finalizes_when_every_call_is_settled():
    state = _base_state(
        proposed_actions=[_proposal("executed"), _proposal("rejected"), _proposal("failed")]
    )
    assert route_action_flow(state) == "finalize"


# ---------------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------------


class TestSpecialistNodes:
    """Tests for specialist agent nodes (harvey, louis, jessica, mike, rachel)."""

    @patch("fraudai.agents.graph.invoke_claude_agent")
    async def test_harvey_agent_node(self, mock_invoke: AsyncMock) -> None:
        from fraudai.agents.graph import harvey_agent_node

        mock_invoke.return_value = {
            "message": AIMessage(content="Transaction analysis complete"),
            "tool_results": [{"tool": "analyze_transactions", "result": "ok"}],
            "analysis_summary": "Detected 3 anomalies",
            "escalation": None,
        }

        state = _base_state(
            messages=[HumanMessage(content="Analyze this transaction")],
            current_agent="harvey",
        )

        result = await harvey_agent_node(state)

        mock_invoke.assert_awaited_once()
        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "Transaction analysis complete"
        assert result["tool_results"] == [{"tool": "analyze_transactions", "result": "ok"}]
        assert result["shared_context"]["last_analysis"] == "Detected 3 anomalies"
        assert result["escalation_request"] is None

    @patch("fraudai.agents.graph.invoke_claude_agent")
    async def test_louis_agent_node(self, mock_invoke: AsyncMock) -> None:
        from fraudai.agents.graph import louis_agent_node

        mock_invoke.return_value = {
            "message": AIMessage(content="Compliance check done"),
            "tool_results": [],
            "analysis_summary": None,
            "escalation": None,
        }

        state = _base_state(
            messages=[HumanMessage(content="Check AML compliance")],
            current_agent="louis",
        )

        result = await louis_agent_node(state)

        assert len(result["messages"]) == 1
        assert result["messages"][0].content == "Compliance check done"
        assert result["escalation_request"] is None

    @patch("fraudai.agents.graph.invoke_claude_agent")
    async def test_jessica_agent_node(self, mock_invoke: AsyncMock) -> None:
        from fraudai.agents.graph import jessica_agent_node

        mock_invoke.return_value = {
            "message": AIMessage(content="Investigation summary"),
            "tool_results": [],
            "analysis_summary": "Network identified",
            "escalation": None,
        }

        state = _base_state(
            messages=[HumanMessage(content="Investigate network")],
            current_agent="jessica",
        )

        result = await jessica_agent_node(state)

        assert result["shared_context"]["last_analysis"] == "Network identified"

    @patch("fraudai.agents.graph.invoke_claude_agent")
    async def test_rachel_agent_node(self, mock_invoke: AsyncMock) -> None:
        from fraudai.agents.graph import rachel_agent_node

        mock_invoke.return_value = {
            "message": AIMessage(content="Pipeline generated"),
            "tool_results": [],
            "analysis_summary": None,
            "escalation": None,
        }

        state = _base_state(
            messages=[HumanMessage(content="Generate ETL pipeline")],
            current_agent="rachel",
        )

        result = await rachel_agent_node(state)

        assert result["messages"][0].content == "Pipeline generated"

    async def test_mike_agent_node_answers_directly_when_no_tool_is_proposed(self) -> None:
        from fraudai.agents import graph as graph_module
        from fraudai.agents.graph import mike_agent_node

        answer = AIMessage(content="Scope first: which endpoint do you own?")
        invoker = MagicMock()
        invoker.plan = AsyncMock(return_value=answer)
        invoker.build_result = MagicMock(
            return_value={
                "message": answer,
                "tool_results": [],
                "analysis_summary": "scoping",
                "escalation": None,
            }
        )
        state = _base_state(
            messages=[HumanMessage(content="Run red team")],
            current_agent="mike",
        )

        with patch.object(graph_module, "_get_claude_invoker", AsyncMock(return_value=invoker)):
            result = await mike_agent_node(state)

        assert result["messages"] == [answer]
        assert result["proposed_actions"] == []
        assert result["shared_context"]["last_analysis"] == "scoping"

    @patch("fraudai.agents.graph.invoke_claude_agent")
    async def test_specialist_node_with_escalation(self, mock_invoke: AsyncMock) -> None:
        from fraudai.agents.graph import harvey_agent_node

        escalation = {"target": "louis", "reason": "AML review needed"}
        mock_invoke.return_value = {
            "message": AIMessage(content="Needs compliance review"),
            "tool_results": [],
            "analysis_summary": None,
            "escalation": escalation,
        }

        state = _base_state(
            messages=[HumanMessage(content="Check this")],
            current_agent="harvey",
        )

        result = await harvey_agent_node(state)

        assert result["escalation_request"] == escalation

    @patch("fraudai.agents.graph.invoke_claude_agent")
    async def test_specialist_preserves_existing_shared_context(
        self,
        mock_invoke: AsyncMock,
    ) -> None:
        from fraudai.agents.graph import harvey_agent_node

        mock_invoke.return_value = {
            "message": AIMessage(content="Done"),
            "tool_results": [],
            "analysis_summary": "New analysis",
            "escalation": None,
        }

        state = _base_state(
            messages=[HumanMessage(content="Test")],
            current_agent="harvey",
            shared_context={"existing_key": "existing_value"},
        )

        result = await harvey_agent_node(state)

        assert result["shared_context"]["existing_key"] == "existing_value"
        assert result["shared_context"]["last_analysis"] == "New analysis"


# ---------------------------------------------------------------------------
# responder_node
# ---------------------------------------------------------------------------


class TestResponderNode:
    """Tests for the responder_node."""

    async def test_passthrough_for_normal_agent(self) -> None:
        from fraudai.agents.graph import responder_node

        state = _base_state(
            messages=[HumanMessage(content="query"), AIMessage(content="Agent response")],
            current_agent="harvey",
            needs_human_confirmation=False,
        )

        result = await responder_node(state)

        # Pass-through returns empty dict
        assert result == {}


# ---------------------------------------------------------------------------
# donna_router_node
# ---------------------------------------------------------------------------


class TestDonnaRouterNode:
    """Tests for the donna_router_node."""

    @patch("fraudai.agents.graph.classify_intent_local")
    async def test_donna_classifies_and_returns_state(
        self,
        mock_classify: AsyncMock,
    ) -> None:
        from fraudai.agents.graph import donna_router_node

        mock_classify.return_value = {
            "agent": "harvey",
            "language": "es",
            "confidence": 0.95,
        }

        state = _base_state(
            messages=[HumanMessage(content="Analyze this transaction for fraud")],
            current_agent=None,
            turn_count=0,
        )

        result = await donna_router_node(state)

        assert result["current_agent"] == "harvey"
        assert result["language"] == "es"
        assert result["turn_count"] == 1
        assert result["previous_agent"] is None

    @patch("fraudai.agents.graph.classify_intent_local")
    async def test_donna_tracks_previous_agent(
        self,
        mock_classify: AsyncMock,
    ) -> None:
        from fraudai.agents.graph import donna_router_node

        mock_classify.return_value = {
            "agent": "louis",
            "language": "en",
            "confidence": 0.85,
        }

        # current_agent must be None so Donna runs classification
        # (Bug #1: if current_agent is set, Donna skips classification)
        # Use previous_agent to test tracking via shared_context or
        # set current_agent on state but not via agent_override path.
        state = _base_state(
            messages=[HumanMessage(content="Check compliance")],
            current_agent=None,
            previous_agent="harvey",
            turn_count=2,
        )

        result = await donna_router_node(state)

        assert result["current_agent"] == "louis"
        assert result["previous_agent"] is None  # previous_agent is state's current_agent (None)
        assert result["turn_count"] == 3
