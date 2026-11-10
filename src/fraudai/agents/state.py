"""Shared state schema for the FraudAI LangGraph.

Extracted into its own module to avoid circular imports between
``graph``, ``base``, and agent implementations.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage  # noqa: TC002 — runtime use by add_messages reducer
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from fraudai.agents.hitl import ActionProposal  # noqa: TC001 — runtime use by LangGraph type hints


class AgentState(TypedDict):
    """Global state shared across all graph nodes.

    Fields follow ADR-003 (F0.5).  The ``messages`` list uses the
    ``add_messages`` reducer so new messages are *appended* rather than
    replaced.
    """

    # Conversation history (accumulative via add_messages reducer)
    messages: Annotated[list[BaseMessage], add_messages]

    # Currently active agent
    current_agent: Literal["donna", "harvey", "louis", "jessica", "mike", "rachel"] | None

    # Previous agent (for escalation and return tracking)
    previous_agent: str | None

    # Session metadata
    session_id: str
    tenant_id: str
    user_tier: Literal["free", "pro", "enterprise"]

    # Shared context between agents (uploaded docs, prior results)
    shared_context: dict[str, Any]

    # User-uploaded documents (references to indexed files)
    uploaded_documents: list[dict[str, str]]

    # Tool call results from the current agent turn
    tool_results: list[dict[str, Any]]

    # Escalation request: current agent asks for cross-domain collaboration
    escalation_request: dict[str, Any] | None

    # Legacy flag, no longer read: HITL state lives in ``proposed_actions``.
    needs_human_confirmation: bool

    # Tool calls planned by an approval-gated agent in the current turn, frozen
    # (id, arguments, target) before the graph pauses for a human decision.
    proposed_actions: list[ActionProposal]

    # Detected session language
    language: Literal["es", "en"]

    # Turn counter for cost control
    turn_count: int
