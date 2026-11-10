"""LangGraph orchestration graph for FraudAI Agent.

Implements the StateGraph defined in ADR-003 (F0.5):
    START -> donna -> {harvey | louis | jessica | mike | rachel | clarify}
    agent -> {responder | escalation_handoff -> target specialist}
    responder -> END

Mike Ross plans with the model first, then every offensive tool call he
proposes is frozen with a server-issued ``action_id`` and waits for its own
human approval (R2):
    mike -> {approve_action <-> execute_action} -> finalize_actions

Each step is its own node because LangGraph restarts a resumed node from its
first line: planning (model call) and execution (side effects) must never sit
in the same node as ``interrupt()``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, TypedDict

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from fraudai.agents.claude_invoker import ClaudeAgentInvoker
from fraudai.agents.donna import DonnaRouter, IntentClassification
from fraudai.agents.hitl import (
    MAX_DECISION_ATTEMPTS,
    NO_APPROVAL_NOTE,
    USER_REJECTION_NOTE,
    ActionProposal,
    ActionStatus,
    action_digest,
    approval_payload,
    build_proposals,
    parse_decision,
)
from fraudai.agents.prompts import AGENT_PROMPTS
from fraudai.agents.state import AgentState
from fraudai.agents.tools import AGENT_TOOLS
from fraudai.core.config import settings

if TYPE_CHECKING:
    from collections.abc import Hashable

    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


# =========================================================================
# Return type contracts
# =========================================================================

# IntentClassification is imported from fraudai.agents.donna (canonical location)
# and re-exported here for backward compatibility.
__all__ = ["IntentClassification"]


class AgentInvocationResult(TypedDict):
    """Return contract for invoke_claude_agent."""

    message: Any  # AIMessage
    tool_results: list[dict[str, Any]]
    analysis_summary: str | None
    escalation: dict[str, Any] | None


# =========================================================================
# Donna Router — lazy singleton
# =========================================================================

_donna_router: DonnaRouter | None = None
_claude_invoker: ClaudeAgentInvoker | None = None
_init_lock = asyncio.Lock()


async def _get_donna_router() -> DonnaRouter:
    """Return the module-level DonnaRouter singleton, creating it on first call."""
    global _donna_router  # noqa: PLW0603
    async with _init_lock:
        if _donna_router is None:
            _donna_router = DonnaRouter(
                ollama_host=settings.ollama_host,
                model=settings.ollama_model,
                groq_api_key=settings.groq_api_key or None,
            )
    return _donna_router


async def classify_intent_local(message: str) -> IntentClassification:
    """Classify user intent using local Ollama model (Llama 3.1 8B).

    Delegates to ``DonnaRouter.classify()`` which calls Ollama with
    JSON mode and falls back to keyword matching on failure.

    Returns:
        Dict with keys ``agent`` (str | None), ``language`` ("es" | "en"),
        and ``confidence`` (float 0.0-1.0).
    """
    router = await _get_donna_router()
    return await router.classify(message)


async def _get_claude_invoker() -> ClaudeAgentInvoker:
    """Return the module-level AgentInvoker singleton, creating it on first call."""
    global _claude_invoker  # noqa: PLW0603
    async with _init_lock:
        if _claude_invoker is None:
            provider = settings.llm_provider
            # Each provider gets only its own credential. Ollama is the personal daemon at
            # OLLAMA_HOST: no key at all (the client sends a non-secret literal there).
            api_keys = {
                "groq": settings.groq_api_key,
                "anthropic": settings.anthropic_api_key,
                "openai": settings.openai_api_key,
                "ollama": "",
            }
            _claude_invoker = ClaudeAgentInvoker(
                api_key=api_keys[provider],
                model=settings.llm_model,
                provider=provider,
                base_url=settings.ollama_host if provider == "ollama" else None,
            )
    return _claude_invoker


async def invoke_claude_agent(
    agent_name: str,
    system_prompt: str,
    tools: list[Any],
    state: AgentState,
) -> AgentInvocationResult:
    """Invoke a Claude-backed specialist agent.

    Delegates to ``ClaudeAgentInvoker.invoke()`` which handles tool binding,
    retry logic, and escalation detection.

    Returns:
        Dict with keys ``message`` (AIMessage), ``tool_results`` (list),
        ``analysis_summary`` (str | None), ``escalation`` (dict | None).
    """
    invoker = await _get_claude_invoker()
    return await invoker.invoke(agent_name, system_prompt, tools, state)


# =========================================================================
# Routing functions
# =========================================================================


_SPECIALISTS = frozenset({"harvey", "louis", "jessica", "mike", "rachel"})


def route_from_donna(state: AgentState) -> str:
    """Route from Donna (or an escalation hand-off) to a specialist or clarification.

    Mike needs no gate here: his offensive tool calls are approved one by one,
    after the model has chosen them, inside his own sub-flow.
    """
    agent = state.get("current_agent")
    if agent in _SPECIALISTS:
        return str(agent)
    return "clarify"


# Counted on ``turn_count``, which the API resets to 0 each turn and Donna
# raises to 1: at most two hand-offs per turn, so agents cannot ping-pong.
_MAX_ESCALATIONS = 3


def route_after_agent(state: AgentState) -> str:
    """Decide whether the agent responds directly or hands off to another specialist.

    Escalation needs a named specialist target different from the current
    agent, and stays under the per-turn cap (Bug #27).
    """
    request = state.get("escalation_request") or {}
    target = request.get("target")
    if (
        target in _SPECIALISTS
        and target != state.get("current_agent")
        and state.get("turn_count", 0) < _MAX_ESCALATIONS
    ):
        return "escalate"
    return "respond"


def route_action_flow(state: AgentState) -> str:
    """Next step of an approval-gated agent's turn.

    Priority: run what is already allowed, then ask about the next call, then
    answer.  With no proposals the model answered directly in the plan step.
    """
    proposals = state.get("proposed_actions") or []
    if not proposals:
        return route_after_agent(state)
    statuses = {p["status"] for p in proposals}
    if "approved" in statuses:
        return "execute"
    if "awaiting_approval" in statuses:
        return "approve"
    return "finalize"


# =========================================================================
# Graph nodes
# =========================================================================


async def donna_router_node(state: AgentState) -> dict[str, Any]:
    """Donna: classify user intent with local model and update routing.

    If ``current_agent`` is already set (via agent_override), skip
    classification and respect the override.
    """
    # Bug #1 fix: respect agent_override from ChatRequest
    if state.get("current_agent") is not None:
        logger.info(
            "Donna: agent_override active, skipping classification -> %s", state["current_agent"]
        )
        return {"turn_count": state.get("turn_count", 0) + 1}

    last_message = state["messages"][-1]
    raw_content = last_message.content if hasattr(last_message, "content") else str(last_message)
    user_text = raw_content if isinstance(raw_content, str) else str(raw_content)

    classification = await classify_intent_local(user_text)

    agent = classification.get("agent")

    # Conversational continuity: if Donna can't classify (no keywords match)
    # and there was a previous agent in this session, continue with that agent.
    if agent is None and state.get("previous_agent"):
        agent = state["previous_agent"]
        logger.info(
            "Donna: no keyword match, continuing with previous agent '%s'",
            agent,
        )

    return {
        "current_agent": agent,
        "previous_agent": state.get("current_agent"),
        "language": classification.get("language", "es"),
        "turn_count": state.get("turn_count", 0) + 1,
    }


_tool_registry: Any | None = None


def set_tool_registry(registry: Any) -> None:
    """Inject a ToolRegistry instance for live RAG-connected tools.

    Called from ``app.py`` lifespan after the registry is created.
    If never called, agents fall back to stub tools from ``AGENT_TOOLS``.
    """
    global _tool_registry  # noqa: PLW0603
    _tool_registry = registry
    logger.info("ToolRegistry injected into graph — agents will use RAG-connected tools")


def _tools_for_agent(agent_name: str) -> list[Any]:
    """Tools for *agent_name*: ToolRegistry (RAG-connected) when injected, else stubs."""
    if _tool_registry is not None:
        try:
            tools: list[Any] = _tool_registry.get_tools_for_agent(agent_name)
        except Exception:
            logger.warning("ToolRegistry failed for '%s', falling back to stubs", agent_name)
            return list(AGENT_TOOLS.get(agent_name, []))
        return tools
    return list(AGENT_TOOLS.get(agent_name, []))


def _agent_update(state: AgentState, response: AgentInvocationResult) -> dict[str, Any]:
    """State update for a finished specialist turn."""
    shared_ctx = dict(state.get("shared_context") or {})
    if response.get("analysis_summary"):
        shared_ctx["last_analysis"] = response["analysis_summary"]

    return {
        "messages": [response["message"]],
        "tool_results": response.get("tool_results", []),
        "shared_context": shared_ctx,
        "escalation_request": response.get("escalation"),
    }


async def _specialist_node(
    agent_name: str,
    state: AgentState,
) -> dict[str, Any]:
    """Generic specialist node — calls invoke_claude_agent with the
    appropriate prompt and tools for *agent_name*.
    """
    response = await invoke_claude_agent(
        agent_name=agent_name,
        system_prompt=AGENT_PROMPTS[agent_name],
        tools=_tools_for_agent(agent_name),
        state=state,
    )
    return _agent_update(state, response)


async def harvey_agent_node(state: AgentState) -> dict[str, Any]:
    """Harvey Specter: transaction fraud analysis."""
    return await _specialist_node("harvey", state)


async def louis_agent_node(state: AgentState) -> dict[str, Any]:
    """Louis Litt: AML/KYC/compliance."""
    return await _specialist_node("louis", state)


async def jessica_agent_node(state: AgentState) -> dict[str, Any]:
    """Jessica Pearson: fraud intelligence and investigation."""
    return await _specialist_node("jessica", state)


async def mike_agent_node(state: AgentState) -> dict[str, Any]:
    """Mike Ross: AI red teaming — plan step.

    Calls the model once and freezes whatever tool calls it proposes into
    ``proposed_actions``.  Nothing runs here; ``approve_action`` and
    ``execute_action`` handle each call afterwards.  If the model proposes no
    call, its message is the answer.
    """
    agent_name = "mike"
    tools = _tools_for_agent(agent_name)
    invoker = await _get_claude_invoker()
    started = time.monotonic()

    plan_message = await invoker.plan(agent_name, AGENT_PROMPTS[agent_name], tools, state)
    calls = list(getattr(plan_message, "tool_calls", None) or [])
    if not calls:
        result = invoker.build_result(agent_name, plan_message, [], started)
        return {**_agent_update(state, result), "proposed_actions": []}

    by_name = {str(getattr(t, "name", "")): t for t in tools}
    proposals = build_proposals(agent_name, state.get("session_id", ""), calls, by_name)
    return {"proposed_actions": proposals}


async def rachel_agent_node(state: AgentState) -> dict[str, Any]:
    """Rachel Zane: data engineering and feature intelligence."""
    return await _specialist_node("rachel", state)


def _first_with_status(
    proposals: list[ActionProposal], status: str
) -> tuple[int, ActionProposal] | None:
    for index, proposal in enumerate(proposals):
        if proposal["status"] == status:
            return index, proposal
    return None


async def action_approval_node(state: AgentState) -> dict[str, Any]:
    """Ask the human about the next proposed call that needs approval.

    One interrupt per call.  The payload is built only from the frozen
    proposal, so replaying this node after a resume is deterministic and
    cannot show or approve anything different.  An answer that is not exactly
    ``{"action_id": <this id>, "approved": <bool>}`` does not approve: the call
    is asked again, and after ``MAX_DECISION_ATTEMPTS`` bad answers it is
    rejected.  ``interrupt()``'s control-flow exception must propagate, so
    nothing here catches broad exceptions.
    """
    proposals = list(state.get("proposed_actions") or [])
    found = _first_with_status(proposals, "awaiting_approval")
    if found is None:
        return {}
    index, action = found

    payload = approval_payload(action)
    approved: bool | None = None
    for _ in range(MAX_DECISION_ATTEMPTS):
        approved = parse_decision(interrupt(payload), action["action_id"])
        if approved is not None:
            break

    if approved:
        proposals[index] = {**action, "status": "approved"}
    else:
        if approved is False:
            proposals[index] = {**action, "status": "rejected", "note": USER_REJECTION_NOTE}
        else:
            proposals[index] = {**action, "status": "refused", "note": NO_APPROVAL_NOTE}
    return {"proposed_actions": proposals}


async def action_executor_node(state: AgentState) -> dict[str, Any]:
    """Run the next approved call, once, and record its outcome on the proposal.

    Runs in its own node, after the approval has been persisted, so replaying
    ``interrupt()`` never re-executes an earlier call.  A call that needs
    approval is re-checked against its frozen digest before it runs.
    """
    proposals = list(state.get("proposed_actions") or [])
    found = _first_with_status(proposals, "approved")
    if found is None:
        return {}
    index, action = found

    if action["requires_approval"]:
        try:
            current = action_digest(
                state.get("session_id", ""), action["agent"], action["tool"], action["arguments"]
            )
        except ValueError:
            current = ""
        if current != action["digest"]:
            proposals[index] = {
                **action,
                "status": "failed",
                "note": "Approved action no longer matches what was reviewed; not executed.",
            }
            return {"proposed_actions": proposals}

    outcome = await ClaudeAgentInvoker.run_tool_call(
        _tools_for_agent(action["agent"]), action["tool"], action["arguments"]
    )
    status: ActionStatus = "executed" if outcome["status"] == "success" else "failed"
    proposals[index] = {**action, "status": status, "result": outcome}
    return {"proposed_actions": proposals}


_REJECTED_PLAN_MESSAGE = "Red teaming action cancelled by user. No offensive tools were executed."
_REFUSED_DETAIL = "(invalid, unreviewable or without a valid approval)"


def _nothing_ran_message(proposals: list[ActionProposal]) -> str:
    """Canned reply when no call ran; says who stopped each one, truthfully."""
    by_user = sum(p["status"] == "rejected" for p in proposals)
    refused = len(proposals) - by_user
    if refused == 0:
        return _REJECTED_PLAN_MESSAGE
    if by_user == 0:
        return f"No action was executed: every proposed call was refused {_REFUSED_DETAIL}."
    return (
        f"No action was executed: {by_user} rejected by the user, "
        f"{refused} refused before approval {_REFUSED_DETAIL}."
    )


def _outcome_for_model(action: ActionProposal) -> dict[str, Any]:
    """Result dict handed to the model for one proposal (executed or not)."""
    if action["result"] is not None:
        return action["result"]
    return {
        "tool_name": action["tool"],
        "status": "error",
        "error": action["note"] or "Not executed.",
    }


async def action_finalize_node(state: AgentState) -> dict[str, Any]:
    """Answer once every proposed call has a final status, then clear them.

    If the user rejected everything, no further model call is made.
    """
    proposals = list(state.get("proposed_actions") or [])
    cleared: dict[str, Any] = {"proposed_actions": []}
    if all(p["status"] in ("rejected", "refused") for p in proposals):
        return {
            **cleared,
            "messages": [AIMessage(content=_nothing_ran_message(proposals))],
            "tool_results": [],
            "escalation_request": None,
        }

    agent_name = proposals[0]["agent"]
    invoker = await _get_claude_invoker()
    started = time.monotonic()
    plan_message = AIMessage(
        content="",
        tool_calls=[
            {"name": p["tool"], "args": p["arguments"], "id": p["tool_call_id"]} for p in proposals
        ],
    )
    outcomes = [(p["tool_call_id"], _outcome_for_model(p)) for p in proposals]
    final_message = await invoker.respond_with_tool_results(
        agent_name, AGENT_PROMPTS[agent_name], state, plan_message, outcomes
    )
    executed = [p["result"] for p in proposals if p["result"] is not None]
    result = invoker.build_result(agent_name, final_message, executed, started)
    return {**_agent_update(state, result), **cleared}


async def escalation_handoff_node(state: AgentState) -> dict[str, Any]:
    """Hand the turn to the specialist named in the escalation request.

    The request is consumed here, so the target cannot be re-escalated by a
    stale flag, and ``turn_count`` advances to enforce the per-turn cap.
    """
    request = state.get("escalation_request") or {}
    return {
        "current_agent": request.get("target"),
        "previous_agent": state.get("current_agent"),
        "escalation_request": None,
        "proposed_actions": [],
        "turn_count": state.get("turn_count", 0) + 1,
    }


_CLARIFICATION_MESSAGE = (
    "Soy Donna Paulsen, directora del bufete FraudAI. "
    "Puedo derivarte al especialista adecuado. Dime en qué necesitas ayuda:\n\n"
    "- **Harvey Specter** — Análisis de transacciones y detección de fraude\n"
    "- **Louis Litt** — Normativa AML, KYC, compliance y reportes regulatorios\n"
    "- **Jessica Pearson** — Investigación de redes de fraude y análisis de grafos\n"
    "- **Mike Ross** — Red teaming y seguridad de modelos de IA\n"
    "- **Rachel Zane** — Pipelines de datos y feature engineering\n\n"
    "Cuéntame tu caso y te derivo al especialista correcto."
)


async def responder_node(state: AgentState) -> dict[str, Any]:
    """Format the final response to the user.

    Handles two cases:
    1. Clarification: Donna couldn't route — ask the user for more detail.
    2. Pass-through: the specialist's response is already in messages.
    """
    # If Donna couldn't classify (agent is None or donna), send clarification
    current = state.get("current_agent")
    if current is None or current == "donna":
        return {
            "messages": [AIMessage(content=_CLARIFICATION_MESSAGE)],
        }

    # Pass-through: the specialist's response is already in messages
    return {}


# =========================================================================
# Graph builder
# =========================================================================


def build_fraud_ai_graph() -> CompiledStateGraph:  # type: ignore[type-arg]
    """Build and compile the FraudAI orchestration graph.

    Returns a compiled ``StateGraph`` with MemorySaver checkpointer
    (suitable for development and testing).  Production deployments
    should replace with ``PostgresSaver``.
    """
    graph = StateGraph(AgentState)

    # --- Nodes ---
    graph.add_node("donna", donna_router_node)
    graph.add_node("harvey", harvey_agent_node)
    graph.add_node("louis", louis_agent_node)
    graph.add_node("jessica", jessica_agent_node)
    graph.add_node("mike", mike_agent_node)
    graph.add_node("rachel", rachel_agent_node)
    graph.add_node("approve_action", action_approval_node)
    graph.add_node("execute_action", action_executor_node)
    graph.add_node("finalize_actions", action_finalize_node)
    graph.add_node("escalation_handoff", escalation_handoff_node)
    graph.add_node("responder", responder_node)

    # --- Edges ---

    # START -> Donna always
    graph.add_edge(START, "donna")

    # Donna -> specialist via conditional routing
    specialist_routes: dict[Hashable, str] = {
        "harvey": "harvey",
        "louis": "louis",
        "jessica": "jessica",
        "mike": "mike",
        "rachel": "rachel",
        "clarify": "responder",
    }
    graph.add_conditional_edges("donna", route_from_donna, specialist_routes)

    # Each plain specialist -> responder OR hand-off to the named specialist
    after_agent: dict[Hashable, str] = {"respond": "responder", "escalate": "escalation_handoff"}
    for agent_name in ("harvey", "louis", "jessica", "rachel"):
        graph.add_conditional_edges(agent_name, route_after_agent, after_agent)

    # Mike: plan -> (execute | approve)* -> finalize -> responder / hand-off
    action_flow: dict[Hashable, str] = {
        "execute": "execute_action",
        "approve": "approve_action",
        "finalize": "finalize_actions",
        **after_agent,
    }
    for node in ("mike", "approve_action", "execute_action"):
        graph.add_conditional_edges(node, route_action_flow, action_flow)
    graph.add_conditional_edges("finalize_actions", route_after_agent, after_agent)

    # Hand-off -> the specialist it names
    graph.add_conditional_edges("escalation_handoff", route_from_donna, specialist_routes)

    # Responder -> END
    graph.add_edge("responder", END)

    # --- Compile ---
    checkpointer = MemorySaver()

    return graph.compile(checkpointer=checkpointer)
