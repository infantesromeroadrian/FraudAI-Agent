"""FastAPI router with all FraudAI Agent API endpoints.

Connects FastAPI to the LangGraph orchestration graph, QdrantStore,
and SessionManager. All endpoints are async.

Reference architecture:
- ADR-003: LangGraph StateGraph orchestrates agents
- ADR-006: FastAPI app served via uvicorn
- F2 spec: SLA targets documented per operation
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from weakref import WeakValueDictionary

from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from fraudai.agents.hitl import APPROVAL_KIND
from fraudai.api import auth as auth_module
from fraudai.api.auth import (
    MAX_USERNAME_LENGTH,
    User,
    check_credentials,
    create_token,
    credentials_from_settings,
    get_current_user,
    require_metrics_access,
)
from fraudai.api.health import (
    HealthDetailResponse,
    ReadinessResponse,
    health_view,
    readiness_view,
)
from fraudai.api.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    ConfirmationRequest,
    FeedbackRequest,
    FileUploadResponse,
    PendingAction,
    ResponseMetadata,
    SessionInfo,
    TokenResponse,
    ToolResult,
)
from fraudai.api.session_manager import (
    SessionData,  # noqa: TC001 - FastAPI resolves dependency annotations at runtime
)
from fraudai.core.config import settings
from fraudai.core.session_context import bind_session, is_valid_session_id, upload_dir

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph

    from fraudai.api.health import HealthMonitor
    from fraudai.api.session_manager import SessionManager
    from fraudai.rag.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {".csv", ".json", ".pdf", ".txt"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
UPLOAD_READ_CHUNK_SIZE = 1024 * 1024  # 1 MiB
# Room for multipart boundaries and part headers on top of the file bytes.
MULTIPART_OVERHEAD_ALLOWANCE = 64 * 1024
MAX_UPLOADS_PER_SESSION = 20


def _get_graph(request: Request) -> CompiledStateGraph:  # type: ignore[type-arg]
    """Extract the compiled graph from app state."""
    graph: CompiledStateGraph = request.app.state.graph  # type: ignore[type-arg]
    return graph


def _get_store(request: Request) -> QdrantStore:
    """Extract the QdrantStore from app state."""
    store: QdrantStore = request.app.state.store
    return store


def _get_session_manager(request: Request) -> SessionManager:
    """Extract the SessionManager from app state."""
    mgr: SessionManager = request.app.state.session_manager
    return mgr


def _get_feedback_store(request: Request) -> list[dict[str, Any]]:
    """Extract the feedback store from app state."""
    store: list[dict[str, Any]] = request.app.state.feedback_store
    return store


def _session_not_found() -> HTTPException:
    # One response for "unknown", "malformed" and "someone else's": a distinct
    # 403 would let a caller probe which session IDs exist in other tenants.
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Session not found.",
    )


def _owned_session(session_mgr: SessionManager, session_id: str, user: User) -> SessionData:
    """Return the session if it is server-issued and belongs to the user's tenant.

    Isolation is per tenant (SR-009): every user of a tenant may use the
    tenant's sessions. Must run before the graph or the filesystem is touched.

    Raises:
        HTTPException 404: malformed, unknown or foreign session ID.
    """
    if not is_valid_session_id(session_id):
        raise _session_not_found()
    session = session_mgr.get_session(session_id)
    if session is None:
        raise _session_not_found()
    if session.tenant_id != user.tenant_id:
        logger.warning(
            "cross_tenant_session_access_denied: user=%s, tenant=%s, session=%s",
            user.user_id,
            user.tenant_id,
            session_id,
        )
        raise _session_not_found()
    return session


async def require_session_owner(
    session_id: str,
    request: Request,
    user: User = Depends(get_current_user),
) -> SessionData:
    """Dependency for routes that carry ``session_id`` in the path or query.

    Routes that carry it in the body call ``_owned_session`` first instead.
    """
    return _owned_session(_get_session_manager(request), session_id, user)


async def enforce_rate_limit(
    request: Request,
    user: User = Depends(get_current_user),
) -> None:
    """Per-tenant request budget for the expensive routes (chat, stream, confirm, upload).

    Runs after authentication, so anonymous traffic cannot spend a tenant's budget, and
    before any session lookup or graph work. An app without ``rate_limiter`` in its state
    is not limited; the production factory always installs one.

    Raises:
        HTTPException 429: budget exhausted, with ``Retry-After`` in seconds.
    """
    limiter = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    retry_after = limiter.check(user.tenant_id)
    if retry_after is not None:
        logger.warning("rate_limited: tenant=%s, user=%s", user.tenant_id, user.user_id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded.",
            headers={"Retry-After": str(retry_after)},
        )


def _resolve_chat_session(
    session_mgr: SessionManager, requested_id: str | None, user: User
) -> SessionData:
    """Create a session for a new conversation or authorize the requested one."""
    if requested_id is None:
        new_id = session_mgr.create_session(tenant_id=user.tenant_id, tier=user.tier)
        created = session_mgr.get_session(new_id)
        if created is None:  # evicted between create and get: cannot happen single-threaded
            raise _session_not_found()
        return created
    return _owned_session(session_mgr, requested_id, user)


def _extract_citations(state: dict[str, Any]) -> list[Citation]:
    """Extract RAG citations from the graph state shared_context."""
    raw_citations = (state.get("shared_context") or {}).get("citations", [])
    citations: list[Citation] = []
    for c in raw_citations:
        try:
            citations.append(Citation(**c))
        except Exception:
            logger.warning("Skipping malformed citation: %s", c)
    return citations


def _extract_tool_results(state: dict[str, Any]) -> list[ToolResult]:
    """Extract tool results from the graph state."""
    raw_results = state.get("tool_results") or []
    results: list[ToolResult] = []
    for r in raw_results:
        try:
            results.append(ToolResult(**r))
        except Exception:
            logger.warning("Skipping malformed tool result: %s", r)
    return results


def _extract_response_text(state: dict[str, Any]) -> str:
    """Extract the final response text from the last AIMessage in state.

    Filters out non-text content blocks (e.g. tool_use) to avoid leaking
    raw tool invocations into the user-visible response.
    """
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                text_parts = [
                    b.get("text", "")
                    if isinstance(b, dict) and b.get("type") == "text"
                    else b
                    if isinstance(b, str)
                    else ""
                    for b in content
                ]
                joined = "\n".join(p for p in text_parts if p)
                if joined.strip():
                    return joined
    return "No response generated."


def _build_metadata(state: dict[str, Any], latency_ms: int) -> ResponseMetadata:
    """Build response metadata from graph state."""
    shared = state.get("shared_context") or {}
    return ResponseMetadata(
        routing_agent=state.get("current_agent") or "donna",
        routing_confidence=shared.get("routing_confidence", 0.0),
        tokens_used=shared.get("tokens_used", 0),
        latency_ms=latency_ms,
        corpus_version=shared.get("corpus_version", "unknown"),
    )


# ---------------------------------------------------------------------------
# Human approval (R2): pending actions and per-session serialisation
# ---------------------------------------------------------------------------

_SPECIALIST_NODES = frozenset({"harvey", "louis", "jessica", "mike", "rachel"})

PENDING_ACTION_MESSAGE = (
    "Approval required before an offensive tool runs. "
    "Review the pending action and confirm or reject it."
)

# One lock per session serialises chat/stream/confirm so two requests cannot
# resume or start the same paused thread at once.  Scope: this process only
# (the checkpointer is in-memory too); several workers would need a shared lock
# and a shared checkpointer.  Weak values: a lock disappears once unused.
_session_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()


def _session_lock(session_id: str) -> asyncio.Lock:
    lock = _session_locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        _session_locks[session_id] = lock
    return lock


async def _pending_action(
    graph: CompiledStateGraph,  # type: ignore[type-arg]
    config: RunnableConfig,
) -> PendingAction | None:
    """The approval the session's thread is paused on, read from its checkpoint."""
    snapshot = await graph.aget_state(config)
    for task in snapshot.tasks:
        for pending in task.interrupts:
            value = pending.value
            if isinstance(value, dict) and value.get("kind") == APPROVAL_KIND:
                return PendingAction.model_validate(value)
    return None


def _approval_conflict(code: str, message: str, pending: PendingAction | None) -> HTTPException:
    """409 whose body re-presents the pending action, so a client that lost it can recover."""
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "code": code,
            "message": message,
            "pending_action": pending.model_dump() if pending is not None else None,
        },
    )


def _chat_response(
    session_id: str,
    final_state: dict[str, Any],
    pending: PendingAction | None,
    latency_ms: int,
    default_agent: str,
) -> ChatResponse:
    """Build the response for a finished or paused turn."""
    if pending is not None:
        return ChatResponse(
            session_id=session_id,
            agent=pending.agent,
            message=PENDING_ACTION_MESSAGE,
            metadata=_build_metadata(final_state, latency_ms),
            pending_action=pending,
        )
    return ChatResponse(
        session_id=session_id,
        agent=final_state.get("current_agent") or default_agent,
        message=_extract_response_text(final_state),
        citations=_extract_citations(final_state),
        tool_results=_extract_tool_results(final_state),
        metadata=_build_metadata(final_state, latency_ms),
    )


# ---------------------------------------------------------------------------
# Auth (SR-010 — OAuth2/JWT)
# ---------------------------------------------------------------------------


def _invalid_credentials() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password.",
        headers={"WWW-Authenticate": "Bearer"},
    )


@router.post("/auth/token", response_model=TokenResponse)
async def login(
    # Default "": an empty or missing field must reach the handler and get the same 401 as
    # wrong credentials, instead of FastAPI's 422 for an "absent" empty form value.
    username: str = Form(default=""),
    password: str = Form(default=""),
) -> TokenResponse:
    """Issue a JWT access token via OAuth2 password grant (D5, minimal local login).

    The user is configured on the server (``AUTH_USERNAME`` / ``AUTH_PASSWORD_HASH`` and
    optionally tenant, tier and admin flag). Identity and tier come from that configuration,
    never from the request. Empty credentials are always rejected.

    - 200: token. 401: wrong or empty credentials. 429: too many recent failures for that
      username (``Retry-After``). 503: staging/production without a configured login.
    - ``development`` with no ``AUTH_*`` configured keeps the open login (any non-empty
      credentials) so the frontend and tests work without a user store; once ``AUTH_*`` is
      configured it is enforced in every environment.

    The token goes in ``Authorization: Bearer <token>``; the client keeps it in memory only
    and logs in again on 401.
    """
    from fraudai.api.auth import JWT_EXPIRATION_HOURS

    creds = credentials_from_settings(settings)
    open_dev_login = creds is None and settings.environment == "development"
    if creds is None and not open_dev_login:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Login is not configured on this server.",
        )
    if not username or not password or len(username) > MAX_USERNAME_LENGTH:
        raise _invalid_credentials()

    throttle = auth_module.login_throttle
    retry_after = throttle.retry_after(username)
    if retry_after is not None:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )

    if creds is not None:
        # scrypt is deliberately slow: keep it off the event loop.
        if not await asyncio.to_thread(check_credentials, creds, username, password):
            throttle.record_failure(username)
            logger.warning("login_failed: attempts_tracked=%d", throttle.tracked_count())
            raise _invalid_credentials()
        token = create_token(
            user_id=creds.username,
            tenant_id=creds.tenant_id,
            tier=creds.tier,
            is_admin=creds.is_admin,
        )
    else:
        token = create_token(
            user_id=username,
            tenant_id="default",
            tier="enterprise",
            email=f"{username}@fraudai.local",
        )
    throttle.reset(username)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=JWT_EXPIRATION_HOURS * 3600,
    )


# ---------------------------------------------------------------------------
# Chat (core interaction -- ADR-003 LangGraph graph invocation)
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request_body: ChatRequest,
    request: Request,
    user: User = Depends(get_current_user),
    _budget: None = Depends(enforce_rate_limit),
) -> ChatResponse:
    """Send a message to FraudAI. Routes via Donna to the appropriate agent.

    Flow: user message -> Donna (classification) -> sub-agent -> response.
    Supports ``agent_override`` to bypass Donna routing.

    SLA targets (F2):
    - Donna routing P95: < 1.5 s
    - First agent response P95: < 5 s
    - Full response P95: < 10 s (streaming recommended for perceived latency)
    """
    graph = _get_graph(request)
    session_mgr = _get_session_manager(request)
    start_time = time.monotonic()

    # Session management: authorize before anything reaches the graph.
    session_data = _resolve_chat_session(session_mgr, request_body.session_id, user)
    session_id = session_data.session_id

    # Get previous agent from session history for conversational continuity
    previous_agent = None
    if session_data.agent_history:
        previous_agent = session_data.agent_history[-1]

    # Build input state — reset current_agent so Donna re-classifies each turn
    input_state: dict[str, Any] = {
        "messages": [HumanMessage(content=request_body.message)],
        "session_id": session_id,
        "tenant_id": user.tenant_id,
        "user_tier": user.tier,
        "language": request_body.language,
        "current_agent": None,  # Force Donna to re-classify on every message
        "previous_agent": previous_agent,  # For conversational continuity
        "escalation_request": None,  # Clear any previous escalation
        "turn_count": 0,  # Escalation cap is per turn, not per session
        "proposed_actions": [],  # A new turn never inherits an earlier approval
    }

    # Agent override: bypass Donna routing
    if request_body.agent_override is not None:
        input_state["current_agent"] = request_body.agent_override

    config = cast("RunnableConfig", {"configurable": {"thread_id": session_id}})

    async with _session_lock(session_id):
        # A paused session must be resolved first: a new message would start
        # a new run over the same thread and could skip or replace the approval.
        waiting = await _pending_action(graph, config)
        if waiting is not None:
            raise _approval_conflict(
                "pending_action_required",
                "Resolve the pending action with /confirm before sending a new message.",
                waiting,
            )
        try:
            with bind_session(session_id):
                final_state = await graph.ainvoke(input_state, config=config)
        except Exception as exc:
            logger.exception("Graph invocation failed for session %s", session_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Agent processing failed. Please try again.",
            ) from exc
        pending = await _pending_action(graph, config)

    latency_ms = int((time.monotonic() - start_time) * 1000)
    response = _chat_response(session_id, final_state, pending, latency_ms, "donna")

    # Record turn in session
    session_mgr.record_turn(session_id, response.agent)

    return response


@router.post("/chat/stream")
async def chat_stream(
    request_body: ChatRequest,
    request: Request,
    user: User = Depends(get_current_user),
    _budget: None = Depends(enforce_rate_limit),
) -> StreamingResponse:
    """SSE streaming endpoint for real-time agent responses.

    Returns a ``text/event-stream`` response with incremental tokens
    as the agent generates its answer. Recommended for all client
    integrations to meet the < 2 s perceived latency target (F2 5.3).

    Event types:
    - ``token``: Incremental text token from the agent.
    - ``citation``: A normative citation from the RAG corpus.
    - ``tool_start``: A tool execution has begun.
    - ``tool_end``: A tool execution completed (includes ToolResult).
    - ``pending_action``: The turn paused for approval; carries the
      ``PendingAction`` (tool, arguments, target, ``action_id``). Nothing
      offensive has run. Resolve it with ``POST /confirm``.
    - ``done``: Stream complete; includes full ResponseMetadata (and
      ``pending_action`` when paused).
    - ``error``: An error occurred during processing.
    """
    graph = _get_graph(request)
    session_mgr = _get_session_manager(request)

    # Session management: authorize before the stream (and the graph) starts.
    session_id = _resolve_chat_session(session_mgr, request_body.session_id, user).session_id

    input_state: dict[str, Any] = {
        "messages": [HumanMessage(content=request_body.message)],
        "session_id": session_id,
        "tenant_id": user.tenant_id,
        "user_tier": user.tier,
        "language": request_body.language,
        "current_agent": None,
        "escalation_request": None,
        "turn_count": 0,
        "proposed_actions": [],
    }

    if request_body.agent_override is not None:
        input_state["current_agent"] = request_body.agent_override

    config = cast("RunnableConfig", {"configurable": {"thread_id": session_id}})

    # Refuse before the stream opens when the session is paused on an approval.
    async with _session_lock(session_id):
        waiting = await _pending_action(graph, config)
    if waiting is not None:
        raise _approval_conflict(
            "pending_action_required",
            "Resolve the pending action with /confirm before sending a new message.",
            waiting,
        )

    async def stream_events() -> AsyncGenerator[str, None]:
        accumulated_content = ""
        agent_name = "donna"
        tool_results_acc: list[dict[str, Any]] = []
        start_time = time.monotonic()

        try:
            # Re-check under the lock: another request may have paused the
            # session between the pre-check and this generator starting.
            if await _pending_action(graph, config) is not None:
                busy = {
                    "type": "error",
                    "code": "pending_action_required",
                    "message": "Resolve the pending action with /confirm first.",
                }
                yield f"data: {json.dumps(busy)}\n\n"
                return

            async for event in graph.astream_events(input_state, config=config, version="v2"):
                event_type = event.get("event", "")

                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        accumulated_content += chunk.content
                        payload = {"type": "token", "content": chunk.content}
                        yield f"data: {json.dumps(payload)}\n\n"

                elif event_type == "on_chat_model_start":
                    # Detect which agent's model is running
                    meta = event.get("metadata", {})
                    langgraph_node = meta.get("langgraph_node", "")
                    # Only specialist nodes are agents: helper nodes such as
                    # ``finalize_actions`` also call the model but are not identities.
                    if langgraph_node in _SPECIALIST_NODES:
                        agent_name = langgraph_node

                elif event_type == "on_chain_end":
                    # Capture non-streamed messages ONLY if no LLM tokens were received.
                    # This handles the case where a node returns an AIMessage directly
                    # (e.g. Donna clarification) without invoking an LLM.
                    if not accumulated_content:
                        output: Any = event.get("data", {}).get("output", {})
                        if isinstance(output, dict):
                            msgs = output.get("messages", [])
                            for msg in msgs:
                                if (
                                    hasattr(msg, "content")
                                    and isinstance(msg, AIMessage)
                                    and msg.content
                                ):
                                    accumulated_content = str(msg.content)
                                    payload = {"type": "token", "content": msg.content}
                                    yield f"data: {json.dumps(payload)}\n\n"

                elif event_type == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    payload = {"type": "tool_start", "tool_name": tool_name}
                    yield f"data: {json.dumps(payload)}\n\n"

                elif event_type == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    output = event.get("data", {}).get("output", "")
                    tool_result = {
                        "tool_name": tool_name,
                        "status": "success",
                        "result": str(output)[:2000],
                        "duration_ms": 0,
                    }
                    tool_results_acc.append(tool_result)
                    payload = {"type": "tool_result", "tool_result": tool_result}
                    yield f"data: {json.dumps(payload)}\n\n"

            pending = await _pending_action(graph, config)
            if pending is not None:
                agent_name = pending.agent
                accumulated_content = PENDING_ACTION_MESSAGE
                pending_payload = {"type": "pending_action", "pending_action": pending.model_dump()}
                yield f"data: {json.dumps(pending_payload)}\n\n"

            # Record turn in session before completing
            session_mgr.record_turn(session_id, agent_name)

            # Build complete ChatResponse for done event
            latency_ms = int((time.monotonic() - start_time) * 1000)
            done_response = {
                "session_id": session_id,
                "agent": agent_name,
                "message": accumulated_content or "No response generated.",
                "citations": [],
                "tool_results": tool_results_acc,
                "pending_action": pending.model_dump() if pending is not None else None,
                "metadata": {
                    "routing_agent": agent_name,
                    "routing_confidence": 0.0,
                    "tokens_used": 0,
                    "latency_ms": latency_ms,
                    "corpus_version": "unknown",
                },
            }
            done_payload = {"type": "done", "response": done_response}
            yield f"data: {json.dumps(done_payload)}\n\n"

        except Exception:
            logger.exception("Stream error for session %s", session_id)
            # The cause stays in the server log; clients get a generic, stable payload.
            error_payload = {
                "type": "error",
                "code": "agent_error",
                "message": "Agent processing failed. Please try again.",
            }
            yield f"data: {json.dumps(error_payload)}\n\n"

    async def event_generator() -> AsyncGenerator[str, None]:
        # The generator runs after this handler returned, so the session must
        # be bound here, inside the task that actually drives the graph.
        async with _session_lock(session_id):
            with bind_session(session_id):
                async for chunk in stream_events():
                    yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# File Upload (UR-002, UR-012)
# ---------------------------------------------------------------------------


def _file_too_large() -> HTTPException:
    return HTTPException(
        status_code=413,
        detail=f"File too large. Maximum: {MAX_FILE_SIZE} bytes (100 MB).",
    )


async def _bounded_body(request: Request) -> AsyncGenerator[bytes, None]:
    """Yield the request body, aborting once it exceeds the upload budget.

    Counts bytes as they arrive, so a body without ``Content-Length`` (chunked)
    is cut at the limit instead of being materialised by the multipart parser.
    """
    budget = MAX_FILE_SIZE + MULTIPART_OVERHEAD_ALLOWANCE
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > budget:
            raise _file_too_large()
        yield chunk


async def _receive_upload(request: Request) -> tuple[str, bytes]:
    """Parse the multipart body of an already-authorized upload.

    The body is parsed here, not by FastAPI's ``UploadFile`` parameter, because
    FastAPI would parse the whole body before the ownership dependency runs.

    Returns:
        ``(client filename, file bytes)``; the bytes never exceed ``MAX_FILE_SIZE``.

    Raises:
        HTTPException: 400 malformed body, 413 too large, 415 wrong content type,
            422 missing ``file`` part.
    """
    content_type = request.headers.get("content-type", "").lower()
    if not content_type.startswith("multipart/form-data"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Expected multipart/form-data.",
        )
    declared = request.headers.get("content-length")
    if declared is not None:
        if not declared.isdigit():
            raise HTTPException(status_code=400, detail="Invalid Content-Length.")
        if int(declared) > MAX_FILE_SIZE + MULTIPART_OVERHEAD_ALLOWANCE:
            raise _file_too_large()

    parser = MultiPartParser(request.headers, _bounded_body(request), max_files=1, max_fields=10)
    try:
        form = await parser.parse()
    except MultiPartException as exc:
        raise HTTPException(status_code=400, detail="Malformed multipart body.") from exc

    try:
        part = form.get("file")
        if not isinstance(part, UploadFile):
            raise HTTPException(status_code=422, detail="Missing 'file' part.")
        if part.size is not None and part.size > MAX_FILE_SIZE:
            raise _file_too_large()
        buffer = bytearray()
        while block := await part.read(UPLOAD_READ_CHUNK_SIZE):
            buffer.extend(block)
            if len(buffer) > MAX_FILE_SIZE:
                raise _file_too_large()
        return part.filename or "unnamed", bytes(buffer)
    finally:
        await form.close()


_UPLOAD_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {"file": {"type": "string", "format": "binary"}},
                }
            }
        },
    }
}


@router.post("/files/upload", response_model=FileUploadResponse, openapi_extra=_UPLOAD_OPENAPI)
async def upload_file(
    request: Request,
    _budget: None = Depends(enforce_rate_limit),
    session: SessionData = Depends(require_session_owner),
) -> FileUploadResponse:
    """Upload a document for session context.

    ``session_id`` is a query parameter; the file is the ``file`` part of a
    multipart body. The session must be server-issued and owned by the caller's
    tenant, and that is checked before any byte of the body is read.
    Accepted formats: CSV, JSON, PDF, TXT.
    Maximum file size: 100 MB (SR-008), enforced while receiving the body.

    The file is indexed into an ephemeral Qdrant collection scoped to the
    session (T-RAG-04). Indexing target: < 30 s (SR-007).
    """
    store = _get_store(request)
    session_mgr = _get_session_manager(request)

    if len(session.uploaded_files) >= MAX_UPLOADS_PER_SESSION:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Upload limit reached ({MAX_UPLOADS_PER_SESSION} files per session).",
        )

    filename, content = await _receive_upload(request)
    size_bytes = len(content)

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    # Use the stored (server-issued) ID, never the raw request value, for paths.
    session_id = session.session_id

    # The on-disk name is server-generated; the client filename never reaches
    # the filesystem, so it cannot traverse or collide.
    file_id = str(uuid.uuid4())
    session_dir = upload_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    file_path = session_dir / f"{file_id}{suffix}"
    file_path.write_bytes(content)

    logger.info(
        "File saved: file_id=%s, size=%d, session=%s",
        file_id,
        size_bytes,
        session_id,
    )

    # Index into Qdrant session collection
    chunks_generated = 0
    indexed = False
    try:
        collection_name = await store.create_session_collection(session_id)
        text_content = content.decode("utf-8", errors="replace")
        raw_chunks = _split_paragraphs(text_content, file_id=file_id, filename=filename)
        if raw_chunks:
            embedder = getattr(request.app.state, "embedder", None)
            if embedder is None:
                logger.warning(
                    "Embedder not available on app.state — skipping indexing for file %s",
                    file_id,
                )
            else:
                texts = [c["text"] for c in raw_chunks]
                embeddings = embedder.encode(texts)
                for chunk, emb in zip(raw_chunks, embeddings, strict=True):
                    chunk["dense_vector"] = emb["dense"]
                    sparse_data = emb.get("sparse")
                    if sparse_data and sparse_data.get("indices"):
                        from qdrant_client import models as _qmodels

                        chunk["sparse_vector"] = _qmodels.SparseVector(
                            indices=sparse_data["indices"],
                            values=sparse_data["values"],
                        )
                chunks_generated = await store.upsert_chunks(collection_name, raw_chunks)
                indexed = True
        logger.info(
            "Indexed %d chunks for file %s in session %s",
            chunks_generated,
            file_id,
            session_id,
        )
    except Exception:
        logger.exception("Failed to index file %s in session %s", file_id, session_id)

    # Record in session
    session_mgr.record_upload(session_id, file_id)

    return FileUploadResponse(
        file_id=file_id,
        filename=filename,
        size_bytes=size_bytes,
        indexed=indexed,
        chunks_generated=chunks_generated,
    )


def _split_paragraphs(
    text: str,
    file_id: str,
    filename: str,
    max_chunk_size: int = 1000,
) -> list[dict[str, Any]]:
    """Split text into paragraph-based chunks for indexing.

    Returns chunk dicts without embeddings; the caller is responsible for
    populating ``dense_vector`` and ``sparse_vector`` from the embedder.
    Production should use RecursiveCharacterTextSplitter with overlap.
    """
    paragraphs = text.split("\n\n")
    chunks: list[dict[str, Any]] = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
            chunks.append(
                {
                    "text": current_chunk.strip(),
                    "dense_vector": [0.0] * 1024,
                    "sparse_vector": None,
                    "metadata": {
                        "file_id": file_id,
                        "filename": filename,
                        "chunk_index": len(chunks),
                    },
                }
            )
            current_chunk = para
        else:
            current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para

    if current_chunk.strip():
        chunks.append(
            {
                "text": current_chunk.strip(),
                "dense_vector": [0.0] * 1024,
                "sparse_vector": None,
                "metadata": {
                    "file_id": file_id,
                    "filename": filename,
                    "chunk_index": len(chunks),
                },
            }
        )

    return chunks


# ---------------------------------------------------------------------------
# Sessions (SR-009 tenant-isolated)
# ---------------------------------------------------------------------------


@router.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(
    session: SessionData = Depends(require_session_owner),
) -> SessionInfo:
    """Get session information and history.

    Returns metadata, agent interaction history, and uploaded file
    references. Scoped to the authenticated user's tenant (SR-009).
    """
    return SessionInfo(
        session_id=session.session_id,
        created_at=session.created_at,
        agent_history=session.agent_history,
        uploaded_files=session.uploaded_files,
        turn_count=session.turn_count,
    )


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    request: Request,
    session: SessionData = Depends(require_session_owner),
    user: User = Depends(get_current_user),
) -> None:
    """Delete a session and all associated data.

    Removes:
    - Session metadata from SessionManager
    - Ephemeral Qdrant collection for uploaded files
    - Audit log entries are **retained** per SEC-004 (1-year retention)
    """
    session_mgr = _get_session_manager(request)
    store = _get_store(request)

    session_id = session.session_id

    # Delete Qdrant session collection
    try:
        await store.delete_session_collection(session_id)
    except Exception:
        logger.warning(
            "Failed to delete Qdrant session collection for %s (may not exist)",
            session_id,
        )

    # Remove from session manager
    session_mgr.delete_session(session_id)

    logger.info("Session deleted: session_id=%s, user=%s", session_id, user.user_id)


# ---------------------------------------------------------------------------
# Red Teaming HITL Confirmation (SEC-006)
# ---------------------------------------------------------------------------


@router.post("/confirm", response_model=ChatResponse)
async def confirm_action(
    request_body: ConfirmationRequest,
    request: Request,
    user: User = Depends(get_current_user),
    _budget: None = Depends(enforce_rate_limit),
) -> ChatResponse:
    """Approve or reject ONE pending offensive tool call (Mike Ross HITL).

    The graph pauses via ``interrupt()`` (ADR-003) on each proposed call and
    ``pending_action`` tells the client exactly what is awaiting a decision.
    ``action_id`` must match that action.

    - ``approved=true``: runs that single call, once.
    - ``approved=false``: cancels it; it never runs.

    A turn may hold several calls: the response then carries the next
    ``pending_action``. 404: session not yours. 409: no pending action, or
    ``action_id`` is stale/unknown (the pending one, if any, is in the body).

    SEC-006: No offensive tool executes without explicit user approval.
    """
    graph = _get_graph(request)
    session_mgr = _get_session_manager(request)
    start_time = time.monotonic()

    # Ownership first: resuming a foreign thread would execute its pending action.
    session_id = _owned_session(session_mgr, request_body.session_id, user).session_id
    config = cast("RunnableConfig", {"configurable": {"thread_id": session_id}})

    async with _session_lock(session_id):
        # Validate against the checkpoint before resuming: a wrong id must not
        # reach the graph, and a repeated or concurrent confirm finds nothing pending.
        waiting = await _pending_action(graph, config)
        if waiting is None:
            raise _approval_conflict("no_pending_action", "No action is awaiting approval.", None)
        if waiting.action_id != request_body.action_id:
            raise _approval_conflict(
                "action_id_mismatch",
                "action_id does not match the pending action.",
                waiting,
            )

        try:
            with bind_session(session_id):
                final_state = await graph.ainvoke(
                    Command(
                        resume={"action_id": waiting.action_id, "approved": request_body.approved}
                    ),
                    config=config,
                )
        except Exception as exc:
            logger.exception("Confirmation resume failed for session %s", session_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process confirmation. The session may not have a pending action.",
            ) from exc
        pending = await _pending_action(graph, config)

    latency_ms = int((time.monotonic() - start_time) * 1000)
    response = _chat_response(session_id, final_state, pending, latency_ms, "mike")

    session_mgr.record_turn(session_id, response.agent)

    return response


# ---------------------------------------------------------------------------
# Feedback (UR-014)
# ---------------------------------------------------------------------------


@router.post("/feedback", status_code=201)
async def submit_feedback(
    request_body: FeedbackRequest,
    request: Request,
    user: User = Depends(get_current_user),
) -> dict[str, str]:
    """Submit feedback for an agent response.

    Rating scale: 1 (bad) to 5 (excellent).
    Feedback is stored for continuous evaluation and prompt tuning.
    """
    feedback_store = _get_feedback_store(request)
    session_mgr = _get_session_manager(request)
    _owned_session(session_mgr, request_body.session_id, user)

    entry: dict[str, Any] = {
        "feedback_id": str(uuid.uuid4()),
        "session_id": request_body.session_id,
        "message_id": request_body.message_id,
        "rating": request_body.rating,
        "comment": request_body.comment,
        "user_id": user.user_id,
        "tenant_id": user.tenant_id,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    db = getattr(request.app.state, "db", None)
    if db is not None:
        try:
            await db.save_feedback(
                request_body.session_id,
                request_body.message_id,
                user.user_id,
                request_body.rating,
                request_body.comment,
            )
        except Exception as exc:  # boundary: any storage failure becomes one stable 503
            logger.exception("Feedback persistence failed")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Feedback could not be stored.",
            ) from exc
    feedback_store.append(entry)

    logger.info(
        "Feedback recorded: feedback_id=%s, session=%s, rating=%d",
        entry["feedback_id"],
        request_body.session_id,
        request_body.rating,
    )

    return {"status": "accepted", "feedback_id": str(entry["feedback_id"])}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def _get_health_monitor(request: Request) -> HealthMonitor | None:
    monitor: HealthMonitor | None = getattr(request.app.state, "health_monitor", None)
    return monitor


@router.get("/livez")
async def livez() -> dict[str, str]:
    """Liveness: the process answers. Touches no dependency, so it never flaps with them."""
    return {"status": "alive"}


@router.get(
    "/readyz",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
)
async def readyz(request: Request) -> JSONResponse:
    """Readiness from the cached dependency snapshot: 200 when ready, 503 otherwise.

    ``starting`` (503) until the first check completes; a snapshot that stops refreshing
    expires. No outbound call happens while answering. ``build_sha`` identifies the image.
    """
    body = readiness_view(_get_health_monitor(request), settings)
    return JSONResponse(
        status_code=200 if body.status == "ready" else 503,
        content=body.model_dump(),
    )


@router.get("/health", response_model=HealthDetailResponse)
async def health(request: Request) -> HealthDetailResponse:
    """Dependency health from the cached snapshot (same data as ``/readyz``).

    Keeps the original fields; ``status`` is 'healthy' (all three components up),
    'degraded' (some) or 'unhealthy' (none, including before the first check, when
    ``checked_at`` is null). Always HTTP 200: use ``/readyz`` for gating.
    No authentication and no outbound calls.
    """
    return health_view(_get_health_monitor(request), settings)


# ---------------------------------------------------------------------------
# Admin (SR-012 observability)
# ---------------------------------------------------------------------------


@router.get("/admin/metrics", dependencies=[Depends(require_metrics_access)])
async def metrics() -> Response:
    """Prometheus-compatible metrics endpoint.

    Exposes:
    - Request count and latency histograms per endpoint
    - Token consumption per agent and per tenant
    - Tool call counts and durations
    - RAG retrieval latency and cache hit rate
    - Active sessions and concurrent users

    Requires the dedicated scrape credential or an admin JWT.
    """
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
