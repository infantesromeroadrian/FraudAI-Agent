"""Pydantic v2 request/response schemas for the FraudAI Agent API.

All models use strict type hints (Python 3.11+) and are designed
for automatic OpenAPI spec generation via FastAPI.

Reference requirements:
- SR-009: Tenant isolation (session_id scoping)
- SR-010: OAuth2/JWT auth, RBAC by tier
- BR-006: REST API as primary interface
- F2 SLAs: latency metadata in every response
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, StrictBool

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class TokenResponse(BaseModel):
    """OAuth2 token response returned after successful authentication."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(
        description="Token lifetime in seconds.",
    )


# ---------------------------------------------------------------------------
# RAG Citations
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A normative citation retrieved from the BOE/EU RAG corpus."""

    boe_id: str = Field(
        description="BOE document identifier (e.g. 'BOE-A-2010-6737').",
    )
    norma_titulo: str = Field(
        description="Full title of the cited regulation.",
    )
    articulo: str = Field(
        description="Specific article or section cited.",
    )
    texto_relevante: str = Field(
        description="Relevant text snippet from the retrieved chunk.",
    )
    score: float = Field(
        ge=0.0,
        description="Retrieval relevance score (unbounded — RRF/reranker scores may exceed 1.0).",
    )


# ---------------------------------------------------------------------------
# Tool Results
# ---------------------------------------------------------------------------


class ToolResult(BaseModel):
    """Result of a tool execution by an agent."""

    tool_name: str = Field(
        description="Canonical tool name (e.g. 'analyze_transactions').",
    )
    status: str = Field(
        description="Execution status: 'success' | 'error' | 'pending'.",
        pattern=r"^(success|error|pending)$",
    )
    result: dict[str, object] | None = Field(
        default=None,
        description=(
            "Tool output payload. None when status is 'pending', or 'error' without a "
            "declared reason; a tool that declined to run (sandbox disabled/unavailable) "
            "reports {status, error_code, error} here."
        ),
    )
    duration_ms: int = Field(
        ge=0,
        description="Execution wall-clock time in milliseconds.",
    )


# ---------------------------------------------------------------------------
# Response Metadata
# ---------------------------------------------------------------------------


class ResponseMetadata(BaseModel):
    """Metadata attached to every chat response for observability (SR-012)."""

    routing_agent: str = Field(
        description="Agent classification by Donna (e.g. 'harvey', 'louis').",
    )
    routing_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Donna's classification confidence (0-1). Fallback triggered when < 0.7.",
    )
    tokens_used: int = Field(
        ge=0,
        description="Total LLM tokens consumed (input + output).",
    )
    latency_ms: int = Field(
        ge=0,
        description="End-to-end latency in milliseconds.",
    )
    corpus_version: str = Field(
        description="RAG corpus snapshot version used for this response (MLR-006).",
    )


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Send a message to FraudAI. Donna routes to the appropriate agent."""

    message: str = Field(
        min_length=1,
        max_length=50_000,
        description="User message text.",
    )
    session_id: str | None = Field(
        default=None,
        description="Existing session ID. None creates a new session.",
    )
    agent_override: str | None = Field(
        default=None,
        description=(
            "Force routing to a specific agent, bypassing Donna. "
            "One of: 'harvey', 'louis', 'jessica', 'mike', 'rachel'."
        ),
    )
    language: str = Field(
        default="es",
        pattern=r"^(es|en)$",
        description="Response language: 'es' (Spanish) or 'en' (English). See BR-007.",
    )


class PendingAction(BaseModel):
    """One offensive tool call waiting for the user's explicit decision (R2).

    Shown verbatim to the user: ``arguments`` is exactly what will run if the
    call is approved.  Resolve it with ``POST /confirm`` using ``action_id``.
    Values inside ``arguments`` and ``target`` come from the model and must be
    rendered as plain text, never as markup.
    """

    kind: Literal["offensive_tool_approval"] = "offensive_tool_approval"
    action_id: str = Field(
        description="Server-issued id of this call; /confirm must echo it.",
    )
    agent: str = Field(description="Agent that proposed the call (e.g. 'mike').")
    tool: str = Field(description="Tool that will run (e.g. 'adversarial_evasion').")
    arguments: dict[str, object] = Field(
        description="Exact arguments the tool will receive.",
    )
    target: str | None = Field(
        default=None,
        description="Target extracted from the arguments, when one is identifiable.",
    )
    warning: str = Field(description="Plain-language risk notice.")


class ChatResponse(BaseModel):
    """Response from a FraudAI agent after processing a user message."""

    session_id: str = Field(
        description="Session identifier (new or existing).",
    )
    agent: str = Field(
        description="Agent that produced this response (e.g. 'harvey', 'louis').",
    )
    message: str = Field(
        description="Agent response in natural language.",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Normative citations from the RAG corpus.",
    )
    tool_results: list[ToolResult] = Field(
        default_factory=list,
        description="Results from tools executed during this turn.",
    )
    metadata: ResponseMetadata
    pending_action: PendingAction | None = Field(
        default=None,
        description=(
            "Set when the turn paused for approval. Nothing offensive has run; "
            "``message`` is a generic notice and the details are in this object."
        ),
    )


# ---------------------------------------------------------------------------
# File Upload
# ---------------------------------------------------------------------------


class FileUploadResponse(BaseModel):
    """Response after uploading a document to a session (UR-012).

    Uploaded files are indexed as ephemeral RAG context for the session.
    Supported formats: CSV, JSON, PDF, TXT. Max size: 100 MB (SR-008).
    """

    file_id: str = Field(
        description="Unique file identifier.",
    )
    filename: str = Field(
        description="Original filename as uploaded.",
    )
    size_bytes: int = Field(
        ge=0,
        description="File size in bytes.",
    )
    indexed: bool = Field(
        description="Whether the file has been indexed into the session RAG context.",
    )
    chunks_generated: int = Field(
        ge=0,
        description="Number of chunks generated during indexing (T-RAG-04).",
    )


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class SessionInfo(BaseModel):
    """Session metadata and history (SR-009 tenant-isolated)."""

    session_id: str
    created_at: str = Field(
        description="ISO 8601 timestamp of session creation.",
    )
    agent_history: list[str] = Field(
        description="Ordered list of agents that participated in this session.",
    )
    uploaded_files: list[str] = Field(
        description="List of file IDs uploaded during this session.",
    )
    turn_count: int = Field(
        ge=0,
        description="Number of user-agent turns in this session.",
    )


# ---------------------------------------------------------------------------
# Red Teaming HITL Confirmation (SEC-006)
# ---------------------------------------------------------------------------


class ConfirmationRequest(BaseModel):
    """Approve or reject a pending red teaming action (Mike Ross HITL).

    Red teaming tools require explicit user confirmation before execution
    to comply with SEC-006 scope control requirements.
    """

    session_id: str = Field(
        description="Session with a pending confirmation.",
    )
    action_id: str = Field(
        min_length=1,
        max_length=64,
        description="``action_id`` of the pending action; stale or unknown ids are refused.",
    )
    approved: StrictBool = Field(
        description="JSON true to approve, false to reject; strings and numbers are refused.",
    )


# ---------------------------------------------------------------------------
# Feedback (UR-014)
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    """User feedback for a specific agent response."""

    session_id: str
    message_id: str = Field(
        description="Identifier of the agent message being rated.",
    )
    rating: int = Field(
        ge=1,
        le=5,
        description="Quality rating: 1 (bad) to 5 (excellent).",
    )
    comment: str = Field(
        default="",
        max_length=2000,
        description="Optional free-text feedback.",
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Health check status for all FraudAI infrastructure components."""

    status: str = Field(
        description="Overall status: 'healthy' | 'degraded' | 'unhealthy'.",
    )
    qdrant: bool = Field(
        description="Qdrant vector database reachable.",
    )
    ollama: bool = Field(
        description="Ollama (Donna local LLM) reachable.",
    )
    claude_api: bool = Field(
        description="Anthropic Claude API reachable.",
    )
    corpus_version: str | None = Field(
        default=None,
        description="Current RAG corpus version, or None if unavailable.",
    )
