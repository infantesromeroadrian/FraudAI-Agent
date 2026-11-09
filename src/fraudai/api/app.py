"""FastAPI application factory for FraudAI Agent.

Creates and configures the ASGI app with CORS, middleware, routes,
and lifespan events. Entry point for both development and production:

    Development:  uv run uvicorn fraudai.api.app:create_app --factory --reload
    Production:   Dockerfile CMD (ADR-006)

Reference: BR-006, ADR-006.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from fraudai.core.config import Settings
    from fraudai.tools.implementations import ToolImplementations
    from fraudai.tools.sandbox import SandboxEngine

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fraudai.api.middleware import MetricsMiddleware
from fraudai.api.routes import router as api_router
from fraudai.api.session_manager import SessionManager
from fraudai.core.config import settings

logger = logging.getLogger(__name__)


class SandboxStartupError(RuntimeError):
    """The sandbox is required (SANDBOX_MODE=docker) but Docker or the image is unusable."""


async def init_sandbox_tools(
    cfg: Settings,
    engine_factory: Callable[[], SandboxEngine] | None = None,
) -> tuple[SandboxEngine | None, ToolImplementations]:
    """Build the tool implementations for the configured sandbox mode (fail closed).

    - ``disabled``: Docker is never touched; sandbox-backed tools answer with an
      explicit ``sandbox_disabled`` error.
    - ``docker`` and healthy: real implementations.
    - ``docker`` but Docker/image unusable: staging and production refuse to
      start; development starts with explicit ``sandbox_unavailable`` errors.
      A failed sandbox never degrades to stub data.
    """
    from docker.errors import DockerException

    from fraudai.tools.implementations import SandboxDisabled, ToolImplementations
    from fraudai.tools.sandbox import SandboxEngine

    if cfg.sandbox_mode == "disabled":
        logger.warning("Sandbox disabled by configuration (SANDBOX_MODE=disabled)")
        return None, ToolImplementations(
            sandbox=None,
            disabled=SandboxDisabled(
                code="sandbox_disabled",
                reason=(
                    "Sandbox-backed tools (including offensive red-teaming tools) are "
                    "disabled in this deployment (SANDBOX_MODE=disabled)."
                ),
            ),
        )

    def default_factory() -> SandboxEngine:
        return SandboxEngine(
            image=cfg.sandbox_image,
            timeout=cfg.sandbox_timeout_seconds,
            mem_limit=cfg.sandbox_mem_limit,
            cpu_count=cfg.sandbox_cpu_count,
            pids_limit=cfg.sandbox_pids_limit,
            max_concurrent=cfg.sandbox_max_concurrent,
        )

    failure = "Docker is not reachable or the sandbox image is missing"
    try:
        engine = (engine_factory or default_factory)()
        if await engine.health_check():
            logger.info("SandboxEngine healthy -- real tool implementations enabled")
            return engine, ToolImplementations(sandbox=engine)
    except (DockerException, OSError):
        logger.exception("Sandbox initialization failed")

    if cfg.environment != "development":
        msg = f"{failure} (image={cfg.sandbox_image}); refusing to start in {cfg.environment}"
        raise SandboxStartupError(msg)
    logger.error("%s -- sandbox-backed tools will return explicit errors", failure)
    return None, ToolImplementations(
        sandbox=None,
        disabled=SandboxDisabled(code="sandbox_unavailable", reason=f"{failure}."),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan: startup and shutdown hooks.

    Startup:
    - Initialize QdrantStore (create collections if missing)
    - Compile LangGraph orchestration graph
    - Create SessionManager
    - Store all on app.state for route access

    Shutdown:
    - Log cleanup (graceful shutdown)
    """
    # --- Startup ---
    from fraudai.agents.graph import build_fraud_ai_graph
    from fraudai.rag.qdrant_store import QdrantStore

    logger.info("Starting FraudAI Agent API (environment=%s)", settings.environment)

    # Qdrant vector store
    store = QdrantStore(host=settings.qdrant_host, port=settings.qdrant_port)
    try:
        await store.initialize()
        logger.info("QdrantStore initialized successfully")
    except Exception:
        logger.exception("QdrantStore initialization failed -- continuing in degraded mode")

    # Sandbox first and outside the degraded-mode try/except below: a required
    # sandbox that is unusable must stop startup instead of being swallowed.
    sandbox, tool_impls = await init_sandbox_tools(settings)

    # RAG retriever + tool registry (Bug #4 fix: wire ToolRegistry into runtime)
    from fraudai.agents.tool_registry import ToolRegistry
    from fraudai.ingestion.embeddings import EmbeddingGenerator
    from fraudai.rag.retriever import LegalRetriever

    try:
        embedder = EmbeddingGenerator(
            device="cpu"
        )  # CPU for API queries; GPU reserved for ingestion
        retriever = LegalRetriever(store=store, embedder=embedder)

        tool_registry = ToolRegistry(
            retriever=retriever,
            sandbox=sandbox,
            tool_impls=tool_impls,
        )
        logger.info(
            "ToolRegistry initialized (retriever=LegalRetriever, sandbox=%s, tool_impls=%s)",
            "enabled" if sandbox is not None else "disabled",
            "enabled" if sandbox is not None else "explicit-errors",
        )
    except Exception:
        logger.exception("ToolRegistry initialization failed — tools will use stubs")
        tool_registry = None
        retriever = None

    # Inject ToolRegistry into graph BEFORE compilation
    if tool_registry is not None:
        from fraudai.agents.graph import set_tool_registry

        set_tool_registry(tool_registry)

    # LangGraph compiled graph
    graph = build_fraud_ai_graph()
    logger.info("LangGraph orchestration graph compiled")

    # Database (SQLite persistence -- graceful degradation to in-memory if unavailable)
    from fraudai.core.database import Database

    db: Database | None = None
    try:
        db = Database()
        await db.initialize()
        logger.info("SQLite database initialized for session/feedback persistence")
    except Exception:
        logger.exception("Database initialization failed -- falling back to in-memory only")
        db = None

    # Session manager (in-memory cache + optional DB persistence)
    session_manager = SessionManager(db=db)
    if db is not None:
        await session_manager.load_from_db()

    # Feedback store (in-memory MVP -- list of deque for bounded memory, Bug #14)
    from collections import deque

    feedback_store: deque[dict[str, Any]] = deque(maxlen=10_000)

    # Publish system info to Prometheus
    from fraudai.core.metrics import SYSTEM_INFO

    SYSTEM_INFO.info(
        {
            "version": "0.1.0",
            "environment": settings.environment,
        }
    )

    # Attach to app.state for route access
    app.state.graph = graph
    app.state.store = store
    app.state.session_manager = session_manager
    app.state.feedback_store = feedback_store
    app.state.tool_registry = tool_registry
    app.state.retriever = retriever
    app.state.embedder = embedder if retriever is not None else None
    app.state.db = db

    yield

    # --- Shutdown ---
    if db is not None:
        await db.close()
    logger.info("Shutting down FraudAI Agent API")


def create_app() -> FastAPI:
    """Create and configure the FraudAI Agent FastAPI application."""

    app = FastAPI(
        title="FraudAI Agent API",
        description=(
            "AI agentic platform for banking fraud detection and AI red teaming. "
            "Multi-agent system with RAG over Spanish/EU financial regulation (BOE). "
            "Agents: Donna (router), Harvey (fraud), Louis (compliance), "
            "Jessica (intelligence), Mike (red teaming), Rachel (data engineering)."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- Structured logging with correlation IDs ---
    # Disabled during uvicorn dev: correlation_id format conflicts with
    # uvicorn's root logger handlers. Tracing works in production via
    # the MetricsMiddleware which sets correlation IDs independently.
    # setup_tracing()

    # --- CORS ---
    # Permissive in development; lock down in production via env config.
    allow_origins = (
        ["*"]
        if settings.environment == "development"
        else [o.strip() for o in settings.cors_allowed_origins.split(",") if o.strip()]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Prometheus metrics + correlation ID middleware ---
    app.add_middleware(MetricsMiddleware)

    # --- Routes ---
    app.include_router(api_router, prefix="/api/v1")

    return app
