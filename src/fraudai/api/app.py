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
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from fraudai.core.config import Settings
    from fraudai.tools.implementations import ToolImplementations
    from fraudai.tools.sandbox import SandboxEngine

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fraudai.api.auth import configure_login_throttle, credentials_from_settings, load_scrape_token
from fraudai.api.health import HealthMonitor, build_checks, served_build_sha
from fraudai.api.limits import TenantRateLimiter
from fraudai.api.logging_config import configure_logging
from fraudai.api.middleware import MetricsMiddleware
from fraudai.api.routes import router as api_router
from fraudai.api.session_manager import SessionManager
from fraudai.core.config import is_volatile_sqlite_path, settings
from fraudai.core.database import Database, DatabaseUnavailableError

logger = logging.getLogger(__name__)


class _UnavailableRetriever:
    """Stand-in used when RAG failed to initialise: ``search_boe`` errors explicitly."""

    async def retrieve(self, *_: Any, **__: Any) -> list[Any]:
        msg = "search_boe is unavailable: the RAG components failed to initialise"
        raise RuntimeError(msg)

    def format_citations(self, *_: Any, **__: Any) -> list[dict[str, Any]]:
        return []


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


async def init_database(cfg: Settings) -> Database | None:
    """Open the configured SQLite database; staging and production never run without it.

    A database that cannot be created or written stops startup outside development, so a
    deployment can never silently lose sessions and feedback to a memory-only fallback.
    Development degrades, loudly, to in-memory state.
    """
    if cfg.environment != "development" and is_volatile_sqlite_path(cfg.database_path):
        msg = f"DATABASE_PATH is a volatile destination; refusing to start in {cfg.environment}"
        raise DatabaseUnavailableError(msg)
    db = Database(cfg.database_path)
    try:
        await db.initialize()
    except DatabaseUnavailableError:
        if cfg.environment != "development":
            logger.exception("Database unavailable; refusing to start in %s", cfg.environment)
            raise
        logger.exception(
            "Database unavailable -- development continues with in-memory sessions "
            "and feedback only (nothing will survive a restart)"
        )
        return None
    logger.info("SQLite database ready for session/feedback persistence")
    return db


def publish_system_info(cfg: Settings) -> None:
    """Expose version, environment and build SHA (``unknown`` if the image has none)."""
    from fraudai.core.metrics import SYSTEM_INFO

    SYSTEM_INFO.info(
        {
            "version": "0.1.0",
            "environment": cfg.environment,
            "build_sha": served_build_sha(cfg) or "unknown",
        }
    )


def _warn_about_missing_operator_config(cfg: Settings) -> None:
    """Outside development a missing login or scrape credential is operator-visible."""
    if cfg.environment == "development":
        return
    if credentials_from_settings(cfg) is None:
        logger.warning(
            "Login is not configured (AUTH_USERNAME / AUTH_PASSWORD_HASH): "
            "POST /auth/token answers 503 until it is"
        )
    if load_scrape_token(cfg) is None:
        logger.warning(
            "No metrics scrape credential (METRICS_SCRAPE_TOKEN_FILE or METRICS_SCRAPE_TOKEN): "
            "only an admin JWT can read /admin/metrics"
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

    embedder: EmbeddingGenerator | None = None
    retriever: LegalRetriever | None = None
    try:
        embedder = EmbeddingGenerator(
            device="cpu"
        )  # CPU for API queries; GPU reserved for ingestion
        retriever = LegalRetriever(store=store, embedder=embedder)
    except Exception:
        logger.exception("RAG initialization failed -- search_boe will report it explicitly")
        embedder = None

    # The registry is built even when RAG is down: dropping it would send the graph
    # back to AGENT_TOOLS stubs and lose the explicit sandbox provider (F3.6).
    tool_registry = ToolRegistry(
        retriever=retriever or cast("LegalRetriever", _UnavailableRetriever()),
        sandbox=sandbox,
        tool_impls=tool_impls,
    )
    logger.info(
        "ToolRegistry initialized (retriever=%s, sandbox=%s, tool_impls=%s)",
        "LegalRetriever" if retriever is not None else "unavailable",
        "enabled" if sandbox is not None else "disabled",
        "enabled" if sandbox is not None else "explicit-errors",
    )

    # Inject ToolRegistry into graph BEFORE compilation
    from fraudai.agents.graph import set_tool_registry

    set_tool_registry(tool_registry)

    # LangGraph compiled graph
    graph = build_fraud_ai_graph()
    logger.info("LangGraph orchestration graph compiled")

    # Database (SQLite): a hard requirement outside development (see init_database).
    db = await init_database(settings)

    # Session manager (in-memory cache + optional DB persistence)
    session_manager = SessionManager(
        db=db, max_sessions_per_tenant=settings.max_sessions_per_tenant
    )
    if db is not None:
        await session_manager.load_from_db()

    # Feedback store (in-memory MVP -- list of deque for bounded memory, Bug #14)
    from collections import deque

    feedback_store: deque[dict[str, Any]] = deque(maxlen=10_000)

    publish_system_info(settings)
    configure_login_throttle(settings)
    _warn_about_missing_operator_config(settings)

    # Cached dependency health: the monitor checks in the background, probes read it.
    checks, required = build_checks(store=store, db=db, cfg=settings)
    health_monitor = HealthMonitor(
        checks,
        required,
        settings.readiness_interval_seconds,
        corpus_version_probe=store.get_corpus_version,
    )
    health_monitor.start()

    # Attach to app.state for route access
    app.state.graph = graph
    app.state.store = store
    app.state.session_manager = session_manager
    app.state.feedback_store = feedback_store
    app.state.tool_registry = tool_registry
    app.state.retriever = retriever
    app.state.embedder = embedder
    app.state.db = db
    app.state.health_monitor = health_monitor
    app.state.rate_limiter = TenantRateLimiter(settings.rate_limit_requests_per_minute)

    yield

    # --- Shutdown: stop the producers first, then drain writes, then close the database ---
    await health_monitor.stop()
    await session_manager.aclose()
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

    # --- Structured logging: JSON lines carrying the request correlation id ---
    configure_logging(level=settings.log_level, fmt=settings.log_format)

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
