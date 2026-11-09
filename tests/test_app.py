"""Tests for FastAPI application factory and lifespan — all external calls are mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI

# ---------------------------------------------------------------------------
# create_app
# ---------------------------------------------------------------------------


class TestCreateApp:
    """Tests for the create_app factory function."""

    def test_returns_fastapi_instance(self) -> None:
        from fraudai.api.app import create_app

        app = create_app()
        assert isinstance(app, FastAPI)

    def test_app_title(self) -> None:
        from fraudai.api.app import create_app

        app = create_app()
        assert app.title == "FraudAI Agent API"

    def test_app_version(self) -> None:
        from fraudai.api.app import create_app

        app = create_app()
        assert app.version == "0.1.0"

    def test_app_docs_url(self) -> None:
        from fraudai.api.app import create_app

        app = create_app()
        assert app.docs_url == "/docs"

    def test_app_redoc_url(self) -> None:
        from fraudai.api.app import create_app

        app = create_app()
        assert app.redoc_url == "/redoc"

    def test_api_routes_included(self) -> None:
        from fraudai.api.app import create_app

        app = create_app()
        # Read paths from the OpenAPI schema: app.routes holds router objects
        # without a ``path`` attribute in recent FastAPI releases.
        route_paths = list(app.openapi()["paths"])
        # At minimum, the /api/v1 prefix should be present via router
        assert any("/api/v1" in p for p in route_paths)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------


class TestLifespan:
    """Tests for the lifespan context manager."""

    @patch("fraudai.agents.graph.build_fraud_ai_graph")
    @patch("fraudai.rag.qdrant_store.QdrantStore.initialize", new_callable=AsyncMock)
    async def test_lifespan_sets_app_state(
        self,
        mock_initialize: AsyncMock,
        mock_build_graph: MagicMock,
    ) -> None:
        """Lifespan should populate app.state with graph, store, session_manager, feedback_store."""
        from fraudai.api.app import lifespan

        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph

        app = FastAPI()

        async with lifespan(app):
            assert app.state.graph is mock_graph
            assert app.state.store is not None
            assert hasattr(app.state, "session_manager")
            assert hasattr(app.state, "feedback_store")
            assert hasattr(app.state.feedback_store, "append")

    @patch("fraudai.agents.graph.build_fraud_ai_graph")
    @patch("fraudai.rag.qdrant_store.QdrantStore.initialize", new_callable=AsyncMock)
    async def test_lifespan_calls_store_initialize(
        self,
        mock_initialize: AsyncMock,
        mock_build_graph: MagicMock,
    ) -> None:
        """Lifespan should call QdrantStore.initialize on startup."""
        from fraudai.api.app import lifespan

        mock_build_graph.return_value = MagicMock()

        app = FastAPI()

        async with lifespan(app):
            pass

        mock_initialize.assert_awaited_once()

    @patch("fraudai.agents.graph.build_fraud_ai_graph")
    @patch("fraudai.rag.qdrant_store.QdrantStore.initialize", new_callable=AsyncMock)
    async def test_lifespan_calls_build_graph(
        self,
        mock_initialize: AsyncMock,
        mock_build_graph: MagicMock,
    ) -> None:
        """Lifespan should call build_fraud_ai_graph on startup."""
        from fraudai.api.app import lifespan

        mock_build_graph.return_value = MagicMock()

        app = FastAPI()

        async with lifespan(app):
            pass

        mock_build_graph.assert_called_once()

    @patch("fraudai.agents.graph.build_fraud_ai_graph")
    @patch(
        "fraudai.rag.qdrant_store.QdrantStore.initialize",
        new_callable=AsyncMock,
        side_effect=ConnectionError("Qdrant down"),
    )
    async def test_lifespan_continues_on_qdrant_failure(
        self,
        mock_initialize: AsyncMock,
        mock_build_graph: MagicMock,
    ) -> None:
        """Lifespan should not crash if QdrantStore.initialize raises (degraded mode)."""
        from fraudai.api.app import lifespan

        mock_build_graph.return_value = MagicMock()

        app = FastAPI()

        # Should not raise — the exception is caught internally
        async with lifespan(app):
            assert app.state.store is not None
            assert app.state.graph is not None

    @patch("fraudai.agents.graph.build_fraud_ai_graph")
    @patch("fraudai.rag.qdrant_store.QdrantStore.initialize", new_callable=AsyncMock)
    async def test_lifespan_creates_session_manager(
        self,
        mock_initialize: AsyncMock,
        mock_build_graph: MagicMock,
    ) -> None:
        """Lifespan should create a SessionManager instance."""
        from fraudai.api.app import lifespan
        from fraudai.api.session_manager import SessionManager

        mock_build_graph.return_value = MagicMock()

        app = FastAPI()

        async with lifespan(app):
            assert isinstance(app.state.session_manager, SessionManager)
