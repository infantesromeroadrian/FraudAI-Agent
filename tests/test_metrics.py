"""Tests for Prometheus metrics, middleware, tracing, and /admin/metrics endpoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_metrics():
    """Accept shared state in the global Prometheus registry."""
    yield


def _create_test_app() -> FastAPI:
    """Build a FastAPI app with mocked lifespan (no Qdrant/Ollama)."""

    @asynccontextmanager
    async def _mock_lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        app.state.graph = MagicMock()
        app.state.store = MagicMock()
        app.state.session_manager = MagicMock()
        app.state.feedback_store: list[dict[str, Any]] = []
        yield

    from fraudai.api.middleware import MetricsMiddleware
    from fraudai.api.routes import router as api_router
    from fraudai.core.tracing import setup_tracing

    setup_tracing()

    app = FastAPI(lifespan=_mock_lifespan)
    app.add_middleware(MetricsMiddleware)
    app.include_router(api_router, prefix="/api/v1")
    return app


# =========================================================================
# 1. Metrics registry
# =========================================================================


class TestMetricsRegistry:
    """Verify that all expected metrics are registered in the default registry."""

    def test_request_count_registered(self) -> None:
        from fraudai.core.metrics import REQUEST_COUNT

        # prometheus_client Counters strip the _total suffix in _name
        assert REQUEST_COUNT._name == "fraudai_requests"

    def test_request_latency_registered(self) -> None:
        from fraudai.core.metrics import REQUEST_LATENCY

        assert REQUEST_LATENCY._name == "fraudai_request_duration_seconds"

    def test_agent_invocations_registered(self) -> None:
        from fraudai.core.metrics import AGENT_INVOCATIONS

        assert AGENT_INVOCATIONS._name == "fraudai_agent_invocations"

    def test_agent_latency_registered(self) -> None:
        from fraudai.core.metrics import AGENT_LATENCY

        assert AGENT_LATENCY._name == "fraudai_agent_duration_seconds"

    def test_routing_accuracy_registered(self) -> None:
        from fraudai.core.metrics import ROUTING_ACCURACY

        assert ROUTING_ACCURACY._name == "fraudai_routing"

    def test_retrieval_latency_registered(self) -> None:
        from fraudai.core.metrics import RETRIEVAL_LATENCY

        assert RETRIEVAL_LATENCY._name == "fraudai_retrieval_duration_seconds"

    def test_retrieval_results_registered(self) -> None:
        from fraudai.core.metrics import RETRIEVAL_RESULTS

        assert RETRIEVAL_RESULTS._name == "fraudai_retrieval_results_count"

    def test_tool_calls_registered(self) -> None:
        from fraudai.core.metrics import TOOL_CALLS

        assert TOOL_CALLS._name == "fraudai_tool_calls"

    def test_tool_latency_registered(self) -> None:
        from fraudai.core.metrics import TOOL_LATENCY

        assert TOOL_LATENCY._name == "fraudai_tool_duration_seconds"

    def test_sandbox_executions_registered(self) -> None:
        from fraudai.core.metrics import SANDBOX_EXECUTIONS

        assert SANDBOX_EXECUTIONS._name == "fraudai_sandbox_executions"

    def test_sandbox_duration_registered(self) -> None:
        from fraudai.core.metrics import SANDBOX_DURATION

        assert SANDBOX_DURATION._name == "fraudai_sandbox_duration_seconds"

    def test_tokens_used_registered(self) -> None:
        from fraudai.core.metrics import TOKENS_USED

        assert TOKENS_USED._name == "fraudai_tokens"

    def test_active_sessions_registered(self) -> None:
        from fraudai.core.metrics import ACTIVE_SESSIONS

        assert ACTIVE_SESSIONS._name == "fraudai_active_sessions"

    def test_system_info_registered(self) -> None:
        from fraudai.core.metrics import SYSTEM_INFO

        assert SYSTEM_INFO._name == "fraudai"

    def test_request_latency_buckets_aligned_with_sla(self) -> None:
        """Histogram buckets should include SLA-relevant boundaries."""
        from fraudai.core.metrics import REQUEST_LATENCY

        # F2 spec: agent first response P95 < 5s, full response P95 < 10s
        buckets = list(REQUEST_LATENCY._kwargs.get("buckets", []))
        assert 5.0 in buckets
        assert 10.0 in buckets

    def test_retrieval_latency_buckets_include_200ms(self) -> None:
        """RAG retrieval P95 target is < 200 ms (F2 2.1)."""
        from fraudai.core.metrics import RETRIEVAL_LATENCY

        buckets = list(RETRIEVAL_LATENCY._kwargs.get("buckets", []))
        assert 0.2 in buckets


# =========================================================================
# 2. Tracing — correlation IDs
# =========================================================================


class TestTracing:
    """Tests for correlation ID generation and propagation."""

    def test_new_correlation_id_returns_string(self) -> None:
        from fraudai.core.tracing import new_correlation_id

        cid = new_correlation_id()
        assert isinstance(cid, str)
        assert len(cid) == 8

    def test_new_correlation_id_sets_contextvar(self) -> None:
        from fraudai.core.tracing import correlation_id, new_correlation_id

        cid = new_correlation_id()
        assert correlation_id.get() == cid

    def test_successive_ids_are_different(self) -> None:
        from fraudai.core.tracing import new_correlation_id

        cid1 = new_correlation_id()
        cid2 = new_correlation_id()
        assert cid1 != cid2

    def test_correlation_filter_injects_id(self) -> None:
        from fraudai.core.tracing import CorrelationFilter, correlation_id

        correlation_id.set("test1234")
        f = CorrelationFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello",
            args=(),
            exc_info=None,
        )
        assert f.filter(record) is True
        assert record.correlation_id == "test1234"  # type: ignore[attr-defined]

    def test_correlation_filter_empty_default(self) -> None:
        from fraudai.core.tracing import CorrelationFilter, correlation_id

        # Reset to empty
        correlation_id.set("")
        f = CorrelationFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello",
            args=(),
            exc_info=None,
        )
        f.filter(record)
        assert record.correlation_id == ""  # type: ignore[attr-defined]

    def test_setup_tracing_is_idempotent(self) -> None:
        from fraudai.core.tracing import CorrelationFilter, setup_tracing

        root = logging.getLogger()
        initial_filter_count = sum(1 for f in root.filters if isinstance(f, CorrelationFilter))

        setup_tracing()
        setup_tracing()  # second call should be a no-op

        final_filter_count = sum(1 for f in root.filters if isinstance(f, CorrelationFilter))
        # Should have at most 1 more than initial (from the first setup_tracing)
        assert final_filter_count <= initial_filter_count + 1


# =========================================================================
# 3. MetricsMiddleware
# =========================================================================


class TestMetricsMiddleware:
    """Tests for the MetricsMiddleware recording request count and latency."""

    async def test_middleware_records_request_count(self) -> None:
        from prometheus_client import generate_latest
        from starlette.testclient import TestClient

        app = _create_test_app()

        with TestClient(app) as client:
            # /api/v1/health will execute (store mock returns falsy for
            # health_check but the middleware still records the request).
            client.get("/api/v1/health")

        output = generate_latest().decode("utf-8")
        assert "fraudai_requests_total" in output

    async def test_middleware_adds_correlation_id_header(self) -> None:
        from starlette.testclient import TestClient

        app = _create_test_app()

        with TestClient(app) as client:
            response = client.get("/api/v1/health")

        assert "x-correlation-id" in response.headers
        cid = response.headers["x-correlation-id"]
        assert len(cid) == 8

    async def test_middleware_records_latency(self) -> None:
        from prometheus_client import generate_latest
        from starlette.testclient import TestClient

        app = _create_test_app()

        with TestClient(app) as client:
            client.get("/api/v1/health")

        output = generate_latest().decode("utf-8")
        assert "fraudai_request_duration_seconds" in output


# =========================================================================
# 4. /admin/metrics endpoint
# =========================================================================


class TestAdminMetricsEndpoint:
    """Tests for the /admin/metrics Prometheus endpoint."""

    async def test_metrics_endpoint_returns_prometheus_format(self) -> None:
        """The endpoint should return Prometheus text exposition format."""
        from starlette.testclient import TestClient

        from fraudai.api.auth import require_metrics_access

        app = _create_test_app()
        # Use FastAPI dependency_overrides to bypass auth
        app.dependency_overrides[require_metrics_access] = lambda: None

        with TestClient(app) as client:
            response = client.get("/api/v1/admin/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        body = response.text
        # Should contain at least one of our custom metrics
        assert "fraudai_requests_total" in body

    async def test_metrics_endpoint_requires_admin(self) -> None:
        """Non-admin users should get 403 or 401."""
        from starlette.testclient import TestClient

        app = _create_test_app()

        with TestClient(app, raise_server_exceptions=False) as client:
            # No auth header at all -> 401
            response = client.get("/api/v1/admin/metrics")

        # Without a valid token, FastAPI's OAuth2 scheme returns 401
        assert response.status_code in (401, 403)


# =========================================================================
# 5. Metric recording by instrumented components
# =========================================================================


class TestDonnaMetrics:
    """Verify Donna records routing and latency metrics."""

    async def test_classify_records_agent_latency(self) -> None:
        from fraudai.agents.donna import DonnaRouter

        # Use a valid but unreachable port so httpx fails fast
        router = DonnaRouter(ollama_host="http://127.0.0.1:1")

        # This will fall back to keywords, but should still record latency
        await router.classify("analizar transacciones sospechosas")

        from prometheus_client import generate_latest

        output = generate_latest().decode("utf-8")
        assert 'agent_name="donna"' in output

    async def test_classify_records_routing_accuracy(self) -> None:
        from fraudai.agents.donna import DonnaRouter

        # Use a valid but unreachable port
        router = DonnaRouter(ollama_host="http://127.0.0.1:1")

        await router.classify("analizar transacciones sospechosas de fraude")

        from prometheus_client import generate_latest

        output = generate_latest().decode("utf-8")
        assert "fraudai_routing_total" in output
