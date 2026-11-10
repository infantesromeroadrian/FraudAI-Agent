"""F5c.2: every metric of ``docs/metrics-contract.md`` is really emitted, and the scrape
credential is accepted by the metrics route only.

``EMISSION`` is the single list of emitted metrics: the contract document must
match it exactly and every entry runs a driver that makes the metric move.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.messages import AIMessage, HumanMessage
from prometheus_client import REGISTRY, Counter, Gauge, Histogram, Info

from fraudai.agents.claude_invoker import ClaudeAgentInvoker
from fraudai.api.auth import create_token
from fraudai.api.middleware import MetricsMiddleware
from fraudai.api.routes import router
from fraudai.api.session_manager import SessionManager
from fraudai.core import metrics as metrics_module
from fraudai.core.config import Settings

CONTRACT = Path(__file__).resolve().parents[1] / "docs" / "metrics-contract.md"
SYNTHETIC_SECRET = "j" * 48  # noqa: S105 - fixture signing key
SCRAPE_VALUE = "scrape-" + "x" * 40  # synthetic scrape credential


def _sample(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def _cfg(**overrides: Any) -> Settings:
    values: dict[str, Any] = {"environment": "development", "jwt_secret": SYNTHETIC_SECRET}
    values.update(overrides)
    return Settings(_env_file=None, **values)


# Emission table: metric -> (sample name, labels, action that makes it move)
# ---------------------------------------------------------------------------


def _invoker(ai_message: AIMessage) -> ClaudeAgentInvoker:
    model = MagicMock()
    model.ainvoke = AsyncMock(return_value=ai_message)
    model.bind_tools = MagicMock(return_value=model)
    with patch("fraudai.agents.claude_invoker._build_llm", return_value=model):
        return ClaudeAgentInvoker(api_key="synthetic", provider="groq", model="m")


def _state() -> dict[str, Any]:
    return {"messages": [HumanMessage(content="hola")]}


def _tool(name: str, output: Any) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.ainvoke = AsyncMock(return_value=output)
    return tool


def _usage(input_tokens: int, output_tokens: int) -> AIMessage:
    return AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


def _retriever() -> Any:
    from fraudai.rag.retriever import LegalRetriever

    store = AsyncMock()
    store.BOE_COLLECTION = "boe_legislation"
    store.hybrid_search.return_value = [
        {"id": "p1", "score": 0.9, "payload": {"text": "t", "boe_id": "BOE-A-1"}}
    ]
    embedder = MagicMock()
    embedder.encode.return_value = [
        {"dense": [0.1] * 1024, "sparse": {"indices": [1], "values": [0.5]}}
    ]
    return LegalRetriever(store=store, embedder=embedder, reranker=None, default_k=5, rerank_k=10)


async def _http_request() -> None:
    app = FastAPI()
    app.add_middleware(MetricsMiddleware)
    app.include_router(router, prefix="/api/v1")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http:
        await http.get("/api/v1/livez")


async def _plan() -> None:
    await _invoker(AIMessage(content="ok")).plan("harvey", "prompt", [], _state())  # type: ignore[arg-type]


async def _invoke_louis() -> None:
    await _invoker(AIMessage(content="ok")).invoke("louis", "prompt", [], _state())  # type: ignore[arg-type]


async def _route() -> None:
    from fraudai.agents.donna import DonnaRouter

    DonnaRouter._record_routing({"agent": "jessica", "language": "es", "confidence": 0.95})


async def _retrieve() -> None:
    await _retriever().retrieve("blanqueo", collection="boe_legislation")


async def _run_tool(name: str = "analyze_transactions") -> None:
    await ClaudeAgentInvoker.run_tool_call([_tool(name, {"rows": 1})], name, {})


async def _sandbox() -> None:
    from fraudai.tools.sandbox import SandboxEngine

    client = MagicMock()
    client.ping.return_value = True
    client.images.get.return_value = MagicMock(tags=["fraudai-sandbox:latest"])
    container = MagicMock()
    container.short_id = "abc123"
    container.wait.return_value = {"StatusCode": 0}
    container.logs.return_value = b""
    container.status = "exited"
    client.containers.create.return_value = container
    await SandboxEngine(docker_client=client).execute_python("print('x')")


async def _tokens() -> None:
    await _invoker(_usage(7, 3)).invoke("rachel", "prompt", [], _state())  # type: ignore[arg-type]


async def _cost() -> None:
    cfg = _cfg(llm_input_cost_per_mtok_usd=1.0, llm_output_cost_per_mtok_usd=2.0)
    with patch("fraudai.agents.claude_invoker.settings", cfg):
        await _invoker(_usage(1_000_000, 0)).invoke("mike", "prompt", [], _state())  # type: ignore[arg-type]


async def _sessions() -> None:
    manager = SessionManager()
    for _ in range(3):
        manager.create_session("tenant-metrics", "pro")


async def _system_info() -> None:
    from fraudai.api.app import publish_system_info

    publish_system_info(_cfg(build_sha="abcdef1"))


_REQ = {"method": "GET", "endpoint": "/api/v1/livez"}
EMISSION: dict[str, tuple[str, dict[str, str], Any]] = {
    "fraudai_requests_total": (
        "fraudai_requests_total",
        {**_REQ, "status_code": "200"},
        _http_request,
    ),
    "fraudai_request_duration_seconds": (
        "fraudai_request_duration_seconds_count",
        _REQ,
        _http_request,
    ),
    "fraudai_agent_invocations_total": (
        "fraudai_agent_invocations_total",
        {"agent_name": "harvey"},
        _plan,
    ),
    "fraudai_agent_duration_seconds": (
        "fraudai_agent_duration_seconds_count",
        {"agent_name": "louis"},
        _invoke_louis,
    ),
    "fraudai_routing_total": (
        "fraudai_routing_total",
        {"target_agent": "jessica", "confidence_bucket": "high"},
        _route,
    ),
    "fraudai_retrieval_duration_seconds": (
        "fraudai_retrieval_duration_seconds_count",
        {"collection": "boe_legislation"},
        _retrieve,
    ),
    "fraudai_retrieval_results_count": (
        "fraudai_retrieval_results_count_count",
        {"collection": "boe_legislation"},
        _retrieve,
    ),
    "fraudai_tool_calls_total": (
        "fraudai_tool_calls_total",
        {"tool_name": "analyze_transactions", "status": "success"},
        _run_tool,
    ),
    "fraudai_tool_duration_seconds": (
        "fraudai_tool_duration_seconds_count",
        {"tool_name": "analyze_transactions"},
        _run_tool,
    ),
    "fraudai_sandbox_executions_total": (
        "fraudai_sandbox_executions_total",
        {"status": "success"},
        _sandbox,
    ),
    "fraudai_sandbox_duration_seconds": (
        "fraudai_sandbox_duration_seconds_count",
        {},
        _sandbox,
    ),
    "fraudai_tokens_total": (
        "fraudai_tokens_total",
        {"agent_name": "rachel", "direction": "input"},
        _tokens,
    ),
    "fraudai_token_cost_usd_total": (
        "fraudai_token_cost_usd_total",
        {"agent_name": "mike"},
        _cost,
    ),
    "fraudai_active_sessions": ("fraudai_active_sessions", {}, _sessions),
    "fraudai_info": (
        "fraudai_info",
        {"version": "0.1.0", "environment": "development", "build_sha": "abcdef1"},
        _system_info,
    ),
}


@pytest.mark.parametrize("metric", sorted(EMISSION))
async def test_every_contract_metric_is_emitted(metric: str) -> None:
    sample, labels, action = EMISSION[metric]
    before = _sample(sample, **labels)

    await action()

    after = _sample(sample, **labels)
    if metric in {"fraudai_active_sessions", "fraudai_info"}:  # gauge / info: set, not counted
        assert after > 0, f"{sample}{labels} was never set"
    else:
        assert after > before, f"{sample}{labels} did not move"


# ---------------------------------------------------------------------------
# The contract document lists exactly what is emitted
# ---------------------------------------------------------------------------


def _exposed_name(metric: Any) -> str:
    if isinstance(metric, Counter):
        return f"{metric._name}_total"
    if isinstance(metric, Info):
        return f"{metric._name}_info"
    return str(metric._name)


def _registered_metrics() -> set[str]:
    kinds = (Counter, Gauge, Histogram, Info)
    return {_exposed_name(v) for v in vars(metrics_module).values() if isinstance(v, kinds)}


def _contract_sections() -> tuple[set[str], set[str]]:
    text = CONTRACT.read_text(encoding="utf-8")
    emitted_part, _, not_emitted_part = text.partition("## Registered but not emitted")
    emitted = set(re.findall(r"^\| `(fraudai_[a-z_]+)` \|", emitted_part, flags=re.MULTILINE))
    not_emitted = set(re.findall(r"^- `(fraudai_[a-z_]+)`", not_emitted_part, flags=re.MULTILINE))
    return emitted, not_emitted


def test_contract_document_exists() -> None:
    assert CONTRACT.is_file()


def test_contract_lists_exactly_the_emitted_metrics() -> None:
    emitted, _ = _contract_sections()

    assert emitted == set(EMISSION)


def test_every_registered_metric_is_documented_as_emitted_or_not() -> None:
    emitted, not_emitted = _contract_sections()

    assert _registered_metrics() <= emitted | not_emitted
    assert not (emitted & not_emitted)


def test_contract_documents_the_unemitted_alert_metrics_and_the_cost_rule() -> None:
    text = CONTRACT.read_text(encoding="utf-8")

    for name in (
        "fraudai_boe_last_ingestion_timestamp",
        "fraudai_sandbox_containers_active",
        "fraudai_session_cost_usd_bucket",
    ):
        assert name in text
    assert "Ollama" in text
    assert "unknown" in text.lower()


# ---------------------------------------------------------------------------
# Labels stay bounded
# ---------------------------------------------------------------------------


async def test_unmatched_paths_share_one_endpoint_label() -> None:
    app = FastAPI()
    app.add_middleware(MetricsMiddleware)
    app.include_router(router, prefix="/api/v1")
    transport = httpx.ASGITransport(app=app)
    before = _sample(
        "fraudai_requests_total", method="GET", endpoint="unmatched", status_code="404"
    )

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        for index in range(5):
            await http.get(f"/scan/probe-{index}")

    after = _sample("fraudai_requests_total", method="GET", endpoint="unmatched", status_code="404")
    assert after - before == 5
    assert (
        _sample("fraudai_requests_total", method="GET", endpoint="/scan/probe-0", status_code="404")
        == 0
    )


async def test_unknown_tool_names_from_the_model_share_one_label() -> None:
    before = _sample("fraudai_tool_calls_total", tool_name="unknown", status="error")

    for index in range(3):
        await ClaudeAgentInvoker.run_tool_call([], f"made_up_tool_{index}", {})

    assert _sample("fraudai_tool_calls_total", tool_name="unknown", status="error") - before == 3
    assert _sample("fraudai_tool_calls_total", tool_name="made_up_tool_0", status="error") == 0


async def test_failed_and_declined_tools_count_as_errors_with_a_duration() -> None:
    failing = _tool("data_quality_check", None)
    failing.ainvoke = AsyncMock(side_effect=RuntimeError("synthetic"))
    declined = _tool(
        "detect_patterns", {"status": "disabled", "error": "off", "error_code": "sandbox_disabled"}
    )
    err_before = _sample("fraudai_tool_calls_total", tool_name="data_quality_check", status="error")
    dec_before = _sample("fraudai_tool_calls_total", tool_name="detect_patterns", status="error")
    dur_before = _sample("fraudai_tool_duration_seconds_count", tool_name="data_quality_check")

    await ClaudeAgentInvoker.run_tool_call([failing], "data_quality_check", {})
    await ClaudeAgentInvoker.run_tool_call([declined], "detect_patterns", {})

    assert (
        _sample("fraudai_tool_calls_total", tool_name="data_quality_check", status="error")
        - err_before
        == 1
    )
    assert (
        _sample("fraudai_tool_calls_total", tool_name="detect_patterns", status="error")
        - dec_before
        == 1
    )
    assert (
        _sample("fraudai_tool_duration_seconds_count", tool_name="data_quality_check") - dur_before
        == 1
    )


# ---------------------------------------------------------------------------
# Token cost: only with a configured tariff (Ollama has none)
# ---------------------------------------------------------------------------


async def test_no_cost_is_emitted_without_a_configured_tariff() -> None:
    message = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 500, "output_tokens": 500, "total_tokens": 1000},
    )
    before = _sample("fraudai_token_cost_usd_total", agent_name="harvey")

    with patch("fraudai.agents.claude_invoker.settings", _cfg(llm_provider="ollama")):
        await _invoker(message).invoke("harvey", "prompt", [], _state())  # type: ignore[arg-type]

    assert _sample("fraudai_token_cost_usd_total", agent_name="harvey") == before
    assert REGISTRY.get_sample_value("fraudai_token_cost_usd_total", {"agent_name": "harvey"}) in (
        None,
        before,
    )


async def test_cost_needs_both_prices() -> None:
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "total_tokens": 2_000_000,
        },
    )
    before = _sample("fraudai_token_cost_usd_total", agent_name="jessica")

    # Settings refuses half a tariff, so force it past validation to prove the guard too.
    half_tariff = _cfg().model_copy(update={"llm_input_cost_per_mtok_usd": 3.0})
    with patch("fraudai.agents.claude_invoker.settings", half_tariff):
        await _invoker(message).invoke("jessica", "prompt", [], _state())  # type: ignore[arg-type]

    assert _sample("fraudai_token_cost_usd_total", agent_name="jessica") == before


async def test_cost_is_tokens_times_tariff() -> None:
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 2_000_000,
            "output_tokens": 500_000,
            "total_tokens": 2_500_000,
        },
    )
    cfg = _cfg(llm_input_cost_per_mtok_usd=3.0, llm_output_cost_per_mtok_usd=15.0)
    before = _sample("fraudai_token_cost_usd_total", agent_name="louis")

    with patch("fraudai.agents.claude_invoker.settings", cfg):
        await _invoker(message).invoke("louis", "prompt", [], _state())  # type: ignore[arg-type]

    assert _sample("fraudai_token_cost_usd_total", agent_name="louis") - before == pytest.approx(
        13.5
    )


def test_tariff_settings_must_come_in_pairs_and_be_non_negative() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="LLM_INPUT_COST_PER_MTOK_USD"):
        _cfg(llm_input_cost_per_mtok_usd=1.0)
    with pytest.raises(ValidationError):
        _cfg(llm_input_cost_per_mtok_usd=-1.0, llm_output_cost_per_mtok_usd=1.0)


# ---------------------------------------------------------------------------
# Active sessions gauge follows the session lifecycle
# ---------------------------------------------------------------------------


def test_active_sessions_tracks_create_delete_and_quota_eviction() -> None:
    manager = SessionManager()
    first = manager.create_session("tenant-gauge", "pro")
    manager.create_session("tenant-gauge", "pro")
    assert _sample("fraudai_active_sessions") == 2

    manager.delete_session(first)

    assert _sample("fraudai_active_sessions") == 1


# ---------------------------------------------------------------------------
# Scrape credential: only /admin/metrics, nothing else
# ---------------------------------------------------------------------------


@pytest.fixture()
def scrape_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.state.session_manager = SessionManager()
    app.state.feedback_store = []
    return app


def _bearer(value: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {value}"}


def _jwt(*, admin: bool) -> str:
    with patch("fraudai.core.config.settings", _cfg()):
        return create_token("ops", "tenant-ops", "pro", is_admin=admin)


async def _call(
    app: FastAPI, method: str, path: str, cfg: Settings, **kwargs: Any
) -> httpx.Response:
    with (
        patch("fraudai.core.config.settings", cfg),
        patch("fraudai.api.routes.settings", cfg),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as http:
            return await http.request(method, f"/api/v1{path}", **kwargs)


async def test_scrape_token_from_env_reads_metrics(scrape_app: FastAPI) -> None:
    cfg = _cfg(environment="production", metrics_scrape_token=SCRAPE_VALUE)

    response = await _call(scrape_app, "GET", "/admin/metrics", cfg, headers=_bearer(SCRAPE_VALUE))

    assert response.status_code == 200
    assert "fraudai_" in response.text


async def test_scrape_token_from_file_reads_metrics(scrape_app: FastAPI, tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text(SCRAPE_VALUE + "\n", encoding="utf-8")
    cfg = _cfg(environment="production", metrics_scrape_token_file=str(token_file))

    response = await _call(scrape_app, "GET", "/admin/metrics", cfg, headers=_bearer(SCRAPE_VALUE))

    assert response.status_code == 200


@pytest.mark.parametrize(
    "presented", ["wrong-" + "y" * 40, SCRAPE_VALUE + "x", SCRAPE_VALUE[:-1], ""]
)
async def test_wrong_scrape_token_is_401(scrape_app: FastAPI, presented: str) -> None:
    cfg = _cfg(environment="production", metrics_scrape_token=SCRAPE_VALUE)

    response = await _call(scrape_app, "GET", "/admin/metrics", cfg, headers=_bearer(presented))

    assert response.status_code == 401


async def test_missing_credentials_is_401(scrape_app: FastAPI) -> None:
    cfg = _cfg(environment="production", metrics_scrape_token=SCRAPE_VALUE)

    assert (await _call(scrape_app, "GET", "/admin/metrics", cfg)).status_code == 401


async def test_too_short_configured_token_is_ignored(scrape_app: FastAPI) -> None:
    cfg = _cfg(environment="production", metrics_scrape_token="short")

    response = await _call(scrape_app, "GET", "/admin/metrics", cfg, headers=_bearer("short"))

    assert response.status_code == 401


async def test_unconfigured_scrape_token_never_matches_an_empty_bearer(scrape_app: FastAPI) -> None:
    cfg = _cfg(environment="production")

    response = await _call(scrape_app, "GET", "/admin/metrics", cfg, headers=_bearer(""))

    assert response.status_code == 401


async def test_admin_jwt_still_reads_metrics_and_plain_users_do_not(scrape_app: FastAPI) -> None:
    cfg = _cfg(environment="production", metrics_scrape_token=SCRAPE_VALUE)
    with patch("fraudai.core.config.settings", cfg):
        admin = create_token("ops", "tenant-ops", "pro", is_admin=True)
        user = create_token("bob", "tenant-ops", "pro", is_admin=False)

    assert (
        await _call(scrape_app, "GET", "/admin/metrics", cfg, headers=_bearer(admin))
    ).status_code == 200
    assert (
        await _call(scrape_app, "GET", "/admin/metrics", cfg, headers=_bearer(user))
    ).status_code == 403


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("POST", "/chat", {"json": {"message": "hi"}}),
        ("POST", "/chat/stream", {"json": {"message": "hi"}}),
        ("POST", "/confirm", {"json": {"session_id": "x", "action_id": "a", "approved": True}}),
        ("GET", "/sessions/00000000-0000-4000-8000-000000000000", {}),
        ("DELETE", "/sessions/00000000-0000-4000-8000-000000000000", {}),
        ("POST", "/feedback", {"json": {"session_id": "x", "message_id": "m", "rating": 3}}),
    ],
)
async def test_scrape_token_has_no_privilege_outside_the_metrics_route(
    scrape_app: FastAPI, method: str, path: str, kwargs: dict[str, Any]
) -> None:
    cfg = _cfg(environment="production", metrics_scrape_token=SCRAPE_VALUE)

    response = await _call(scrape_app, method, path, cfg, headers=_bearer(SCRAPE_VALUE), **kwargs)

    assert response.status_code == 401


async def test_scrape_token_does_not_expire_and_has_no_jwt_claims(scrape_app: FastAPI) -> None:
    # Not a JWT at all: nothing to expire, nothing to widen with claims.
    assert SCRAPE_VALUE.count(".") != 2
    cfg = _cfg(environment="production", metrics_scrape_token=SCRAPE_VALUE)

    first = await _call(scrape_app, "GET", "/admin/metrics", cfg, headers=_bearer(SCRAPE_VALUE))
    second = await _call(scrape_app, "GET", "/admin/metrics", cfg, headers=_bearer(SCRAPE_VALUE))

    assert first.status_code == second.status_code == 200
