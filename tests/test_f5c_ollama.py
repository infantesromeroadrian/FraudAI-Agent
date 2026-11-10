"""F5c.6: the personal Ollama daemon as a first-class provider (D2).

The real ``langchain-openai`` client talks to a synthetic OpenAI-compatible server bound to
127.0.0.1 (a stand-in for the daemon), so the wire format, the literal key and the base URL
are checked without any real provider, key or spend. The real-daemon smoke is integration.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from prometheus_client import REGISTRY
from pydantic import ValidationError

from fraudai.agents import graph as graph_module
from fraudai.agents.claude_invoker import ClaudeAgentInvoker, _build_llm
from fraudai.core.config import Settings

if TYPE_CHECKING:
    from collections.abc import Iterator

MODEL = "gpt-oss:120b-cloud"

# The CI test job installs only the dev extra; these tests need the OpenAI-compatible client.
# They skip without it, unless FRAUDAI_REQUIRE_OPENAI_TESTS=1 (then a missing extra fails).
_HAS_OPENAI = importlib.util.find_spec("langchain_openai") is not None
requires_openai = pytest.mark.skipif(
    not _HAS_OPENAI and os.environ.get("FRAUDAI_REQUIRE_OPENAI_TESTS") != "1",
    reason="needs the openai extra (uv sync --extra openai)",
)
SYNTHETIC_FOREIGN_KEYS = {
    "anthropic_api_key": "synthetic-anthropic-key",
    "groq_api_key": "synthetic-groq-key",
    "openai_api_key": "synthetic-openai-key",
}


class _FakeDaemon:
    """Records requests and answers like an OpenAI-compatible chat endpoint."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.replies: list[dict[str, Any]] = []
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        assert self.server is not None
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def start(self) -> None:
        daemon = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                daemon.requests.append(
                    {
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "body": body,
                    }
                )
                reply = daemon.replies.pop(0) if daemon.replies else _text_reply("default")
                payload = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)


def _text_reply(text: str, prompt: int = 11, completion: int = 5) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


def _tool_reply(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments)},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    }


@pytest.fixture()
def daemon() -> Iterator[_FakeDaemon]:
    fake = _FakeDaemon()
    fake.start()
    try:
        yield fake
    finally:
        fake.stop()


def _sample(name: str, **labels: str) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


# ---------------------------------------------------------------------------
# _build_llm("ollama")
# ---------------------------------------------------------------------------


class TestBuildLlm:
    @requires_openai
    def test_ollama_uses_the_openai_compatible_client_against_the_v1_endpoint(self) -> None:
        llm = _build_llm("ollama", "", MODEL, 1024, 0.2, base_url="http://daemon.test:11434")

        assert type(llm).__name__ == "ChatOpenAI"
        assert llm.openai_api_base == "http://daemon.test:11434/v1"  # type: ignore[attr-defined]
        assert llm.model_name == MODEL  # type: ignore[attr-defined]

    @requires_openai
    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("http://h:11434", "http://h:11434/v1"),
            ("http://h:11434/", "http://h:11434/v1"),
            ("http://h:11434/v1", "http://h:11434/v1"),
            ("http://h:11434/v1/", "http://h:11434/v1"),
            ("  https://ollama.internal  ", "https://ollama.internal/v1"),
        ],
    )
    def test_base_url_is_normalised(self, given: str, expected: str) -> None:
        llm = _build_llm("ollama", "", MODEL, 1024, 0.2, base_url=given)

        assert llm.openai_api_base == expected  # type: ignore[attr-defined]

    @requires_openai
    def test_the_key_sent_is_the_non_secret_literal_never_a_provider_key(self) -> None:
        llm = _build_llm(
            "ollama", "synthetic-groq-key", MODEL, 1024, 0.2, base_url="http://daemon.test:11434"
        )

        assert llm.openai_api_key.get_secret_value() == "ollama"  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "base_url", [None, "", "   ", "daemon:11434", "ftp://h", "file:///x", "http://"]
    )
    def test_ollama_needs_an_http_base_url(self, base_url: str | None) -> None:
        with pytest.raises(ValueError, match="base_url"):
            _build_llm("ollama", "", MODEL, 1024, 0.2, base_url=base_url)

    @pytest.mark.parametrize("provider", ["openai", "anthropic", "groq"])
    def test_other_providers_refuse_a_custom_base_url_so_keys_cannot_be_redirected(
        self, provider: str
    ) -> None:
        with pytest.raises(ValueError, match="base_url"):
            _build_llm(provider, "synthetic-key", MODEL, 1024, 0.2, base_url="http://evil.test")

    @requires_openai
    def test_openai_keeps_using_its_own_key(self) -> None:
        llm = _build_llm("openai", "synthetic-openai-key", "gpt-4o", 1024, 0.2)

        assert llm.openai_api_key.get_secret_value() == "synthetic-openai-key"  # type: ignore[attr-defined]

    def test_unknown_provider_lists_ollama_among_the_supported(self) -> None:
        with pytest.raises(ValueError, match="ollama"):
            _build_llm("nope", "", MODEL, 1024, 0.2)


# ---------------------------------------------------------------------------
# Real SDK against the synthetic daemon
# ---------------------------------------------------------------------------


def _state(text: str = "analiza esta operación") -> dict[str, Any]:
    return {"messages": [HumanMessage(content=text)]}


@requires_openai
async def test_a_plain_answer_goes_to_v1_chat_completions_with_the_literal_key(
    daemon: _FakeDaemon,
) -> None:
    daemon.replies = [_text_reply("sin anomalías")]
    invoker = ClaudeAgentInvoker(
        api_key="", model=MODEL, provider="ollama", base_url=daemon.base_url
    )

    result = await invoker.invoke("harvey", "Eres Harvey.", [], _state())  # type: ignore[arg-type]

    assert result["message"].content == "sin anomalías"
    request = daemon.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["authorization"] == "Bearer ollama"
    assert request["body"]["model"] == MODEL
    roles = [m["role"] for m in request["body"]["messages"]]
    assert roles == ["system", "user"]


@requires_openai
async def test_tool_calling_round_trip_through_the_sdk(daemon: _FakeDaemon) -> None:
    daemon.replies = [
        _tool_reply("risk_scoring", {"data_path": "tx.csv"}),
        _text_reply("riesgo bajo"),
    ]
    calls: list[str] = []

    @tool("risk_scoring")
    async def risk_scoring(data_path: str) -> dict[str, Any]:
        """Score the risk of a data file."""
        calls.append(data_path)
        return {"score": 0.1}

    invoker = ClaudeAgentInvoker(
        api_key="", model=MODEL, provider="ollama", base_url=daemon.base_url
    )

    result = await invoker.invoke("harvey", "Eres Harvey.", [risk_scoring], _state())  # type: ignore[arg-type]

    assert calls == ["tx.csv"]
    assert result["message"].content == "riesgo bajo"
    first, second = (r["body"] for r in daemon.requests)
    assert [t["function"]["name"] for t in first["tools"]] == ["risk_scoring"]
    assert "tools" not in second or not second["tools"], "the follow-up call has no tools"
    assert any(m["role"] == "tool" for m in second["messages"])


@requires_openai
async def test_token_usage_from_the_daemon_reaches_the_metrics_without_a_cost(
    daemon: _FakeDaemon,
) -> None:
    daemon.replies = [_text_reply("ok", prompt=123, completion=45)]
    invoker = ClaudeAgentInvoker(
        api_key="", model=MODEL, provider="ollama", base_url=daemon.base_url
    )
    in_before = _sample("fraudai_tokens_total", agent_name="jessica", direction="input")
    out_before = _sample("fraudai_tokens_total", agent_name="jessica", direction="output")
    cost_before = _sample("fraudai_token_cost_usd_total", agent_name="jessica")

    cfg = Settings(_env_file=None, llm_provider="ollama", ollama_host=daemon.base_url)
    with patch("fraudai.agents.claude_invoker.settings", cfg):
        await invoker.invoke("jessica", "prompt", [], _state())  # type: ignore[arg-type]

    assert (
        _sample("fraudai_tokens_total", agent_name="jessica", direction="input") - in_before == 123
    )
    assert (
        _sample("fraudai_tokens_total", agent_name="jessica", direction="output") - out_before == 45
    )
    assert _sample("fraudai_token_cost_usd_total", agent_name="jessica") == cost_before


# ---------------------------------------------------------------------------
# Wiring from settings (graph._get_claude_invoker)
# ---------------------------------------------------------------------------


@pytest.fixture()
def fresh_invoker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph_module, "_claude_invoker", None)


def _wired(cfg: Settings) -> MagicMock:
    created = MagicMock()
    with (
        patch.object(graph_module, "settings", cfg),
        patch.object(graph_module, "ClaudeAgentInvoker", created),
    ):
        asyncio.run(graph_module._get_claude_invoker())
    return created


def _kwargs(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "llm_provider": "ollama",
        "llm_model": MODEL,
        "ollama_host": "http://host.docker.internal:11434",
        **SYNTHETIC_FOREIGN_KEYS,
    }
    values.update(overrides)
    return values


def test_ollama_is_wired_to_the_configured_host_and_model_and_no_foreign_key(
    fresh_invoker: None,
) -> None:
    created = _wired(Settings(_env_file=None, **_kwargs()))

    kwargs = created.call_args.kwargs
    assert kwargs["provider"] == "ollama"
    assert kwargs["model"] == MODEL
    assert kwargs["base_url"] == "http://host.docker.internal:11434"
    assert kwargs["api_key"] == ""
    assert not any("synthetic" in str(v) for v in kwargs.values())


@pytest.mark.parametrize(
    ("provider", "key_field", "expected_key"),
    [
        ("groq", "groq_api_key", "synthetic-groq-key"),
        ("anthropic", "anthropic_api_key", "synthetic-anthropic-key"),
        ("openai", "openai_api_key", "synthetic-openai-key"),
    ],
)
def test_other_providers_keep_their_own_key_and_never_get_a_base_url(
    provider: str, key_field: str, expected_key: str, fresh_invoker: None
) -> None:
    created = _wired(Settings(_env_file=None, **_kwargs(llm_provider=provider)))

    kwargs = created.call_args.kwargs
    assert kwargs["api_key"] == expected_key
    assert kwargs.get("base_url") is None
    assert kwargs["provider"] == provider


@requires_openai
async def test_a_real_call_through_the_wiring_sends_only_the_literal_key(
    daemon: _FakeDaemon, fresh_invoker: None
) -> None:
    daemon.replies = [_text_reply("hola")]
    cfg = Settings(_env_file=None, **_kwargs(ollama_host=daemon.base_url))

    with patch.object(graph_module, "settings", cfg):
        result = await graph_module.invoke_claude_agent("louis", "prompt", [], _state())  # type: ignore[arg-type]

    assert result["message"].content == "hola"
    sent = daemon.requests[0]
    assert sent["headers"]["authorization"] == "Bearer ollama"
    assert all("synthetic" not in json.dumps(v) for v in sent.values())
    graph_module._claude_invoker = None


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestSettings:
    def test_provider_is_a_closed_set(self) -> None:
        with pytest.raises(ValidationError):
            Settings(_env_file=None, llm_provider="olama")

    @pytest.mark.parametrize("provider", ["anthropic", "groq", "openai", "ollama"])
    def test_supported_providers(self, provider: str) -> None:
        assert Settings(_env_file=None, llm_provider=provider).llm_provider == provider

    def test_ollama_needs_an_http_host_and_a_model(self) -> None:
        with pytest.raises(ValidationError, match="OLLAMA_HOST"):
            Settings(_env_file=None, llm_provider="ollama", ollama_host="daemon:11434")
        with pytest.raises(ValidationError, match="LLM_MODEL"):
            Settings(_env_file=None, llm_provider="ollama", llm_model="")

    def test_openai_key_has_its_own_setting_and_defaults_empty(self) -> None:
        assert Settings(_env_file=None).openai_api_key == ""

    def test_the_default_provider_is_unchanged(self) -> None:
        assert Settings(_env_file=None).llm_provider == "groq"


# ---------------------------------------------------------------------------
# Donna must not send the user's message to a provider that was not selected
# ---------------------------------------------------------------------------

AMBIGUOUS_MESSAGE = "zzzz qqqq xxxx"  # no keyword matches: Donna must ask an LLM


@pytest.fixture()
def fresh_donna(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph_module, "_donna_router", None)


async def _classify_ambiguous(cfg: Settings) -> tuple[AsyncMock, AsyncMock]:
    from fraudai.agents.donna import DonnaRouter

    groq = AsyncMock(return_value={"agent": "harvey", "language": "es", "confidence": 0.9})
    ollama = AsyncMock(return_value={"agent": "louis", "language": "es", "confidence": 0.9})
    with (
        patch.object(graph_module, "settings", cfg),
        patch.object(DonnaRouter, "_call_groq", groq),
        patch.object(DonnaRouter, "_call_ollama", ollama),
    ):
        await graph_module.classify_intent_local(AMBIGUOUS_MESSAGE)
    return groq, ollama


async def test_ollama_provider_never_gives_donna_the_groq_key(fresh_donna: None) -> None:
    cfg = Settings(_env_file=None, **_kwargs(llm_provider="ollama"))

    groq, ollama = await _classify_ambiguous(cfg)

    assert groq.await_count == 0, "the message must not go to a provider that is not selected"
    assert ollama.await_count == 1
    assert graph_module._donna_router is not None
    assert graph_module._donna_router._groq_api_key is None


@pytest.mark.parametrize("provider", ["groq", "anthropic", "openai"])
async def test_other_providers_keep_donnas_previous_groq_behaviour(
    provider: str, fresh_donna: None
) -> None:
    cfg = Settings(_env_file=None, **_kwargs(llm_provider=provider))

    groq, ollama = await _classify_ambiguous(cfg)

    assert groq.await_count == 1
    assert ollama.await_count == 0


async def test_donna_without_any_groq_key_still_uses_ollama(fresh_donna: None) -> None:
    cfg = Settings(_env_file=None, **_kwargs(llm_provider="groq", groq_api_key=""))

    groq, ollama = await _classify_ambiguous(cfg)

    assert (groq.await_count, ollama.await_count) == (0, 1)
