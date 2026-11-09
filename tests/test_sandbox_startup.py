"""Sandbox startup policy: fail closed, never fall back to stub data (F3, ADR-007).

All Docker access is faked: these tests pin the *decision logic* (which mode
starts, which refuses, what the tools answer). Real isolation is checked
against a Docker daemon elsewhere.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from docker.errors import DockerException
from fastapi.testclient import TestClient

from fraudai.api.app import SandboxStartupError, create_app, init_sandbox_tools
from fraudai.core.config import Settings

_JWT = "x" * 40


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"jwt_secret": _JWT}
    return Settings(**{**base, **overrides})  # type: ignore[arg-type]


def _engine(healthy: bool) -> MagicMock:
    engine = MagicMock()
    engine.health_check = AsyncMock(return_value=healthy)
    return engine


async def test_disabled_mode_never_touches_docker_and_answers_explicitly() -> None:
    factory = MagicMock()

    engine, tools = await init_sandbox_tools(
        _settings(sandbox_mode="disabled", environment="production"), engine_factory=factory
    )

    factory.assert_not_called()
    assert engine is None
    result = await tools.adversarial_evasion(target_endpoint="https://model.example/p")
    assert result["status"] == "disabled"
    assert result["error_code"] == "sandbox_disabled"
    assert "SANDBOX_MODE=disabled" in result["error"]


async def test_healthy_docker_mode_enables_real_tools() -> None:
    healthy = _engine(True)

    engine, tools = await init_sandbox_tools(_settings(), engine_factory=lambda: healthy)

    assert engine is healthy
    assert tools._disabled is None
    assert tools._sandbox is healthy


@pytest.mark.parametrize("environment", ["staging", "production"])
async def test_unhealthy_sandbox_refuses_to_start_outside_development(environment: str) -> None:
    with pytest.raises(SandboxStartupError, match=environment):
        await init_sandbox_tools(
            _settings(environment=environment), engine_factory=lambda: _engine(False)
        )


@pytest.mark.parametrize("environment", ["staging", "production"])
async def test_docker_connection_error_also_refuses_to_start(environment: str) -> None:
    def broken() -> MagicMock:
        raise DockerException("daemon not running")

    with pytest.raises(SandboxStartupError):
        await init_sandbox_tools(_settings(environment=environment), engine_factory=broken)


async def test_development_without_docker_starts_with_explicit_errors_not_stubs() -> None:
    engine, tools = await init_sandbox_tools(
        _settings(environment="development"), engine_factory=lambda: _engine(False)
    )

    assert engine is None
    result = await tools.prompt_injection_suite(target_endpoint="https://model.example/p")
    assert result["status"] == "disabled"
    assert result["error_code"] == "sandbox_unavailable"


def test_default_settings_use_docker_mode_and_bounded_limits() -> None:
    cfg = _settings()

    assert cfg.sandbox_mode == "docker"
    assert (cfg.sandbox_mem_limit, cfg.sandbox_pids_limit, cfg.sandbox_max_concurrent) == (
        "2g",
        128,
        2,
    )


def test_unknown_sandbox_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="sandbox_mode"):
        _settings(sandbox_mode="off")


def test_startup_error_is_not_swallowed_by_the_degraded_mode_handlers() -> None:
    """A required but broken sandbox must abort the app lifespan, not log and continue."""
    with (
        patch(
            "fraudai.api.app.init_sandbox_tools", AsyncMock(side_effect=SandboxStartupError("x"))
        ),
        pytest.raises(SandboxStartupError),
        TestClient(create_app()),
    ):
        pass
