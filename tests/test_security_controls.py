"""Security controls: SSRF guard, code-injection safety, path checks, JWT secret."""

from __future__ import annotations

import ast
import socket
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from fraudai.core.config import Settings
from fraudai.tools.implementations import ToolImplementations

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def networked_sandbox() -> MagicMock:
    sandbox = MagicMock()
    sandbox.execute_python = AsyncMock()
    return sandbox


@pytest.fixture()
def impls(networked_sandbox: MagicMock) -> ToolImplementations:
    return ToolImplementations(sandbox=MagicMock(), sandbox_networked=networked_sandbox)


def _assigned_value(code: str, name: str) -> object:
    """Return the literal assigned to a top-level name in generated code."""
    for node in ast.parse(code).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not assigned in generated code")


# ---------------------------------------------------------------------------
# SSRF: only public http(s) targets reach the networked sandbox
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://127.0.0.1/predict",
        "http://2130706433/predict",  # 127.0.0.1 written as an integer
        "http://0x7f.1/predict",  # 127.0.0.1 in hex shorthand
        "http://[::ffff:127.0.0.1]/predict",  # IPv4-mapped IPv6
        "http://[::1]/predict",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://10.0.0.5/predict",
        "http://100.64.0.1/predict",  # carrier-grade NAT
        "http://[fd00::1]/predict",  # unique local IPv6
        "ftp://93.184.216.34/predict",
        "file:///etc/passwd",
        "http://user:secret@93.184.216.34/predict",
        "not a url",
    ],
)
def test_non_public_or_non_http_targets_are_blocked(endpoint: str) -> None:
    with pytest.raises(PermissionError):
        ToolImplementations(sandbox=MagicMock())._validate_target_endpoint(endpoint)


def test_hostname_resolving_to_a_private_address_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ToolImplementations,
        "_resolve_host",
        staticmethod(lambda _host: ["93.184.216.34", "10.0.0.7"]),
    )

    with pytest.raises(PermissionError):
        ToolImplementations(sandbox=MagicMock())._validate_target_endpoint(
            "https://looks-public.example/predict"
        )


def test_unresolvable_hostname_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_host: str) -> list[str]:
        raise socket.gaierror("no such host")

    monkeypatch.setattr(ToolImplementations, "_resolve_host", staticmethod(fail))

    with pytest.raises(PermissionError):
        ToolImplementations(sandbox=MagicMock())._validate_target_endpoint(
            "https://missing.example/predict"
        )


def test_public_https_target_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ToolImplementations, "_resolve_host", staticmethod(lambda _host: ["93.184.216.34"])
    )

    ToolImplementations(sandbox=MagicMock())._validate_target_endpoint(
        "https://model.example/predict"
    )


async def test_blocked_target_never_reaches_the_sandbox(
    impls: ToolImplementations, networked_sandbox: MagicMock
) -> None:
    with pytest.raises(PermissionError):
        await impls.prompt_injection_suite(target_endpoint="http://169.254.169.254/")

    networked_sandbox.execute_python.assert_not_called()


# ---------------------------------------------------------------------------
# Code injection: tool arguments are data, never code
# ---------------------------------------------------------------------------

_HOSTILE_ENDPOINT = 'https://model.example/p"\nimport os; os.system("id")  # '


def test_endpoint_is_embedded_as_a_string_literal_in_evasion_code() -> None:
    code = ToolImplementations._build_adversarial_evasion_code(_HOSTILE_ENDPOINT, "sign_noise")

    assert _assigned_value(code, "TARGET_ENDPOINT") == _HOSTILE_ENDPOINT
    assert _assigned_value(code, "ATTACK_TYPE") == "sign_noise"


def test_endpoint_is_embedded_as_a_string_literal_in_injection_code() -> None:
    code = ToolImplementations._build_prompt_injection_code(_HOSTILE_ENDPOINT, None)

    assert _assigned_value(code, "TARGET_ENDPOINT") == _HOSTILE_ENDPOINT


@pytest.mark.parametrize("attack_type", ["fgsm", 'x"; import os; "', ""])
async def test_unknown_perturbation_is_rejected_before_the_sandbox(
    impls: ToolImplementations, networked_sandbox: MagicMock, attack_type: str
) -> None:
    with pytest.raises(ValueError, match="Unknown perturbation"):
        await impls.adversarial_evasion(
            target_endpoint="https://model.example/predict", attack_type=attack_type
        )

    networked_sandbox.execute_python.assert_not_called()


# ---------------------------------------------------------------------------
# Path checks for sandbox input files
# ---------------------------------------------------------------------------


@pytest.fixture()
def uploads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base = tmp_path / "uploads"
    base.mkdir()
    monkeypatch.setattr(ToolImplementations, "ALLOWED_INPUT_DIRS", [str(base)])
    return base


def test_file_inside_the_allowed_directory_is_read(uploads: Path) -> None:
    (uploads / "tx.csv").write_text("id\n1\n")

    data = ToolImplementations(sandbox=MagicMock())._read_input_file(str(uploads / "tx.csv"))

    assert data == b"id\n1\n"


def test_sibling_directory_sharing_the_prefix_is_rejected(uploads: Path) -> None:
    sibling = uploads.parent / "uploads_evil"
    sibling.mkdir()
    (sibling / "secret.csv").write_text("x")

    with pytest.raises(PermissionError):
        ToolImplementations(sandbox=MagicMock())._read_input_file(str(sibling / "secret.csv"))


def test_symlink_escaping_the_allowed_directory_is_rejected(uploads: Path) -> None:
    outside = uploads.parent / "outside.csv"
    outside.write_text("x")
    (uploads / "link.csv").symlink_to(outside)

    with pytest.raises(PermissionError):
        ToolImplementations(sandbox=MagicMock())._read_input_file(str(uploads / "link.csv"))


# ---------------------------------------------------------------------------
# JWT secret in production
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "secret",
    ["fraudai-dev-secret-change-in-production", "change-this-in-production", "short"],
)
def test_production_refuses_placeholder_or_short_jwt_secret(secret: str) -> None:
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        Settings(_env_file=None, environment="production", jwt_secret=secret)


def test_production_accepts_a_long_random_jwt_secret() -> None:
    settings = Settings(_env_file=None, environment="production", jwt_secret="k" * 48)

    assert settings.environment == "production"


def test_development_keeps_the_default_secret_for_local_use() -> None:
    assert Settings(_env_file=None, environment="development").jwt_secret
