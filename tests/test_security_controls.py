"""Security controls: SSRF guard, code-injection safety, path checks, JWT secret."""

from __future__ import annotations

import ast
import asyncio
import socket
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from fraudai.core import session_context
from fraudai.core.config import Settings
from fraudai.core.session_context import active_session_id, bind_session
from fraudai.tools.egress import (
    EgressDeniedError,
    EgressPolicy,
    EgressTarget,
    alias_for,
    is_public_address,
)
from fraudai.tools.implementations import SandboxDisabled, ToolImplementations
from fraudai.tools.sandbox import SandboxResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from fraudai.tools.egress import IPAddress


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


SESSION_A = "11111111-1111-4111-8111-111111111111"
SESSION_B = "22222222-2222-4222-8222-222222222222"


@pytest.fixture()
def uploads_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "fraudai_uploads"
    root.mkdir()
    monkeypatch.setattr(session_context, "uploads_root", lambda: root)
    return root


def _session_file(root: Path, session_id: str, name: str, content: str = "id\n1\n") -> Path:
    directory = root / session_id
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_text(content)
    return path


def _read(data_path: str, *, session_id: str | None) -> bytes:
    impls = ToolImplementations(sandbox=MagicMock())
    if session_id is None:
        return impls._read_input_file(data_path)
    with bind_session(session_id):
        return impls._read_input_file(data_path)


def test_file_of_the_active_session_is_read_by_absolute_path(uploads_root: Path) -> None:
    path = _session_file(uploads_root, SESSION_A, "tx.csv")

    assert _read(str(path), session_id=SESSION_A) == b"id\n1\n"


def test_relative_name_resolves_inside_the_session_directory(uploads_root: Path) -> None:
    _session_file(uploads_root, SESSION_A, "tx.csv")

    assert _read("tx.csv", session_id=SESSION_A) == b"id\n1\n"


def test_upload_of_another_session_is_denied(uploads_root: Path) -> None:
    other = _session_file(uploads_root, SESSION_B, "secret.csv", "victim")

    with pytest.raises(PermissionError):
        _read(str(other), session_id=SESSION_A)
    with pytest.raises(PermissionError):
        _read(f"../{SESSION_B}/secret.csv", session_id=SESSION_A)


def test_no_active_session_denies_everything(uploads_root: Path) -> None:
    path = _session_file(uploads_root, SESSION_A, "tx.csv")

    with pytest.raises(PermissionError, match="no active session"):
        _read(str(path), session_id=None)


def test_session_binding_does_not_leak_after_the_scope(uploads_root: Path) -> None:
    path = _session_file(uploads_root, SESSION_A, "tx.csv")
    _read(str(path), session_id=SESSION_A)

    with pytest.raises(PermissionError):
        _read(str(path), session_id=None)


@pytest.mark.parametrize(
    "bad_session", ["../etc", "not-a-uuid", "", "ABCDEF01-0000-4000-8000-000000000000"]
)
def test_non_canonical_active_session_id_is_denied(uploads_root: Path, bad_session: str) -> None:
    # bind_session refuses these; force the variable to prove the reader does too.
    token = session_context._active_session_id.set(bad_session)
    try:
        with pytest.raises(PermissionError):
            ToolImplementations(sandbox=MagicMock())._read_input_file("tx.csv")
    finally:
        session_context._active_session_id.reset(token)


@pytest.mark.parametrize(
    "data_path",
    ["../outside.csv", "../../etc/passwd", "/etc/passwd", "sub/../../outside.csv", "."],
)
def test_traversal_and_foreign_absolute_paths_are_denied(
    uploads_root: Path, data_path: str
) -> None:
    (uploads_root / "outside.csv").write_text("x")
    _session_file(uploads_root, SESSION_A, "tx.csv")

    with pytest.raises(PermissionError):
        _read(data_path, session_id=SESSION_A)


def test_rest_of_the_temp_directory_is_denied(uploads_root: Path, tmp_path: Path) -> None:
    sandbox_input = tmp_path / "other.csv"
    sandbox_input.write_text("x")
    _session_file(uploads_root, SESSION_A, "tx.csv")

    with pytest.raises(PermissionError):
        _read(str(sandbox_input), session_id=SESSION_A)


def test_sibling_directory_sharing_the_prefix_is_denied(uploads_root: Path) -> None:
    sibling = uploads_root / f"{SESSION_A}_evil"
    sibling.mkdir()
    (sibling / "secret.csv").write_text("x")

    with pytest.raises(PermissionError):
        _read(str(sibling / "secret.csv"), session_id=SESSION_A)


def test_symlink_escaping_the_session_directory_is_denied(uploads_root: Path) -> None:
    outside = uploads_root / "outside.csv"
    outside.write_text("x")
    victim = _session_file(uploads_root, SESSION_B, "secret.csv", "victim")
    directory = uploads_root / SESSION_A
    directory.mkdir()
    (directory / "link.csv").symlink_to(outside)
    (directory / "peer.csv").symlink_to(victim)

    with pytest.raises(PermissionError):
        _read("link.csv", session_id=SESSION_A)
    with pytest.raises(PermissionError):
        _read("peer.csv", session_id=SESSION_A)


def test_missing_file_inside_the_session_is_not_found(uploads_root: Path) -> None:
    (uploads_root / SESSION_A).mkdir()

    with pytest.raises(FileNotFoundError):
        _read("nope.csv", session_id=SESSION_A)


def test_session_directory_itself_is_not_readable(uploads_root: Path) -> None:
    (uploads_root / SESSION_A).mkdir()

    with pytest.raises(PermissionError):
        _read(str(uploads_root / SESSION_A), session_id=SESSION_A)


# ---------------------------------------------------------------------------
# Session binding lifecycle
# ---------------------------------------------------------------------------


def test_binding_is_restored_after_an_error() -> None:
    with pytest.raises(RuntimeError), bind_session(SESSION_A):
        assert active_session_id() == SESSION_A
        raise RuntimeError("boom")

    assert active_session_id() is None


def test_nested_binding_restores_the_outer_session() -> None:
    with bind_session(SESSION_A):
        with bind_session(SESSION_B):
            assert active_session_id() == SESSION_B
        assert active_session_id() == SESSION_A


@pytest.mark.parametrize("bad", ["", "../x", "not-a-uuid", "ABCDEF01-0000-4000-8000-000000000000"])
def test_binding_refuses_non_canonical_session_ids(bad: str) -> None:
    with pytest.raises(ValueError, match="invalid session id"), bind_session(bad):
        pass

    assert active_session_id() is None


async def test_concurrent_requests_do_not_see_each_others_session() -> None:
    both_bound = asyncio.Event()
    waiting = 0
    seen: dict[str, list[str | None]] = {}

    async def request(session_id: str) -> None:
        nonlocal waiting
        with bind_session(session_id):
            seen[session_id] = [active_session_id()]
            waiting += 1
            if waiting == 2:
                both_bound.set()
            await both_bound.wait()  # both sessions are bound at the same time here
            await asyncio.sleep(0)
            seen[session_id].append(active_session_id())

    await asyncio.gather(request(SESSION_A), request(SESSION_B))

    assert seen == {SESSION_A: [SESSION_A, SESSION_A], SESSION_B: [SESSION_B, SESSION_B]}
    assert active_session_id() is None


async def test_cancelled_request_does_not_leak_its_binding() -> None:
    entered = asyncio.Event()

    async def request() -> None:
        with bind_session(SESSION_A):
            entered.set()
            await asyncio.sleep(60)

    task = asyncio.create_task(request())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert active_session_id() is None


# ---------------------------------------------------------------------------
# Environment is a closed set (M5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "environment",
    ["prod", "PRODUCTION", "Production", "production ", "test", "", "dev"],
)
def test_unknown_environment_fails_closed(environment: str) -> None:
    with pytest.raises(ValidationError, match="environment"):
        Settings(_env_file=None, environment=environment)


@pytest.mark.parametrize("environment", ["development", "staging", "production"])
def test_known_environments_are_accepted(environment: str) -> None:
    assert (
        Settings(_env_file=None, environment=environment, jwt_secret="k" * 48).environment
        == environment
    )


def test_environment_from_the_process_env_is_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "prod")

    with pytest.raises(ValidationError, match="environment"):
        Settings(_env_file=None)


# ---------------------------------------------------------------------------
# JWT secret outside development (staging and production)
# ---------------------------------------------------------------------------

SECRET_REQUIRED_ENVIRONMENTS = ["staging", "production"]


@pytest.mark.parametrize("environment", SECRET_REQUIRED_ENVIRONMENTS)
@pytest.mark.parametrize(
    "secret",
    ["fraudai-dev-secret-change-in-production", "change-this-in-production", "short"],
)
def test_non_development_refuses_placeholder_or_short_jwt_secret(
    environment: str, secret: str
) -> None:
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        Settings(_env_file=None, environment=environment, jwt_secret=secret)


@pytest.mark.parametrize("environment", SECRET_REQUIRED_ENVIRONMENTS)
def test_non_development_refuses_the_built_in_default_secret(
    environment: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("JWT_SECRET", raising=False)

    with pytest.raises(ValidationError, match="JWT_SECRET"):
        Settings(_env_file=None, environment=environment)


@pytest.mark.parametrize("environment", SECRET_REQUIRED_ENVIRONMENTS)
def test_non_development_error_names_the_environment(environment: str) -> None:
    with pytest.raises(ValidationError, match=f"ENVIRONMENT={environment}"):
        Settings(_env_file=None, environment=environment, jwt_secret="short")


@pytest.mark.parametrize("environment", SECRET_REQUIRED_ENVIRONMENTS)
def test_non_development_accepts_a_long_random_jwt_secret(environment: str) -> None:
    settings = Settings(_env_file=None, environment=environment, jwt_secret="k" * 48)

    assert settings.environment == environment
    assert settings.jwt_secret == "k" * 48


def test_development_keeps_the_default_secret_for_local_use() -> None:
    assert Settings(_env_file=None, environment="development").jwt_secret


# ---------------------------------------------------------------------------
# ADR-007: validated target is pinned; nothing but that address may be dialled
# ---------------------------------------------------------------------------


def _policy(
    *addresses: str, allowed: Callable[[IPAddress], bool] = is_public_address
) -> EgressPolicy:
    return EgressPolicy(resolver=lambda _host: list(addresses), address_allowed=allowed)


def test_validate_pins_one_resolved_address_and_keeps_hostname_as_alias_only() -> None:
    target = _policy("2606:4700::1", "93.184.216.34").validate("https://Model.Example:8443/p")

    assert target == EgressTarget(
        scheme="https", hostname="model.example", address="93.184.216.34", port=8443
    )


def test_default_ports_follow_the_scheme() -> None:
    assert _policy("93.184.216.34").validate("http://m.example/x").port == 80
    assert _policy("93.184.216.34").validate("https://m.example/x").port == 443


def test_resolver_is_consulted_once_per_validation() -> None:
    calls: list[str] = []

    def resolver(host: str) -> list[str]:
        calls.append(host)
        return ["93.184.216.34"]

    EgressPolicy(resolver=resolver).validate("https://m.example/")

    assert calls == ["m.example"]  # no second lookup that rebinding could poison


@pytest.mark.parametrize(
    "address",
    [
        "169.254.169.254",  # IMDS
        "169.254.1.1",  # link-local
        "172.17.0.1",  # default Docker gateway
        "172.19.0.1",  # user-defined Docker bridge gateway
        "192.168.1.1",
        "100.100.100.200",  # CGNAT metadata service
        "::ffff:10.0.0.1",
        "fe80::1",
        "64:ff9b::a00:1",  # NAT64 embedding 10.0.0.1
        "2002:a00:1::1",  # 6to4 embedding 10.0.0.1
        "224.0.0.1",
    ],
)
def test_any_non_public_resolved_address_blocks_the_target(address: str) -> None:
    with pytest.raises(EgressDeniedError):
        _policy("93.184.216.34", address).validate("https://m.example/")


def test_invalid_port_and_garbage_addresses_are_blocked() -> None:
    with pytest.raises(EgressDeniedError):
        _policy("93.184.216.34").validate("https://m.example:99999/")
    with pytest.raises(EgressDeniedError):
        _policy("not-an-ip").validate("https://m.example/")


def test_policy_can_be_injected_for_a_synthetic_target_without_relaxing_the_default() -> None:
    synthetic = _policy("172.17.0.9", allowed=lambda ip: str(ip) == "172.17.0.9")

    assert synthetic.validate("http://target.test:8080/").address == "172.17.0.9"
    with pytest.raises(EgressDeniedError):
        _policy("172.17.0.9").validate("http://target.test:8080/")  # default policy still blocks


def test_dns_alias_rejects_ip_literals_and_odd_names() -> None:
    literal = EgressTarget(
        scheme="http", hostname="93.184.216.34", address="93.184.216.34", port=80
    )
    odd = EgressTarget(scheme="http", hostname="bad_name.example", address="93.184.216.34", port=80)

    with pytest.raises(EgressDeniedError):
        alias_for(literal)
    with pytest.raises(EgressDeniedError):
        alias_for(odd)
    assert (
        alias_for(EgressTarget(scheme="http", hostname="ok.example", address="1.1.1.1", port=80))
        == "ok.example"
    )


async def test_validated_target_is_handed_to_the_sandbox(
    impls: ToolImplementations,
    networked_sandbox: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ToolImplementations, "_resolve_host", staticmethod(lambda _host: ["93.184.216.34"])
    )
    networked_sandbox.execute_python.return_value = SandboxResult(
        exit_code=0,
        stdout="",
        stderr="",
        output_files={"result.json": b"{}"},
        duration_seconds=0.1,
    )

    await impls.prompt_injection_suite(target_endpoint="https://model.example:8443/p")

    egress = networked_sandbox.execute_python.call_args.kwargs["egress"]
    assert (egress.hostname, egress.address, egress.port) == (
        "model.example",
        "93.184.216.34",
        8443,
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda: ToolImplementations._build_adversarial_evasion_code(
            "https://m.example/p", "sign_noise"
        ),
        lambda: ToolImplementations._build_prompt_injection_code("https://m.example/p", None),
    ],
)
def test_generated_http_client_refuses_redirects_and_environment_proxies(
    build: Callable[[], str],
) -> None:
    code = build()

    assert "urlopen(" not in code  # urllib's default opener follows 3xx
    assert "HTTPRedirectHandler" in code
    assert "return None" in code  # redirect_request refuses every hop
    assert "ProxyHandler({})" in code


# ---------------------------------------------------------------------------
# D4 mode B: explicit error, never a stub, when the sandbox is off
# ---------------------------------------------------------------------------

_DISABLED = SandboxDisabled(code="sandbox_disabled", reason="turned off in this deployment")


@pytest.mark.parametrize(
    "call",
    [
        lambda t: t.adversarial_evasion(target_endpoint="https://model.example/p"),
        lambda t: t.prompt_injection_suite(target_endpoint="https://model.example/p"),
        lambda t: t.analyze_transactions(data_path="x.csv"),
        lambda t: t.detect_patterns(data_path="x.csv", algorithms=["isolation_forest"]),
        lambda t: t.risk_scoring(data_path="x.csv"),
        lambda t: t.graph_analysis(data_path="x.csv"),
        lambda t: t.data_quality_check(data_path="x.csv"),
    ],
)
async def test_disabled_sandbox_returns_an_explicit_error_for_every_sandbox_tool(
    call: Callable[[ToolImplementations], Awaitable[dict[str, Any]]],
) -> None:
    tools = ToolImplementations(sandbox=None, disabled=_DISABLED)

    result = await call(tools)

    assert result["status"] == "disabled"
    assert result["error_code"] == "sandbox_disabled"
    assert "turned off" in result["error"]


async def test_disabled_sandbox_does_not_block_tools_that_need_no_sandbox() -> None:
    tools = ToolImplementations(sandbox=None, disabled=_DISABLED)

    result = await tools.compliance_checklist(regulation="PSD2")

    assert result.get("status") != "disabled"
