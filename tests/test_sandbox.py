"""Tests for the sandbox execution engine.

All Docker SDK interactions are mocked -- no real Docker daemon is required.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest

from fraudai.tools.egress import EgressDeniedError, EgressTarget
from fraudai.tools.sandbox import (
    OutputCaps,
    SandboxEngine,
    SandboxError,
    SandboxResult,
    read_output_tree,
    validate_input_name,
)

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_docker_client() -> MagicMock:
    """Return a mock ``docker.DockerClient``."""
    client = MagicMock()
    client.ping.return_value = True
    # Images
    client.images.get.return_value = MagicMock(tags=["fraudai-sandbox:latest"])
    return client


@pytest.fixture()
def mock_container() -> MagicMock:
    """Return a mock Docker container with sane defaults."""
    container = MagicMock()
    container.short_id = "abc123"
    container.start.return_value = None
    container.wait.return_value = {"StatusCode": 0}
    container.logs.return_value = b""
    container.stop.return_value = None
    container.remove.return_value = None
    return container


@pytest.fixture()
def engine(mock_docker_client: MagicMock) -> SandboxEngine:
    """Return a ``SandboxEngine`` with a mocked Docker client."""
    return SandboxEngine(docker_client=mock_docker_client)


# ---------------------------------------------------------------------------
# SandboxResult model
# ---------------------------------------------------------------------------


class TestSandboxResult:
    def test_defaults(self) -> None:
        result = SandboxResult(
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=1.0,
        )
        assert result.exit_code == 0
        assert result.stdout == "ok"
        assert result.stderr == ""
        assert result.output_files == {}
        assert result.timed_out is False

    def test_with_output_files(self) -> None:
        result = SandboxResult(
            exit_code=0,
            stdout="",
            stderr="",
            output_files={"report.json": b'{"ok": true}'},
            duration_seconds=2.5,
        )
        assert "report.json" in result.output_files
        assert result.duration_seconds == 2.5


# ---------------------------------------------------------------------------
# execute_python
# ---------------------------------------------------------------------------


class TestExecutePython:
    async def test_basic_execution(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        mock_docker_client.containers.create.return_value = mock_container
        mock_container.wait.return_value = {"StatusCode": 0}

        def _logs_side_effect(stdout: bool = True, stderr: bool = True) -> bytes:
            if stdout and not stderr:
                return b"hello world\n"
            return b""

        mock_container.logs.side_effect = _logs_side_effect

        result = await engine.execute_python("print('hello world')")

        assert result.exit_code == 0
        assert "hello world" in result.stdout
        assert result.timed_out is False

        # Container must have been created and started
        mock_docker_client.containers.create.assert_called_once()
        mock_container.start.assert_called_once()
        # Container must ALWAYS be removed
        mock_container.remove.assert_called_once_with(force=True)

    async def test_with_input_files(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        mock_docker_client.containers.create.return_value = mock_container
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b""

        input_files = {
            "data.csv": b"col1,col2\n1,2\n3,4\n",
            "config.json": b'{"threshold": 0.5}',
        }

        result = await engine.execute_python(
            "import pandas as pd; df = pd.read_csv('/workspace/data.csv')",
            input_files=input_files,
        )

        assert result.exit_code == 0
        mock_container.remove.assert_called_once_with(force=True)

    async def test_nonzero_exit_code(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        mock_docker_client.containers.create.return_value = mock_container
        mock_container.wait.return_value = {"StatusCode": 1}

        def _logs_side_effect(stdout: bool = True, stderr: bool = True) -> bytes:
            if not stdout and stderr:
                return b"NameError: name 'foo' is not defined\n"
            return b""

        mock_container.logs.side_effect = _logs_side_effect

        result = await engine.execute_python("foo()")

        assert result.exit_code == 1
        assert "NameError" in result.stderr
        assert result.timed_out is False
        mock_container.remove.assert_called_once_with(force=True)


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestTimeout:
    async def test_container_stopped_on_timeout(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        mock_docker_client.containers.create.return_value = mock_container
        # Simulate timeout by raising an exception from container.wait
        mock_container.wait.side_effect = Exception("read timeout")
        mock_container.logs.return_value = b""

        result = await engine.execute_python(
            "import time; time.sleep(9999)",
            timeout=5,
        )

        assert result.timed_out is True
        assert result.exit_code == -1
        # Container should be stopped and then removed
        mock_container.stop.assert_called_once_with(timeout=5)
        mock_container.remove.assert_called_once_with(force=True)

    async def test_custom_timeout_passed(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """When caller provides a custom timeout it overrides the engine default."""
        mock_docker_client.containers.create.return_value = mock_container
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b""

        await engine.execute_python("pass", timeout=30)

        # wait() should have been called with the custom timeout
        mock_container.wait.assert_called_once_with(timeout=30)


# ---------------------------------------------------------------------------
# Security options
# ---------------------------------------------------------------------------


class TestSecurityOptions:
    async def test_no_network_by_default(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        mock_docker_client.containers.create.return_value = mock_container

        await engine.execute_python("pass")

        create_kwargs = mock_docker_client.containers.create.call_args.kwargs
        assert create_kwargs["network_mode"] == "none"
        assert "network" not in create_kwargs
        mock_docker_client.networks.create.assert_not_called()

    async def test_cap_drop_all(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        mock_docker_client.containers.create.return_value = mock_container
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b""

        await engine.execute_python("pass")

        create_kwargs = mock_docker_client.containers.create.call_args
        assert create_kwargs.kwargs["cap_drop"] == ["ALL"]

    async def test_no_new_privileges(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        mock_docker_client.containers.create.return_value = mock_container
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b""

        await engine.execute_python("pass")

        create_kwargs = mock_docker_client.containers.create.call_args
        assert "no-new-privileges" in create_kwargs.kwargs["security_opt"]

    async def test_memory_limit(
        self,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        mock_docker_client.containers.create.return_value = mock_container
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b""

        engine = SandboxEngine(docker_client=mock_docker_client, mem_limit="4g")
        await engine.execute_python("pass")

        create_kwargs = mock_docker_client.containers.create.call_args
        assert create_kwargs.kwargs["mem_limit"] == "4g"
        assert create_kwargs.kwargs["memswap_limit"] == "4g"

    async def test_cpu_quota_uses_nano_cpus(
        self,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        mock_docker_client.containers.create.return_value = mock_container
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b""

        engine = SandboxEngine(docker_client=mock_docker_client, cpu_count=1.5)
        await engine.execute_python("pass")

        # ``cpu_count`` is Windows-only in the Engine API; the CFS quota is nano_cpus.
        create_kwargs = mock_docker_client.containers.create.call_args
        assert create_kwargs.kwargs["nano_cpus"] == 1_500_000_000
        assert "cpu_count" not in create_kwargs.kwargs

    async def test_pids_limit(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        mock_docker_client.containers.create.return_value = mock_container
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b""

        await engine.execute_python("pass")

        create_kwargs = mock_docker_client.containers.create.call_args
        assert create_kwargs.kwargs["pids_limit"] == 128


# ---------------------------------------------------------------------------
# Cleanup -- container must ALWAYS be removed
# ---------------------------------------------------------------------------


class TestCleanup:
    async def test_container_removed_on_success(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        mock_docker_client.containers.create.return_value = mock_container
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b""

        await engine.execute_python("pass")

        mock_container.remove.assert_called_once_with(force=True)

    async def test_container_removed_on_failure(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        mock_docker_client.containers.create.return_value = mock_container
        mock_container.wait.return_value = {"StatusCode": 1}
        mock_container.logs.return_value = b""

        await engine.execute_python("exit(1)")

        mock_container.remove.assert_called_once_with(force=True)

    async def test_container_removed_on_timeout(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        mock_docker_client.containers.create.return_value = mock_container
        mock_container.wait.side_effect = Exception("timeout")
        mock_container.logs.return_value = b""

        await engine.execute_python("pass", timeout=1)

        mock_container.remove.assert_called_once_with(force=True)

    async def test_container_removed_even_if_logs_fail(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        """If collecting logs raises, the container is still removed."""
        mock_docker_client.containers.create.return_value = mock_container
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.side_effect = Exception("connection reset")

        with pytest.raises(Exception, match="connection reset"):
            await engine.execute_python("pass")

        mock_container.remove.assert_called_once_with(force=True)


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    async def test_healthy(self, engine: SandboxEngine, mock_docker_client: MagicMock) -> None:
        mock_docker_client.ping.return_value = True
        mock_docker_client.images.get.return_value = MagicMock()

        assert await engine.health_check() is True

    async def test_docker_not_running(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
    ) -> None:
        import docker.errors

        mock_docker_client.ping.side_effect = docker.errors.APIError("connection refused")

        assert await engine.health_check() is False

    async def test_image_not_found(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
    ) -> None:
        import docker.errors

        mock_docker_client.ping.return_value = True
        mock_docker_client.images.get.side_effect = docker.errors.ImageNotFound("not found")

        assert await engine.health_check() is False

    async def test_ping_returns_false(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
    ) -> None:
        mock_docker_client.ping.return_value = False

        assert await engine.health_check() is False


# ---------------------------------------------------------------------------
# execute_script
# ---------------------------------------------------------------------------


class TestExecuteScript:
    async def test_basic_script(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        mock_docker_client.containers.create.return_value = mock_container
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b""

        result = await engine.execute_script("/scripts/analyze.py", args=["--mode", "fast"])

        assert result.exit_code == 0
        create_kwargs = mock_docker_client.containers.create.call_args
        assert create_kwargs.kwargs["command"] == [
            "python",
            "/scripts/analyze.py",
            "--mode",
            "fast",
        ]
        mock_container.remove.assert_called_once_with(force=True)

    async def test_script_no_args(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        mock_docker_client.containers.create.return_value = mock_container
        mock_container.wait.return_value = {"StatusCode": 0}
        mock_container.logs.return_value = b""

        await engine.execute_script("/scripts/check.py")

        create_kwargs = mock_docker_client.containers.create.call_args
        assert create_kwargs.kwargs["command"] == ["python", "/scripts/check.py"]


# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------


class TestEngineDefaults:
    def test_default_values(self, mock_docker_client: MagicMock) -> None:
        engine = SandboxEngine(docker_client=mock_docker_client)
        assert engine._image == "fraudai-sandbox:latest"
        assert engine._timeout == 300
        assert engine._mem_limit == "2g"
        assert engine._cpu_count == 2.0
        assert engine._pids_limit == 128
        assert engine._max_concurrent == 2

    def test_custom_values(self, mock_docker_client: MagicMock) -> None:
        engine = SandboxEngine(
            docker_client=mock_docker_client,
            image="custom:v1",
            timeout=120,
            mem_limit="2g",
            cpu_count=1,
            pids_limit=64,
            max_concurrent=3,
        )
        assert engine._image == "custom:v1"
        assert engine._timeout == 120
        assert engine._mem_limit == "2g"
        assert engine._cpu_count == 1
        assert engine._pids_limit == 64
        assert engine._max_concurrent == 3


# ---------------------------------------------------------------------------
# ADR-007 hardening -- container options (Docker SDK mocked: these assert what
# is *requested*; the effect is proven separately against a real daemon)
# ---------------------------------------------------------------------------


class TestIsolationOptions:
    async def test_readonly_rootfs_noexec_tmpfs_and_readonly_inputs(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        mock_docker_client.containers.create.return_value = mock_container

        await engine.execute_python("pass", input_files={"data.csv": b"a\n"})

        kwargs = mock_docker_client.containers.create.call_args.kwargs
        assert kwargs["read_only"] is True
        assert kwargs["tmpfs"] == {"/tmp": "rw,noexec,nosuid,nodev,size=64m"}
        mounts = {m["Target"]: m for m in kwargs["mounts"]}
        assert mounts["/workspace"]["ReadOnly"] is True
        assert mounts["/workspace/output"]["ReadOnly"] is False
        assert kwargs["working_dir"] == "/workspace"
        assert "volumes" not in kwargs

    async def test_resource_limits_and_ulimits(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        mock_docker_client.containers.create.return_value = mock_container

        await engine.execute_python("pass")

        kwargs = mock_docker_client.containers.create.call_args.kwargs
        assert kwargs["mem_limit"] == kwargs["memswap_limit"] == "2g"
        assert kwargs["pids_limit"] == 128
        names = {u["Name"]: (u["Soft"], u["Hard"]) for u in kwargs["ulimits"]}
        assert names["fsize"] == (4 * 1024 * 1024, 4 * 1024 * 1024)
        assert names["core"] == (0, 0)
        assert kwargs["init"] is True
        assert kwargs["labels"]["fraudai.sandbox"] == "1"

    async def test_rejects_unsafe_input_names_before_any_container(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
    ) -> None:
        for name in ("../evil", "/etc/passwd", "output/x", "a/../../b", ""):
            with pytest.raises(ValueError, match="(?i)sandbox input"):
                await engine.execute_python("pass", input_files={name: b"x"})
        mock_docker_client.containers.create.assert_not_called()

    def test_validate_input_name_accepts_nested_relative_paths(self) -> None:
        assert str(validate_input_name("data/a.csv")) == "data/a.csv"


# ---------------------------------------------------------------------------
# ADR-007 egress topology (Docker SDK mocked)
# ---------------------------------------------------------------------------

_TARGET = EgressTarget(
    scheme="https", hostname="api.example.test", address="93.184.216.34", port=443
)


def _egress_client(mock_docker_client: MagicMock) -> tuple[MagicMock, MagicMock, MagicMock]:
    network = MagicMock()
    network.name = "fraudai-sbx-test"
    sandbox, proxy = MagicMock(), MagicMock()
    for container in (sandbox, proxy):
        container.short_id = "c0ffee"
        container.wait.return_value = {"StatusCode": 0}
        container.logs.return_value = b"READY"
        container.status = "running"
    mock_docker_client.networks.create.return_value = network
    # the forwarder is created first, then the sandbox
    mock_docker_client.containers.create.side_effect = [proxy, sandbox]
    return network, proxy, sandbox


class TestEgressTopology:
    async def test_dedicated_internal_network_without_host_address(
        self, engine: SandboxEngine, mock_docker_client: MagicMock
    ) -> None:
        network, _, _ = _egress_client(mock_docker_client)

        await engine.execute_python("pass", egress=_TARGET)

        net_kwargs = mock_docker_client.networks.create.call_args.kwargs
        assert net_kwargs["internal"] is True
        # ``internal`` alone leaves the bridge gateway (the host) reachable.
        assert net_kwargs["options"] == {"com.docker.network.bridge.inhibit_ipv4": "true"}
        assert net_kwargs["labels"]["fraudai.sandbox"] == "1"
        network.remove.assert_called_once_with()

    async def test_forwarder_is_pinned_to_the_validated_address(
        self, engine: SandboxEngine, mock_docker_client: MagicMock
    ) -> None:
        network, proxy, _ = _egress_client(mock_docker_client)

        await engine.execute_python("pass", egress=_TARGET)

        proxy_kwargs = mock_docker_client.containers.create.call_args_list[0].kwargs
        assert proxy_kwargs["command"][-2:] == ["93.184.216.34", "443"]
        assert proxy_kwargs["read_only"] is True
        assert proxy_kwargs["cap_drop"] == ["ALL"]
        # the hostname is only a DNS alias on the sandbox network, never the destination
        network.connect.assert_called_once_with(proxy, aliases=["api.example.test"])

    async def test_sandbox_joins_only_the_internal_network(
        self, engine: SandboxEngine, mock_docker_client: MagicMock
    ) -> None:
        _egress_client(mock_docker_client)

        await engine.execute_python("pass", egress=_TARGET)

        sandbox_kwargs = mock_docker_client.containers.create.call_args_list[1].kwargs
        assert sandbox_kwargs["network"] == "fraudai-sbx-test"
        assert "network_mode" not in sandbox_kwargs

    async def test_everything_removed_including_network(
        self, engine: SandboxEngine, mock_docker_client: MagicMock
    ) -> None:
        network, proxy, sandbox = _egress_client(mock_docker_client)

        await engine.execute_python("pass", egress=_TARGET)

        sandbox.remove.assert_called_once_with(force=True)
        proxy.remove.assert_called_once_with(force=True)
        network.remove.assert_called_once_with()

    async def test_partial_setup_failure_cleans_up_what_exists(
        self, engine: SandboxEngine, mock_docker_client: MagicMock
    ) -> None:
        network, proxy, _ = _egress_client(mock_docker_client)
        proxy.start.side_effect = RuntimeError("daemon refused")

        with pytest.raises(RuntimeError, match="daemon refused"):
            await engine.execute_python("pass", egress=_TARGET)

        proxy.remove.assert_called_once_with(force=True)
        network.remove.assert_called_once_with()

    async def test_forwarder_that_never_gets_ready_aborts_the_run(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        network, proxy, _ = _egress_client(mock_docker_client)
        proxy.logs.return_value = b""
        proxy.status = "exited"

        with pytest.raises(SandboxError, match="not become ready"):
            await engine.execute_python("pass", egress=_TARGET)

        assert mock_docker_client.containers.create.call_count == 1  # sandbox never created
        network.remove.assert_called_once_with()

    async def test_ip_literal_target_refused_before_any_docker_call(
        self, engine: SandboxEngine, mock_docker_client: MagicMock
    ) -> None:
        literal = EgressTarget(
            scheme="http", hostname="93.184.216.34", address="93.184.216.34", port=80
        )

        with pytest.raises(EgressDeniedError, match="IP-literal"):
            await engine.execute_python("pass", egress=literal)

        mock_docker_client.networks.create.assert_not_called()
        mock_docker_client.containers.create.assert_not_called()


# ---------------------------------------------------------------------------
# Global concurrency gate
# ---------------------------------------------------------------------------


class TestConcurrencyGate:
    async def test_runs_never_exceed_max_concurrent(self, mock_docker_client: MagicMock) -> None:
        engine = SandboxEngine(docker_client=mock_docker_client, max_concurrent=2)
        active = 0
        peak = 0

        def fake_create(**_: Any) -> MagicMock:
            container = MagicMock()
            container.short_id = "x"
            container.logs.return_value = b""

            def wait(timeout: int) -> dict[str, int]:
                import time

                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                time.sleep(0.05)
                active -= 1
                return {"StatusCode": 0}

            container.wait.side_effect = wait
            return container

        mock_docker_client.containers.create.side_effect = fake_create

        await asyncio.gather(*(engine.execute_python("pass") for _ in range(6)))

        assert peak == 2

    def test_max_concurrent_must_be_positive(self, mock_docker_client: MagicMock) -> None:
        with pytest.raises(ValueError, match="max_concurrent"):
            SandboxEngine(docker_client=mock_docker_client, max_concurrent=0)


# ---------------------------------------------------------------------------
# Output collection -- real filesystem, hostile content
# ---------------------------------------------------------------------------


class TestReadOutputTree:
    def test_reads_regular_files_including_nested(self, tmp_path: Path) -> None:
        (tmp_path / "sub").mkdir()
        (tmp_path / "result.json").write_bytes(b"{}")
        (tmp_path / "sub" / "a.txt").write_bytes(b"a")

        files, rejected = read_output_tree(tmp_path, OutputCaps())

        assert files == {"result.json": b"{}", "sub/a.txt": b"a"}
        assert rejected == []

    def test_symlinks_are_never_followed(self, tmp_path: Path) -> None:
        secret = tmp_path / "outside" / "secret.txt"
        secret.parent.mkdir()
        secret.write_bytes(b"host secret")
        out = tmp_path / "out"
        out.mkdir()
        os.symlink(secret, out / "leak.txt")
        os.symlink(secret.parent, out / "dirlink")
        (out / "ok.txt").write_bytes(b"ok")

        files, rejected = read_output_tree(out, OutputCaps())

        assert files == {"ok.txt": b"ok"}
        assert rejected == ["dirlink", "leak.txt"]
        assert b"host secret" not in b"".join(files.values())

    def test_special_files_are_refused(self, tmp_path: Path) -> None:
        os.mkfifo(tmp_path / "pipe")

        files, rejected = read_output_tree(tmp_path, OutputCaps())

        assert files == {}
        assert rejected == ["pipe"]

    def test_caps_on_file_size_total_and_count(self, tmp_path: Path) -> None:
        (tmp_path / "big").write_bytes(b"x" * 11)
        for index in range(3):
            (tmp_path / f"f{index}").write_bytes(b"12345")

        files, rejected = read_output_tree(
            tmp_path, OutputCaps(max_total_bytes=12, max_file_bytes=10, max_files=2)
        )

        assert "big" in rejected  # over the per-file cap
        assert sorted(files) == ["f0", "f1"]  # count cap reached
        assert rejected == ["big", "f2"]
        assert sum(len(v) for v in files.values()) <= 12
