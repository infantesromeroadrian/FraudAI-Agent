"""Tests for the sandbox execution engine.

All Docker SDK interactions are mocked -- no real Docker daemon is required.
"""

from __future__ import annotations

import asyncio
import os
import threading
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
    from collections.abc import Callable
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
    container.status = "exited"  # what reload() reports once the container has stopped
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
    network.attrs = {"EnableIPv6": False, "IPAM": {"Config": [{"Subnet": "172.30.0.0/16"}]}}
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
        assert net_kwargs["options"] == {
            "com.docker.network.bridge.inhibit_ipv4": "true",
            "com.docker.network.enable_ipv6": "false",
        }
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

        files, rejected, *_ = read_output_tree(tmp_path, OutputCaps())

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

        files, rejected, *_ = read_output_tree(out, OutputCaps())

        assert files == {"ok.txt": b"ok"}
        assert rejected == ["dirlink", "leak.txt"]
        assert b"host secret" not in b"".join(files.values())

    def test_special_files_are_refused(self, tmp_path: Path) -> None:
        os.mkfifo(tmp_path / "pipe")

        files, rejected, *_ = read_output_tree(tmp_path, OutputCaps())

        assert files == {}
        assert rejected == ["pipe"]

    def test_caps_on_file_size_total_and_count(self, tmp_path: Path) -> None:
        (tmp_path / "big").write_bytes(b"x" * 11)
        for index in range(3):
            (tmp_path / f"f{index}").write_bytes(b"12345")

        files, rejected, *_ = read_output_tree(
            tmp_path, OutputCaps(max_total_bytes=12, max_file_bytes=10, max_files=2)
        )

        assert "big" in rejected  # over the per-file cap
        assert sorted(files) == ["f0", "f1"]  # count cap reached
        assert rejected == ["big", "f2"]
        assert sum(len(v) for v in files.values()) <= 12


# ---------------------------------------------------------------------------
# Correction round -- every test below was written red first
# ---------------------------------------------------------------------------


class TestOutputsAreReadOnlyAfterTheWriterIsGone:
    """CX1: never read /workspace/output while the sandbox may still be writing."""

    @staticmethod
    def _broken_stop(container: MagicMock) -> None:
        container.wait.side_effect = TimeoutError("wait failed")
        container.stop.side_effect = RuntimeError("stop failed")
        container.kill.side_effect = RuntimeError("kill failed")
        container.status = "running"  # reload() keeps reporting a live writer

    async def test_unconfirmed_stop_discards_outputs_and_raises(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._broken_stop(mock_container)
        mock_container.remove.side_effect = RuntimeError("remove failed")
        mock_docker_client.containers.create.return_value = mock_container
        reads: list[object] = []
        monkeypatch.setattr(
            "fraudai.tools.sandbox.read_output_tree", lambda *a, **k: reads.append(a)
        )

        with pytest.raises(SandboxError, match="outputs were discarded"):
            await engine.execute_python("pass")

        assert reads == []  # zero reads when the end of the writer cannot be ensured

    async def test_forced_removal_is_verified_before_outputs_are_read(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import docker.errors

        self._broken_stop(mock_container)
        events: list[str] = []
        removed = False

        def remove(force: bool = False) -> None:
            nonlocal removed
            removed = True
            events.append("remove")

        def reload() -> None:
            if removed:
                raise docker.errors.NotFound("gone")

        mock_container.remove.side_effect = remove
        mock_container.reload.side_effect = reload
        mock_docker_client.containers.create.return_value = mock_container
        monkeypatch.setattr(
            "fraudai.tools.sandbox.read_output_tree",
            lambda *a, **k: events.append("read") or ({}, [], 0, False),
        )

        result = await engine.execute_python("pass")

        assert result.timed_out is True
        assert events.index("remove") < events.index("read")

    async def test_stopped_container_outputs_are_still_collected(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        mock_container.wait.side_effect = TimeoutError("took too long")
        mock_container.status = "exited"
        mock_docker_client.containers.create.return_value = mock_container

        result = await engine.execute_python("pass")

        assert result.timed_out is True
        mock_container.stop.assert_called_once_with(timeout=5)
        mock_container.remove.assert_called_once_with(force=True)

    def test_a_directory_swapped_for_a_symlink_mid_walk_is_not_followed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The live writer wins the race after the walker decided ``sub`` is a directory."""
        canary = tmp_path / "outside" / "canary.txt"
        canary.parent.mkdir()
        canary.write_bytes(b"HOST-CANARY")
        out = tmp_path / "out"
        (out / "sub").mkdir(parents=True)
        (out / "sub" / "inner.txt").write_bytes(b"inner")
        (out / "ok.txt").write_bytes(b"ok")
        real_scandir = os.scandir
        calls = 0

        def racing_scandir(path: Any = ".") -> Any:
            nonlocal calls
            calls += 1
            if calls == 2:  # descending into ``sub``: the writer swaps it right now
                (out / "sub").rename(out / "sub-original")
                os.symlink(canary.parent, out / "sub")
            return real_scandir(path)

        monkeypatch.setattr(os, "scandir", racing_scandir)

        collection = read_output_tree(out, OutputCaps())
        files = collection[0]

        assert b"HOST-CANARY" not in b"".join(files.values())
        assert "sub/canary.txt" not in files
        assert files["ok.txt"] == b"ok"


class TestCancellationDuringCreate:
    """CX2: a cancelled await must not orphan what Docker creates afterwards."""

    @staticmethod
    def _blocking(
        entered: threading.Event, release: threading.Event, created: MagicMock
    ) -> Callable[..., MagicMock]:
        def create(**_: Any) -> MagicMock:
            entered.set()
            release.wait(timeout=5)  # Docker is still working when the caller gives up
            return created

        return create

    @pytest.mark.parametrize("blocked", ["network", "proxy", "sandbox"])
    async def test_resource_created_after_cancel_is_removed_before_the_gate_opens(
        self, mock_docker_client: MagicMock, blocked: str
    ) -> None:
        engine = SandboxEngine(docker_client=mock_docker_client, max_concurrent=1)
        network, proxy, sandbox = _egress_client(mock_docker_client)
        entered, release = threading.Event(), threading.Event()
        created = {"network": network, "proxy": proxy, "sandbox": sandbox}[blocked]
        slow = self._blocking(entered, release, created)
        if blocked == "network":
            mock_docker_client.networks.create.side_effect = slow
        else:
            ordered = {"proxy": 0, "sandbox": 1}[blocked]
            fast = [proxy, sandbox]

            def containers_create(**kwargs: Any) -> MagicMock:
                index = containers_create.calls  # type: ignore[attr-defined]
                containers_create.calls += 1  # type: ignore[attr-defined]
                return slow(**kwargs) if index == ordered else fast[index]

            containers_create.calls = 0  # type: ignore[attr-defined]
            mock_docker_client.containers.create.side_effect = containers_create

        task = asyncio.create_task(engine.execute_python("pass", egress=_TARGET))
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

        removed = {
            "network": network.remove.called,
            "proxy": proxy.remove.called,
            "sandbox": sandbox.remove.called,
        }
        assert removed[blocked], f"{blocked} created after the cancel was never removed"
        # the gate was released only after cleanup: the next run is not blocked
        sandbox_ok = MagicMock(short_id="n", status="exited")
        sandbox_ok.wait.return_value = {"StatusCode": 0}
        sandbox_ok.logs.return_value = b""
        mock_docker_client.containers.create.side_effect = None
        mock_docker_client.containers.create.return_value = sandbox_ok
        result = await asyncio.wait_for(engine.execute_python("pass"), timeout=5)
        assert result.exit_code == 0


class TestOutputWorkIsBounded:
    """CX3: hostile output cannot make the host walk or remember unbounded entries."""

    def test_many_small_entries_keep_stored_rejections_bounded(self, tmp_path: Path) -> None:
        for index in range(4096):
            (tmp_path / f"e{index:04d}").write_bytes(b"")

        files, rejected, *_ = read_output_tree(tmp_path, OutputCaps())[:2]

        assert len(files) == 64
        assert len(rejected) <= 64  # was 4032 remembered names


class TestConfiguredTimeoutIsACeiling:
    """W2: SANDBOX_TIMEOUT_SECONDS bounds every call, tools may only ask for less."""

    @pytest.mark.parametrize(("requested", "expected"), [(300, 60), (30, 30)])
    async def test_effective_timeout_never_exceeds_the_engine_limit(
        self,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
        requested: int,
        expected: int,
    ) -> None:
        mock_docker_client.containers.create.return_value = mock_container
        engine = SandboxEngine(docker_client=mock_docker_client, timeout=60)

        await engine.execute_python("pass", timeout=requested)

        mock_container.wait.assert_called_once_with(timeout=expected)


class TestEffectiveIPv6IsVerified:
    """Chief: passing enable_ipv6=False is not proof; inspect the network that exists."""

    @pytest.mark.parametrize(
        "attrs",
        [
            {"EnableIPv6": True, "IPAM": {"Config": [{"Subnet": "172.30.0.0/16"}]}},
            {"IPAM": {"Config": [{"Subnet": "172.30.0.0/16"}]}},  # unknown
            {"EnableIPv6": False, "IPAM": {"Config": [{"Subnet": "fd00:1::/64"}]}},
        ],
    )
    async def test_network_that_may_carry_ipv6_aborts_before_any_container(
        self, engine: SandboxEngine, mock_docker_client: MagicMock, attrs: dict[str, Any]
    ) -> None:
        network, _, _ = _egress_client(mock_docker_client)
        network.attrs = attrs

        with pytest.raises(SandboxError, match="IPv6"):
            await engine.execute_python("pass", egress=_TARGET)

        mock_docker_client.containers.create.assert_not_called()
        network.remove.assert_called_once_with()

    async def test_network_is_created_with_ipv6_explicitly_off(
        self, engine: SandboxEngine, mock_docker_client: MagicMock
    ) -> None:
        _egress_client(mock_docker_client)

        await engine.execute_python("pass", egress=_TARGET)

        options = mock_docker_client.networks.create.call_args.kwargs["options"]
        assert options["com.docker.network.enable_ipv6"] == "false"


class TestBoundedOutputWalk:
    """CX3 (after the fix): explicit, bounded work and metadata for hostile trees."""

    def test_entry_budget_stops_the_walk_before_it_is_materialised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for index in range(4096):
            (tmp_path / f"e{index:04d}").write_bytes(b"")
        pulled = 0
        real_scandir = os.scandir

        class CountingScandir:
            def __init__(self, handle: Any) -> None:
                self._handle = handle

            def __enter__(self) -> CountingScandir:
                return self

            def __exit__(self, *exc: object) -> None:
                self._handle.close()

            def __iter__(self) -> CountingScandir:
                return self

            def __next__(self) -> Any:
                nonlocal pulled
                entry = next(self._handle)
                pulled += 1
                return entry

        monkeypatch.setattr(os, "scandir", lambda fd: CountingScandir(real_scandir(fd)))
        caps = OutputCaps()

        collection = read_output_tree(tmp_path, caps)

        assert pulled <= caps.max_entries + 1  # +1: the peek that detects "there is more"
        assert collection.truncated is True
        assert len(collection.files) == caps.max_files
        assert len(collection.rejected) <= caps.max_rejected_names
        assert collection.rejected_total >= len(collection.rejected)
        assert collection.rejected == sorted(collection.rejected)

    def test_rejections_are_counted_beyond_the_names_that_are_stored(self, tmp_path: Path) -> None:
        for index in range(10):
            (tmp_path / f"f{index}").write_bytes(b"x")

        collection = read_output_tree(tmp_path, OutputCaps(max_files=2, max_rejected_names=3))

        assert len(collection.files) == 2
        assert len(collection.rejected) == 3
        assert collection.rejected_total == 8
        assert collection.truncated is False  # nothing was skipped unseen: all 10 were visited

    def test_result_json_survives_a_flood_of_junk_entries(self, tmp_path: Path) -> None:
        (tmp_path / "result.json").write_bytes(b'{"ok": true}')
        for index in range(3000):
            (tmp_path / f"a{index:04d}").write_bytes(b"")

        collection = read_output_tree(tmp_path, OutputCaps())

        assert collection.files["result.json"] == b'{"ok": true}'
        assert collection.truncated is True

    def test_depth_is_capped(self, tmp_path: Path) -> None:
        deepest = tmp_path
        for level in range(12):
            deepest = deepest / f"d{level}"
        deepest.mkdir(parents=True)
        (deepest / "deep.txt").write_bytes(b"deep")
        (tmp_path / "d0" / "shallow.txt").write_bytes(b"shallow")

        collection = read_output_tree(tmp_path, OutputCaps(max_depth=4))

        assert collection.files == {"d0/shallow.txt": b"shallow"}
        assert collection.truncated is True
        assert any(name.endswith("d3") for name in collection.rejected)

    async def test_engine_reports_truncation_explicitly(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_docker_client.containers.create.return_value = mock_container
        monkeypatch.setattr(
            "fraudai.tools.sandbox.read_output_tree",
            lambda *a, **k: ({"a": b"1"}, ["x"], 4000, True),
        )

        result = await engine.execute_python("pass")

        assert (result.rejected_outputs, result.rejected_total, result.outputs_truncated) == (
            ["x"],
            4000,
            True,
        )

    def test_live_writer_cannot_redirect_the_walk_with_swapped_directories(
        self, tmp_path: Path
    ) -> None:
        """Hammer: a thread flips ``sub`` between a directory and a symlink while we walk."""
        canary = tmp_path / "outside" / "canary.txt"
        canary.parent.mkdir()
        canary.write_bytes(b"HOST-CANARY")
        out = tmp_path / "out"
        (out / "sub").mkdir(parents=True)
        (out / "sub" / "inner.txt").write_bytes(b"inner")
        stop = threading.Event()

        def flip() -> None:
            while not stop.is_set():
                try:
                    (out / "sub").rename(out / "sub-dir")
                    os.symlink(canary.parent, out / "sub")
                    (out / "sub").unlink()
                    (out / "sub-dir").rename(out / "sub")
                except OSError:
                    continue

        thread = threading.Thread(target=flip, daemon=True)
        thread.start()
        try:
            for _ in range(300):
                files = read_output_tree(out, OutputCaps()).files
                assert b"HOST-CANARY" not in b"".join(files.values())
        finally:
            stop.set()
            thread.join(timeout=5)


class TestStopEscalation:
    async def test_kill_is_used_when_stop_fails_and_outputs_are_then_read(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        state = {"status": "running"}
        mock_container.wait.side_effect = TimeoutError("slow")
        mock_container.stop.side_effect = RuntimeError("stop failed")
        mock_container.kill.side_effect = lambda: state.update(status="exited")
        type(mock_container).status = property(lambda _self: state["status"])
        mock_docker_client.containers.create.return_value = mock_container

        result = await engine.execute_python("pass")

        assert result.timed_out is True
        mock_container.kill.assert_called_once_with()
        mock_container.remove.assert_called_once_with(force=True)  # normal cleanup only


class TestCleanupSurvivesCancellation:
    async def test_cancelling_during_the_run_still_removes_everything_first(
        self, mock_docker_client: MagicMock, mock_container: MagicMock
    ) -> None:
        engine = SandboxEngine(docker_client=mock_docker_client, max_concurrent=1)
        waiting, release = threading.Event(), threading.Event()
        removed = threading.Event()

        def wait(timeout: int) -> dict[str, int]:
            waiting.set()
            release.wait(timeout=5)
            return {"StatusCode": 0}

        def remove(force: bool = False) -> None:
            release.set()  # removing the container unblocks the waiting thread
            removed.set()

        mock_container.wait.side_effect = wait
        mock_container.remove.side_effect = remove
        mock_docker_client.containers.create.return_value = mock_container

        task = asyncio.create_task(engine.execute_python("pass"))
        assert await asyncio.to_thread(waiting.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert removed.is_set()


class TestResolverIsPinnedInert:
    """DNS finding: the embedded resolver forwards to whatever upstream is reachable,
    so the sandbox's upstream is fixed to an inert address instead of the daemon's."""

    async def test_networked_sandbox_has_no_external_dns_upstream(
        self, engine: SandboxEngine, mock_docker_client: MagicMock
    ) -> None:
        _egress_client(mock_docker_client)

        await engine.execute_python("pass", egress=_TARGET)

        sandbox_kwargs = mock_docker_client.containers.create.call_args_list[1].kwargs
        assert sandbox_kwargs["dns"] == ["127.0.0.1"]

    async def test_sandbox_without_network_does_not_set_dns(
        self, engine: SandboxEngine, mock_docker_client: MagicMock, mock_container: MagicMock
    ) -> None:
        mock_docker_client.containers.create.return_value = mock_container

        await engine.execute_python("pass")

        assert "dns" not in mock_docker_client.containers.create.call_args.kwargs


class TestForcedRemovalThatLeavesAStoppedContainer:
    """Residual CX1: ``exited`` is not ``gone``; the final removal must still happen."""

    @staticmethod
    def _stateful(mock_container: MagicMock) -> dict[str, Any]:
        import docker.errors

        state: dict[str, Any] = {"status": "running", "exists": True, "remove_calls": 0}

        def stop(timeout: int = 10) -> None:
            raise RuntimeError("stop failed")

        def kill() -> None:
            raise RuntimeError("kill failed")

        def reload() -> None:
            if not state["exists"]:
                raise docker.errors.NotFound("gone")

        mock_container.wait.side_effect = TimeoutError("wait failed")
        mock_container.stop.side_effect = stop
        mock_container.kill.side_effect = kill
        mock_container.reload.side_effect = reload
        type(mock_container).status = property(lambda _self: state["status"])
        return state

    async def test_exited_container_whose_removal_errored_is_removed_again(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        state = self._stateful(mock_container)

        def remove(force: bool = False) -> None:
            state["remove_calls"] += 1
            if state["remove_calls"] == 1:
                state["status"] = "exited"  # the daemon killed it, then reported an error
                raise RuntimeError("removal reported an error")
            state["exists"] = False

        mock_container.remove.side_effect = remove
        mock_docker_client.containers.create.return_value = mock_container

        result = await engine.execute_python("pass")

        assert result.timed_out is True  # stopped: the writer is gone, outputs may be read
        assert state["exists"] is False, "an exited container was left behind"
        assert state["remove_calls"] == 2

    async def test_really_removed_container_is_not_removed_a_second_time(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
    ) -> None:
        state = self._stateful(mock_container)

        def remove(force: bool = False) -> None:
            state["remove_calls"] += 1
            state["exists"] = False

        mock_container.remove.side_effect = remove
        mock_docker_client.containers.create.return_value = mock_container

        result = await engine.execute_python("pass")

        assert result.timed_out is True
        assert state["remove_calls"] == 1  # NotFound confirmed the absence

    async def test_unknown_state_after_removal_failure_discards_outputs(
        self,
        engine: SandboxEngine,
        mock_docker_client: MagicMock,
        mock_container: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._stateful(mock_container)
        mock_container.reload.side_effect = RuntimeError("daemon unreachable")
        mock_container.remove.side_effect = RuntimeError("remove failed")
        mock_docker_client.containers.create.return_value = mock_container
        reads: list[object] = []
        monkeypatch.setattr("fraudai.tools.sandbox.read_output_tree", lambda *a: reads.append(a))

        with pytest.raises(SandboxError, match="outputs were discarded"):
            await engine.execute_python("pass")

        assert reads == []
