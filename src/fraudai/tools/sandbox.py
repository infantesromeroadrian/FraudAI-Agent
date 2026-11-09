"""Sandbox execution engine -- ephemeral Docker containers with security isolation.

Implements ADR-004 as amended by ADR-007: each execution runs in a throwaway
container with a read-only root filesystem, a ``noexec`` tmpfs, read-only
inputs, a single writable output directory that is read back without following
symlinks, enforced CPU/memory/pids limits and a global concurrency gate.

Network policy: **none by default** (``network_mode="none"``). Only when the
caller passes a validated :class:`~fraudai.tools.egress.EgressTarget` does the
sandbox get a network, and then it is a dedicated ``internal`` bridge *without
a host-side IP* that contains nothing but a forwarding container. That
container is the only hop with a route out, and it dials the single validated
address, so the sandbox cannot reach the host, the Docker gateway, link-local
metadata services, other containers or any other address, whether or not the
code running inside cooperates. ``internal`` alone does not give this
guarantee (the bridge gateway stays reachable); ``inhibit_ipv4`` is what removes
the host address from the network.
"""

from __future__ import annotations

import asyncio
import logging
import os
import stat
import tempfile
import time
import uuid
import weakref
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import docker
import docker.errors
from docker.types import LogConfig, Mount, Ulimit
from pydantic import BaseModel

from fraudai.core.metrics import SANDBOX_DURATION, SANDBOX_EXECUTIONS
from fraudai.tools.egress import FORWARDER_SOURCE, EgressTarget, alias_for

if TYPE_CHECKING:
    from docker.models.containers import Container
    from docker.models.networks import Network

logger = logging.getLogger(__name__)

_MIB = 1024 * 1024
_RESOURCE_LABEL = "fraudai.sandbox"
_PROXY_READY_TIMEOUT_SECONDS = 10.0
_NANO = 1_000_000_000


class SandboxError(RuntimeError):
    """The sandbox could not be prepared or run (not a failure of the user code)."""


class SandboxResult(BaseModel):
    """Result of a sandboxed execution."""

    exit_code: int
    stdout: str
    stderr: str
    output_files: dict[str, bytes] = {}
    # Output entries that were refused (symlink, special file, over a cap).
    rejected_outputs: list[str] = []
    duration_seconds: float
    timed_out: bool = False

    model_config = {"arbitrary_types_allowed": True}


@dataclass(frozen=True, slots=True)
class OutputCaps:
    """Limits applied when reading ``/workspace/output`` back to the host."""

    max_total_bytes: int = 8 * _MIB
    max_file_bytes: int = 4 * _MIB
    max_files: int = 64


def read_output_tree(root: Path, caps: OutputCaps) -> tuple[dict[str, bytes], list[str]]:
    """Read regular files below *root* without ever following a symlink.

    The sandboxed code controls this tree, so every entry is untrusted:
    symlinks (to files or directories), devices, FIFOs and anything beyond the
    caps are refused and reported by name instead of being read.

    Returns:
        ``(files, rejected)`` where ``files`` maps POSIX relative paths to bytes.
    """
    files: dict[str, bytes] = {}
    rejected: list[str] = []
    total = 0
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                relative = str(PurePosixPath(*Path(entry.path).relative_to(root).parts))
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                    continue
                if not stat.S_ISREG(info.st_mode):
                    rejected.append(relative)
                    continue
                if (
                    len(files) >= caps.max_files
                    or info.st_size > caps.max_file_bytes
                    or total + info.st_size > caps.max_total_bytes
                ):
                    rejected.append(relative)
                    continue
                content = _read_regular_file(entry.path, caps.max_file_bytes)
                if content is None:
                    rejected.append(relative)
                    continue
                total += len(content)
                files[relative] = content
    return files, sorted(rejected)


def _read_regular_file(path: str, limit: int) -> bytes | None:
    """Open with ``O_NOFOLLOW`` (guards the stat/open race) and re-check the type."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            return None
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            data = handle.read(limit + 1)
    finally:
        os.close(descriptor)
    return data if len(data) <= limit else None


def validate_input_name(filename: str) -> PurePosixPath:
    """Return the relative path for an input file or raise ``ValueError``."""
    path = PurePosixPath(filename)
    if path.is_absolute() or not path.parts or ".." in path.parts or "." in path.parts:
        msg = f"Invalid sandbox input file name: {filename!r}"
        raise ValueError(msg)
    if path.parts[0] == "output":
        msg = "Sandbox input files must not be placed under the output directory"
        raise ValueError(msg)
    return path


# One gate per (event loop, capacity): asyncio semaphores bind to the loop that
# first awaits them, and all engines sharing a capacity share the gate.
_GATES: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[int, asyncio.Semaphore]] = (
    weakref.WeakKeyDictionary()
)


def _concurrency_gate(capacity: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    return _GATES.setdefault(loop, {}).setdefault(capacity, asyncio.Semaphore(capacity))


@dataclass(slots=True)
class _RunResources:
    """Docker objects created by one run, registered as soon as they exist so a
    failure half-way through setup still cleans up exactly what was created."""

    container: Container | None = None
    proxy: Container | None = None
    network: Network | None = None


class SandboxEngine:
    """Executes code in ephemeral Docker containers with security isolation.

    Each execution creates a new container, mounts the inputs read-only, runs
    the code, reads the outputs back and destroys everything it created in a
    ``finally`` block.

    Security posture (ADR-004 / ADR-007):
      - no network unless an ``EgressTarget`` is supplied (see module docs)
      - read-only root filesystem, ``/tmp`` tmpfs with ``noexec,nosuid,nodev``
      - inputs bind-mounted read-only; only ``/workspace/output`` is writable
      - ``cap_drop=["ALL"]`` and ``no-new-privileges``; Docker's default seccomp
        profile (no custom profile is shipped)
      - memory (no swap), CPU quota (``nano_cpus``), ``pids_limit``, per-file
        ``fsize`` ulimit and a global concurrency gate
    """

    SANDBOX_IMAGE: str = "fraudai-sandbox:latest"
    DEFAULT_TIMEOUT: int = 300
    DEFAULT_MEM_LIMIT: str = "2g"
    DEFAULT_CPU_COUNT: float = 2.0
    DEFAULT_PIDS_LIMIT: int = 128
    DEFAULT_MAX_CONCURRENT: int = 2
    TMPFS_OPTIONS: str = "rw,noexec,nosuid,nodev,size=64m"
    WORKSPACE_DIR: str = "/workspace"
    OUTPUT_DIR: str = "/workspace/output"
    MAX_LOG_BYTES: int = _MIB

    def __init__(
        self,
        docker_client: docker.DockerClient | None = None,
        image: str = SANDBOX_IMAGE,
        timeout: int = DEFAULT_TIMEOUT,
        mem_limit: str = DEFAULT_MEM_LIMIT,
        cpu_count: float = DEFAULT_CPU_COUNT,
        pids_limit: int = DEFAULT_PIDS_LIMIT,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        output_caps: OutputCaps | None = None,
    ) -> None:
        if max_concurrent < 1:
            msg = "max_concurrent must be at least 1"
            raise ValueError(msg)
        self._client = docker_client or docker.from_env()
        self._image = image
        self._timeout = timeout
        self._mem_limit = mem_limit
        self._cpu_count = cpu_count
        self._pids_limit = pids_limit
        self._max_concurrent = max_concurrent
        self._output_caps = output_caps or OutputCaps()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute_python(
        self,
        code: str,
        input_files: dict[str, bytes] | None = None,
        timeout: int | None = None,
        egress: EgressTarget | None = None,
    ) -> SandboxResult:
        """Execute Python code in a sandboxed container.

        Args:
            code: Python source code to execute.
            input_files: Mapping of ``{filename: content}`` mounted read-only
                under ``/workspace/``.
            timeout: Override the default timeout (seconds).
            egress: Validated destination the sandbox may reach. ``None`` (the
                default) means no network at all.
        """
        return await self._run_container(
            command=["python", "-c", code],
            input_files=input_files,
            timeout=timeout if timeout is not None else self._timeout,
            egress=egress,
        )

    async def execute_script(
        self,
        script_path: str,
        args: list[str] | None = None,
        input_files: dict[str, bytes] | None = None,
        timeout: int | None = None,
        egress: EgressTarget | None = None,
    ) -> SandboxResult:
        """Execute a pre-built analysis script that already exists in the image."""
        return await self._run_container(
            command=["python", script_path, *(args or [])],
            input_files=input_files,
            timeout=timeout if timeout is not None else self._timeout,
            egress=egress,
        )

    async def health_check(self) -> bool:
        """Check that Docker is reachable and the sandbox image exists."""
        try:
            ping_ok: bool = await asyncio.to_thread(self._client.ping)
            if not ping_ok:
                return False
            await asyncio.to_thread(self._client.images.get, self._image)
            return True
        except (docker.errors.APIError, docker.errors.ImageNotFound):
            logger.warning("Sandbox health check failed: image '%s' not found", self._image)
            return False
        except Exception:
            logger.exception("Sandbox health check failed unexpectedly")
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _container_user() -> str | None:
        """Run as the host uid/gid so bind-mounted output is owned (and removable) by us."""
        uid, gid = os.getuid(), os.getgid()
        return f"{uid}:{gid}" if uid != 0 else None

    def _hardening(self, run_id: str, role: str) -> dict[str, Any]:
        """Options shared by the sandbox and the egress forwarder container."""
        return {
            "read_only": True,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges"],
            "ipc_mode": "private",
            "init": True,
            "labels": {_RESOURCE_LABEL: "1", f"{_RESOURCE_LABEL}.run": run_id, "role": role},
            "log_config": LogConfig(type="json-file", config={"max-size": "1m", "max-file": "1"}),
            "environment": {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"},
            "detach": True,
        }

    def _write_workspace(
        self, workspace: Path, output: Path, input_files: dict[str, bytes] | None
    ) -> None:
        workspace.mkdir()
        output.mkdir()
        for filename, content in (input_files or {}).items():
            relative = validate_input_name(filename)
            destination = workspace.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        # Same-uid containers need no widened permissions; a root host process
        # (the only case where we keep the image user) needs a writable output.
        if os.getuid() == 0:
            output.chmod(0o777)

    async def _create_egress(
        self, target: EgressTarget, run_id: str, resources: _RunResources
    ) -> Network:
        """Create the sandbox-only network and the forwarder pinned to ``target``."""
        alias = alias_for(target)
        network: Network = await asyncio.to_thread(
            self._client.networks.create,
            name=f"fraudai-sbx-{run_id}",
            driver="bridge",
            internal=True,
            options={"com.docker.network.bridge.inhibit_ipv4": "true"},
            labels={_RESOURCE_LABEL: "1", f"{_RESOURCE_LABEL}.run": run_id},
        )
        resources.network = network
        proxy: Container = await asyncio.to_thread(
            self._client.containers.create,
            image=self._image,
            command=["python", "-c", FORWARDER_SOURCE, target.address, str(target.port)],
            name=f"fraudai-sbxproxy-{run_id}",
            mem_limit="128m",
            memswap_limit="128m",
            nano_cpus=_NANO // 2,
            pids_limit=64,
            # Lets the non-root forwarder bind the target port even when < 1024.
            sysctls={"net.ipv4.ip_unprivileged_port_start": "0"},
            ulimits=[Ulimit(name="nofile", soft=256, hard=256)],
            user=self._container_user(),
            **self._hardening(run_id, "egress-proxy"),
        )
        resources.proxy = proxy
        await asyncio.to_thread(network.connect, proxy, aliases=[alias])
        await asyncio.to_thread(proxy.start)
        await self._wait_for_proxy(proxy)
        return network

    async def _wait_for_proxy(self, proxy: Container) -> None:
        deadline = time.monotonic() + _PROXY_READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            await asyncio.to_thread(proxy.reload)
            if proxy.status not in ("created", "running"):
                break
            logs: bytes = await asyncio.to_thread(proxy.logs, stdout=True, stderr=False)
            if b"READY" in logs:
                return
            await asyncio.sleep(0.1)
        msg = "Sandbox egress forwarder did not become ready"
        raise SandboxError(msg)

    async def _run_container(
        self,
        command: list[str],
        input_files: dict[str, bytes] | None,
        timeout: int,
        egress: EgressTarget | None = None,
    ) -> SandboxResult:
        """Create an ephemeral container, run *command*, and return results.

        Everything created here (sandbox, forwarder, network, host tmpdir) is
        removed in a ``finally`` block regardless of success, failure or timeout.
        """
        if egress is not None:
            alias_for(egress)  # refuse unreachable-by-design targets before any effect
        async with _concurrency_gate(self._max_concurrent):
            return await self._run_gated(command, input_files, timeout, egress)

    async def _run_gated(
        self,
        command: list[str],
        input_files: dict[str, bytes] | None,
        timeout: int,
        egress: EgressTarget | None,
    ) -> SandboxResult:
        run_id = uuid.uuid4().hex[:12]
        resources = _RunResources()
        tmpdir_obj = tempfile.TemporaryDirectory(prefix="fraudai-sandbox-")
        tmpdir = Path(tmpdir_obj.name)

        try:
            workspace_host = tmpdir / "workspace"
            output_host = workspace_host / "output"
            self._write_workspace(workspace_host, output_host, input_files)

            if egress is not None:
                network = await self._create_egress(egress, run_id, resources)
                network_args: dict[str, Any] = {"network": network.name}
            else:
                network_args = {"network_mode": "none"}

            max_file = self._output_caps.max_file_bytes
            resources.container = container = await asyncio.to_thread(
                self._client.containers.create,
                image=self._image,
                command=command,
                name=f"fraudai-sbx-{run_id}-run",
                mem_limit=self._mem_limit,
                memswap_limit=self._mem_limit,  # No swap
                # ``cpu_count`` is Windows-only in the Engine API; nano_cpus is the
                # CFS quota that is actually enforced on Linux.
                nano_cpus=int(self._cpu_count * _NANO),
                pids_limit=self._pids_limit,
                ulimits=[
                    Ulimit(name="fsize", soft=max_file, hard=max_file),
                    Ulimit(name="nofile", soft=256, hard=256),
                    Ulimit(name="core", soft=0, hard=0),
                ],
                tmpfs={"/tmp": self.TMPFS_OPTIONS},
                mounts=[
                    Mount(
                        target=self.WORKSPACE_DIR,
                        source=str(workspace_host),
                        type="bind",
                        read_only=True,
                    ),
                    Mount(
                        target=self.OUTPUT_DIR,
                        source=str(output_host),
                        type="bind",
                        read_only=False,
                    ),
                ],
                working_dir=self.WORKSPACE_DIR,
                user=self._container_user(),
                **network_args,
                **{
                    **self._hardening(run_id, "sandbox"),
                    "environment": {
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONUNBUFFERED": "1",
                        "HOME": "/tmp",
                        "MPLCONFIGDIR": "/tmp/matplotlib",
                    },
                },
            )

            logger.info(
                "Sandbox container %s created (image=%s, timeout=%ds, egress=%s)",
                container.short_id,
                self._image,
                timeout,
                "forwarder" if egress is not None else "none",
            )

            await asyncio.to_thread(container.start)
            start_time = time.monotonic()

            timed_out = False
            try:
                wait_result: dict[str, Any] = await asyncio.to_thread(
                    container.wait,
                    timeout=timeout,
                )
                exit_code: int = wait_result.get("StatusCode", -1)
            except Exception:
                # Timeout or unexpected error -- stop the container
                timed_out = True
                exit_code = -1
                logger.warning(
                    "Sandbox container %s timed out after %ds, stopping",
                    container.short_id,
                    timeout,
                )
                try:
                    await asyncio.to_thread(container.stop, timeout=5)
                except Exception:
                    logger.debug(
                        "Failed to stop container %s gracefully, will force-remove",
                        container.short_id,
                    )

            duration = time.monotonic() - start_time

            stdout_bytes = await self._read_logs(container, stdout=True)
            stderr_bytes = await self._read_logs(container, stdout=False)

            collected_files, rejected = await asyncio.to_thread(
                read_output_tree, output_host, self._output_caps
            )
            if rejected:
                logger.warning(
                    "Sandbox container %s: %d output entrie(s) refused",
                    container.short_id,
                    len(rejected),
                )

            logger.info(
                "Sandbox container %s finished "
                "(exit_code=%d, duration=%.2fs, timed_out=%s, output_files=%d)",
                container.short_id,
                exit_code,
                duration,
                timed_out,
                len(collected_files),
            )

            SANDBOX_DURATION.observe(duration)
            if timed_out:
                SANDBOX_EXECUTIONS.labels(status="timeout").inc()
            elif exit_code == 0:
                SANDBOX_EXECUTIONS.labels(status="success").inc()
            else:
                SANDBOX_EXECUTIONS.labels(status="error").inc()

            return SandboxResult(
                exit_code=exit_code,
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                output_files=collected_files,
                rejected_outputs=rejected,
                duration_seconds=round(duration, 3),
                timed_out=timed_out,
            )

        finally:
            await self._cleanup(resources)
            try:
                tmpdir_obj.cleanup()
            except Exception:
                logger.debug("Tmpdir cleanup failed for %s", tmpdir)

    async def _read_logs(self, container: Container, *, stdout: bool) -> bytes:
        data: bytes = await asyncio.to_thread(container.logs, stdout=stdout, stderr=not stdout)
        return data[-self.MAX_LOG_BYTES :]

    @staticmethod
    async def _cleanup(resources: _RunResources) -> None:
        """Remove only the resources this run created, sandbox first."""
        for victim in (resources.container, resources.proxy):
            if victim is None:
                continue
            try:
                await asyncio.to_thread(victim.remove, force=True)
                logger.debug("Sandbox container %s removed", victim.short_id)
            except Exception:
                logger.exception("Failed to remove sandbox container %s", victim.short_id)
        if resources.network is not None:
            try:
                await asyncio.to_thread(resources.network.remove)
            except Exception:
                logger.exception("Failed to remove sandbox network %s", resources.network.name)
