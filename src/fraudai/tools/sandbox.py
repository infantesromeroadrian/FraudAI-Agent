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
import itertools
import logging
import os
import stat
import tempfile
import time
import uuid
import weakref
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, NamedTuple

import docker
import docker.errors
from docker.types import LogConfig, Mount, Ulimit
from pydantic import BaseModel

from fraudai.core.metrics import SANDBOX_DURATION, SANDBOX_EXECUTIONS
from fraudai.tools.egress import FORWARDER_SOURCE, EgressTarget, alias_for

if TYPE_CHECKING:
    from collections.abc import Callable

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
    # Total refusals (``rejected_outputs`` stores at most OutputCaps.max_rejected_names)
    # and whether the bounded walk stopped early; both are explicit, never silent.
    rejected_total: int = 0
    outputs_truncated: bool = False
    duration_seconds: float
    timed_out: bool = False

    model_config = {"arbitrary_types_allowed": True}


@dataclass(frozen=True, slots=True)
class OutputCaps:
    """Limits applied when reading ``/workspace/output`` back to the host.

    Besides bytes and file count, the walk itself is bounded: the sandbox
    controls how many entries exist, so the host never lists, sorts or
    remembers more than ``max_entries`` / ``max_rejected_names`` of them.
    """

    max_total_bytes: int = 8 * _MIB
    max_file_bytes: int = 4 * _MIB
    max_files: int = 64
    max_entries: int = 1024  # entries visited across the whole tree
    max_depth: int = 8
    max_rejected_names: int = 64  # names reported back; the rest are only counted


class OutputCollection(NamedTuple):
    """What came back from the output directory, with explicit truncation."""

    files: dict[str, bytes]
    rejected: list[str]  # sorted, at most ``max_rejected_names`` entries
    rejected_total: int  # every refusal seen, including names not stored
    truncated: bool  # the walk stopped at the entry/depth budget


class _Walk:
    """Mutable bookkeeping for one bounded, descriptor-relative tree walk."""

    def __init__(self, caps: OutputCaps) -> None:
        self.caps = caps
        self.files: dict[str, bytes] = {}
        self.rejected: list[str] = []
        self.rejected_total = 0
        self.total_bytes = 0
        self.visited = 0
        self.truncated = False

    def reject(self, relative: str) -> None:
        self.rejected_total += 1
        if len(self.rejected) < self.caps.max_rejected_names:
            self.rejected.append(relative)

    def entries_left(self) -> int:
        return self.caps.max_entries - self.visited


def read_output_tree(root: Path, caps: OutputCaps) -> OutputCollection:
    """Read regular files below *root* without ever following a symlink.

    The sandboxed code controls this tree and may still be altering it, so
    every entry is untrusted. The walk is descriptor-relative: *root* is opened
    once, each directory is opened with ``O_NOFOLLOW`` relative to its parent
    descriptor and listed through that descriptor, so a path component swapped
    for a symlink after it was inspected is refused instead of followed. Symlinks,
    devices, FIFOs and anything beyond the caps are reported (bounded) rather
    than read. ``result.json`` is looked up first so junk entries cannot starve
    the one file the tools depend on.
    """
    walk = _Walk(caps)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        _take_file(walk, root_fd, "result.json", "result.json")
        _walk_directory(walk, root_fd, "", depth=0)
    finally:
        os.close(root_fd)
    return OutputCollection(
        files=dict(sorted(walk.files.items())),
        rejected=sorted(walk.rejected),
        rejected_total=walk.rejected_total,
        truncated=walk.truncated,
    )


def _walk_directory(walk: _Walk, dir_fd: int, prefix: str, depth: int) -> None:
    budget = walk.entries_left()
    if budget <= 0:
        walk.truncated = True
        return
    with os.scandir(dir_fd) as entries:
        # islice: never materialise more than the remaining budget of a huge directory
        names = sorted(entry.name for entry in itertools.islice(entries, budget))
        if next(entries, None) is not None:
            walk.truncated = True
    walk.visited += len(names)
    for name in names:
        relative = f"{prefix}{name}"
        if relative == "result.json" and "result.json" in walk.files:
            continue  # already taken first
        try:
            info = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        except OSError:
            walk.reject(relative)
            continue
        if stat.S_ISDIR(info.st_mode):
            if depth + 1 >= walk.caps.max_depth:
                walk.truncated = True
                walk.reject(relative)
                continue
            try:
                child_fd = os.open(
                    name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd
                )
            except OSError:  # replaced by a symlink (ELOOP) or removed since the stat
                walk.reject(relative)
                continue
            try:
                _walk_directory(walk, child_fd, f"{relative}/", depth + 1)
            finally:
                os.close(child_fd)
            continue
        _take_file(walk, dir_fd, name, relative, info)


def _take_file(
    walk: _Walk,
    dir_fd: int,
    name: str,
    relative: str,
    info: os.stat_result | None = None,
) -> None:
    """Read one entry of ``dir_fd`` if it is a regular file within the caps."""
    if relative in walk.files:
        return
    if info is None:
        try:
            info = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
        except OSError:
            return  # absent: nothing to report for the optional priority lookup
        walk.visited += 1
    caps = walk.caps
    if not stat.S_ISREG(info.st_mode):
        walk.reject(relative)
        return
    if (
        len(walk.files) >= caps.max_files
        or info.st_size > caps.max_file_bytes
        or walk.total_bytes + info.st_size > caps.max_total_bytes
    ):
        walk.reject(relative)
        return
    content = _read_regular_file(dir_fd, name, caps.max_file_bytes)
    if content is None:
        walk.reject(relative)
        return
    walk.total_bytes += len(content)
    walk.files[relative] = content


def _read_regular_file(dir_fd: int, name: str, limit: int) -> bytes | None:
    """Open relative to the directory with ``O_NOFOLLOW`` and re-check the type."""
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd)
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
    container_gone: bool = False  # removed (and verified) before the normal cleanup


async def _wait_done(task: asyncio.Future[Any]) -> bool:
    """Wait for *task* to finish even if the caller is cancelled meanwhile.

    ``asyncio.to_thread`` cannot stop the thread when its await is cancelled, so
    a Docker create may complete after the caller gave up. Callers that must not
    orphan what it creates wait here, collect the result, and re-raise the
    cancellation afterwards. Returns whether a cancellation was absorbed.
    """
    cancelled = False
    while not task.done():
        try:
            await asyncio.wait({task})
        except asyncio.CancelledError:
            cancelled = True
    return cancelled


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

    @staticmethod
    async def _create_tracked(
        resources: _RunResources, slot: str, factory: Callable[..., Any], /, **kwargs: Any
    ) -> Any:
        """Run a Docker create so that its result is registered even if we are cancelled.

        The resource is stored in ``resources`` before the cancellation (if any)
        propagates, so the ``finally`` cleanup removes it before the concurrency
        gate opens again.
        """
        task = asyncio.ensure_future(asyncio.to_thread(factory, **kwargs))
        cancelled = await _wait_done(task)
        if not task.cancelled() and task.exception() is None:
            setattr(resources, slot, task.result())
        if cancelled:
            raise asyncio.CancelledError
        return task.result()

    @staticmethod
    def _require_ipv4_only(network: Network) -> None:
        """Fail closed unless the network that actually exists carries no IPv6.

        ``inhibit_ipv4`` removes the host's IPv4 address from the bridge but says
        nothing about IPv6, and daemon defaults (``default-network-opts``) can turn
        it on. The SDK omits ``enable_ipv6=False`` from the request, so the
        inspected state is the only evidence: anything but an explicit
        ``EnableIPv6 == False`` (or any IPv6 subnet) aborts the run.
        """
        attrs = network.attrs or {}
        configs = (attrs.get("IPAM") or {}).get("Config") or []
        has_ipv6_subnet = any(":" in str(config.get("Subnet", "")) for config in configs)
        if attrs.get("EnableIPv6") is not False or has_ipv6_subnet:
            msg = "Sandbox egress network may carry IPv6 (enabled or unknown); refusing to run"
            raise SandboxError(msg)

    async def _create_egress(
        self, target: EgressTarget, run_id: str, resources: _RunResources
    ) -> Network:
        """Create the sandbox-only network and the forwarder pinned to ``target``."""
        alias = alias_for(target)
        network: Network = await self._create_tracked(
            resources,
            "network",
            self._client.networks.create,
            name=f"fraudai-sbx-{run_id}",
            driver="bridge",
            internal=True,
            options={
                "com.docker.network.bridge.inhibit_ipv4": "true",
                "com.docker.network.enable_ipv6": "false",
            },
            labels={_RESOURCE_LABEL: "1", f"{_RESOURCE_LABEL}.run": run_id},
        )
        await asyncio.to_thread(network.reload)
        self._require_ipv4_only(network)
        proxy: Container = await self._create_tracked(
            resources,
            "proxy",
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
        # The configured timeout is a ceiling: callers may ask for less, never more.
        timeout = min(timeout, self._timeout)
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
                # The embedded resolver answers the forwarder's alias itself but forwards
                # every other name to whatever upstream is reachable. Pin that upstream to
                # an inert address so name resolution never depends on the daemon's DNS.
                network_args: dict[str, Any] = {"network": network.name, "dns": ["127.0.0.1"]}
            else:
                network_args = {"network_mode": "none"}

            max_file = self._output_caps.max_file_bytes
            container = await self._create_tracked(
                resources,
                "container",
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
                writer_state = "stopped"
            except Exception:
                # Timeout or unexpected error: the writer may still be alive.
                timed_out = True
                exit_code = -1
                logger.warning(
                    "Sandbox container %s timed out after %ds, stopping",
                    container.short_id,
                    timeout,
                )
                writer_state = await self._ensure_stopped(container, resources)

            duration = time.monotonic() - start_time

            if writer_state == "unconfirmed":
                # Never read outputs a live sandbox could still be rewriting.
                msg = "Sandbox container could not be confirmed stopped; outputs were discarded"
                raise SandboxError(msg)

            if writer_state == "removed":
                stdout_bytes = stderr_bytes = b""  # logs left with the container
            else:
                stdout_bytes = await self._read_logs(container, stdout=True)
                stderr_bytes = await self._read_logs(container, stdout=False)

            collection = await asyncio.to_thread(read_output_tree, output_host, self._output_caps)
            collected_files, rejected = collection[0], collection[1]
            if collection[2] or collection[3]:
                logger.warning(
                    "Sandbox container %s: %d output entrie(s) refused (truncated=%s)",
                    container.short_id,
                    collection[2],
                    collection[3],
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
                rejected_total=collection[2],
                outputs_truncated=collection[3],
                duration_seconds=round(duration, 3),
                timed_out=timed_out,
            )

        finally:
            # Cleanup must finish even if this task is being cancelled, and it
            # finishes before the concurrency gate is released by our caller.
            cancelled = await _wait_done(asyncio.ensure_future(self._cleanup(resources)))
            try:
                tmpdir_obj.cleanup()
            except Exception:
                logger.debug("Tmpdir cleanup failed for %s", tmpdir)
            if cancelled:
                raise asyncio.CancelledError

    async def _container_is_stopped(self, container: Container) -> bool:
        """True when Docker reports the container exited, dead or already removed."""
        try:
            await asyncio.to_thread(container.reload)
        except docker.errors.NotFound:
            return True
        except Exception:
            logger.exception("Could not inspect sandbox container %s", container.short_id)
            return False
        return bool(container.status in ("exited", "dead"))

    async def _ensure_stopped(self, container: Container, resources: _RunResources) -> str:
        """Make sure the sandbox can no longer write; return how that was established.

        Graceful stop, then kill, then forced removal; each step is verified with
        an inspect. ``"stopped"``: exited; ``"removed"``: had to be force-removed
        and is gone; ``"unconfirmed"``: still possibly alive (outputs must not be read).
        """
        for step, action in (
            ("stop", lambda: container.stop(timeout=5)),
            ("kill", container.kill),
        ):
            try:
                await asyncio.to_thread(action)
            except Exception:
                logger.warning("Sandbox container %s: %s failed", container.short_id, step)
            if await self._container_is_stopped(container):
                return "stopped"
        try:
            await asyncio.to_thread(container.remove, force=True)
        except Exception:
            logger.exception("Forced removal of sandbox container %s failed", container.short_id)
        if await self._container_is_stopped(container):
            resources.container_gone = True
            return "removed"
        return "unconfirmed"

    async def _read_logs(self, container: Container, *, stdout: bool) -> bytes:
        data: bytes = await asyncio.to_thread(container.logs, stdout=stdout, stderr=not stdout)
        return data[-self.MAX_LOG_BYTES :]

    @staticmethod
    async def _cleanup(resources: _RunResources) -> None:
        """Remove only the resources this run created, sandbox first."""
        for victim in (resources.container, resources.proxy):
            if victim is None or (victim is resources.container and resources.container_gone):
                continue
            try:
                await asyncio.to_thread(victim.remove, force=True)
                logger.debug("Sandbox container %s removed", victim.short_id)
            except docker.errors.NotFound:
                logger.debug("Sandbox container %s was already gone", victim.short_id)
            except Exception:
                logger.exception("Failed to remove sandbox container %s", victim.short_id)
        if resources.network is not None:
            try:
                await asyncio.to_thread(resources.network.remove)
            except Exception:
                logger.exception("Failed to remove sandbox network %s", resources.network.name)
