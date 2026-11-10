"""Helpers for the Docker integration battery (F3.5, ADR-007).

Everything here talks to a real Docker daemon. Synthetic services are plain
containers we own and label (``fraudai.it``); they log every connection so a
test can prove, from the *service side*, whether traffic arrived. A negative
result is only meaningful next to a positive control: the same probe, run from
a path that is allowed to reach the service, must produce hits first.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar
from unittest import mock

import docker
import docker.errors
import docker.models.containers
import docker.models.networks

from fraudai.tools import sandbox as sandbox_module
from fraudai.tools.egress import EgressPolicy, EgressTarget
from fraudai.tools.sandbox import SandboxEngine, SandboxResult

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Iterator

    from docker.models.containers import Container

_T = TypeVar("_T")

REQUIRE_ENV = "FRAUDAI_DOCKER_TESTS"
IMAGE_ENV = "FRAUDAI_SANDBOX_IMAGE"
IT_LABEL = "fraudai.it"
SANDBOX_LABEL = "fraudai.sandbox"
SERVICE_READY_SECONDS = 15.0
CONTROL_TIMEOUT_SECONDS = 60
TARGET_ALIAS = "synthetic-target.test"
TARGET_PORT = 8080
PRIVATE_PORT = 6333  # a stand-in for an internal Qdrant
# Simulated metadata endpoint in the link-local range. Deliberately NOT 169.254.169.254:
# CI runners are cloud VMs whose real IMDS lives there, and a local Docker network must
# not shadow it. The real address is only ever fed to the (pure) policy tests.
IMDS_ADDRESS = "169.254.77.77"
IMDS_SUBNET = "169.254.77.0/24"
IMDS_GATEWAY = "169.254.77.1"
IMDS_CONTROL_RANGE = "169.254.77.76/30"
IMDS_PORT = 80
UNROUTED_TEST_NET = "192.0.2.1"  # RFC 5737 TEST-NET-1, never routable


def docker_tests_required() -> bool:
    """True when the battery is mandatory: missing Docker/image must fail, not skip."""
    return os.environ.get(REQUIRE_ENV) == "1"


def sandbox_image() -> str:
    return os.environ.get(IMAGE_ENV) or SandboxEngine.SANDBOX_IMAGE


# Minimal TCP/HTTP service. It logs ``ACCEPT <peer>`` for every TCP connection
# and ``HIT <peer> <request line> host=<Host header>`` for every request, so the
# tests can attribute traffic by the Host label the probe chose.
SERVICE_SOURCE = """\
import socket, sys, threading

port, mode, location = int(sys.argv[1]), sys.argv[2], sys.argv[3]
BODY = b'{"prediction": 1, "response": "synthetic-ok"}'


def read_request(conn):
    data = b""
    while b"\\r\\n\\r\\n" not in data and len(data) < 8192:
        chunk = conn.recv(4096)
        if not chunk:
            return data
        data += chunk
    head, _, rest = data.partition(b"\\r\\n\\r\\n")
    length = 0
    for line in head.split(b"\\r\\n")[1:]:
        if line.lower().startswith(b"content-length:"):
            length = min(int(line.split(b":", 1)[1]), 1 << 20)
    while len(rest) < length:
        chunk = conn.recv(4096)
        if not chunk:
            break
        rest += chunk
    return head


def handle(conn, peer):
    conn.settimeout(3)
    try:
        head = read_request(conn)
        if head:
            lines = head.split(b"\\r\\n")
            host = ""
            for line in lines[1:]:
                if line.lower().startswith(b"host:"):
                    host = line[5:].strip().decode(errors="replace")
            print("HIT", peer, lines[0].decode(errors="replace"), "host=" + host, flush=True)
            if mode == "redirect":
                reply = (
                    b"HTTP/1.0 302 Found\\r\\nLocation: " + location.encode()
                    + b"\\r\\nContent-Length: 0\\r\\n\\r\\n"
                )
            else:
                reply = (
                    b"HTTP/1.0 200 OK\\r\\nContent-Type: application/json\\r\\n"
                    b"Content-Length: " + str(len(BODY)).encode() + b"\\r\\n\\r\\n" + BODY
                )
            conn.sendall(reply)
    except OSError:
        pass
    finally:
        conn.close()


server = socket.socket()
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", port))
server.listen(64)
print("READY", flush=True)
while True:
    conn, addr = server.accept()
    print("ACCEPT", addr[0], flush=True)
    threading.Thread(target=handle, args=(conn, addr[0]), daemon=True).start()
"""

# Synthetic DNS upstream: answers every A query with a documentation address and logs
# ``QUERY <name> <client>`` so a test can count what actually reached it.
DNS_SERVICE_SOURCE = """\
import socket, sys

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", int(sys.argv[1])))
print("READY", flush=True)
while True:
    data, peer = sock.recvfrom(512)
    labels, index = [], 12
    while index < len(data) and data[index]:
        labels.append(data[index + 1 : index + 1 + data[index]].decode(errors="replace"))
        index += data[index] + 1
    question = data[12 : index + 5]
    print("QUERY", ".".join(labels), peer[0], flush=True)
    head = data[:2] + b"\\x81\\x80" + data[4:6] + b"\\x00\\x01\\x00\\x00\\x00\\x00"
    pointer = b"\\xc0\\x0c\\x00\\x01\\x00\\x01\\x00\\x00\\x00\\x3c\\x00\\x04"
    sock.sendto(head + question + pointer + bytes([203, 0, 113, 9]), peer)
"""

# Connects to every target and reports what happened. Run from a control
# container (must connect) and from the sandbox (must not): same code, same
# targets, so a difference is attributable to the network topology.
NET_PROBE_SOURCE = """\
import json, os, socket

TARGETS = json.loads(__TARGETS__)
NAMES = json.loads(__NAMES__)
OWN_GATEWAY_PORTS = json.loads(__OWN_GATEWAY_PORTS__)


def attempt(label, host, port):
    try:
        conn = socket.create_connection((host, port), timeout=3)
    except OSError as exc:
        return {"connected": False, "errno": exc.errno, "error": type(exc).__name__}
    try:
        conn.settimeout(3)
        conn.sendall(("GET /probe HTTP/1.0\\r\\nHost: " + label + "\\r\\n\\r\\n").encode())
        reply = conn.recv(64)
    except OSError:
        reply = b""
    finally:
        conn.close()
    return {"connected": True, "reply": reply.decode(errors="replace")}


def routes():
    rows = []
    with open("/proc/net/route") as handle:
        next(handle)
        for line in handle:
            fields = line.split()
            rows.append({"iface": fields[0], "dest": fields[1], "mask": fields[7]})
    return rows


def own_first_host():
    for row in routes():
        if row["dest"] != "00000000" and row["iface"] != "lo":
            base = int.from_bytes(bytes.fromhex(row["dest"])[::-1], "big") + 1
            return socket.inet_ntoa(base.to_bytes(4, "big"))
    return None


def interfaces_up():
    # Default tunnel devices (ip6gre0, erspan0, ...) exist but are DOWN; only UP counts.
    names = []
    for name in sorted(os.listdir("/sys/class/net")):
        try:
            with open("/sys/class/net/%s/flags" % name) as handle:
                if int(handle.read(), 16) & 1:
                    names.append(name)
        except OSError:
            continue
    return names


def resolve(name):
    try:
        infos = socket.getaddrinfo(name, None, proto=socket.IPPROTO_TCP)
    except OSError:
        return []
    return sorted({info[4][0] for info in infos})


report = {
    "targets": {label: attempt(label, host, port) for label, host, port in TARGETS},
    "names": {name: resolve(name) for name in NAMES},
    "interfaces": interfaces_up(),
    "routes": routes(),
    "own_first_host": own_first_host(),
}
gateway = report["own_first_host"]
for port in OWN_GATEWAY_PORTS:
    key = "own-gateway:%d" % port
    if gateway is None:
        report["targets"][key] = {"connected": False, "error": "no-subnet"}
    else:
        report["targets"][key] = attempt("own-gateway", gateway, port)
print("PROBE:" + json.dumps(report))
"""


def render_net_probe(
    targets: list[tuple[str, str, int]],
    names: list[str] | None = None,
    own_gateway_ports: list[int] | None = None,
) -> str:
    """Return the probe source with its (JSON-encoded) inputs substituted."""
    return (
        NET_PROBE_SOURCE.replace("__TARGETS__", repr(json.dumps(targets)))
        .replace("__NAMES__", repr(json.dumps(names or [])))
        .replace("__OWN_GATEWAY_PORTS__", repr(json.dumps(own_gateway_ports or [])))
    )


def parse_probe(stdout: str) -> dict[str, object]:
    for line in stdout.splitlines():
        if line.startswith("PROBE:"):
            parsed: dict[str, object] = json.loads(line[len("PROBE:") :])
            return parsed
    msg = f"probe printed no PROBE line; stdout was: {stdout[-500:]!r}"
    raise AssertionError(msg)


@dataclass(frozen=True, slots=True)
class SyntheticService:
    """A container we own that logs every connection it receives."""

    container: Container
    port: int

    @property
    def name(self) -> str:
        return str(self.container.name)

    def address(self, network: str = "bridge") -> str:
        self.container.reload()
        networks = self.container.attrs["NetworkSettings"]["Networks"]
        return str(networks[network]["IPAddress"])

    def log_lines(self) -> list[str]:
        raw: bytes = self.container.logs(stdout=True, stderr=True)
        return raw.decode(errors="replace").splitlines()

    def accepts(self) -> int:
        """TCP connections accepted so far (a probe's mere connect counts)."""
        return sum(1 for line in self.log_lines() if line.startswith("ACCEPT"))

    def queries(self, name: str | None = None) -> int:
        """DNS queries received so far (DNS service only), optionally for one name."""
        return sum(
            1
            for line in self.log_lines()
            if line.startswith("QUERY") and (name is None or line.split()[1] == name)
        )

    def hits(self, label: str | None = None) -> int:
        """HTTP requests received so far, optionally only those with ``Host: label``."""
        return sum(
            1
            for line in self.log_lines()
            if line.startswith("HIT") and (label is None or line.endswith(f"host={label}"))
        )


def start_service(  # noqa: PLR0913 - explicit fixture knobs, all keyword-only
    client: docker.DockerClient,
    image: str,
    *,
    session: str,
    port: int,
    mode: str = "ok",
    location: str = "-",
    network: str | None = None,
    network_mode: str | None = None,
    low_ports: bool = False,
    ipv4_address: str | None = None,
    source: str = SERVICE_SOURCE,
) -> SyntheticService:
    """Start a synthetic service and wait until it logs READY."""
    command = ["python", "-c", source, str(port), mode, location]
    name = f"fraudai-it-{session}-{secrets.token_hex(3)}"
    environment = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    sysctls = {"net.ipv4.ip_unprivileged_port_start": "0"} if low_ports else None
    if ipv4_address is not None:
        # The high-level API cannot pin an address; the low-level one can.
        assert network is not None
        created = client.api.create_container(
            image,
            command,
            name=name,
            labels={IT_LABEL: session},
            environment=environment,
            host_config=client.api.create_host_config(
                read_only=True,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                mem_limit="128m",
                pids_limit=64,
                network_mode=network,
                sysctls=sysctls,
            ),
            networking_config=client.api.create_networking_config(
                {network: client.api.create_endpoint_config(ipv4_address=ipv4_address)}
            ),
        )
        client.api.start(created["Id"])
        container = client.containers.get(created["Id"])
    else:
        options: dict[str, object] = {}
        if network_mode is not None:
            options["network_mode"] = network_mode
        elif network is not None:
            options["network"] = network
        if sysctls is not None:
            options["sysctls"] = sysctls
        container = client.containers.run(
            image,
            command,
            name=name,
            detach=True,
            labels={IT_LABEL: session},
            read_only=True,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges"],
            mem_limit="128m",
            pids_limit=64,
            environment=environment,
            **options,  # type: ignore[arg-type]
        )
    deadline = time.monotonic() + SERVICE_READY_SECONDS
    while time.monotonic() < deadline:
        container.reload()
        logs: bytes = container.logs()
        if b"READY" in logs:
            return SyntheticService(container=container, port=port)
        if container.status not in ("created", "running"):
            break
        time.sleep(0.1)
    logs = container.logs()
    container.remove(force=True)
    msg = f"synthetic service on port {port} did not become ready: {logs[-400:]!r}"
    raise RuntimeError(msg)


def start_host_namespace_service(
    client: docker.DockerClient, image: str, *, session: str
) -> SyntheticService:
    """Service listening in the Docker host's network namespace.

    On Linux that is the real host; on Docker Desktop it is the Linux VM, not
    macOS. Either way it is the "host port" a sandbox must not be able to reach.
    Retries on a port clash instead of assuming a free port.
    """
    last_error: Exception | None = None
    for _ in range(8):
        port = 20000 + secrets.randbelow(25000)
        try:
            return start_service(client, image, session=session, port=port, network_mode="host")
        except (RuntimeError, docker.errors.APIError) as exc:
            last_error = exc
    msg = "could not start a host-namespace synthetic service"
    raise RuntimeError(msg) from last_error


def run_control(  # noqa: PLR0913 - explicit keyword-only knobs
    client: docker.DockerClient,
    image: str,
    code: str,
    *,
    session: str,
    network: str,
    dns: list[str] | None = None,
) -> str:
    """Run *code* in a plain (non-sandbox) container on *network*; return stdout."""
    container = client.containers.run(
        image,
        ["python", "-c", code],
        network=network,
        dns=dns,
        detach=True,
        labels={IT_LABEL: session},
        read_only=True,
        cap_drop=["ALL"],
        security_opt=["no-new-privileges"],
        mem_limit="128m",
        pids_limit=64,
    )
    try:
        status = container.wait(timeout=CONTROL_TIMEOUT_SECONDS)
        stdout: bytes = container.logs(stdout=True, stderr=False)
        if status.get("StatusCode") != 0:
            stderr: bytes = container.logs(stdout=False, stderr=True)
            msg = f"control container failed: {stderr[-400:]!r}"
            raise RuntimeError(msg)
        return stdout.decode(errors="replace")
    finally:
        container.remove(force=True)


def sandbox_resources(client: docker.DockerClient) -> frozenset[str]:
    """Containers and networks the engine labelled as its own, right now."""
    selector = {"label": f"{SANDBOX_LABEL}=1"}
    containers = {f"container:{c.id}" for c in client.containers.list(all=True, filters=selector)}
    networks = {f"network:{n.id}" for n in client.networks.list(filters=selector)}
    return frozenset(containers | networks)


def default_bridge_gateway(client: docker.DockerClient) -> str:
    network = client.networks.get("bridge")
    return str(network.attrs["IPAM"]["Config"][0]["Gateway"])


def run_async(coro: Coroutine[Any, Any, _T]) -> _T:
    return asyncio.run(coro)


def run_without_leaks(
    client: docker.DockerClient, coro: Coroutine[Any, Any, SandboxResult]
) -> SandboxResult:
    """Run one engine call and require that it left no labelled resource behind."""
    before = sandbox_resources(client)
    try:
        return run_async(coro)
    finally:
        leaked = sandbox_resources(client) - before
        assert not leaked, f"sandbox resources leaked: {sorted(leaked)}"


def synthetic_policy(address: str) -> EgressPolicy:
    """Test-only policy: the hostname resolves to *address*, and only it is allowed.

    The production policy is untouched and stays strict (see the default-policy tests).
    """
    return EgressPolicy(
        resolver=lambda _host: [address], address_allowed=lambda ip: str(ip) == address
    )


def validated_target(
    address: str, port: int = TARGET_PORT, host: str = TARGET_ALIAS
) -> EgressTarget:
    """Build the pinned target through the real ``EgressPolicy.validate``."""
    return synthetic_policy(address).validate(f"http://{host}:{port}/predict")


@dataclass(frozen=True, slots=True)
class World:
    """The synthetic services every network test points at."""

    target: SyntheticService  # the only destination the egress forwarder may dial
    redirect_target: SyntheticService  # answers 302 -> private
    private: SyntheticService  # stands in for an internal service (e.g. Qdrant)
    host_service: SyntheticService  # listens in the Docker host network namespace
    imds: SyntheticService  # simulated link-local metadata endpoint (our own, empty)
    imds_network: str
    bridge_gateway: str


def probe_targets(world: World) -> list[tuple[str, str, int]]:
    return [
        ("direct-target-ip", world.target.address(), TARGET_PORT),
        ("bridge-gateway", world.bridge_gateway, world.host_service.port),
        ("private-container", world.private.address(), PRIVATE_PORT),
    ]


@dataclass(frozen=True, slots=True)
class Counters:
    """Service-side traffic counters at one instant."""

    host: int
    private: int
    imds: int
    target_by_label: dict[str, int]

    @staticmethod
    def take(world: World) -> Counters:
        labels = ("via-alias", "direct-target-ip", "alias-other-port", "probe")
        return Counters(
            host=world.host_service.accepts(),
            private=world.private.accepts(),
            imds=world.imds.accepts(),
            target_by_label={label: world.target.hits(label) for label in labels},
        )


@dataclass(frozen=True, slots=True)
class NetworkEvidence:
    """Probe reports plus service-side counters around each phase.

    Phases run in this order: controls (allowed paths; services must see
    traffic), sandbox without network, sandbox with the egress forwarder.
    """

    control_bridge: dict[str, Any]
    control_imds: dict[str, Any]
    control_naive_internal: dict[str, Any]
    control_dns: dict[str, Any]
    sandbox_none: dict[str, Any]
    sandbox_egress: dict[str, Any]
    before_controls: Counters
    after_controls: Counters
    after_none: Counters
    after_egress: Counters


# --- lifecycle instrumentation -------------------------------------------------------
BLOCK_TIMEOUT_SECONDS = 60.0
WRITER_TIMEOUT_SECONDS = 30.0
_ENGINE_SLOT_BY_ROLE = {"sandbox": "sandbox", "egress-proxy": "proxy"}


def _is_sandbox_container(container: Any) -> bool:
    name = str(container.name)
    return name.startswith("fraudai-sbx-") and name.endswith("-run")


class LifecycleProbe:
    """Pass-through instrumentation of the Docker SDK calls the engine makes.

    Nothing is faked: every call still reaches the real daemon. The probe can
    (a) hold one engine ``create`` (network, proxy or sandbox) at a deterministic
    point with events instead of sleeps, (b) make selected ``Container`` methods of
    the *sandbox* container raise, and (c) record which resources the engine
    created and what state the sandbox was in when its outputs were read.

    It patches the SDK classes, not instances: ``DockerClient.networks`` returns a
    new collection on every access, so instance patches would silently not apply.
    Every id it saw is force-removed on exit, so a failing test cannot leak.
    """

    def __init__(self, client: docker.DockerClient) -> None:
        self.client = client
        self.block_slot: str | None = None
        self.entered = threading.Event()  # the held create has started
        self.release = threading.Event()  # let the held create proceed
        self.finished = threading.Event()  # the held create returned a resource
        self.networks: list[str] = []
        self.containers: dict[str, list[str]] = {"sandbox": [], "proxy": []}
        self.fail_methods: dict[str, int] = {}  # method -> failures left (-1: always)
        self.wait_for_writer = False
        # The next N sandbox ``remove`` calls really stop the container, then raise: the
        # daemon did the work but the client saw an error (the residual CX1 scenario).
        self.remove_stops_then_fails = 0
        self.remove_calls = 0  # every ``remove`` the engine attempted on the sandbox
        self.state_after_failed_remove: str | None = None
        self.reads: list[list[str]] = []  # sandbox states at each read_output_tree call
        self.on_container_create: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None
        self.on_network_create: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        self.before_network_remove: Callable[[Any], None] | None = None
        self._patches = contextlib.ExitStack()

    # -- patches ------------------------------------------------------------------
    def __enter__(self) -> LifecycleProbe:
        containers = docker.models.containers
        networks = docker.models.networks
        original_net_create = networks.NetworkCollection.create
        original_ctr_create = containers.ContainerCollection.create
        original_net_remove = networks.Network.remove
        original_read = sandbox_module.read_output_tree

        def net_create(coll: Any, *args: Any, **kwargs: Any) -> Any:
            engine_owned = (kwargs.get("labels") or {}).get(SANDBOX_LABEL) == "1"
            held = engine_owned and self._hold("network")
            if engine_owned and self.on_network_create is not None:
                kwargs = self.on_network_create(kwargs)
            network = original_net_create(coll, *args, **kwargs)
            if engine_owned:
                self.networks.append(str(network.id))
            if held:
                self.finished.set()
            return network

        def ctr_create(coll: Any, *args: Any, **kwargs: Any) -> Any:
            role = (kwargs.get("labels") or {}).get("role")
            slot = _ENGINE_SLOT_BY_ROLE.get(str(role))
            held = slot is not None and self._hold(slot)
            if slot is not None and self.on_container_create is not None:
                kwargs = self.on_container_create(slot, kwargs)
            container = original_ctr_create(coll, *args, **kwargs)
            if slot is not None:
                self.containers[slot].append(str(container.id))
            if held:
                self.finished.set()
            return container

        def net_remove(network: Any, *args: Any, **kwargs: Any) -> Any:
            if self.before_network_remove is not None:
                self.before_network_remove(network)
            return original_net_remove(network, *args, **kwargs)

        def read_tree(root: Any, caps: Any) -> Any:
            states = []
            for container_id in self.containers["sandbox"]:
                try:
                    states.append(str(self.client.containers.get(container_id).status))
                except docker.errors.NotFound:
                    states.append("gone")
            self.reads.append(states)
            return original_read(root, caps)

        self._patches.enter_context(
            mock.patch.object(networks.NetworkCollection, "create", net_create)
        )
        self._patches.enter_context(
            mock.patch.object(containers.ContainerCollection, "create", ctr_create)
        )
        self._patches.enter_context(mock.patch.object(networks.Network, "remove", net_remove))
        self._patches.enter_context(
            mock.patch.object(sandbox_module, "read_output_tree", read_tree)
        )
        for method in ("wait", "stop", "kill", "remove"):
            self._patches.enter_context(
                mock.patch.object(
                    containers.Container,
                    method,
                    self._failing(method, getattr(containers.Container, method)),
                )
            )
        return self

    def __exit__(self, *exc: object) -> None:
        self.release.set()  # never leave a worker thread parked
        self._patches.close()
        for container_id in [*self.containers["sandbox"], *self.containers["proxy"]]:
            with contextlib.suppress(docker.errors.NotFound, docker.errors.APIError):
                self.client.containers.get(container_id).remove(force=True)
        for network_id in self.networks:
            with contextlib.suppress(docker.errors.NotFound, docker.errors.APIError):
                self.client.networks.get(network_id).remove()

    # -- helpers --------------------------------------------------------------------
    def _hold(self, slot: str) -> bool:
        if slot != self.block_slot:
            return False
        self.entered.set()
        if not self.release.wait(timeout=BLOCK_TIMEOUT_SECONDS):
            msg = f"probe was never released while holding the {slot} create"
            raise RuntimeError(msg)
        return True

    def _failing(self, method: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(container: Any, *args: Any, **kwargs: Any) -> Any:
            if method == "remove" and _is_sandbox_container(container):
                self.remove_calls += 1
                if self.remove_stops_then_fails > 0:
                    self.remove_stops_then_fails -= 1
                    self.state_after_failed_remove = self._really_stop(container)
                    msg = "injected remove failure after the container stopped"
                    raise RuntimeError(msg)
            left = self.fail_methods.get(method, 0)
            if left != 0 and _is_sandbox_container(container):
                if method == "wait" and self.wait_for_writer:
                    self._await_writer(container)
                if left > 0:
                    self.fail_methods[method] = left - 1
                msg = f"injected {method} failure"
                raise RuntimeError(msg)
            return original(container, *args, **kwargs)

        return wrapper

    def _really_stop(self, container: Any) -> str:
        """Kill the container through the low-level API (not patched) and wait until it exited.

        Returns the status the daemon reports, which proves the container still
        exists in that state. A bounded condition wait, not a sleep.
        """
        with contextlib.suppress(docker.errors.APIError):
            self.client.api.kill(container.id)
        deadline = time.monotonic() + WRITER_TIMEOUT_SECONDS
        status = "unknown"
        while time.monotonic() < deadline:
            status = str(self.client.api.inspect_container(container.id)["State"]["Status"])
            if status in ("exited", "dead"):
                break
            time.sleep(0.02)
        return status

    @staticmethod
    def _await_writer(container: Any) -> None:
        """Block until the sandbox process has written a tick (a condition, not a sleep)."""
        source = next(
            m["Source"]
            for m in container.attrs["Mounts"]
            if m["Destination"] == "/workspace/output"
        )
        tick = Path(source) / "tick.txt"
        deadline = time.monotonic() + WRITER_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if tick.exists() and tick.stat().st_size > 0:
                return
            time.sleep(0.02)
        msg = "the sandbox writer never produced output"
        raise RuntimeError(msg)


@contextlib.contextmanager
def dns_upstream_in_engine_network(
    probe: LifecycleProbe, dns: SyntheticService, *, offer: bool
) -> Iterator[None]:
    """Make *dns* reachable on the engine's per-run network.

    With ``offer=True`` the sandbox is also *configured* to use it as its upstream
    (a mutation of the engine's ``dns`` pin, used as a positive control). The DNS
    container is detached just before the engine removes the network.
    """
    attached: list[str] = []

    def on_create(slot: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        if slot != "sandbox":
            return kwargs
        network = probe.client.networks.get(kwargs["network"])
        network.connect(dns.container)
        attached.append(str(network.name))
        if offer:
            return {**kwargs, "dns": [dns.address(str(network.name))]}
        return kwargs

    def before_remove(network: Any) -> None:
        if network.name in attached:
            with contextlib.suppress(docker.errors.APIError):
                network.disconnect(dns.container, force=True)

    probe.on_container_create = on_create
    probe.before_network_remove = before_remove
    try:
        yield
    finally:
        probe.on_container_create = None
        probe.before_network_remove = None
