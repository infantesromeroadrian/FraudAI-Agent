"""Helpers for the Docker integration battery (F3.5, ADR-007).

Everything here talks to a real Docker daemon. Synthetic services are plain
containers we own and label (``fraudai.it``); they log every connection so a
test can prove, from the *service side*, whether traffic arrived. A negative
result is only meaningful next to a positive control: the same probe, run from
a path that is allowed to reach the service, must produce hits first.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import docker
import docker.errors

from fraudai.tools.egress import EgressPolicy, EgressTarget
from fraudai.tools.sandbox import SandboxEngine, SandboxResult

if TYPE_CHECKING:
    from collections.abc import Coroutine

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
) -> SyntheticService:
    """Start a synthetic service and wait until it logs READY."""
    command = ["python", "-c", SERVICE_SOURCE, str(port), mode, location]
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


def run_control(
    client: docker.DockerClient, image: str, code: str, *, session: str, network: str
) -> str:
    """Run *code* in a plain (non-sandbox) container on *network*; return stdout."""
    container = client.containers.run(
        image,
        ["python", "-c", code],
        network=network,
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
