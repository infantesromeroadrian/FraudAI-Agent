"""F3 correction round -- DNS upstream and effective IPv4-only network, on real Docker.

DNS. Docker's embedded resolver answers the forwarder's alias itself but forwards
every other name to a configured upstream, *even on an ``internal`` network*. That
is only an exposure when an upstream is both reachable and configured; the engine
pins ``dns=["127.0.0.1"]``. The evidence is therefore a pair measured on our own
synthetic upstream (UDP/53, counts every query it receives):

* control: a client configured with the upstream queries it (plain network, and
  also an ``internal`` network -- the finding);
* treatment: the engine's sandbox, with that upstream *reachable on its own
  network*, sends it zero queries; and the alias still resolves to the forwarder
  and reaches the authorised target.

A mutation of the pin (offering the upstream to the sandbox) must bring the query
back, otherwise a zero would prove nothing. No real DNS name or server is used.

IPv6. The effective network is inspected at the moment the sandbox is created,
and adverse or unknown IPv6 states must abort before any container exists.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest import mock

import docker.models.networks
import pytest
from docker.types import IPAMConfig, IPAMPool

from fraudai.tools.sandbox import SandboxEngine, SandboxError
from tests.integration.harness import (
    DNS_SERVICE_SOURCE,
    IT_LABEL,
    TARGET_ALIAS,
    TARGET_PORT,
    LifecycleProbe,
    SyntheticService,
    dns_upstream_in_engine_network,
    parse_probe,
    run_async,
    run_control,
    run_without_leaks,
    start_service,
    validated_target,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tests.integration.conftest import DockerEnv
    from tests.integration.harness import World

PROBE_NAME = "probe.f35synthetic.test"
HOST_HEADER = f"{TARGET_ALIAS}:{TARGET_PORT}"

RESOLVE_CODE = f"""\
import socket
try:
    print("RESOLVED", socket.gethostbyname({PROBE_NAME!r}))
except OSError as exc:
    print("FAILED", type(exc).__name__)
"""

# Runs inside the engine's sandbox: the probe name must not resolve, the alias must,
# and the authorised target must still be reachable through it.
SANDBOX_DNS_CODE = f"""\
import json, socket, urllib.request

def resolve(name):
    try:
        return socket.gethostbyname(name)
    except OSError:
        return None

report = {{"probe_name": resolve({PROBE_NAME!r}), "alias": resolve({TARGET_ALIAS!r})}}
try:
    with urllib.request.urlopen("http://{HOST_HEADER}/dns-check", timeout=10) as reply:
        report["target_status"] = reply.status
except OSError as exc:
    report["target_status"] = type(exc).__name__
print("PROBE:" + json.dumps(report))
"""

SANDBOX_IPV6_CODE = """\
import json, socket

global_addresses = []
try:
    for line in open("/proc/net/if_inet6"):
        fields = line.split()
        if fields and fields[3] == "00":  # scope 0 = global
            global_addresses.append(fields[0])
except OSError:
    pass
try:
    sock = socket.socket(socket.AF_INET6)
    sock.settimeout(3)
    sock.connect(("2001:db8::1", 80))  # RFC 3849 documentation prefix
    v6_connect = "connected"
except OSError as exc:
    v6_connect = exc.errno
print("PROBE:" + json.dumps({"global": global_addresses, "v6_connect": v6_connect}))
"""


def _synthetic_dns(env: DockerEnv, **placement: Any) -> SyntheticService:
    return start_service(
        env.client,
        env.image,
        session=env.session,
        port=53,
        source=DNS_SERVICE_SOURCE,
        low_ports=True,
        **placement,
    )


@pytest.fixture()
def dns_server(docker_env: DockerEnv) -> Iterator[SyntheticService]:
    service = _synthetic_dns(docker_env)
    try:
        yield service
    finally:
        service.container.remove(force=True)


def _client_resolution(env: DockerEnv, network: str, upstream_ip: str) -> str:
    output = run_control(
        env.client,
        env.image,
        RESOLVE_CODE,
        session=env.session,
        network=network,
        dns=[upstream_ip],
    )
    return output.strip()


# --- DNS ------------------------------------------------------------------------------


def test_control_client_with_a_configured_upstream_queries_it(docker_env: DockerEnv) -> None:
    network = docker_env.client.networks.create(
        f"fraudai-it-{docker_env.session}-dns-plain", labels={IT_LABEL: docker_env.session}
    )
    server = _synthetic_dns(docker_env, network=network.name)
    try:
        outcome = _client_resolution(docker_env, network.name, server.address(network.name))

        assert outcome == "RESOLVED 203.0.113.9"
        assert server.queries(PROBE_NAME) >= 1
    finally:
        server.container.remove(force=True)
        network.remove()


def test_internal_network_still_forwards_to_a_reachable_configured_upstream(
    docker_env: DockerEnv,
) -> None:
    """The finding: ``internal`` + ``inhibit_ipv4`` does not stop the embedded resolver."""
    network = docker_env.client.networks.create(
        f"fraudai-it-{docker_env.session}-dns-internal",
        internal=True,
        options={"com.docker.network.bridge.inhibit_ipv4": "true"},
        labels={IT_LABEL: docker_env.session},
    )
    server = _synthetic_dns(docker_env, network=network.name)
    try:
        outcome = _client_resolution(docker_env, network.name, server.address(network.name))

        assert outcome == "RESOLVED 203.0.113.9"
        assert server.queries(PROBE_NAME) >= 1
    finally:
        server.container.remove(force=True)
        network.remove()


def test_engine_sandbox_sends_no_query_to_an_upstream_reachable_on_its_network(
    docker_env: DockerEnv, engine: SandboxEngine, world: World, dns_server: SyntheticService
) -> None:
    target_before = world.target.hits(HOST_HEADER)
    egress = validated_target(world.target.address())
    with (
        LifecycleProbe(docker_env.client) as probe,
        dns_upstream_in_engine_network(probe, dns_server, offer=False),
    ):
        result = run_without_leaks(
            docker_env.client, engine.execute_python(SANDBOX_DNS_CODE, egress=egress)
        )

    assert result.exit_code == 0, result.stderr
    report = parse_probe(result.stdout)
    assert report["probe_name"] is None
    assert dns_server.queries() == 0, "the sandbox leaked a DNS query to the reachable upstream"
    # the authorised path keeps working: the alias resolves locally to the forwarder ...
    assert report["alias"] not in (None, world.target.address(), "203.0.113.9")
    # ... and the target answered through it
    assert report["target_status"] == 200
    assert world.target.hits(HOST_HEADER) == target_before + 1


def test_control_offering_the_upstream_to_the_engine_sandbox_brings_the_query_back(
    docker_env: DockerEnv, engine: SandboxEngine, world: World, dns_server: SyntheticService
) -> None:
    """Mutation as control: without the 127.0.0.1 pin an offered, reachable upstream is used."""
    egress = validated_target(world.target.address())
    with (
        LifecycleProbe(docker_env.client) as probe,
        dns_upstream_in_engine_network(probe, dns_server, offer=True),
    ):
        result = run_without_leaks(
            docker_env.client, engine.execute_python(SANDBOX_DNS_CODE, egress=egress)
        )

    assert result.exit_code == 0, result.stderr
    report = parse_probe(result.stdout)
    assert report["probe_name"] == "203.0.113.9"
    assert dns_server.queries(PROBE_NAME) >= 1


# --- IPv6 -----------------------------------------------------------------------------


def test_engine_network_is_effectively_ipv4_only(
    docker_env: DockerEnv, engine: SandboxEngine, world: World
) -> None:
    effective: dict[str, Any] = {}

    def observe(slot: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        if slot == "sandbox":
            network = docker_env.client.networks.get(kwargs["network"])
            network.reload()
            effective["EnableIPv6"] = network.attrs.get("EnableIPv6")
            effective["Internal"] = network.attrs.get("Internal")
            effective["Subnets"] = [
                str(c.get("Subnet")) for c in (network.attrs.get("IPAM") or {}).get("Config") or []
            ]
        return kwargs

    with LifecycleProbe(docker_env.client) as probe:
        probe.on_container_create = observe
        result = run_without_leaks(
            docker_env.client,
            engine.execute_python(
                SANDBOX_IPV6_CODE, egress=validated_target(world.target.address())
            ),
        )

    assert result.exit_code == 0, result.stderr
    assert effective["EnableIPv6"] is False
    assert effective["Internal"] is True
    assert effective["Subnets"], "the network must report its subnets"
    assert all(":" not in subnet for subnet in effective["Subnets"]), effective
    inside = parse_probe(result.stdout)
    assert inside["global"] == []  # no global IPv6 address
    assert inside["v6_connect"] != "connected"


def test_network_with_ipv6_enabled_aborts_before_any_container_exists(
    docker_env: DockerEnv, engine: SandboxEngine, world: World, no_leaks: None
) -> None:
    def make_adverse(kwargs: dict[str, Any]) -> dict[str, Any]:
        # A real network the daemon accepts, but with IPv6 on: what a daemon-level
        # default (default-network-opts) could produce behind the engine's back.
        options = {**kwargs["options"], "com.docker.network.enable_ipv6": "true"}
        ipam = IPAMConfig(
            pool_configs=[IPAMPool(subnet="198.18.35.0/24"), IPAMPool(subnet="fd00:f35:0:77::/64")]
        )
        return {**kwargs, "options": options, "enable_ipv6": True, "ipam": ipam}

    with LifecycleProbe(docker_env.client) as probe:
        probe.on_network_create = make_adverse
        with pytest.raises(SandboxError, match="IPv6"):
            _run_egress_hello(engine, world)

        assert len(probe.networks) == 1  # the adverse network really was created ...
        assert probe.containers == {"sandbox": [], "proxy": []}  # ... and nothing ran in it


def test_unknown_ipv6_state_aborts_before_any_container_exists(
    docker_env: DockerEnv, engine: SandboxEngine, world: World, no_leaks: None
) -> None:
    """Fail closed when the daemon's answer does not state ``EnableIPv6`` at all."""
    real_reload = docker.models.networks.Network.reload

    def reload_without_ipv6(network: Any) -> None:
        real_reload(network)
        network.attrs.pop("EnableIPv6", None)

    with (
        LifecycleProbe(docker_env.client) as probe,
        mock.patch.object(docker.models.networks.Network, "reload", reload_without_ipv6),
    ):
        with pytest.raises(SandboxError, match="IPv6"):
            _run_egress_hello(engine, world)

        assert probe.containers == {"sandbox": [], "proxy": []}


def _run_egress_hello(engine: SandboxEngine, world: World) -> None:
    run_async(engine.execute_python("print(1)", egress=validated_target(world.target.address())))
