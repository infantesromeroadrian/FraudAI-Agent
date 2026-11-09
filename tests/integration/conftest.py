"""Fixtures for the Docker integration battery (F3.5).

Gate: ``FRAUDAI_DOCKER_TESTS=1`` makes the battery mandatory -- an unreachable
daemon or a missing sandbox image is a hard failure, never a skip. Without the
flag every test here is skipped with an explicit reason (opt-in, because the
tests create containers and networks on the machine running them).

The sandbox image comes from ``FRAUDAI_SANDBOX_IMAGE`` (default: the engine's
own default tag). Nothing is pulled or published by the tests.
"""

from __future__ import annotations

import platform
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import docker
import docker.errors
import docker.types
import pytest

from fraudai.tools.sandbox import SandboxEngine
from tests.integration.harness import (
    IMDS_ADDRESS,
    IMDS_CONTROL_RANGE,
    IMDS_GATEWAY,
    IMDS_PORT,
    IMDS_SUBNET,
    IT_LABEL,
    PRIVATE_PORT,
    TARGET_ALIAS,
    TARGET_PORT,
    UNROUTED_TEST_NET,
    Counters,
    NetworkEvidence,
    World,
    default_bridge_gateway,
    docker_tests_required,
    parse_probe,
    probe_targets,
    render_net_probe,
    run_control,
    run_without_leaks,
    sandbox_image,
    sandbox_resources,
    start_host_namespace_service,
    start_service,
    validated_target,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

HERE = Path(__file__).parent
SKIP_REASON = (
    "Docker integration battery is opt-in: set FRAUDAI_DOCKER_TESTS=1 "
    "(mandatory in CI; missing Docker or image then fails instead of skipping)"
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # This hook sees every collected item, so only touch the ones under this directory.
    for item in items:
        if not Path(str(item.fspath)).is_relative_to(HERE):
            continue
        item.add_marker(pytest.mark.docker)
        if not docker_tests_required():
            item.add_marker(pytest.mark.skip(reason=SKIP_REASON))


def pytest_report_header(config: pytest.Config) -> list[str]:
    """Record the Docker/OS versions in every log, Linux CI included."""
    if not docker_tests_required():
        return []
    try:
        client = docker.from_env()
        version = client.version()
        image = client.images.get(sandbox_image())
        server = f"{version.get('Version')} {version.get('Os')}/{version.get('Arch')}"
        return [
            f"docker: server={server} kernel={version.get('KernelVersion')} "
            f"api={version.get('ApiVersion')}",
            f"host: {platform.system()} {platform.release()} {platform.machine()}",
            f"sandbox image: {sandbox_image()} id={image.id}",
        ]
    except Exception as exc:  # reported, not raised: the fixtures fail the tests
        return [f"docker: UNAVAILABLE ({type(exc).__name__}: {exc})"]


@dataclass(frozen=True, slots=True)
class DockerEnv:
    client: docker.DockerClient
    image: str
    session: str


def _remove_session_resources(client: docker.DockerClient, session: str) -> None:
    selector = {"label": f"{IT_LABEL}={session}"}
    for container in client.containers.list(all=True, filters=selector):
        container.remove(force=True)
    for network in client.networks.list(filters=selector):
        network.remove()


@pytest.fixture(scope="session")
def docker_env() -> Iterator[DockerEnv]:
    image = sandbox_image()
    try:
        client = docker.from_env()
        client.ping()
    except Exception as exc:
        pytest.fail(f"Docker daemon unreachable ({type(exc).__name__}: {exc})", pytrace=False)
    try:
        client.images.get(image)
    except docker.errors.ImageNotFound:
        pytest.fail(f"Sandbox image {image!r} not found; build it first", pytrace=False)
    session = secrets.token_hex(4)
    try:
        yield DockerEnv(client=client, image=image, session=session)
    finally:
        _remove_session_resources(client, session)
        leftovers = client.containers.list(all=True, filters={"label": f"{IT_LABEL}={session}"})
        assert not leftovers, f"test containers leaked: {[c.name for c in leftovers]}"


@pytest.fixture(scope="session")
def engine(docker_env: DockerEnv) -> SandboxEngine:
    return SandboxEngine(docker_client=docker_env.client, image=docker_env.image, timeout=120)


@pytest.fixture()
def no_leaks(docker_env: DockerEnv) -> Iterator[None]:
    """Fail the test if it leaves sandbox containers/networks behind (also on failure)."""
    before = sandbox_resources(docker_env.client)
    yield
    leaked = sandbox_resources(docker_env.client) - before
    assert not leaked, f"sandbox resources leaked: {sorted(leaked)}"


@pytest.fixture(scope="session")
def world(docker_env: DockerEnv) -> World:
    client, image, session = docker_env.client, docker_env.image, docker_env.session
    imds_network = f"fraudai-it-{session}-imds"
    ipam = docker.types.IPAMConfig(
        pool_configs=[
            docker.types.IPAMPool(
                subnet=IMDS_SUBNET,
                iprange=IMDS_CONTROL_RANGE,  # controls take .76; the service is pinned to .77
                gateway=IMDS_GATEWAY,
            )
        ]
    )
    client.networks.create(imds_network, driver="bridge", ipam=ipam, labels={IT_LABEL: session})
    imds = start_service(
        client,
        image,
        session=session,
        port=IMDS_PORT,
        network=imds_network,
        low_ports=True,
        ipv4_address=IMDS_ADDRESS,
    )
    assert imds.address(imds_network) == IMDS_ADDRESS
    private = start_service(client, image, session=session, port=PRIVATE_PORT)
    target = start_service(client, image, session=session, port=TARGET_PORT)
    redirect_target = start_service(
        client,
        image,
        session=session,
        port=TARGET_PORT,
        mode="redirect",
        location=f"http://{private.address()}:{PRIVATE_PORT}/internal",
    )
    return World(
        target=target,
        redirect_target=redirect_target,
        private=private,
        host_service=start_host_namespace_service(client, image, session=session),
        imds=imds,
        imds_network=imds_network,
        bridge_gateway=default_bridge_gateway(client),
    )


@pytest.fixture(scope="session")
def network_evidence(  # noqa: PLR0915 - one linear scenario, split would hide the ordering
    docker_env: DockerEnv, engine: SandboxEngine, world: World
) -> NetworkEvidence:
    client, image, session = docker_env.client, docker_env.image, docker_env.session
    host_port = world.host_service.port
    before_controls = Counters.take(world)

    # --- Phase 1: controls on paths that ARE allowed to reach each service.
    bridge_code = render_net_probe(probe_targets(world))
    control_bridge = parse_probe(
        run_control(client, image, bridge_code, session=session, network="bridge")
    )
    imds_code = render_net_probe([("imds-sim", IMDS_ADDRESS, IMDS_PORT)])
    control_imds = parse_probe(
        run_control(client, image, imds_code, session=session, network=world.imds_network)
    )
    # A naive `--internal` network (no inhibit_ipv4): its gateway reaches host services.
    # This is exactly the gap the production topology closes, so it must connect here.
    naive = client.networks.create(
        f"fraudai-it-{session}-naive", driver="bridge", internal=True, labels={IT_LABEL: session}
    )
    try:
        control_naive = parse_probe(
            run_control(
                client,
                image,
                render_net_probe([], own_gateway_ports=[host_port]),
                session=session,
                network=naive.name,
            )
        )
    finally:
        naive.remove()
    # DNS control: on a shared user-defined network a container name resolves.
    dns_net = client.networks.create(f"fraudai-it-{session}-dns", labels={IT_LABEL: session})
    try:
        dns_net.connect(world.private.container)
        control_dns = parse_probe(
            run_control(
                client,
                image,
                render_net_probe([], names=[world.private.name]),
                session=session,
                network=dns_net.name,
            )
        )
        dns_net.disconnect(world.private.container)
    finally:
        dns_net.remove()
    after_controls = Counters.take(world)

    # --- Phase 2: sandbox with NO network (the default).
    none_targets = [
        *probe_targets(world),
        ("imds-sim", IMDS_ADDRESS, IMDS_PORT),
        ("via-alias", TARGET_ALIAS, TARGET_PORT),
    ]
    none_code = render_net_probe(
        none_targets, names=[TARGET_ALIAS, world.private.name], own_gateway_ports=[host_port]
    )
    sandbox_none = run_without_leaks(client, engine.execute_python(none_code, timeout=90))
    assert sandbox_none.exit_code == 0, sandbox_none.stderr
    after_none = Counters.take(world)

    # --- Phase 3: sandbox with the egress forwarder pinned to the synthetic target.
    egress_targets = [
        ("via-alias", TARGET_ALIAS, TARGET_PORT),
        ("alias-other-port", TARGET_ALIAS, TARGET_PORT + 1),
        *probe_targets(world),
        ("imds-sim", IMDS_ADDRESS, IMDS_PORT),
        ("unrouted-test-net", UNROUTED_TEST_NET, 80),
    ]
    egress_code = render_net_probe(
        egress_targets, names=[TARGET_ALIAS, world.private.name], own_gateway_ports=[host_port]
    )
    sandbox_egress = run_without_leaks(
        client,
        engine.execute_python(
            egress_code, timeout=90, egress=validated_target(world.target.address())
        ),
    )
    assert sandbox_egress.exit_code == 0, sandbox_egress.stderr
    after_egress = Counters.take(world)

    return NetworkEvidence(
        control_bridge=control_bridge,
        control_imds=control_imds,
        control_naive_internal=control_naive,
        control_dns=control_dns,
        sandbox_none=parse_probe(sandbox_none.stdout),
        sandbox_egress=parse_probe(sandbox_egress.stdout),
        before_controls=before_controls,
        after_controls=after_controls,
        after_none=after_none,
        after_egress=after_egress,
    )
