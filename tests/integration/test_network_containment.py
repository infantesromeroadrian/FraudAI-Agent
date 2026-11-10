"""F3.5 -- network negatives, proven against a real Docker daemon.

Every claim is a pair: a *positive control* (the same probe from a path that is
allowed to reach the synthetic service must connect and be logged by the
service) and a *negative* (from the real ``SandboxEngine`` it must not connect
and the service must log zero new connections). A service that never answered
would prove nothing, so the controls are asserted first and by themselves.

Limits: "host" is the Docker host network namespace (the Linux VM on Docker
Desktop, the runner on Linux CI); IMDS is a simulated link-local endpoint we
own (169.254.77.77), never the real 169.254.169.254; no public address or external DNS
name is contacted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tests.integration.harness import (
    IMDS_ADDRESS,
    PRIVATE_PORT,
    TARGET_ALIAS,
    TARGET_PORT,
)

if TYPE_CHECKING:
    from tests.integration.harness import NetworkEvidence, World


def _outcome(report: dict[str, Any], label: str) -> dict[str, Any]:
    outcome: dict[str, Any] = report["targets"][label]
    return outcome


def test_positive_controls_reach_every_synthetic_service(
    network_evidence: NetworkEvidence, world: World
) -> None:
    """The services are alive and reachable from allowed paths; this validates every negative."""
    ev = network_evidence
    for label in ("direct-target-ip", "bridge-gateway", "private-container"):
        assert _outcome(ev.control_bridge, label)["connected"], label
    assert _outcome(ev.control_imds, "imds-sim")["connected"]
    # naive `--internal` network: the bridge gateway DOES reach host services
    assert _outcome(ev.control_naive_internal, f"own-gateway:{world.host_service.port}")[
        "connected"
    ]
    # ... and the services themselves recorded those connections
    assert ev.after_controls.host > ev.before_controls.host
    assert ev.after_controls.private > ev.before_controls.private
    assert ev.after_controls.imds > ev.before_controls.imds
    assert (
        ev.after_controls.target_by_label["direct-target-ip"]
        > ev.before_controls.target_by_label["direct-target-ip"]
    )


def test_sandbox_has_no_network_by_default(network_evidence: NetworkEvidence) -> None:
    ev = network_evidence
    report = ev.sandbox_none

    assert report["interfaces"] == ["lo"]
    assert all(not outcome["connected"] for outcome in report["targets"].values()), report[
        "targets"
    ]
    assert report["names"][TARGET_ALIAS] == []
    # service side: zero new connections of any kind during the no-network run
    assert ev.after_none.host == ev.after_controls.host
    assert ev.after_none.private == ev.after_controls.private
    assert ev.after_none.imds == ev.after_controls.imds
    assert ev.after_none.target_by_label == ev.after_controls.target_by_label


def test_egress_reaches_only_the_validated_target_through_the_forwarder(
    network_evidence: NetworkEvidence,
) -> None:
    ev = network_evidence
    via_alias = _outcome(ev.sandbox_egress, "via-alias")

    assert via_alias["connected"], via_alias  # positive: the allowed path works
    assert via_alias["reply"].startswith("HTTP/1.0 200")
    assert ev.after_egress.target_by_label["via-alias"] == (
        ev.after_none.target_by_label["via-alias"] + 1
    )
    # the target's real address is not routable from the sandbox ...
    assert not _outcome(ev.sandbox_egress, "direct-target-ip")["connected"]
    assert (
        ev.after_egress.target_by_label["direct-target-ip"]
        == ev.after_controls.target_by_label["direct-target-ip"]
    )
    # ... and the forwarder exposes only the pinned port
    assert not _outcome(ev.sandbox_egress, "alias-other-port")["connected"]


def test_egress_sandbox_is_attached_only_to_the_internal_network(
    network_evidence: NetworkEvidence,
) -> None:
    report = network_evidence.sandbox_egress

    assert report["interfaces"] == ["eth0", "lo"]
    default_routes = [row for row in report["routes"] if row["dest"] == "00000000"]
    assert default_routes == [], "sandbox network must have no default route"
    # the alias resolves to the forwarder on the internal network, not to the target
    [forwarder_ip] = report["names"][TARGET_ALIAS]
    assert forwarder_ip != network_evidence.control_bridge["names"].get(TARGET_ALIAS)
    assert report["own_first_host"] is not None


def test_egress_blocks_docker_gateway_and_host_ports(
    network_evidence: NetworkEvidence, world: World
) -> None:
    ev = network_evidence
    own_gateway = f"own-gateway:{world.host_service.port}"

    assert not _outcome(ev.sandbox_egress, "bridge-gateway")["connected"]
    assert not _outcome(ev.sandbox_egress, own_gateway)["connected"]
    # service side: the host-namespace listener saw nothing from either sandbox run
    assert ev.after_egress.host == ev.after_controls.host


def test_egress_blocks_other_containers_on_the_default_bridge(
    network_evidence: NetworkEvidence,
) -> None:
    ev = network_evidence

    assert not _outcome(ev.sandbox_egress, "private-container")["connected"]
    assert ev.after_egress.private == ev.after_controls.private, (
        f"internal service on :{PRIVATE_PORT} received sandbox traffic"
    )


def test_egress_blocks_link_local_metadata_endpoint(network_evidence: NetworkEvidence) -> None:
    ev = network_evidence

    imds = _outcome(ev.sandbox_egress, "imds-sim")
    assert not imds["connected"], f"{IMDS_ADDRESS} reachable: {imds}"
    assert "reply" not in imds  # never any content
    assert ev.after_egress.imds == ev.after_controls.imds
    assert not _outcome(ev.sandbox_egress, "unrouted-test-net")["connected"]


def test_dns_gives_no_lateral_reach(network_evidence: NetworkEvidence, world: World) -> None:
    ev = network_evidence

    # control: on a shared user network the container name does resolve
    assert ev.control_dns["names"][world.private.name], "DNS control did not resolve"
    # sandbox: neither run can resolve a sibling container's name
    assert ev.sandbox_egress["names"][world.private.name] == []
    assert ev.sandbox_none["names"][world.private.name] == []
    assert TARGET_PORT == 8080  # alias port used above; keeps the probe table honest
