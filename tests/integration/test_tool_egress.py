"""F3.5 -- Mike's tools through the real ``ToolImplementations`` + ``SandboxEngine``.

Real: the tools, the code they generate, the egress policy, the sandbox,
Docker, the forwarder container and the sandbox's own ``urllib`` client.
Synthetic: the target services (ours, on the local Docker bridge) and the
resolver/address predicate injected into ``EgressPolicy`` *in the test only*;
the production policy is exercised unmodified in the default-policy tests.

Services are alive (positive control from a plain container) before any
"zero hits" claim is made, and "zero" is read from the service's own log.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from fraudai.tools.egress import EgressDeniedError, EgressPolicy
from fraudai.tools.implementations import ToolImplementations
from fraudai.tools.sandbox import SandboxEngine
from tests.integration.harness import (
    PRIVATE_PORT,
    TARGET_ALIAS,
    TARGET_PORT,
    parse_probe,
    render_net_probe,
    run_async,
    run_control,
    run_without_leaks,
    synthetic_policy,
    validated_target,
)

if TYPE_CHECKING:
    from tests.integration.conftest import DockerEnv
    from tests.integration.harness import World

HOST_HEADER = f"{TARGET_ALIAS}:{TARGET_PORT}"  # what urllib sends for a non-default port
EVASION_REQUESTS = 40  # 20 samples, original + perturbed query each
TARGET_URL = f"http://{TARGET_ALIAS}:{TARGET_PORT}/predict"


class _StageSpy(SandboxEngine):
    """Real engine that counts how many runs got past argument validation."""

    stages_started = 0

    async def _run_gated(self, *args: Any, **kwargs: Any) -> Any:
        type(self).stages_started += 1
        return await super()._run_gated(*args, **kwargs)


def _tools(engine: SandboxEngine, policy: EgressPolicy) -> ToolImplementations:
    return ToolImplementations(sandbox=engine, egress_policy=policy)


def _prove_private_service_is_reachable(docker_env: DockerEnv, world: World) -> None:
    """Positive control: a plain container on the bridge reaches the private service."""
    before = world.private.accepts()
    code = render_net_probe([("probe", world.private.address(), PRIVATE_PORT)])
    report = parse_probe(
        run_control(
            docker_env.client, docker_env.image, code, session=docker_env.session, network="bridge"
        )
    )
    assert report["targets"]["probe"]["connected"]  # type: ignore[index]
    assert world.private.accepts() > before


def test_adversarial_evasion_reaches_the_pinned_target_end_to_end(
    docker_env: DockerEnv, engine: SandboxEngine, world: World
) -> None:
    tools = _tools(engine, synthetic_policy(world.target.address()))
    before = world.target.hits(HOST_HEADER)

    result = run_async(tools.adversarial_evasion(TARGET_URL))

    assert result["status"] == "success", result
    answered = [p for p in result["original_predictions"] if p.get("prediction") == 1]
    assert len(answered) == 20  # every sample got the synthetic target's answer
    assert world.target.hits(HOST_HEADER) - before == EVASION_REQUESTS


def test_prompt_injection_suite_reaches_the_pinned_target_end_to_end(
    engine: SandboxEngine, world: World
) -> None:
    tools = _tools(engine, synthetic_policy(world.target.address()))
    before = world.target.hits(HOST_HEADER)

    result = run_async(tools.prompt_injection_suite(TARGET_URL))

    assert result["status"] == "success", result
    assert result["total_payloads_tested"] > 0
    assert world.target.hits(HOST_HEADER) - before == result["total_payloads_tested"]


def test_redirect_to_private_service_is_not_followed_by_generated_client(
    docker_env: DockerEnv, engine: SandboxEngine, world: World
) -> None:
    _prove_private_service_is_reachable(docker_env, world)
    private_before = world.private.accepts()
    redirect_before = world.redirect_target.hits(HOST_HEADER)
    tools = _tools(engine, synthetic_policy(world.redirect_target.address()))

    result = run_async(tools.adversarial_evasion(TARGET_URL))

    assert result["status"] == "success", result
    # the validated target WAS contacted (so a missing private hit is not a dead path) ...
    assert world.redirect_target.hits(HOST_HEADER) > redirect_before
    assert all("302" in entry.get("error", "") for entry in result["original_predictions"])
    # ... and the private service it redirected to saw nothing
    assert world.private.accepts() == private_before


def test_network_blocks_a_redirect_even_when_the_client_follows_it(
    docker_env: DockerEnv, engine: SandboxEngine, world: World
) -> None:
    """Defence in depth: a stock urllib client (follows 302) still cannot reach the private host."""
    _prove_private_service_is_reachable(docker_env, world)
    private_before = world.private.accepts()
    redirect_before = world.redirect_target.hits(HOST_HEADER)
    code = f"""\
import json, urllib.error, urllib.request

try:
    urllib.request.urlopen({TARGET_URL!r}, timeout=10)
    outcome = {{"followed_to_body": True}}
except urllib.error.HTTPError as exc:
    outcome = {{"kind": "HTTPError", "code": exc.code}}
except urllib.error.URLError as exc:
    outcome = {{"kind": "URLError", "reason": str(exc.reason)}}
print("PROBE:" + json.dumps(outcome))
"""
    egress = validated_target(world.redirect_target.address())

    result = run_without_leaks(docker_env.client, engine.execute_python(code, egress=egress))

    assert result.exit_code == 0, result.stderr
    outcome = parse_probe(result.stdout)
    assert world.redirect_target.hits(HOST_HEADER) == redirect_before + 1
    # URLError (not HTTPError 302) means urllib really tried the second hop and the network said no
    assert outcome["kind"] == "URLError", outcome
    assert "unreachable" in str(outcome["reason"]).lower(), outcome
    assert world.private.accepts() == private_before


def test_dns_rebinding_after_validation_cannot_change_the_destination(
    docker_env: DockerEnv, engine: SandboxEngine, world: World
) -> None:
    _prove_private_service_is_reachable(docker_env, world)
    private_before = world.private.accepts()
    target_before = world.target.hits(HOST_HEADER)
    target_ip, private_ip = world.target.address(), world.private.address()
    resolutions: list[str] = []

    def rebinding_resolver(_host: str) -> list[str]:
        resolutions.append(_host)
        return [target_ip] if len(resolutions) == 1 else [private_ip]

    policy = EgressPolicy(
        resolver=rebinding_resolver, address_allowed=lambda ip: str(ip) == target_ip
    )
    tools = _tools(engine, policy)

    result = run_async(tools.adversarial_evasion(TARGET_URL))

    assert result["status"] == "success", result
    assert resolutions == [TARGET_ALIAS]  # resolved once; nothing re-resolves afterwards
    assert world.target.hits(HOST_HEADER) - target_before == EVASION_REQUESTS
    assert world.private.accepts() == private_before


# --- default (production) policy, no relaxation -------------------------------------

REJECTED_BY_DEFAULT_POLICY = [
    "http://127.0.0.1:8080/predict",
    "http://localhost:8080/predict",
    "http://[::1]:8080/predict",
    "http://10.0.0.5/predict",
    "http://172.17.0.1:6333/",
    "http://192.168.1.10/predict",
    "http://169.254.169.254/latest/meta-data/",
    "http://[fe80::1]/predict",
    "http://[fd00::1]/predict",
    "http://[::ffff:10.0.0.1]/predict",
    "http://0.0.0.0/predict",
    "http://2130706433/predict",  # decimal form of 127.0.0.1
    "http://user:secret@models.example.test/predict",
    "ftp://models.example.test/predict",
]


@pytest.mark.parametrize("url", REJECTED_BY_DEFAULT_POLICY)
def test_default_policy_validate_rejects_on_its_own(url: str) -> None:
    """Layer 1 in isolation: IP literals are also refused later by the sandbox
    engine, so the tool-level test below cannot tell which layer said no."""
    with pytest.raises(EgressDeniedError):
        EgressPolicy().validate(url)


@pytest.mark.parametrize("url", REJECTED_BY_DEFAULT_POLICY)
def test_default_policy_rejects_private_loopback_and_link_local_targets(
    docker_env: DockerEnv, url: str
) -> None:
    spy = _StageSpy(docker_client=docker_env.client, image=docker_env.image)
    tools = ToolImplementations(sandbox=spy)  # default EgressPolicy, nothing injected
    started_before = _StageSpy.stages_started

    with pytest.raises(PermissionError):
        run_async(tools.adversarial_evasion(url))
    with pytest.raises(PermissionError):
        run_async(tools.prompt_injection_suite(url))

    assert _StageSpy.stages_started == started_before, (
        "a container was prepared for a denied target"
    )


def test_default_predicate_accepts_a_global_address_so_rejections_are_not_blanket() -> None:
    policy = EgressPolicy(resolver=lambda _host: ["1.1.1.1"])  # data only; never contacted

    target = policy.validate("http://models.example.test/predict")

    assert (target.hostname, target.address, target.port) == ("models.example.test", "1.1.1.1", 80)


@pytest.mark.parametrize(
    "url",
    [
        "http://203.0.113.10:8080/predict",
        "https://203.0.113.10/predict",
        "http://[2001:db8::1]/predict",
    ],
)
def test_ip_literal_targets_are_refused_even_when_the_policy_would_allow_the_address(
    docker_env: DockerEnv, url: str
) -> None:
    """Final contract: every IP-literal destination is rejected, HTTP included."""
    permissive = EgressPolicy(
        resolver=lambda host: [host.strip("[]")], address_allowed=lambda _ip: True
    )
    spy = _StageSpy(docker_client=docker_env.client, image=docker_env.image)
    tools = ToolImplementations(sandbox=spy, egress_policy=permissive)
    started_before = _StageSpy.stages_started

    with pytest.raises(PermissionError, match="IP-literal"):
        run_async(tools.adversarial_evasion(url))

    assert _StageSpy.stages_started == started_before
