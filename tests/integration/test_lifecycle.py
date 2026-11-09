"""F3 correction round -- sandbox lifecycle on a real daemon.

Cancellation. ``asyncio.to_thread`` cannot stop a Docker create that is already
running, so a cancelled caller used to leave the late-created network / forwarder /
sandbox behind. Here the engine's create is *held* inside the worker thread with
events (no sleeps), the caller is cancelled, the create is released, and the
engine must collect and remove what the late create produced before it finishes,
free its concurrency slot, and let the next run work.

Stop failures. With a sandbox process still writing, ``wait``/``stop``/``kill`` are
made to fail (the real daemon still executes everything else); outputs may only
be read once the sandbox is confirmed gone, and are discarded -- with the error
propagated -- when that cannot be confirmed.

Accepted limit (not fixed by the engine): ``SIGKILL`` of the Python process gives
no ``finally`` a chance to run, so a sandbox can outlive it. That is asserted as
a characterisation, with cleanup by the exact ids it created.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any

import docker
import docker.errors
import pytest

from fraudai.tools.sandbox import OutputCaps, SandboxEngine, SandboxError, SandboxResult
from tests.integration.harness import (
    LifecycleProbe,
    run_async,
    run_without_leaks,
    sandbox_resources,
    validated_target,
)

if TYPE_CHECKING:
    from tests.integration.conftest import DockerEnv
    from tests.integration.harness import World

STEP_TIMEOUT_SECONDS = 60
WRITER_CODE = """\
import time

n = 0
while True:
    with open("/workspace/output/tick.txt", "w") as handle:
        handle.write(str(n))
    n += 1
    time.sleep(0.01)
"""


# --- cancellation during create ---------------------------------------------------------


def _exists(client: docker.DockerClient, kind: str, resource_id: str) -> bool:
    try:
        (client.networks if kind == "network" else client.containers).get(resource_id)
    except docker.errors.NotFound:
        return False
    return True


@pytest.mark.parametrize("slot", ["network", "proxy", "sandbox"])
def test_cancel_during_create_collects_and_removes_the_late_resource(
    docker_env: DockerEnv, world: World, slot: str
) -> None:
    client = docker_env.client
    engine = SandboxEngine(
        docker_client=client, image=docker_env.image, timeout=STEP_TIMEOUT_SECONDS, max_concurrent=1
    )
    egress = validated_target(world.target.address())
    before = sandbox_resources(client)

    with LifecycleProbe(client) as probe:
        probe.block_slot = slot

        async def scenario() -> dict[str, Any]:
            task = asyncio.ensure_future(engine.execute_python("print(1)", egress=egress))
            assert await asyncio.to_thread(probe.entered.wait, STEP_TIMEOUT_SECONDS)
            task.cancel()
            # Let the loop deliver the cancellation to the engine (FIFO callbacks, no timing)
            # before the held create is allowed to finish.
            for _ in range(3):
                await asyncio.sleep(0)
            probe.release.set()
            try:
                await task
                cancelled = False
            except asyncio.CancelledError:
                cancelled = True
            outcome: dict[str, Any] = {
                "cancelled": cancelled,
                "late_create_finished": probe.finished.is_set(),
                "left_behind": sandbox_resources(client) - before,
                "orphan_tasks": [t for t in asyncio.all_tasks() if t is not asyncio.current_task()],
            }
            # The only concurrency slot was released: a following run is not blocked.
            follow_up = await asyncio.wait_for(
                engine.execute_python("print('next')"), STEP_TIMEOUT_SECONDS
            )
            outcome["next_run"] = (follow_up.exit_code, follow_up.stdout.strip())
            return outcome

        outcome = run_async(scenario())

        late_id = probe.networks[0] if slot == "network" else probe.containers[slot][0]
        assert outcome["cancelled"], "the caller's cancellation must propagate"
        assert outcome["late_create_finished"], "the held create must really have completed late"
        assert not _exists(client, "network" if slot == "network" else "container", late_id)
        assert outcome["left_behind"] == frozenset()
        assert outcome["orphan_tasks"] == []
        assert outcome["next_run"] == (0, "next")
    assert sandbox_resources(client) == before


# --- wait / stop / kill failing with an active writer ------------------------------------


def _run_with_active_writer(env: DockerEnv, probe: LifecycleProbe) -> SandboxResult:
    probe.wait_for_writer = True
    engine = SandboxEngine(docker_client=env.client, image=env.image, timeout=STEP_TIMEOUT_SECONDS)
    return run_async(engine.execute_python(WRITER_CODE))


def test_failed_wait_stops_the_writer_before_outputs_are_read(docker_env: DockerEnv) -> None:
    before = sandbox_resources(docker_env.client)
    with LifecycleProbe(docker_env.client) as probe:
        probe.fail_methods = {"wait": -1}

        result = _run_with_active_writer(docker_env, probe)

        assert result.timed_out is True
        assert probe.reads == [["exited"]]  # read only after the daemon reported it exited
        assert int(result.output_files["tick.txt"]) >= 0  # the writer was active, output intact
    assert sandbox_resources(docker_env.client) == before


def test_failed_wait_stop_and_kill_force_remove_the_writer_before_outputs_are_read(
    docker_env: DockerEnv,
) -> None:
    before = sandbox_resources(docker_env.client)
    with LifecycleProbe(docker_env.client) as probe:
        probe.fail_methods = {"wait": -1, "stop": -1, "kill": -1}

        result = _run_with_active_writer(docker_env, probe)

        assert result.timed_out is True
        assert probe.reads == [["gone"]]  # the container no longer existed when read
        assert "tick.txt" in result.output_files
        assert (result.stdout, result.stderr) == ("", "")  # its logs left with it
    assert sandbox_resources(docker_env.client) == before


def test_unconfirmed_removal_discards_outputs_and_propagates_the_error(
    docker_env: DockerEnv,
) -> None:
    before = sandbox_resources(docker_env.client)
    with LifecycleProbe(docker_env.client) as probe:
        # wait/stop/kill fail, and the first (forced) removal fails too: the engine
        # cannot prove the writer is gone. The retry in cleanup is allowed to work.
        probe.fail_methods = {"wait": -1, "stop": -1, "kill": -1, "remove": 1}

        with pytest.raises(SandboxError, match="discarded"):
            _run_with_active_writer(docker_env, probe)

        assert probe.reads == [], "outputs were read although the writer was unconfirmed"
    assert sandbox_resources(docker_env.client) == before


LOGGING_WRITER_CODE = 'print("writer-started", flush=True)\n' + WRITER_CODE


def test_exited_container_whose_removal_errored_is_removed_again_by_the_final_cleanup(
    docker_env: DockerEnv,
) -> None:
    """Residual CX1: the daemon stopped the writer but ``remove`` raised anyway.

    The container then exists in state ``exited`` -- not ``NotFound`` -- so the
    engine may read its outputs but must still remove it at the end. Treating
    "stopped" as "gone" skipped that second removal and left it behind.
    """
    client = docker_env.client
    before = sandbox_resources(client)
    engine = SandboxEngine(
        docker_client=client, image=docker_env.image, timeout=STEP_TIMEOUT_SECONDS
    )

    with LifecycleProbe(client) as probe:
        probe.wait_for_writer = True
        probe.fail_methods = {"wait": -1, "stop": -1, "kill": -1}
        probe.remove_stops_then_fails = 1

        async def scenario() -> tuple[SandboxResult, list[Any]]:
            result = await engine.execute_python(LOGGING_WRITER_CODE)
            return result, [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

        result, orphan_tasks = run_async(scenario())

        sandbox_id = probe.containers["sandbox"][0]
        assert result.timed_out is True
        assert probe.state_after_failed_remove == "exited"  # stopped, yet still present
        assert probe.remove_calls == 2, "the final cleanup must remove it a second time"
        # stopped (not NotFound): outputs and logs were read from the exited container,
        # unlike the force-removed case where its logs left with it
        assert probe.reads == [["exited"]]
        assert "writer-started" in result.stdout
        assert int(result.output_files["tick.txt"]) >= 0
        # absence is confirmed before the run is over, by the engine and not by the probe's
        # own cleanup (which only runs when this block exits)
        assert not _exists(client, "container", sandbox_id)
        assert orphan_tasks == []
    assert sandbox_resources(client) == before


# --- output flood and timeout ceiling, measured on the real bind mount ----------------------


def test_junk_flood_in_output_is_bounded_and_keeps_result_json(
    docker_env: DockerEnv, engine: SandboxEngine
) -> None:
    caps = OutputCaps()
    code = """\
import os

out = "/workspace/output"
for index in range(5000):
    open("%s/junk%05d" % (out, index), "w").close()
with open(out + "/result.json", "w") as handle:  # written last on purpose
    handle.write("{}")
"""

    started = time.monotonic()
    result = run_without_leaks(docker_env.client, engine.execute_python(code))
    elapsed = time.monotonic() - started

    assert result.exit_code == 0, result.stderr
    assert result.outputs_truncated is True
    assert result.output_files["result.json"] == b"{}"
    assert len(result.output_files) <= caps.max_files
    assert len(result.rejected_outputs) <= caps.max_rejected_names
    assert result.rejected_total > 0
    assert elapsed < STEP_TIMEOUT_SECONDS


def test_configured_timeout_is_a_ceiling_for_longer_requests(docker_env: DockerEnv) -> None:
    short = SandboxEngine(docker_client=docker_env.client, image=docker_env.image, timeout=5)

    result = run_without_leaks(
        docker_env.client, short.execute_python("import time\ntime.sleep(60)\n", timeout=300)
    )

    assert result.timed_out is True
    assert result.duration_seconds < 30  # ~5 s, nowhere near the 300 s that was asked for


# --- accepted limit: SIGKILL -------------------------------------------------------------

SIGKILLED_ENGINE = """\
import asyncio, sys
from fraudai.tools.sandbox import SandboxEngine

engine = SandboxEngine(image=sys.argv[1], timeout=300)
asyncio.run(engine.execute_python("import time\\ntime.sleep(300)\\n", timeout=300))
"""


def test_sigkill_of_the_engine_process_leaves_the_sandbox_behind_until_removed_by_id(
    docker_env: DockerEnv,
) -> None:
    """Characterises the accepted limit; if cleanup of stale resources is ever added, update it."""
    client = docker_env.client
    before = sandbox_resources(client)
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", SIGKILLED_ENGINE, docker_env.image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + STEP_TIMEOUT_SECONDS
        running: list[Any] = []
        while time.monotonic() < deadline and not running:
            running = [
                c
                for c in client.containers.list(filters={"label": "fraudai.sandbox=1"})
                if f"container:{c.id}" not in before
            ]
            if not running:
                time.sleep(0.2)
        assert running, "the engine process never started a sandbox"

        child.kill()
        child.wait(timeout=30)

        survivors = [
            c
            for c in client.containers.list(filters={"label": "fraudai.sandbox=1"})
            if c in running
        ]
        assert survivors, "expected the sandbox to outlive a SIGKILLed engine (accepted limit)"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=30)
        leftovers = sandbox_resources(client) - before  # exactly what this test created
        for resource in sorted(leftovers):  # containers first, then networks
            kind, _, resource_id = resource.partition(":")
            if kind == "container":
                client.containers.get(resource_id).remove(force=True)
        for resource in sorted(leftovers):
            kind, _, resource_id = resource.partition(":")
            if kind == "network":
                client.networks.get(resource_id).remove()
    assert sandbox_resources(client) == before
