"""F3.5 -- filesystem, process and output negatives against a real Docker daemon.

Every assertion observes what the kernel enforces on code running inside the
real ``SandboxEngine`` container (errno values, process counts, files that come
back to the host); none of them inspects configuration passed to Docker.

``EROFS`` is asserted for read-only paths on purpose: Linux reports it before
any permission check, so an unprivileged writer gets ``EROFS`` only when the
mount really is read-only (a writable root would answer ``EACCES``).

Limits: Docker Desktop (Mac) and ubuntu-latest are different kernels; CI is the
source of truth for the sandbox. Seccomp is Docker's default profile.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import docker.errors
import pytest

from fraudai.tools.sandbox import OutputCaps, SandboxEngine
from tests.integration.harness import run_async, run_without_leaks, validated_target

if TYPE_CHECKING:
    from pathlib import Path

    from tests.integration.conftest import DockerEnv

MIB = 1024 * 1024
# errno numbers of the Linux container, not of the machine running pytest
# (EAGAIN is 11 on Linux but 35 on macOS).
EACCES, EROFS, ENOSPC, EAGAIN = 13, 30, 28, 11
TMPFS_LIMIT_BYTES = 64 * MIB  # SandboxEngine.TMPFS_OPTIONS: size=64m
SMALL_PIDS_LIMIT = 16
PID_ATTEMPTS = 40  # finite and small: never an unbounded loop
INPUT_CANARY = "synthetic-input-canary"

HARDENING_PROBE = """\
import json, os, subprocess

report = {}


def try_write(path):
    try:
        with open(path, "w") as handle:
            handle.write("f35")
    except OSError as exc:
        return {"written": False, "errno": exc.errno}
    return {"written": True}


report["write_attempts"] = {
    path: try_write(path)
    for path in (
        "/f35-rootfs-canary",
        "/etc/f35-canary",
        "/usr/local/f35-canary",
        "/workspace/f35-new-input",
        "/workspace/data.txt",
    )
}
report["input_read"] = open("/workspace/data.txt", "rb").read().decode()
with open("/workspace/output/ok.txt", "w") as handle:
    handle.write("output-ok")

script = "/tmp/f35.sh"
with open(script, "w") as handle:
    handle.write("#!/bin/sh\\necho exec-ran\\n")
os.chmod(script, 0o755)
try:
    subprocess.run([script], capture_output=True, check=True)
    report["direct_exec"] = {"ran": True}
except OSError as exc:
    report["direct_exec"] = {"ran": False, "errno": exc.errno}
report["via_sh"] = subprocess.run(
    ["/bin/sh", script], capture_output=True, text=True
).stdout.strip()

# Several files below the 4 MiB per-file ulimit, so the tmpfs size is what stops us.
chunk, written, fill_errno = b"x" * (1 << 20), 0, None
try:
    for index in range(30):
        with open("/tmp/fill%02d" % index, "wb") as handle:
            for _ in range(3):
                handle.write(chunk)
                handle.flush()
                written += len(chunk)
except OSError as exc:
    fill_errno = exc.errno
report["tmpfs"] = {"written": written, "errno": fill_errno}

status = {}
for line in open("/proc/self/status"):
    key, _, value = line.partition(":")
    status[key] = value.strip()
report["identity"] = {
    "uid": os.getuid(),
    "cap_eff": status["CapEff"],
    "no_new_privs": status["NoNewPrivs"],
}
print("PROBE:" + json.dumps(report))
"""

PIDS_PROBE = f"""\
import json, subprocess

children, error = [], None
for _ in range({PID_ATTEMPTS}):
    try:
        children.append(subprocess.Popen(["sleep", "5"]))
    except OSError as exc:
        error = exc.errno
        break
started = len(children)
for child in children:
    child.kill()
for child in children:
    child.wait()
print("PROBE:" + json.dumps({{"started": started, "errno": error}}))
"""


def _probe(stdout: str) -> dict[str, Any]:
    line = next(line for line in stdout.splitlines() if line.startswith("PROBE:"))
    parsed: dict[str, Any] = json.loads(line[len("PROBE:") :])
    return parsed


@pytest.fixture(scope="module")
def hardening(docker_env: DockerEnv, engine: SandboxEngine) -> dict[str, Any]:
    result = run_without_leaks(
        docker_env.client,
        engine.execute_python(HARDENING_PROBE, input_files={"data.txt": INPUT_CANARY.encode()}),
    )
    assert result.exit_code == 0, result.stderr
    assert result.output_files["ok.txt"] == b"output-ok"  # the one writable path works
    return _probe(result.stdout)


def test_root_filesystem_and_inputs_are_read_only(hardening: dict[str, Any]) -> None:
    attempts = hardening["write_attempts"]

    assert hardening["input_read"] == INPUT_CANARY  # control: inputs are readable
    assert {path: a.get("errno") for path, a in attempts.items()} == {
        path: EROFS for path in attempts
    }, attempts


def test_tmp_is_noexec_and_size_bounded(hardening: dict[str, Any]) -> None:
    assert hardening["via_sh"] == "exec-ran"  # control: the script itself is valid
    assert hardening["direct_exec"] == {"ran": False, "errno": EACCES}

    tmpfs = hardening["tmpfs"]
    assert tmpfs["written"] >= MIB  # control: /tmp is writable
    assert tmpfs["written"] <= TMPFS_LIMIT_BYTES
    assert tmpfs["errno"] == ENOSPC


def test_process_is_unprivileged_with_no_capabilities(hardening: dict[str, Any]) -> None:
    identity = hardening["identity"]

    assert identity["uid"] != 0
    assert int(identity["cap_eff"], 16) == 0
    assert identity["no_new_privs"] == "1"


def test_pids_limit_caps_process_creation(docker_env: DockerEnv) -> None:
    limited = SandboxEngine(
        docker_client=docker_env.client,
        image=docker_env.image,
        timeout=60,
        pids_limit=SMALL_PIDS_LIMIT,
    )
    generous = SandboxEngine(docker_client=docker_env.client, image=docker_env.image, timeout=60)

    control = run_without_leaks(docker_env.client, generous.execute_python(PIDS_PROBE))
    capped = run_without_leaks(docker_env.client, limited.execute_python(PIDS_PROBE))

    assert control.exit_code == 0, control.stderr
    assert _probe(control.stdout) == {"started": PID_ATTEMPTS, "errno": None}  # control
    assert capped.exit_code == 0, capped.stderr
    outcome = _probe(capped.stdout)
    assert 0 < outcome["started"] < SMALL_PIDS_LIMIT
    assert outcome["errno"] == EAGAIN


def test_symlinks_and_special_files_in_output_are_not_followed(
    docker_env: DockerEnv, engine: SandboxEngine, tmp_path: Path
) -> None:
    canary = tmp_path / "host-canary.txt"
    canary.write_text("HOST-SECRET-CANARY")
    secret_dir = tmp_path / "host-secret-dir"
    secret_dir.mkdir()
    (secret_dir / "inside.txt").write_text("HOST-SECRET-IN-DIR")
    code = f"""\
import json, os

out = "/workspace/output"
os.symlink({str(canary)!r}, out + "/leak-file")
os.symlink({str(secret_dir)!r}, out + "/leak-dir")
os.mkdir(out + "/sub")
os.symlink({str(canary)!r}, out + "/sub/nested-leak")
os.mkfifo(out + "/fifo")
with open(out + "/ok.txt", "w") as handle:
    handle.write("regular")
try:
    open({str(canary)!r}).read()
    sees_host_path = True
except OSError:
    sees_host_path = False
print("PROBE:" + json.dumps({{"sees_host_path": sees_host_path}}))
"""

    result = run_without_leaks(docker_env.client, engine.execute_python(code))

    assert result.exit_code == 0, result.stderr
    assert _probe(result.stdout) == {"sees_host_path": False}  # container cannot read it itself
    assert result.output_files == {"ok.txt": b"regular"}  # control: regular files come back
    assert result.rejected_outputs == ["fifo", "leak-dir", "leak-file", "sub/nested-leak"]
    assert b"HOST-SECRET" not in b"".join(result.output_files.values())
    assert canary.read_text() == "HOST-SECRET-CANARY"


def test_collected_output_respects_size_caps_and_reports_written_volume(
    docker_env: DockerEnv, engine: SandboxEngine, record_property: Any
) -> None:
    caps = OutputCaps()
    code = f"""\
import json, os

out = "/workspace/output"
chunk = b"a" * {MIB}
report = {{}}


def write(name, mib):
    written, error = 0, None
    try:
        with open(out + "/" + name, "wb") as handle:
            for _ in range(mib):
                handle.write(chunk)
                handle.flush()
                written += len(chunk)
    except OSError as exc:
        error = exc.errno
    report[name] = {{"written": written, "errno": error}}


write("over-one-file.bin", 5)
for name in ("a.bin", "b.bin", "c.bin"):
    write(name, 3)
print("PROBE:" + json.dumps(report))
"""

    result = run_without_leaks(docker_env.client, engine.execute_python(code))

    assert result.exit_code == 0, result.stderr
    written = _probe(result.stdout)
    # control: the process really did write more than a single file may return
    assert written["a.bin"] == {"written": 3 * MIB, "errno": None}
    returned = result.output_files
    assert all(len(content) <= caps.max_file_bytes for content in returned.values())
    assert sum(len(content) for content in returned.values()) <= caps.max_total_bytes
    assert result.rejected_outputs, "9 MiB of 3 MiB files must exceed the 8 MiB total cap"
    # Evaluated, not asserted: the caps apply when outputs are collected. How much a
    # run may write to the host before that is whatever the per-file ulimit allows.
    total_written = sum(entry["written"] for entry in written.values())
    record_property("bytes_written_by_sandbox", total_written)
    record_property("bytes_returned_to_host", sum(len(c) for c in returned.values()))
    print(f"written_by_sandbox={total_written} returned={sum(map(len, returned.values()))}")


def test_collected_output_respects_file_count_cap(
    docker_env: DockerEnv, engine: SandboxEngine
) -> None:
    caps = OutputCaps()
    code = f"""\
for index in range({caps.max_files + 6}):
    with open("/workspace/output/f%03d.txt" % index, "w") as handle:
        handle.write("x")
"""

    result = run_without_leaks(docker_env.client, engine.execute_python(code))

    assert result.exit_code == 0, result.stderr
    assert len(result.output_files) == caps.max_files
    assert len(result.rejected_outputs) == 6


def test_timeout_stops_the_container_and_leaves_nothing_behind(
    docker_env: DockerEnv, engine: SandboxEngine
) -> None:
    result = run_without_leaks(
        docker_env.client, engine.execute_python("import time\ntime.sleep(60)\n", timeout=3)
    )

    assert result.timed_out is True
    assert result.exit_code == -1
    assert result.duration_seconds < 30


def test_failed_setup_cleans_up_network_and_forwarder(
    docker_env: DockerEnv, no_leaks: None
) -> None:
    """The egress network is created before the missing image is noticed."""
    broken = SandboxEngine(
        docker_client=docker_env.client,
        image="fraudai-sandbox:f35-image-that-does-not-exist",
        timeout=30,
    )
    target = validated_target("198.51.100.7")  # TEST-NET-2; never contacted

    with pytest.raises(docker.errors.ImageNotFound):
        run_async(broken.execute_python("print('unreachable')", egress=target))
