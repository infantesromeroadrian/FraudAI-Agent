"""Logic tests for deploy.sh / rollback.sh with SIMULATED docker, git and curl.

These tests run the REAL shell scripts but put stub executables first on PATH
(see stub_tools.py). They prove the scripts' decisions -- immutable SHA tags, no
re-tagging or deletion, refusal of mislabelled images, the mandatory served-SHA check,
rollback to the previous deployment -- not that Docker, Compose or the application
behave that way. The SHAs below are placeholders for the doubles, NOT commits of any
repository, and no Git command that writes is ever run. A real two-commit deployment
needs real commits made by git-master and is verified separately.
Run: python3 -m unittest discover -s scripts/tests -v
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(os.environ.get("F5A_SCRIPTS_DIR") or Path(__file__).resolve().parent.parent)
STUB = Path(__file__).resolve().parent / "stub_tools.py"
SHA_A = "a" * 40  # placeholder identities for the doubles only
SHA_B = "b" * 40


class Harness:
    def __init__(self, images: dict | None = None, running: str | None = None) -> None:
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        self.state = root / "state.json"
        self.log = root / "calls.log"
        self.bin = root / "bin"
        self.bin.mkdir()
        for tool in ("docker", "git", "curl"):
            wrapper = self.bin / tool
            wrapper.write_text(f'#!/bin/sh\nexec python3 "{STUB}" {tool} "$@"\n')
            wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
        self.state.write_text(
            json.dumps({"images": images or {}, "running": running, "fail_up": False})
        )
        self.env_file = root / "synthetic.env"
        self.env_file.write_text("GRAFANA_PASSWORD=synthetic\n")
        self.state_dir = root / "deploy-state"

    def run(self, script: str, *args: str, **extra_env: str) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "STUB_STATE": str(self.state),
            "STUB_LOG": str(self.log),
            "STUB_SHA_A": SHA_A,
            "STUB_SHA_B": SHA_B,
            "FRAUDAI_ENV_FILE": str(self.env_file),
            "FRAUDAI_STATE_DIR": str(self.state_dir),
            **extra_env,
        }
        return subprocess.run(  # noqa: S603
            ["bash", str(SCRIPTS / script), *args],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            check=False,
        )

    def calls(self) -> list[list]:
        return (
            [json.loads(line) for line in self.log.read_text().splitlines()]
            if self.log.exists()
            else []
        )

    def current(self) -> dict:
        return json.loads(self.state.read_text())


def image(sha_label: str, ident: str) -> dict:
    return {"id": f"sha256:{ident}", "label": sha_label}


class DeployAndRollback(unittest.TestCase):
    def harness(self, **kw: object) -> Harness:
        h = Harness(**kw)  # type: ignore[arg-type]
        self.addCleanup(h.dir.cleanup)
        return h

    # -- deploy ----------------------------------------------------------------
    def test_deploy_names_the_image_by_sha_at_build_time_and_never_retags(self) -> None:
        h = self.harness()
        result = h.run("deploy.sh", "--ref", "HEAD", "--no-frontend")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        state = h.current()
        self.assertIn(f"fraudai-agent:{SHA_B}", state["images"])
        self.assertEqual(state["images"][f"fraudai-agent:{SHA_B}"]["label"], SHA_B)
        builds = [c for c in h.calls() if c[0] == "docker" and c[1] == "build"]
        self.assertEqual(len(builds), 1)
        self.assertIn(f"fraudai-agent:{SHA_B}", builds[0])
        self.assertNotIn("latest", " ".join(map(str, builds[0])))
        mutating = [
            c
            for c in h.calls()
            if c[0] == "docker"
            and c[1] in {"tag", "rmi", "rm"}
            or c[:3] == ["docker", "image", "rm"]
        ]
        self.assertEqual(mutating, [], "deploy must never retag or delete images")

    def test_two_deployments_keep_both_images_and_record_the_previous_one(self) -> None:
        h = self.harness()
        self.assertEqual(h.run("deploy.sh", "--ref", "HEAD~1", "--no-frontend").returncode, 0)
        self.assertEqual(h.run("deploy.sh", "--ref", "HEAD", "--no-frontend").returncode, 0)
        images = h.current()["images"]
        self.assertIn(f"fraudai-agent:{SHA_A}", images)
        self.assertIn(f"fraudai-agent:{SHA_B}", images)
        entries = [
            line.split("\t") for line in (h.state_dir / "deployments.log").read_text().splitlines()
        ]
        self.assertEqual(
            [e[1:4] for e in entries], [["deploy", SHA_A, "-"], ["deploy", SHA_B, SHA_A]]
        )

    def test_rollback_to_previous_runs_the_old_image_and_verifies_the_served_sha(self) -> None:
        h = self.harness()
        h.run("deploy.sh", "--ref", "HEAD~1", "--no-frontend")
        h.run("deploy.sh", "--ref", "HEAD", "--no-frontend")
        builds_before = h.current().get("builds")
        result = h.run("rollback.sh", "--previous")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        state = h.current()
        self.assertEqual(state["running"], f"fraudai-agent:{SHA_A}")
        self.assertEqual(state.get("builds"), builds_before, "rollback must not build anything")
        self.assertIn(f"the API serves build_sha {SHA_A}", result.stdout)
        self.assertIn(f"fraudai-agent:{SHA_B}", state["images"], "the replaced image must survive")
        self.assertEqual(h.run("rollback.sh", "--previous").returncode, 0)  # and forward again

    def test_deploy_refuses_to_overwrite_a_tag_with_a_foreign_revision(self) -> None:
        h = self.harness(images={f"fraudai-agent:{SHA_B}": image("c" * 40, "foreign000000")})
        result = h.run("deploy.sh", "--ref", "HEAD", "--no-frontend")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to overwrite", result.stderr)
        self.assertEqual(h.current()["images"][f"fraudai-agent:{SHA_B}"]["label"], "c" * 40)

    def test_deploy_reuses_an_existing_correctly_labelled_image_without_rebuilding(self) -> None:
        h = self.harness(images={f"fraudai-agent:{SHA_B}": image(SHA_B, "good00000000")})
        result = h.run("deploy.sh", "--ref", "HEAD", "--no-frontend")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(h.current().get("builds", 0), 0)

    def test_a_build_without_the_revision_label_is_rejected(self) -> None:
        h = self.harness()
        result = h.run("deploy.sh", "--ref", "HEAD", "--no-frontend", STUB_BUILD_LABEL="unset")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not carry revision", result.stderr)

    # -- the served SHA cannot be skipped ----------------------------------------
    def test_a_served_sha_that_differs_fails_the_deployment(self) -> None:
        h = self.harness()
        result = h.run("deploy.sh", "--ref", "HEAD", "--no-frontend", STUB_SERVED_SHA=SHA_A)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("the API serves build_sha", result.stderr)

    def test_a_missing_served_sha_fails_the_deployment(self) -> None:
        h = self.harness()
        result = h.run("deploy.sh", "--ref", "HEAD", "--no-frontend", STUB_SERVED_SHA="")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("<none>", result.stderr)

    def test_there_is_no_switch_to_skip_the_served_sha_check(self) -> None:
        h = self.harness()
        for script in ("deploy.sh", "rollback.sh"):
            result = h.run(script, "--no-served-sha-check", SHA_A)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unknown argument", result.stderr)

    # -- rollback refusals ------------------------------------------------------------
    def test_rollback_refuses_an_image_whose_label_is_not_its_tag(self) -> None:
        h = self.harness(
            images={
                f"fraudai-agent:{SHA_A}": image(SHA_B, "mislabelled00"),
                f"fraudai-agent:{SHA_B}": image(SHA_B, "good00000000"),
            },
            running=f"fraudai-agent:{SHA_B}",
        )
        result = h.run("rollback.sh", SHA_A)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mislabelled", result.stderr)
        self.assertEqual(
            h.current()["running"], f"fraudai-agent:{SHA_B}", "the running deployment is untouched"
        )
        self.assertEqual(
            [c for c in h.calls() if c[1:3] == ["compose", "up"] or "up" in c[3:5]], []
        )

    def test_rollback_to_an_unknown_image_is_refused(self) -> None:
        h = self.harness(
            images={f"fraudai-agent:{SHA_B}": image(SHA_B, "good00000000")},
            running=f"fraudai-agent:{SHA_B}",
        )
        result = h.run("rollback.sh", SHA_A)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not found locally", result.stderr)

    def test_previous_without_history_is_refused(self) -> None:
        h = self.harness()
        result = h.run("rollback.sh", "--previous")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No previous deployment", result.stderr)

    def test_rollback_fails_when_the_restored_container_serves_another_sha(self) -> None:
        h = self.harness(
            images={
                f"fraudai-agent:{SHA_A}": image(SHA_A, "old000000000"),
                f"fraudai-agent:{SHA_B}": image(SHA_B, "good00000000"),
            },
            running=f"fraudai-agent:{SHA_B}",
        )
        result = h.run("rollback.sh", SHA_A, STUB_SERVED_SHA=SHA_B)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Rollback verification failed", result.stderr)

    def test_list_shows_only_sha_tags(self) -> None:
        h = self.harness(
            images={
                f"fraudai-agent:{SHA_A}": image(SHA_A, "old000000000"),
                "fraudai-agent:latest": image("", "x00000000000"),
            }
        )
        result = h.run("rollback.sh", "--list")
        self.assertEqual(result.returncode, 0)
        self.assertIn(SHA_A, result.stdout)
        self.assertNotIn("latest", result.stdout)


if __name__ == "__main__":
    unittest.main()
