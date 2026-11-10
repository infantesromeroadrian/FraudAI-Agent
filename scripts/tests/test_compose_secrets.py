"""Secrets contract of the production Compose file (synthetic values only).

The local .env is the only persistent place for secrets. These tests show, with fake
values and throw-away containers (no application image, no deployment):
  * the metrics credential reaches containers through Compose's ``environment`` secret
    source, read from the dotenv file handed over with --env-file, readable by the API's
    uid (1000) and by Prometheus' (nobody) -- and Compose aborts when it is missing;
  * the API service receives an allow-list of variables, never the whole .env;
  * no token file or exported variable is part of the contract.
Docker (with the compose plugin) is REQUIRED: the tests fail, not skip, without it.
Run: python3 -m unittest discover -s scripts/tests -v
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
COMPOSE = ROOT / "docker-compose.prod.yml"
PYTHON_IMAGE = "python:3.11.18-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39"
SYNTHETIC = {
    "GRAFANA_PASSWORD": "synthetic-grafana-" + uuid.uuid4().hex,
    "METRICS_SCRAPE_TOKEN": "synthetic-metrics-" + uuid.uuid4().hex,
    "FRAUDAI_SMOKE_PASSWORD": "synthetic-smoke-" + uuid.uuid4().hex,
    "JWT_SECRET": "synthetic-jwt-" + uuid.uuid4().hex,
}


def docker(
    *args: str, env: dict[str, str] | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    base = {k: v for k, v in os.environ.items() if k not in SYNTHETIC}
    return subprocess.run(  # noqa: S603
        ["docker", *args],
        capture_output=True,
        text=True,
        env={**base, **(env or {})},
        timeout=180,
        check=check,
    )


class ComposeSecretsContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dotenv = Path(cls.tmp.name) / "synthetic.env"
        cls.dotenv.write_text("".join(f"{k}={v}\n" for k, v in SYNTHETIC.items()))
        cls.text = COMPOSE.read_text()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def rendered(self) -> dict:
        out = docker(
            "compose",
            "-f",
            str(COMPOSE),
            "--env-file",
            str(self.dotenv),
            "config",
            "--format",
            "json",
            env={"FRAUDAI_IMAGE": "fraudai-agent:t", "FRONTEND_IMAGE": "fraudai-frontend:t"},
        )
        return json.loads(out.stdout)

    # -- contract of the file ---------------------------------------------------
    def test_scrape_credential_is_an_environment_secret_not_a_file(self) -> None:
        secret = self.rendered()["secrets"]["fraudai_metrics_token"]
        self.assertEqual(secret.get("environment"), "METRICS_SCRAPE_TOKEN")
        self.assertNotIn("file", secret)
        self.assertNotIn("metrics-token", self.text)
        self.assertNotIn("FRAUDAI_METRICS_TOKEN_FILE", self.text)

    def test_only_api_and_prometheus_are_granted_the_secret(self) -> None:
        services = self.rendered()["services"]
        granted = sorted(
            n
            for n, s in services.items()
            if "fraudai_metrics_token" in (s.get("secrets") or [])
            or any(
                x.get("source") == "fraudai_metrics_token"
                for x in (s.get("secrets") or [])
                if isinstance(x, dict)
            )
        )
        self.assertEqual(granted, ["fraudai", "prometheus"])

    def test_api_gets_an_allow_list_not_the_whole_dotenv(self) -> None:
        api = self.rendered()["services"]["fraudai"]
        self.assertNotIn("env_file", api)
        env = api["environment"]
        self.assertIn("JWT_SECRET", env)  # a listed name is passed through from the dotenv
        for leaked in ("GRAFANA_PASSWORD", "FRAUDAI_SMOKE_PASSWORD", "METRICS_SCRAPE_TOKEN"):
            self.assertNotIn(leaked, env, f"{leaked} must not enter the API container environment")
        self.assertEqual(env["METRICS_SCRAPE_TOKEN_FILE"], "/run/secrets/fraudai_metrics_token")
        self.assertEqual(env["SANDBOX_MODE"], "disabled")
        self.assertNotIn("BUILD_SHA", env)

    # -- the mechanism, with throw-away containers ----------------------------------
    def read_secret_as(self, uid: str) -> str:
        probe = Path(self.tmp.name) / f"probe-{uid}.yml"
        probe.write_text(
            "name: f5a-secret-test\nservices:\n  reader:\n"
            f'    image: {PYTHON_IMAGE}\n    user: "{uid}"\n    secrets: [fraudai_metrics_token]\n'
            "    environment:\n      - EXPECT_SHA256\n"
            '    command: ["python", "-c", "import hashlib,os,stat; p=\'/run/secrets/fraudai_metrics_token\'; d=open(p,\'rb\').read(); '
            "print('match' if hashlib.sha256(d).hexdigest()==os.environ['EXPECT_SHA256'] else 'DIFFERENT', oct(stat.S_IMODE(os.stat(p).st_mode)))\"]\n"
            "secrets:\n  fraudai_metrics_token:\n    environment: METRICS_SCRAPE_TOKEN\n"
        )
        expected = hashlib.sha256(SYNTHETIC["METRICS_SCRAPE_TOKEN"].encode()).hexdigest()
        try:
            result = docker(
                "compose",
                "-f",
                str(probe),
                "--env-file",
                str(self.dotenv),
                "run",
                "--rm",
                "-T",
                "reader",
                env={"EXPECT_SHA256": expected},
            )
        finally:
            docker(
                "compose",
                "-f",
                str(probe),
                "--env-file",
                str(self.dotenv),
                "down",
                "-v",
                "--remove-orphans",
                check=False,
            )
        self.assertNotIn(SYNTHETIC["METRICS_SCRAPE_TOKEN"], result.stdout + result.stderr)
        return result.stdout.strip().splitlines()[-1]

    def test_missing_scrape_token_aborts_when_the_stack_is_started(self) -> None:
        """`config` does not validate the variable; starting a service that needs it does."""
        probe = Path(self.tmp.name) / "probe-missing.yml"
        probe.write_text(
            "name: f5a-secret-test\nservices:\n  reader:\n"
            f'    image: {PYTHON_IMAGE}\n    secrets: [fraudai_metrics_token]\n    command: ["true"]\n'
            "secrets:\n  fraudai_metrics_token:\n    environment: METRICS_SCRAPE_TOKEN\n"
        )
        empty = Path(self.tmp.name) / "empty.env"
        empty.write_text("GRAFANA_PASSWORD=x\n")
        try:
            result = docker(
                "compose",
                "-f",
                str(probe),
                "--env-file",
                str(empty),
                "run",
                "--rm",
                "-T",
                "reader",
                check=False,
            )
        finally:
            docker(
                "compose",
                "-f",
                str(probe),
                "--env-file",
                str(empty),
                "down",
                "-v",
                "--remove-orphans",
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("METRICS_SCRAPE_TOKEN", result.stderr + result.stdout)

    def test_dotenv_value_reaches_a_container_run_as_the_api_user(self) -> None:
        self.assertTrue(self.read_secret_as("1000").startswith("match"))

    def test_dotenv_value_reaches_a_container_run_as_prometheus_user(self) -> None:
        self.assertTrue(self.read_secret_as("65534").startswith("match"))


if __name__ == "__main__":
    unittest.main()
