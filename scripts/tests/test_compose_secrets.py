"""Secrets contract of the production Compose file (synthetic values only, real containers).

The local .env is the only persistent place for secrets. The metrics scrape credential is a
variable: the API (read_only: true, so Compose `secrets` cannot be attached to it) and
Prometheus both receive METRICS_SCRAPE_TOKEN; Prometheus' static entrypoint writes its
bearer file into a private tmpfs and unsets the variable before exec.

Every container test uses its OWN Compose project and labels, synthetic values, the
pinned local images and no application image; cleanup is limited to those ids. A stand-in
for the API runs with the same hardening flags as the real service (read-only rootfs,
non-root, no capabilities). Docker (with the compose plugin) is REQUIRED: tests fail, not
skip, without it.
Run: python3 -m unittest discover -s scripts/tests -v
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import tempfile
import time
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
COMPOSE = ROOT / "docker-compose.prod.yml"
ENTRYPOINT = ROOT / "monitoring" / "prometheus" / "entrypoint.sh"
PYTHON_IMAGE = "python:3.11.18-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39"
PROM_IMAGE = "prom/prometheus:v2.53.0@sha256:075b1ba2c4ebb04bc3a6ab86c06ec8d8099f8fda1c96ef6d104d9bb1def1d8bc"
SYNTHETIC = {
    "GRAFANA_PASSWORD": "synthetic-grafana-" + uuid.uuid4().hex,
    "METRICS_SCRAPE_TOKEN": "synthetic-metrics-" + uuid.uuid4().hex,
    "FRAUDAI_SMOKE_PASSWORD": "synthetic-smoke-" + uuid.uuid4().hex,
    "JWT_SECRET": "synthetic-jwt-" + uuid.uuid4().hex,
}
STUB_API = (
    "import http.server,os;T=os.environ['METRICS_SCRAPE_TOKEN']\n"
    "class H(http.server.BaseHTTPRequestHandler):\n"
    "    def do_GET(s):\n"
    "        ok=s.path=='/api/v1/admin/metrics' and s.headers.get('Authorization')=='Bearer '+T\n"
    "        s.send_response(200 if ok else 401);s.send_header('Content-Type','text/plain');s.end_headers()\n"
    "        s.wfile.write(b'fraudai_active_sessions 1\\n' if ok else b'no')\n"
    "    def log_message(*a):pass\n"
    "http.server.ThreadingHTTPServer(('0.0.0.0',8000),H).serve_forever()\n"
)


def docker(
    *args: str,
    env: dict[str, str] | None = None,
    check: bool = True,
    timeout: int = 240,
) -> subprocess.CompletedProcess[str]:
    base = {k: v for k, v in os.environ.items() if k not in SYNTHETIC}
    return subprocess.run(  # noqa: S603
        ["docker", *args],
        capture_output=True,
        text=True,
        env={**base, **(env or {})},
        timeout=timeout,
        check=check,
    )


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ComposeContract(unittest.TestCase):
    """What the real docker-compose.prod.yml declares (rendered, never started)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dotenv = Path(cls.tmp.name) / "synthetic.env"
        cls.dotenv.write_text("".join(f"{k}={v}\n" for k, v in SYNTHETIC.items()))
        cls.text = COMPOSE.read_text()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def rendered(self, dotenv: Path | None = None, **env: str) -> dict:
        out = docker(
            "compose",
            "-f",
            str(COMPOSE),
            "--env-file",
            str(dotenv or self.dotenv),
            "config",
            "--format",
            "json",
            env={"FRAUDAI_IMAGE": "fraudai-agent:t", "FRONTEND_IMAGE": "fraudai-frontend:t", **env},
        )
        return json.loads(out.stdout)

    def test_the_api_stays_read_only_without_socket_or_host_network(self) -> None:
        api = self.rendered()["services"]["fraudai"]
        self.assertIs(api["read_only"], True)
        self.assertNotIn("network_mode", api)
        self.assertNotIn("docker.sock", json.dumps(api))
        self.assertEqual(api["cap_drop"], ["ALL"])

    def test_no_compose_secret_is_used_anywhere(self) -> None:
        rendered = self.rendered()
        self.assertNotIn("secrets", rendered)
        for name, service in rendered["services"].items():
            self.assertNotIn("secrets", service, f"{name} must not use Compose secrets")

    def test_the_api_gets_an_allow_list_with_the_scrape_variable_only(self) -> None:
        api = self.rendered()["services"]["fraudai"]
        self.assertNotIn("env_file", api)
        env = api["environment"]
        self.assertEqual(env["METRICS_SCRAPE_TOKEN"], SYNTHETIC["METRICS_SCRAPE_TOKEN"])
        self.assertIn("JWT_SECRET", env)
        for leaked in ("GRAFANA_PASSWORD", "FRAUDAI_SMOKE_PASSWORD"):
            self.assertNotIn(leaked, env, f"{leaked} must not enter the API container")
        self.assertNotIn("METRICS_SCRAPE_TOKEN_FILE", env)
        self.assertEqual(env["SANDBOX_MODE"], "disabled")
        self.assertNotIn("BUILD_SHA", env)

    def test_prometheus_gets_the_variable_a_tmpfs_and_the_static_entrypoint(self) -> None:
        prom = self.rendered()["services"]["prometheus"]
        self.assertEqual(prom["entrypoint"], ["/bin/sh", "/etc/prometheus/entrypoint.sh"])
        self.assertEqual(
            prom["environment"], {"METRICS_SCRAPE_TOKEN": SYNTHETIC["METRICS_SCRAPE_TOKEN"]}
        )
        tmpfs = json.dumps(prom.get("tmpfs") or prom.get("volumes"))
        self.assertIn("/run/prometheus-credentials", tmpfs)
        self.assertNotIn("enable-lifecycle", " ".join(prom["command"]))
        self.assertNotIn("docker.sock", json.dumps(prom))
        self.assertTrue(all(p["host_ip"] == "127.0.0.1" for p in prom["ports"]))

    def test_the_entrypoint_contains_no_secret_and_unsets_before_exec(self) -> None:
        text = ENTRYPOINT.read_text()
        self.assertLess(
            text.index("unset METRICS_SCRAPE_TOKEN"), text.index("exec /bin/prometheus")
        )
        self.assertIn("umask 077", text)
        for value in SYNTHETIC.values():
            self.assertNotIn(value, text)

    def test_a_missing_scrape_token_aborts_compose(self) -> None:
        empty = Path(self.tmp.name) / "no-token.env"
        empty.write_text("GRAFANA_PASSWORD=x\n")
        result = docker(
            "compose",
            "-f",
            str(COMPOSE),
            "--env-file",
            str(empty),
            "config",
            "-q",
            check=False,
            env={"FRAUDAI_IMAGE": "a:b", "FRONTEND_IMAGE": "c:d"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("METRICS_SCRAPE_TOKEN", result.stderr)

    def test_ports_follow_the_dotenv_and_a_process_variable_wins(self) -> None:
        dotenv = Path(self.tmp.name) / "ports.env"
        dotenv.write_text(
            "GRAFANA_PASSWORD=x\nMETRICS_SCRAPE_TOKEN=y\nFRAUDAI_PORT=18080\nFRONTEND_PORT=13000\n"
        )
        services = self.rendered(dotenv)["services"]
        published = lambda n: sorted(str(p["published"]) for p in services[n]["ports"])  # noqa: E731
        self.assertEqual(published("fraudai"), ["18080"])
        self.assertEqual(published("frontend"), ["13000"])
        self.assertIn(
            "http://localhost:13000", services["fraudai"]["environment"]["CORS_ALLOWED_ORIGINS"]
        )
        self.assertEqual(
            sorted(
                str(p["published"])
                for p in self.rendered(dotenv, FRAUDAI_PORT="19090")["services"]["fraudai"]["ports"]
            ),
            ["19090"],
        )

    def test_project_name_matches_the_label_the_scripts_query(self) -> None:
        self.assertEqual(self.rendered()["name"], "fraudai-local")


class PrometheusCredentialInRealContainers(unittest.TestCase):
    """The real Prometheus service definition, started alone in its own project."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.project = "f5a-cred-" + uuid.uuid4().hex[:8]
        cls.dotenv = Path(cls.tmp.name) / "synthetic.env"
        cls.dotenv.write_text("".join(f"{k}={v}\n" for k, v in SYNTHETIC.items()))
        cls.stubs: list[str] = []
        cls.env = {
            "FRAUDAI_IMAGE": PYTHON_IMAGE,
            "FRONTEND_IMAGE": PYTHON_IMAGE,
            "PROMETHEUS_PORT": str(free_port()),
        }

    @classmethod
    def tearDownClass(cls) -> None:
        for stub in cls.stubs:
            docker("rm", "-f", stub, check=False)
        docker(
            "compose",
            "-p",
            cls.project,
            "-f",
            str(COMPOSE),
            "--env-file",
            str(cls.dotenv),
            "down",
            "-v",
            "--remove-orphans",
            check=False,
            env=cls.env,
        )
        cls.tmp.cleanup()

    @classmethod
    def compose(cls, *args: str) -> subprocess.CompletedProcess[str]:
        return docker(
            "compose",
            "-p",
            cls.project,
            "-f",
            str(COMPOSE),
            "--env-file",
            str(cls.dotenv),
            *args,
            env=cls.env,
        )

    def start_api_stand_in(self, token: str) -> str:
        """Same hardening as the real API service: read-only rootfs, non-root, no caps."""
        name = f"{self.project}-api-{uuid.uuid4().hex[:6]}"
        docker(
            "run",
            "-d",
            "--name",
            name,
            "--label",
            f"f5a.test={self.project}",
            "--network",
            f"{self.project}_default",
            "--network-alias",
            "fraudai",
            "--read-only",
            "--user",
            "1000:1000",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "-e",
            "METRICS_SCRAPE_TOKEN",
            PYTHON_IMAGE,
            "python",
            "-c",
            STUB_API,
            env={"METRICS_SCRAPE_TOKEN": token},
        )
        self.stubs.append(name)
        return name

    def query_up(self, container: str) -> str:
        out = docker(
            "exec",
            container,
            "wget",
            "-qO-",
            "http://localhost:9090/api/v1/query?query=up%7Bjob%3D%22fraudai%22%7D",
            check=False,
        )
        try:
            result = json.loads(out.stdout)["data"]["result"]
            return str(result[0]["value"][1]) if result else "absent"
        except (ValueError, KeyError, IndexError):
            return "error"

    def wait_for_up(self, container: str, wanted: str, seconds: int = 90) -> str:
        deadline = time.monotonic() + seconds
        value = "error"
        while time.monotonic() < deadline:
            value = self.query_up(container)
            if value == wanted:
                return value
            time.sleep(3)
        return value

    def test_credential_reaches_prometheus_through_a_private_tmpfs_and_authenticates(self) -> None:
        token = SYNTHETIC["METRICS_SCRAPE_TOKEN"]
        self.compose("up", "-d", "--no-deps", "prometheus")
        prom = self.compose("ps", "-q", "prometheus").stdout.strip()
        self.assertTrue(prom)

        # 1. the file exists in the tmpfs, private to the Prometheus user, and equals the token
        shown = docker(
            "exec", prom, "sh", "-c", "sha256sum /run/prometheus-credentials/metrics-token"
        ).stdout
        self.assertEqual(shown.split()[0], hashlib.sha256(token.encode()).hexdigest())
        modes = docker(
            "exec",
            prom,
            "sh",
            "-c",
            "stat -c '%a %u %n' /run/prometheus-credentials /run/prometheus-credentials/metrics-token",
        ).stdout
        self.assertEqual(modes.split("\n")[0].split()[:2], ["700", "65534"])
        self.assertEqual(modes.split("\n")[1].split()[:2], ["600", "65534"])
        self.assertIn(
            "tmpfs",
            docker(
                "exec", prom, "sh", "-c", "grep ' /run/prometheus-credentials ' /proc/mounts"
            ).stdout,
        )

        # 2. the variable is gone from the running process and nothing printed it
        environ = docker("exec", prom, "sh", "-c", "tr '\\0' '\\n' < /proc/1/environ").stdout
        self.assertNotIn("METRICS_SCRAPE_TOKEN", environ)
        self.assertNotIn(token, docker("logs", prom).stdout + docker("logs", prom).stderr)

        # 3. HTTP authentication, against a read-only non-root API stand-in
        good = self.start_api_stand_in(token)
        self.assertEqual(self.wait_for_up(prom, "1"), "1", "Prometheus could not authenticate")
        self.assertNotIn(token, docker("logs", good).stdout + docker("logs", good).stderr)
        flags = docker(
            "inspect",
            "--format",
            "{{.HostConfig.ReadonlyRootfs}} {{.Config.User}} {{.HostConfig.NetworkMode}}",
            good,
        ).stdout.split()
        self.assertEqual(flags[0], "true")
        self.assertEqual(flags[1], "1000:1000")
        self.assertNotEqual(flags[2], "host")

        # 4. negative: a target that expects a DIFFERENT token is not scraped successfully
        docker("rm", "-f", good)
        self.start_api_stand_in("a-different-synthetic-token-0123456789abcdef")
        self.assertEqual(self.wait_for_up(prom, "0"), "0", "a wrong token must not authenticate")

        # 5. flags of the real service: loopback binding, no lifecycle endpoint, no socket, no host network
        inspected = json.loads(docker("inspect", prom).stdout)[0]
        self.assertNotIn(
            "--web.enable-lifecycle", inspected["Args"] + inspected["Config"].get("Cmd", [])
        )
        self.assertNotEqual(inspected["HostConfig"]["NetworkMode"], "host")
        self.assertFalse(inspected["HostConfig"]["Privileged"])
        for binding in inspected["HostConfig"]["PortBindings"].values():
            self.assertEqual({b["HostIp"] for b in binding}, {"127.0.0.1"})
        self.assertFalse([m for m in inspected["Mounts"] if "docker.sock" in m.get("Source", "")])

    def test_a_missing_or_short_token_stops_the_entrypoint_without_printing_it(self) -> None:
        short = "short-synthetic-value"
        result = docker(
            "run",
            "--rm",
            "--entrypoint",
            "/bin/sh",
            "-e",
            "METRICS_SCRAPE_TOKEN",
            "--tmpfs",
            "/run/prometheus-credentials:mode=0700,uid=65534,gid=65534,size=64k",
            "-v",
            f"{ENTRYPOINT}:/etc/prometheus/entrypoint.sh:ro",
            PROM_IMAGE,
            "/etc/prometheus/entrypoint.sh",
            check=False,
            env={"METRICS_SCRAPE_TOKEN": short},
        )
        self.assertEqual(result.returncode, 64)
        self.assertNotIn(short, result.stdout + result.stderr)
        self.assertIn("METRICS_SCRAPE_TOKEN", result.stderr)


if __name__ == "__main__":
    unittest.main()
