"""Regression A3: red-teaming HTTP clients must not follow redirects to private hosts.

Audit finding: ``_validate_target_endpoint`` checks only the *initial* URL,
but the code the tools generate calls ``urllib.request.urlopen``, which follows
3xx responses, so a public-looking target can bounce the sandbox to an
internal address (metadata service, LAN, localhost).

This test is RED on purpose until redirects are not followed or are
re-validated. What is real: ``ToolImplementations.adversarial_evasion`` /
``prompt_injection_suite``, the code they build, and the stdlib ``urllib``
client inside that code, executed in a separate Python process.

Documented limits of the harness (all sockets stay on 127.0.0.1):
- ``_resolve_host`` is replaced so the initial loopback URL is *treated as*
  public by the validator; the validator's IP policy itself still runs.
- The sandbox is replaced by ``_LocalProcessSandbox``, which runs the
  generated script with the host interpreter and only remaps
  ``/workspace/output``. This does NOT prove anything about Docker isolation
  or egress policy (that is F3.5).
- The "private" destination is a second loopback listener on another port.
- The child process installs a ``sys.addaudithook`` that rejects any socket
  connect/sendto whose address is not one of the two synthetic listeners, so
  generated code can never reach 93.184.216.34 or any other local service.
  urllib and its redirect handling are untouched. If a future fix pins the
  resolved IP (F3), only this harness may need adapting (e.g. mapping the
  simulated public address to the first listener); the assertion stays.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any

import pytest

from fraudai.tools.implementations import ToolImplementations
from fraudai.tools.sandbox import SandboxResult

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

# A global address literal fed only to the validator's IP policy; it is never
# contacted (the documentation ranges are not global, so they would be blocked).
SIMULATED_PUBLIC_ADDRESS = "93.184.216.34"
SCRIPT_TIMEOUT_SECONDS = 120


class _Listener:
    """Loopback HTTP server that records the paths it receives."""

    def __init__(self, redirect_to: str | None = None) -> None:
        self.requests: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                outer.requests.append(f"{self.command} {self.path}")
                if redirect_to is None:
                    body = b'{"prediction": 0, "response": "ok"}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    # 302 on POST: urllib re-issues it as GET and follows it.
                    self.send_response(302)
                    self.send_header("Location", redirect_to)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

            do_GET = do_POST = _handle  # noqa: N815

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> _Listener:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


_SOCKET_GUARD = """\
import sys as _sys

_ALLOWED_ADDRESSES = {allowed!r}


def _socket_guard(event, args):
    if event in ("socket.connect", "socket.sendto"):
        address = args[1]
        if not (isinstance(address, tuple) and tuple(address[:2]) in _ALLOWED_ADDRESSES):
            raise OSError("test harness blocked non-listener address: " + repr(address))


_sys.addaudithook(_socket_guard)
"""


class _LocalProcessSandbox:
    """Stand-in for ``SandboxEngine.execute_python`` that runs code locally."""

    def __init__(self, workdir: Path, allowed_ports: tuple[int, ...]) -> None:
        self._workdir = workdir
        self._allowed = {("127.0.0.1", port) for port in allowed_ports}

    async def execute_python(self, code: str, timeout: int = 300, **_: Any) -> SandboxResult:
        output_dir = self._workdir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        script = self._workdir / "generated_tool_code.py"
        guard = _SOCKET_GUARD.format(allowed=self._allowed)
        script.write_text(guard + code.replace("/workspace/output", str(output_dir)))

        started = time.monotonic()
        proc = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-I", str(script)],
            capture_output=True,
            text=True,
            timeout=min(timeout, SCRIPT_TIMEOUT_SECONDS),
            cwd=self._workdir,
            check=False,
        )
        result_file = output_dir / "result.json"
        return SandboxResult(
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            output_files={"result.json": result_file.read_bytes()} if result_file.exists() else {},
            duration_seconds=time.monotonic() - started,
        )


@pytest.fixture()
def listeners() -> Iterator[tuple[_Listener, _Listener]]:
    with _Listener() as private_target:
        redirect = f"http://127.0.0.1:{private_target.port}/internal-only"
        with _Listener(redirect_to=redirect) as public_looking:
            yield public_looking, private_target


async def test_generated_http_client_does_not_follow_redirect_to_private_host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    listeners: tuple[_Listener, _Listener],
) -> None:
    public_looking, private_target = listeners
    monkeypatch.setattr(
        ToolImplementations,
        "_resolve_host",
        staticmethod(lambda hostname: [SIMULATED_PUBLIC_ADDRESS]),
    )
    tools = ToolImplementations(
        sandbox=object(),  # type: ignore[arg-type]  # never used by these two tools
        sandbox_networked=_LocalProcessSandbox(  # type: ignore[arg-type]
            tmp_path, allowed_ports=(public_looking.port, private_target.port)
        ),
    )
    endpoint = f"http://127.0.0.1:{public_looking.port}/predict"

    evasion = await tools.adversarial_evasion(endpoint)
    after_evasion = list(private_target.requests)
    injection = await tools.prompt_injection_suite(endpoint)

    # Fixture sanity: the validated entry point was contacted by the real
    # generated client, so a missing hit below cannot be a broken harness.
    assert evasion["status"] == "success", evasion
    assert injection["status"] == "success", injection
    assert public_looking.requests, "initial endpoint was never contacted"

    leaked = len(private_target.requests)
    assert leaked == 0, (
        "generated urllib client followed a redirect to a private host: "
        f"{len(after_evasion)} request(s) from adversarial_evasion, {leaked} in total "
        f"(first: {private_target.requests[0]!r})"
    )
