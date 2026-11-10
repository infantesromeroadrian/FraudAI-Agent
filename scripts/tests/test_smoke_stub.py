"""Behaviour tests for scripts/smoke_test.py against a local STUB of the API contract.

The stub implements just enough of login / readyz / chat / upload / confirm (F4 and F5c
contracts) to show that the smoke test passes a correct server and FAILS on each
deviation that matters: implicit approval, execution while the sandbox is disabled, a
bad login, a served SHA that differs, token leakage. It is a simulation: these tests do
not prove the real stack works. No Git is involved.
Run: python3 -m unittest discover -s scripts/tests -v
"""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
import uuid
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "smoke_test.py"
TOKEN = "stub-token-" + uuid.uuid4().hex
SERVED_SHA = "d" * 40  # placeholder served identity for the double


class Stub:
    def __init__(self, **flaws: bool) -> None:
        self.flaws = flaws
        self.pending: dict[str, dict[str, object]] = {}
        self.owned: set[str] = set()
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _send(self, status: int, body: object) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self) -> bytes:
                return self.rfile.read(int(self.headers.get("Content-Length") or 0))

            def _authed(self) -> bool:
                return self.headers.get("Authorization") == f"Bearer {TOKEN}"

            def do_GET(self) -> None:  # noqa: N802
                if self.path.endswith("/readyz"):
                    sha = "e" * 40 if outer.flaws.get("other_sha") else SERVED_SHA
                    return self._send(
                        200,
                        {
                            "status": "ready",
                            "checks": {"qdrant": True, "llm": True},
                            "build_sha": sha,
                        },
                    )
                return self._send(404, {})

            def do_POST(self) -> None:  # noqa: N802
                raw = self._body()
                path = self.path.split("?")[0]
                if path.endswith("/auth/token"):
                    fields = dict(p.split("=", 1) for p in raw.decode().split("&"))
                    good = fields.get("password", "").startswith("smoke-password-")
                    if outer.flaws.get("bad_login") or not good:
                        return self._send(401, {"detail": "Incorrect username or password."})
                    return self._send(200, {"access_token": TOKEN, "token_type": "bearer"})
                if not self._authed():
                    return self._send(401, {"detail": "unauthorized"})
                if path.endswith("/files/upload"):
                    session = self.path.split("session_id=")[-1]
                    if session not in outer.owned:
                        return self._send(404, {"detail": "Session not found."})
                    return self._send(200, {"file_id": "f1", "filename": "smoke.csv"})
                payload = json.loads(raw or b"{}")
                if path.endswith("/chat"):
                    return self._chat(payload)
                if path.endswith("/confirm"):
                    return self._confirm(payload)
                return self._send(404, {})

            def _chat(self, payload: dict[str, object]) -> None:
                session = str(payload.get("session_id") or uuid.uuid4())
                outer.owned.add(session)
                if session in outer.pending:
                    if outer.flaws.get("implicit_approval"):
                        outer.pending.pop(session)
                        return self._send(
                            200, {"session_id": session, "message": "done", "pending_action": None}
                        )
                    return self._send(409, {"detail": {"code": "pending_action_required"}})
                if payload.get("agent_override") == "mike":
                    target = next(
                        w for w in str(payload["message"]).split() if w.startswith("http://")
                    )
                    action = {
                        "kind": "offensive_tool_approval",
                        "action_id": uuid.uuid4().hex,
                        "tool": "adversarial_evasion",
                        "arguments": {"target_endpoint": target},
                        "target": target,
                    }
                    outer.pending[session] = action
                    return self._send(
                        200,
                        {
                            "session_id": session,
                            "agent": "mike",
                            "message": "notice",
                            "pending_action": action,
                            "tool_results": [],
                        },
                    )
                return self._send(
                    200,
                    {
                        "session_id": session,
                        "agent": "donna",
                        "message": "I am working.",
                        "pending_action": None,
                    },
                )

            def _confirm(self, payload: dict[str, object]) -> None:
                session = str(payload.get("session_id"))
                action = outer.pending.get(session)
                if not isinstance(payload.get("approved"), bool):
                    return self._send(422, {"detail": "approved must be boolean"})
                if action is None:
                    return self._send(409, {"detail": {"code": "no_pending_action"}})
                if payload.get("action_id") != action["action_id"]:
                    return self._send(409, {"detail": {"code": "action_id_mismatch"}})
                outer.pending.pop(session)
                if outer.flaws.get("executes"):
                    urllib.request.urlopen(str(action["target"]), timeout=5).read()  # noqa: S310
                    return self._send(
                        200,
                        {
                            "session_id": session,
                            "tool_results": [{"status": "success", "result": {}}],
                        },
                    )
                result = {"status": "disabled", "error_code": "sandbox_disabled", "error": "off"}
                return self._send(
                    200,
                    {
                        "session_id": session,
                        "tool_results": [{"status": "error", "result": result}],
                    },
                )

            def log_message(self, *_: object) -> None:
                return

        socketserver.TCPServer.allow_reuse_address = True
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


SYNTHETIC_PASSWORD = "smoke-password-" + uuid.uuid4().hex


def run_smoke(
    stub: Stub,
    *extra: str,
    credentials: bool = True,
    via: str = "dotenv",
    dotenv_text: str | None = None,
) -> tuple[int, str, str, list[str]]:
    """Run the smoke script; returns (exit code, output, log text, the argv that was used).

    ``via="dotenv"`` (the documented way) puts the credentials only in a synthetic dotenv
    file; ``via="env"`` uses the internal process-variable interface.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FRAUDAI_SMOKE_", "AUTH_"))}
    with tempfile.TemporaryDirectory() as scratch:
        log = Path(scratch) / "smoke.log"
        dotenv = Path(scratch) / "synthetic.env"
        text = dotenv_text
        if text is None and credentials and via == "dotenv":
            text = f"AUTH_USERNAME=smoke-user\nFRAUDAI_SMOKE_PASSWORD={SYNTHETIC_PASSWORD}\n"
        if text is not None:
            dotenv.write_text(text)
        if credentials and via == "env":
            env.update(
                FRAUDAI_SMOKE_USERNAME="smoke-user", FRAUDAI_SMOKE_PASSWORD=SYNTHETIC_PASSWORD
            )
        argv = [
            sys.executable,
            str(SCRIPT),
            "--base-url",
            stub.url,
            "--log",
            str(log),
            "--env-file",
            str(dotenv),
            "--timeout",
            "10",
            *extra,
        ]
        proc = subprocess.run(  # noqa: S603
            argv, capture_output=True, text=True, env=env, timeout=120, check=False
        )
        return (
            proc.returncode,
            proc.stdout + proc.stderr,
            log.read_text() if log.exists() else "",
            argv,
        )


class SmokeAgainstStub(unittest.TestCase):
    def stub(self, **flaws: bool) -> Stub:
        stub = Stub(**flaws)
        self.addCleanup(stub.close)
        return stub

    def test_correct_server_passes_every_step(self) -> None:
        code, out, *_ = run_smoke(self.stub(), "--expect-sha", SERVED_SHA)
        self.assertEqual(code, 0, out)
        self.assertIn("synthetic target received no request", out)
        self.assertNotIn("[FAIL", out)

    def test_server_that_executes_despite_disabled_sandbox_fails(self) -> None:
        code, out, *_ = run_smoke(self.stub(executes=True))
        self.assertEqual(code, 1, out)
        self.assertRegex(out, r"\[FAIL\s*\] synthetic target received no request")
        self.assertRegex(
            out, r"\[FAIL\s*\] HITL: approval ends in an explicit sandbox_disabled error"
        )

    def test_implicit_approval_by_chat_message_fails(self) -> None:
        code, out, *_ = run_smoke(self.stub(implicit_approval=True))
        self.assertEqual(code, 1, out)
        self.assertRegex(out, r"\[FAIL\s*\] negative: a chat message does not approve")

    def test_rejected_login_stops_before_any_authenticated_step(self) -> None:
        code, out, *_ = run_smoke(self.stub(bad_login=True))
        self.assertEqual(code, 1, out)
        self.assertNotIn("chat (real LLM)", out)

    def test_a_different_served_sha_fails_when_one_is_expected(self) -> None:
        code, out, *_ = run_smoke(self.stub(other_sha=True), "--expect-sha", SERVED_SHA)
        self.assertEqual(code, 1, out)
        self.assertRegex(out, r"\[FAIL\s*\] readiness \(before\)")

    def test_missing_credentials_are_never_guessed(self) -> None:
        code, out, *_ = run_smoke(self.stub(), credentials=False)
        self.assertEqual(code, 3)
        self.assertIn("FRAUDAI_SMOKE_PASSWORD", out)
        self.assertIn(".env", out.replace("synthetic.env", ".env"))

    def test_credentials_are_read_from_the_dotenv_file_only(self) -> None:
        code, out, log, argv = run_smoke(self.stub(), "--skip-llm")
        self.assertEqual(code, 2, out)  # login worked (skip-llm => 2), so the file was used
        self.assertRegex(out, r"\[PASS\s*\] login")
        self.assertNotIn(SYNTHETIC_PASSWORD, out + log + " ".join(argv))

    def test_quoted_values_and_username_fallback_to_auth_username(self) -> None:
        text = f"# comment\nexport AUTH_USERNAME=\"smoke-user\"\nFRAUDAI_SMOKE_PASSWORD='{SYNTHETIC_PASSWORD}'\nOTHER=ignored\n"
        code, out, *_ = run_smoke(self.stub(), "--skip-llm", dotenv_text=text)
        self.assertEqual(code, 2, out)
        self.assertRegex(out, r"\[PASS\s*\] login")

    def test_a_dotenv_without_the_password_stops_before_any_request(self) -> None:
        code, out, *_ = run_smoke(self.stub(), dotenv_text="AUTH_USERNAME=smoke-user\n")
        self.assertEqual(code, 3)
        self.assertNotIn("readiness", out)

    def test_process_variables_remain_an_accepted_internal_interface(self) -> None:
        code, out, *_ = run_smoke(self.stub(), "--skip-llm", via="env", dotenv_text="")
        self.assertEqual(code, 2, out)
        self.assertRegex(out, r"\[PASS\s*\] login")

    def test_skipping_the_llm_is_reported_and_never_counts_as_success(self) -> None:
        code, out, *_ = run_smoke(self.stub(), "--skip-llm")
        self.assertEqual(code, 2, out)
        self.assertIn("SKIPPED", out)

    def test_log_and_output_never_contain_the_token_or_password(self) -> None:
        code, out, log, argv = run_smoke(self.stub())
        self.assertEqual(code, 0, out)
        for text in (out, log):
            self.assertNotIn(TOKEN, text)
            self.assertNotIn("smoke-password-", text)
        self.assertFalse(
            [a for a in argv if "smoke-password-" in a], "the password must never be in argv"
        )


if __name__ == "__main__":
    unittest.main()
