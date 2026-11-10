#!/usr/bin/env python3
"""End-to-end smoke test of the local stack: login -> chat -> upload -> HITL -> health.

Everything it sends stays on loopback and on a synthetic target served by this very
script, so no external host is ever contacted. The log never contains tokens or
credentials (the bearer token lives only in memory; it is masked in every line).

Credentials come from the local ``.env`` (the only persistent place for secrets; see
.env.example): FRAUDAI_SMOKE_PASSWORD, and FRAUDAI_SMOKE_USERNAME or else AUTH_USERNAME. The
file is read by this process only: nothing goes through argv or the shell, and no value is
ever printed or logged. Process variables of the same names are accepted as an internal
interface (they take precedence) but are not the documented way to provide secrets.

Exit status: 0 only when every step passed (SKIPPED steps are reported, never counted
as passed). Standard library only; runs with the system python3 (>= 3.9).
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

API = "/api/v1"
DEFAULT_ENV_FILE = Path(
    os.environ.get("FRAUDAI_ENV_FILE") or Path(__file__).resolve().parent.parent / ".env"
)
SYNTHETIC_CSV = (
    "transaction_id,amount,merchant,country\n"
    "t-001,12.50,synthetic-shop,ES\n"
    "t-002,9800.00,synthetic-exchange,ES\n"
    "t-003,15.00,synthetic-cafe,ES\n"
)


def read_dotenv(path: Path, wanted: set[str]) -> dict[str, str]:
    """Values of the ``wanted`` keys in a dotenv file; nothing is evaluated or expanded."""
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        if key not in wanted:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def resolve_credentials(env_file: Path) -> tuple[str, str]:
    """(username, password) from process variables, else from the dotenv file."""
    keys = {"FRAUDAI_SMOKE_USERNAME", "FRAUDAI_SMOKE_PASSWORD", "AUTH_USERNAME"}
    from_file = read_dotenv(env_file, keys)
    username = (
        os.environ.get("FRAUDAI_SMOKE_USERNAME")
        or from_file.get("FRAUDAI_SMOKE_USERNAME")
        or from_file.get("AUTH_USERNAME")
        or ""
    )
    password = (
        os.environ.get("FRAUDAI_SMOKE_PASSWORD") or from_file.get("FRAUDAI_SMOKE_PASSWORD") or ""
    )
    return username, password


class SyntheticTarget:
    """A local HTTP service that only counts the requests it receives."""

    def __init__(self) -> None:
        self.hits: list[str] = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _answer(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                outer.hits.append(f"{self.command} {self.path}")
                body = b'{"prediction": 0}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = do_POST = _answer  # noqa: N815

            def log_message(self, *_: object) -> None:
                return

        socketserver.TCPServer.allow_reuse_address = True
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class Smoke:
    def __init__(self, base_url: str, log_path: str | None, timeout: float) -> None:
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.token: str | None = None
        self.results: list[tuple[str, str, str]] = []
        self._log = open(log_path, "w", encoding="utf-8") if log_path else None  # noqa: SIM115
        self._secrets: set[str] = set()

    # -- logging (secrets masked) ------------------------------------------------
    def say(self, text: str) -> None:
        for secret in self._secrets:
            if secret:
                text = text.replace(secret, "***")
        line = text
        print(line, flush=True)
        if self._log:
            self._log.write(line + "\n")
            self._log.flush()

    def record(self, name: str, status: str, detail: str = "") -> bool:
        self.results.append((name, status, detail))
        self.say(f"[{status:7}] {name}" + (f" -- {detail}" if detail else ""))
        return status == "PASS"

    # -- HTTP ---------------------------------------------------------------------
    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: object | None = None,
        form: dict[str, str] | None = None,
        raw: tuple[bytes, str] | None = None,
        auth: bool = True,
        token: str | None = None,
    ) -> tuple[int, object]:
        headers: dict[str, str] = {}
        data: bytes | None = None
        if json_body is not None:
            data, headers["Content-Type"] = json.dumps(json_body).encode(), "application/json"
        elif form is not None:
            data = urllib.parse.urlencode(form).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        elif raw is not None:
            data, headers["Content-Type"] = raw
        bearer = token if token is not None else (self.token if auth else None)
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                status, payload = resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            status, payload = exc.code, exc.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return 0, {"transport_error": type(exc).__name__}
        try:
            return status, json.loads(payload or b"null")
        except ValueError:
            return status, {"non_json_body_bytes": len(payload)}

    # -- steps --------------------------------------------------------------------
    def health(self, label: str, expect_sha: str | None = None) -> bool:
        """GET /readyz (F5c): 200 = ready; build_sha is the revision the API serves."""
        status, body = self.request("GET", f"{API}/readyz", auth=False)
        if not isinstance(body, dict):
            return self.record(label, "FAIL", f"HTTP {status}")
        served = body.get("build_sha")
        detail = f"HTTP {status} status={body.get('status')} checks={body.get('checks')} build_sha={str(served)[:12]}"
        ok = status == 200 and body.get("status") == "ready"
        if expect_sha is not None:
            ok = ok and served == expect_sha
            detail += f" expected={expect_sha[:12]}"
        return self.record(label, "PASS" if ok else "FAIL", detail)

    def unauthenticated_is_rejected(self) -> bool:
        status, _ = self.request("POST", f"{API}/chat", json_body={"message": "x"}, auth=False)
        bad_status, _ = self.request(
            "POST", f"{API}/chat", json_body={"message": "x"}, token="not-a-valid-token"
        )
        ok = status in (401, 403) and bad_status in (401, 403)
        return self.record(
            "negative: chat without / with a bogus token is rejected",
            "PASS" if ok else "FAIL",
            f"no token -> {status}, bogus token -> {bad_status}",
        )

    def wrong_password_is_rejected(self, username: str) -> bool:
        status, body = self.request(
            "POST",
            f"{API}/auth/token",
            form={"username": username, "password": "not-the-password-" + uuid.uuid4().hex},
            auth=False,
        )
        leaked = isinstance(body, dict) and "access_token" in body
        return self.record(
            "negative: a wrong password is rejected (401)",
            "PASS" if status == 401 and not leaked else "FAIL",
            f"HTTP {status}",
        )

    def login(self, username: str, password: str) -> bool:
        status, body = self.request(
            "POST",
            f"{API}/auth/token",
            form={"username": username, "password": password},
            auth=False,
        )
        token = body.get("access_token") if isinstance(body, dict) else None
        if status != 200 or not isinstance(token, str) or not token:
            return self.record("login", "FAIL", f"HTTP {status}")
        self.token = token
        self._secrets.add(token)  # from now on masked in every line
        return self.record("login", "PASS", "token kept in memory only")

    def chat(self, message: str, **extra: object) -> tuple[int, dict[str, object]]:
        body: dict[str, object] = {"message": message, "language": "en", **extra}
        status, payload = self.request("POST", f"{API}/chat", json_body=body)
        return status, payload if isinstance(payload, dict) else {}

    def plain_chat(self) -> str | None:
        status, body = self.chat("Reply with one short sentence confirming you are working.")
        text = str(body.get("message") or "")
        session_id = str(body.get("session_id") or "")
        ok = status == 200 and bool(text.strip()) and bool(session_id)
        self.record(
            "chat (real LLM)",
            "PASS" if ok else "FAIL",
            f"HTTP {status}, agent={body.get('agent')}, reply_chars={len(text)}",
        )
        return session_id if ok else None

    def upload(self, session_id: str) -> bool:
        boundary = uuid.uuid4().hex
        payload = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="smoke.csv"\r\nContent-Type: text/csv\r\n\r\n{SYNTHETIC_CSV}\r\n'
            f"--{boundary}--\r\n"
        ).encode()
        ctype = f"multipart/form-data; boundary={boundary}"
        query = urllib.parse.urlencode({"session_id": session_id})
        status, body = self.request("POST", f"{API}/files/upload?{query}", raw=(payload, ctype))
        file_id = body.get("file_id") if isinstance(body, dict) else None
        ok = self.record(
            "upload (synthetic CSV)",
            "PASS" if status == 200 and file_id else "FAIL",
            f"HTTP {status}",
        )
        foreign = urllib.parse.urlencode({"session_id": str(uuid.uuid4())})
        status, _ = self.request("POST", f"{API}/files/upload?{foreign}", raw=(payload, ctype))
        return (
            self.record(
                "negative: upload to a session that is not yours is 404",
                "PASS" if status == 404 else "FAIL",
                f"HTTP {status}",
            )
            and ok
        )

    # -- HITL -----------------------------------------------------------------------
    def propose(self, target_url: str, attempts: int) -> tuple[str, dict[str, object]] | None:
        prompt = (
            "Use your adversarial_evasion tool against the model endpoint "
            f"{target_url} with attack_type sign_noise. Call the tool now."
        )
        for attempt in range(1, attempts + 1):
            status, body = self.chat(prompt, agent_override="mike")
            pending = body.get("pending_action")
            if status == 200 and isinstance(pending, dict):
                return str(body["session_id"]), pending
            self.say(
                f"  attempt {attempt}: no pending_action (HTTP {status}); the model did not call the tool"
            )
        return None

    def confirm(
        self, session_id: str, action_id: str, approved: object
    ) -> tuple[int, dict[str, object]]:
        status, body = self.request(
            "POST",
            f"{API}/confirm",
            json_body={"session_id": session_id, "action_id": action_id, "approved": approved},
        )
        return status, body if isinstance(body, dict) else {}

    def hitl(self, target: SyntheticTarget, attempts: int, expect_sandbox: str) -> bool:
        url = f"http://127.0.0.1:{target.port}/predict"
        proposal = self.propose(url, attempts)
        if proposal is None:
            return self.record(
                "HITL: offensive tool is proposed, not run",
                "FAIL",
                f"no pending_action in {attempts} attempts",
            )
        session_id, pending = proposal
        action_id = str(pending.get("action_id") or "")
        ok = self.record(
            "HITL: offensive tool is proposed, not run",
            "PASS" if pending.get("kind") == "offensive_tool_approval" and action_id else "FAIL",
            f"tool={pending.get('tool')} target={pending.get('target')}",
        )

        # nothing may be approved implicitly: a new message while paused is refused
        status, body = self.chat("go ahead", session_id=session_id)
        detail = body.get("detail") if isinstance(body.get("detail"), dict) else {}
        ok &= self.record(
            "negative: a chat message does not approve the pending action",
            "PASS" if status == 409 and detail.get("code") == "pending_action_required" else "FAIL",
            f"HTTP {status} code={detail.get('code') if isinstance(detail, dict) else '-'}",
        )
        status, body = self.confirm(session_id, "wrong-action-id", True)
        detail = body.get("detail") if isinstance(body.get("detail"), dict) else {}
        ok &= self.record(
            "negative: a wrong action_id is refused and consumes nothing",
            "PASS"
            if status == 409
            and isinstance(detail, dict)
            and detail.get("code") == "action_id_mismatch"
            else "FAIL",
            f"HTTP {status}",
        )
        status, _ = self.confirm(session_id, action_id, "true")
        ok &= self.record(
            "negative: approval must be a JSON boolean",
            "PASS" if status == 422 else "FAIL",
            f"HTTP {status}",
        )

        # the single real decision: approve exactly what was shown
        status, body = self.confirm(session_id, action_id, True)
        results = body.get("tool_results") if isinstance(body.get("tool_results"), list) else []
        first = results[0] if results and isinstance(results[0], dict) else {}
        inner = first.get("result") if isinstance(first.get("result"), dict) else {}
        if expect_sandbox == "disabled":
            good = (
                status == 200
                and first.get("status") == "error"
                and inner.get("error_code") == "sandbox_disabled"
            )
            what = "approval ends in an explicit sandbox_disabled error"
        else:
            good = status == 200 and bool(results)
            what = "approval executes"
        ok &= self.record(
            f"HITL: {what}",
            "PASS" if good else "FAIL",
            f"HTTP {status} tool_status={first.get('status')} error_code={inner.get('error_code')}",
        )
        status, body = self.confirm(session_id, action_id, True)
        detail = body.get("detail") if isinstance(body.get("detail"), dict) else {}
        ok &= self.record(
            "negative: repeating the same approval is refused",
            "PASS"
            if status == 409
            and isinstance(detail, dict)
            and detail.get("code") == "no_pending_action"
            else "FAIL",
            f"HTTP {status}",
        )
        time.sleep(0.3)
        ok &= self.record(
            "synthetic target received no request (zero execution)",
            "PASS" if not target.hits else "FAIL",
            f"requests={len(target.hits)}",
        )
        return ok


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="dotenv holding the credentials (default: ./.env)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="default: http://127.0.0.1:<FRAUDAI_PORT from .env, else 8000>",
    )
    parser.add_argument("--log", default=os.environ.get("FRAUDAI_SMOKE_LOG"))
    parser.add_argument(
        "--timeout", type=float, default=180.0, help="per-request seconds (real LLM calls are slow)"
    )
    parser.add_argument(
        "--expect-sha",
        default=os.environ.get("FRAUDAI_SMOKE_EXPECT_SHA"),
        help="fail unless /readyz serves exactly this build_sha (full commit SHA)",
    )
    parser.add_argument("--hitl-attempts", type=int, default=3)
    parser.add_argument(
        "--expect-sandbox",
        choices=["disabled"],
        default="disabled",
        help="Compose runs with SANDBOX_MODE=disabled; the approval must end in sandbox_disabled",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="skip the real-LLM steps; they are reported as SKIPPED and the run exits 2",
    )
    args = parser.parse_args(argv)

    env_file = Path(args.env_file)
    username, password = resolve_credentials(env_file)
    if not username or not password:
        print(
            f"Missing smoke credentials. Put FRAUDAI_SMOKE_PASSWORD (and AUTH_USERNAME or FRAUDAI_SMOKE_USERNAME) "
            f"in {env_file}; not guessing credentials, nothing was sent.",
            file=sys.stderr,
        )
        return 3
    if args.base_url is None:
        port = (
            read_dotenv(env_file, {"FRAUDAI_PORT"}).get("FRAUDAI_PORT")
            or os.environ.get("FRAUDAI_PORT")
            or "8000"
        )
        args.base_url = os.environ.get("FRAUDAI_SMOKE_BASE_URL") or f"http://127.0.0.1:{port}"

    smoke = Smoke(args.base_url, args.log, args.timeout)
    smoke._secrets.add(password)
    smoke.say(
        f"FraudAI smoke test against {args.base_url} ({time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})"
    )
    target = SyntheticTarget()
    try:
        alive = smoke.health("readiness (before)", args.expect_sha)
        smoke.unauthenticated_is_rejected()
        smoke.wrong_password_is_rejected(username)
        if alive and smoke.login(username, password):
            if args.skip_llm:
                smoke.record("chat (real LLM)", "SKIPPED", "--skip-llm")
                smoke.record("HITL", "SKIPPED", "--skip-llm")
            else:
                session_id = smoke.plain_chat()
                if session_id:
                    smoke.upload(session_id)
                smoke.hitl(target, args.hitl_attempts, args.expect_sandbox)
        smoke.health("readiness (after)", args.expect_sha)
    finally:
        target.close()

    failed = [name for name, status, _ in smoke.results if status == "FAIL"]
    skipped = [name for name, status, _ in smoke.results if status == "SKIPPED"]
    passed = sum(1 for _, status, _ in smoke.results if status == "PASS")
    smoke.say(f"SUMMARY: {passed} passed, {len(failed)} failed, {len(skipped)} skipped")
    if failed:
        return 1
    return 2 if skipped else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
