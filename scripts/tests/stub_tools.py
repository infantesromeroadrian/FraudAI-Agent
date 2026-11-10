#!/usr/bin/env python3
"""SIMULATED docker / git / curl used by test_deploy_rollback_doubles.py.

Nothing here talks to a real Docker daemon or a real repository: it models only the
handful of commands scripts/lib.sh, deploy.sh and rollback.sh issue, keeps its state
in a JSON file ($STUB_STATE) and appends every call to $STUB_LOG, so a test can assert
what the scripts did (for instance that no `docker tag` or `image rm` ever ran).
Read-only Git only: this stub never writes to any repository.
"""

from __future__ import annotations

import json
import os
import sys

STATE = os.environ["STUB_STATE"]
LOG = os.environ["STUB_LOG"]
SHA_A = os.environ["STUB_SHA_A"]
SHA_B = os.environ["STUB_SHA_B"]


def load() -> dict:
    with open(STATE, encoding="utf-8") as handle:
        return json.load(handle)


def save(state: dict) -> None:
    with open(STATE, "w", encoding="utf-8") as handle:
        json.dump(state, handle)


def fmt_value(fmt: str, image: dict, image_ref: str) -> str:
    if "index .Config.Labels" in fmt:
        state = load()
        running = find_image(state, state["running"]) if state.get("running") else None
        if os.environ.get("STUB_UNLABELLED_RUNNING") and running and running["id"] == image["id"]:
            return ""  # the running image has no revision label
        return image.get("label", "")
    if ".Id" in fmt:
        return image["id"]
    if ".Created" in fmt:
        return "2026-01-01T00:00:00Z"
    return ""


def find_image(state: dict, ref: str):
    if ref in state["images"]:
        return state["images"][ref]
    for image in state["images"].values():
        if image["id"] == ref:
            return image
    return None


def docker(args: list[str]) -> int:
    state = load()
    if args[:2] == ["compose", "version"]:
        return 0
    if args and args[0] == "compose":
        rest = [a for a in args[1:]]
        # drop "-f FILE" and "--env-file FILE"
        cleaned: list[str] = []
        skip = False
        for item in rest:
            if skip:
                skip = False
                continue
            if item in {"-f", "--env-file"}:
                skip = True
                continue
            cleaned.append(item)
        verb = cleaned[0]
        if verb in {"ps", "up"}:
            # docker-compose.prod.yml declares FRAUDAI_IMAGE / FRONTEND_IMAGE as required
            # (${VAR:?...}); real Compose fails to interpolate when either is missing.
            for required in ("FRAUDAI_IMAGE", "FRONTEND_IMAGE"):
                if not os.environ.get(required):
                    sys.stderr.write(f"required variable {required} is missing a value\n")
                    return 15
        if verb == "ps":
            return print("cid-fraudai") or 0 if state.get("running") else 0
        if verb == "up":
            image = os.environ.get("FRAUDAI_IMAGE", "")
            if "fraudai" in cleaned or any(s in cleaned for s in ("qdrant", "grafana")):
                if state["fail_up"]:
                    return 1
                state["running"] = image
                save(state)
            return 0
        if verb in {"exec", "logs"}:
            return 0
        return 0
    if args[0] == "ps":
        if os.environ.get("STUB_PS_FAIL"):
            sys.stderr.write("Cannot connect to the Docker daemon (simulated)\n")
            return 1
        if state.get("running"):
            print("cid-fraudai")
        return 0
    if args[:2] == ["image", "inspect"]:
        fmt = args[args.index("--format") + 1] if "--format" in args else ""
        ref = args[-1]
        image = find_image(state, ref)
        if image is None:
            return 1
        if fmt:
            print(fmt_value(fmt, image, ref))
        return 0
    if args[:2] == ["image", "ls"]:
        for tag in state["images"]:
            if tag.startswith("fraudai-agent:"):
                print(tag.split(":", 1)[1])
        return 0
    if args[0] == "inspect":
        if os.environ.get("STUB_INSPECT_FAIL"):
            sys.stderr.write("Error response from daemon: inspect failed (simulated)\n")
            return 1
        image = find_image(state, state["running"]) if state.get("running") else None
        print(image["id"] if image else "")
        return 0
    if args[0] == "build":
        sys.stdin.buffer.read()
        tag = args[args.index("-t") + 1]
        label = ""
        for index, item in enumerate(args):
            if item == "--build-arg" and args[index + 1].startswith("GIT_SHA="):
                label = args[index + 1].split("=", 1)[1]
            if item == "--label" and args[index + 1].startswith(
                "org.opencontainers.image.revision="
            ):
                label = args[index + 1].split("=", 1)[1]
        if os.environ.get("STUB_BUILD_LABEL"):
            label = os.environ["STUB_BUILD_LABEL"]
        state["images"][tag] = {
            "id": "sha256:" + tag.replace(":", "")[:12].ljust(12, "0") + str(len(state["images"])),
            "label": label,
        }
        state["builds"] = state.get("builds", 0) + 1
        save(state)
        return 0
    return 0


def git(args: list[str]) -> int:
    if args[0] == "rev-parse":
        ref = args[-1].split("^", 1)[0]
        mapping = {"HEAD": SHA_B, "HEAD~1": SHA_A, SHA_A: SHA_A, SHA_B: SHA_B}
        if ref in mapping:
            print(mapping[ref])
            return 0
        return 128
    if args[0] == "show":
        print("2026-01-01T00:00:00+00:00")
        return 0
    if args[0] == "archive":
        sys.stdout.buffer.write(b"SIMULATED-TAR")
        return 0
    if args[0] in {"status", "cat-file"}:
        return 0
    sys.stderr.write(f"stub git: unsupported (and never mutating): {args}\n")
    return 99


def curl(args: list[str]) -> int:
    state = load()
    url = args[-1]
    served = os.environ.get("STUB_SERVED_SHA")
    running = state.get("running")
    if served is None and running:
        image = find_image(state, running)
        served = image["label"] if image else None
    if "readyz" in url:
        body = json.dumps(
            {
                "status": "ready",
                "checks": {"qdrant": True, "llm": True},
                "checked_at": "2026-01-01T00:00:00Z",
                "build_sha": served or None,
            }
        )
        if "-w" in args:
            print("200" if running else "000", end="")
        else:
            print(body)
        return 0 if running else 7
    return 0


def main() -> int:
    tool, args = sys.argv[1], sys.argv[2:]
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write(
            json.dumps([tool, *args, {"FRAUDAI_IMAGE": os.environ.get("FRAUDAI_IMAGE", "")}]) + "\n"
        )
    return {"docker": docker, "git": git, "curl": curl}[tool](args)


if __name__ == "__main__":
    sys.exit(main())
