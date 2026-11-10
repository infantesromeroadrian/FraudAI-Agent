import { afterEach, beforeEach, describe, it } from "node:test";
import assert from "node:assert/strict";

import { clearSession, getToken, setToken } from "../src/lib/auth";
import { describeToolFailure, resolvePending, formatArguments } from "../src/lib/hitl";
import { chatResponse, installFetch, json, PENDING } from "./helpers";

let restore: () => void = () => undefined;
beforeEach(() => {
  clearSession("logout");
  setToken("synthetic.jwt.value");
});
afterEach(() => restore());

const NEXT = { ...PENDING, action_id: "22222222-2222-4222-8222-222222222222", tool: "second_tool" };

describe("resolvePending (F5b.3 /confirm)", () => {
  it("sends the exact action_id and a JSON boolean", async () => {
    const f = installFetch([json(200, chatResponse())]);
    restore = f.restore;
    await resolvePending("s-1", PENDING.action_id, true);
    assert.match(f.calls[0].url, /\/confirm$/);
    assert.equal(f.calls[0].headers.get("authorization"), "Bearer synthetic.jwt.value");
    assert.deepEqual(JSON.parse(String(f.calls[0].init.body)), {
      session_id: "s-1",
      action_id: PENDING.action_id,
      approved: true,
    });
  });

  it("approved:false is sent as boolean false, never coerced", async () => {
    const f = installFetch([json(200, chatResponse())]);
    restore = f.restore;
    await resolvePending("s-1", PENDING.action_id, false);
    assert.strictEqual(JSON.parse(String(f.calls[0].init.body)).approved, false);
  });

  it("200 final answer -> outcome answer", async () => {
    const f = installFetch([json(200, chatResponse({ message: "Red teaming action cancelled by user." }))]);
    restore = f.restore;
    const o = await resolvePending("s-1", PENDING.action_id, false);
    assert.equal(o.kind, "answer");
  });

  it("200 with next pending_action -> outcome next", async () => {
    const f = installFetch([json(200, chatResponse({ pending_action: NEXT }))]);
    restore = f.restore;
    const o = await resolvePending("s-1", PENDING.action_id, true);
    assert.equal(o.kind, "next");
    if (o.kind === "next") assert.equal(o.pending.action_id, NEXT.action_id);
  });

  it("409 no_pending_action -> nothing pending, no retry", async () => {
    const f = installFetch([
      json(409, { detail: { code: "no_pending_action", message: "Nothing awaiting", pending_action: null } }),
    ]);
    restore = f.restore;
    const o = await resolvePending("s-1", PENDING.action_id, true);
    assert.equal(o.kind, "no_pending");
    assert.equal(f.calls.length, 1);
  });

  it("409 action_id_mismatch -> re-render the real pending action, not consumed", async () => {
    const f = installFetch([
      json(409, { detail: { code: "action_id_mismatch", message: "stale", pending_action: NEXT } }),
    ]);
    restore = f.restore;
    const o = await resolvePending("s-1", PENDING.action_id, true);
    assert.equal(o.kind, "stale");
    if (o.kind === "stale") assert.equal(o.pending.action_id, NEXT.action_id);
    assert.equal(f.calls.length, 1, "no automatic re-approval of the replacement");
  });

  it("404 -> session_lost; 422/500 -> error keeping the action retryable", async () => {
    let f = installFetch([json(404, { detail: "Session not found." })]);
    restore = f.restore;
    assert.equal((await resolvePending("s", "a", true)).kind, "session_lost");
    f.restore();
    f = installFetch([json(500, { detail: "Failed to process confirmation" })]);
    restore = f.restore;
    const o = await resolvePending("s", "a", true);
    assert.equal(o.kind, "error");
    f.restore();
    f = installFetch([
      () => {
        throw new TypeError("fetch failed");
      },
    ]);
    restore = f.restore;
    assert.equal((await resolvePending("s", "a", true)).kind, "error");
  });

  it("401 clears the token and reports unauthorized", async () => {
    const f = installFetch([json(401, { detail: "expired" })]);
    restore = f.restore;
    const o = await resolvePending("s", "a", true);
    assert.equal(o.kind, "unauthorized");
    assert.equal(getToken(), null);
  });
});

describe("describeToolFailure (sandbox disabled/unavailable)", () => {
  it("explains status=error results with error_code", () => {
    const text = describeToolFailure({
      tool_name: "adversarial_evasion",
      status: "error",
      result: {
        status: "error",
        error_code: "sandbox_disabled",
        error: "Offensive execution is disabled.",
      },
      duration_ms: 0,
    });
    assert.ok(text);
    assert.match(text, /sandbox_disabled/);
    assert.match(text, /disabled/i);
    assert.match(text, /not executed|did not run|nothing/i);
  });
  it("returns null for successful results", () => {
    assert.equal(
      describeToolFailure({ tool_name: "t", status: "success", result: { a: 1 }, duration_ms: 1 }),
      null,
    );
  });
});

describe("formatArguments", () => {
  it("pretty-prints arguments as plain text, preserving markup literally", () => {
    const out = formatArguments({ target_endpoint: "<img src=x onerror=alert(1)>", n: 3 });
    assert.ok(out.includes("<img src=x onerror=alert(1)>"));
    assert.ok(out.includes('"n": 3'));
  });
});
