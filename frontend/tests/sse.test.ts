import { afterEach, beforeEach, describe, it } from "node:test";
import assert from "node:assert/strict";

import { clearSession, getToken, setToken } from "../src/lib/auth";
import { streamChat, type StreamHandlers } from "../src/lib/sse";
import { chatResponse, ev, installFetch, json, PENDING, sse, waitFor } from "./helpers";

let restore: () => void = () => undefined;
beforeEach(() => {
  clearSession("logout");
  setToken("synthetic.jwt.value");
});
afterEach(() => restore());

interface Log {
  tokens: string[];
  tools: string[];
  pending: unknown[];
  done: unknown[];
  errors: Array<{ message: string; code?: string }>;
}

function collect(): { log: Log; handlers: StreamHandlers } {
  const log: Log = { tokens: [], tools: [], pending: [], done: [], errors: [] };
  return {
    log,
    handlers: {
      onToken: (t) => log.tokens.push(t),
      onToolStart: (n) => log.tools.push(n),
      onToolResult: () => undefined,
      onPendingAction: (p) => log.pending.push(p),
      onDone: (r) => log.done.push(r),
      onError: (message, code) => log.errors.push({ message, code }),
    },
  };
}

describe("streamChat (F5b.3 SSE)", () => {
  it("sends Bearer and decodes events split across chunks, incl. CRLF", async () => {
    const body = ev({ type: "token", content: "ho" }) + ev({ type: "token", content: "la" });
    const mid = Math.floor(body.length / 2);
    const f = installFetch([
      sse([
        body.slice(0, mid),
        body.slice(mid),
        ev({ type: "done", response: chatResponse({ agent: "donna" }) }).replace(/\n/g, "\r\n"),
      ]),
    ]);
    restore = f.restore;
    const { log, handlers } = collect();
    streamChat({ message: "x" }, handlers);
    await waitFor(() => log.done.length === 1);
    assert.equal(f.calls[0].headers.get("authorization"), "Bearer synthetic.jwt.value");
    assert.equal(log.tokens.join(""), "hola");
    assert.deepEqual(log.errors, []);
  });

  it("dispatches pending_action before done and passes done.response.pending_action", async () => {
    const f = installFetch([
      sse([
        ev({ type: "pending_action", pending_action: PENDING }),
        ev({ type: "done", response: chatResponse({ pending_action: PENDING, message: "notice" }) }),
      ]),
    ]);
    restore = f.restore;
    const { log, handlers } = collect();
    streamChat({ message: "x" }, handlers);
    await waitFor(() => log.done.length === 1);
    assert.deepEqual(log.pending, [PENDING]);
    assert.deepEqual((log.done[0] as { pending_action: unknown }).pending_action, PENDING);
    assert.deepEqual(log.tools, [], "no tool_start for a paused call");
  });

  it("409 pending_action_required recovers the pending action without an error", async () => {
    const f = installFetch([
      json(409, {
        detail: { code: "pending_action_required", message: "Resolve first", pending_action: PENDING },
      }),
    ]);
    restore = f.restore;
    const { log, handlers } = collect();
    streamChat({ message: "x", session_id: "s-1" }, handlers);
    await waitFor(() => log.pending.length === 1);
    assert.deepEqual(log.pending, [PENDING]);
    assert.deepEqual(log.errors, []);
  });

  it("stream error event keeps its code", async () => {
    const f = installFetch([
      sse([ev({ type: "error", code: "pending_action_required", message: "paused meanwhile" })]),
    ]);
    restore = f.restore;
    const { log, handlers } = collect();
    streamChat({ message: "x" }, handlers);
    await waitFor(() => log.errors.length === 1);
    assert.equal(log.errors[0].code, "pending_action_required");
  });

  it("HTTP 401 clears the token and reports a sign-in error", async () => {
    const f = installFetch([json(401, { detail: "expired" })]);
    restore = f.restore;
    const { log, handlers } = collect();
    streamChat({ message: "x" }, handlers);
    await waitFor(() => log.errors.length === 1);
    assert.equal(getToken(), null);
    assert.match(log.errors[0].message, /sign in/i);
  });

  it("logout aborts the active stream and drops late events", async () => {
    let push: (s: string) => void = () => undefined;
    const enc = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      start(c) {
        push = (s) => c.enqueue(enc.encode(s));
      },
    });
    const f = installFetch([new Response(stream, { status: 200 })]);
    restore = f.restore;
    const { log, handlers } = collect();
    const controller = streamChat({ message: "x" }, handlers);
    await waitFor(() => f.calls.length === 1);
    push(ev({ type: "token", content: "a" }));
    await waitFor(() => log.tokens.length === 1);
    clearSession("logout");
    assert.equal(controller.signal.aborted, true);
    push(ev({ type: "token", content: "late" }));
    await new Promise((r) => setTimeout(r, 30));
    assert.deepEqual(log.tokens, ["a"]);
    assert.deepEqual(log.errors, [], "abort is not an error");
  });
});
