import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  canDecide,
  initialPendingState,
  pendingReducer,
  type PendingAction_ as Act,
  type PendingState,
} from "../src/lib/pending";
import { PENDING } from "./helpers";

const A = PENDING;
const B = { ...PENDING, action_id: "22222222-2222-4222-8222-222222222222", tool: "second_tool" };

function run(actions: Act[], from: PendingState = initialPendingState): PendingState {
  return actions.reduce(pendingReducer, from);
}

// Mirrors the hook: stream announces A (existing session), then the done of
// the same turn arrives; the user decides on A.
const streamAnnouncesA: Act[] = [{ type: "stream_start" }, { type: "stream_pending", pending: A }];

describe("pending-action ordering (P2-1)", () => {
  it("does not allow deciding while the SSE turn is still open", () => {
    const s = run(streamAnnouncesA);
    assert.equal(s.pending?.action_id, A.action_id);
    assert.equal(canDecide(s), false);
    const after = pendingReducer(s, { type: "decision_start" });
    assert.equal(after.confirming, false, "decision_start is refused while streaming");
  });

  it("order 1: done closes the turn first, then the decision resolves A -> nothing pending", () => {
    let s = run([...streamAnnouncesA, { type: "stream_done", pending: A }]);
    assert.equal(canDecide(s), true);
    s = run([{ type: "decision_start" }, { type: "decision_result", pending: null }], s);
    assert.equal(s.pending, null);
    assert.equal(s.confirming, false);
  });

  it("order 1 with A -> B: B stays after the decision", () => {
    let s = run([...streamAnnouncesA, { type: "stream_done", pending: A }]);
    s = run([{ type: "decision_start" }, { type: "decision_result", pending: B }], s);
    assert.equal(s.pending?.action_id, B.action_id);
  });

  it("order 2 (decision result before the old done): a stale done cannot restore A", () => {
    // Defence in depth: even if the decision were applied while the old turn
    // is still open, its late done must not bring A back.
    let s = run(streamAnnouncesA);
    s = { ...s, streaming: false }; // force the decision through (e.g. stream cancelled)
    s = run([{ type: "decision_start" }, { type: "decision_result", pending: null }], s);
    s = pendingReducer({ ...s, streaming: true }, { type: "stream_done", pending: A });
    assert.equal(s.pending, null);
  });

  it("order 2 with A -> B: a stale done cannot replace B with A", () => {
    let s = run(streamAnnouncesA);
    s = { ...s, streaming: false };
    s = run([{ type: "decision_start" }, { type: "decision_result", pending: B }], s);
    s = pendingReducer({ ...s, streaming: true }, { type: "stream_done", pending: A });
    assert.equal(s.pending?.action_id, B.action_id);
  });

  it("a done of a NEW turn (started after the decision) does set its pending", () => {
    let s = run([...streamAnnouncesA, { type: "stream_done", pending: A }]);
    s = run([{ type: "decision_start" }, { type: "decision_result", pending: null }], s);
    s = run([{ type: "stream_start" }, { type: "stream_done", pending: B }], s);
    assert.equal(s.pending?.action_id, B.action_id);
  });

  it("failed decision keeps the same action and allows retry", () => {
    let s = run([...streamAnnouncesA, { type: "stream_done", pending: A }]);
    s = run([{ type: "decision_start" }, { type: "decision_result", pending: "keep" }], s);
    assert.equal(s.pending?.action_id, A.action_id);
    assert.equal(canDecide(s), true);
  });

  it("double decision is refused while one is in flight; reset clears everything", () => {
    let s = run([...streamAnnouncesA, { type: "stream_done", pending: A }, { type: "decision_start" }]);
    assert.equal(canDecide(s), false);
    s = pendingReducer(s, { type: "reset" });
    assert.deepEqual(s, initialPendingState);
  });

  it("409 recovery (no done follows): stream_end releases the decision", () => {
    const s = run([...streamAnnouncesA, { type: "stream_end" }]);
    assert.equal(canDecide(s), true);
  });
});
