import type { PendingAction } from "./types";

// ---------------------------------------------------------------------------
// Ordering of the pending approval between the SSE turn and /confirm.
//
// Pure reducer used by useChat (not a copy): the hook dispatches these
// actions synchronously and renders from the resulting state.
//
// Rules:
//  1. No decision may start while the SSE turn is still open (`streaming`):
//     the dialog is shown but disabled until `done`/error/cancel.
//  2. Defence in depth: a `done` that belongs to a stream started BEFORE the
//     latest decision result (epoch mismatch) never writes `pending`, so a
//     stale turn cannot restore an already-resolved A or replace B with A.
// ---------------------------------------------------------------------------

export interface PendingState {
  pending: PendingAction | null;
  streaming: boolean;
  confirming: boolean;
  /** Incremented by every decision result. */
  epoch: number;
  /** `epoch` captured when the current/last stream started. */
  streamEpoch: number;
}

export const initialPendingState: PendingState = {
  pending: null,
  streaming: false,
  confirming: false,
  epoch: 0,
  streamEpoch: 0,
};

export type PendingAction_ =
  | { type: "stream_start" }
  | { type: "stream_pending"; pending: PendingAction }
  | { type: "stream_done"; pending: PendingAction | null }
  /** error, cancel or 409 recovery: the turn is over, no `done` follows */
  | { type: "stream_end" }
  | { type: "decision_start" }
  /** `pending`: next action, null = nothing pending, "keep" = request failed */
  | { type: "decision_result"; pending: PendingAction | null | "keep" }
  | { type: "reset" };

export function canDecide(s: PendingState): boolean {
  return s.pending !== null && !s.streaming && !s.confirming;
}

export function pendingReducer(s: PendingState, a: PendingAction_): PendingState {
  switch (a.type) {
    case "stream_start":
      return { ...s, streaming: true, streamEpoch: s.epoch };
    case "stream_pending":
      return s.streamEpoch === s.epoch ? { ...s, pending: a.pending } : s;
    case "stream_done":
      return {
        ...s,
        streaming: false,
        pending: a.pending && s.streamEpoch === s.epoch ? a.pending : s.pending,
      };
    case "stream_end":
      return { ...s, streaming: false };
    case "decision_start":
      return canDecide(s) ? { ...s, confirming: true } : s;
    case "decision_result":
      return {
        ...s,
        confirming: false,
        epoch: s.epoch + 1,
        pending: a.pending === "keep" ? s.pending : a.pending,
      };
    case "reset":
      return initialPendingState;
  }
}
