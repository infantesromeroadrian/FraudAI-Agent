import type { ChatResponse, PendingAction, ToolResult } from "./types";
import { ApiError, confirmAction } from "./api";

// ---------------------------------------------------------------------------
// Human-in-the-loop (F4 contract). Pure helpers so the flow is testable.
// ---------------------------------------------------------------------------

export type ConfirmOutcome =
  /** Decision applied, final answer (nothing else pending). */
  | { kind: "answer"; response: ChatResponse }
  /** Decision applied, the plan has another call awaiting approval. */
  | { kind: "next"; response: ChatResponse; pending: PendingAction }
  /** 409 no_pending_action: already decided / repeated / server restarted. */
  | { kind: "no_pending"; message: string }
  /** 409 action_id_mismatch: show the real pending action instead (not consumed). */
  | { kind: "stale"; pending: PendingAction; message: string }
  /** 404: session unknown or not ours. */
  | { kind: "session_lost"; message: string }
  /** 401: token cleared by the API layer; sign in again. */
  | { kind: "unauthorized"; message: string }
  /** Anything else: the action stays pending and the user may retry. */
  | { kind: "error"; message: string };

/**
 * Sends exactly one decision for exactly one `action_id`. Never retries and
 * never approves on behalf of the user.
 */
export async function resolvePending(
  sessionId: string,
  actionId: string,
  approved: boolean,
): Promise<ConfirmOutcome> {
  try {
    const response = await confirmAction(sessionId, actionId, approved);
    if (response.pending_action) {
      return { kind: "next", response, pending: response.pending_action };
    }
    return { kind: "answer", response };
  } catch (err: unknown) {
    if (err instanceof ApiError) {
      if (err.status === 401) {
        return { kind: "unauthorized", message: err.message };
      }
      if (err.status === 404) {
        return { kind: "session_lost", message: "This session no longer exists. Start a new conversation." };
      }
      if (err.status === 409) {
        if (err.code === "action_id_mismatch" && err.pendingAction) {
          return {
            kind: "stale",
            pending: err.pendingAction,
            message:
              "That request was out of date. Review the action that is currently awaiting approval.",
          };
        }
        if (err.code === "no_pending_action") {
          return {
            kind: "no_pending",
            message:
              "Nothing is awaiting approval (already decided, or the server restarted). The action was not executed by this request.",
          };
        }
      }
      return { kind: "error", message: err.message };
    }
    return {
      kind: "error",
      message: err instanceof Error ? err.message : "Could not submit the decision.",
    };
  }
}

/**
 * Explicit failure text for a tool that ended in status "error", e.g.
 * result={status,error_code,error} when the sandbox is disabled/unavailable.
 * Returns null for non-failures.
 */
export function describeToolFailure(tool: ToolResult): string | null {
  if (tool.status !== "error") return null;
  const r = (tool.result ?? {}) as Record<string, unknown>;
  const code = typeof r.error_code === "string" ? r.error_code : "unknown_error";
  const detail = typeof r.error === "string" ? r.error : "No further detail.";
  return `Tool "${tool.tool_name}" failed (${code}): ${detail} The action was not executed.`;
}

/** Plain-text rendering of model-supplied arguments (never HTML/markdown). */
export function formatArguments(args: Record<string, unknown>): string {
  try {
    return JSON.stringify(args, null, 2) ?? "{}";
  } catch {
    return String(args);
  }
}
