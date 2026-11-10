import type {
  ChatRequest,
  ChatResponse,
  PendingAction,
  ToolResult,
  SSEEvent,
} from "./types";
import {
  API_BASE,
  authHeaders,
  errorFromResponse,
  pendingFromDetail,
} from "./api";
import { getToken, onSessionCleared } from "./auth";

// ---------------------------------------------------------------------------
// Active streams: aborted when the session ends (logout / 401)
// ---------------------------------------------------------------------------

const activeStreams = new Set<AbortController>();

onSessionCleared(() => {
  activeStreams.forEach((c) => c.abort());
  activeStreams.clear();
});

export interface StreamHandlers {
  onToken: (content: string) => void;
  onToolStart: (toolName: string) => void;
  onToolResult: (result: ToolResult) => void;
  /**
   * Server paused for human approval. `recovered` is true when it came from a
   * 409 on the request itself (no `done` event will follow).
   */
  onPendingAction: (pending: PendingAction, recovered: boolean) => void;
  onDone: (response: ChatResponse) => void;
  onError: (error: string, code?: string, status?: number) => void;
}

// ---------------------------------------------------------------------------
// SSE line parser
// ---------------------------------------------------------------------------

/**
 * Parse a single SSE "data:" line into a typed event.
 * Returns null for keep-alive or malformed lines.
 */
export function parseSSELine(line: string): SSEEvent | null {
  if (!line.startsWith("data:")) return null;

  const json = line.slice("data:".length).trim();
  if (!json || json === "[DONE]") return null;

  try {
    return JSON.parse(json) as SSEEvent;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Stream chat — fetch + ReadableStream SSE client
// ---------------------------------------------------------------------------

/**
 * Opens an SSE connection to POST /chat/stream and dispatches
 * typed callbacks as events arrive.
 *
 * Returns an AbortController so the caller can cancel the stream. Once
 * aborted (user cancel, logout, 401) no further callback is invoked.
 */
export function streamChat(
  chatRequest: ChatRequest,
  handlers: StreamHandlers,
): AbortController {
  const controller = new AbortController();
  activeStreams.add(controller);

  const live = () => !controller.signal.aborted;

  const dispatch = (event: SSEEvent) => {
    if (!live()) return;
    switch (event.type) {
      case "token":
        handlers.onToken(event.content);
        break;
      case "tool_start":
        handlers.onToolStart(event.tool_name);
        break;
      case "tool_result":
        handlers.onToolResult(event.tool_result);
        break;
      case "pending_action":
        handlers.onPendingAction(event.pending_action, false);
        break;
      case "done":
        handlers.onDone(event.response);
        break;
      case "error":
        handlers.onError(event.message, event.code);
        break;
    }
  };

  const run = async () => {
    const usedToken = getToken();
    try {
      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...authHeaders(usedToken),
        },
        body: JSON.stringify(chatRequest),
        signal: controller.signal,
      });

      if (!res.ok) {
        const err = await errorFromResponse(res, usedToken, "Stream request failed");
        if (!live() && res.status !== 401) return;
        // F4: the session is paused; recover the pending action, no error.
        if (res.status === 409 && err.code === "pending_action_required") {
          const pending = pendingFromDetail(err.detail);
          if (pending && live()) {
            handlers.onPendingAction(pending, true);
            return;
          }
        }
        // 401 aborts this controller via clearSession; still tell the caller.
        handlers.onError(err.message, err.code, err.status);
        return;
      }

      const reader = res.body?.getReader();
      if (!reader) {
        handlers.onError("No readable stream in response");
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";

      const flush = (chunk: string) => {
        for (const line of chunk.split("\n")) {
          const event = parseSSELine(line);
          if (event) dispatch(event);
        }
      };

      while (live()) {
        const { done, value } = await reader.read();
        if (done) break;

        // Normalise CRLF so event separators are always "\n\n"
        buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");

        // SSE events are separated by double newlines
        const parts = buffer.split("\n\n");
        // Keep the last (possibly incomplete) chunk in the buffer
        buffer = parts.pop() ?? "";
        for (const part of parts) flush(part);
      }
      buffer += decoder.decode();
      if (live() && buffer.trim()) flush(buffer);
    } catch (err: unknown) {
      if (!live() || (err instanceof DOMException && err.name === "AbortError")) {
        // User cancelled / session ended — not an error
        return;
      }
      handlers.onError(err instanceof Error ? err.message : "Unknown streaming error");
    } finally {
      activeStreams.delete(controller);
    }
  };

  void run();

  return controller;
}
