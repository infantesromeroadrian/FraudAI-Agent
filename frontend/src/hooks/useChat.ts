"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import type {
  Message,
  AgentName,
  ChatResponse,
  PendingAction,
  ToolResult,
} from "@/lib/types";
import { streamChat } from "@/lib/sse";
import {
  uploadFile as apiUploadFile,
  sendFeedback as apiFeedback,
} from "@/lib/api";
import { onSessionCleared } from "@/lib/auth";
import { resolvePending } from "@/lib/hitl";
import {
  canDecide,
  initialPendingState,
  pendingReducer,
  type PendingAction_,
  type PendingState,
} from "@/lib/pending";

// ---------------------------------------------------------------------------
// Unique ID generator (avoids crypto dep in client components)
// ---------------------------------------------------------------------------

let counter = 0;
function uid(): string {
  counter += 1;
  return `msg_${Date.now()}_${counter}`;
}

function assistantNote(content: string, agent?: string): Message {
  return { id: uid(), role: "assistant", content, agent, timestamp: new Date() };
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentAgent, setCurrentAgent] = useState<AgentName | null>(null);
  // F4: pending/streaming/confirming live in one reducer (lib/pending.ts) so
  // an old SSE turn can never restore or replace an already-decided action.
  // The ref mirrors the state synchronously for guards inside callbacks.
  const [pendingState, setPendingState] = useState<PendingState>(initialPendingState);
  const stateRef = useRef<PendingState>(initialPendingState);
  const dispatch = useCallback((a: PendingAction_) => {
    stateRef.current = pendingReducer(stateRef.current, a);
    setPendingState(stateRef.current);
  }, []);
  const { pending: pendingAction, streaming: isStreaming, confirming: isConfirming } = pendingState;
  const [confirmError, setConfirmError] = useState<string | null>(null);
  const [confirmNotice, setConfirmNotice] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const streamBufferRef = useRef<string>("");
  const rafRef = useRef<number | null>(null);
  const assistantIdRef = useRef<string>("");
  // Bumped whenever state of a previous session/user must be discarded.
  const generationRef = useRef(0);

  const stopRaf = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
  }, []);

  // Drop everything that belonged to the previous user/session.
  const resetAll = useCallback(() => {
    generationRef.current += 1;
    stopRaf();
    abortRef.current?.abort();
    abortRef.current = null;
    streamBufferRef.current = "";
    dispatch({ type: "reset" });
    setMessages([]);
    setSessionId(null);
    setCurrentAgent(null);
    setConfirmError(null);
    setConfirmNotice(null);
  }, [stopRaf, dispatch]);

  // 401 / logout: cancel stream and clear sessions + pending.
  useEffect(() => {
    const off = onSessionCleared(resetAll);
    return () => {
      off();
      generationRef.current += 1;
      stopRaf();
      abortRef.current?.abort();
    };
  }, [resetAll, stopRaf]);

  // -----------------------------------------------------------------------
  // Send a chat message (streaming)
  // -----------------------------------------------------------------------

  const sendMessage = useCallback(
    async (content: string, agentOverride?: AgentName) => {
      // While an action awaits approval the server rejects chat (409): the
      // decision must come first, and a message never approves anything.
      if (!content.trim() || stateRef.current.streaming || stateRef.current.pending) return;

      const gen = generationRef.current;
      const alive = () => generationRef.current === gen;

      // Append user message
      const userMsg: Message = {
        id: uid(),
        role: "user",
        content,
        timestamp: new Date(),
      };

      const assistantId = uid();

      setMessages((prev) => [
        ...prev,
        userMsg,
        {
          id: assistantId,
          role: "assistant",
          content: "",
          timestamp: new Date(),
          isStreaming: true,
        },
      ]);

      dispatch({ type: "stream_start" });
      streamBufferRef.current = "";
      assistantIdRef.current = assistantId;

      // Flush accumulated tokens to React state at ~30fps
      // This prevents React 18 batching from swallowing individual token updates
      const flushBuffer = () => {
        const buf = streamBufferRef.current;
        const aid = assistantIdRef.current;
        if (buf) {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === aid ? { ...m, content: buf } : m,
            ),
          );
        }
        rafRef.current = requestAnimationFrame(flushBuffer);
      };
      rafRef.current = requestAnimationFrame(flushBuffer);

      const finish = () => {
        stopRaf();
        dispatch({ type: "stream_end" });
        abortRef.current = null;
      };

      const controller = streamChat(
        {
          message: content,
          session_id: sessionId,
          agent_override: agentOverride ?? null,
          language: "es",
        },
        {
          // accumulate in ref, RAF flushes to state
          onToken: (token: string) => {
            if (alive()) streamBufferRef.current += token;
          },
          onToolStart: (toolName: string) => {
            if (!alive()) return;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      tool_results: [
                        ...(m.tool_results ?? []),
                        { tool_name: toolName, status: "pending" as const, result: null, duration_ms: 0 },
                      ],
                    }
                  : m,
              ),
            );
          },
          onToolResult: (result: ToolResult) => {
            if (!alive()) return;
            setMessages((prev) =>
              prev.map((m) => {
                if (m.id !== assistantId) return m;
                const tools = (m.tool_results ?? []).map((t) =>
                  t.tool_name === result.tool_name && t.status === "pending" ? result : t,
                );
                return { ...m, tool_results: tools };
              }),
            );
          },
          onPendingAction: (pending: PendingAction, recovered: boolean) => {
            if (!alive()) return;
            setConfirmError(null);
            setConfirmNotice(null);
            dispatch({ type: "stream_pending", pending });
            if (recovered) {
              // 409: the message was discarded by the server; no `done` follows.
              finish();
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? {
                        ...m,
                        content:
                          "An action is awaiting your approval. Your message was not sent: decide on the action, then send it again.",
                        isStreaming: false,
                      }
                    : m,
                ),
              );
            }
          },
          // stop RAF, set final content with metadata
          onDone: (response: ChatResponse) => {
            if (!alive()) return;
            stopRaf();

            setSessionId(response.session_id);
            setCurrentAgent(response.agent as AgentName);

            const finalContent = streamBufferRef.current || response.message || "No response generated.";
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                      ...m,
                      content: finalContent,
                      agent: response.agent,
                      citations: response.citations,
                      tool_results: response.tool_results?.length ? response.tool_results : m.tool_results,
                      metadata: response.metadata,
                      isStreaming: false,
                    }
                  : m,
              ),
            );
            dispatch({ type: "stream_done", pending: response.pending_action ?? null });
            abortRef.current = null;
          },
          // stop RAF, show error
          onError: (error: string, code?: string, status?: number) => {
            if (!alive()) return;
            finish();
            if (status === 404) {
              // Server does not know this session: next message starts a new one.
              setSessionId(null);
            }
            const text =
              code === "pending_action_required"
                ? "An action became pending while your message was being sent. Send the message again to review it."
                : `Error: ${error}`;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, content: text, isStreaming: false } : m,
              ),
            );
          },
        },
      );

      abortRef.current = controller;
    },
    [sessionId, dispatch, stopRaf],
  );

  // -----------------------------------------------------------------------
  // File upload (session must already exist: IDs are issued by the server)
  // -----------------------------------------------------------------------

  const uploadFile = useCallback(
    async (file: File) => {
      if (!sessionId) {
        setMessages((prev) => [
          ...prev,
          assistantNote(
            "Send a message first: the server creates the session on your first message. Then attach the file.",
          ),
        ]);
        return;
      }
      const gen = generationRef.current;
      try {
        const result = await apiUploadFile(sessionId, file);
        if (generationRef.current !== gen) return;
        setMessages((prev) => [
          ...prev,
          assistantNote(
            `File "${result.filename}" uploaded (${result.chunks_generated} chunks indexed).`,
            "donna",
          ),
        ]);
      } catch (err: unknown) {
        if (generationRef.current !== gen) return;
        setMessages((prev) => [
          ...prev,
          assistantNote(
            `Upload failed: ${err instanceof Error ? err.message : "Unknown error"}`,
          ),
        ]);
      }
    },
    [sessionId],
  );

  // -----------------------------------------------------------------------
  // HITL decision: one explicit approve/reject for the exact pending action
  // -----------------------------------------------------------------------

  const confirmPending = useCallback(
    async (approved: boolean) => {
      const pending = stateRef.current.pending;
      // Not while the SSE turn is open, not twice, never without a pending.
      if (!pending || !sessionId || !canDecide(stateRef.current)) return;

      const gen = generationRef.current;
      dispatch({ type: "decision_start" });
      setConfirmError(null);
      setConfirmNotice(null);

      const outcome = await resolvePending(sessionId, pending.action_id, approved);
      if (generationRef.current !== gen) return; // session ended meanwhile

      switch (outcome.kind) {
        case "answer": {
          const r = outcome.response;
          dispatch({ type: "decision_result", pending: null });
          setCurrentAgent(r.agent as AgentName);
          setMessages((prev) => [
            ...prev,
            {
              id: uid(),
              role: "assistant",
              content: r.message,
              agent: r.agent,
              citations: r.citations,
              tool_results: r.tool_results,
              metadata: r.metadata,
              timestamp: new Date(),
            },
          ]);
          break;
        }
        case "next":
          dispatch({ type: "decision_result", pending: outcome.pending });
          setConfirmNotice(
            approved
              ? "Decision recorded. Another action now needs your approval."
              : "Rejection recorded. Another action now needs your decision.",
          );
          break;
        case "no_pending":
          dispatch({ type: "decision_result", pending: null });
          setMessages((prev) => [...prev, assistantNote(outcome.message)]);
          break;
        case "stale":
          dispatch({ type: "decision_result", pending: outcome.pending });
          setConfirmError(outcome.message);
          break;
        case "session_lost":
          dispatch({ type: "decision_result", pending: null });
          setSessionId(null);
          setMessages((prev) => [...prev, assistantNote(outcome.message)]);
          break;
        case "unauthorized":
          // Token already cleared by the API layer; the gate shows sign-in.
          break;
        case "error":
          // Action stays pending; user may retry the same explicit decision.
          dispatch({ type: "decision_result", pending: "keep" });
          setConfirmError(outcome.message);
          break;
      }
    },
    [sessionId, dispatch],
  );

  // -----------------------------------------------------------------------
  // Feedback
  // -----------------------------------------------------------------------

  const rateFeedback = useCallback(
    async (messageId: string, rating: number) => {
      if (!sessionId) return;
      try {
        await apiFeedback(sessionId, messageId, rating);
      } catch {
        // Silently fail — feedback is non-critical
      }
    },
    [sessionId],
  );

  // -----------------------------------------------------------------------
  // Clear / cancel
  // -----------------------------------------------------------------------

  const clearChat = resetAll;

  const cancelStream = useCallback(() => {
    stopRaf();
    abortRef.current?.abort();
    abortRef.current = null;
    dispatch({ type: "stream_end" });
    setMessages((prev) =>
      prev.map((m) => (m.isStreaming ? { ...m, isStreaming: false, content: streamBufferRef.current || m.content } : m)),
    );
  }, [stopRaf, dispatch]);

  return {
    messages,
    isStreaming,
    sessionId,
    currentAgent,
    pendingAction,
    isConfirming,
    confirmError,
    confirmNotice,
    sendMessage,
    uploadFile,
    confirmPending,
    rateFeedback,
    clearChat,
    cancelStream,
  };
}
