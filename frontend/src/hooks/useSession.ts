"use client";

import { useState, useCallback, useEffect } from "react";
import type { SessionInfo, HealthResponse } from "@/lib/types";
import { getSession, deleteSession, healthCheck } from "@/lib/api";

// ---------------------------------------------------------------------------
// Session list. Kept in memory only: it belongs to the signed-in user and is
// dropped with the component on logout / 401 (the AuthGate unmounts this
// tree). Session IDs are issued by the server; nothing is persisted.
// ---------------------------------------------------------------------------

interface StoredSession {
  id: string;
  createdAt: string;
  turnCount: number;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useSession() {
  const [sessions, setSessions] = useState<StoredSession[]>([]);
  const [activeSession, setActiveSession] = useState<SessionInfo | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isLoadingHealth, setIsLoadingHealth] = useState(false);

  // -----------------------------------------------------------------------
  // Health check
  // -----------------------------------------------------------------------

  const checkHealth = useCallback(async () => {
    setIsLoadingHealth(true);
    try {
      const result = await healthCheck();
      setHealth(result);
    } catch {
      setHealth({
        status: "unhealthy",
        qdrant: false,
        ollama: false,
        claude_api: false,
        corpus_version: null,
      });
    } finally {
      setIsLoadingHealth(false);
    }
  }, []);

  // Poll health every 30 seconds
  useEffect(() => {
    void checkHealth();
    const interval = setInterval(() => void checkHealth(), 30_000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  // -----------------------------------------------------------------------
  // Track session
  // -----------------------------------------------------------------------

  const trackSession = useCallback((sessionId: string) => {
    setSessions((prev) => {
      if (prev.some((s) => s.id === sessionId)) return prev;
      return [
        { id: sessionId, createdAt: new Date().toISOString(), turnCount: 0 },
        ...prev,
      ];
    });
  }, []);

  // -----------------------------------------------------------------------
  // Load session info
  // -----------------------------------------------------------------------

  const loadSession = useCallback(async (sessionId: string) => {
    try {
      const info = await getSession(sessionId);
      setActiveSession(info);
      return info;
    } catch {
      return null;
    }
  }, []);

  // -----------------------------------------------------------------------
  // Delete session
  // -----------------------------------------------------------------------

  const removeSession = useCallback(async (sessionId: string) => {
    try {
      await deleteSession(sessionId);
    } catch {
      // Session may already be gone — continue cleanup
    }

    setSessions((prev) => prev.filter((s) => s.id !== sessionId));
    setActiveSession((prev) =>
      prev?.session_id === sessionId ? null : prev,
    );
  }, []);

  return {
    sessions,
    activeSession,
    health,
    isLoadingHealth,
    trackSession,
    loadSession,
    removeSession,
    checkHealth,
  };
}
