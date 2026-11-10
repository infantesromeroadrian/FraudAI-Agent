"use client";

import { useState, useCallback, useEffect } from "react";
import type { AgentName } from "@/lib/types";
import { useSession } from "@/hooks/useSession";
import { Sidebar } from "@/components/layout/Sidebar";
import { Header } from "@/components/layout/Header";
import { AuthGate } from "@/components/auth/AuthGate";
import { logout } from "@/lib/api";

/**
 * Chat layout: sidebar + header + content area.
 *
 * This is a client component because it manages interactive
 * state (sidebar selection, agent override, sessions).
 */
export default function ChatLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <AuthGate>
      <ChatShell>{children}</ChatShell>
    </AuthGate>
  );
}

function ChatShell({ children }: { children: React.ReactNode }) {
  const {
    sessions,
    health,
    trackSession,
    removeSession,
  } = useSession();

  const [selectedAgent, setSelectedAgent] = useState<AgentName | null>(null);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [currentAgent, setCurrentAgent] = useState<AgentName | null>(null);

  // Track session when it changes
  useEffect(() => {
    if (currentSessionId) {
      trackSession(currentSessionId);
    }
  }, [currentSessionId, trackSession]);

  const handleNewChat = useCallback(() => {
    setCurrentSessionId(null);
    setCurrentAgent(null);
    setSelectedAgent(null);
  }, []);

  const handleSelectSession = useCallback((sessionId: string) => {
    setCurrentSessionId(sessionId);
  }, []);

  const handleDeleteSession = useCallback(
    async (sessionId: string) => {
      await removeSession(sessionId);
      if (currentSessionId === sessionId) {
        setCurrentSessionId(null);
        setCurrentAgent(null);
      }
    },
    [currentSessionId, removeSession],
  );

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <Sidebar
        sessions={sessions}
        health={health}
        currentSessionId={currentSessionId}
        selectedAgent={selectedAgent}
        onNewChat={handleNewChat}
        onSelectSession={handleSelectSession}
        onDeleteSession={handleDeleteSession}
        onSelectAgent={setSelectedAgent}
      />

      {/* Main content */}
      <main className="flex flex-1 flex-col overflow-hidden">
        <Header
          sessionId={currentSessionId}
          currentAgent={currentAgent}
          selectedAgent={selectedAgent}
          onLogout={logout}
        />
        <div className="flex-1 overflow-hidden">{children}</div>
      </main>
    </div>
  );
}
