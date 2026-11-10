"use client";

import type { AgentName } from "@/lib/types";
import { AGENTS } from "@/lib/types";
import { Badge } from "@/components/ui/Badge";

interface HeaderProps {
  sessionId: string | null;
  currentAgent: AgentName | null;
  selectedAgent: AgentName | null;
  onLogout: () => void;
}

export function Header({ sessionId, currentAgent, selectedAgent, onLogout }: HeaderProps) {
  const displayAgent = selectedAgent ?? currentAgent;
  const agentInfo = displayAgent ? AGENTS[displayAgent] : null;

  return (
    <header className="flex items-center justify-between border-b border-zinc-800 bg-zinc-900/80 px-4 py-2.5 backdrop-blur-sm">
      <div className="flex items-center gap-3">
        {agentInfo ? (
          <Badge color={`${agentInfo.color} text-white`}>
            {agentInfo.icon} {agentInfo.displayName}
          </Badge>
        ) : (
          <span className="text-sm text-zinc-400">Auto-routing (Donna)</span>
        )}
        {selectedAgent && (
          <span className="text-[10px] text-zinc-500">
            Agent override active
          </span>
        )}
      </div>

      <div className="flex items-center gap-3">
        {sessionId && (
          <span className="font-mono text-[10px] text-zinc-600">
            Session: {sessionId.slice(0, 8)}
          </span>
        )}
        <button
          type="button"
          onClick={onLogout}
          className="rounded px-2 py-1 text-xs text-zinc-400 hover:bg-zinc-800 hover:text-zinc-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-500"
        >
          Sign out
        </button>
      </div>
    </header>
  );
}
