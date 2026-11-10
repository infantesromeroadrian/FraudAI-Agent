"use client";

import type { AgentName } from "@/lib/types";
import { useChat } from "@/hooks/useChat";
import { MessageList } from "./MessageList";
import { ChatInput } from "./ChatInput";
import { AgentIndicator } from "./AgentIndicator";
import ConfirmationDialog from "./ConfirmationDialog";

export function ChatWindow() {
  const {
    messages,
    isStreaming,
    currentAgent,
    pendingAction,
    isConfirming,
    confirmError,
    confirmNotice,
    confirmPending,
    sendMessage,
    uploadFile,
    rateFeedback,
    cancelStream,
  } = useChat();

  return (
    <div className="flex h-full flex-col">
      {/* Message list (scrollable) */}
      <MessageList messages={messages} onRate={rateFeedback} />

      {/* Agent streaming indicator */}
      <AgentIndicator
        agent={currentAgent as AgentName | null}
        isStreaming={isStreaming}
      />

      {/* Input bar */}
      <ChatInput
        onSend={sendMessage}
        onFileUpload={uploadFile}
        isStreaming={isStreaming}
        onCancel={cancelStream}
        disabled={pendingAction !== null}
      />

      {/* Human approval: shown from the server's pending_action */}
      {pendingAction && (
        <ConfirmationDialog
          pending={pendingAction}
          isSubmitting={isConfirming}
          error={confirmError}
          notice={confirmNotice}
          decisionsBlocked={isStreaming}
          onDecision={(approved) => void confirmPending(approved)}
        />
      )}
    </div>
  );
}
