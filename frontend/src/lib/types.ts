// ---------------------------------------------------------------------------
// TypeScript types matching src/fraudai/api/schemas.py
// ---------------------------------------------------------------------------

// --- Chat ---

export interface ChatRequest {
  message: string;
  session_id?: string | null;
  agent_override?: string | null;
  language?: "es" | "en";
}

export interface Citation {
  boe_id: string;
  norma_titulo: string;
  articulo: string;
  texto_relevante: string;
  /** Retrieval confidence 0-1 */
  score: number;
}

export interface ToolResult {
  tool_name: string;
  status: "success" | "error" | "pending";
  result: Record<string, unknown> | null;
  duration_ms: number;
}

export interface ResponseMetadata {
  routing_agent: string;
  routing_confidence: number;
  tokens_used: number;
  latency_ms: number;
  corpus_version: string;
}

/** Per-call human approval request (F4 contract). Model-originated: render as plain text. */
export interface PendingAction {
  kind: "offensive_tool_approval";
  action_id: string;
  agent: string;
  tool: string;
  arguments: Record<string, unknown>;
  target: string | null;
  warning?: string | null;
}

export interface ChatResponse {
  session_id: string;
  agent: string;
  message: string;
  citations: Citation[];
  tool_results: ToolResult[];
  metadata: ResponseMetadata;
  /** Always present from the server; null when not paused. */
  pending_action?: PendingAction | null;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

// --- File Upload ---

export interface FileUploadResponse {
  file_id: string;
  filename: string;
  size_bytes: number;
  indexed: boolean;
  chunks_generated: number;
}

// --- Session ---

export interface SessionInfo {
  session_id: string;
  created_at: string;
  agent_history: string[];
  uploaded_files: string[];
  turn_count: number;
}

// --- HITL Confirmation ---

export interface ConfirmationRequest {
  session_id: string;
  action_id: string;
  approved: boolean;
}

// --- Feedback ---

export interface FeedbackRequest {
  session_id: string;
  message_id: string;
  rating: number;
  comment?: string;
}

// --- Health ---

export interface HealthResponse {
  status: "healthy" | "degraded" | "unhealthy";
  qdrant: boolean;
  ollama: boolean;
  claude_api: boolean;
  corpus_version: string | null;
}

// ---------------------------------------------------------------------------
// Frontend-only types
// ---------------------------------------------------------------------------

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  agent?: string;
  citations?: Citation[];
  tool_results?: ToolResult[];
  metadata?: ResponseMetadata;
  timestamp: Date;
  isStreaming?: boolean;
}

export type AgentName =
  | "donna"
  | "harvey"
  | "louis"
  | "jessica"
  | "mike"
  | "rachel";

export interface AgentInfo {
  name: AgentName;
  displayName: string;
  description: string;
  /** Tailwind color class (e.g. "bg-agent-harvey") */
  color: string;
  /** Tailwind text color class */
  textColor: string;
  icon: string;
}

// ---------------------------------------------------------------------------
// Agent registry
// ---------------------------------------------------------------------------

export const AGENTS: Record<AgentName, AgentInfo> = {
  donna: {
    name: "donna",
    displayName: "Donna",
    description: "Routing & orchestration",
    color: "bg-agent-donna",
    textColor: "text-agent-donna",
    icon: "\uD83D\uDCCB",
  },
  harvey: {
    name: "harvey",
    displayName: "Harvey",
    description: "Regulatory compliance (PBC/AML)",
    color: "bg-agent-harvey",
    textColor: "text-agent-harvey",
    icon: "\u2696\uFE0F",
  },
  louis: {
    name: "louis",
    displayName: "Louis",
    description: "Transaction analysis & anomalies",
    color: "bg-agent-louis",
    textColor: "text-agent-louis",
    icon: "\uD83D\uDCCA",
  },
  jessica: {
    name: "jessica",
    displayName: "Jessica",
    description: "Risk assessment & SAR drafting",
    color: "bg-agent-jessica",
    textColor: "text-agent-jessica",
    icon: "\uD83D\uDEE1\uFE0F",
  },
  mike: {
    name: "mike",
    displayName: "Mike",
    description: "Red teaming & penetration testing",
    color: "bg-agent-mike",
    textColor: "text-agent-mike",
    icon: "\uD83D\uDD25",
  },
  rachel: {
    name: "rachel",
    displayName: "Rachel",
    description: "Legal research & BOE citations",
    color: "bg-agent-rachel",
    textColor: "text-agent-rachel",
    icon: "\uD83D\uDCDA",
  },
};

// ---------------------------------------------------------------------------
// SSE event types
// ---------------------------------------------------------------------------

export type SSEEventType =
  | "token"
  | "pending_action"
  | "tool_start"
  | "tool_result"
  | "done"
  | "error";

export interface SSETokenEvent {
  type: "token";
  content: string;
}

export interface SSEToolStartEvent {
  type: "tool_start";
  tool_name: string;
}

export interface SSEToolResultEvent {
  type: "tool_result";
  tool_result: ToolResult;
}

export interface SSEPendingActionEvent {
  type: "pending_action";
  pending_action: PendingAction;
}

export interface SSEDoneEvent {
  type: "done";
  response: ChatResponse;
}

export interface SSEErrorEvent {
  type: "error";
  message: string;
  code?: string;
}

export type SSEEvent =
  | SSETokenEvent
  | SSEToolStartEvent
  | SSEToolResultEvent
  | SSEPendingActionEvent
  | SSEDoneEvent
  | SSEErrorEvent;
