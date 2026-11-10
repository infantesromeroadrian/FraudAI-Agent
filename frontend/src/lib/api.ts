import type {
  ChatRequest,
  ChatResponse,
  FileUploadResponse,
  SessionInfo,
  HealthResponse,
  FeedbackRequest,
  PendingAction,
  TokenResponse,
} from "./types";
import { clearSession, getToken, handleUnauthorized, setToken } from "./auth";

// ---------------------------------------------------------------------------
// Base configuration
// ---------------------------------------------------------------------------

/** Inlined at build time (NEXT_PUBLIC_*): must be the URL the BROWSER uses. */
export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    /** Parsed `detail` of the error body (string, object or null). */
    public detail: unknown = null,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** `detail.code` for structured errors (e.g. F4 409 conflicts). */
  get code(): string | undefined {
    const d = this.detail;
    if (d && typeof d === "object" && "code" in d) {
      const c = (d as { code?: unknown }).code;
      return typeof c === "string" ? c : undefined;
    }
    return undefined;
  }

  /** `detail.pending_action` for F4 409 conflicts, if the server sent one. */
  get pendingAction(): PendingAction | null {
    return pendingFromDetail(this.detail);
  }
}

export function pendingFromDetail(detail: unknown): PendingAction | null {
  if (detail && typeof detail === "object" && "pending_action" in detail) {
    const p = (detail as { pending_action?: unknown }).pending_action;
    if (p && typeof p === "object" && typeof (p as PendingAction).action_id === "string") {
      return p as PendingAction;
    }
  }
  return null;
}

/** Human-readable message out of a FastAPI error body. */
export function messageFromDetail(detail: unknown, fallback: string): string {
  if (typeof detail === "string" && detail) return detail;
  if (detail && typeof detail === "object") {
    if (Array.isArray(detail)) {
      const msgs = detail
        .map((d) => (d && typeof d === "object" ? (d as { msg?: unknown }).msg : null))
        .filter((m): m is string => typeof m === "string");
      if (msgs.length) return msgs.join("; ");
    } else {
      const m = (detail as { message?: unknown }).message;
      if (typeof m === "string" && m) return m;
    }
  }
  return fallback;
}

/** Builds an ApiError from a non-OK response and applies 401 handling. */
export async function errorFromResponse(
  res: Response,
  usedToken: string | null,
  fallback: string,
): Promise<ApiError> {
  if (res.status === 401) handleUnauthorized(usedToken);
  let detail: unknown = null;
  let text = "";
  try {
    text = await res.text();
    const parsed: unknown = JSON.parse(text);
    detail =
      parsed && typeof parsed === "object" && "detail" in parsed
        ? (parsed as { detail: unknown }).detail
        : parsed;
  } catch {
    detail = text || null;
  }
  const message =
    res.status === 401
      ? "Authentication required. Please sign in again."
      : messageFromDetail(detail, `${fallback} (HTTP ${res.status})`);
  return new ApiError(res.status, message, detail);
}

export function authHeaders(usedToken: string | null): Record<string, string> {
  return usedToken ? { Authorization: `Bearer ${usedToken}` } : {};
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const usedToken = getToken();
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(usedToken),
      ...(options.headers as Record<string, string> | undefined),
    },
  });

  if (!res.ok) {
    throw await errorFromResponse(res, usedToken, "Request failed");
  }

  // Handle 204 No Content
  if (res.status === 204) {
    return undefined as T;
  }

  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Login (D5: token kept in memory only)
// ---------------------------------------------------------------------------

export type LoginErrorKind =
  | "invalid_credentials"
  | "server_unavailable"
  | "rate_limited"
  | "network"
  | "unexpected";

export class LoginError extends Error {
  constructor(
    public kind: LoginErrorKind,
    message: string,
  ) {
    super(message);
    this.name = "LoginError";
  }
}

export async function login(username: string, password: string): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}/auth/token`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ username, password }).toString(),
    });
  } catch {
    throw new LoginError(
      "network",
      "Cannot reach the server. Check that the API is running and try again.",
    );
  }

  if (res.status === 401) {
    throw new LoginError("invalid_credentials", "Invalid username or password.");
  }
  if (res.status === 429) {
    const wait = Number(res.headers.get("Retry-After"));
    throw new LoginError(
      "rate_limited",
      Number.isFinite(wait) && wait > 0
        ? `Too many failed sign-in attempts. Try again in ${Math.ceil(wait)} seconds.`
        : "Too many failed sign-in attempts. Try again later.",
    );
  }
  if (res.status === 503 || res.status === 501) {
    throw new LoginError(
      "server_unavailable",
      "Sign-in is unavailable: authentication is not configured on the server. Contact the administrator.",
    );
  }
  if (!res.ok) {
    throw new LoginError("unexpected", `Sign-in failed (HTTP ${res.status}).`);
  }

  let data: Partial<TokenResponse> | null = null;
  try {
    data = (await res.json()) as Partial<TokenResponse>;
  } catch {
    data = null;
  }
  if (!data || typeof data.access_token !== "string" || !data.access_token) {
    throw new LoginError("unexpected", "Sign-in failed: unexpected server response.");
  }
  setToken(data.access_token);
}

export function logout(): void {
  clearSession("logout");
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

export async function sendMessage(
  chatRequest: ChatRequest,
): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify(chatRequest),
  });
}

// ---------------------------------------------------------------------------
// File Upload (session_id in the query string; multipart carries only `file`)
// ---------------------------------------------------------------------------

export async function uploadFile(
  sessionId: string,
  file: File,
): Promise<FileUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const usedToken = getToken();
  const res = await fetch(
    `${API_BASE}/files/upload?session_id=${encodeURIComponent(sessionId)}`,
    {
      method: "POST",
      // Do NOT set Content-Type — browser sets multipart boundary automatically
      headers: authHeaders(usedToken),
      body: formData,
    },
  );

  if (!res.ok) {
    throw await errorFromResponse(res, usedToken, "Upload failed");
  }

  return res.json() as Promise<FileUploadResponse>;
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

export async function getSession(sessionId: string): Promise<SessionInfo> {
  return request<SessionInfo>(`/sessions/${encodeURIComponent(sessionId)}`);
}

export async function deleteSession(sessionId: string): Promise<void> {
  return request<void>(`/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
  });
}

// ---------------------------------------------------------------------------
// HITL Confirmation (F4 contract: JSON boolean `approved`, exact `action_id`)
// ---------------------------------------------------------------------------

export async function confirmAction(
  sessionId: string,
  actionId: string,
  approved: boolean,
): Promise<ChatResponse> {
  return request<ChatResponse>("/confirm", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      action_id: actionId,
      approved: approved === true,
    }),
  });
}

// ---------------------------------------------------------------------------
// Feedback
// ---------------------------------------------------------------------------

export async function sendFeedback(
  sessionId: string,
  messageId: string,
  rating: number,
  comment?: string,
): Promise<void> {
  const body: FeedbackRequest = {
    session_id: sessionId,
    message_id: messageId,
    rating,
    ...(comment ? { comment } : {}),
  };

  return request<void>("/feedback", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// ---------------------------------------------------------------------------
// Health (public endpoint)
// ---------------------------------------------------------------------------

export async function healthCheck(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}
