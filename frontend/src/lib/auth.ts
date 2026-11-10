// ---------------------------------------------------------------------------
// In-memory auth session (D5).
//
// The bearer token lives ONLY in this module-level variable: never in
// any browser persistent storage or cookie, never in React state that could be
// persisted. A page reload therefore requires signing in again by design.
// Identity (user, tenant, tier) is decided by the server from the token; the
// client never decodes or asserts it.
// ---------------------------------------------------------------------------

export type ClearReason = "logout" | "unauthorized";

export interface AuthSnapshot {
  authenticated: boolean;
  /** Message shown on the login form after a forced sign-out. */
  notice: string | null;
}

const UNAUTHORIZED_NOTICE =
  "Your session expired or was rejected by the server. Please sign in again.";

let token: string | null = null;
let snapshot: AuthSnapshot = { authenticated: false, notice: null };

const subscribers = new Set<() => void>();
const clearedListeners = new Set<() => void>();

function publish(next: AuthSnapshot): void {
  snapshot = next;
  subscribers.forEach((l) => l());
}

export function getToken(): string | null {
  return token;
}

export function setToken(value: string): void {
  token = value;
  publish({ authenticated: true, notice: null });
}

/** Stable object identity between changes (useSyncExternalStore friendly). */
export function getAuthSnapshot(): AuthSnapshot {
  return snapshot;
}

export function subscribeAuth(listener: () => void): () => void {
  subscribers.add(listener);
  return () => {
    subscribers.delete(listener);
  };
}

/**
 * Teardown hooks run when the session ends (logout or 401): streams are
 * aborted and chat/pending state of the previous user is dropped.
 */
export function onSessionCleared(listener: () => void): () => void {
  clearedListeners.add(listener);
  return () => {
    clearedListeners.delete(listener);
  };
}

export function clearSession(reason: ClearReason = "logout"): void {
  token = null;
  clearedListeners.forEach((l) => l());
  publish({
    authenticated: false,
    notice: reason === "unauthorized" ? UNAUTHORIZED_NOTICE : null,
  });
}

/**
 * 401 handling for a request that was sent with `usedToken`. A late 401
 * belonging to a previous user's token must not log out the current user.
 */
export function handleUnauthorized(usedToken: string | null): void {
  if (usedToken !== null && usedToken === token) {
    clearSession("unauthorized");
  }
}
