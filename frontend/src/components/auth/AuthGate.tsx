"use client";

import { useSyncExternalStore, type ReactNode } from "react";
import { getAuthSnapshot, subscribeAuth } from "@/lib/auth";
import { LoginForm } from "./LoginForm";

/**
 * Renders children only while an in-memory token exists. On logout or 401 the
 * children unmount, which drops chat messages, session list and pending
 * actions of the previous user and aborts any open stream.
 */
export function AuthGate({ children }: { children: ReactNode }) {
  const snap = useSyncExternalStore(subscribeAuth, getAuthSnapshot, getAuthSnapshot);
  if (!snap.authenticated) return <LoginForm notice={snap.notice} />;
  return <>{children}</>;
}
