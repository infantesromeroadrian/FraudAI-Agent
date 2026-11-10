"use client";

import { useId, useState, type FormEvent } from "react";
import { login, LoginError } from "@/lib/api";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";

interface LoginFormProps {
  /** Shown when the previous session ended (401) — asks to sign in again. */
  notice: string | null;
}

/**
 * Minimal sign-in form. The token returned by the server is kept in memory by
 * `login()`; this component never stores credentials or the token anywhere
 * and discards the password from state as soon as the request ends.
 */
export function LoginForm({ notice }: LoginFormProps) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const errorId = useId();

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (isSubmitting || !username.trim() || !password) return;
    setIsSubmitting(true);
    setError(null);
    try {
      await login(username.trim(), password);
      // AuthGate swaps to the app on success; nothing else to do.
    } catch (err) {
      setError(
        err instanceof LoginError
          ? err.message
          : "Sign-in failed. Please try again.",
      );
      setIsSubmitting(false);
    } finally {
      setPassword("");
    }
  }

  return (
    <div className="flex h-screen items-center justify-center bg-zinc-950 px-4">
      <form
        onSubmit={handleSubmit}
        className="w-full max-w-sm space-y-4 rounded-xl border border-zinc-800 bg-zinc-900 p-6 shadow-xl"
        aria-labelledby="login-title"
        aria-describedby={error ? errorId : undefined}
      >
        <div>
          <h1 id="login-title" className="text-lg font-semibold text-zinc-100">
            Sign in to FraudAI Agent
          </h1>
          <p className="mt-1 text-xs text-zinc-500">
            Your session is kept in memory only; reloading the page signs you out.
          </p>
        </div>

        {notice && !error && (
          <p
            role="status"
            className="rounded-lg border border-yellow-500/40 bg-yellow-900/20 px-3 py-2 text-sm text-yellow-200"
          >
            {notice}
          </p>
        )}

        <Input
          label="Username"
          name="username"
          autoComplete="username"
          autoFocus
          required
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          disabled={isSubmitting}
        />
        <Input
          label="Password"
          name="password"
          type="password"
          autoComplete="current-password"
          required
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          disabled={isSubmitting}
        />

        {error && (
          <p
            id={errorId}
            role="alert"
            className="rounded-lg border border-red-500/40 bg-red-900/30 px-3 py-2 text-sm text-red-300"
          >
            {error}
          </p>
        )}

        <Button
          type="submit"
          className="w-full"
          disabled={isSubmitting || !username.trim() || !password}
          aria-busy={isSubmitting}
        >
          {isSubmitting ? "Signing in…" : "Sign in"}
        </Button>
      </form>
    </div>
  );
}
