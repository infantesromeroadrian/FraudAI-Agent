"use client";

import { useEffect, useRef, type KeyboardEvent } from "react";
import type { PendingAction } from "../../lib/types";
import { formatArguments } from "../../lib/hitl";

// ---------------------------------------------------------------------------
// HITL Confirmation Dialog — per-call human approval (F4 contract, SEC-006)
//
// Everything model-originated (tool, arguments, target, warning) is rendered
// as plain text nodes: never markdown or HTML. There is deliberately no
// "close" / Escape path: the only ways out are an explicit Reject or Approve.
// ---------------------------------------------------------------------------

interface ConfirmationDialogProps {
  pending: PendingAction;
  /** A decision request is in flight: both buttons are disabled. */
  isSubmitting: boolean;
  error: string | null;
  notice: string | null;
  onDecision: (approved: boolean) => void;
}

export default function ConfirmationDialog({
  pending,
  isSubmitting,
  error,
  notice,
  onDecision,
}: ConfirmationDialogProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const rejectRef = useRef<HTMLButtonElement>(null);
  const approveRef = useRef<HTMLButtonElement>(null);
  const previousFocus = useRef<HTMLElement | null>(null);

  // Remember where the user was and give focus back when the dialog closes.
  useEffect(() => {
    previousFocus.current = document.activeElement as HTMLElement | null;
    return () => previousFocus.current?.focus?.();
  }, []);

  // Land on the safe choice (Reject) for each action and after each request.
  useEffect(() => {
    if (!isSubmitting) rejectRef.current?.focus();
  }, [pending.action_id, isSubmitting]);

  // Keep Tab inside the modal; Escape intentionally does nothing.
  function handleKeyDown(e: KeyboardEvent<HTMLDivElement>) {
    if (e.key !== "Tab") return;
    const focusables = dialogRef.current?.querySelectorAll<HTMLElement>(
      'button:not([disabled]), [tabindex="0"]',
    );
    if (!focusables || focusables.length === 0) {
      e.preventDefault();
      dialogRef.current?.focus();
      return;
    }
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    const active = document.activeElement;
    if (e.shiftKey && (active === first || active === dialogRef.current)) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && active === last) {
      e.preventDefault();
      first.focus();
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div
        ref={dialogRef}
        tabIndex={-1}
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="hitl-dialog-title"
        aria-describedby="hitl-dialog-desc"
        aria-busy={isSubmitting}
        onKeyDown={handleKeyDown}
        className="mx-4 max-h-[90vh] w-full max-w-lg overflow-y-auto rounded-xl border border-orange-500/40 bg-gray-900 p-6 shadow-2xl focus:outline-none"
      >
        {/* Warning header */}
        <div className="mb-4 flex items-start gap-3">
          <span
            className="mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-orange-500/20 text-xl"
            aria-hidden="true"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="currentColor"
              className="h-6 w-6 text-orange-400"
            >
              <path
                fillRule="evenodd"
                d="M11.484 2.17a.75.75 0 0 1 1.032 0 11.209 11.209 0 0 0 7.877 3.08.75.75 0 0 1 .722.515 12.74 12.74 0 0 1 .635 3.985c0 5.369-3.525 9.921-8.392 11.473a.75.75 0 0 1-.468 0C7.907 19.671 4.38 15.12 4.38 9.75c0-1.394.224-2.735.635-3.985a.75.75 0 0 1 .722-.516 11.21 11.21 0 0 0 7.877-3.08ZM12 8.25a.75.75 0 0 1 .75.75v3.75a.75.75 0 0 1-1.5 0V9a.75.75 0 0 1 .75-.75Zm0 8.25a.75.75 0 1 0 0-1.5.75.75 0 0 0 0 1.5Z"
                clipRule="evenodd"
              />
            </svg>
          </span>
          <div>
            <h2
              id="hitl-dialog-title"
              className="text-lg font-semibold text-orange-300"
            >
              Offensive Action Confirmation
            </h2>
            <p className="mt-1 text-xs text-orange-400/80">
              Nothing has run yet. Review exactly what will be executed, then
              approve or reject this single call.
            </p>
          </div>
        </div>

        {/* What will run (plain text only) */}
        <dl
          id="hitl-dialog-desc"
          className="mb-4 space-y-3 rounded-lg border border-gray-700 bg-gray-800 p-4 text-sm text-gray-300"
        >
          <div>
            <dt className="text-xs uppercase tracking-wide text-gray-500">Agent</dt>
            <dd className="break-words">{pending.agent}</dd>
          </div>
          <div>
            <dt className="text-xs uppercase tracking-wide text-gray-500">Tool</dt>
            <dd className="break-words font-mono text-gray-100">{pending.tool}</dd>
          </div>
          <div>
            <dt className="text-xs uppercase tracking-wide text-gray-500">Target</dt>
            <dd className="break-all font-mono text-gray-100">
              {pending.target ?? "No specific target identified — you approve the whole call."}
            </dd>
          </div>
          <div>
            <dt className="text-xs uppercase tracking-wide text-gray-500">
              Arguments (exactly as they will be executed)
            </dt>
            <dd>
              <pre
                tabIndex={0}
                aria-label="Tool arguments"
                className="mt-1 max-h-60 overflow-auto whitespace-pre-wrap break-words rounded bg-gray-900 p-2 font-mono text-xs text-gray-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-gray-500"
              >
                {formatArguments(pending.arguments)}
              </pre>
            </dd>
          </div>
          {pending.warning ? (
            <div>
              <dt className="text-xs uppercase tracking-wide text-orange-400">Warning</dt>
              <dd className="break-words text-orange-200">{pending.warning}</dd>
            </div>
          ) : null}
        </dl>

        {notice && (
          <div
            className="mb-4 rounded-lg border border-blue-500/40 bg-blue-900/30 px-4 py-2 text-sm text-blue-200"
            role="status"
          >
            {notice}
          </div>
        )}

        {error && (
          <div
            className="mb-4 rounded-lg border border-red-500/40 bg-red-900/30 px-4 py-2 text-sm text-red-300"
            role="alert"
          >
            {error}
          </div>
        )}

        <p className="mb-4 text-xs text-gray-500">
          Action ID:{" "}
          <code className="break-all rounded bg-gray-800 px-1.5 py-0.5 font-mono text-gray-400">
            {pending.action_id}
          </code>
        </p>

        <p className="sr-only" role="status" aria-live="polite">
          {isSubmitting ? "Submitting your decision…" : ""}
        </p>

        {/* Buttons: explicit decision only */}
        <div className="flex items-center justify-end gap-3">
          <button
            ref={rejectRef}
            type="button"
            onClick={() => onDecision(false)}
            disabled={isSubmitting}
            className="rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-red-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-400 focus-visible:ring-offset-2 focus-visible:ring-offset-gray-900 disabled:cursor-not-allowed disabled:opacity-50"
            aria-label={`Reject ${pending.tool}`}
          >
            {isSubmitting ? "Sending…" : "Reject"}
          </button>
          <button
            ref={approveRef}
            type="button"
            onClick={() => onDecision(true)}
            disabled={isSubmitting}
            className="rounded-lg bg-green-700 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-green-600 focus:outline-none focus-visible:ring-2 focus-visible:ring-green-400 focus-visible:ring-offset-2 focus-visible:ring-offset-gray-900 disabled:cursor-not-allowed disabled:opacity-50"
            aria-label={`Approve ${pending.tool}`}
          >
            {isSubmitting ? "Sending…" : "Approve"}
          </button>
        </div>
      </div>
    </div>
  );
}
