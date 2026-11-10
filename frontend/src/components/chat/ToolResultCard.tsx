"use client";

import { useState } from "react";
import type { ToolResult } from "@/lib/types";
import { describeToolFailure } from "@/lib/hitl";

// ---------------------------------------------------------------------------
// Tool Result Card — collapsible display for agent tool execution results
// ---------------------------------------------------------------------------

interface ToolResultCardProps {
  /** Accepts either `result` or `toolResult` for compatibility. */
  result?: ToolResult;
  toolResult?: ToolResult;
}

const STATUS_STYLES: Record<
  ToolResult["status"],
  { dot: string; bg: string; border: string; label: string }
> = {
  success: {
    dot: "bg-green-400",
    bg: "bg-green-500/10",
    border: "border-green-500/30",
    label: "Success",
  },
  error: {
    dot: "bg-red-400",
    bg: "bg-red-500/10",
    border: "border-red-500/30",
    label: "Error",
  },
  pending: {
    dot: "bg-yellow-400 animate-pulse",
    bg: "bg-yellow-500/10",
    border: "border-yellow-500/30",
    label: "Pending",
  },
};

/** Heuristic: collapse if the JSON stringified result exceeds this length. */
const COLLAPSE_THRESHOLD = 200;

export function ToolResultCard({ result: resultProp, toolResult }: ToolResultCardProps) {
  const result = resultProp ?? toolResult;
  const resultJson = result?.result
    ? JSON.stringify(result.result, null, 2)
    : null;

  const isLargeResult =
    resultJson !== null && resultJson.length > COLLAPSE_THRESHOLD;

  // Hooks must run before any early return (rules-of-hooks).
  const [expanded, setExpanded] = useState(!isLargeResult);

  if (!result) return null;
  const style = STATUS_STYLES[result.status];
  const failure = describeToolFailure(result);

  return (
    <div
      className={`overflow-hidden rounded-lg border ${style.border} ${style.bg}`}
      role="region"
      aria-label={`Tool result: ${result.tool_name} — ${style.label}`}
    >
      {/* Header */}
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-3 px-4 py-2.5 text-left transition-colors hover:bg-white/5 focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500"
        aria-expanded={expanded}
        aria-controls={`tool-result-${result.tool_name}`}
      >
        {/* Tool icon */}
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 20 20"
          fill="currentColor"
          className="h-4 w-4 shrink-0 text-gray-400"
          aria-hidden="true"
        >
          <path
            fillRule="evenodd"
            d="M14.5 10a4.5 4.5 0 0 0 4.284-5.882c-.105-.324-.51-.391-.752-.15L15.34 6.66a.454.454 0 0 1-.493.1 3.29 3.29 0 0 1-1.604-1.604.454.454 0 0 1 .1-.493l2.691-2.692c.241-.242.174-.647-.15-.752a4.5 4.5 0 0 0-5.873 4.575c.055.873-.128 1.808-.8 2.368l-7.23 6.024a2.724 2.724 0 1 0 3.837 3.837l6.024-7.23c.56-.672 1.495-.855 2.368-.8.096.007.193.01.291.01ZM5 16a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"
            clipRule="evenodd"
          />
        </svg>

        {/* Tool name */}
        <span className="flex-1 font-mono text-sm font-medium text-gray-200">
          {result.tool_name}
        </span>

        {/* Duration */}
        <span className="shrink-0 text-xs text-gray-500">
          {result.duration_ms}ms
        </span>

        {/* Status badge */}
        <span className="flex shrink-0 items-center gap-1.5 rounded-full bg-gray-800/50 px-2.5 py-0.5 text-xs">
          <span
            className={`inline-block h-2 w-2 rounded-full ${style.dot}`}
            aria-hidden="true"
          />
          <span className="text-gray-300">{style.label}</span>
        </span>

        {/* Chevron */}
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 20 20"
          fill="currentColor"
          className={`h-4 w-4 shrink-0 text-gray-500 transition-transform duration-200 ${
            expanded ? "rotate-180" : ""
          }`}
          aria-hidden="true"
        >
          <path
            fillRule="evenodd"
            d="M5.22 8.22a.75.75 0 0 1 1.06 0L10 11.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 9.28a.75.75 0 0 1 0-1.06Z"
            clipRule="evenodd"
          />
        </svg>
      </button>

      {/* Explicit failure (e.g. sandbox_disabled / sandbox_unavailable) */}
      {failure && (
        <p
          role="alert"
          className="border-t border-red-500/30 px-4 py-2 text-sm text-red-300"
        >
          {failure}
        </p>
      )}

      {/* Content — JSON or error */}
      {expanded && (
        <div
          id={`tool-result-${result.tool_name}`}
          className="border-t border-gray-700/50 px-4 py-3"
        >
          {resultJson ? (
            <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-gray-300">
              {resultJson}
            </pre>
          ) : (
            <p className="text-sm italic text-gray-500">
              {result.status === "pending"
                ? "Waiting for result..."
                : "No output available."}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
