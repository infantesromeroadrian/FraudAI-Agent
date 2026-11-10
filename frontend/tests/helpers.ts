// Shared test helpers: scripted fetch + SSE bodies. Synthetic data only.

export interface RecordedCall {
  url: string;
  init: RequestInit;
  headers: Headers;
}

type Responder = Response | ((call: RecordedCall) => Response | Promise<Response>);

export function installFetch(responders: Responder[]): {
  calls: RecordedCall[];
  restore: () => void;
} {
  const calls: RecordedCall[] = [];
  const original = globalThis.fetch;
  let i = 0;
  globalThis.fetch = (async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const call: RecordedCall = {
      url: String(input),
      init,
      headers: new Headers(init.headers),
    };
    calls.push(call);
    const r = responders[Math.min(i, responders.length - 1)];
    i += 1;
    return typeof r === "function" ? r(call) : r.clone();
  }) as typeof fetch;
  return {
    calls,
    restore: () => {
      globalThis.fetch = original;
    },
  };
}

export function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/** SSE body delivered as the given raw chunks (to exercise chunk boundaries). */
export function sse(chunks: string[], status = 200): Response {
  const enc = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(enc.encode(c));
      controller.close();
    },
  });
  return new Response(stream, {
    status,
    headers: { "Content-Type": "text/event-stream" },
  });
}

export function ev(obj: unknown): string {
  return `data: ${JSON.stringify(obj)}\n\n`;
}

export const PENDING = {
  kind: "offensive_tool_approval" as const,
  action_id: "11111111-1111-4111-8111-111111111111",
  agent: "mike",
  tool: "adversarial_evasion",
  arguments: { target_endpoint: "https://target.example/api", attack_type: "sign_noise" },
  target: "https://target.example/api",
  warning: "Offensive tool: requires explicit approval.",
};

export function chatResponse(over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    session_id: "s-1",
    agent: "mike",
    message: "ok",
    citations: [],
    tool_results: [],
    metadata: {
      routing_agent: "mike",
      routing_confidence: 1,
      tokens_used: 0,
      latency_ms: 1,
      corpus_version: "x",
    },
    pending_action: null,
    ...over,
  };
}

export function waitFor(cond: () => boolean, ms = 1000): Promise<void> {
  return new Promise((resolve, reject) => {
    const t0 = Date.now();
    const tick = () => {
      if (cond()) return resolve();
      if (Date.now() - t0 > ms) return reject(new Error("waitFor timeout"));
      setTimeout(tick, 5);
    };
    tick();
  });
}
