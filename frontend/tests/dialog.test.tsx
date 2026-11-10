import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { renderToStaticMarkup } from "react-dom/server";

import ConfirmationDialog from "../src/components/chat/ConfirmationDialog";
import type { PendingAction } from "../src/lib/types";
import { PENDING } from "./helpers";

function render(over: Partial<PendingAction> = {}, isSubmitting = false, error: string | null = null): string {
  return renderToStaticMarkup(
    <ConfirmationDialog
      pending={{ ...PENDING, ...over }}
      isSubmitting={isSubmitting}
      error={error}
      notice={null}
      onDecision={() => undefined}
    />,
  );
}

describe("ConfirmationDialog (F5b.3)", () => {
  it("shows tool, target, effective arguments and action id before any decision", () => {
    const html = render();
    assert.match(html, /adversarial_evasion/);
    assert.match(html, /https:\/\/target\.example\/api/);
    assert.match(html, /sign_noise/);
    assert.match(html, new RegExp(PENDING.action_id));
    assert.match(html, /Nothing has run yet/);
  });

  it("renders model-originated text as inert text (no HTML injection)", () => {
    const html = render({
      tool: "<script>alert(1)</script>",
      target: "<img src=x onerror=alert(1)>",
      arguments: { a: "<b>bold</b>" },
      warning: "<iframe src=//evil></iframe>",
    });
    assert.ok(!/<script>|<img |<iframe|<b>bold/.test(html));
    assert.match(html, /&lt;script&gt;/);
  });

  it("states when there is no target (null) instead of hiding it", () => {
    assert.match(render({ target: null }), /No specific target/);
  });

  it("offers only explicit Approve/Reject, both disabled while submitting", () => {
    const idle = render();
    assert.match(idle, /Approve adversarial_evasion/);
    assert.match(idle, /Reject adversarial_evasion/);
    assert.ok(!/Cancel|Close/.test(idle), "no tacit dismissal");
    const busy = render({}, true);
    assert.equal((busy.match(/disabled=""/g) ?? []).length, 2);
  });

  it("shows an alert for conflicts / failures", () => {
    assert.match(render({}, false, "That request was out of date."), /role="alert"[^>]*>That request was out of date\./);
  });
});
