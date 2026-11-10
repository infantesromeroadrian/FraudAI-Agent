import { afterEach, beforeEach, describe, it } from "node:test";
import assert from "node:assert/strict";

import { clearSession, setToken } from "../src/lib/auth";
import { ApiError, uploadFile } from "../src/lib/api";
import { installFetch, json } from "./helpers";

const OK = { file_id: "f1", filename: "a.csv", size_bytes: 3, indexed: true, chunks_generated: 1 };
let restore: () => void = () => undefined;
beforeEach(() => {
  clearSession("logout");
  setToken("synthetic.jwt.value");
});
afterEach(() => restore());

describe("uploadFile (F5b.2)", () => {
  it("sends Bearer header, session_id in the query and only the file part", async () => {
    const f = installFetch([json(200, OK)]);
    restore = f.restore;
    const file = new File(["a,b\n1,2\n"], "a.csv", { type: "text/csv" });
    const res = await uploadFile("sess id/ñ", file);
    assert.equal(res.file_id, "f1");
    const call = f.calls[0];
    const url = new URL(call.url);
    assert.match(url.pathname, /\/files\/upload$/);
    assert.equal(url.searchParams.get("session_id"), "sess id/ñ");
    assert.equal(call.headers.get("authorization"), "Bearer synthetic.jwt.value");
    // browser must set the multipart boundary itself
    assert.equal(call.headers.get("content-type"), null);
    const form = call.init.body as FormData;
    assert.ok(form instanceof FormData);
    assert.deepEqual([...form.keys()], ["file"]);
  });

  it("surfaces server limits as readable errors", async () => {
    const cases: Array<[number, string, RegExp]> = [
      [413, "File too large. Maximum: 104857600 bytes (100 MB).", /too large/i],
      [415, "Content-Type must be multipart/form-data.", /multipart/i],
      [429, "Upload limit reached (20 files per session).", /limit reached/i],
      [404, "Session not found.", /session not found/i],
      [400, "Unsupported file type '.exe'.", /unsupported/i],
    ];
    for (const [status, detail, re] of cases) {
      const f = installFetch([json(status, { detail })]);
      restore = f.restore;
      await assert.rejects(uploadFile("s", new File(["x"], "a.txt")), (e: unknown) => {
        assert.ok(e instanceof ApiError);
        assert.equal(e.status, status);
        assert.match(e.message, re);
        assert.ok(!e.message.startsWith("{"), "message must not be raw JSON");
        return true;
      });
      f.restore();
    }
  });

  it("401 on upload clears the session", async () => {
    const f = installFetch([json(401, { detail: "expired" })]);
    restore = f.restore;
    await assert.rejects(uploadFile("s", new File(["x"], "a.txt")), ApiError);
    const { getToken } = await import("../src/lib/auth");
    assert.equal(getToken(), null);
  });
});
