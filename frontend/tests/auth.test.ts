import { afterEach, beforeEach, describe, it } from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

import { clearSession, getAuthSnapshot, getToken, onSessionCleared } from "../src/lib/auth";
import { ApiError, login, LoginError, healthCheck, sendMessage } from "../src/lib/api";
import { installFetch, json } from "./helpers";

const TOKEN_OK = { access_token: "synthetic.jwt.value", token_type: "bearer", expires_in: 3600 };

let restore: () => void = () => undefined;
beforeEach(() => clearSession("logout"));
afterEach(() => restore());

describe("login (F5b.1)", () => {
  it("posts urlencoded username/password to /auth/token and keeps the token in memory", async () => {
    const f = installFetch([json(200, TOKEN_OK)]);
    restore = f.restore;
    await login("analyst", "pw-synthetic");
    assert.equal(f.calls.length, 1);
    assert.match(f.calls[0].url, /\/auth\/token$/);
    assert.equal(f.calls[0].init.method, "POST");
    const body = new URLSearchParams(String(f.calls[0].init.body));
    assert.equal(body.get("username"), "analyst");
    assert.equal(body.get("password"), "pw-synthetic");
    assert.match(f.calls[0].headers.get("content-type") ?? "", /application\/x-www-form-urlencoded/);
    assert.equal(f.calls[0].headers.get("authorization"), null);
    assert.equal(getToken(), "synthetic.jwt.value");
    assert.equal(getAuthSnapshot().authenticated, true);
  });

  it("never touches localStorage, sessionStorage or cookies", async () => {
    const trap = new Proxy(
      {},
      {
        get() {
          throw new Error("storage access");
        },
        set() {
          throw new Error("storage access");
        },
      },
    );
    const g = globalThis as Record<string, unknown>;
    g.localStorage = trap;
    g.sessionStorage = trap;
    g.document = trap;
    const f = installFetch([json(200, TOKEN_OK), json(200, { status: "healthy" })]);
    restore = () => {
      f.restore();
      delete g.localStorage;
      delete g.sessionStorage;
      delete g.document;
    };
    await login("analyst", "pw-synthetic");
    await healthCheck();
    clearSession("logout");
  });

  it("maps 401 to invalid_credentials and leaves no token", async () => {
    const f = installFetch([json(401, { detail: "Incorrect" })]);
    restore = f.restore;
    await assert.rejects(login("a", "b"), (e: unknown) => {
      assert.ok(e instanceof LoginError);
      assert.equal(e.kind, "invalid_credentials");
      return true;
    });
    assert.equal(getToken(), null);
  });

  it("maps 503 (and 501) to server_unavailable with a clear message", async () => {
    for (const status of [503, 501]) {
      const f = installFetch([json(status, { detail: "Authentication is not configured." })]);
      restore = f.restore;
      await assert.rejects(login("a", "b"), (e: unknown) => {
        assert.ok(e instanceof LoginError);
        assert.equal(e.kind, "server_unavailable");
        assert.match(e.message, /not configured|unavailable/i);
        return true;
      });
      f.restore();
    }
    assert.equal(getToken(), null);
  });

  it("maps 429 (login throttle) to rate_limited and mentions Retry-After", async () => {
    const f = installFetch([
      new Response(JSON.stringify({ detail: "Too many failed login attempts. Try again later." }), {
        status: 429,
        headers: { "Content-Type": "application/json", "Retry-After": "42" },
      }),
    ]);
    restore = f.restore;
    await assert.rejects(login("a", "b"), (e: unknown) => {
      assert.ok(e instanceof LoginError);
      assert.equal(e.kind, "rate_limited");
      assert.match(e.message, /too many/i);
      assert.match(e.message, /42/);
      return true;
    });
    assert.equal(getToken(), null);
  });

  it("reports network failures without leaking the password", async () => {
    const f = installFetch([
      () => {
        throw new TypeError("fetch failed");
      },
    ]);
    restore = f.restore;
    await assert.rejects(login("a", "secret-pw"), (e: unknown) => {
      assert.ok(e instanceof LoginError);
      assert.equal(e.kind, "network");
      assert.ok(!e.message.includes("secret-pw"));
      return true;
    });
  });

  it("rejects a malformed token response", async () => {
    const f = installFetch([json(200, { nope: true })]);
    restore = f.restore;
    await assert.rejects(login("a", "b"), LoginError);
    assert.equal(getToken(), null);
  });
});

describe("Authorization and 401 handling", () => {
  it("sends the in-memory bearer token on API calls", async () => {
    const f = installFetch([json(200, TOKEN_OK), json(200, {})]);
    restore = f.restore;
    await login("a", "b");
    await sendMessage({ message: "hola" });
    assert.equal(f.calls[1].headers.get("authorization"), "Bearer synthetic.jwt.value");
  });

  it("401 clears the token, notifies listeners and asks to sign in again", async () => {
    const f = installFetch([json(200, TOKEN_OK), json(401, { detail: "expired" })]);
    restore = f.restore;
    let cleared = 0;
    const off = onSessionCleared(() => {
      cleared += 1;
    });
    await login("a", "b");
    await assert.rejects(sendMessage({ message: "x" }), (e: unknown) => {
      assert.ok(e instanceof ApiError);
      assert.equal(e.status, 401);
      return true;
    });
    off();
    assert.equal(getToken(), null);
    assert.equal(cleared, 1);
    assert.match(getAuthSnapshot().notice ?? "", /sign in/i);
  });

  it("a late 401 for the previous user's token does not log out the new user", async () => {
    let release: (r: Response) => void = () => undefined;
    const late = new Promise<Response>((res) => {
      release = res;
    });
    const f = installFetch([json(200, TOKEN_OK), () => late, json(200, { ...TOKEN_OK, access_token: "second.jwt" })]);
    restore = f.restore;
    await login("a", "b");
    const inflight = sendMessage({ message: "x" }).catch(() => undefined);
    clearSession("logout");
    await login("c", "d");
    release(json(401, { detail: "old" }));
    await inflight;
    assert.equal(getToken(), "second.jwt");
  });

  it("logout clears the token and listeners run", () => {
    let n = 0;
    const off = onSessionCleared(() => {
      n += 1;
    });
    clearSession("logout");
    off();
    assert.equal(n, 1);
  });
});

describe("no dev-token in the frontend (F5b.1 acceptance)", () => {
  function walk(dir: string, out: string[] = []): string[] {
    for (const name of readdirSync(dir)) {
      if (name === "node_modules" || name === ".next" || name === ".test-build") continue;
      const p = join(dir, name);
      if (statSync(p).isDirectory()) walk(p, out);
      else out.push(p);
    }
    return out;
  }
  it("has no dev-token / persistent token storage in src", () => {
    const files = walk(join(__dirname, "..", "..", "src")).filter((f) => /\.(ts|tsx)$/.test(f));
    assert.ok(files.length > 10);
    for (const f of files) {
      const text = readFileSync(f, "utf8");
      assert.ok(!/dev-token/.test(text), `dev-token in ${f}`);
      assert.ok(!/(localStorage|sessionStorage|document\.cookie)/.test(text), `storage use in ${f}`);
    }
  });
});
