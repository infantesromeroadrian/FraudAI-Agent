<p align="center">
  <img src="assets/logo.png" alt="FraudAI Agent" width="200" />
</p>

<h1 align="center">FraudAI Agent</h1>

<p align="center">
  <strong>AI-powered multi-agent platform for banking fraud detection and AI red teaming in FinTech</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/LangGraph-0.3+-purple?logo=langchain" alt="LangGraph" />
  <img src="https://img.shields.io/badge/FastAPI-0.115+-green?logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Next.js-15-black?logo=next.js" alt="Next.js" />
  <img src="https://img.shields.io/badge/Qdrant-Vector_DB-red" alt="Qdrant" />
  <a href="https://github.com/infantesromeroadrian/FraudAI-Agent/actions/workflows/ci.yml"><img src="https://github.com/infantesromeroadrian/FraudAI-Agent/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License" />
</p>

---

## What is FraudAI Agent?

A multi-agent prototype for fraud investigation. Six specialized AI agents — inspired by the characters of *Suits* — are orchestrated with LangGraph to cover fraud detection, regulatory compliance, fraud-network analysis, red teaming and data engineering.

Agents can call tools (transaction analysis, compliance reports, network graphs, robustness and prompt-injection probes). Tool code is meant to run in ephemeral Docker containers through [`SandboxEngine`](src/fraudai/tools/sandbox.py). See [Implementation status](#implementation-status) for what is implemented, simulated or pending; this is a portfolio project, not a hardened or production-deployed system.

### Agents

| Agent | Role | Personality |
|-------|------|-------------|
| **Donna Paulsen** | Router | Classifies intent, routes to the right specialist |
| **Harvey Specter** | Transaction Fraud Detection | Direct, confident. Analyzes transactions, detects anomalies, scores risk |
| **Louis Litt** | AML / KYC / Compliance | Meticulous. Cites exact legal articles, generates SAR reports |
| **Jessica Pearson** | Fraud Intelligence | Strategic. Network graph analysis, identity resolution |
| **Mike Ross** | AI Red Teaming | Creative. Random-perturbation robustness probes, prompt injection testing |
| **Rachel Zane** | Data Engineering | Rigorous. ETL pipelines, feature engineering, data quality |

---

## Architecture

<p align="center">
  <img src="docs/architecture/system-architecture.svg" alt="Architecture" width="800" />
</p>

**Key components** (see [Implementation status](#implementation-status) for caveats):
- **LangGraph StateGraph** routes between agents with conversational continuity ([`graph.py`](src/fraudai/agents/graph.py)).
- **RAG pipeline code** with Qdrant hybrid search (dense BGE-M3 + sparse BM25): [`rag/`](src/fraudai/rag/), [`ingestion/`](src/fraudai/ingestion/). The repository ships **no corpus**: `data/raw` and `data/processed` are empty, so the size and content of the indexed legal corpus are not verified here.
- **Sandbox engine** that starts ephemeral Docker containers with `cap_drop=ALL` and `no-new-privileges` ([`sandbox.py`](src/fraudai/tools/sandbox.py)). Network is disabled by default; red-team tools turn it on ([`implementations.py`](src/fraudai/tools/implementations.py)).
- **LLM invoker** for Anthropic, Groq and OpenAI-compatible providers ([`claude_invoker.py`](src/fraudai/agents/claude_invoker.py)). The Donna router additionally supports a local Ollama model ([`donna.py`](src/fraudai/agents/donna.py)).
- **JWT authentication** ([`auth.py`](src/fraudai/api/auth.py)). Chat, stream, confirm, upload and session routes check that the session belongs to the caller's tenant, and unknown or foreign sessions return a uniform 404 ([`test_tenant_isolation.py`](tests/test_tenant_isolation.py), [`test_regression_a1_session_owner.py`](tests/test_regression_a1_session_owner.py), [`test_regression_a2_confirm_owner.py`](tests/test_regression_a2_confirm_owner.py); [PR #3](https://github.com/infantesromeroadrian/FraudAI-Agent/pull/3)). That is tenant isolation, not role-based access control (RBAC): the token carries a `tier` claim but no per-tier access enforcement is implemented.
- **Prometheus + Grafana** monitoring config with alert rules in [`monitoring/alerts/rules.yml`](monitoring/alerts/rules.yml); several rules are known to be inert (see the [status table](#implementation-status)).

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM Orchestration** | LangGraph, LangChain |
| **LLM Providers** | Anthropic, Groq, OpenAI-compatible (agent invoker); Ollama (Donna router only) |
| **Vector DB** | Qdrant (hybrid search) |
| **Embeddings** | BGE-M3 (1024-dim, multilingual) |
| **Backend** | FastAPI, Python >=3.11,<3.12 ([`pyproject.toml`](pyproject.toml); `.python-version` 3.11.16) |
| **Frontend** | Next.js 15.5.28, React 18.3, Tailwind CSS ([`frontend/package.json`](frontend/package.json)) |
| **Auth** | JWT (PyJWT) with OAuth2 |
| **Sandbox** | Docker containers (`cap_drop=ALL`, `no-new-privileges`; Docker default seccomp profile) |
| **Monitoring** | Prometheus, Grafana |
| **CI/CD** | GitHub Actions (lint, typecheck, test, docker, security) |
| **Database** | SQLite via aiosqlite (code in [`database.py`](src/fraudai/core/database.py); not validated in the production container), Qdrant (vectors) |

---

## Quick Start

### Prerequisites
- Docker + Docker Compose
- Python 3.11.x and [uv](https://docs.astral.sh/uv/) (uv 0.12.19 is pinned only in the Dockerfile and the Python 3.11.18 container CI job; native CI jobs and local installs are not version-pinned; `.python-version` is 3.11.16)
- Node.js 20 (CI uses 20.20.2) with pnpm 9.15.9 (`packageManager` in `frontend/package.json`)
- An LLM provider API key. `.env.example` defaults to `LLM_PROVIDER=groq` with `GROQ_API_KEY` ([console.groq.com](https://console.groq.com)); `ANTHROPIC_API_KEY` is also supported

### 1. Clone and configure

```bash
git clone https://github.com/infantesromeroadrian/FraudAI-Agent.git
cd FraudAI-Agent
cp .env.example .env
# Edit .env: set your provider key (default provider: groq -> GROQ_API_KEY)
```

### 2. Start infrastructure

```bash
docker compose up -d  # Qdrant + Ollama
```

### 3. Install and run backend

```bash
uv sync --frozen --extra dev
uv run python -m uvicorn fraudai.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

### 4. Install and run frontend

```bash
cd frontend && pnpm install --frozen-lockfile && pnpm dev
```

### 5. Open the app

- **Chat UI**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard

### 6. (Optional) Index Spanish legal corpus

The pipeline downloads from the BOE (Official State Gazette) and needs network access, Qdrant and an embedding model; it is not exercised by the unit tests against the real BOE.

```bash
uv run python -m fraudai.ingestion --mode full
```

---

## Project Structure

```
src/fraudai/
  agents/          # 6 agent definitions, prompts, LangGraph orchestration
  api/             # FastAPI endpoints, auth, session management
  core/            # Config, database, metrics, tracing
  evaluation/      # RAGAS benchmark (50 questions, 5 categories)
  ingestion/       # BOE download pipeline, text extraction, chunking
  rag/             # Retriever, reranker, prompt templates
  tools/           # Sandbox engine, tool implementations

frontend/          # Next.js 15 chat UI with SSE streaming
monitoring/        # Prometheus alerts, Grafana dashboards
docs/              # ADRs, requirements, API reference, runbook
tests/             # unit/integration tests + live smoke tests (need a running backend)
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Requirements (F0)](docs/F0_requirements.md) | Business, user, system, and ML requirements |
| [Architecture Decisions (F0.5)](docs/F0.5_ADRs.md) | 6 ADRs. Design intent, not current state: e.g. ADR-004 seccomp profile and ADR-006 Postgres / `fraudai.api.main:app` are not implemented |
| [Backlog (F0.5)](docs/F0.5_backlog.md) | 74 user stories, 8 sprints, MVP scope |
| [Technical Spec (F2)](docs/F2_spec.md) | Metrics, baselines, SLAs, fairness evaluation |
| [API Reference (F6)](docs/F6_api_reference.md) | All endpoints with curl examples |
| [Deployment Guide (F6)](docs/F6_deployment_guide.md) | Prerequisites, config; its production section is not validated |
| [Operations Runbook (F6)](docs/F6_runbook.md) | Common issues, rollback, scaling |

---

## Testing

```bash
# Unit + integration tests, excluding the live smoke tests.
# Reference, GitHub Actions on main@8e4003a (2026-11-09), run 42692893880:
#   667 passed in Linux Python 3.11.16, 91.93 % line coverage;
#   667 passed in a python:3.11.18 container, 92.35 % line coverage.
# These are CI numbers for that commit only: not a local-machine result, and not
# end-to-end, production or LLM evidence. CI fails below 80 % coverage.
uv run python -m pytest tests/ -q --ignore=tests/test_smoke_live.py

# Live smoke tests (requires running backend)
./scripts/smoke-test.sh

# Coverage report
uv run python -m pytest tests/ --cov=fraudai --cov-report=html
```

---

## Key Design Decisions

- **Hybrid LLM routing**: keywords -> Groq API (if a key is set) -> Ollama (fallback), see [`donna.py`](src/fraudai/agents/donna.py). No latency benchmark exists, so no timing is claimed
- **Conversational continuity**: Follow-up messages stay with the current agent
- **RAG with auto-filtering**: mentions of specific laws are meant to trigger metadata filters ([`retriever.py`](src/fraudai/rag/retriever.py))
- **Sandboxed execution by design**: tool code runs in Docker with dropped capabilities (see status below for limits)
- **Human-in-the-loop (HITL) for red teaming**: the graph pauses for approval before Mike Ross runs ([`graph.py`](src/fraudai/agents/graph.py), `POST /confirm` in [`routes.py`](src/fraudai/api/routes.py)). The mechanism exists; it is not yet an operational guarantee (open security findings, pending)

---

## Implementation status

This is a **portfolio project** for AI agent architecture in FinTech. It is not production-deployed and its production configuration is not validated. Local reproducible development is the supported goal; cloud deployment is out of scope for now.

| Area | Implemented (code + tests) | Simulated / partial | Pending |
|---|---|---|---|
| Agents and routing | LangGraph graph, Donna routing ([`test_graph.py`](tests/test_graph.py), [`test_donna.py`](tests/test_donna.py)) | Agent answers depend on the configured LLM; no quality benchmark | Ollama cloud-model support (decided, not verified) |
| Tools | Real implementations behind the sandbox ([`test_tool_implementations.py`](tests/test_tool_implementations.py), [`test_tool_registry.py`](tests/test_tool_registry.py)) | Sandbox tests use a mocked Docker client ([`test_sandbox.py`](tests/test_sandbox.py)). Docker-backed tools are unavailable when the sandbox or registry cannot be initialized; the production Compose does not mount a Docker socket, so real sandbox execution there is **not** supported | Live sandbox test; production wiring |
| Sandbox isolation | `cap_drop=ALL`, `no-new-privileges`, network disabled by default | Docker default seccomp only (no custom profile). Red-team tools run **with network**, filtered by a URL check ([`test_security_controls.py`](tests/test_security_controls.py)) | Egress policy; custom seccomp |
| Authentication | JWT validation ([`test_auth.py`](tests/test_auth.py)); staging and production refuse to start without a valid `JWT_SECRET`, and only `ENVIRONMENT=development` (the default) accepts `dev-token` and free login | Development accepts any login and `dev-token` as enterprise tier | User store (`/auth/token` returns 501 in staging and production); tier/role enforcement (RBAC) |
| Authorization and approval | Per-session tenant ownership on chat, stream, confirm, upload and session routes, with server-generated UUIDs and uniform 404 ([PR #3](https://github.com/infantesromeroadrian/FraudAI-Agent/pull/3); tests listed under Key components). HITL pause/resume in the graph | The approval mechanism is present | Approval bound to a specific `action_id`/`pending_action` (F4); other audit findings. Not an overall guarantee of authorization or approval |
| RAG | Retriever, reranker, ingestion code and tests | Evaluator computes context precision/recall only; `faithfulness` and `answer_relevancy` are `0.0` placeholders | Corpus not included; no verified document count |
| Persistence | SQLite code and tests ([`test_database.py`](tests/test_database.py)) | Works in local development | Not validated in the production container |
| Monitoring | Prometheus metrics, 23 rules in `rules.yml` | At least 7 rules are known to be inert (never fire) | Rule review |
| Production Compose | [`docker-compose.prod.yml`](docker-compose.prod.yml) exists | Publishes the API, Prometheus and Grafana on all interfaces | Hardening; explicit disabling of offensive tools; not tested end to end |

**Known limits:**
- `adversarial_evasion` is a black-box **random-perturbation** robustness probe. It uses no gradients, so it is not an implementation of FGSM, PGD, C&W or DeepFool.
- Sandbox isolation is covered by unit tests with a mocked Docker client and by image builds in CI, not by a live escape test.
- Session ownership checks (A1) and approval of another tenant's session (A2) are fixed and covered by regression tests. Other audit findings remain open, notably SSRF through redirects and egress policy (F3) and approval bound to a specific pending action (F4). Treat approval and sandbox controls as unverified until those land.
- No production deployment and no complete local smoke run have been verified. Ollama with a cloud subscription is a decision (D2) whose access, model and quota are not verified; the current agent invoker supports Anthropic, Groq and OpenAI.

---

## Security Notes

- **Red-team targets.** The networked sandbox only accepts public `http(s)` endpoints. Every resolved address of the host must be public; loopback, private, link-local, CGNAT, unique-local IPv6 and IPv4-mapped forms are rejected, as are URLs with credentials. Residual risk: DNS rebinding between the check and the sandbox connection; closing it needs an egress policy on the sandbox network.
- **Generated code.** Tool arguments are embedded in sandbox code as Python literals (`repr`), and the perturbation type comes from a fixed allow-list, so a crafted endpoint or argument cannot inject code.
- **Input files.** Sandbox inputs must resolve inside the allowed directories; prefix look-alikes and symlinks that escape them are rejected.
- **Local services.** The development [`docker-compose.yml`](docker-compose.yml) publishes Qdrant and Ollama on `127.0.0.1`; neither has authentication enabled. The production Compose publishes the API, Prometheus and Grafana on all interfaces, so do not expose it as is.
- **Supply chain.** Python dependencies install from the committed `uv.lock` with `uv sync --frozen`; the frontend uses `pnpm install --frozen-lockfile`; the sandbox image uses a hashed `requirements.lock`. External images are pinned by digest, while local `latest` tags remain temporary until F3/F5a. Builds are pinned, not claimed to be bit-for-bit identical. GitHub Actions are pinned to commit SHAs. The weekly [Security workflow](.github/workflows/security.yml) runs `pip-audit`, a frontend audit and TruffleHog (repository secrets, not image layers).
- **Vulnerability scans.** On main@8e4003a (2026-11-09), CI run [42692893880](https://github.com/infantesromeroadrian/FraudAI-Agent/actions/runs/42692893880) and Security run [42692893879](https://github.com/infantesromeroadrian/FraudAI-Agent/actions/runs/42692893879) succeeded. Trivy found 0 fixable HIGH/CRITICAL vulnerabilities in the API and sandbox images (artifact `trivy-reports`). The full frontend audit still reports a moderate `minimatch` issue in tooling only, with no production advisories. `pip-audit` cannot audit `torch 2.15.0+cpu` (not on PyPI); it is covered by the Trivy image scan. This is not a claim of zero vulnerabilities.
- **Type checking.** CI runs `uv run mypy src/fraudai/` with `strict = true` set in [`pyproject.toml`](pyproject.toml), except `warn_unused_ignores = false`. Specific modules have their own exceptions, and there is no global `ignore_missing_imports`.

---

## Author

**Adrian Infantes** — Senior AI Engineer specializing in FinTech, fraud detection, and AI red teaming.

- [LinkedIn](https://linkedin.com/in/adrianinfantes)
- [GitHub](https://github.com/infantesromeroadrian)

---

## License

MIT License. See [LICENSE](LICENSE).
