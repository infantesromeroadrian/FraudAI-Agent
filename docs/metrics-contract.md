# Metrics contract (F5c.2)

This file lists exactly the Prometheus metrics the API emits today. Dashboards and alerts
(F5a.3) may use only the names in the first table. `tests/test_f5c_metrics.py` fails if this
list and the code drift apart: every row has a test that makes the metric move, and every
metric defined in `src/fraudai/core/metrics.py` appears here.

- Endpoint: `GET /api/v1/admin/metrics`, Prometheus text format, per-process registry.
- Access: a dedicated static scrape credential (`METRICS_SCRAPE_TOKEN_FILE` / `METRICS_SCRAPE_TOKEN`)
  or an admin JWT. The scrape credential is valid only on this route.
- Everything is per process. With more than one worker, aggregate in Prometheus.
- Labels are bounded by design: no user text, no session ids, no tenant ids, no paths of
  uploaded files, no model-chosen free text.

## Emitted metrics

| Metric | Type | Labels | Emitted by |
|---|---|---|---|
| `fraudai_requests_total` | counter | `method`, `endpoint`, `status_code` | `api/middleware.py` for every request |
| `fraudai_request_duration_seconds` | histogram | `method`, `endpoint` | `api/middleware.py` |
| `fraudai_agent_invocations_total` | counter | `agent_name` | `agents/claude_invoker.py` (`plan`) |
| `fraudai_agent_duration_seconds` | histogram | `agent_name` | `agents/claude_invoker.py` (`build_result`) and `agents/donna.py` |
| `fraudai_routing_total` | counter | `target_agent`, `confidence_bucket` | `agents/donna.py` |
| `fraudai_retrieval_duration_seconds` | histogram | `collection` | `rag/retriever.py` |
| `fraudai_retrieval_results_count` | histogram | `collection` | `rag/retriever.py` |
| `fraudai_tool_calls_total` | counter | `tool_name`, `status` | `agents/claude_invoker.py` (`run_tool_call`) |
| `fraudai_tool_duration_seconds` | histogram | `tool_name` | `agents/claude_invoker.py` (`run_tool_call`) |
| `fraudai_sandbox_executions_total` | counter | `status` | `tools/sandbox.py` |
| `fraudai_sandbox_duration_seconds` | histogram | none | `tools/sandbox.py` |
| `fraudai_tokens_total` | counter | `agent_name`, `direction` | `agents/claude_invoker.py` (`build_result`) |
| `fraudai_token_cost_usd_total` | counter | `agent_name` | `agents/claude_invoker.py`, only with a configured tariff |
| `fraudai_active_sessions` | gauge | none | `api/session_manager.py` |
| `fraudai_info` | info | `version`, `environment`, `build_sha` | `api/app.py` at startup |

Prometheus exposes histograms as `<name>_bucket`, `<name>_sum` and `<name>_count`.

### Label values

- `endpoint`: the matched route template (for example `/api/v1/sessions/{session_id}`). A
  request that matches no route is labelled `unmatched`, so scanners cannot create series.
- `method`: one of `GET`, `POST`, `PUT`, `PATCH`, `DELETE`, `HEAD`, `OPTIONS`; any other
  value the client sends is reported as `OTHER`, so invented methods cannot create series.
- `status_code`: the HTTP status as a string. `/api/v1/chat/stream` is measured until the
  response starts, not until the last event.
- `agent_name`, `target_agent`: `donna`, `harvey`, `louis`, `jessica`, `mike`, `rachel`
  (and `clarify` for `target_agent`).
- `confidence_bucket`: `high` (>= 0.85), `medium` (>= 0.70), `low`.
- `tool_name`: the registered tool name, or `unknown` when the model asked for a tool that
  does not exist. `status`: `success` or `error`. A tool that declares it did not run
  (sandbox disabled or unavailable) counts as `error`.
- `direction`: `input` or `output`.
- `fraudai_sandbox_executions_total{status}`: `success`, `error`, `timeout`.
- `fraudai_info{build_sha}`: the value of `BUILD_SHA` when it is valid, otherwise the literal
  `unknown`. It is never computed.

### Semantics worth knowing

- `fraudai_tool_calls_total` / `fraudai_tool_duration_seconds` count executed or attempted
  calls. Offensive calls rejected or refused before approval (F4) never reach a tool, so they
  are not counted.
- `fraudai_tokens_total` comes from the usage the provider reports. If a provider reports
  none, nothing is emitted for that call.
- `fraudai_active_sessions` is the number of sessions held in this process memory, after
  quota evictions and deletions. It includes sessions restored from SQLite at startup.

### Token cost

`fraudai_token_cost_usd_total` is emitted only when BOTH `LLM_INPUT_COST_PER_MTOK_USD` and
`LLM_OUTPUT_COST_PER_MTOK_USD` are set; the value added per call is
`input_tokens * input_price / 1e6 + output_tokens * output_price / 1e6`.

With Ollama (the personal daemon, subscription, no per-token price) no tariff is configured, so
this metric is **not emitted at all**. The cost is unknown, not zero: do not build panels that
read its absence as "free", and do not set a tariff of 0 to make a panel draw.

## Registered but not emitted

Every metric defined in `core/metrics.py` is emitted above, so nothing registered is silent.

These names appear in `monitoring/alerts/rules.yml` but do not exist in the application, so
those rules can never fire until F5a.3 rewrites them against the table above:

- `fraudai_boe_last_ingestion_timestamp`
- `fraudai_sandbox_containers_active`
- `fraudai_session_cost_usd_bucket`

## Readiness is not a metric

Whether the service is ready comes from `GET /api/v1/readyz` (200/503, cached) and is served
with `build_sha`; there is no readiness gauge. Use the Prometheus `up` series for scrape
health.
