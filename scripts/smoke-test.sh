#!/usr/bin/env bash
# Smoke test of the local Compose stack: login -> chat (real LLM) -> upload -> HITL -> health.
# Credentials are read from your local .env (FRAUDAI_SMOKE_PASSWORD, and AUTH_USERNAME or
# FRAUDAI_SMOKE_USERNAME) by the Python script itself: nothing to export, nothing in argv or
# shell history. Targets only loopback and a synthetic local service.
#
# Usage: ./scripts/smoke-test.sh [--log FILE] [--env-file FILE] [--expect-sha SHA] [--skip-llm]
# Exit: 0 all passed; 1 a step failed; 2 something was skipped; 3 missing credentials.
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]}")/smoke_test.py" "$@"
