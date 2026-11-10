#!/usr/bin/env bash
# Shared helpers for deploy.sh / rollback.sh / smoke-test.sh (sourced, not executed).
# Compatible with the bash 3.2 that ships with macOS.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${PROJECT_DIR}/docker-compose.prod.yml"
ENV_FILE="${FRAUDAI_ENV_FILE:-${PROJECT_DIR}/.env}"
API_REPO="fraudai-agent"
# shellcheck disable=SC2034  # used by the scripts that source this file
FRONTEND_REPO="fraudai-frontend"
REVISION_LABEL="org.opencontainers.image.revision"
STATE_DIR="${FRAUDAI_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/fraudai-agent}"
STATE_LOG="${STATE_DIR}/deployments.log"
FRAUDAI_PORT="${FRAUDAI_PORT:-8000}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[+]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[~]${NC} $*"; }
log_error() { echo -e "${RED}[!]${NC} $*" >&2; }
die()       { log_error "$*"; exit 1; }

is_full_sha() { [[ "$1" =~ ^[0-9a-f]{40}$ ]]; }

# Revision label of an image reference ("" when the image or label is missing).
image_revision() {
    docker image inspect --format "{{ index .Config.Labels \"${REVISION_LABEL}\" }}" "$1" 2>/dev/null || true
}

image_id() { docker image inspect --format '{{.Id}}' "$1" 2>/dev/null || true; }

# Compose wrapper: always the same file, env file for interpolation and image variables.
compose() {
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" "$@"
}

# SHA of the image the running API container was created from ("" when not running).
running_sha() {
    local cid
    cid="$(compose ps -q fraudai 2>/dev/null | head -n 1)"
    [[ -n "$cid" ]] || return 0
    local img
    img="$(docker inspect --format '{{.Image}}' "$cid" 2>/dev/null || true)"
    [[ -n "$img" ]] && image_revision "$img"
}

# JSON field of GET /api/v1/readyz (first argument = field name). The body is parsed even
# on 503 (starting / not_ready), because build_sha is present in every answer.
served_field() {
    curl -sS --max-time 5 "http://127.0.0.1:${FRAUDAI_PORT}/api/v1/readyz" 2>/dev/null |
        python3 -c 'import json,sys; v=json.load(sys.stdin).get(sys.argv[1]); print("" if v is None else v)' "$1" 2>/dev/null || true
}

# Ready = /readyz answers 200 (qdrant + LLM checks passed in the API's own snapshot).
wait_for_ready() {
    local retries="${1:-60}" code
    for _ in $(seq 1 "$retries"); do
        code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:${FRAUDAI_PORT}/api/v1/readyz" 2>/dev/null || true)"
        [[ "$code" == "200" ]] && return 0
        sleep 2
    done
    return 1
}

# Verify that what is RUNNING is the image we asked for, in three independent ways, all
# mandatory (there is deliberately no switch to skip the last one):
#   1. the container's image ID is the ID of the SHA tag,
#   2. that image's own revision label equals the SHA,
#   3. the build_sha the API itself serves on /readyz equals the SHA.
verify_running_sha() {
    local sha="$1" cid want_id have_id label served
    cid="$(compose ps -q fraudai | head -n 1)"
    [[ -n "$cid" ]] || { log_error "fraudai container is not running."; return 1; }
    want_id="$(image_id "${API_REPO}:${sha}")"
    have_id="$(docker inspect --format '{{.Image}}' "$cid")"
    label="$(image_revision "$have_id")"
    [[ -n "$want_id" && "$want_id" == "$have_id" ]] ||
        { log_error "container image ${have_id} is not ${API_REPO}:${sha} (${want_id})."; return 1; }
    [[ "$label" == "$sha" ]] ||
        { log_error "image revision label '${label}' != ${sha}."; return 1; }
    log_info "container image ${have_id#sha256:} carries revision ${label}"
    served="$(served_field build_sha)"
    if [[ "$served" != "$sha" ]]; then
        log_error "the API serves build_sha '${served:-<none>}' on /api/v1/readyz, expected ${sha}."
        return 1
    fi
    log_info "the API serves build_sha ${served}"
}

record_state() {  # kind sha previous image_id
    mkdir -p "$STATE_DIR"
    printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "${3:--}" "${4:--}" >> "$STATE_LOG"
}

# Last SHA recorded as deployed before the current one ("" when none).
previous_recorded_sha() {
    [[ -f "$STATE_LOG" ]] || return 0
    local current="$1"
    awk -F'\t' -v cur="$current" '$3 != cur && $3 ~ /^[0-9a-f]{40}$/ {last=$3} END {print last}' "$STATE_LOG"
}
