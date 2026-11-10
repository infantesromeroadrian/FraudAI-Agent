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
COMPOSE_PROJECT="fraudai-local"  # must equal `name:` in docker-compose.prod.yml (tested)
LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# Non-secret setting with the SAME precedence Compose applies to ${NAME:-default}:
# a non-empty process variable, else the --env-file value, else the default. The dotenv is
# parsed by dotenv_get.py, never sourced or evaluated by a shell.
setting() {
    local name="$1" default="$2" value=""
    if [[ -n "${!name:-}" ]]; then
        printf '%s' "${!name}"
        return 0
    fi
    if [[ -f "$ENV_FILE" ]]; then
        value="$(python3 "${LIB_DIR}/dotenv_get.py" "$ENV_FILE" "$name" 2>/dev/null || true)"
    fi
    printf '%s' "${value:-$default}"
}

# Ports published on 127.0.0.1 (and used for probes, the compiled API URL and the summary).
resolve_port() {
    local name="$1" default="$2" value digits
    value="$(setting "$name" "$default")"
    if [[ "$value" =~ ^[0-9]+$ ]]; then
        # Leading zeros are removed with string operations (Bash arithmetic would read
        # "080000" as an invalid octal and overflow on very long numbers), and the length
        # is bounded before any arithmetic.
        digits="${value#"${value%%[!0]*}"}"
        if [[ -n "$digits" && ${#digits} -le 5 ]] && (( 10#$digits >= 1 && 10#$digits <= 65535 )); then
            printf '%s' "$digits"
            return 0
        fi
    fi
    die "${name} must be a decimal port number between 1 and 65535 (got a different value)."
}
FRAUDAI_PORT="$(resolve_port FRAUDAI_PORT 8000)"
# shellcheck disable=SC2034  # used by the scripts that source this file
FRONTEND_PORT="$(resolve_port FRONTEND_PORT 3000)"
# shellcheck disable=SC2034
GRAFANA_PORT="$(resolve_port GRAFANA_PORT 3001)"
# shellcheck disable=SC2034
PROMETHEUS_PORT="$(resolve_port PROMETHEUS_PORT 9090)"

# Compose wrapper: always the same file, env file for interpolation and image variables.
compose() {
    docker compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" "$@"
}

# ID of the running API container, found by Compose's own labels. This deliberately does
# NOT go through `docker compose`: Compose must interpolate the required FRAUDAI_IMAGE /
# FRONTEND_IMAGE first, which a plain shell does not provide, and that failure would look
# exactly like "nothing is running". Prints the id ("" when no container is running);
# returns 2 when Docker itself cannot be queried, which callers must treat as an error.
running_container_id() {
    local out
    if ! out="$(docker ps --filter "label=com.docker.compose.project=${COMPOSE_PROJECT}" \
        --filter "label=com.docker.compose.service=fraudai" --format '{{.ID}}' 2>&1)"; then
        log_error "cannot query Docker for the running API container: ${out}"
        return 2
    fi
    printf '%s\n' "$out" | head -n 1
}

# SHA of the image the running API container was created from. Three outcomes, never mixed:
#   - no container running: prints nothing, returns 0 (genuine absence);
#   - a container was found but cannot be inspected, or its image cannot: error, returns 2;
#   - the image carries no revision label: the revision is UNKNOWN, error, returns 3.
# Unlike image_revision (lenient by design, used for tag checks), nothing is swallowed
# here: callers abort before building or changing anything.
running_sha() {
    local cid img rev
    cid="$(running_container_id)" || return 2
    [[ -n "$cid" ]] || return 0
    if ! img="$(docker inspect --format '{{.Image}}' "$cid" 2>&1)" || [[ -z "$img" ]]; then
        log_error "cannot inspect the running API container ${cid}: ${img:-empty image id}"
        return 2
    fi
    if ! rev="$(docker image inspect --format "{{ index .Config.Labels \"${REVISION_LABEL}\" }}" "$img" 2>&1)"; then
        log_error "cannot inspect the image of the running API container ${cid}: ${rev}"
        return 2
    fi
    if [[ -z "$rev" ]]; then
        log_error "the running API image's revision is unknown: it has no ${REVISION_LABEL} label."
        return 3
    fi
    printf '%s\n' "$rev"
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
    cid="$(running_container_id)" || return 1
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
