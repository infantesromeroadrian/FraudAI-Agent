#!/usr/bin/env bash
# =============================================================================
# FraudAI Agent -- deploy a REAL git commit to the local Compose stack.
#
# Usage: ./scripts/deploy.sh [--ref <rev>] [--no-frontend] [--with-sandbox]
#
#   --ref <rev>              commit to deploy (default HEAD); resolved to a full SHA
#   --no-frontend            do not build/start the frontend (e.g. before F5b lands)
#   --with-sandbox           also build fraudai-sandbox:<sha> for an API run on a Docker
#                            host (D4-B); the Compose API itself never runs sandboxes
#
# Guarantees (F5a.1):
#   * Images are named by the full commit SHA at BUILD time (-t fraudai-agent:<sha>);
#     nothing is ever re-tagged, overwritten or deleted, so the previous deployment's
#     image is still there for rollback.
#   * The build context is `git archive <sha>`, never the working tree: the image
#     content is exactly that commit, whatever is uncommitted in the checkout.
#   * An existing tag is reused only if its revision label equals the SHA.
#   * After `up`, the running container is checked against the tag (image ID, label)
#     and against the build_sha served by the API itself on /api/v1/readyz. That last
#     check is mandatory: a deployment whose served SHA is missing or different fails.
# =============================================================================
set -euo pipefail

# shellcheck source=scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

REF="HEAD"
WITH_FRONTEND=true
WITH_SANDBOX=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref) REF="${2:?--ref needs a revision}"; shift 2 ;;
        --no-frontend) WITH_FRONTEND=false; shift ;;
        --with-sandbox) WITH_SANDBOX=true; shift ;;
        --help|-h) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

command -v docker >/dev/null || die "docker is not installed."
docker compose version >/dev/null 2>&1 || die "docker compose plugin is not available."
command -v git >/dev/null || die "git is required to resolve the commit to deploy."
[[ -f "$ENV_FILE" ]] || die "Env file not found: ${ENV_FILE} (copy .env.example to .env and fill it in)."

cd "$PROJECT_DIR"
SHA="$(git rev-parse --verify "${REF}^{commit}")" || die "Cannot resolve '${REF}' to a commit."
is_full_sha "$SHA" || die "Unexpected commit id '${SHA}'."
COMMIT_DATE="$(git show -s --format=%cI "$SHA")"
if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
    log_warn "Working tree has uncommitted changes; they are NOT part of the image (built from commit ${SHA})."
fi

API_IMAGE="${API_REPO}:${SHA}"
FRONTEND_IMAGE="${FRONTEND_REPO}:${SHA}"

# --- build helpers ----------------------------------------------------------
# build_once <image> <description> <build command...>
# Reuses an existing tag only when its revision label matches; never overwrites.
build_once() {
    local image="$1" what="$2"
    shift 2
    if docker image inspect "$image" >/dev/null 2>&1; then
        [[ "$(image_revision "$image")" == "$SHA" ]] ||
            die "${image} exists but its revision label is '$(image_revision "$image")'; refusing to overwrite a mutable/foreign tag."
        log_info "${what}: ${image} already built for this commit; reusing it (no rebuild, no re-tag)."
        return 0
    fi
    log_info "Building ${what}: ${image}"
    "$@"
    [[ "$(image_revision "$image")" == "$SHA" ]] ||
        die "${image} was built but does not carry revision ${SHA}."
}

build_api() {
    git archive --format=tar "$SHA" |
        docker build --build-arg "GIT_SHA=${SHA}" --build-arg "BUILD_DATE=${COMMIT_DATE}" \
            -t "$API_IMAGE" -f Dockerfile -
}

build_frontend() {
    git cat-file -e "${SHA}:frontend/Dockerfile" 2>/dev/null ||
        die "frontend/Dockerfile does not exist in ${SHA} (delivered by F5b). Use --no-frontend until it is integrated."
    local args=() arg
    # NEXT_PUBLIC_API_URL is baked into the browser bundle: it must be the URL the BROWSER
    # uses to reach the published API port (F5b handoff), never a Compose-internal name.
    for arg in ${FRONTEND_BUILD_ARGS:-NEXT_PUBLIC_API_URL=http://localhost:${FRAUDAI_PORT}/api/v1}; do args+=(--build-arg "$arg"); done
    git archive --format=tar "${SHA}:frontend" |
        docker build "${args[@]+"${args[@]}"}" --label "${REVISION_LABEL}=${SHA}" \
            -t "$FRONTEND_IMAGE" -f Dockerfile -
}

build_sandbox() {
    git archive --format=tar "${SHA}:src/fraudai/tools/sandbox_image" |
        docker build --label "${REVISION_LABEL}=${SHA}" -t "fraudai-sandbox:${SHA}" -f Dockerfile -
}

PREVIOUS_SHA="$(running_sha || true)"

build_once "$API_IMAGE" "FraudAI API image" build_api
if [[ "$WITH_FRONTEND" == "true" ]]; then
    build_once "$FRONTEND_IMAGE" "frontend image" build_frontend
fi
if [[ "$WITH_SANDBOX" == "true" ]]; then
    build_once "fraudai-sandbox:${SHA}" "sandbox image" build_sandbox
fi

# --- start --------------------------------------------------------------------
export FRAUDAI_IMAGE="$API_IMAGE"
export FRONTEND_IMAGE
SERVICES=(qdrant fraudai prometheus grafana)
if [[ "$WITH_FRONTEND" == "true" ]]; then SERVICES+=(frontend); fi

log_info "Starting services: ${SERVICES[*]}"
compose up -d "${SERVICES[@]}"

log_info "Waiting for the API to be ready (/api/v1/readyz = 200)..."
wait_for_ready 90 || { compose logs --tail 40 fraudai >&2; die "API did not become ready."; }

verify_running_sha "$SHA" || die "Deployment verification failed; the previous image is untouched (${PREVIOUS_SHA:-none})."

# --- Ollama reachability (D2): from inside the container, no model is pulled ----
if compose exec -T fraudai python -c "
import os, urllib.request
host = os.environ.get('OLLAMA_HOST', '').rstrip('/')
urllib.request.urlopen(host + '/api/version', timeout=5).read()
" >/dev/null 2>&1; then
    log_info "Ollama daemon reachable from the API container."
else
    log_warn "Ollama is NOT reachable from the API container (is the host daemon running?)."
fi

record_state deploy "$SHA" "${PREVIOUS_SHA:-}" "$(image_id "$API_IMAGE")"

echo ""
echo "============================================"
echo "  Deployed commit : ${SHA}"
echo "  Image           : ${API_IMAGE} ($(image_id "$API_IMAGE" | cut -c8-19))"
echo "  Previous        : ${PREVIOUS_SHA:-none}"
echo "  API             : http://127.0.0.1:${FRAUDAI_PORT}/api/v1/readyz"
echo "  Frontend        : http://127.0.0.1:${FRONTEND_PORT:-3000}"
echo "  Grafana         : http://127.0.0.1:${GRAFANA_PORT:-3001}"
echo "  Prometheus      : http://127.0.0.1:${PROMETHEUS_PORT:-9090}"
echo "  Rollback        : ./scripts/rollback.sh --previous   (or <full sha>)"
echo "============================================"
