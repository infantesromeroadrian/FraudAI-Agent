#!/usr/bin/env bash
# =============================================================================
# FraudAI Agent -- roll the API back to an image that already exists locally.
#
# Usage: ./scripts/rollback.sh <sha|--previous>
#        ./scripts/rollback.sh --list
#
# Rolling back never builds, re-tags or deletes anything: it points the API
# container at fraudai-agent:<sha> (built earlier by deploy.sh) and then verifies
# that the container runs that image and that the API serves that same SHA
# (build_sha on /api/v1/readyz); that check cannot be skipped.
# =============================================================================
set -euo pipefail

# shellcheck source=scripts/lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

TARGET=""
LIST=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list) LIST=true; shift ;;
        --previous) TARGET="--previous"; shift ;;
        --help|-h) sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*) die "Unknown argument: $1" ;;
        *) TARGET="$1"; shift ;;
    esac
done

command -v docker >/dev/null || die "docker is not installed."
[[ -f "$ENV_FILE" ]] || die "Env file not found: ${ENV_FILE}."

if [[ "$LIST" == "true" ]]; then
    echo "Local ${API_REPO} images by commit (revision label):"
    docker image ls "$API_REPO" --format '{{.Tag}}' | grep -E '^[0-9a-f]{40}$' | while read -r tag; do
        printf '  %s  %s  %s\n' "$tag" "$(image_id "${API_REPO}:${tag}" | cut -c8-19)" \
            "$(docker image inspect --format '{{.Created}}' "${API_REPO}:${tag}")"
    done
    exit 0
fi

[[ -n "$TARGET" ]] || die "Missing target: pass a commit SHA, --previous or --list."

CURRENT_SHA="$(running_sha)" || die "Cannot determine the version that is running now; refusing to guess."
if [[ "$TARGET" == "--previous" ]]; then
    [[ -n "$CURRENT_SHA" ]] || die "Nothing is running, so there is no \"previous\" to go back from: pass an explicit SHA (see --list)."
    TARGET="$(previous_recorded_sha "$CURRENT_SHA")"
    [[ -n "$TARGET" ]] || die "No previous deployment recorded in ${STATE_LOG}."
fi

if ! is_full_sha "$TARGET"; then
    # allow an unambiguous prefix of a locally built tag
    matches="$(docker image ls "$API_REPO" --format '{{.Tag}}' | grep -E "^${TARGET}[0-9a-f]*$" | grep -E '^[0-9a-f]{40}$' || true)"
    [[ "$(echo "$matches" | grep -c .)" == "1" ]] || die "'${TARGET}' does not identify exactly one local image (use --list)."
    TARGET="$matches"
fi
SHA="$TARGET"
IMAGE="${API_REPO}:${SHA}"

docker image inspect "$IMAGE" >/dev/null 2>&1 || die "Image ${IMAGE} not found locally (see --list)."
[[ "$(image_revision "$IMAGE")" == "$SHA" ]] ||
    die "${IMAGE} does not carry revision ${SHA} in its label; refusing to roll back to a mislabelled image."

log_info "Rolling back: ${CURRENT_SHA:-<not running>} -> ${SHA}"
export FRAUDAI_IMAGE="$IMAGE"
# The frontend image is only touched when one was built for that commit.
SERVICES=(fraudai)
if docker image inspect "${FRONTEND_REPO}:${SHA}" >/dev/null 2>&1; then
    export FRONTEND_IMAGE="${FRONTEND_REPO}:${SHA}"
    SERVICES+=(frontend)
else
    export FRONTEND_IMAGE="${FRONTEND_IMAGE:-${FRONTEND_REPO}:none}"
    log_warn "No ${FRONTEND_REPO}:${SHA}; the frontend container is left as it is."
fi

compose up -d --no-deps "${SERVICES[@]}"
wait_for_ready 90 || { compose logs --tail 40 fraudai >&2; die "API did not become ready after rollback."; }
verify_running_sha "$SHA" || die "Rollback verification failed."

record_state rollback "$SHA" "${CURRENT_SHA:-}" "$(image_id "$IMAGE")"
log_info "Rollback complete: serving ${SHA}. The image of ${CURRENT_SHA:-the replaced deployment} is untouched."
