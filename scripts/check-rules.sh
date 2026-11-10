#!/usr/bin/env bash
# Validate monitoring: promtool (from the pinned Prometheus image) + "no unknown metric".
# Usage: ./scripts/check-rules.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROM_IMAGE="prom/prometheus:v2.53.0@sha256:075b1ba2c4ebb04bc3a6ab86c06ec8d8099f8fda1c96ef6d104d9bb1def1d8bc"

echo "== promtool check rules"
docker run --rm --entrypoint promtool \
    -v "${PROJECT_DIR}/monitoring/alerts:/rules:ro" "$PROM_IMAGE" check rules /rules/rules.yml

echo "== promtool check config (rules + scrape config, with a throw-away credential file)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
printf 'placeholder-not-a-credential' > "${scratch}/metrics-token"
docker run --rm --entrypoint promtool \
    -v "${PROJECT_DIR}/monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
    -v "${PROJECT_DIR}/monitoring/alerts:/etc/prometheus/rules:ro" \
    -v "${scratch}/metrics-token:/run/prometheus-credentials/metrics-token:ro" \
    "$PROM_IMAGE" check config /etc/prometheus/prometheus.yml

echo "== every metric used by rules and dashboards is exposed"
python3 "${PROJECT_DIR}/scripts/check_rule_metrics.py" \
    "${PROJECT_DIR}/monitoring/alerts/rules.yml" \
    "${PROJECT_DIR}"/monitoring/grafana/dashboards/*.json
