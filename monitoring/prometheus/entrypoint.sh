#!/bin/sh
# Static entrypoint (no secrets inside): hands Prometheus its scrape credential as a file
# in a private tmpfs, then starts Prometheus WITHOUT the credential in its environment.
#
# The value arrives as the process variable METRICS_SCRAPE_TOKEN (Compose takes it from the
# local .env, the only persistent place for secrets). Prometheus can only read a bearer
# token from a file, and Compose `secrets` cannot be used with the read_only API service,
# so the file is created here, at start, in /run/prometheus-credentials (a tmpfs owned by
# the Prometheus user, mode 0700, mounted by Compose). Nothing is written to the image, to
# a volume or to the host, and the value is never printed or put on a command line.
set -eu

dir=/run/prometheus-credentials
file="${dir}/metrics-token"

if [ -z "${METRICS_SCRAPE_TOKEN:-}" ] || [ "${#METRICS_SCRAPE_TOKEN}" -lt 32 ]; then
    echo "METRICS_SCRAPE_TOKEN is missing or shorter than 32 characters (see .env.example)" >&2
    exit 64
fi

umask 077
printf '%s' "$METRICS_SCRAPE_TOKEN" > "$file"
unset METRICS_SCRAPE_TOKEN

exec /bin/prometheus "$@"
