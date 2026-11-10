# =============================================================================
# FraudAI Agent -- Production Dockerfile (multi-stage, reproducible)
#
# Build:  docker build -t fraudai-agent:latest .
# Run:    docker run -p 8000:8000 --env-file .env fraudai-agent:latest
#
# Reproducibility: external base images are pinned by multi-platform index
# digest (linux/amd64 + linux/arm64) and dependencies are installed from the
# committed uv.lock with `uv sync --frozen`.
# NOTE: sentence-transformers pulls torch, so the image is far above 500 MB.
# Reference: ADR-006
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder -- install locked dependencies + the project (non-editable)
# ---------------------------------------------------------------------------
FROM python:3.11.18-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39 AS builder

WORKDIR /app

# System deps for building Python packages that lack wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# uv pinned by digest (0.12.19)
COPY --from=ghcr.io/astral-sh/uv:0.12.19@sha256:04d046b13e60d6bcec73cbc5e1cad25d680dea90c8573340950a0ac2d1aef424 /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

# 1) Third-party deps only (cached layer; depends solely on the lock files)
COPY pyproject.toml uv.lock ./
# extra openai: the OpenAI-compatible provider (ChatOpenAI) is the D2 route to Ollama
RUN uv sync --frozen --no-dev --no-install-project --no-editable --extra openai

# 2) The project itself, installed into the venv (NOT editable: no link to /app/src)
COPY src/ ./src/
RUN uv sync --frozen --no-dev --no-editable --extra openai && \
    rm -rf /root/.cache

# ---------------------------------------------------------------------------
# Stage 2: Runtime -- minimal image with only the virtualenv
# ---------------------------------------------------------------------------
FROM python:3.11.18-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39 AS runtime

WORKDIR /app

# Runtime system deps (curl for Docker healthcheck)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    # Base-image build tooling (setuptools/wheel + vendored jaraco.context) is not
    # used at runtime (app runs from /opt/venv) and carries fixable HIGH CVEs.
    pip uninstall -y setuptools wheel

# Locked virtualenv (contains the installed fraudai package)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# Non-root user for security. /app/data (SQLite) and the Hugging Face cache are created
# here, owned by that user, so named volumes mounted on them are writable by the
# effective user even though the root filesystem is read-only in Compose.
RUN useradd -m -s /bin/bash -u 1000 fraudai && \
    mkdir -p /app/data /home/fraudai/.cache/huggingface && \
    chown -R fraudai:fraudai /app/data /home/fraudai/.cache
ENV HF_HOME=/home/fraudai/.cache/huggingface

# Build identity, baked into the image (scripts/deploy.sh passes the real commit SHA
# resolved from git; there is no default so a build without it is visibly unlabelled).
# The image label is the source of truth; BUILD_SHA (the name F5c's API reads) carries
# the same value, so /api/v1/readyz can report the revision the container really runs.
# Compose must never override BUILD_SHA: scripts/deploy.sh compares the served value
# with the image label and refuses a mismatch.
ARG GIT_SHA=unset
ARG BUILD_DATE=unset
LABEL org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.title="fraudai-agent"
ENV BUILD_SHA="${GIT_SHA}"
USER fraudai

EXPOSE 8000

# Liveness only (/livez calls no dependency): the container is restarted when the process
# is dead, never because Qdrant or the LLM are down. Readiness is /api/v1/readyz,
# checked by the deploy/rollback scripts. start-period covers loading the embedding
# model on a cold Hugging Face cache.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/livez >/dev/null || exit 1

# Production server: uvicorn with app factory
# Workers controlled via FRAUDAI_WORKERS env var (default 1 in CMD, override in compose)
CMD ["python", "-m", "uvicorn", "fraudai.api.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000"]
