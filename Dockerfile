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
RUN uv sync --frozen --no-dev --no-install-project --no-editable

# 2) The project itself, installed into the venv (NOT editable: no link to /app/src)
COPY src/ ./src/
RUN uv sync --frozen --no-dev --no-editable && \
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

# Non-root user for security
RUN useradd -m -s /bin/bash -u 1000 fraudai
USER fraudai

EXPOSE 8000

# Healthcheck against the FastAPI /api/v1/health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Production server: uvicorn with app factory
# Workers controlled via FRAUDAI_WORKERS env var (default 1 in CMD, override in compose)
CMD ["python", "-m", "uvicorn", "fraudai.api.app:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000"]
