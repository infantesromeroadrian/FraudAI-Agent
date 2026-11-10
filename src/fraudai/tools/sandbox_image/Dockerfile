FROM python:3.11.18-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39

# Analysis dependencies used by Harvey, Jessica, Mike, and Rachel.
# requirements.lock is the full transitive closure (exact versions + sha256) of
# pandas, numpy, scikit-learn, scipy, networkx and matplotlib; pip refuses any
# file whose hash is not listed. Nothing is installed at runtime -- the image is immutable.
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes --no-deps -r /tmp/requirements.lock \
    # Base-image build tooling is unused at runtime and carries fixable HIGH CVEs.
    && pip uninstall -y setuptools wheel \
    && rm -rf /root/.cache /tmp/requirements.lock

# Headless plotting and no bytecode writes: the engine runs this image with a
# read-only root filesystem, so nothing but /tmp and /workspace/output is writable.
ENV MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1

# Security: non-root user with minimal shell
RUN useradd -m -s /bin/bash sandbox
USER sandbox
WORKDIR /workspace

# Pre-create output directory so the sandbox user can write to it
RUN mkdir -p /workspace/output

# No ENTRYPOINT -- commands are passed at runtime by SandboxEngine
