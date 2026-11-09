FROM python:3.11.18-slim-bookworm@sha256:a83ba18aef0f865b1a4a611c2cf460fe446f2795673bbb9fe43be570439e3b39

# Analysis dependencies used by Harvey, Jessica, Mike, and Rachel.
# Exact versions (aligned with the project uv.lock where the package is locked;
# matplotlib is not part of the lock, pinned to the PyPI release at 2026-11-09).
# Nothing is installed at runtime -- the image is immutable.
RUN pip install --no-cache-dir \
    pandas==3.1.1 \
    numpy==2.4.6 \
    scikit-learn==1.9.1 \
    scipy==1.17.1 \
    networkx==3.6.1 \
    matplotlib==3.11.2 \
    # Base-image build tooling is unused at runtime and carries fixable HIGH CVEs.
    && pip uninstall -y setuptools wheel \
    && rm -rf /root/.cache

# Security: non-root user with minimal shell
RUN useradd -m -s /bin/bash sandbox
USER sandbox
WORKDIR /workspace

# Pre-create output directory so the sandbox user can write to it
RUN mkdir -p /workspace/output

# No ENTRYPOINT -- commands are passed at runtime by SandboxEngine
