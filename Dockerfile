# GANDALF KRMHD - CPU Backend
# Minimal container for running KRMHD simulations on CPU
# Suitable for: Testing, small simulations, HPC clusters without GPU

FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/anjor/gandalf"
LABEL org.opencontainers.image.description="KRMHD spectral solver for magnetized plasma turbulence (CPU backend)"
LABEL org.opencontainers.image.licenses="MIT"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY examples/ ./examples/
COPY tests/ ./tests/

# Install uv for fast package management
RUN pip install --no-cache-dir uv

# Install GANDALF and dependencies (CPU-only JAX)
RUN uv pip install --system -e .

# Verify installation
RUN python -c 'import krmhd; print("KRMHD imported successfully")' && \
    python -c 'import jax; print(f"JAX version: {jax.__version__}"); print(f"Devices: {jax.devices()}")'

# Create output directory
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV JAX_PLATFORMS=cpu

# Default command: run minimal forcing example
CMD ["python", "examples/forcing_minimal.py"]

# Usage examples:
#
# Build:
#   docker build -t gandalf-krmhd .
#
# Run minimal example:
#   docker run gandalf-krmhd
#
# Run custom script:
#   docker run gandalf-krmhd python examples/decaying_turbulence.py
#
# Interactive session:
#   docker run -it gandalf-krmhd bash
#
# Mount local directory for outputs:
#   docker run -v $(pwd)/output:/app/output gandalf-krmhd python examples/decaying_turbulence.py
#
# For HPC (convert to Singularity):
#   singularity pull docker://ghcr.io/anjor/gandalf:latest
#   singularity exec gandalf_latest.sif python examples/decaying_turbulence.py
