# GANDALF KRMHD - Metal Backend (Apple Silicon)
# Container for running KRMHD simulations on Apple Silicon GPUs
# Requires: macOS 13.0+ (Ventura), M1/M2/M3 Mac

FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/anjor/gandalf"
LABEL org.opencontainers.image.description="KRMHD spectral solver for magnetized plasma turbulence (Metal backend for Apple Silicon)"
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

# Install GANDALF with Metal GPU support
RUN uv pip install --system -e ".[metal]"

# Verify installation
RUN python -c 'import krmhd; print("KRMHD imported successfully")' && \
    python -c 'import jax; print(f"JAX version: {jax.__version__}"); print(f"Devices: {jax.devices()}")'

# Create output directory
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ENABLE_PJRT_COMPATIBILITY=1

# Default command: run minimal forcing example
CMD ["python", "examples/forcing_minimal.py"]

# Usage examples:
#
# Build (on Apple Silicon Mac):
#   docker build -t gandalf-krmhd:metal -f Dockerfile.metal .
#
# Run minimal example:
#   docker run gandalf-krmhd:metal
#
# Run custom script:
#   docker run gandalf-krmhd:metal python examples/decaying_turbulence.py
#
# Interactive session:
#   docker run -it gandalf-krmhd:metal bash
#
# Mount local directory for outputs:
#   docker run -v $(pwd)/output:/app/output gandalf-krmhd:metal python examples/decaying_turbulence.py
#
# Verify Metal GPU is detected:
#   docker run gandalf-krmhd:metal python -c 'import jax; print(jax.devices())'
#
# Note: Metal backend requires running on Apple Silicon hardware.
# The container will fall back to CPU if Metal is not available.
