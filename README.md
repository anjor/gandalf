# KRMHD: Kinetic Reduced MHD Spectral Solver

A modern Python implementation of a Kinetic Reduced Magnetohydrodynamics (KRMHD) solver using spectral methods for studying turbulent magnetized plasmas in astrophysical environments.

## Overview

KRMHD is a spectral code designed to simulate turbulence in weakly collisional magnetized plasmas, incorporating kinetic effects such as Landau damping via Hermite velocity-space representation. This is a modern rewrite of the legacy [GANDALF](https://github.com/anjor/gandalf) Fortran+CUDA implementation, leveraging JAX for automatic differentiation and GPU acceleration via Apple's Metal backend.

## Physics Model

The KRMHD model splits plasma dynamics into two coupled sectors:

### Active (Alfvenic) Sector
- **Stream function phi**: Describes perpendicular plasma flow
- **Parallel vector potential A_parallel**: Represents perpendicular magnetic field fluctuations
- These fields drive the turbulent energy cascade

### Passive (Slow Mode) Sector
- **Parallel magnetic field delta_B_parallel**: Field-aligned magnetic fluctuations
- **Electron density/pressure n_e, p_e**: Thermodynamic perturbations
- Advected by Alfvenic turbulence without back-reaction

Key physical processes include:
- Poisson bracket nonlinearities: `{f,g} = z_hat · (grad_f x grad_g)`
- Parallel electron kinetic effects (Landau damping)
- Spectral energy transfer across scales
- Selective decay and energy partition

## Installation

### Prerequisites
- Python 3.10 or higher (Python 3.11 recommended for containers and CI)
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- **Optional**: GPU acceleration (Metal/CUDA) for better performance

**Note on Python versions:** While the package supports Python 3.10-3.12, Docker containers and CI workflows use Python 3.11 as the tested baseline. Local installations can use any supported version.

### System Requirements

**Minimum (CPU-only, 32³ resolution):**
- RAM: 4 GB
- Disk space: 1 GB (code + outputs)
- CPU: Any modern multi-core processor

**Recommended (GPU, 128³ resolution):**
- RAM: 8-16 GB
- Disk space: 5-10 GB (for output files)
- GPU options:
  - **Apple Silicon**: M1/M2/M3 with 8+ GB unified memory
  - **NVIDIA**: GPU with 8+ GB VRAM (CUDA 11.8+ or 12.x)
  - **Driver requirements** (NVIDIA): CUDA 12.x drivers (compatible with driver 525.60.13+)

**Production (256³+ resolution):**
- RAM: 32-64 GB
- Disk space: 50-100 GB (for long simulations)
- GPU: 16+ GB VRAM or HPC cluster resources

**Notes:**
- Memory scales as O(N³ × M) where N is grid size, M is number of Hermite moments
- Typical field memory: ~270 MB per complex field at 256³ resolution
- Output files can be large: ~500 MB per checkpoint for 128³ simulations

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/anjor/gandalf.git
cd gandalf

# Install core dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Verify JAX installation
python -c 'import jax; print(jax.devices())'
```

## Platform Support

KRMHD supports multiple compute backends through JAX. Choose the appropriate installation method for your hardware:

### macOS with Apple Silicon (Metal)

For M1/M2/M3 Macs with GPU acceleration:

```bash
# Install with Metal GPU support
uv sync --extra metal

# Verify Metal device
python -c 'import jax; print(jax.devices())'
# Should show: [METAL(id=0)]
```

**Notes:**
- `jax-metal` is experimental but functional
- GPU operations are transparent (no code changes needed)
- For compatibility issues, set: `export ENABLE_PJRT_COMPATIBILITY=1`
- See [Apple's JAX documentation](https://developer.apple.com/metal/jax/)

### Linux with NVIDIA GPU (CUDA)

For CUDA-enabled systems (HPC clusters, workstations):

```bash
# Install JAX with CUDA support
uv pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Verify CUDA device
python -c 'import jax; print(jax.devices())'
# Should show: [cuda:0]
```

**Note:** Replace `cuda12` with your CUDA version (11.8, 12.0, etc.). See [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for details.

### CPU-only (All platforms)

For development or testing without GPU:

```bash
# Standard installation (uses CPU)
uv sync

# JAX will use CPU backend automatically
python -c 'import jax; print(jax.devices())'
# Should show: [CpuDevice(id=0)]
```

**Performance note:** CPU mode works but is significantly slower (~10-50x) than GPU-accelerated runs. Suitable for 128² resolution or smaller.

### Manual Installation

For systems without uv:

```bash
python -m venv venv
source venv/bin/activate
pip install -e .

# For Metal GPU support on macOS:
pip install -e ".[metal]"
```

### Using Docker Containers

Docker containers provide a pre-configured environment with all dependencies installed. This is especially useful for:
- **HPC clusters** requiring Singularity/Apptainer containers
- **Reproducibility** across different systems
- **Quick testing** without local installation
- **Avoiding** JAX/GPU driver conflicts

#### Quick Start with Docker

```bash
# Pull and run the latest CPU version
docker run -it ghcr.io/anjor/gandalf:latest python examples/forcing_minimal.py

# Or run interactively
docker run -it ghcr.io/anjor/gandalf:latest bash
```

#### Available Images

Pre-built images are automatically published to GitHub Container Registry:

| Image | Description | Use Case |
|-------|-------------|----------|
| `ghcr.io/anjor/gandalf:latest` | CPU backend | Testing, HPC without GPU |
| `ghcr.io/anjor/gandalf:v0.1.0` | Specific version (CPU) | Reproducibility |
| `ghcr.io/anjor/gandalf:latest-cuda` | NVIDIA GPU (CUDA 12) | Production GPU runs |

**Note**: Metal (Apple Silicon) images must be built locally - see below.

#### Running with GPU Support

**NVIDIA GPUs** (requires [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)):

```bash
# Run with GPU access
docker run --gpus all ghcr.io/anjor/gandalf:latest-cuda python examples/decaying_turbulence.py

# Verify GPU is detected
docker run --gpus all ghcr.io/anjor/gandalf:latest-cuda python -c 'import jax; print(jax.devices())'
```

**Apple Silicon** (build locally):

```bash
# Clone repository first
git clone https://github.com/anjor/gandalf.git
cd gandalf

# Build Metal-enabled image
docker build -t gandalf-krmhd:metal -f Dockerfile.metal .

# Run with Metal support
docker run -it gandalf-krmhd:metal python examples/decaying_turbulence.py
```

#### Mounting Local Directories

To save outputs to your local machine:

```bash
# Create output directory
mkdir -p output

# Run with volume mount
docker run -v $(pwd)/output:/app/output ghcr.io/anjor/gandalf:latest \
  python examples/decaying_turbulence.py

# Outputs will appear in ./output/
```

#### Using on HPC Clusters (Singularity/Apptainer)

Most HPC systems use Singularity (now Apptainer) instead of Docker:

```bash
# Convert Docker image to Singularity
singularity pull gandalf_latest.sif docker://ghcr.io/anjor/gandalf:latest

# Run simulation
singularity exec gandalf_latest.sif python examples/decaying_turbulence.py

# With GPU support (CUDA)
singularity pull gandalf_cuda.sif docker://ghcr.io/anjor/gandalf:latest-cuda
singularity exec --nv gandalf_cuda.sif python examples/decaying_turbulence.py
```

**HPC batch script example** (SLURM):

```bash
#!/bin/bash
#SBATCH --job-name=gandalf
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

module load singularity

# Run simulation
singularity exec --nv gandalf_cuda.sif \
  python /app/examples/alfvenic_cascade_benchmark.py --resolution 128
```

#### Building Custom Images

For development or customization:

```bash
# Clone repository
git clone https://github.com/anjor/gandalf.git
cd gandalf

# Build CPU version
docker build -t gandalf-krmhd:dev .

# Build CUDA version
docker build -t gandalf-krmhd:dev-cuda -f Dockerfile.cuda .

# Build Metal version (macOS only)
docker build -t gandalf-krmhd:dev-metal -f Dockerfile.metal .

# Run your custom build
docker run -it gandalf-krmhd:dev bash
```

#### Container Best Practices

**For reproducibility**:
- Use specific version tags: `ghcr.io/anjor/gandalf:v0.1.0`
- Include container version in paper methods section
- Archive container with data: `docker save` or `singularity build --sandbox`

**For performance**:
- Use GPU images (`-cuda`) on GPU-enabled systems
- Mount data directories as read-only where possible: `-v $(pwd)/data:/data:ro`
- Use `/app/output` as working directory for temporary files

**For debugging**:
- Interactive shell: `docker run -it <image> bash`
- Check JAX devices: `docker run <image> python -c 'import jax; print(jax.devices())'`
- Inspect installed versions: `docker run <image> pip list`

## Installation Troubleshooting

### Verifying Your Installation

After installation, always verify that JAX is correctly configured:

```bash
python -c 'import jax; print(f"JAX version: {jax.__version__}"); print(f"Devices: {jax.devices()}")'
```

**Expected output:**
- **Metal (macOS)**: `[METAL(id=0)]`
- **CUDA (Linux/GPU)**: `[cuda:0]` or `[gpu:0]`
- **CPU (all platforms)**: `[CpuDevice(id=0)]`

### Common Issues and Solutions

#### Issue: "No module named 'jax'"

**Cause**: JAX not installed or virtual environment not activated

**Solution**:
```bash
# If using uv
uv sync
source .venv/bin/activate

# If using pip
pip install -e .
```

#### Issue: JAX Metal fails silently on macOS

**Symptoms**: Installation succeeds but `jax.devices()` shows `[CpuDevice(id=0)]` instead of `[METAL(id=0)]`

**Solution 1**: Enable PJRT compatibility
```bash
export ENABLE_PJRT_COMPATIBILITY=1
python -c 'import jax; print(jax.devices())'
```

**Solution 2**: Reinstall jax-metal
```bash
uv pip uninstall jax-metal
uv pip install --upgrade jax-metal
```

**Solution 3**: Check macOS version
- Metal backend requires macOS 13.0+ (Ventura or later)
- For older macOS versions, use CPU backend

#### Issue: CUDA version mismatch

**Symptoms**: `RuntimeError: CUDA version mismatch` or `CUDNN_STATUS_NOT_INITIALIZED`

**Solution**: Match JAX CUDA version to your system CUDA:
```bash
# Check your CUDA version
nvcc --version
# or
nvidia-smi

# Install matching JAX version (e.g., for CUDA 12.x)
uv pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# For CUDA 11.8
uv pip install --upgrade "jax[cuda11_cudnn86]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

See [JAX GPU installation guide](https://jax.readthedocs.io/en/latest/installation.html#nvidia-gpu) for all CUDA versions.

#### Issue: "RuntimeError: Unable to initialize backend 'cuda'"

**Cause**: CUDA libraries not found or incompatible driver

**Solution**:
```bash
# Check CUDA libraries are accessible
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify CUDA installation
python -c 'import jaxlib; print(jaxlib.xla_extension_version)'

# If still failing, try CPU-only mode
python -c 'import jax; jax.config.update("jax_platform_name", "cpu"); print(jax.devices())'
```

#### Issue: HPC cluster module conflicts

**Symptoms**: Import errors, version conflicts, or permission errors on HPC systems

**Solution**: Use module system and local installation
```bash
# Load required modules (adjust for your cluster)
module load python/3.10
module load cuda/12.0  # if using GPU

# Create local environment
python -m venv ~/.venv/gandalf
source ~/.venv/gandalf/bin/activate
pip install -e /path/to/gandalf

# For systems requiring Singularity/Apptainer
# See Docker section for container usage
```

#### Issue: "ImportError: cannot import name 'sparse' from 'jax.experimental'"

**Cause**: JAX version too old

**Solution**:
```bash
uv pip install --upgrade "jax>=0.4.20"
```

#### Issue: Out of memory errors during simulation

**Symptoms**: `RuntimeError: RESOURCE_EXHAUSTED: Out of memory`

**Solutions**:
1. **Reduce resolution**: Start with 32³ or 64³ before scaling to 128³+
2. **Use CPU for memory-intensive runs**: `jax.config.update("jax_platform_name", "cpu")`
3. **Monitor memory**:
   ```python
   from jax.lib import xla_bridge
   print(f"Backend: {xla_bridge.get_backend().platform}")
   ```
4. **Clear JAX cache**: `rm -rf ~/.cache/jax_cache/`

#### Issue: Slow performance even with GPU

**Symptoms**: Simulation takes much longer than expected

**Diagnostics**:
```python
import jax
import time

# Verify GPU is actually being used
print(f"Devices: {jax.devices()}")

# Test compilation overhead
@jax.jit
def test_fn(x):
    return x @ x.T

# First call includes compilation (slow)
x = jax.numpy.ones((1000, 1000))
start = time.time()
_ = test_fn(x).block_until_ready()
print(f"First call (with compilation): {time.time() - start:.3f}s")

# Second call is fast
start = time.time()
_ = test_fn(x).block_until_ready()
print(f"Second call (compiled): {time.time() - start:.3f}s")
```

**Solutions**:
- First timestep is always slow (JIT compilation)
- Use `.block_until_ready()` for accurate timing
- Check Metal GPU is active: `System Settings > General > About > Graphics`

#### Issue: "uv: command not found"

**Cause**: `uv` not installed or not in PATH

**Solution**:
```bash
# Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or use pip/pipx
pip install uv
# or
pipx install uv

# Add to PATH (macOS/Linux)
export PATH="$HOME/.cargo/bin:$PATH"

# Or use pip directly (no uv needed)
python -m venv venv
source venv/bin/activate
pip install -e .
```

### Platform-Specific Notes

#### macOS (Apple Silicon)

**Best practices:**
- Use Metal backend for 2-5x speedup over CPU
- Monitor Activity Monitor → GPU History during runs
- For large simulations (256³+), consider cloud GPU or HPC cluster

**Known limitations:**
- Metal backend is experimental (may have edge case bugs)
- Some operations fall back to CPU automatically
- Memory limited by unified RAM (8-64 GB typical)

#### Linux (NVIDIA GPU)

**Best practices:**
- Use `nvidia-smi` to monitor GPU usage during runs
- For multi-GPU systems, specify device: `CUDA_VISIBLE_DEVICES=0 python script.py`
- Load CUDA modules before Python: `module load cuda/12.0`

**Known limitations:**
- CUDA version must match JAX installation exactly
- Older GPUs (compute capability < 5.0) may not be supported

#### HPC Clusters

**Best practices:**
- Use container images (Docker/Singularity) for reproducibility
- Request GPU resources in batch scripts: `#SBATCH --gres=gpu:1`
- Install in user directory: `pip install --user -e .`
- Use `module spider jax` to check available pre-installed versions

**Common HPC commands:**
```bash
# Check available GPUs
sinfo -o "%20N %10c %10m %25f %10G"

# Interactive GPU session
salloc --gres=gpu:1 --time=01:00:00

# Load environment
module load python/3.10 cuda/12.0
source ~/.venv/gandalf/bin/activate
```

### Still Having Issues?

1. **Check GitHub Issues**: https://github.com/anjor/gandalf/issues
2. **JAX Installation Guide**: https://jax.readthedocs.io/en/latest/installation.html
3. **File a bug report**: Include output of:
   ```bash
   python -c 'import jax; print(jax.__version__); print(jax.devices())'
   uv --version  # or: pip --version
   python --version
   ```

## Project Structure

```
krmhd/
├── src/krmhd/          # Main package
│   ├── spectral.py     # ✅ FFT operations, derivatives, dealiasing (2D/3D)
│   ├── hermite.py      # ✅ Hermite basis (kinetic closures)
│   ├── physics.py      # ✅ KRMHD state, Poisson brackets, Elsasser RHS, Hermite RHS
│   ├── timestepping.py # ✅ GANDALF integrating factor + RK2 timestepper
│   ├── diagnostics.py  # ✅ Energy spectra (1D, k⊥, k∥), history, visualization
│   ├── forcing.py      # ✅ Gaussian white noise forcing for driven turbulence
│   ├── io.py          # HDF5 checkpointing (TODO)
│   └── validation.py   # Linear physics tests (TODO)
├── tests/             # Test suite (285 tests including performance benchmarks)
├── examples/          # Example scripts and tutorials
│   ├── decaying_turbulence.py   # ✅ Turbulent cascade with diagnostics
│   ├── driven_turbulence.py     # ✅ Forced turbulence with steady-state balance
│   ├── forcing_minimal.py       # ✅ Minimal forcing example (50 lines)
│   └── orszag_tang.py           # ✅ Orszag-Tang vortex benchmark
└── pyproject.toml     # Project metadata
```

## Quick Start: Running Examples

### Example 1: Minimal Forcing Example

The `examples/forcing_minimal.py` script shows basic forcing usage (~50 lines):

```bash
# Run the minimal example (takes ~2 seconds on M1 Pro)
uv run python examples/forcing_minimal.py
```

**What it does:**
1. Initializes a simple Alfvén wave
2. Applies forcing at large scales (k ~ 2-5)
3. Evolves 10 timesteps with forcing + dissipation
4. Prints energy and injection rate at each step

**Perfect for:** Learning the forcing API basics before diving into comprehensive examples.

### Example 2: Driven Turbulence Simulation

The `examples/driven_turbulence.py` script demonstrates forced turbulence (~317 lines):

```bash
# Run the driven turbulence example (takes ~20 seconds on M1 Pro)
uv run python examples/driven_turbulence.py
```

**What it does:**
1. Applies Gaussian white noise forcing at large scales (k ~ 2-5)
2. Evolves 200 timesteps to approach steady state
3. Tracks energy injection rate and energy balance
4. Computes energy spectra showing inertial range
5. Generates publication-ready plots

**Output files** (saved to `examples/output/`):
- `driven_energy_history.png` - Energy evolution and magnetic fraction
- `driven_energy_spectra.png` - E(k), E(k⊥), E(k∥) with forcing band highlighted

### Example 3: Decaying Turbulence Simulation

The `examples/decaying_turbulence.py` script demonstrates unforced turbulent decay:

```bash
# Run the example (takes ~1-2 minutes on M1 Pro)
uv run python examples/decaying_turbulence.py
```

**What it does:**
1. Initializes a turbulent k^(-5/3) spectrum (Kolmogorov)
2. Evolves 100 timesteps using GANDALF integrator
3. Tracks energy history E(t) showing exponential decay
4. Computes energy spectra E(k), E(k⊥), E(k∥)
5. Demonstrates selective decay (magnetic energy dominates)

**Output files** (saved to `examples/output/`):
- `energy_history.png` - Energy evolution showing selective decay
- `final_state.png` - 2D slices of φ and A∥ fields
- `energy_spectra.png` - Three-panel plot with 1D, perpendicular, and parallel spectra

### Example 4: Orszag-Tang Vortex

The `examples/orszag_tang.py` script runs a classic nonlinear MHD benchmark:

```bash
# Run the Orszag-Tang vortex (takes ~1-2 minutes on M1 Pro)
uv run python examples/orszag_tang.py
```

**What it does:**
1. Initializes incompressible Orszag-Tang vortex in RMHD formulation
2. Tests nonlinear dynamics, energy cascade, and current sheet formation
3. Evolves to t=1.0 (standard benchmark time)
4. Tracks energy partition and magnetic/kinetic energy ratio
5. Validates spectral method accuracy for complex flows

**Create publication-quality energy plots:**

```bash
# Run simulation and create energy evolution plot
uv run python scripts/plot_energy_evolution.py --run-simulation

# Or just plot existing data
uv run python scripts/plot_energy_evolution.py
```

This creates a plot showing Total, Kinetic, and Magnetic energy vs. time (normalized by Alfvén crossing time τ_A = L/v_A), matching the style of thesis Figure 2.1.

### Example 5: Custom Simulation with Forcing

```python
import jax
from krmhd import (
    SpectralGrid3D,
    initialize_random_spectrum,
    gandalf_step,
    compute_cfl_timestep,
    force_alfven_modes,
    compute_energy_injection_rate,
    EnergyHistory,
    energy_spectrum_perpendicular,
    plot_energy_history,
    plot_state,
)

# Initialize JAX random key for forcing
key = jax.random.PRNGKey(42)

# 1. Initialize grid and state
grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)
state = initialize_random_spectrum(
    grid,
    M=20,              # Number of Hermite moments
    alpha=5/3,         # Kolmogorov spectrum
    amplitude=1.0,
    k_min=1.0,
    k_max=10.0,
)

# 2. Set up energy tracking
history = EnergyHistory()

# 3. Time evolution loop with forcing
dt = 0.01
for step in range(100):
    # Record energy before forcing
    history.append(state)

    # Apply forcing at large scales
    state_before = state
    state, key = force_alfven_modes(
        state,
        amplitude=0.3,
        k_min=2.0,
        k_max=5.0,
        dt=dt,
        key=key
    )

    # Measure energy injection
    eps_inj = compute_energy_injection_rate(state_before, state, dt)

    # Advance one timestep (cascade + dissipation)
    state = gandalf_step(state, dt, eta=0.01, v_A=1.0)

    print(f"Step {step}: t={state.time:.3f}, E={history.E_total[-1]:.3e}, ε_inj={eps_inj:.2e}")

# 4. Analyze and visualize
k_perp, E_perp = energy_spectrum_perpendicular(state)
plot_energy_history(history, filename='energy.png')
plot_state(state, filename='final_state.png')
```

### Post-Processing: Analyzing Checkpoint Files

The `plot_checkpoint_spectrum.py` script analyzes saved checkpoints without rerunning simulations:

```bash
# Plot spectra from a checkpoint file
uv run python examples/plot_checkpoint_spectrum.py examples/output/checkpoints/checkpoint_t0300.0.h5

# Thesis-style formatting (for papers/presentations)
uv run python examples/plot_checkpoint_spectrum.py --thesis-style checkpoint.h5

# Custom output filename
uv run python examples/plot_checkpoint_spectrum.py --output fig_spectrum.png checkpoint.h5

# Interactive display
uv run python examples/plot_checkpoint_spectrum.py --show checkpoint.h5

# Batch process multiple checkpoints
for f in examples/output/checkpoints/checkpoint_*.h5; do
    uv run python examples/plot_checkpoint_spectrum.py "$f"
done
```

**What it does:**
- Loads checkpoint data (state, grid, metadata) from HDF5 file
- Computes perpendicular energy spectra: E_kin(k⊥) and E_mag(k⊥)
- Separates kinetic (from φ) and magnetic (from A∥) contributions
- Generates publication-quality plots with k⊥^(-5/3) reference lines
- Prints energy summary (total energy, magnetic fraction, time)

**Output formats:**
1. **Standard style** (default): Total spectrum + kinetic/magnetic comparison
2. **Thesis style** (`--thesis-style`): Side-by-side kinetic and magnetic panels with clean formatting

**Physics interpretation:**
- **Good k⊥^(-5/3) match** (n ~ 3-10): Healthy turbulent cascade in inertial range
- **High magnetic fraction** (f_mag > 0.5): Selective decay underway (normal)
- **Exponential cutoff at high-k**: Hyper-dissipation working correctly
- **Flat/rising spectrum at high-k**: Under-dissipated, increase η

The script automatically provides interpretation guidance in the console output.

### Generating Custom Plots

```python
import matplotlib.pyplot as plt
from krmhd import energy_spectrum_1d, plot_energy_spectrum

# Compute spectrum
k, E_k = energy_spectrum_1d(state, n_bins=32)

# Option 1: Use built-in plotting
plot_energy_spectrum(k, E_k, reference_slope=-5/3, filename='spectrum.png')

# Option 2: Custom matplotlib plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.loglog(k, E_k, 'b-', linewidth=2, label='E(k)')
ax.loglog(k, k**(-5/3), 'k--', label='k^(-5/3)')
ax.set_xlabel('|k|')
ax.set_ylabel('E(k)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('my_spectrum.png', dpi=150)
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_diagnostics.py

# Run with verbose output
uv run pytest -xvs

# Exclude slow tests (like benchmarks)
uv run pytest -m "not slow and not benchmark"
```

### Performance Benchmarks

The code includes comprehensive performance benchmarks for the Poisson bracket operator (the computational bottleneck):

```bash
# Run all benchmarks with detailed timing output
pytest tests/test_performance.py -v -s

# Run only benchmark tests
pytest -m benchmark -v -s

# Run directly without pytest
python tests/test_performance.py
```

**Baseline performance (M1 Pro, 128³ resolution):**
- ~28 ms per Poisson bracket call
- ~35 calls/second throughput
- ~5.3 hours for 100K timesteps (typical long simulation)

Benchmarks test 2D (64²-512²) and 3D (32³-256³) resolutions, measuring compilation time, runtime, throughput, and scaling behavior. See `tests/test_performance.py` for detailed baseline data.

**Note**: Benchmarks are currently excluded from CI (use `-m "not benchmark"` in regular test runs). CI integration for automated regression detection is planned as future work.

### Code Quality

```bash
# Format code
uv run ruff format

# Lint
uv run ruff check

# Type checking
uv run mypy src/krmhd
```

## Normalization Convention

**Box Size and Time:**
- **Standard**: Use Lx = Ly = Lz = 1.0 (unit box) for benchmarks
- **Alfvén time**: τ_A = Lz/v_A (parallel Alfvén crossing time)
- **Wavenumbers**: k = (2π/L) × n for mode number n

**Field Normalization:**
- **Mean field**: B₀ = 1/√(4π) ≈ 0.282 in code units
- **Alfvén velocity**: v_A = 1.0 (normalized)
- **Energy**: Total energy (volume integral), computed via Parseval's theorem

**Important for Benchmarks:**
- Orszag-Tang vortex uses Lx = Ly = Lz = 1.0 to match thesis Figure 2.1
- With Lx=1.0, fundamental wavenumber k=2π gives correct energy scale
- Using Lx=2π would make k=1, reducing energy by factor (2π)² ≈ 39.5
- See CLAUDE.md for detailed normalization discussion (Issue #78)

## Forced Turbulence: Parameter Selection Guide

**CRITICAL for forced turbulence simulations**: Energy injection (forcing) must balance energy removal (dissipation) to avoid instabilities.

### Quick Start Parameters (Validated Stable)

| Resolution | η (dissipation) | amplitude | Runtime | Notes |
|------------|----------------|-----------|---------|-------|
| 32³        | 1.0            | 0.05      | 50+ τ_A | Recommended starting point |
| 64³        | 20.0           | 0.01      | 50+ τ_A | Requires strong η (anomaly) |
| 128³       | 2.0            | 0.05      | 50+ τ_A | Moderate parameters work |

**Force modes**: `[1, 2]` (large scales, k = 2π and 4π)
**Hyper-dissipation**: `r=2, n=2` (recommended)

### How to Choose Parameters for Your Simulation

**The fundamental constraint**: Energy injection rate ≤ Dissipation rate

**Starting recipe** (conservative, guaranteed stable):
```python
from krmhd import gandalf_step, force_alfven_modes

# 1. Start with conservative parameters
eta = 5.0
force_amplitude = 0.01
force_modes = [1, 2]  # Large scales

# 2. Test run (20 τ_A with diagnostics)
for step in range(n_steps):
    state, key = force_alfven_modes(state, amplitude=force_amplitude,
                                    k_min=1.0, k_max=3.0, dt=dt, key=key)
    state = gandalf_step(state, dt=dt, eta=eta, nu=eta, v_A=1.0,
                        hyper_r=2, hyper_n=2)

    # Monitor energy: should plateau after spin-up
    E = compute_energy(state)['total']
```

**With diagnostics**:
```bash
uv run python examples/alfvenic_cascade_benchmark.py \
  --resolution 64 --total-time 20 --save-diagnostics
```

### Warning Signs of Instability

Watch for these during your simulation:
1. **Energy grows exponentially** (straight line on log plot) → Reduce forcing OR increase η
2. **Max velocity > 100** → Energy pile-up, reduce forcing
3. **CFL violations** (CFL > 1.0) → Reduce timestep (but CFL < 1 ≠ stability!)

### Troubleshooting

**Problem**: Energy grows exponentially after 10-20 τ_A

**Not the issue**:
- ✅ Timestep size (unless CFL > 1.0)
- ✅ Dissipation implementation (verified Issue #82)

**Root cause**: Energy injection/dissipation imbalance

**Solutions** (pick ONE):
- **Reduce forcing**: `amplitude → amplitude / 2`
- **Increase dissipation**: `eta → eta × 2`
- **Quick fix for 64³**: Use `eta=20.0, amplitude=0.01` (known stable)

### Why Normalized Hyper-Dissipation Requires Careful Tuning

Normalized hyper-dissipation (`r=2`) concentrates damping at high-k:
- **Low-k modes (k=2)**: Only 1.6% energy decay over 10 τ_A
- **High-k modes (k=8)**: 98% energy decay over 10 τ_A

This is **by design** to preserve the inertial range. But it means:
- Forcing at low-k adds energy where dissipation is weak
- Energy must cascade to high-k before being removed
- If cascade is too slow → energy accumulates → instability

**For details**: See CLAUDE.md "Forced Turbulence: Parameter Selection Guide" section or `docs/ISSUE82_SUMMARY.md`

### Automated Diagnostics

Use built-in tools to detect instabilities early:
```bash
# Run with diagnostics
uv run python examples/alfvenic_cascade_benchmark.py \
  --resolution 64 --save-diagnostics --diagnostic-interval 5

# Analyze results
uv run python examples/analyze_64cubed_detailed.py
```

Tracks:
- Max velocity evolution
- CFL number
- High-k energy pile-up
- Critical balance ratio
- Energy injection vs dissipation

## Typical Physical Parameters

Reference values for astrophysical plasmas:

- **Plasma beta**: 0.01 - 100 (ratio of thermal to magnetic pressure)
- **Temperature ratio tau**: 1 - 10 (T_i/T_e)
- **Resolution**: 128³ to 512³ grid points (3D spectral)

## Validation Tests

The code includes validation against:

1. **Linear dispersion relations**: Alfven waves (ω² = k∥²v_A²) - tests in `test_physics.py`
2. **Orszag-Tang vortex**: Standard MHD benchmark - `examples/orszag_tang.py` ✅
3. **Decaying turbulence**: k^(-5/3) inertial range spectrum - `examples/decaying_turbulence.py` ✅
4. **Energy conservation**: 0.0086% error in inviscid runs (Issue #44 resolved)
5. **Landau damping**: Emerges from parallel streaming in Hermite moment evolution

## Current Status

### Completed ✅
- **Project infrastructure** (Issue #1): UV package management, dependencies, testing framework
- **Spectral methods** (Issues #2, #19): Full 2D/3D spectral grids with FFT operations
  - Real-space ↔ Fourier-space transforms (rfftn for memory efficiency)
  - Spatial derivatives (∂/∂x, ∂/∂y, ∂/∂z) and Laplacian operators
  - 2/3 dealiasing for nonlinear terms
  - Lazy evaluation with caching via SpectralField2D/3D
- **Poisson solver** (Issue #3): Spectral solver for ∇²φ = ω
  - 2D and 3D implementations with perpendicular/full options
  - Proper k=0 mode handling
- **Hermite basis** (Issue #21): Infrastructure for kinetic physics
  - Orthonormal Hermite functions (quantum harmonic oscillator eigenstates)
  - Moment projection and distribution reconstruction
  - Foundation for Landau closures
- **Poisson brackets** (Issue #4): Nonlinear advection operator
  - poisson_bracket_2d() and poisson_bracket_3d() with spectral derivatives
  - 2/3 dealiasing applied, JIT-compiled
- **KRMHD equations** (Issues #5-6): Full Elsasser formulation
  - KRMHDState with z± Elsasser variables
  - z_plus_rhs() and z_minus_rhs() using GANDALF energy-conserving formulation
  - Multiple initialization functions (Alfvén waves, turbulent spectra)
  - Energy conservation: 0.0086% error (Issue #44 resolved!)
- **Kinetic physics** (Issues #22-24, #49): Hermite moment hierarchy
  - g0_rhs(), g1_rhs(), gm_rhs() implement full kinetic coupling
  - Lenard-Bernstein collision operator: C[gm] = -νmgm
  - Hermite moment closures (truncation, ready for advanced)
- **Time integration** (Issues #8, #47): GANDALF integrating factor + RK2
  - Integrating factor e^(±ikz·t) handles linear propagation exactly
  - RK2 (midpoint method) for nonlinear terms
  - Exponential dissipation factors for resistivity and collisions
  - CFL timestep calculator
- **Diagnostics** (Issue #9): Energy spectra and visualization
  - energy_spectrum_1d(), energy_spectrum_perpendicular(), energy_spectrum_parallel()
  - EnergyHistory for tracking E(t), magnetic fraction, dissipation rate
  - Visualization functions: plot_state(), plot_energy_history(), plot_energy_spectrum()
  - Examples: decaying_turbulence.py and orszag_tang.py demonstrate full workflow
- **Forcing mechanisms** (Issue #29): Gaussian white noise forcing for driven turbulence
  - gaussian_white_noise_fourier(): Band-limited stochastic forcing
  - force_alfven_modes(): Forces z⁺=z⁻ identically (drives u⊥ only, not B⊥)
  - force_slow_modes(): Independent forcing for δB∥
  - compute_energy_injection_rate(): Energy diagnostics for balance validation
  - Hermitian symmetry enforcement for rfft format (critical for direct Fourier operations)
- **Validation examples** (Issues #11-12, #27): Physics benchmarks
  - Orszag-Tang vortex: Nonlinear dynamics and current sheet formation
  - Decaying turbulence: Spectral cascade and selective decay
  - Driven turbulence: Forced steady-state with energy balance (ε_inj ≈ ε_diss)
  - Minimal forcing: 50-line example showing basic forcing workflow
  - **Kinetic FDT validation**: Drive single k-modes, measure |gₘ|² spectrum, compare with theory (Issue #27)
  - All with comprehensive diagnostics and visualization
- **Performance benchmarks** (Issue #35): Comprehensive performance benchmarks with catastrophic failure detection
  - 2D Poisson bracket: 64² to 512² resolution scaling
  - 3D Poisson bracket: 32³ to 256³ resolution scaling (primary use case)
  - Realistic workload tests for 128³ turbulence simulations
  - Baseline performance data for M1 Pro (JAX with Metal)
  - Throughput metrics, memory usage, and O(N³ log N) scaling analysis
  - Can run via pytest or standalone: `python tests/test_performance.py`

**Test Coverage:** 285 passing tests across all modules (includes 10 performance benchmarks)

### In Progress 🚧
- **Extended validation** (Issue #10): Kinetic Alfven waves, Landau damping with exact dispersion
- **Advanced diagnostics** (Issue #25): True k∥ spectra via FFT along curved field lines

### Planned 📋
- **Production features** (Issues #13, #15, #28, #30): HDF5 I/O, hyper-dissipation, configuration files

## References

- Original GANDALF code: https://github.com/anjor/gandalf-original
- JAX documentation: https://jax.readthedocs.io/
- Schekochihin et al. (2009): "Astrophysical Gyrokinetics" (ApJS 182, 310)
- Kunz et al. (2018): "Kinetic turbulence in pressure-anisotropic plasmas" (JPP 84)

## License

See [LICENSE](LICENSE) for details.

## Contact

For questions or collaboration inquiries, contact anjor@umd.edu.
