# KRMHD: Kinetic Reduced MHD Spectral Solver

A modern Python implementation of a Kinetic Reduced Magnetohydrodynamics (KRMHD) solver using spectral methods for studying turbulent magnetized plasmas in astrophysical environments.

## Overview

KRMHD is a spectral code designed to simulate turbulence in weakly collisional magnetized plasmas, incorporating kinetic effects such as Landau damping and finite Larmor radius corrections. This is a modern rewrite of the legacy [GANDALF](https://github.com/anjor/gandalf) Fortran+CUDA implementation, leveraging JAX for automatic differentiation and GPU acceleration via Apple's Metal backend.

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
- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- **Optional**: GPU acceleration (Metal/CUDA) for better performance

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

## Project Structure

```
krmhd/
├── src/krmhd/          # Main package
│   ├── spectral.py     # ✅ FFT operations, derivatives, dealiasing (2D/3D)
│   ├── hermite.py      # ✅ Hermite basis (kinetic closures)
│   ├── physics.py      # ✅ KRMHD state, Poisson brackets, Elsasser RHS, Hermite RHS
│   ├── timestepping.py # ✅ GANDALF integrating factor + RK2 timestepper
│   ├── diagnostics.py  # ✅ Energy spectra (1D, k⊥, k∥), history, visualization
│   ├── forcing.py      # Turbulence forcing (TODO)
│   ├── io.py          # HDF5 checkpointing (TODO)
│   └── validation.py   # Linear physics tests (TODO)
├── tests/             # Test suite (191 tests)
├── examples/          # Example scripts and tutorials
│   ├── decaying_turbulence.py  # ✅ Turbulent cascade with diagnostics
│   └── orszag_tang.py          # ✅ Orszag-Tang vortex benchmark
└── pyproject.toml     # Project metadata
```

## Quick Start: Running Examples

### Example 1: Decaying Turbulence Simulation

The `examples/decaying_turbulence.py` script demonstrates a complete KRMHD workflow:

```bash
# Run the example (takes ~1-2 minutes on M1 Pro)
uv run python examples/decaying_turbulence.py
```

**What it does:**
1. Initializes a turbulent k^(-5/3) spectrum (Kolmogorov)
2. Evolves 100 timesteps using GANDALF integrator
3. Tracks energy history E(t)
4. Computes energy spectra E(k), E(k⊥), E(k∥)
5. Generates publication-ready plots

**Output files** (saved to `examples/output/`):
- `energy_history.png` - Energy evolution showing selective decay
- `final_state.png` - 2D slices of φ and A∥ fields
- `energy_spectra.png` - Three-panel plot with 1D, perpendicular, and parallel spectra

### Example 2: Orszag-Tang Vortex

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

**Output files** (saved to `examples/output/`):
- `orszag_tang_energy.png` - Energy evolution and partitioning
- `orszag_tang_initial.png` - Initial condition fields
- `orszag_tang_final.png` - Final state showing developed structures
- `orszag_tang_spectrum.png` - Energy spectrum with k^(-5/3) reference

### Example 3: Custom Simulation

```python
from krmhd import (
    SpectralGrid3D,
    initialize_random_spectrum,
    gandalf_step,
    compute_cfl_timestep,
    EnergyHistory,
    energy_spectrum_perpendicular,
    plot_energy_history,
    plot_state,
)

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

# 3. Time evolution loop
for step in range(100):
    # Record energy
    history.append(state)

    # Compute adaptive timestep
    dt = compute_cfl_timestep(state, v_A=1.0, cfl_safety=0.3)

    # Advance one timestep
    state = gandalf_step(state, dt, eta=0.01, v_A=1.0)

    print(f"Step {step}: t={state.time:.3f}, E={history.E_total[-1]:.3e}")

# 4. Analyze and visualize
k_perp, E_perp = energy_spectrum_perpendicular(state)
plot_energy_history(history, filename='energy.png')
plot_state(state, filename='final_state.png')
```

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
```

### Code Quality

```bash
# Format code
uv run ruff format

# Lint
uv run ruff check

# Type checking
uv run mypy src/krmhd
```

## Typical Physical Parameters

Reference values for astrophysical plasmas:

- **Plasma beta**: 0.01 - 100 (ratio of thermal to magnetic pressure)
- **Temperature ratio tau**: 1 - 10 (T_i/T_e)
- **Resolution**: 128³ to 512³ grid points (3D spectral)
- **Scale range**: k_max rho_s << 1 (KRMHD valid only at scales larger than ion gyroradius)

## Validation Tests

The code includes validation against:

1. **Linear dispersion relations**: Alfven waves (ω² = k∥²v_A²) - tests in `test_physics.py`
2. **Orszag-Tang vortex**: Standard MHD benchmark - `examples/orszag_tang.py` ✅
3. **Decaying turbulence**: k^(-5/3) inertial range spectrum - `examples/decaying_turbulence.py` ✅
4. **Energy conservation**: 0.0086% error in inviscid runs (Issue #44 resolved)
5. **Kinetic Alfven waves**: FLR corrections (planned - Issue #10)
6. **Landau damping**: Analytical damping rates (planned - Issue #27)

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
- **Validation examples** (Issues #11-12): Physics benchmarks
  - Orszag-Tang vortex: Nonlinear dynamics and current sheet formation
  - Decaying turbulence: Spectral cascade and selective decay
  - Both with comprehensive diagnostics and visualization

**Test Coverage:** 191 passing tests across all modules

### Planned 📋
- **Extended validation** (Issues #10, #27): Kinetic Alfven waves, Landau damping, FDT validation
- **Advanced diagnostics** (Issues #25-26): Field line following, phase mixing analysis
- **Production features** (Issues #13-15, #28-30): HDF5 I/O, forcing, hyper-dissipation, configuration files

## References

- Original GANDALF code: https://github.com/anjor/gandalf
- JAX documentation: https://jax.readthedocs.io/
- Schekochihin et al. (2009): "Astrophysical Gyrokinetics" (ApJS 182, 310)
- Kunz et al. (2018): "Kinetic turbulence in pressure-anisotropic plasmas" (JPP 84)

## License

See [LICENSE](LICENSE) for details.

## Contact

For questions or collaboration inquiries, contact anjor@umd.edu.
