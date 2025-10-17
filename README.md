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
│   ├── forcing.py      # ✅ Gaussian white noise forcing for driven turbulence
│   ├── io.py          # HDF5 checkpointing (TODO)
│   └── validation.py   # Linear physics tests (TODO)
├── tests/             # Test suite (222 tests)
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

**Test Coverage:** 275 passing tests across all modules

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
