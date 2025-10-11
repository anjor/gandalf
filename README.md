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
- macOS with Apple Silicon (M1/M2/M3) for Metal GPU acceleration
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/anjor/gandalf.git
cd gandalf

# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Verify JAX Metal installation
python -c 'import jax; print(jax.devices())'
```

You should see output indicating a Metal device:
```
[METAL(id=0)]
```

### Manual Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

## Metal Backend Notes

JAX uses Apple's Metal plugin for GPU acceleration on Apple Silicon. This is experimental but functional. Key points:

- The `jax-metal` package is installed automatically
- GPU operations are transparent (no code changes needed)
- For compatibility issues, set: `export ENABLE_PJRT_COMPATIBILITY=1`
- See [Apple's JAX documentation](https://developer.apple.com/metal/jax/) for details

## Project Structure

```
krmhd/
├── src/krmhd/          # Main package
│   ├── spectral.py     # FFT operations, derivatives, dealiasing
│   ├── physics.py      # KRMHD equations, Poisson brackets
│   ├── timestepping.py # RK4/RK45 integrators
│   ├── hermite.py      # Hermite basis (kinetic closures)
│   ├── diagnostics.py  # Energy spectra, fluxes
│   ├── io.py          # HDF5 checkpointing
│   └── validation.py   # Linear physics tests
├── tests/             # Test suite
├── examples/          # Example scripts
└── pyproject.toml     # Project metadata
```

## Development

### Running Tests

```bash
uv run pytest
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
- **Resolution**: 256^2 to 1024^2 grid points
- **Scale range**: k_max rho_s ~ 1-2 (resolve ion gyroradius scales)

## Validation Tests

The code includes validation against:

1. **Linear dispersion relations**: Alfven waves, kinetic Alfven waves
2. **Landau damping**: Analytical damping rates
3. **Orszag-Tang vortex**: Standard MHD benchmark
4. **Decaying turbulence**: k^(-5/3) inertial range spectrum
5. **Energy conservation**: Sub-machine precision for inviscid runs

## Current Status

- [x] Project structure and dependencies
- [ ] Core spectral infrastructure
- [ ] KRMHD equation implementation
- [ ] Time integration
- [ ] Linear validation suite
- [ ] Nonlinear turbulence benchmarks
- [ ] Production diagnostics

## References

- Original GANDALF code: https://github.com/anjor/gandalf
- JAX documentation: https://jax.readthedocs.io/
- Schekochihin et al. (2009): "Astrophysical Gyrokinetics" (ApJS 182, 310)
- Kunz et al. (2018): "Kinetic turbulence in pressure-anisotropic plasmas" (JPP 84)

## License

See [LICENSE](LICENSE) for details.

## Contact

For questions or collaboration inquiries, contact anjor@umd.edu.
