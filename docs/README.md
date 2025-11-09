# GANDALF Documentation

**G-and-Alf**: Solves for the distribution function **G** (slow modes) and **Alf**vén waves.

A modern JAX-based implementation of Kinetic Reduced MHD (KRMHD) for studying turbulent magnetized plasmas.

## Documentation Overview

This documentation is organized into four main sections:

### 🚀 [Running Simulations](running_simulations.md)
Step-by-step guide for getting started with GANDALF:
- Running your first simulation
- Using the config YAML system
- Command-line arguments and options
- Understanding output files
- Post-processing and visualization
- Common pitfalls and solutions

**Start here if you're new to the code.**

### 📊 [Parameter Scans](parameter_scans.md)
Workflows for systematic parameter studies:
- Scanning plasma parameters (β, temperature ratio)
- Resolution convergence studies
- Dissipation parameter optimization
- Batch processing strategies
- Data organization and comparison

**Essential for research workflows.**

### ⚙️ [Numerical Methods](numerical_methods.md)
Technical details of the implementation:
- 3D Fourier spectral method
- GANDALF integrating factor timestepping
- Dealiasing and FFT conventions
- Hyper-dissipation formulation
- Hermite moment expansion
- CFL condition and stability

**Read this to understand what the code does under the hood.**

### 🔬 [Physics Validity Regimes](physics_validity.md)
When and where KRMHD is valid:
- RMHD ordering (ε << 1 constraint)
- Critical balance and anisotropic cascade
- Kinetic effects (Landau damping, phase mixing)
- Valid parameter ranges (β, τ, resolution)
- When the code breaks down

**Critical for correct interpretation of results.**

## Quick Start

```bash
# Simplest example - 2 second runtime
uv run python examples/forcing_minimal.py

# Full featured turbulence simulation
uv run python examples/alfvenic_cascade_benchmark.py --resolution 64

# Using config files
uv run python run_simulation.py --config configs/decaying_turbulence.yaml
```

## Example Gallery

The `examples/` directory contains working scripts demonstrating various physics:

| Example | Physics | Runtime | Purpose |
|---------|---------|---------|---------|
| `forcing_minimal.py` | White noise forcing | ~2s | Minimal working example |
| `decaying_turbulence.py` | Turbulent cascade | ~2min | Energy decay, spectra |
| `orszag_tang.py` | Current sheet formation | ~5min | Nonlinear benchmark |
| `alfvenic_cascade_benchmark.py` | Forced turbulence | ~10min | Steady-state turbulence |
| `hyper_dissipation_demo.py` | Dissipation comparison | ~3min | r=1 vs r=2 hyper-dissipation |
| `kinetic_fdt_validation.py` | Landau damping | ~5min | Kinetic physics validation |

See [Running Simulations](running_simulations.md) for detailed walkthrough of each example.

## Configuration Files

Pre-configured YAML templates in `configs/`:
- `test_quick.yaml` - Fast test run (32³, 5 Alfvén times)
- `decaying_turbulence.yaml` - Standard turbulent decay
- `driven_turbulence.yaml` - Forced steady-state turbulence
- `high_resolution.yaml` - Production run (128³)
- `hyper_dissipation.yaml` - High-order dissipation (r=4)

Generate a new template:
```bash
uv run python run_simulation.py --generate-template my_simulation.yaml
```

## Physics Background

GANDALF solves the **Kinetic Reduced MHD** equations, which describe turbulent plasmas in the presence of a strong guide field B₀. The model captures:

- **Alfvénic turbulence**: Perpendicular cascade driven by Poisson bracket nonlinearity
- **Kinetic effects**: Landau damping, phase mixing via Hermite velocity-space expansion
- **Anisotropic cascade**: Critical balance k∥ ~ k⊥^(2/3)
- **Collisional damping**: Lenard-Bernstein collision operator

Valid for weak turbulence (δB⊥/B₀ << 1) at scales larger than the ion gyroradius.

## Key References

- **Original GANDALF**: [github.com/anjor/gandalf-original](https://github.com/anjor/gandalf-original) (Fortran+CUDA)
- **Thesis**: See `gandalf_thesis_chapter.pdf` in repository root
- **KRMHD Theory**: Schekochihin et al. (2009), ApJS 182:310

## Getting Help

- **Issues**: File bug reports or feature requests at [github.com/anjor/gandalf/issues](https://github.com/anjor/gandalf/issues)
- **Examples not working?** Check [Running Simulations](running_simulations.md) troubleshooting section
- **Physics questions?** See [Physics Validity Regimes](physics_validity.md)
- **Performance issues?** See [Numerical Methods](numerical_methods.md) for resolution scaling

## Contributing

See `CLAUDE.md` for detailed physics model and code architecture (oriented toward development).
