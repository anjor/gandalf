# KRMHD Examples

This directory contains example simulations and validation scripts demonstrating the capabilities of the KRMHD spectral solver.

## Directory Structure

### `benchmarks/` - Core Simulation Examples

Standard test cases and demonstrations of the solver's physics capabilities. Start here if you're new to the code.

**Quick Start:**
```bash
# Minimal forcing example (~2 seconds)
uv run python examples/benchmarks/forcing_minimal.py

# Comprehensive driven turbulence (~20 seconds)
uv run python examples/benchmarks/driven_turbulence.py

# Decaying turbulence cascade (~1-2 minutes)
uv run python examples/benchmarks/decaying_turbulence.py
```

See [benchmarks/README.md](benchmarks/README.md) for detailed descriptions and usage.

### `validation/` - Physics Validation Scripts

Tests that verify the numerical implementation against analytical theory and expected physical behaviors.

**Examples:**
```bash
# Compare hyper-dissipation orders (r=1 vs r=2)
uv run python examples/validation/hyper_dissipation_demo.py

# Validate kinetic physics against fluctuation-dissipation theorem
uv run python examples/validation/kinetic_fdt_validation.py

# Spectral method convergence tests
uv run python examples/validation/grid_convergence_tests.py
```

See [validation/README.md](validation/README.md) for full validation suite documentation.

## Output Directory Structure

By default, examples create output in:
```
examples/
├── output/
│   ├── checkpoints/       # HDF5 checkpoint files
│   ├── timeseries/        # Energy history data
│   └── figures/           # Generated plots
```

## Common Workflows

### Running a Benchmark Simulation
```bash
# Run Orszag-Tang vortex with default parameters
uv run python examples/benchmarks/orszag_tang.py

# Custom resolution and runtime
uv run python examples/benchmarks/decaying_turbulence.py --resolution 128 --total-time 50
```

### Analyzing Results
```bash
# Plot energy spectrum from checkpoint
uv run python scripts/plot_checkpoint_spectrum.py examples/output/checkpoints/checkpoint_t0300.0.h5

# Visualize field line structure
uv run python scripts/field_line_visualization.py --checkpoint examples/output/checkpoints/checkpoint.h5
```

### Configuration-Based Runs
Instead of command-line arguments, use YAML configuration files:
```bash
# Run from config (see configs/ directory)
uv run python scripts/run_simulation.py --config configs/decaying_turbulence.yaml

# Generate template config
uv run python scripts/run_simulation.py --template > my_config.yaml
```

## Performance Guidelines

| Resolution | Runtime (typical) | Memory | Recommended Use |
|-----------|------------------|--------|-----------------|
| 32³       | 2-10 seconds     | ~50 MB | Quick tests, development |
| 64³       | 10-60 seconds    | ~200 MB | Standard benchmarks |
| 128³      | 5-30 minutes     | ~1 GB | Production runs (M1 Pro) |
| 256³      | 1-4 hours        | ~8 GB | High-resolution (HPC recommended) |

**Note:** Timings are approximate and depend on total simulation time (τ_A), diagnostic frequency, and forcing parameters.

## Parameter Selection Guide

### Forcing Amplitude and Dissipation

For forced turbulence, balance energy injection (forcing) with energy removal (dissipation):

```python
# Conservative (guaranteed stable)
eta = 5.0              # Hyper-resistivity
force_amplitude = 0.01 # Forcing strength

# Moderate (test first!)
eta = 2.0
force_amplitude = 0.05

# Resolution-dependent validated values
# 32³: eta=1.0, amplitude=0.05
# 64³: eta=20.0, amplitude=0.01 (anomalous, under investigation)
# 128³: eta=2.0, amplitude=0.05
```

### Hyper-Dissipation Orders

```python
hyper_r = 2  # RECOMMENDED for production (stable, validated)
hyper_r = 4  # Thesis value (sharper cutoff, but stability issues in forced runs)
```

See `CLAUDE.md` for detailed discussion of numerical stability and parameter constraints.

## Getting Help

- **Code documentation**: See docstrings in each script
- **Physics background**: See `CLAUDE.md` for KRMHD equations and numerical methods
- **Troubleshooting**: Check `CLAUDE.md` sections on "Forced Turbulence Parameter Selection" and "Detecting Numerical Instabilities"
- **Post-processing**: See `scripts/README.md` for visualization and analysis tools

## Contributing New Examples

When adding new example scripts:
1. Include comprehensive docstring with physics context
2. Support command-line arguments with `argparse`
3. Print progress and diagnostic information
4. Save outputs to `examples/output/` subdirectories
5. Update relevant README (this file, benchmarks/, or validation/)
