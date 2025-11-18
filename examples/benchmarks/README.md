# KRMHD Benchmark Simulations

This directory contains the core simulation examples demonstrating the physics capabilities of the KRMHD spectral solver. These benchmarks cover standard MHD test cases, turbulent cascades, and forced turbulence.

## Quick Start Guide

**New users should start here:**

```bash
# 1. Minimal forcing example (~2 seconds, 32³)
uv run python examples/benchmarks/forcing_minimal.py

# 2. Comprehensive driven turbulence (~20 seconds, 64³)
uv run python examples/benchmarks/driven_turbulence.py

# 3. Decaying turbulence with diagnostics (~1-2 minutes, 64³)
uv run python examples/benchmarks/decaying_turbulence.py
```

## Benchmark Scripts

### `forcing_minimal.py` - Minimal Forcing Example ⭐ START HERE

**Purpose:** Simplest possible forced turbulence example. Minimal code (~50 lines) for quick experimentation.

**Physics:** White noise forcing at large scales (k=1-2), driving Alfvénic turbulence with energy cascade to small scales.

**Runtime:** ~2 seconds (32³ resolution, 10 τ_A)

**Usage:**
```bash
# Run with default parameters
uv run python examples/benchmarks/forcing_minimal.py
```

**What it demonstrates:**
- Basic forced turbulence setup
- Energy injection and dissipation balance
- Minimal working example for users to modify

**Output:** Console progress only (no diagnostics saved)

---

### `driven_turbulence.py` - Comprehensive Driven Turbulence

**Purpose:** Full-featured forced turbulence example with comprehensive diagnostics, energy tracking, and spectrum visualization.

**Physics:** Gaussian white noise forcing at modes n=1-2, driving steady-state Alfvénic turbulence. Monitors energy injection rate, dissipation rate, and spectral evolution.

**Runtime:** ~20 seconds (64³ resolution, 50 τ_A)

**Usage:**
```bash
# Run with default parameters
uv run python examples/benchmarks/driven_turbulence.py

# Custom resolution and runtime
uv run python examples/benchmarks/driven_turbulence.py --resolution 128 --total-time 100

# Save diagnostics
uv run python examples/benchmarks/driven_turbulence.py --save-checkpoints --checkpoint-interval 10
```

**What it demonstrates:**
- Energy injection diagnostics
- Steady-state turbulence achievement
- Time-averaged spectra computation
- Energy history tracking

**Output:**
- Energy history plot: `driven_turbulence_energy.png`
- Final spectrum plot: `driven_turbulence_spectrum.png`
- Console: Energy injection/dissipation balance

---

### `decaying_turbulence.py` - Decaying Turbulence Cascade

**Purpose:** Decaying (unforced) turbulence initialized with k^(-5/3) spectrum. Studies energy decay, selective decay (magnetic energy dominates), and spectral evolution.

**Physics:** Inviscid MHD turbulence conserves ideal invariants, leading to selective decay where magnetic energy is preferentially preserved over kinetic energy. Spectrum maintains approximate k^(-5/3) power law during decay.

**Runtime:** ~1-2 minutes (64³ resolution, 100 τ_A)

**Usage:**
```bash
# Run with default parameters
uv run python examples/benchmarks/decaying_turbulence.py

# Higher resolution
uv run python examples/benchmarks/decaying_turbulence.py --resolution 128 --total-time 200

# Adjust dissipation
uv run python examples/benchmarks/decaying_turbulence.py --eta 0.5 --hyper-r 4
```

**What it demonstrates:**
- Energy conservation (inviscid limit)
- Selective decay (f_mag → 1 over time)
- k^(-5/3) spectral slope maintenance
- Exponential energy decay with dissipation

**Output:**
- Energy history plot: `decaying_turbulence_energy.png`
- Time-averaged spectrum: `decaying_turbulence_spectrum.png`
- Console: Magnetic fraction evolution

---

### `orszag_tang.py` - Orszag-Tang Vortex

**Purpose:** Standard incompressible MHD test case. Validates nonlinear dynamics, current sheet formation, and energy conservation.

**Physics:** 2D flow pattern with prescribed initial conditions. Develops thin current sheets and vortical structures. Exhibits complex nonlinear evolution with characteristic energy oscillations.

**Runtime:** ~10 seconds (128² × 32 resolution, 10 τ_A)

**Usage:**
```bash
# Run with default parameters (matches thesis Figure 2.1)
uv run python examples/benchmarks/orszag_tang.py

# Higher resolution
uv run python examples/benchmarks/orszag_tang.py --resolution 256 --total-time 20
```

**What it demonstrates:**
- Nonlinear MHD dynamics
- Energy conservation (~0.01% error typical)
- Current sheet formation
- Standard MHD benchmark validation

**Output:**
- Energy history plot: `orszag_tang_energy.png`
- Console: Energy conservation error

**Reference:** Thesis Section 2.6.2, Figure 2.1

---

### `alfvenic_cascade_benchmark.py` - Alfvénic Cascade (Thesis Reproduction)

**Purpose:** Reproduce thesis Figure 2.2 showing steady-state turbulent cascade with k⊥^(-5/3) critical-balance spectrum. This is the primary validation benchmark for forced turbulence.

**Physics:** Forced Alfvénic turbulence with balanced energy injection and dissipation, achieving quasi-steady state. Separate kinetic E_kin(k⊥) and magnetic E_mag(k⊥) spectra should both follow k⊥^(-5/3) in inertial range.

**Runtime:**
- 32³: ~5-10 minutes (50 τ_A)
- 64³: ~15-25 minutes (50 τ_A)
- 128³: ~1-2 hours (50 τ_A)

**Usage:**
```bash
# Default: 64³ resolution, 50 τ_A total, average last 20 τ_A
uv run python examples/benchmarks/alfvenic_cascade_benchmark.py

# Different resolutions (32³, 64³, 128³)
uv run python examples/benchmarks/alfvenic_cascade_benchmark.py --resolution 32
uv run python examples/benchmarks/alfvenic_cascade_benchmark.py --resolution 128

# Custom runtime and averaging window
uv run python examples/benchmarks/alfvenic_cascade_benchmark.py --total-time 100 --averaging-start 80

# Save diagnostics for instability investigation
uv run python examples/benchmarks/alfvenic_cascade_benchmark.py --save-diagnostics --diagnostic-interval 5
```

**What it demonstrates:**
- Thesis Figure 2.2 reproduction
- Steady-state forced turbulence
- k⊥^(-5/3) critical-balance spectrum
- Time-averaged spectra (last 20 τ_A by default)
- Kinetic vs magnetic spectrum separation

**Output:**
- Two-panel spectrum plot: `alfvenic_cascade_spectrum_[resolution].png`
  - Left: Kinetic spectrum with k⊥^(-5/3) reference
  - Right: Magnetic spectrum with k⊥^(-5/3) reference
- Console: Energy variation during averaging (target: ΔE/⟨E⟩ < 10%)

**Parameters (resolution-dependent):**
- **32³**: η=1.0, amplitude=0.05, r=2 (stable)
- **64³**: η=20.0, amplitude=0.01, r=2 (anomalous, see CLAUDE.md "Forced Turbulence Parameter Selection" or [Issue #82](https://github.com/anjor/gandalf/issues/82))
- **128³**: η=2.0, amplitude=0.05, r=2 (stable)

**Reference:** Thesis Section 2.6.3, Figure 2.2

**Note:** See `CLAUDE.md` section "Forced Turbulence: Parameter Selection Guide" for discussion of stability constraints.

## Parameter Selection Guidelines

### Resolution and Runtime

| Resolution | Recommended Use | Typical Runtime | Memory |
|-----------|----------------|-----------------|--------|
| 32³       | Quick tests, development | 2-10 seconds | ~50 MB |
| 64³       | Standard benchmarks | 10-60 seconds | ~200 MB |
| 128³      | Production runs | 5-30 minutes | ~1 GB |
| 256³      | High-resolution (HPC) | 1-4 hours | ~8 GB |

### Hyper-Dissipation Parameters

**Recommended for production:**
```python
eta = 2.0       # Hyper-resistivity coefficient
hyper_r = 2     # Dissipation order (STABLE at all tested resolutions)
```

**Thesis value (use with caution - may exhibit numerical instability):**
```python
eta = 0.5-1.0   # Lower coefficient compensates for higher order
hyper_r = 4     # Sharper cutoff, but can cause exponential energy growth in forced runs
                # See CLAUDE.md "Forced Turbulence Parameter Selection" and Issue #82
```

### Forcing Parameters

**Conservative (guaranteed stable):**
```python
force_amplitude = 0.01
force_modes = [1, 2]  # Mode numbers (n=1 is fundamental)
eta = 5.0
```

**Balanced (production):**
```python
force_amplitude = 0.05
force_modes = [1, 2]
eta = 2.0  # Resolution-dependent, see CLAUDE.md
```

**Warning signs of instability:**
- Energy grows exponentially on log plot
- max_velocity > 100 (should saturate at O(1-10))
- Flat/rising spectrum at high-k (spectral pile-up)

See `CLAUDE.md` for detailed parameter selection workflow.

## Common Workflows

### Quick Physics Test
```bash
# Run minimal example to verify installation
uv run python examples/benchmarks/forcing_minimal.py
```

### Standard Benchmark Suite
```bash
# Run all benchmarks at default resolutions
uv run python examples/benchmarks/forcing_minimal.py
uv run python examples/benchmarks/driven_turbulence.py
uv run python examples/benchmarks/decaying_turbulence.py
uv run python examples/benchmarks/orszag_tang.py
```

### Thesis Figure Reproduction
```bash
# Reproduce Figure 2.1 (Orszag-Tang)
uv run python examples/benchmarks/orszag_tang.py

# Reproduce Figure 2.2 (Alfvénic cascade)
uv run python examples/benchmarks/alfvenic_cascade_benchmark.py --resolution 128
```

### Production Run
```bash
# Long-time 128³ simulation with checkpointing
uv run python examples/benchmarks/alfvenic_cascade_benchmark.py \
  --resolution 128 \
  --total-time 100 \
  --averaging-start 80 \
  --save-diagnostics \
  --diagnostic-interval 10
```

## Analyzing Outputs

After running benchmarks, use post-processing scripts:

```bash
# Plot spectrum from checkpoint
uv run python scripts/plot_checkpoint_spectrum.py examples/output/checkpoints/checkpoint_t0300.0.h5

# Thesis-style formatting
uv run python scripts/plot_checkpoint_spectrum.py --thesis-style checkpoint.h5

# Visualize field lines
uv run python scripts/field_line_visualization.py --checkpoint checkpoint.h5
```

## Troubleshooting

### Energy grows exponentially
**Diagnosis:** Energy injection > dissipation
**Solution:** Increase `eta` or decrease `force_amplitude` (see CLAUDE.md Issue #82)

### Flat spectrum at high-k
**Diagnosis:** Under-dissipated, energy piling up at Nyquist
**Solution:** Increase `eta` or use higher `hyper_r`

### Crash with NaN/Inf
**Diagnosis:** Timestep too large or numerical instability
**Solution:** Reduce `dt` or adjust forcing/dissipation balance

### Memory errors
**Diagnosis:** Resolution too high for available RAM
**Solution:** Reduce resolution or run on HPC system

## Related Documentation

- **Validation Tests:** See [../validation/README.md](../validation/README.md) for convergence tests and physics verification
- **Post-Processing Tools:** See [../../scripts/README.md](../../scripts/README.md) for spectrum plotting and analysis tools
- **Main Examples Guide:** See [../README.md](../README.md) for overview and navigation

## Physics References

- **RMHD Theory:** Strauss (1976), Zank & Matthaeus (1992)
- **Critical Balance:** Goldreich & Sridhar (1995)
- **Selective Decay:** Matthaeus & Montgomery (1980)
- **Orszag-Tang:** Orszag & Tang (1979)
- **This Implementation:** See CLAUDE.md for equations and numerical methods
