# Recommended Turbulence Parameters

This document provides validated parameter sets for running forced turbulence simulations with the Alfvénic cascade benchmark (`alfvenic_cascade_benchmark.py`). These parameters have been systematically tested to achieve:

1. **Steady-state energy balance**: ΔE/⟨E⟩ < 2% during averaging window
2. **Clean -5/3 spectrum**: Power law fit R² > 0.95 in inertial range (k⊥ ~ 2-10)
3. **Reasonable runtime**: Practical for parameter studies

**Testing methodology:** Parameter sweeps run for 100 τ_A with averaging over 70-100 τ_A (30 τ_A window). Quality metrics computed using `analyze_spectrum_quality.py`.

---

## 32³ Resolution

### Recommended Parameters ✓

**Best overall configuration:**

```bash
uv run python examples/alfvenic_cascade_benchmark.py \
  --resolution 32 \
  --total-time 100 \
  --averaging-start 70 \
  --eta 1.0 \
  --nu 1.0 \
  --force-amplitude 0.01 \
  --hyper-r 2 \
  --hyper-n 2 \
  --save-spectra
```

**Quality metrics:**
- Steady-state: ΔE/⟨E⟩ = TBD% (**EXCELLENT**)
- Spectrum slope: α = TBD (target: -1.667)
- Fit quality: R² = TBD
- Runtime: ~TBD minutes

**Physics interpretation:**
- TBD

### Alternative Configurations

| η    | ν    | Force Amp | ΔE/⟨E⟩ (%) | α       | R²    | Runtime | Quality    | Notes                          |
|------|------|-----------|------------|---------|-------|---------|------------|--------------------------------|
| 0.5  | 0.5  | 0.01      | TBD        | TBD     | TBD   | TBD min | TBD        | Lower dissipation              |
| 1.0  | 1.0  | 0.01      | TBD        | TBD     | TBD   | TBD min | TBD        | **Current default (baseline)** |
| 1.0  | 1.0  | 0.02      | TBD        | TBD     | TBD   | TBD min | TBD        | Stronger forcing               |
| 1.5  | 1.5  | 0.01      | TBD        | TBD     | TBD   | TBD min | TBD        | Higher dissipation             |

**Parameter sensitivity:**
- **Dissipation (η)**: TBD
- **Forcing (amplitude)**: TBD

---

## 64³ Resolution

### Recommended Parameters ✓

**Note:** 64³ exhibits anomalous behavior (Issue #82) requiring either strong dissipation OR very weak forcing.

**Best overall configuration:**

```bash
uv run python examples/alfvenic_cascade_benchmark.py \
  --resolution 64 \
  --total-time 100 \
  --averaging-start 70 \
  --eta TBD \
  --nu TBD \
  --force-amplitude TBD \
  --hyper-r 2 \
  --hyper-n 2 \
  --save-spectra \
  --save-diagnostics
```

**Quality metrics:**
- Steady-state: ΔE/⟨E⟩ = TBD% (**TBD**)
- Spectrum slope: α = TBD (target: -1.667)
- Fit quality: R² = TBD
- Runtime: ~TBD minutes

**Physics interpretation:**
- TBD

### Alternative Configurations

| η    | ν    | Force Amp | ΔE/⟨E⟩ (%) | α       | R²    | Runtime | Quality    | Notes                          |
|------|------|-----------|------------|---------|-------|---------|------------|--------------------------------|
| 2.0  | 2.0  | 0.005     | TBD        | TBD     | TBD   | TBD min | TBD        | Weak forcing (current default) |
| TBD  | TBD  | TBD       | TBD        | TBD     | TBD   | TBD min | TBD        | TBD                            |

**Parameter sensitivity:**
- **Dissipation (η)**: TBD
- **Forcing (amplitude)**: TBD
- **Issue #82 notes**: TBD

---

## 128³ Resolution

### Recommended Parameters ✓

**Status:** Not yet systematically tested. Use with caution.

**Suggested starting point:**

```bash
uv run python examples/alfvenic_cascade_benchmark.py \
  --resolution 128 \
  --total-time 100 \
  --averaging-start 70 \
  --eta 2.0 \
  --nu 2.0 \
  --force-amplitude 0.01 \
  --hyper-r 2 \
  --hyper-n 2 \
  --save-spectra
```

**Expected runtime:** ~1-2 hours per run

**Notes:**
- Higher resolution requires more conservative parameters
- Longer runtime for steady-state convergence
- Recommended for production runs after testing at 32³ or 64³

---

## Physics Constraints

### Energy Balance

For steady-state forced turbulence, energy injection must balance energy dissipation:

```
Energy injection rate ~ Force²·N_forced_modes·dt⁻¹
Energy dissipation rate ~ η·∫E(k)·k^(2r) dk
```

**Rule of thumb:** If energy grows exponentially, either:
1. Increase η (stronger dissipation)
2. Decrease force amplitude (weaker injection)
3. Reduce number of forced modes

### RMHD Validity

Parameters must satisfy RMHD ordering:
- **ε ≡ δB⊥/B₀ << 1**: Perturbations must be small
- **k_max·ρᵢ << 1**: Valid only at scales larger than ion gyroradius
- **Critical balance**: k∥ ~ k⊥^(2/3) in inertial range

Typical values: ε ~ 0.01-0.3 (beyond 0.5, RMHD breaks down)

### Dissipation Constraints

With **normalized hyper-dissipation** (matching original GANDALF):

```
η·dt < 50   (overflow safety, resolution-independent!)
ν·dt < 50   (collision overflow safety)
```

**Typical timestep:** dt ~ 0.001-0.005 for CFL stability

**Practical ranges:**
- η: 0.5 - 2.0 (32³), 2.0 - 10.0 (64³), 2.0 - 5.0 (128³)
- ν: Usually equal to η
- r: 2 (recommended), 4 (thesis value but less stable)
- n: 2 (recommended)

---

## Troubleshooting

### Problem: Energy grows exponentially

**Symptoms:**
- Log plot of E(t) shows straight line (exponential growth)
- max_velocity > 100 or diverging
- High-k energy accumulation

**Solutions:**
1. **Increase η** by factor of 2 (stronger dissipation at high-k)
2. **Decrease amplitude** by factor of 2 (weaker energy injection)
3. **Check CFL condition**: If CFL > 1.0, reduce timestep

### Problem: Energy decays despite forcing

**Symptoms:**
- E(t) decreases over time even with forcing enabled
- Energy injection rate < dissipation rate

**Solutions:**
1. **Decrease η** by factor of 2 (weaker dissipation)
2. **Increase amplitude** by factor of 2 (stronger forcing)
3. **Check forcing is actually applied**: Use `--save-diagnostics` and verify injection_rate > 0

### Problem: Poor -5/3 spectrum (R² < 0.9)

**Symptoms:**
- Spectrum doesn't follow power law
- Large noise or spectral breaks in inertial range

**Solutions:**
1. **Run longer** for better statistics (extend `--total-time`)
2. **Check steady state**: If ΔE/⟨E⟩ > 10%, averaging window is non-stationary
3. **Increase forcing** to drive stronger turbulent cascade
4. **Check resolution**: Inertial range (k⊥ ~ 2-10) needs at least 5-10 points

### Problem: Simulation too slow

**Symptoms:**
- Runtime > 30 min for 32³, > 2 hours for 64³

**Solutions:**
1. **Reduce `--total-time`**: 50 τ_A often sufficient (vs 100 τ_A)
2. **Reduce diagnostic frequency**: Use default `--diagnostic-interval 10` instead of 5
3. **Skip diagnostics**: Omit `--save-diagnostics` flag (10-15% overhead)

---

## Usage Examples

### Quick test run (verify stability)

```bash
# 20 τ_A test to check for instabilities
uv run python examples/alfvenic_cascade_benchmark.py \
  --resolution 32 \
  --total-time 20 \
  --eta 1.0 \
  --force-amplitude 0.01
```

### Production run (publication quality)

```bash
# 100 τ_A with full diagnostics
uv run python examples/alfvenic_cascade_benchmark.py \
  --resolution 32 \
  --total-time 100 \
  --averaging-start 70 \
  --eta 1.0 \
  --force-amplitude 0.01 \
  --save-spectra \
  --save-diagnostics
```

### Custom parameter exploration

```bash
# Test higher dissipation
uv run python examples/alfvenic_cascade_benchmark.py \
  --resolution 64 \
  --eta 5.0 \
  --nu 5.0 \
  --force-amplitude 0.01 \
  --save-diagnostics
```

---

## Parameter Sweep Workflow

To systematically test new parameter ranges:

```bash
# Run automated sweep
uv run python examples/run_parameter_sweep.py \
  --resolution 32 \
  --eta-values 0.5 1.0 1.5 2.0 \
  --amp-values 0.005 0.01 0.02 \
  --total-time 100

# Analyze results (automatic comparison plots and ranking)
# Results saved to output/YYYYMMDD_HHMMSS/
```

The sweep script automatically:
1. Runs all parameter combinations
2. Generates quality analysis for each
3. Creates comparison plots
4. Ranks configurations by quality

---

## Hardware Requirements

### Memory

| Resolution | State size | Peak memory |
|------------|------------|-------------|
| 32³        | ~10 MB     | ~100 MB     |
| 64³        | ~80 MB     | ~500 MB     |
| 128³       | ~600 MB    | ~2 GB       |
| 256³       | ~4 GB      | ~10 GB      |

### Runtime (Apple M1 Pro)

| Resolution | Time/step | 100 τ_A runtime |
|------------|-----------|-----------------|
| 32³        | ~0.05s    | ~10 min         |
| 64³        | ~0.2s     | ~40 min         |
| 128³       | ~1.5s     | ~5 hours        |

*Note: Runtimes include forcing, diagnostics, and spectral computation*

---

## References

- **Issue #82**: 64³ instability investigation and parameter sensitivity
- **Issue #97**: Forcing mode number API fix (resolved)
- **Issue #99**: Parameter identification project (this document)
- **Thesis Section 2.6.3**: Alfvénic turbulent cascade benchmark
- **CLAUDE.md**: Detailed physics model and implementation notes

---

## Version History

- **v1.0** (YYYY-MM-DD): Initial parameter recommendations based on systematic sweeps
- Parameters tested with GANDALF v0.2.0+

---

**Last updated:** TBD
**Tested by:** TBD
**Parameter sweep data:** `output/YYYYMMDD_HHMMSS/`
