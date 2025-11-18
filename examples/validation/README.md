# KRMHD Physics Validation Suite

This directory contains scripts that validate the numerical implementation against analytical theory, expected physical behaviors, and convergence properties. These tests verify that the code correctly implements the KRMHD equations and kinetic physics.

## Validation Scripts

### `hyper_dissipation_demo.py` - Hyper-Dissipation Comparison

**Purpose:** Compare standard dissipation (r=1) with hyper-dissipation (r=2) side-by-side to demonstrate the benefits of selective small-scale damping.

**Physics:** Hyper-dissipation operators -η∇^(2r)⊥ concentrate damping at high-k while preserving large-scale dynamics. Higher r means sharper cutoff and better energy preservation in the inertial range.

**Runtime:** ~20-30 seconds (64³ resolution, 10 τ_A each)

**Usage:**
```bash
# Run comparison with default parameters
uv run python examples/validation/hyper_dissipation_demo.py

# Custom resolution and dissipation coefficient
uv run python examples/validation/hyper_dissipation_demo.py --resolution 128 --eta 1.0
```

**What it validates:**
- Hyper-dissipation implementation correctness
- Energy preservation: r=2 shows 24% decay vs r=1 shows 84% decay (at η=0.5)
- Spectral behavior: Sharp exponential cutoff at high-k for r=2
- Large-scale preservation: Inertial range (k ~ 2-5) unaffected by hyper-dissipation

**Output:**
- Side-by-side comparison plot: energy evolution (top) and spectra (bottom)
- Console: Energy decay percentages for each case

**Expected results:**
- r=1: Significant energy loss (~80-90% over 10 τ_A)
- r=2: Minimal energy loss (~20-30% over 10 τ_A)
- Both: Exponential cutoff at high-k, but steeper for r=2

---

### `kinetic_fdt_validation.py` - Kinetic Fluctuation-Dissipation Theorem

**Purpose:** Validate kinetic physics implementation by comparing Hermite moment spectrum |g_m(k)|² against analytical predictions from fluctuation-dissipation theorem (FDT).

**Physics:** Force single Fourier mode in stream function φ, measure time-averaged response in Hermite moments g_m(k). FDT relates fluctuations to dissipation, providing analytical expressions for |g_m|² including:
- Landau resonance (plasma dispersion function Z)
- Finite Larmor radius corrections (Bessel functions I_m)
- Collisional damping (exponential decay with m)

**Runtime:** ~10 seconds (32³ resolution, 50 τ_A)

**Usage:**
```bash
# Run with default parameters
uv run python examples/validation/kinetic_fdt_validation.py

# Different plasma beta and temperature ratio
uv run python examples/validation/kinetic_fdt_validation.py --beta 1.0 --tau 2.0

# Vary collision frequency
uv run python examples/validation/kinetic_fdt_validation.py --nu 0.1
```

**What it validates:**
- Hermite moment evolution (g0_rhs, g1_rhs, gm_rhs)
- Collision operators (exponential damping)
- Kinetic parameter Λ implementation
- Time-averaging convergence

**Output:**
- Log-linear plot: |g_m|² vs moment number m
- Comparison with analytical FDT prediction
- Console: RMS error between numerical and analytical spectra

**Expected results:**
- Exponential decay: |g_m|² ~ exp(-αm) for collisional case
- Power law decay: |g_m|² ~ m^(-3/2) for phase mixing (collisionless)
- Good agreement with theory (RMS error < 10%)

**Reference:** Thesis Section 3.4, Equations 3.37 & 3.58

---

### `grid_convergence_tests.py` - Spectral Method Convergence

**Purpose:** Demonstrate exponential convergence of spectral methods by comparing solutions at multiple resolutions (32³, 64³, 128³).

**Physics:** Spectral methods achieve exponential convergence for smooth solutions: error ~ exp(-c·N) where N is grid points. This is superior to finite difference methods which converge algebraically.

**Runtime:** ~2-5 minutes (runs 3 resolutions)

**Usage:**
```bash
# Run full convergence test suite
uv run python examples/validation/grid_convergence_tests.py

# Test specific simulation type
uv run python examples/validation/grid_convergence_tests.py --sim-type alfven_wave

# Custom resolutions
uv run python examples/validation/grid_convergence_tests.py --resolutions 32 64 128 256
```

**What it validates:**
- Spectral derivative accuracy
- Dealiasing effectiveness
- Resolution independence of physics
- Exponential error convergence

**Output:**
- Convergence plot: log(error) vs resolution
- Exponential fit showing convergence rate
- Console: Convergence order estimate

**Expected results:**
- Exponential convergence: error decreases by factor ~100 per resolution doubling
- Physics-independent: k^(-5/3) slope maintained at all resolutions
- Dealiasing effective: No spurious energy at high-k

**Test cases:**
- Alfvén wave propagation (linear test)
- Decaying turbulence (nonlinear test)
- Orszag-Tang vortex (complex nonlinear test)

---

### `hermite_convergence.py` - Hermite Moment Truncation

**Purpose:** Study convergence of kinetic solution with increasing Hermite moment truncation (M = 2, 4, 8, 16).

**Physics:** Kinetic distribution f(v∥) is expanded in Hermite polynomials. Higher moments (larger M) capture fine velocity-space structure. Convergence indicates sufficient resolution of velocity space.

**Runtime:** ~1-2 minutes (32³ spatial, 4 moment truncations)

**Usage:**
```bash
# Run with default parameters
uv run python examples/validation/hermite_convergence.py

# Different truncation values
uv run python examples/validation/hermite_convergence.py --max-moments 2 4 8 16 32

# Higher spatial resolution
uv run python examples/validation/hermite_convergence.py --resolution 64
```

**What it validates:**
- Hermite moment hierarchy coupling
- Truncation closure effects
- Velocity-space resolution requirements
- Convergence diagnostics

**Output:**
- Multi-panel plot: Energy evolution for each M
- Convergence metrics vs M
- Console: Moment energy distribution

**Expected results:**
- Convergence: Solutions agree for M ≥ 8 typical
- Energy cascade: E_m decays exponentially with m
- Truncation effects: M too small shows artificial damping

**Interpretation:**
- Collisional: Lower M sufficient (moments damped by collisions)
- Collisionless: Higher M needed (phase mixing populates high-m)
- Production runs: M = 8-16 typical compromise

---

## Validation Workflows

### Quick Validation Check
Run all validation scripts with default parameters to verify installation:
```bash
uv run python examples/validation/hyper_dissipation_demo.py
uv run python examples/validation/kinetic_fdt_validation.py
uv run python examples/validation/grid_convergence_tests.py
uv run python examples/validation/hermite_convergence.py
```

### Parameter Study
Explore physics parameter space:
```bash
# Vary plasma beta
for beta in 0.01 0.1 1.0 10.0; do
    uv run python examples/validation/kinetic_fdt_validation.py --beta $beta
done

# Vary dissipation order
for r in 1 2 4; do
    uv run python examples/validation/hyper_dissipation_demo.py --hyper-r $r
done
```

### Publication-Quality Figures
Generate high-resolution validation plots:
```bash
# Convergence study at high resolution
uv run python examples/validation/grid_convergence_tests.py --resolutions 64 128 256

# Hermite convergence with extended moment range
uv run python examples/validation/hermite_convergence.py --max-moments 2 4 8 16 32
```

## Validation Checklist

Before trusting results from production runs, verify:

- [ ] **Hyper-dissipation:** Exponential cutoff at high-k, energy preservation in inertial range
- [ ] **Kinetic FDT:** |g_m|² matches analytical prediction within ~10%
- [ ] **Grid convergence:** Exponential error decay with increasing resolution
- [ ] **Hermite convergence:** Energy distribution converged by M=8 or M=16

## Understanding Validation Results

### Hyper-Dissipation Demo

**Good results:**
- r=2 preserves 70-80% energy over 10 τ_A
- Sharp exponential cutoff at k > 0.7 k_max
- Inertial range unaffected (k ~ 2-10)

**Warning signs:**
- Flat spectrum at high-k → increase η
- Excessive damping (>50% loss for r=2) → decrease η

### Kinetic FDT

**Good results:**
- Exponential |g_m|² decay with m
- RMS error < 10% vs analytical prediction
- Time-averaged spectrum converged after ~10 τ_A

**Warning signs:**
- Growing |g_m|² at high-m → increase ν (collisions)
- Poor agreement with theory → check kinetic parameter Λ
- Non-convergent time average → run longer

### Grid Convergence

**Good results:**
- Error decreases by factor ~100 per doubling
- Convergence order > 2 (exponential)
- Physics (k^(-5/3)) maintained at all resolutions

**Warning signs:**
- Convergence order < 2 → check dealiasing
- Different physics at different N → numerical artifact
- Error plateau → insufficient resolution

### Hermite Convergence

**Good results:**
- Solutions agree within ~5% for M ≥ 8
- Energy in m > M/2 negligible (<1%)
- Smooth exponential E_m decay

**Warning signs:**
- Significant energy at m ~ M → increase M
- Non-exponential E_m → check collision operator
- Convergence not achieved → may need M > 16

## Related Tests

Additional validation is performed in the test suite (`tests/` directory):
- Unit tests: Individual function validation
- Integration tests: Multi-step timestepping
- Linear physics: Alfvén wave dispersion
- Energy conservation: Inviscid runs

See `pytest` output for comprehensive test results.

## Adding New Validations

When adding new validation scripts:
1. Compare against analytical theory when possible
2. Include convergence study (vary relevant parameter)
3. Generate quantitative error metrics (RMS, L2 norm, etc.)
4. Create clear visualization with reference curves
5. Document expected results and warning signs
6. Update this README with usage and interpretation

## Physics References

- **Spectral Methods:** Boyd (2001), Canuto et al. (2007)
- **Hermite Expansion:** Hammett & Perkins (1990)
- **FDT:** Howes et al. (2006), Told et al. (2015)
- **RMHD Theory:** Strauss (1976), Schekochihin et al. (2009)
- **This Implementation:** See CLAUDE.md and thesis references
