# Physics Validity Regimes

GANDALF solves the **Kinetic Reduced MHD (KRMHD)** equations, which are valid for **weak turbulence** in a **strong guide field**. This document explains when and where KRMHD is applicable, and how to check whether your simulation results are physically meaningful.

## RMHD Ordering and Validity

### The Small Parameter ε

RMHD is an asymptotic expansion in the small parameter:

```
ε ~ δB⊥/B₀ ~ ρᵢ/L << 1
```

where:
- `δB⊥` = perpendicular magnetic field perturbation amplitude
- `B₀` = guide field strength (assumed constant and aligned with ẑ)
- `ρᵢ` = ion gyroradius
- `L` = characteristic perpendicular scale length

**Physical meaning:** Magnetic field lines are only slightly perturbed from straight. The perpendicular magnetic field is much weaker than the guide field.

### Valid Range

**KRMHD is valid when:**
```
ε ~ 0.01 - 0.3
```

**Interpretation:**

| ε value | Regime | Description |
|---------|--------|-------------|
| ε ~ 0.01 | Very weak | Nearly straight field lines, small amplitude waves |
| ε ~ 0.1 | Moderate | Visible field line wandering, well-developed turbulence |
| ε ~ 0.3 | Strong | Approaching RMHD breakdown, critical balance still holds |
| ε ~ 0.5 | **Invalid** | RMHD ordering violated, need full MHD |
| ε ~ 1.0 | **Invalid** | Strong turbulence, reconnection, full MHD required |

**Field line wandering:** At the outer scale L_z (parallel box size), perpendicular displacement is:
```
δr⊥ ~ ε L_z
```

For ε = 0.1 and L_z = 2π, field lines wander by ~0.2π ≈ 0.6 units (always << L_z ✓).

### How to Check ε in Your Simulation

Compute the perturbation amplitude:

```python
from krmhd.physics import elsasser_to_physical

# Convert Elsasser variables to physical fields
phi, A_parallel = elsasser_to_physical(state.z_plus, state.z_minus)

# Compute perpendicular magnetic field
phi_real = jnp.fft.irfftn(phi)
A_parallel_real = jnp.fft.irfftn(A_parallel)

# δB⊥ comes from curl of A∥ẑ (so δB⊥ ~ |∇⊥A∥|)
# For spectral methods, |∇⊥A∥| ~ k⊥|A∥|
k_perp_typical = 2.0  # Typical energy-carrying scale
epsilon = k_perp_typical * jnp.std(A_parallel_real) / (1/jnp.sqrt(4*jnp.pi))

print(f"ε ~ {epsilon:.3f}")
```

**Rule of thumb:** For initial conditions with `energy=0.01` at 64³ resolution, ε ~ 0.1-0.2 (valid).

**Warning signs:**
- ε > 0.3: Approaching RMHD breakdown
- ε > 0.5: Results likely unphysical
- Field line wandering δr⊥ ~ L: Model invalid

## Ordering Hierarchy

RMHD assumes the following ordering:

1. **Perpendicular vs parallel gradients:**
   ```
   k⊥ >> k∥   or   k∥/k⊥ ~ ε
   ```
   Turbulence is **anisotropic**, with much stronger gradients perpendicular to B₀.

2. **Magnetic perturbations:**
   ```
   δB ~ ε B₀   (first order in ε)
   ```
   Perpendicular magnetic field is weak compared to guide field.

3. **Perpendicular flow:**
   ```
   v⊥ ~ εv_A
   ```
   Flow speeds are subsonic and sub-Alfvénic.

**Consequences:**
- Alfvén waves propagate predominantly parallel to B₀
- Nonlinear cascade is driven by perpendicular Poisson bracket {φ, ·}
- Parallel dynamics are linear (wave propagation)

## Critical Balance

In turbulent KRMHD, the cascade is **critically balanced**:

```
τ_nl ~ τ_A
```

where:
- `τ_nl = (k⊥v⊥)⁻¹` = nonlinear eddy turnover time
- `τ_A = (k∥v_A)⁻¹` = Alfvén wave crossing time

**Implication:** As energy cascades to smaller perpendicular scales (increasing k⊥), the parallel wavenumber adjusts to maintain balance:

```
k∥ ~ k⊥^(2/3)
```

This is the **anisotropic cascade** characteristic of KRMHD turbulence.

### Spectrum Predictions

Critical balance leads to:

```
E(k⊥) ~ k⊥^(-5/3)   (perpendicular spectrum, Kolmogorov-like)
E(k∥) ~ k∥^(-2)      (parallel spectrum, steeper)
```

**In GANDALF:**
- Use `energy_spectrum_perpendicular()` to compute E(k⊥)
- Expect k⊥^(-5/3) in inertial range (k ~ 2-10 for 64³ resolution)
- Deviations indicate dissipation range or insufficient forcing

### Checking Critical Balance

The `alfvenic_cascade_benchmark.py` example computes the critical balance ratio:

```python
# In diagnostics
critical_balance_ratio = tau_nl / tau_A
# Should be ~ 1.0 in inertial range
```

**Interpretation:**
- Ratio ~ 1.0: Critical balance holds ✓
- Ratio >> 1.0: Nonlinear cascade dominates (may indicate RMHD breakdown)
- Ratio << 1.0: Linear propagation dominates (weak turbulence, possibly invalid regime)

Enable with `--save-diagnostics`:
```bash
uv run python examples/alfvenic_cascade_benchmark.py --save-diagnostics
```

## Kinetic Effects

KRMHD captures kinetic physics via Hermite moment expansion in parallel velocity space.

### Landau Damping

**When it matters:**
```
k∥v_te ~ ω   (resonance condition)
```

where:
- `v_te` = electron thermal speed
- `ω` = wave frequency (~ k∥v_A for Alfvén waves)

Landau damping is significant when:
```
k∥ρₑ ~ k∥v_te/Ωₑ ~ 1
```

For typical KRMHD:
- `k∥ ~ 1-5` (low mode numbers in parallel direction)
- `ρₑ ~ 0.01-0.1` (electron gyroradius in code units)
- Product k∥ρₑ ~ 0.01-0.5 (marginal to significant damping)

**In simulations:** Hermite moment hierarchy captures this automatically. Higher moments (m > 2) are needed for strong Landau damping.

### Phase Mixing

Energy in velocity space cascades to higher Hermite moments:
```
E_m ~ m^(-3/2)   (phase mixing, collisionless)
E_m ~ m^(-1/2)   (phase unmixing, Landau resonance)
```

**Practical consideration:** Need M ≥ 4 Hermite moments to resolve kinetic effects. Default M=8 in most examples.

**Check convergence:**
```python
from krmhd.diagnostics import hermite_moment_energy

# Compute energy distribution in velocity space
E_m = hermite_moment_energy(state, grid)

# Should decay exponentially with m
# If E_m[7] / E_m[0] > 0.1, need more moments (increase M)
```

### Collisional vs Collisionless

**Collisionless regime:**
```
ν << ω   (collision frequency << wave frequency)
```

Landau damping and phase mixing dominate. This is the "kinetic" regime where KRMHD differs from fluid RMHD.

**Collisional regime:**
```
ν >> ω
```

Collisions damp high-m moments before they can develop. KRMHD reduces to fluid RMHD with an effective closure.

**In GANDALF:** Collision parameter `nu` controls this:
- `nu = 0.0`: Purely collisionless (Landau damping only)
- `nu = 0.1-1.0`: Moderate collisionality (hybrid regime)
- `nu >> 1.0`: Collisional (fluid-like behavior)

Typical choice: `nu = 0.5-1.0` (hyper-collisions with n=2).

## Valid Parameter Ranges

### Plasma Beta (β)

```
β = 8πp / B₀²
```

Ratio of thermal to magnetic pressure.

**GANDALF handles:**
- **Low-β** (β ~ 0.01-0.1): Magnetically dominated, strong guide field
- **Order unity** (β ~ 1): Comparable thermal and magnetic pressure
- **High-β** (β ~ 10-100): Thermally dominated

**Typical astrophysical:**
- Solar wind: β ~ 0.1-10
- Magnetosheath: β ~ 1-10
- Hot accretion flows: β ~ 1-100

**No formal upper limit** in KRMHD, but:
- β >> 100: Guide field becomes very weak, ε constraint harder to satisfy
- β << 0.01: Nearly force-free, slow modes become negligible

### Temperature Ratio (τ)

```
τ = Tᵢ / Tₑ
```

Ion to electron temperature ratio.

**GANDALF handles:** τ ~ 1-10

**Typical astrophysical:**
- Solar wind: τ ~ 1-3
- Tokamak edge: τ ~ 1-5
- Low-collisionality plasmas: τ can be >> 1

**Physical effects:**
- τ = 1: Electrons and ions at same temperature
- τ > 1: Ions hotter, affects Landau damping rates
- τ >> 10: May need separate ion and electron kinetic equations (beyond KRMHD)

### Resolution Requirements

Spectral methods require resolving the **inertial range** and **dissipation range**.

**Minimum for turbulence:** 64³
- Inertial range: k ~ 2-8 (~3 modes)
- Dissipation range: k ~ 8-21 (dealiasing boundary)

**Recommended for production:** 128³
- Inertial range: k ~ 2-16 (~7 modes)
- Dissipation range: k ~ 16-42

**High-fidelity:** 256³ or higher
- Inertial range: k ~ 2-32 (>10 modes)
- Dissipation range: k ~ 32-85

**Constraint:** Must have `k_max ρᵢ << 1` (spectral method valid only at scales larger than ion gyroradius).

For box size L = 2π and N points:
```
k_max = N/3 × (2π/L) = N/3   (after 2/3 dealiasing)
```

If ρᵢ ~ 0.01 in code units:
```
k_max ρᵢ = (N/3) × 0.01 ~ N/300
```

For N=128: k_max ρᵢ ~ 0.4 (marginal, KRMHD still valid)
For N=256: k_max ρᵢ ~ 0.8 (approaching breakdown, ion-scale effects)

**Rule of thumb:** Stay below 128³ unless you're sure ρᵢ is small enough.

### Dissipation Parameters

**Hyper-resistivity (η):**
- **r=2:** Stable, validated at 32³-128³ with η ~ 0.5-2.0
- **r=4:** Thesis value, narrower dissipation range but **unstable at 64³ in forced turbulence** (Issue #82)
- **Higher r:** Use with caution, overflow-safe but stability TBD

**Hyper-collisions (ν):**
- **n=2:** Recommended (matches r=2)
- **n=4:** For aggressive velocity-space damping
- Typical: ν ~ 0.5-1.0

**Overflow constraint:**
```
η·dt < 50   and   ν·dt < 50
```

Always satisfied with normalized dissipation (automatic validation in code).

See [Running Simulations](running_simulations.md) for parameter selection guidelines.

## When the Code is NOT Valid

### 1. Strong Turbulence (ε ~ 1)

**Symptom:** δB⊥/B₀ ~ 1

**Why invalid:** RMHD ordering breaks down. Need full MHD to capture:
- Magnetic reconnection
- Large-scale field reversals
- Isotropic (not anisotropic) cascade

**What to do:** Reduce initial energy or forcing amplitude to keep ε < 0.3.

### 2. Ion Gyroscale Physics (k ρᵢ ~ 1)

**Symptom:** Resolution approaching ion gyroradius scales

**Why invalid:** RMHD assumes k⊥ρᵢ << 1. At ion scales, need:
- Full gyrokinetic equations
- Finite Larmor radius (FLR) effects
- Gyro-averaging operators

**What to do:** Reduce k_max (lower resolution) or increase box size L (fewer modes).

### 3. Extreme Anisotropy (k∥ >> k⊥)

**Symptom:** Parallel gradients dominate

**Why invalid:** RMHD assumes k∥/k⊥ ~ ε << 1. If k∥ >> k⊥:
- Perpendicular cascade assumption fails
- Different turbulence regime (fiber-like modes)

**What to do:** This is rare in turbulence (critical balance prevents it). If seen, check initial conditions for numerical artifacts.

### 4. No Guide Field (B₀ → 0)

**Symptom:** Guide field too weak compared to perturbations

**Why invalid:** RMHD expansion requires strong B₀. Without it:
- No preferred direction
- Isotropic MHD turbulence
- Different cascade dynamics

**What to do:** GANDALF is not designed for this regime. Use standard MHD code.

### 5. Compressibility (δρ/ρ ~ 1)

**Symptom:** Large density fluctuations

**Why invalid:** RMHD treats density/pressure as passive scalars (slow modes don't back-react). Large compressibility requires:
- Coupled slow mode dynamics
- Sound waves
- Compressible MHD

**What to do:** GANDALF is designed for nearly incompressible regime. For compressible turbulence, use different model.

## Validity Checklist

Before trusting your simulation results:

### 1. Check Perturbation Amplitude
```python
# Compute ε ~ δB⊥/B₀
epsilon = jnp.std(A_parallel_real) * k_typical / B0
assert epsilon < 0.3, f"ε = {epsilon:.2f} too large (RMHD breakdown)"
```

### 2. Check Anisotropy
```python
# Critical balance: k∥ ~ k⊥^(2/3)
k_parallel_typical = ...  # From spectrum
k_perp_typical = ...
assert k_parallel < k_perp, "Anisotropy violated"
```

### 3. Check Resolution
```python
# Ensure k_max ρᵢ << 1
k_max = (grid.Nx // 3)  # After dealiasing
rho_i = 0.01  # Example value in code units
assert k_max * rho_i < 1.0, "Approaching ion scales"
```

### 4. Check Hermite Convergence
```python
# Energy should decay exponentially with m
E_m = hermite_moment_energy(state, grid)
ratio = E_m[-1] / E_m[0]  # Highest / lowest moment
assert ratio < 0.1, "Need more Hermite moments"
```

### 5. Check Energy Conservation (Inviscid)
```python
# For eta=0, nu=0, energy should be conserved
relative_error = abs(E_final - E_initial) / E_initial
assert relative_error < 0.01, "Energy not conserved (numerical error)"
```

### 6. Check Spectrum Slope
```python
# Should have k⊥^(-5/3) in inertial range
# Fit spectrum between k=2 and k=10
slope = fit_power_law(k_perp, E_perp, k_min=2, k_max=10)
assert -1.8 < slope < -1.5, f"Slope {slope:.2f} not k^(-5/3)"
```

## Red Flags

**Stop and investigate if you see:**

1. **Exponential energy growth** (not reaching steady state)
   - Likely: Numerical instability, not physics
   - See [Running Simulations](running_simulations.md) troubleshooting

2. **Flat spectrum** (no k^(-5/3) cascade)
   - Likely: Insufficient forcing or over-damped
   - Try reducing `eta` or increasing `force_amplitude`

3. **Spectrum pile-up at high-k**
   - Likely: Dealiasing failure or insufficient dissipation
   - Check `energy_highk` diagnostic

4. **Field line wandering δr⊥ ~ L**
   - Likely: ε too large (RMHD breakdown)
   - Reduce initial energy or forcing

5. **Hermite moments not decaying**
   - Likely: Need more moments (increase M)
   - Or collision parameter `nu` too small

6. **NaN or Inf in output**
   - Likely: Timestep too large (CFL violation)
   - Or numerical instability (reduce forcing, increase dissipation)

## Summary

**KRMHD is valid for:**
- Weak turbulence: ε ~ 0.01-0.3
- Strong guide field: B₀ >> δB⊥
- Anisotropic cascade: k⊥ >> k∥
- Sub-gyroradius scales: kρᵢ << 1
- Nearly incompressible: δρ/ρ << 1

**Check validity by:**
- Computing ε from simulation output
- Verifying k⊥^(-5/3) spectrum
- Checking Hermite moment decay
- Monitoring critical balance ratio

**If invalid:**
- Reduce energy or forcing (lower ε)
- Change resolution (stay above ion scales)
- Use different model (full MHD or gyrokinetics)

## Further Reading

- **RMHD Derivation:** Strauss (1976), Phys. Fluids 19:134
- **Critical Balance:** Goldreich & Sridhar (1995), ApJ 438:763
- **KRMHD Theory:** Schekochihin et al. (2009), ApJS 182:310
- **Thesis:** See `gandalf_thesis_chapter.pdf` in repository

For numerical validation, see:
- [Numerical Methods](numerical_methods.md) - Algorithm details
- [Running Simulations](running_simulations.md) - Examples with expected results
