# Numerical Methods

GANDALF uses **pseudo-spectral methods** in space and a **specialized integrating factor timestepper** for temporal evolution. This document explains the numerical algorithms and implementation choices.

## Overview

| Component | Method | Key Feature |
|-----------|--------|-------------|
| **Spatial discretization** | 3D Fourier spectral | Spectral accuracy (exponential convergence) |
| **Time integration** | GANDALF integrating factor + RK2 | Exact linear propagation, 2nd-order nonlinear |
| **Dealiasing** | 2/3 rule | Prevents aliasing errors in nonlinear terms |
| **Poisson solver** | Direct (Fourier space) | k²φ = -k⊥²A∥ solved exactly |
| **Dissipation** | Normalized hyper-dissipation | Resolution-independent parameters |
| **Kinetic physics** | Hermite moment expansion | Captures Landau damping, phase mixing |

## Spectral Method

### Why Spectral?

Spectral methods represent fields as sums of basis functions (Fourier modes):

```
f(x,y,z) = Σ f̂(kx,ky,kz) exp(i(kx·x + ky·y + kz·z))
```

**Advantages:**
- **Spectral accuracy:** Error decreases exponentially with resolution (vs algebraically for finite differences)
- **Exact derivatives:** ∂f/∂x = Σ (ikx) f̂(kx,ky,kz) exp(...)
- **Easy Poisson solver:** ∇²φ = ω → -k²φ̂ = ω̂ → φ̂ = -ω̂/k²
- **Clean dealiasing:** 2/3 rule exactly removes aliasing

**Disadvantages:**
- Requires periodic boundary conditions (not suitable for walls or open boundaries)
- Memory intensive (all modes stored globally)
- Load balancing on parallel machines more complex than domain decomposition

For KRMHD turbulence in periodic boxes, spectral methods are ideal.

### 3D Fourier Decomposition

GANDALF decomposes fields in all three directions:

```
f(x,y,z) → f̂(kx, ky, kz)
```

**Wavenumbers:**
```
kx = (2π/Lx) × nx,  nx = 0, 1, ..., Nx/2 (rfft format)
ky = (2π/Ly) × ny,  ny = -Ny/2, ..., Ny/2-1
kz = (2π/Lz) × nz,  nz = -Nz/2, ..., Nz/2-1
```

**Grid structure:** For Nx × Ny × Nz real-space grid:
- **Real space:** (Nz, Ny, Nx) array, real values
- **Fourier space:** (Nz, Ny, Nx//2+1) array, complex values

### rfft Format (Memory Efficiency)

Since real fields satisfy f(-k) = f*(k) (Hermitian symmetry), we only store half of the kx modes:

```python
import jax.numpy as jnp

# Forward transform (real → complex)
f_real = ...  # Shape: (Nz, Ny, Nx)
f_hat = jnp.fft.rfftn(f_real, axes=(0, 1, 2))  # Shape: (Nz, Ny, Nx//2+1)

# Inverse transform (complex → real)
f_real_recovered = jnp.fft.irfftn(f_hat, s=(Nz, Ny, Nx))
```

**Memory savings:** Factor of ~2 (store Nx//2+1 instead of Nx modes in x-direction).

**Reality condition:** Enforced automatically by `irfftn`, but must be maintained manually when:
- Adding forcing directly in Fourier space
- Modifying specific modes
- Setting initial conditions

**Special planes:**
- `kx=0` plane: Must be real (imaginary part automatically zeroed by `irfftn`)
- `kx=Nyquist` plane (if Nx even): Must also be real

### Derivative Operators

Spatial derivatives are **exact** in Fourier space:

```
∂f/∂x ↔ i·kx·f̂
∂f/∂y ↔ i·ky·f̂
∂f/∂z ↔ i·kz·f̂
```

**Implementation:**
```python
from krmhd.spectral import derivative_x, derivative_y, derivative_z

df_dx = derivative_x(f_hat, grid)  # Returns i·kx·f̂
df_dy = derivative_y(f_hat, grid)  # Returns i·ky·f̂
df_dz = derivative_z(f_hat, grid)  # Returns i·kz·f̂
```

**Laplacian:**
```python
laplacian = grid.k_squared * f_hat  # ∇²f ↔ -k²f̂
```

For perpendicular Laplacian only (KRMHD uses this):
```python
laplacian_perp = grid.k_perp_squared * f_hat  # ∇⊥²f ↔ -(kx²+ky²)f̂
```

**No numerical diffusion:** Unlike finite differences, spectral derivatives have zero numerical viscosity.

## Dealiasing (2/3 Rule)

### Why Dealiasing?

Nonlinear terms like {f,g} involve products in real space:

```
h(x) = f(x) · g(x)
```

**Problem:** If f has modes up to k_max and g has modes up to k_max, their product h has modes up to 2·k_max. On a grid with Nyquist frequency k_Nyquist = N/2, modes beyond k_Nyquist **wrap around** (aliasing) and corrupt lower modes.

**Example:**
- Grid: N=64 → k_Nyquist = 32
- Mode k=30 × Mode k=30 = Mode k=60
- But k=60 > k_Nyquist → wraps to k = 60 - 64 = -4 (aliases to k=4)

This causes spurious energy transfer and violates conservation laws.

### The 2/3 Rule

**Solution:** Keep only modes with |k| ≤ k_max = (2/3)·k_Nyquist.

For N=64:
- k_Nyquist = 32
- k_max = 21 (after dealiasing)
- Products: k=21 × k=21 = k=42 ≤ 2·k_max = 42 ✓ (no aliasing)

**Implementation:**
```python
from krmhd.spectral import SpectralGrid3D

grid = SpectralGrid3D(Nx=64, Ny=64, Nz=64, Lx=2*np.pi, Ly=2*np.pi, Lz=2*np.pi)

# Compute nonlinear term (e.g., Poisson bracket)
f_hat = ...
g_hat = ...

# Transform to real space, multiply, transform back
f_real = jnp.fft.irfftn(f_hat, s=(grid.Nz, grid.Ny, grid.Nx))
g_real = jnp.fft.irfftn(g_hat, s=(grid.Nz, grid.Ny, grid.Nx))
product_real = f_real * g_real
product_hat = jnp.fft.rfftn(product_real)

# CRITICAL: Apply dealiasing mask
product_hat_dealiased = grid.dealias(product_hat)
```

**Dealiasing mask:** Sets to zero all modes with:
```
|kx| > (2/3)·(Nx/2)  OR  |ky| > (2/3)·(Ny/2)  OR  |kz| > (2/3)·(Nz/2)
```

**Performance:** Dealiasing is cheap (just a mask multiplication), but FFTs dominate cost.

**Importance:** Without dealiasing, energy grows spuriously and simulations blow up.

## Poisson Solver

KRMHD requires solving for the stream function φ from the vorticity:

```
∇⊥²φ = ω = -∇⊥²A∥
```

In Fourier space, this is trivial:

```
-k⊥² φ̂ = -k⊥² Â∥
→ φ̂ = Â∥
```

But more generally, for ∇²φ = ω:

```
-k² φ̂ = ω̂
→ φ̂ = -ω̂ / k²
```

**Singularity at k=0:** The k=0 mode represents the spatial mean. For Poisson equation, the mean is arbitrary (set to zero):

```python
def poisson_solve_3d(omega_hat, grid):
    """Solve ∇²φ = ω for φ."""
    phi_hat = jnp.where(
        grid.k_squared > 0,
        -omega_hat / grid.k_squared,
        0.0  # Set k=0 mode to zero (mean = 0)
    )
    return phi_hat
```

**No iteration needed:** Direct solve in one step (advantage of spectral methods).

**In KRMHD:** Elsasser formulation eliminates need for Poisson solve in timestepping. But diagnostics (computing φ from z±) use it:

```python
from krmhd.physics import elsasser_to_physical

phi, A_parallel = elsasser_to_physical(z_plus, z_minus)
# Uses φ = (z⁺ + z⁻)/2 and A∥ = (z⁺ - z⁻)/2 (no Poisson solve!)
```

## Time Integration: GANDALF Method

### The Challenge

KRMHD equations have both **linear** and **nonlinear** terms:

```
∂z±/∂t = ±i·kz·z± + N±[z±,z∓] - dissipation
         \_______/   \__________/
          Linear      Nonlinear
```

**Linear term:** Represents Alfvén wave propagation (oscillatory, frequency ω = kz·v_A)

**Problem with standard RK methods:**
- Standard RK4 requires Δt < 2π/ω for stability (very small timesteps!)
- For kz=10 and v_A=1: Δt < 0.6
- Wasteful for slow nonlinear dynamics (τ_nl ~ 1/k⊥v⊥ >> τ_A)

### Integrating Factor Method

**Idea:** Treat linear term **exactly**, only approximate nonlinear term.

**Derivation:** Rewrite equation as:
```
∂z±/∂t = ±i·kz·z± + N±
```

Multiply by integrating factor exp(∓i·kz·t):
```
∂/∂t [z± · exp(∓i·kz·t)] = N± · exp(∓i·kz·t)
```

**Solution:**
```
z±(t+Δt) = exp(±i·kz·Δt) · [z±(t) + ∫₀^Δt N±(t') · exp(∓i·kz·t') dt']
```

**Interpretation:**
- Factor `exp(±i·kz·Δt)` propagates linear waves **exactly** (no stability restriction!)
- Integral of N± is approximated (e.g., with RK2)

### RK2 (Midpoint Method)

For the nonlinear integral, use 2nd-order Runge-Kutta:

**Note:** The code below is simplified pseudocode for illustration. The actual implementation in `timestepping.py` includes additional details (Hermite moments, collision operators, validation). See source code for complete implementation.

```python
def gandalf_step(state, dt, eta, nu, hyper_r=2, hyper_n=2, force_z_plus=None, force_z_minus=None):
    """Single timestep with GANDALF integrating factor + RK2 (simplified pseudocode)."""

    # 1. Compute RHS at t (stage 1)
    k1_plus = z_plus_rhs(state, grid) + forcing_plus
    k1_minus = z_minus_rhs(state, grid) + forcing_minus

    # 2. Midpoint state (evolve by dt/2 with integrating factor)
    z_plus_mid = apply_integrating_factor(state.z_plus + 0.5*dt*k1_plus, grid, dt/2)
    z_minus_mid = apply_integrating_factor(state.z_minus + 0.5*dt*k1_minus, grid, -dt/2)

    state_mid = KRMHDState(z_plus=z_plus_mid, z_minus=z_minus_mid, ...)

    # 3. Compute RHS at t+dt/2 (stage 2)
    k2_plus = z_plus_rhs(state_mid, grid) + forcing_plus
    k2_minus = z_minus_rhs(state_mid, grid) + forcing_minus

    # 4. Full step (evolve by dt with integrating factor)
    z_plus_new = apply_integrating_factor(state.z_plus + dt*k2_plus, grid, dt)
    z_minus_new = apply_integrating_factor(state.z_minus + dt*k2_minus, grid, -dt)

    # 5. Apply dissipation (exponential factors)
    z_plus_new *= exp(-η·(k⊥²/k⊥²_max)^r·dt)
    z_minus_new *= exp(-η·(k⊥²/k⊥²_max)^r·dt)

    return KRMHDState(z_plus=z_plus_new, z_minus=z_minus_new, ...)
```

**Accuracy:** 2nd-order in time for nonlinear terms, exact for linear propagation.

**Stability:** CFL condition based on nonlinear advection, NOT wave propagation.

### Integrating Factor Application

```python
def apply_integrating_factor(field, grid, dt, v_A=1.0):
    """Multiply by exp(i·kz·dt) for z⁺ or exp(-i·kz·dt) for z⁻."""
    phase = 1j * grid.kz * (v_A * dt)
    return field * jnp.exp(phase)
```

**Key insight:** This is just a phase shift in Fourier space (cheap operation).

**For z⁺:** exp(+i·kz·v_A·dt) (propagates in +ẑ direction)
**For z⁻:** exp(-i·kz·v_A·dt) (propagates in -ẑ direction)

### Why "GANDALF"?

This method was developed in the original Fortran+CUDA GANDALF code and is essential for efficient KRMHD simulations. Without it, timesteps would be 10-100× smaller.

## Hyper-Dissipation

### Motivation

At high resolution, energy cascades to small scales and accumulates at the grid Nyquist scale (spectral pile-up). Need dissipation to remove this energy.

**Standard dissipation** (η∇²):
- Damps all scales uniformly
- Removes energy from inertial range (bad!)

**Hyper-dissipation** (η∇^(2r) with r>1):
- Concentrates dissipation at high-k
- Preserves inertial range dynamics

### Normalized Formulation

**Key innovation:** Normalize by k_max to make parameters resolution-independent.

**Resistivity:**
```
exp(-η · (k⊥²/k⊥²_max)^r · dt)
```

where k⊥²_max is evaluated at the 2/3 dealiasing boundary:
```python
idxmax = (grid.Nx - 1) // 3
idymax = (grid.Ny - 1) // 3
k_perp_max_squared = grid.kx[idxmax]**2 + grid.ky[idymax]**2
```

**Advantages:**
- **Resolution-independent constraint:** η·dt < 50 (same for 64³, 128³, 256³!)
- **Practical range:** η ~ 0.5-2.0 (matching thesis)
- **High-order safe:** r=4, r=8 satisfy overflow constraint at ANY resolution

**Hyper-collisions** (for Hermite moments):
```
exp(-ν · (m/M)^(2n) · dt)
```

Normalized by M (maximum moment index).

### Parameter Selection

| Order | Description | Stability | Use case |
|-------|-------------|-----------|----------|
| r=1 | Standard dissipation | ✅ Stable | Reference, linear tests |
| r=2 | Moderate hyper | ✅ Stable (validated 32³-128³) | **Recommended** for production |
| r=4 | Thesis value | ⚠️ Unstable at 64³ forced turbulence | Use with caution (Issue #82) |
| r=8 | Maximum thesis | ❓ Stability TBD | Experimental |

**Overflow constraint:**
```
η·dt < 50   (safe)
η·dt < 20   (conservative, warning issued)
```

Code automatically validates and throws `ValueError` if violated.

**Typical values:**
- 32³: η=1.0, r=2
- 64³: η=20.0, r=2 (anomalous, under investigation) OR η=2.0 with amplitude=0.01
- 128³: η=2.0, r=2

See [Running Simulations](running_simulations.md) for forced turbulence parameter selection.

### Implementation

Applied as multiplicative factor after timestepping:

```python
# Compute dissipation factor
factor = jnp.exp(-eta * (grid.k_perp_squared / k_perp_max_squared)**hyper_r * dt)

# Apply to Elsasser variables
z_plus_new = z_plus_new * factor
z_minus_new = z_minus_new * factor

# Same for Hermite moments (with m/M normalization)
```

**Energy decay:**
```
E(t) = E₀ · exp(-2η⟨(k⊥²/k⊥²_max)^r⟩·t)
```

where ⟨·⟩ is spectrum-weighted average.

## Hermite Moment Expansion

### Velocity-Space Representation

KRMHD captures kinetic physics by expanding the distribution function in **Hermite polynomials**:

```
g(x,y,z,v∥,t) = Σ gₘ(x,y,z,t) · Hₘ(v∥/v_th)
```

where Hₘ are **orthonormal Hermite functions** (quantum harmonic oscillator eigenfunctions):

```
Hₘ(ξ) = (π^(1/4) √(2^m m!))^(-1) · exp(-ξ²/2) · Hₘ(ξ)
```

**Hermite polynomials:**
- H₀ = 1
- H₁ = 2ξ
- H₂ = 4ξ² - 2
- H₃ = 8ξ³ - 12ξ
- Recurrence: Hₘ₊₁ = 2ξHₘ - 2m·Hₘ₋₁

**Moments gₘ:** Fourier-space coefficients of velocity-space expansion.

### Moment Hierarchy

Evolution equations couple moments:

```
∂g₀/∂t = {φ,g₀} - √2·k∥·Re[g₁]

∂g₁/∂t = {φ,g₁} + (1-1/Λ)·∇∥φ·g₀ - k∥·√2·g₀ - √2·k∥·g₂ + field line advection

∂gₘ/∂t = {φ,gₘ} - k∥·√(m)·gₘ₋₁ - k∥·√(m+1)·gₘ₊₁ + field line advection  (m ≥ 2)
```

**Coupling:**
- Perpendicular advection: {φ,gₘ} (Poisson bracket)
- Parallel streaming: k∥·√m·gₘ₋₁ + k∥·√(m+1)·gₘ₊₁ (wave propagation in v∥)
- Field line advection: Along curved field lines (mixing of k∥)

**Truncation:** In practice, keep M moments (e.g., M=8):
- m=0: Density perturbation
- m=1: Parallel flow
- m ≥ 2: Higher velocity moments (phase mixing, Landau damping)

**Closure:** Assume gₘ₊₁ = 0 (simplest) or use more sophisticated closures.

### Collisions

Lenard-Bernstein collision operator:

```
C[gₘ] = -νₘ·gₘ
```

with νₘ = ν·m (m-dependent damping rate).

**Conservation:**
- m=0 (particles): ν₀ = 0 (conserved)
- m=1 (momentum): ν₁ = 0 (conserved)
- m ≥ 2: Damped exponentially

**Hyper-collisions:** Generalize to νₘ = ν·(m/M)^(2n) with n ≥ 1.

**Implementation:** Applied as exponential factor (same as resistivity):

```python
# Only damp m ≥ 2
for m in range(2, M+1):
    damping_factor = jnp.exp(-nu * (m/M)**(2*hyper_n) * dt)
    g_moments[m] = g_moments[m] * damping_factor
```

### Energy in Velocity Space

Energy distributed across moments:

```
E_m = ∫ |gₘ(k)|² dk
```

**Expected behavior:**
- Phase mixing (collisionless): E_m ~ m^(-3/2)
- Phase unmixing (Landau resonance): E_m ~ m^(-1/2)

**Convergence check:**
```python
from krmhd.diagnostics import hermite_moment_energy

E_m = hermite_moment_energy(state, grid)
ratio = E_m[-1] / E_m[0]  # Should be < 0.1 for converged
```

If ratio > 0.1, increase M (more moments needed).

## CFL Condition and Timestep Selection

### CFL Constraint

For explicit timestepping, the **Courant-Friedrichs-Lewy (CFL)** condition must be satisfied:

```
CFL = max(|v⊥|) · dt / dx < 1
```

where:
- `|v⊥|` = maximum perpendicular velocity
- `dt` = timestep
- `dx` = grid spacing (smallest of dx, dy, dz)

**Physical meaning:** Fluid should not cross more than one grid cell per timestep.

**For KRMHD:** Perpendicular velocity comes from stream function:
```
v⊥ ~ |∇φ| ~ k⊥|φ|
```

### Automatic Timestep Calculation

```python
from krmhd.timestepping import compute_cfl_timestep

dt_safe = compute_cfl_timestep(state, grid, cfl_safety_factor=0.5)
```

**Algorithm:**
1. Compute φ = (z⁺ + z⁻)/2
2. Estimate max velocity: v_max ~ k_typical · max(|φ|)
3. Compute dx_min = min(Lx/Nx, Ly/Ny, Lz/Nz)
4. Return dt = cfl_safety_factor · dx_min / v_max

**Safety factor:** Typically 0.5 (conservative) to 0.8 (aggressive).

**Note:** GANDALF integrating factor removes linear wave CFL constraint (only nonlinear advection matters).

### Typical Timesteps

| Resolution | Box size | Typical dt | Steps per τ_A |
|------------|----------|-----------|---------------|
| 32³ | 2π | 0.01-0.02 | 50-100 |
| 64³ | 2π | 0.005-0.01 | 100-200 |
| 128³ | 2π | 0.0025-0.005 | 200-400 |

For forced turbulence with moderate amplitude, dt ~ 0.005 is typical.

## Memory and Performance

### Memory Scaling

Storage for one complex 3D field:
```
Memory = Nz × Ny × (Nx//2+1) × 16 bytes (complex128)
```

**Examples:**
- 64³: 64 × 64 × 33 × 16 = 2.1 MB per field
- 128³: 128 × 128 × 65 × 16 = 17 MB per field
- 256³: 256 × 256 × 129 × 16 = 135 MB per field

**Total state:**
- 2 Elsasser variables (z±): 2 × 17 MB = 34 MB (128³)
- 1 slow mode (B∥): 17 MB
- M=8 Hermite moments: 8 × 17 MB = 136 MB
- **Total ~ 187 MB** for 128³ (fits in GPU memory easily)

For 256³: ~ 1.5 GB (still manageable on modern GPUs).

### Computational Cost

**FFTs dominate:** Each timestep requires:
- ~6 FFTs for Elsasser RHS (forward + backward for each Poisson bracket term)
- ~3M FFTs for Hermite moments
- Total: ~30 FFTs/step for M=8

**FFT cost:** O(N³ log N) per 3D FFT

**Rough scaling:**
- 64³: ~0.01 s/step (M1 Pro)
- 128³: ~0.1 s/step (8× slower, expect 8³/64³ × log(128)/log(64) ≈ 9×)
- 256³: ~1 s/step (estimated)

**For 50 τ_A at 128³ with dt=0.005:**
- Steps: 50 / 0.005 = 10,000
- Time: 10,000 × 0.1 s = 1000 s ≈ 17 minutes ✓ (matches observed)

### Optimization Tips

1. **Use JIT compilation:**
   ```python
   from jax import jit

   @jit
   def step_function(state):
       return gandalf_step(state, dt=0.005, eta=1.0)
   ```

2. **Reduce resolution for testing:**
   - 32³ for quick tests (seconds)
   - 64³ for parameter scans (minutes)
   - 128³+ for production (hours)

3. **Use float32 instead of float64:**
   ```bash
   JAX_ENABLE_X64=0 uv run python examples/forcing_minimal.py
   ```
   Saves memory and ~2× faster, but less accurate.

4. **Profile to find bottlenecks:**
   ```python
   import jax
   jax.profiler.start_trace("/tmp/jax_trace")
   # Run simulation
   jax.profiler.stop_trace()
   ```

## Validation and Testing

GANDALF includes extensive tests (448+ passing):

```bash
# Run all tests
uv run pytest tests/

# Specific modules
uv run pytest tests/test_spectral.py      # Spectral operations
uv run pytest tests/test_physics.py       # Poisson bracket, RHS
uv run pytest tests/test_timestepping.py  # Integrating factor, convergence
uv run pytest tests/test_hermite.py       # Hermite basis
uv run pytest tests/test_diagnostics.py   # Spectra, energy
```

**Key validation tests:**
1. **Derivative accuracy:** Compare spectral derivatives to analytical
2. **Dealiasing effectiveness:** Check energy conservation without/with dealiasing
3. **Alfvén wave dispersion:** ω = k∥v_A (linear physics)
4. **Energy conservation:** <0.01% error for inviscid runs
5. **Convergence order:** 2nd-order in time for RK2
6. **Hermite orthogonality:** ∫ Hₘ Hₙ = δₘₙ

**Benchmark examples:**
- `orszag_tang.py`: Nonlinear MHD benchmark
- `alfvenic_cascade_benchmark.py`: k⊥^(-5/3) turbulent spectrum
- `kinetic_fdt_validation.py`: Landau damping rates

## Summary

| Component | Implementation | Key Parameters |
|-----------|---------------|----------------|
| **Spatial** | 3D Fourier (rfft) | Nx, Ny, Nz, Lx, Ly, Lz |
| **Dealiasing** | 2/3 rule | k_max = (2/3)·k_Nyquist |
| **Time** | GANDALF + RK2 | dt (CFL-limited) |
| **Dissipation** | Normalized hyper | η, r (typically η=1-2, r=2) |
| **Kinetic** | Hermite moments | M (typically 8), ν, n |
| **Poisson** | Direct (k-space) | Exact solve |

**Strengths:**
- Spectral accuracy
- Exact linear propagation
- Resolution-independent parameters
- Energy conservation (inviscid limit)

**Limitations:**
- Periodic boundaries only
- Memory intensive (global modes)
- CFL constraint on nonlinear terms

## Further Reading

- **Spectral Methods:** Boyd (2001), "Chebyshev and Fourier Spectral Methods"
- **Integrating Factor:** Cox & Matthews (2002), J. Comp. Phys. 176:430
- **GANDALF Method:** Original thesis (see `gandalf_thesis_chapter.pdf`)
- **Hermite Expansion:** Grad (1949), Comm. Pure Appl. Math. 2:331

For practical usage:
- [Running Simulations](running_simulations.md) - Examples and workflows
- [Parameter Scans](parameter_scans.md) - Systematic studies
- [Physics Validity](physics_validity.md) - When to trust results
