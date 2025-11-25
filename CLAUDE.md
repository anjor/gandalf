# KRMHD Spectral Solver Project

## Context
You are implementing a **full Kinetic Reduced MHD (KRMHD)** solver using spectral methods for studying turbulent magnetized plasmas in astrophysical environments. This code is a modern rewrite of GANDALF (https://github.com/anjor/gandalf-original), a legacy Fortran+CUDA implementation.

**Scope:** This is a **kinetic** implementation using Hermite moment expansion in parallel velocity space (v∥). This captures Landau damping, phase mixing, and collisional effects. We are NOT implementing a simplified fluid closure.

The user is an expert plasma physicist familiar with gyrokinetics, Landau damping, and turbulence theory. No basic explanations needed.

## Physics Model

### KRMHD Hierarchy
- **Active (Alfvénic) sector**: 
  - Stream function φ (perpendicular flow)
  - Parallel vector potential A∥ (perpendicular magnetic field)
  - These drive the turbulent cascade
  
- **Passive (slow mode) sector**:
  - Parallel magnetic field perturbation δB∥
  - Electron density/pressure perturbations n_e, p_e
  - These are advected by Alfvénic turbulence but don't back-react

### Key Equations

```
∂A∥/∂t + {φ, A∥} = -∇∥φ + η∇²A∥
∂δB∥/∂t + {φ, δB∥} = -∇∥u∥ + D∇²δB∥
∂n_e/∂t + {φ, n_e} = -∇∥u∥ + Landau damping terms
```
where {f,g} = ẑ·(∇f × ∇g) is the Poisson bracket.

### Landau Closure
- Parallel electron pressure response includes kinetic effects
- Implemented via Hermite representation in v∥ or closure approximation
- Critical for proper damping at k∥v_te ~ ω

### RMHD Ordering and Validity

RMHD is an asymptotic expansion in the small parameter ε:
```
ε ~ δB⊥/B₀  << 1
```

**Ordering hierarchy:**
- δB ~ ε B₀ (magnetic perturbations, first order)
- k∥/k⊥ ~ ε (anisotropic cascade favoring k⊥ >> k∥)

**Physical consequences:**
- Field line wandering: δr⊥ ~ ε Lz (always << Lz in valid RMHD)
- Critical balance: k∥ ~ k⊥^(2/3) in turbulent cascade

### Normalization Convention

**Thesis Convention (Original GANDALF):**
- **Box size**: Lx = Ly = Lz = 1.0 (unit box). Perpendicular lengths are in units of ion
  gyroradius, parallel length is in units of a larger length scale.
- **Alfvén time**: τ_A = Lz/v_A = 1.0 (unit time)
- **Wavenumbers**: k = 2πn for integer mode number n
- **Energy**: Total energy (volume integral), not energy density

**Code Convention (This Implementation):**
- **Box size**: Can be arbitrary (Lx, Ly, Lz)
- **For Orszag-Tang benchmark**: Use Lx = Ly = Lz = 1.0 to match thesis
- **Alfvén time**: τ_A = Lz/v_A (parallel Alfvén crossing time)
- **Wavenumbers**: k = (2π/L) × n for mode number n
- **Energy**: Computed via Parseval's theorem with proper FFT normalization

**Field Normalization:**
- **Magnetic field**: B = B₀ẑ + δB where δB comes from perturbations
  - Perpendicular: δB⊥ = ∇ × (Ψ ẑ) with Ψ = (z⁺ - z⁻)/2
  - Parallel: Bz = 1 + δB∥ (state.B_parallel in Fourier space)
- **Field line following**: Requires full B field, so B₀ must be added in real space
- **Important**: Never add constants in Fourier space - only k=0 mode is affected!

## Technical Stack

### Core Dependencies
- **JAX**: Primary framework (runs on Apple Metal)
- **jax.numpy**: Array operations
- **jax.scipy.fft**: Spectral transforms
- **Pydantic**: Data validation and settings management (v2.0+)
- **h5py**: Data I/O
- **matplotlib**: Diagnostics/visualization
- **Optax**: For any optimization needs (optional)

### Numerical Methods
- **Spatial dimensions**: 3D Fourier spectral (x, y, z) with z ∥ B₀
  - Perpendicular (x,y): Standard 2D FFT with 2/3 dealiasing
  - Parallel (z): 1D FFT for ∂/∂z operations and field line following
  - Grid structure: Nz × Ny × (Nx//2+1) in Fourier space (rfft in x)
- **Parallel velocity**: Hermite polynomials (if fully kinetic) or fluid closure
- **Time stepping**: GANDALF integrating factor + RK2 (exact linear propagation, 2nd-order nonlinear)
- **Poisson solver**: k²φ = ∇²⊥A∥ in Fourier space
- **Field line following**: Requires full 3D for interpolation across z-planes

## Code Architecture

### Module Structure
```
krmhd/
├── spectral.py      # ✅ COMPLETE: FFT operations, derivatives, dealiasing (2D/3D)
├── physics.py       # ✅ COMPLETE: Poisson bracket, Elsasser RHS, Hermite moment RHS
├── timestepping.py  # ✅ COMPLETE: GANDALF integrating factor + RK2 timestepper (includes collisions)
├── hermite.py       # ✅ COMPLETE: Hermite basis for kinetic physics
├── diagnostics.py   # ✅ COMPLETE: Energy spectra (1D, k⊥, k∥), history, visualization
├── forcing.py       # ✅ COMPLETE: Gaussian white noise forcing, energy injection diagnostics (Issue #29)
├── io.py            # ✅ COMPLETE: HDF5 checkpointing and timeseries I/O (Issue #13)
└── validation.py    # Linear physics tests (Issue #10)

examples/
├── forcing_minimal.py              # ✅ COMPLETE: Minimal forcing example (~50 lines, 2s runtime)
├── driven_turbulence.py            # ✅ COMPLETE: Comprehensive driven turbulence with forcing (317 lines, 20s runtime)
├── decaying_turbulence.py          # ✅ COMPLETE: Turbulent cascade with diagnostics (Issue #12)
├── orszag_tang.py                  # ✅ COMPLETE: Orszag-Tang vortex benchmark (Issue #11)
├── alfvenic_cascade_benchmark.py   # ✅ COMPLETE: Alfvénic cascade benchmark (Thesis Section 2.6.3)
├── hyper_dissipation_demo.py       # ✅ COMPLETE: Hyper-dissipation r=1 vs r=2 comparison (Issue #28)
└── plot_checkpoint_spectrum.py     # ✅ COMPLETE: Post-processing tool for checkpoint analysis
```

### Design Principles
1. **Functional style**: Pure functions, no side effects
2. **JAX patterns**: Use vmap/scan instead of loops
3. **JIT everything**: @jax.jit on all performance-critical functions
4. **Explicit typing**: Full type hints for clarity
5. **Physics transparency**: Variable names match physics notation
6. **UV for package management**: Use uv for package management and/or running ad hoc
scripts
7. **Commit quick**: Commit logical changes quickly and often to ensure there are
checkpoints to return to.

## Implementation Guidelines

### Spectral Operations
- Maintain reality condition: `f(-k) = f*(k)`
- Apply dealiasing after EVERY nonlinear operation
- Use rfftn/irfftn for 3D real fields to save memory (rfft in x-direction only)
- Preserve ∇·B = 0 constraint (automatically satisfied with spectral methods)

### Performance Targets
- 128³ resolution should run efficiently on M1 Pro (matches original GANDALF)

### Energy Conservation
- Alfven energy E = E_magnetic + E_kinetic. Slow modes are conserved separately: E_compressive
- GANDALF formulation conserves perpendicular gradient energy exactly
- Inviscid (η=0) runs: ~0.0086% relative error (200× better than simplified formulation)
- Viscous (η>0) runs: Exponential energy decay E(t) = E₀ exp(-2ηk⊥²t)
- Monitor energy with diagnostics.EnergyHistory for tracking E(t)

### Hyper-dissipation (Issue #28)

Hyper-dissipation operators provide selective damping of small-scale modes while preserving large-scale turbulent dynamics. This is critical for preventing spectral pile-up at grid Nyquist scales without over-dissipating the inertial range.

**Physics:**
- **Standard dissipation** (r=1): -η∇²⊥ ~ -ηk⊥² (uniform damping across scales)
- **Hyper-dissipation** (r>1): -η∇^(2r)⊥ ~ -ηk⊥^(2r) (concentrated at high-k)
- **Hyper-collisions** (n>1): -νm^n for Hermite moments (damps high-m modes)
  - **IMPORTANT**: Uses m^n (not m^(2n)) to match original GANDALF alpha_m parameter
  - Spatial hyper-dissipation uses k^(2r) because ∇² ~ k², but moment collisions use m^n directly

**Normalized Dissipation (Matching Original GANDALF):**
This implementation uses **normalized hyper-dissipation** matching the original GANDALF:
- **Resistivity**: exp(-η·(k⊥²/k⊥²_max)^r·dt) instead of exp(-η·k⊥^(2r)·dt)
- **Collisions**: exp(-ν·(m/M)^n·dt) instead of exp(-ν·m^n·dt)
  - Original GANDALF: alpha_m parameter (typically 3, default 2)
  - JAX implementation: hyper_n parameter (same exponent, no factor of 2)

**Key advantages:**
- **Resolution-independent constraints**: η·dt < 50 and ν·dt < 50 (independent of N, M, r, n!)
- **Practical parameter range**: η ~ 0.5-1.0, ν ~ 0.5-1.0 (matching thesis)
- **High-order overflow-safe**: r=4, r=8 satisfy overflow constraint at ANY resolution
- **No overflow issues**: Normalized dissipation rate at k_max or M is simply the coefficient
- **IMPORTANT**: Overflow safety ≠ numerical stability. High-order (r≥4) may require
  additional parameter tuning or exhibit instabilities in forced turbulence (see Issue #82)

**Detecting Numerical Instabilities:**
When running turbulence simulations, watch for these warning signs:
1. **Monitor max field amplitude**: Track `max(|z±|)` during evolution
   - Healthy: Saturates to O(1-10) values in forced turbulence
   - Warning: Exponential growth beyond O(100) indicates instability
2. **Plot E(t) on log scale**: Detect exponential growth
   - Healthy: Steady plateau or slow linear growth in forced runs
   - Warning: Straight line on log-scale = exponential blow-up
3. **Check for NaN/Inf**: Added in alfvenic_cascade_benchmark.py:350-360
   - Terminates with diagnostic message if detected
4. **Spectrum pile-up**: Look for energy accumulating at k_max
   - Healthy: Sharp exponential cutoff from hyper-dissipation
   - Warning: Flat spectrum extending to k_max = dealiasing failure


See Diagnostics section for full details on compute_turbulence_diagnostics().

**Implementation details:**
- **Hyper-resistivity**: Applied via multiplicative factor exp(-η·(k⊥²/k⊥²_max)^r·dt) to z± in Fourier space
  - Normalization by k⊥²_max at 2/3 dealiasing boundary (matches damping_kernel.cu:50)
  - idxmax = (Nx-1)//3, idymax = (Ny-1)//3 following original GANDALF
- **Hyper-collisions**: Applied via multiplicative factor exp(-ν·(m/M)^n·dt) to Hermite moments (m≥2)
  - Normalization by M (max moment) matching timestep.cu:111 (alpha_m parameter)
  - Exponent is n (not 2n) to match original GANDALF directly
  - m=0,1 exempt (conserves particles and momentum)
- **Reality condition**: Preserved (dissipation factors are real and uniform across modes)
- **Energy conservation**: Modified to E(t) = E₀·exp(-2η⟨(k⊥²/k⊥²_max)^r⟩·t) for hyper-viscous decay
- **References**:
  - Original GANDALF: damping_kernel.cu:50 (resistivity), timestep.cu:111 (collisions)
  - This implementation: timestepping.py:366-371 (k_max calculation), :444-456 (application)

## Validation Suite

### Linear Physics Tests
1. **Alfvén wave dispersion**: ω² = k∥²v_A²
2. **Slow mode damping**: Verify passive advection without back-reaction
3. **Landau damping**: Emerges from parallel streaming in Hermite moments

### Nonlinear Benchmarks
1. **Orszag-Tang vortex**: Standard MHD test (fluid limit)
2. **Decaying turbulence**: Check k⁻⁵/³ spectrum
3. **Alfvénic cascade** (Thesis Section 2.6.3, Figure 2.2): Forced turbulence with k⊥⁻⁵/³ critical-balance spectrum
4. **Energy partition**: Verify equipartition in steady state
5. **Selective decay**: Magnetic energy should dominate at late times

### Critical Checks
- Slow modes must remain passive (no spurious coupling)
- Phase mixing in v∥ (if using Hermite representation)
- CFL condition: dt < min(dx/v_A, dx²/η)

## Common Pitfalls

1. **Hermite truncation**: Need enough moments to resolve Landau resonance
2. **Dealiasing**: Forgetting it causes spurious energy growth
3. **Parallel gradient**: ∇∥ = ik∥ only valid for periodic boundary conditions
4. **Initial conditions**: Must satisfy ∇·B = 0 and reality conditions
5. **Slow mode coupling**: Any back-reaction indicates coding error
6. **Hermitian symmetry in forcing**: When adding forcing directly in Fourier space, must explicitly enforce reality on kx=0 and kx=Nyquist planes (rfft format requirement). JAX's irfftn doesn't get called to fix violations automatically.

## Reference Parameters
Typical astrophysical parameters:
- β (plasma beta): 0.01 - 100
- τ (T_i/T_e): 1 - 10
- Resolution: 128³ to 512³ (3D spatial grid)

### Note on k∥ Independence
For straight, uniform B₀ = B₀ẑ, different k∥ modes evolve independently through the
perpendicular Poisson bracket {f,g} = ẑ·(∇⊥f × ∇⊥g). However, we use full 3D grids for:
1. **Field line following**: Interpolation across z-planes couples k∥ modes
2. **Parallel structure**: Track energy distribution in k∥ space
3. **Initial conditions**: Allow z-dependent structures (wave packets, etc.)
4. **Landau physics**: Need k∥ distribution for proper kinetic damping

## Questions for User
Track any clarifications needed during development

