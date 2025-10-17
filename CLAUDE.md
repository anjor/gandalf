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
ε ~ δB⊥/B₀ ~ ρᵢ/L  << 1
```

**Ordering hierarchy:**
- δB ~ ε B₀ (magnetic perturbations, first order)
- k∥/k⊥ ~ ε (anisotropic cascade favoring k⊥ >> k∥)

**Validity constraints:**
- **MUST HAVE**: ε << 1 for RMHD to be valid
- **Typical range**: ε ~ 0.01–0.3
  - ε ~ 0.01: Very weak perturbations, nearly straight field lines
  - ε ~ 0.1: Moderate perturbations, visible field line wandering
  - ε ~ 0.3: Strong perturbations, approaching RMHD breakdown
- **Beyond ε ~ 0.5**: RMHD ordering breaks down, need full MHD
- **k_max ρᵢ << 1**: Valid only at scales larger than ion gyroradius

**Physical consequences:**
- Field line wandering: δr⊥ ~ ε Lz (always << Lz in valid RMHD)
- Critical balance: k∥ ~ k⊥^(2/3) in turbulent cascade

**IMPORTANT**: Code comments or documentation referring to "strong turbulence" with
δB⊥/B₀ ~ 1 are **physically incorrect** - such regimes violate RMHD validity.

### Normalization Convention
- **Mean field**: B₀ = 1 in code units (normalized Alfvén velocity v_A = 1)
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
- **Time stepping**: RK4 or RK45 adaptive
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
├── io.py           # HDF5 checkpointing (Issue #13)
└── validation.py    # Linear physics tests (Issue #10)

examples/
├── forcing_minimal.py          # ✅ COMPLETE: Minimal forcing example (~50 lines, 2s runtime)
├── driven_turbulence.py        # ✅ COMPLETE: Comprehensive driven turbulence with forcing (317 lines, 20s runtime)
├── decaying_turbulence.py      # ✅ COMPLETE: Turbulent cascade with diagnostics (Issue #12)
├── orszag_tang.py              # ✅ COMPLETE: Orszag-Tang vortex benchmark (Issue #11)
└── hyper_dissipation_demo.py   # ✅ COMPLETE: Hyper-dissipation r=1 vs r=2 comparison (Issue #28)
```

**spectral.py** includes:
- SpectralGrid2D/3D: Pydantic models with validation and pre-computed wavenumbers
- SpectralField2D/3D: Lazy evaluation with caching
- Derivative operators: derivative_x, derivative_y, derivative_z
- Laplacian: Full 3D or perpendicular-only
- Dealiasing: 2/3 rule implementation
- 50 comprehensive tests validating all functionality

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
- 256³ should be feasible for production runs on M1 Pro
- 512³ or higher requires HPC/cluster resources
- Memory estimate: ~270 MB per complex field at 256³ resolution

### Energy Conservation
- Total energy E = E_magnetic + E_kinetic + E_compressive
- GANDALF formulation conserves perpendicular gradient energy exactly
- Inviscid (η=0) runs: ~0.0086% relative error (200× better than simplified formulation)
- Viscous (η>0) runs: Exponential energy decay E(t) = E₀ exp(-2ηk⊥²t)
- Monitor energy with diagnostics.EnergyHistory for tracking E(t)

**✅ Issue #44 RESOLVED:** Implemented correct GANDALF energy-conserving formulation from thesis. The key insight is using k⊥⁻¹[{z⁺,-k⊥²z⁻} + {z⁻,-k⊥²z⁺} + k⊥²{z⁺,z⁻}] instead of simple -k⊥²{z∓,z±}. This conserves perpendicular gradient energy to machine precision.

### Hyper-dissipation (Issue #28)

Hyper-dissipation operators provide selective damping of small-scale modes while preserving large-scale turbulent dynamics. This is critical for preventing spectral pile-up at grid Nyquist scales without over-dissipating the inertial range.

**Physics:**
- **Standard dissipation** (r=1): -η∇²⊥ ~ -ηk⊥² (uniform damping across scales)
- **Hyper-dissipation** (r>1): -η∇^(2r)⊥ ~ -ηk⊥^(2r) (concentrated at high-k)
- **Hyper-collisions** (n>1): -νm^(2n) for Hermite moments (damps high-m modes)

**Recommended parameters:**
- **r = 2, n = 2**: Optimal balance for typical turbulence studies
  - Creates steep exponential cutoff at k⊥ > k_max/2
  - Preserves inertial range (k⊥ ~ 2-10) with minimal artificial damping
  - Prevents spurious reflections from grid Nyquist boundary
- **r = 1, n = 1**: Standard dissipation (reference case)
- **r ≥ 3, n ≥ 3**: Generally impractical (requires tiny coefficients)

**Parameter selection constraints:**
- **Overflow safety**: Dissipation rate must satisfy `η·k_max^(2r)·dt < 50` (see timestepping.py:51-61)
- **Practical limits** for typical grids (k_max ~ 30-60):
  ```
  r=1 (standard):     η < 0.01          (no overflow risk)
  r=2 (recommended):  η < 1e-5 to 1e-4  (safe range)
  r=3:                η < 1e-11         (impractically small!)
  r=8:                η < 1e-29         (unusable)
  ```
- **Collision parameters**: Similar constraints apply for ν with M (max Hermite moment)
- **Timestep dependence**: Smaller dt allows larger η, ν (rate·dt is the critical factor)

**Usage in gandalf_step():**
```python
state = gandalf_step(
    state,
    dt=0.01,
    eta=1e-4,     # Hyper-resistivity coefficient
    nu=1e-4,      # Hyper-collision coefficient
    v_A=1.0,
    hyper_r=2,    # Hyper-resistivity order (default: 1)
    hyper_n=2     # Hyper-collision order (default: 1)
)
```

**Automatic validation:**
- `ValueError` raised if `rate·dt ≥ 50` (overflow risk)
- `RuntimeWarning` issued if `rate·dt ≥ 20` (moderate risk)
- Error messages provide safe parameter recommendations

**When to use:**
- **Turbulence studies**: Essential for 256³ and higher resolutions
- **Spectral pile-up**: When energy accumulates at high-k Nyquist boundary
- **Long-time evolution**: Prevents small-scale instabilities in multi-τ simulations
- **Clean inertial range**: When you need minimal artificial dissipation at k ~ 2-10

**When NOT to use:**
- **Linear physics tests**: Use standard dissipation (r=1, n=1) for analytical comparison
- **Coarse grids**: For Nx < 64, standard dissipation may be sufficient
- **Dissipation studies**: When you're specifically investigating viscous effects

**Example:**
See `examples/hyper_dissipation_demo.py` for side-by-side comparison of r=1 vs r=2, showing:
- Energy preservation: 83.6% decay (r=1) vs 24.3% decay (r=2)
- Spectral behavior: Steep exponential cutoff at high-k for r=2
- Large-scale dynamics: Inertial range (k ~ 2-5) preserved with hyper-dissipation

**Implementation details:**
- Hyper-resistivity: Applied via multiplicative factor exp(-η·k⊥^(2r)·dt) to z± in Fourier space
- Hyper-collisions: Applied via multiplicative factor exp(-ν·m^(2n)·dt) to Hermite moments
- Reality condition: Preserved (dissipation factors are real and uniform across modes)
- Energy conservation: Modified to E(t) = E₀·exp(-2η⟨k⊥^(2r)⟩·t) for hyper-viscous decay

**References:**
- Implementation: physics.py (hyperdiffusion, hypercollision functions)
- Integration: timestepping.py (gandalf_step with overflow validation)
- Tests: test_physics.py (unit tests), test_timestepping.py (integration tests)
- Example: examples/hyper_dissipation_demo.py

## Validation Suite

### Linear Physics Tests
1. **Alfvén wave dispersion**: ω² = k∥²v_A²
2. **Kinetic Alfvén waves**: Include FLR corrections if β ~ 1
3. **Slow mode damping**: Verify passive advection without back-reaction
4. **Landau damping**: Compare damping rates with analytical theory

### Nonlinear Benchmarks
1. **Orszag-Tang vortex**: Standard MHD test (fluid limit)
2. **Decaying turbulence**: Check k⁻⁵/³ spectrum
3. **Energy partition**: Verify equipartition in steady state
4. **Selective decay**: Magnetic energy should dominate at late times

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

## Current Development Status

**Test Coverage:** 275 passing tests across all modules

### Completed (Issues #1-3, #21)
- [x] Basic spectral infrastructure (2D/3D, Issue #2)
  - SpectralGrid2D/3D with Pydantic validation
  - SpectralField2D/3D with lazy evaluation
  - All derivative operators (x, y, z) and Laplacian
  - 2/3 dealiasing implementation
  - Memory-efficient rfft2/rfftn transforms
- [x] Poisson solver (Issue #3)
  - 2D solver: poisson_solve_2d() for ∇²φ = ω
  - 3D solver: poisson_solve_3d() with perpendicular/full options
  - Handles k=0 mode correctly (set to zero)
  - 12 comprehensive tests with manufactured solutions
- [x] Hermite basis infrastructure (Issue #21)
  - Orthonormal Hermite functions (quantum harmonic oscillator eigenfunctions)
  - Moment projection: distribution_to_moments()
  - Distribution reconstruction: moments_to_distribution()
  - Orthogonality verification utilities

### Fluid RMHD Core (Issues #4-8) ✅ COMPLETE
- [x] Poisson bracket implementation (Issue #4) ✅
  - poisson_bracket_2d() and poisson_bracket_3d() in physics.py
  - Computes {f,g} = ẑ·(∇f × ∇g) with spectral derivatives
  - 2/3 dealiasing applied, JIT-compiled with static args
  - 12 comprehensive tests: analytical solutions, symmetry, linearity
- [x] KRMHD state and initialization (Issue #5) ✅
  - KRMHDState dataclass with fluid + kinetic fields (z±, B∥, g moments)
  - initialize_hermite_moments() for velocity-space distribution
  - initialize_alfven_wave() for linear wave tests
  - initialize_kinetic_alfven_wave() for kinetic response
  - initialize_random_spectrum() for turbulent IC with k^-α power law
  - energy() function for diagnostics (E_magnetic, E_kinetic, E_compressive)
  - 10 comprehensive tests: state validation, initialization, energy calculation
- [x] Alfvén dynamics (Issue #6, #44, #48) ✅
  - Elsasser variable formulation: z± = φ ± A∥
  - physical_to_elsasser() and elsasser_to_physical() conversions
  - z_plus_rhs() and z_minus_rhs() using GANDALF energy-conserving formulation
  - Energy conservation: 0.0086% error (Issue #44 resolved!)
  - 20+ comprehensive tests including:
    * Elsasser conversion (roundtrip, pure waves, reality condition)
    * Linear Alfvén wave dispersion (validates ω = k∥v_A)
    * Energy conservation (validates inviscid limit)
- [x] Time integration (Issue #8, #47) ✅
  - GANDALF integrating factor + RK2 timestepper (gandalf_step)
  - Integrating factor e^(±ikz·t) handles linear propagation exactly
  - RK2 (midpoint method) for nonlinear terms (2nd-order accurate)
  - Exponential dissipation factors for η and ν
  - CFL timestep calculator: compute_cfl_timestep()
  - 12 comprehensive tests: convergence, wave propagation, dissipation
- [ ] Passive scalar evolution (Issue #7 - deferred, B∥ is legacy field)

### Kinetic Physics (Issues #22-24, #49) ✅ COMPLETE
- [x] Hermite moment hierarchy (Issue #22, #49) ✅
  - g0_rhs(), g1_rhs(), gm_rhs() implement thesis Eqs. 2.7-2.9
  - Full coupling: perpendicular advection + parallel streaming + field line advection
  - Hermite recurrence relations for m ≥ 2
  - Integrated into timestepping with collision damping
  - 10+ comprehensive tests: shape, coupling, collisions
- [x] Collision operators (Issue #23) ✅
  - Lenard-Bernstein collision operator: C[gm] = -νmgm (thesis Eq. 2.5)
  - Exponential damping: gm → gm × exp(-νm·δt)
  - Conservation: m=0 (particles) and m=1 (momentum) exempt
  - Vectorized implementation in timestepping.py:402-418
  - Integrated into gandalf_step() function
- [x] Hermite closures (Issue #24, partial) ✅
  - Truncation closure for highest moment: gM+1 = 0 (implicit default)
  - Kinetic parameter Λ in g1_rhs: (1-1/Λ) coupling factor
  - Ready for advanced closures (gM+1 = gM-1 for better convergence)
  - Utility functions (closure_zero, check_hermite_convergence) deferred

### Diagnostics (Issues #9, #25-26)
- [x] Basic diagnostics - energy, spectra (Issue #9) ✅
  - energy_spectrum_1d(): Spherically-averaged E(|k|)
  - energy_spectrum_perpendicular(): E(k⊥) for perpendicular cascade
  - energy_spectrum_parallel(): E(k∥) for field-line structure (kz-based, not true field line following)
  - EnergyHistory: Track E(t), magnetic fraction, dissipation rate
  - Visualization: plot_state(), plot_energy_history(), plot_energy_spectrum()
  - Examples: decaying_turbulence.py and orszag_tang.py demonstrate full workflow
  - 23 comprehensive tests: normalization, single modes, zero states
- [x] Field line following - k∥ spectra (Issue #25, partial) ✅
  - **Infrastructure complete**: Spectral interpolation via FFT padding (2× resolution)
  - follow_field_line(): RK2 integration with spectral-accurate interpolation
  - compute_magnetic_field_components(): B = (Bx, By, Bz) on fine grid
  - plot_field_lines(): 3D visualization of field line wandering
  - Example: field_line_visualization.py demonstrates all functionality
  - **Key improvement**: Spectral interpolation (vs bilinear in original GANDALF)
  - **TODO**: True k∥ spectrum via FFT along curved field lines (deferred)
- [x] Phase mixing diagnostics (Issue #26) ✅
  - hermite_flux(): Compute Γₘ,ₖ = -k∥·√(2(m+1))·Im[gₘ₊₁·g*ₘ] energy flux between moments
  - hermite_moment_energy(): Energy distribution E_m vs m in velocity space
  - phase_mixing_energy(), phase_unmixing_energy(): Decompose modes by flux direction
  - Visualization: plot_hermite_flux_spectrum(), plot_hermite_moment_energy(), plot_phase_mixing_2d()
  - 16 comprehensive tests: shape, reality, conservation, energy decomposition
  - **Ready for**: Issue #27 (Kinetic FDT validation)

### Forcing Mechanisms (Issue #29) ✅ COMPLETE
- [x] Gaussian white noise forcing (Issue #29) ✅
  - gaussian_white_noise_fourier(): Band-limited stochastic forcing with δ-correlated time statistics
  - force_alfven_modes(): Forces z⁺=z⁻ identically (drives u⊥ only, not B⊥)
  - force_slow_modes(): Independent forcing for δB∥
  - compute_energy_injection_rate(): Energy diagnostics for balance validation
  - **Critical physics**: z⁺=z⁻ forcing drives φ (flow) only, prevents spurious magnetic reconnection
  - **Hermitian symmetry**: Explicit enforcement for rfft format (kx=0 and kx=Nyquist planes must be real)
  - **White noise scaling**: amplitude/√dt for time-independent energy injection
  - 28 comprehensive tests: reality condition, Hermitian symmetry, energy injection, input validation
  - Examples: forcing_minimal.py (50 lines, 2s) and driven_turbulence.py (317 lines, 20s)

### Validation (Issues #10-12, #27)
- [x] Linear physics tests (Issue #10, partial) ✅
  - Alfvén wave dispersion tests (test_physics.py:1498)
  - Wave frequency validation (test_timestepping.py:334)
  - Energy conservation tests
  - Kinetic Alfvén waves and Landau damping still needed
- [x] Orszag-Tang vortex (Issue #11) ✅
  - Full implementation in examples/orszag_tang.py
  - Incompressible RMHD adaptation
  - Tests nonlinear dynamics and current sheet formation
  - Validates spectral method accuracy
- [x] Decaying turbulence (Issue #12) ✅
  - Full implementation in examples/decaying_turbulence.py
  - k^(-5/3) turbulent spectrum initialization
  - Energy history tracking and selective decay
  - Spectrum analysis with visualization
- [x] Kinetic FDT validation (Issue #27) ✅
  - Infrastructure complete: Force single k-modes, measure |gₘ|² spectrum, time-average
  - 3 passing tests: Basic validation, parameter scaling, energy balance
  - Example script: examples/kinetic_fdt_validation.py demonstrates workflow
  - **Status**: Placeholder analytical models (simplified power-law + exp cutoff)
  - **TODO**: Implement exact thesis equations (Eqs 3.37, 3.58) with plasma dispersion functions
  - **Physics validated**: Spectrum decays exponentially with m (collisional damping works)

### Production Features (Issues #13-15, #28, #30)
- [ ] HDF5 I/O (Issue #13)
- [ ] Hyper-dissipation (Issue #28)
- [ ] Integrating factor timestepper (Issue #30, optional)
- [ ] Configuration files and run scripts (Issue #15)

## Reference Parameters
Typical astrophysical parameters:
- β (plasma beta): 0.01 - 100
- τ (T_i/T_e): 1 - 10
- Resolution: 128³ to 512³ (3D spatial grid)
- k_max ρ_s << 1 (KRMHD valid only at scales larger than ion gyroradius)

### Note on k∥ Independence
For straight, uniform B₀ = B₀ẑ, different k∥ modes evolve independently through the
perpendicular Poisson bracket {f,g} = ẑ·(∇⊥f × ∇⊥g). However, we use full 3D grids for:
1. **Field line following**: Interpolation across z-planes couples k∥ modes
2. **Parallel structure**: Track energy distribution in k∥ space
3. **Initial conditions**: Allow z-dependent structures (wave packets, etc.)
4. **Landau physics**: Need k∥ distribution for proper kinetic damping

## Questions for User
Track any clarifications needed during development


ULTRATHINK.
