# KRMHD Spectral Solver Project

## Context
You are implementing a Kinetic Reduced MHD (KRMHD) solver using spectral methods for studying turbulent magnetized plasmas in astrophysical environments. This code is a modern rewrite of GANDALF (https://github.com/anjor/gandalf), a legacy Fortran+CUDA implementation.

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

## Technical Stack

### Core Dependencies
- **JAX**: Primary framework (runs on Apple Metal)
- **jax.numpy**: Array operations
- **jax.scipy.fft**: Spectral transforms
- **Optax**: For any optimization needs
- **h5py**: Data I/O
- **matplotlib**: Diagnostics/visualization

### Numerical Methods
- **Perpendicular (x,y)**: Fourier spectral with 2/3 dealiasing
- **Parallel velocity**: Hermite polynomials (if fully kinetic) or fluid closure
- **Time stepping**: RK4 or RK45 adaptive
- **Poisson solver**: k²φ = ∇²⊥A∥ in Fourier space

## Code Architecture

### Module Structure
```
krmhd/
├── spectral.py      # FFT operations, derivatives, dealiasing
├── physics.py       # KRMHD equations, Poisson brackets
├── timestepping.py  # RK4/RK45 integrators
├── hermite.py       # Hermite basis functions (if needed)
├── diagnostics.py   # Energy, spectra, fluxes
├── io.py           # HDF5 checkpointing
└── validation.py    # Linear physics tests
```

### Design Principles
1. **Functional style**: Pure functions, no side effects
2. **JAX patterns**: Use vmap/scan instead of loops
3. **JIT everything**: @jax.jit on all performance-critical functions
4. **Explicit typing**: Full type hints for clarity
5. **Physics transparency**: Variable names match physics notation

## Implementation Guidelines

### Spectral Operations
- Maintain reality condition: `f(-k) = f*(k)`
- Apply dealiasing after EVERY nonlinear operation
- Use rfft2/irfft2 for real fields to save memory
- Preserve ∇·B = 0 constraint (automatically satisfied in 2D)

### Performance Targets
- 256² resolution should run in real-time on M1 Pro
- 512² should be feasible for production runs
- Memory usage < 8GB for typical resolutions

### Energy Conservation
- Total energy E = E_magnetic + E_kinetic + E_compressive
- Should conserve to ~1e-10 relative error for inviscid runs
- Monitor energy injection/dissipation balance

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

## Current Development Status
- [ ] Basic spectral infrastructure
- [ ] KRMHD equation implementation
- [ ] Linear validation tests
- [ ] Nonlinear turbulence runs
- [ ] Production diagnostics

## Reference Parameters
Typical astrophysical parameters:
- β (plasma beta): 0.01 - 100
- τ (T_i/T_e): 1 - 10  
- Resolution: 256² to 1024²
- k_max ρ_s: ~1-2 (resolve ion scales)

## Questions for User
Track any clarifications needed during development


ULTRATHINK.
