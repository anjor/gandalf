## KRMHD Solver: Step-by-Step Implementation Plan

### Step 1: Project Bootstrap
```bash
# First command to Claude Code
Create a new Python project structure for a KRMHD spectral solver:
- pyproject.toml with JAX, h5py, matplotlib dependencies  
- krmhd/ package with __init__.py
- tests/ directory
- examples/ directory
- README.md with basic project description
Ensure JAX will work on Apple M1/M2 Metal backend.
```

### Step 2: Core Spectral Infrastructure
```bash
Create krmhd/spectral.py with:
1. A SpectralField2D class that wraps JAX arrays
2. FFT/IFFT operations using jax.numpy.fft
3. Derivative operators (∂x, ∂y) in Fourier space
4. Dealising function (2/3 rule)
Include a simple test that verifies ∂x(sin(x)) = cos(x)
```

### Step 3: Poisson Solver
```bash
In krmhd/spectral.py add:
1. Poisson solver: solve ∇²φ = ω for φ given vorticity ω
2. Use k² in Fourier space (handle k=0 carefully)
3. Test with manufactured solution ω = sin(x)cos(y)
Verify the solver returns φ with correct boundary conditions.
```

### Step 4: Poisson Bracket
```bash
Create krmhd/physics.py with:
1. poisson_bracket(f, g) function computing {f,g} = ẑ·(∇f × ∇g)
2. Use spectral derivatives and proper dealiasing
3. Test with f=sin(x), g=cos(y) - should give sin(x)sin(y)
This is the core nonlinearity in KRMHD.
```

### Step 5: KRMHD State and Initialization
```bash
In krmhd/physics.py add:
1. KRMHDState dataclass with fields: phi, A_parallel, B_parallel, time
2. initialize_alfven_wave() function for linear wave test
3. initialize_random_spectrum() for turbulence with specified k^-α spectrum
4. energy() function computing E_magnetic + E_kinetic
```

### Step 6: Time Evolution (Alfvén Dynamics)
```bash
In krmhd/physics.py add the RHS for active fields:
1. dphi_dt computing RHS of stream function equation
2. dA_parallel_dt computing RHS of vector potential equation
3. Include viscosity/resistivity terms
4. Test: initialized Alfvén wave should propagate at correct speed
```

### Step 7: Passive Scalar Evolution
```bash
Extend krmhd/physics.py with slow mode evolution:
1. dB_parallel_dt for parallel magnetic field (passive scalar)
2. Ensure it's only advected, no back-reaction on phi or A_parallel
3. Test: B_parallel should be passively mixed by turbulent flow
```

### Step 8: Time Integrator
```bash
Create krmhd/timestepping.py with:
1. RK4 integrator that takes (state, dt, rhs_function)
2. CFL condition calculator based on max(v_Alfven, v_flow)
3. Test with Alfvén wave - verify 4th order convergence
```

### Step 9: Basic Diagnostics
```bash
Create krmhd/diagnostics.py with:
1. energy_spectrum(state) returning k, E(k)
2. energy_history tracking total, magnetic, kinetic energy
3. plot_state() for quick visualization of phi, A_parallel, B_parallel
Test by running decaying turbulence and plotting E(t)
```

### Step 10: Linear Physics Validation
```bash
Create tests/test_linear_physics.py with:
1. Alfvén wave dispersion test (ω vs k_parallel)
2. Check wave propagates at v_A
3. Verify energy conservation to machine precision
This validates our basic physics implementation.
```

### Step 11: Orszag-Tang Vortex
```bash
Create examples/orszag_tang.py:
1. Classic MHD test problem setup
2. Run to t=1.0, compare with known results
3. Check for correct shock formation
This tests nonlinear dynamics.
```

### Step 12: Decaying Turbulence Run
```bash
Create examples/decaying_turbulence.py:
1. Initialize with k^-1 spectrum
2. Run until cascade develops
3. Verify k^-5/3 inertial range
4. Check slow modes remain passive
This is our first physics research run.
```

### Step 13: HDF5 I/O
```bash
Create krmhd/io.py with:
1. save_checkpoint(state, filename) 
2. load_checkpoint(filename)
3. save_diagnostics(diagnostics_dict, filename)
Test with a turbulence run - checkpoint and restart.
```

### Step 14: Landau Damping Closure
```bash
Extend krmhd/physics.py with:
1. Simple Landau damping model for electron pressure
2. Add damping term to appropriate equations
3. Verify damping rate matches theory for linear waves
This completes the physics model.
```

### Step 15: Production Features
```bash
Create krmhd/config.py with:
1. YAML configuration file parser
2. Parameter validation
3. Run script that takes config file as input
Create example configs for standard cases.
```

---

**Execution notes for Claude Code:**
- Complete each step fully before moving to the next
- Run tests immediately after implementing each feature
- If a test fails, debug before proceeding
- Keep functions small and focused
- Use @jax.jit liberally but test without it first

Start with Step 1 and proceed sequentially. Each step builds on the previous ones, so order matters.
