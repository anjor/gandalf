# Running Simulations

This guide walks you through running GANDALF simulations, from your first test to production turbulence studies.

## Prerequisites

GANDALF requires Python 3.10+ and uses `uv` for package management:

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/anjor/gandalf.git
cd gandalf

# Install dependencies (handled automatically by uv)
uv sync
```

JAX will automatically detect and use Apple Metal GPU on M1/M2/M3 Macs, or CUDA GPUs on Linux.

## Your First Simulation

The simplest way to see GANDALF in action is `forcing_minimal.py` (~50 lines, 2 second runtime):

```bash
uv run python examples/forcing_minimal.py
```

**What this does:**
- Creates a 32³ spectral grid
- Initializes random Alfvénic fluctuations
- Applies white noise forcing at k=1,2 (large scales)
- Evolves for 5 Alfvén times with hyper-dissipation (r=2, η=1.0)
- Prints energy every 0.5 τ_A

**Expected output:**
```
t=0.00: E_total=0.0234, E_mag=0.0117, E_kin=0.0117
t=0.50: E_total=0.0245, E_mag=0.0125, E_kin=0.0120
t=1.00: E_total=0.0251, E_mag=0.0128, E_kin=0.0123
...
t=5.00: E_total=0.0268, E_mag=0.0136, E_kin=0.0132
```

Energy grows slightly as forcing adds energy to the system. Magnetic and kinetic energies stay roughly equal (equipartition).

### Understanding the Code

Key parts of `forcing_minimal.py`:

```python
# 1. Create spectral grid
grid = SpectralGrid3D(Nx=32, Ny=32, Nz=32, Lx=2*np.pi, Ly=2*np.pi, Lz=2*np.pi)

# 2. Initialize state
state = initialize_random_spectrum(grid, energy=0.01, alpha=5/3)

# 3. Time evolution loop
for step in range(n_steps):
    # Add forcing (white noise at k=1,2)
    force_z_plus, force_z_minus = force_alfven_modes(
        grid, amplitude=0.05, k_min=1, k_max=2, dt=dt, key=key
    )

    # Evolve one timestep
    state = gandalf_step(state, dt=dt, eta=1.0, hyper_r=2, force_z_plus=force_z_plus)
```

**Box size parameters (Lx, Ly, Lz):**
- The default `Lx = Ly = Lz = 2π` gives integer wavenumbers (k⊥ = 1, 2, 3, ...)
- **Important:** Lz sets both k∥ spacing AND Alfvén time τ_A = Lz/v_A (with Lz = 2π, v_A = 1 → τ_A ≈ 6.28)

**See also:** [Numerical Methods: Box Size and Time Normalization](numerical_methods.md#box-size-and-time-normalization) for detailed explanation.

**Key parameters:**
- `Nx, Ny, Nz`: Grid resolution (32³ = 32,768 points)
- `Lx, Ly, Lz`: Box size (2π gives wavenumbers k = 1, 2, 3, ...)
- `energy`: Initial total energy
- `alpha`: Spectral slope (5/3 for Kolmogorov turbulence)
- `amplitude`: Forcing strength
- `eta`: Hyper-dissipation coefficient
- `hyper_r`: Dissipation order (r=2 is stable and validated)

## Decaying Turbulence

For a more complete example with diagnostics and visualization:

```bash
uv run python examples/decaying_turbulence.py
```

**Runtime:** ~2 minutes on M1 Pro (64³ resolution, 20 Alfvén times)

**What this does:**
- Initializes turbulence with k^(-5/3) spectrum
- Evolves without forcing (decaying turbulence)
- Computes energy spectra E(k⊥) every 2 τ_A
- Saves diagnostic plots to current directory
- Demonstrates selective decay (magnetic energy dominates at late times)

**Output files:**
- `decaying_turbulence_energy.png` - Energy vs time
- `decaying_turbulence_spectrum_t*.png` - Spectra at different times

**Expected behavior:**
- Total energy decays exponentially (viscous dissipation)
- Spectrum maintains k⊥^(-5/3) shape while amplitude decreases
- Magnetic fraction increases from ~0.5 to ~0.7 (selective decay)

### Customizing Parameters

Edit the script to change:

```python
# Resolution (trade-off: accuracy vs runtime)
Nx = 64  # Try 32 (faster) or 128 (more accurate)

# Dissipation strength
eta = 0.5  # Lower = less dissipation, slower decay

# Simulation time
t_final = 50.0  # Longer simulation

# Initial energy
energy = 0.1  # Stronger initial turbulence
```

**Warning:** Increasing resolution scales cubically! 128³ takes ~8× longer than 64³.

## Forced Turbulence (Steady State)

The most sophisticated example is the Alfvénic cascade benchmark:

```bash
# 64³ resolution (recommended for testing)
uv run python examples/alfvenic_cascade_benchmark.py --resolution 64

# Higher resolution (production runs)
uv run python examples/alfvenic_cascade_benchmark.py --resolution 128
```

**Runtime:**
- 64³: ~15-25 minutes
- 128³: ~1-2 hours

**What this does:**
- Applies continuous white noise forcing at k=1,2
- Energy injection balances dissipation → steady state
- Develops k⊥^(-5/3) turbulent cascade
- Computes time-averaged spectra over last 20 τ_A
- Validates critical balance (k∥ ~ k⊥^(2/3))

**Output:**
```
Running Alfvénic cascade benchmark (Resolution: 64³)
Parameters: η=20.0, r=2, forcing amplitude=0.01

t=0.0: E=0.0234
t=5.0: E=0.0312  (growing - spin-up phase)
t=10.0: E=0.0389
...
t=30.0: E=0.0421  (steady state reached)
...
Averaging from t=30.0 to t=50.0
Energy variation during averaging: ΔE/⟨E⟩ = 3.2% ✓

Time-averaged spectrum shows k⊥^(-1.67) scaling (expected: -1.67)
```

Produces `alfvenic_cascade_*_spectrum.png` with kinetic and magnetic spectra.

### Important: Parameter Selection

**Critical insight from Issue #82:** Forced turbulence requires careful balance between energy injection (forcing) and energy removal (dissipation).

**Recommended parameters by resolution:**

| Resolution | η (dissipation) | Force amplitude | Forcing modes | Status |
|------------|----------------|-----------------|---------------|---------|
| 32³ | 1.0 | 0.05 | [1, 2] | ✅ Stable |
| 64³ | 20.0 | 0.01 | [1, 2] | ✅ Stable (requires 10× stronger η, see Note below) |
| 128³ | 2.0 | 0.05 | [1, 2] | ✅ Stable |

**Note on 64³ parameters:** This resolution requires anomalously strong dissipation (η=20.0) or weak forcing (amplitude=0.01) compared to the smooth scaling seen at 32³ and 128³. The root cause is under investigation (Issue #82). **This is an empirical finding specific to this resolution; no theoretical explanation is yet available.** For production work at 64³, use these validated parameters; do not extrapolate from 32³ or 128³.

**Warning signs of instability:**
- Energy grows exponentially (not reaching plateau)
- Maximum velocity exceeds O(100)
- Energy accumulates at high-k (spectral pile-up)

**If you see instability:**
1. Reduce forcing: `--force-amplitude 0.01` (half the injection)
2. Increase dissipation: `--eta 10.0` (double η)
3. Use diagnostics: `--save-diagnostics` (see below)

See [Parameter Scans](parameter_scans.md) for systematic parameter exploration.

### Advanced Diagnostics

Track detailed turbulence metrics during evolution:

```bash
uv run python examples/alfvenic_cascade_benchmark.py \
  --resolution 64 \
  --save-diagnostics \
  --diagnostic-interval 5
```

This saves `diagnostics_R64.h5` containing:
- `max_velocity`: Peak flow speed (detects blow-up)
- `cfl_number`: Timestep stability indicator
- `max_nonlinear`: Cascade rate extremes
- `energy_highk`: Spectral pile-up detector
- `critical_balance_ratio`: Validates RMHD ordering

Analyze with:
```bash
uv run python examples/analyze_issue82_diagnostics.py --resolution 64
```

Produces 6-panel diagnostic plots showing evolution of all metrics.

## Using Configuration Files

For reproducible runs, use YAML config files:

```bash
# Run with existing config
uv run python run_simulation.py --config configs/decaying_turbulence.yaml

# Generate a template
uv run python run_simulation.py --generate-template my_config.yaml
```

### Example Config

```yaml
# my_turbulence_run.yaml
simulation:
  name: "my_turbulence_run"
  output_dir: "output/my_run"

grid:
  Nx: 64
  Ny: 64
  Nz: 64
  Lx: 6.283185307179586  # 2π
  Ly: 6.283185307179586
  Lz: 6.283185307179586

initial_condition:
  type: "random_spectrum"
  energy: 0.01
  alpha: 1.6667  # k^(-5/3) spectrum

timestepping:
  dt: 0.005
  t_final: 20.0
  cfl_safety_factor: 0.5

dissipation:
  eta: 1.0
  nu: 1.0
  hyper_r: 2
  hyper_n: 2

diagnostics:
  energy_interval: 0.5
  spectrum_interval: 2.0
  checkpoint_interval: 10.0
```

**Advantages:**
- Reproducibility (commit configs to git)
- Easy parameter sweeps (generate multiple configs)
- Self-documenting (parameters and values in one place)

## Command-Line Arguments

Most examples accept command-line arguments:

```bash
# Alfvénic cascade benchmark
uv run python examples/alfvenic_cascade_benchmark.py \
  --resolution 128 \           # Grid size (32, 64, 128)
  --total-time 50.0 \          # Simulation time (τ_A)
  --averaging-start 30.0 \     # When to start averaging
  --eta 2.0 \                  # Dissipation coefficient
  --hyper-r 2 \                # Dissipation order
  --force-amplitude 0.05 \     # Forcing strength
  --save-diagnostics \         # Enable detailed diagnostics
  --diagnostic-interval 5      # Diagnostic save frequency

# Decaying turbulence
uv run python examples/decaying_turbulence.py \
  --resolution 128 \
  --eta 0.5 \
  --t-final 30.0

# Orszag-Tang vortex
uv run python examples/orszag_tang.py \
  --resolution 128 \
  --t-final 3.0
```

Use `--help` to see available options:
```bash
uv run python examples/alfvenic_cascade_benchmark.py --help
```

## Output Files and Directories

GANDALF produces several types of output:

### 1. Console Output
Real-time progress printed to terminal:
```
t=0.00: E_total=0.0234, E_mag=0.0117, E_kin=0.0117
t=0.50: E_total=0.0245, E_mag=0.0125, E_kin=0.0120
```

### 2. Diagnostic Plots (PNG)
- `*_energy.png` - Energy vs time
- `*_spectrum.png` - Energy spectra E(k⊥)
- `*_structures.png` - 2D slices of fields

Located in current directory (or `output_dir` if using configs).

### 3. HDF5 Checkpoints
Full state snapshots for resuming:
```
output/
├── checkpoint_t00.h5
├── checkpoint_t10.h5
└── checkpoint_t20.h5
```

Load with:
```python
from krmhd.io import load_checkpoint
state, metadata = load_checkpoint("output/checkpoint_t10.h5")
```

### 4. Time Series Data (HDF5)
Energy history for post-processing:
```python
from krmhd.io import load_timeseries
history = load_timeseries("output/energy_history.h5")
# history.times, history.total_energy, history.magnetic_fraction, etc.
```

### 5. Turbulence Diagnostics (HDF5)
Detailed metrics (when using `--save-diagnostics`):
```
diagnostics_R64.h5  # Contains max_velocity, cfl_number, etc.
```

## Visualization

### Quick Plots
Most examples include built-in plotting:
```python
from krmhd.diagnostics import plot_energy_history, plot_energy_spectrum

# Energy vs time
plot_energy_history(history, filename="my_energy.png")

# Spectrum at current time
E_perp, k_perp = energy_spectrum_perpendicular(state)
plot_energy_spectrum(k_perp, E_perp, filename="my_spectrum.png")
```

### Field Visualization
```python
from krmhd.diagnostics import plot_state

# 2D slices of φ, A∥, δB∥ at z=0 plane
plot_state(state, filename="fields.png")
```

### Custom Analysis
```python
import h5py
import matplotlib.pyplot as plt

# Load checkpoint
with h5py.File("output/checkpoint_t20.h5", "r") as f:
    z_plus = f["z_plus"][:]  # Shape: (Nz, Ny, Nx//2+1), complex
    times = f.attrs["time"]

# Transform to real space
phi_real = jax.numpy.fft.irfftn(state.phi, s=(grid.Nz, grid.Ny, grid.Nx))

# Plot custom slice
plt.imshow(phi_real[0, :, :])  # z=0 plane
plt.colorbar()
plt.savefig("phi_slice.png")
```

## Checkpointing and Resuming

### Automatic Checkpointing
When using `run_simulation.py`:
```yaml
diagnostics:
  checkpoint_interval: 10.0  # Save every 10 τ_A
```

### Manual Checkpointing
```python
from krmhd.io import save_checkpoint, load_checkpoint

# Save
save_checkpoint("my_checkpoint.h5", state, metadata={"time": t, "step": step})

# Resume
state, metadata = load_checkpoint("my_checkpoint.h5")
t_start = metadata["time"]

# Continue evolution from t_start
for t in range(t_start, t_final, dt):
    state = gandalf_step(state, dt=dt, eta=1.0)
```

**Use cases:**
- Long runs (checkpoint every 10-20 τ_A in case of crashes)
- Parameter changes (load checkpoint, modify parameters, continue)
- Ensemble runs (load checkpoint, apply different forcing realization)

## Common Pitfalls and Solutions

### 1. Simulation Blows Up (Energy → ∞)

**Symptoms:**
- Energy grows exponentially
- NaN or Inf in output
- Maximum velocity > 1000

**Causes & Solutions:**

| Cause | Solution |
|-------|----------|
| Timestep too large (CFL > 1) | Reduce `dt` or use `compute_cfl_timestep()`. **Note: CFL < 1 is necessary but not sufficient for stability in forced turbulence (see Issue #82).** |
| Insufficient dissipation | Increase `eta` (e.g., 1.0 → 2.0) |
| Forcing too strong | Reduce `force_amplitude` (0.05 → 0.01) |
| Dealiasing failed | Check `grid.dealias()` is called after nonlinear terms |

**Debug workflow:**
```bash
# Enable diagnostics to see WHERE it fails
uv run python examples/alfvenic_cascade_benchmark.py \
  --save-diagnostics --diagnostic-interval 5

# Analyze
uv run python examples/analyze_issue82_diagnostics.py --resolution 64
```

### 2. Spectrum Doesn't Match Theory

**Expected:** k⊥^(-5/3) in inertial range (k ~ 2-10)

**If spectrum is too steep:**
- Dissipation too strong → Reduce `eta`
- Hyper-dissipation order too high → Use r=2 instead of r=4
- Resolution too low → Increase Nx, Ny, Nz

**If spectrum is too shallow:**
- Dissipation too weak → Increase `eta`
- Spectral pile-up at high-k → Enable `--save-diagnostics`, check `energy_highk`

### 3. Energy Not Conserved (Inviscid Case)

For inviscid runs (`eta=0`, `nu=0`), energy should be conserved to ~0.01%:

**If energy drifts:**
- Dealiasing not applied → Check `grid.dealias()` calls
- Timestep too large → Reduce `dt`
- Using wrong RHS formulation → Use GANDALF formulation (default)

**Test energy conservation:**
```bash
uv run python tests/test_timestepping.py::test_gandalf_step_energy_conservation
```

### 4. Slow Performance

**Expected runtime (M1 Pro):**
- 32³: seconds to minutes
- 64³: minutes to ~20 minutes
- 128³: ~1-2 hours
- 256³: ~8-16 hours (not extensively tested)

**If slower than expected:**
- Ensure JAX is using GPU: `jax.devices()` should show Metal/CUDA
- Check for Python loops (use `jax.vmap` or `jax.lax.scan` instead)
- Profile with: `JAX_ENABLE_X64=0` (use float32 instead of float64)

**Memory issues:**
- 256³ requires ~60GB RAM
- Reduce resolution or use checkpointing to save intermediate states

### 5. Config File Not Found

```bash
# Error: FileNotFoundError: configs/my_config.yaml

# Solution: Use absolute path or run from repo root
cd /path/to/gandalf
uv run python run_simulation.py --config configs/my_config.yaml

# Or generate template in current directory
uv run python run_simulation.py --generate-template ./my_config.yaml
```

### 6. Import Errors

```bash
# Error: ModuleNotFoundError: No module named 'krmhd'

# Solution: Run via `uv run` to use the virtual environment
uv run python examples/forcing_minimal.py

# Or activate venv manually
source .venv/bin/activate
python examples/forcing_minimal.py
```

## Troubleshooting Checklist

Before filing an issue, try:

1. ✅ Run a known-working example: `uv run python examples/forcing_minimal.py`
2. ✅ Check Python version: `python --version` (need 3.10+)
3. ✅ Update dependencies: `uv sync`
4. ✅ Verify JAX installation: `uv run python -c "import jax; print(jax.devices())"`
5. ✅ Run tests: `uv run pytest tests/` (should pass 448+ tests)
6. ✅ Check for NaN/Inf: Enable `--save-diagnostics` and analyze

If still stuck, file an issue with:
- Full command used
- Error message or unexpected output
- Output of `uv run python -c "import jax; print(jax.__version__)"`
- Resolution and parameters used

## Next Steps

- **Parameter studies:** See [Parameter Scans](parameter_scans.md)
- **Understanding the physics:** See [Physics Validity Regimes](physics_validity.md)
- **Algorithm details:** See [Numerical Methods](numerical_methods.md)
- **More examples:** Browse `examples/` directory

## Quick Reference

```bash
# Fastest test
uv run python examples/forcing_minimal.py

# Standard turbulence
uv run python examples/decaying_turbulence.py

# Forced steady-state (64³)
uv run python examples/alfvenic_cascade_benchmark.py --resolution 64

# With diagnostics
uv run python examples/alfvenic_cascade_benchmark.py \
  --resolution 64 --save-diagnostics --diagnostic-interval 5

# Using config files
uv run python run_simulation.py --config configs/decaying_turbulence.yaml

# Generate config template
uv run python run_simulation.py --generate-template my_config.yaml

# Run tests
uv run pytest tests/

# Check JAX device
uv run python -c "import jax; print(jax.devices())"
```
