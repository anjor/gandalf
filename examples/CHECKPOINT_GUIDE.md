# Checkpoint Guide for KRMHD Turbulence Simulations

This guide shows how to use checkpoint save/resume functionality in `alfvenic_cascade_benchmark.py` and related scripts.

## Why Use Checkpoints?

Checkpoints are essential for:
- **Long-running simulations**: Save progress and avoid starting over if something fails
- **Parameter tuning**: Resume from a stable state and try different dissipation/forcing parameters
- **Failure recovery**: Auto-save when CFL violations or energy spikes are detected
- **Iterative exploration**: Build up turbulence gradually with changing parameters

## Basic Usage

### Save Periodic Checkpoints

Save checkpoints every 20 Alfvén times:

```bash
uv run python examples/alfvenic_cascade_benchmark.py \
  --resolution 64 \
  --total-time 200 \
  --checkpoint-interval-time 20.0 \
  --checkpoint-dir checkpoints_64cubed/
```

This creates:
- `checkpoints_64cubed/checkpoint_t020.0.h5`
- `checkpoints_64cubed/checkpoint_t040.0.h5`
- `checkpoints_64cubed/checkpoint_t060.0.h5`
- ...
- `checkpoints_64cubed/checkpoint_final_t200.0.h5` (at end)

### Save Checkpoints by Steps

Save every 5000 steps instead of by time:

```bash
uv run python examples/alfvenic_cascade_benchmark.py \
  --resolution 64 \
  --checkpoint-interval-steps 5000
```

### Auto-Save on Issues

Automatically save if CFL violation or energy spike detected (enabled by default):

```bash
uv run python examples/alfvenic_cascade_benchmark.py \
  --resolution 64 \
  --checkpoint-interval-time 20.0 \
  --checkpoint-on-issues  # Enabled by default
```

If instability detected at t=161.5:
- Creates: `checkpoint_t161.5_CFL_VIOLATION.h5` or `checkpoint_t161.5_HIGH_VELOCITY.h5`

To disable: `--no-checkpoint-on-issues`

## Resuming from Checkpoints

### Basic Resume

Continue from where you left off:

```bash
uv run python examples/alfvenic_cascade_benchmark.py \
  --resume-from checkpoints_64cubed/checkpoint_t160.0.h5 \
  --total-time 250  # Run from t=160 to t=250
```

**Important**: `--total-time` is the **target end time** (not additional time).

### Resume with Modified Parameters

This is the **key feature** - resume from a checkpoint but change dissipation/forcing:

```bash
# Original run failed at t=160 with eta=4.5, amplitude=0.055
# Resume with stronger dissipation and weaker forcing:

uv run python examples/alfvenic_cascade_benchmark.py \
  --resume-from checkpoints_64cubed/checkpoint_t160.0.h5 \
  --eta 10.0 \              # OVERRIDE: 2.2x stronger dissipation
  --force-amplitude 0.02 \  # OVERRIDE: 2.75x weaker forcing
  --total-time 250 \
  --checkpoint-interval-time 20.0  # Continue saving checkpoints
```

The script will print:
```
RESUMING FROM CHECKPOINT: checkpoints_64cubed/checkpoint_t160.0.h5
✓ Loaded checkpoint from t=160.00 τ_A (step 34200)

  Original parameters from checkpoint:
    eta: 4.5
    force_amplitude: 0.055

  Parameter overrides:
    eta: 4.5 → 10.0
    force_amplitude: 0.055 → 0.02
```

## Common Workflows

### Workflow 1: Long Run with Checkpoints

```bash
# Initial run (saves checkpoints every 20 τ_A)
uv run python examples/alfvenic_cascade_benchmark.py \
  --resolution 64 --total-time 200 \
  --checkpoint-interval-time 20.0 \
  --checkpoint-dir checkpoints_stable/

# If stable: continue to t=300 for better statistics
uv run python examples/alfvenic_cascade_benchmark.py \
  --resume-from checkpoints_stable/checkpoint_final_t200.0.h5 \
  --total-time 300 \
  --checkpoint-interval-time 20.0
```

### Workflow 2: Recover from Instability (Issue #82/#99)

```bash
# Run with diagnostics to detect instability early
uv run python examples/alfvenic_cascade_benchmark.py \
  --resolution 64 --total-time 200 \
  --eta 4.5 --force-amplitude 0.055 \
  --checkpoint-interval-time 20.0 \
  --checkpoint-dir checkpoints_64cubed/ \
  --save-diagnostics --diagnostic-interval 5

# FAILED at t=161.5 with CFL violation
# Auto-saved: checkpoints_64cubed/checkpoint_t161.5_CFL_VIOLATION.h5

# Resume from PREVIOUS stable checkpoint with safer parameters
uv run python examples/alfvenic_cascade_benchmark.py \
  --resume-from checkpoints_64cubed/checkpoint_t160.0.h5 \
  --eta 10.0 \              # Increase dissipation
  --force-amplitude 0.02 \  # Reduce forcing
  --total-time 250 \
  --checkpoint-interval-time 20.0 \
  --save-diagnostics
```

### Workflow 3: Parameter Sweep with Checkpoints

Resume capability also works with `run_parameter_sweep.py`:

```bash
python examples/run_parameter_sweep.py \
  --resolution 64 \
  --total-time 100 \
  --checkpoint-interval-time 25.0 \  # Save every 25 τ_A
  --checkpoint-final  # Save final state for each configuration
```

Each parameter combination gets its own checkpoint directory.

## Checkpoint File Contents

Checkpoint files (HDF5 format) contain:

**State data:**
- All fields: z⁺, z⁻, B∥, Hermite moments g
- Grid configuration: Nx, Ny, Nz, Lx, Ly, Lz
- State parameters: M, beta_i, v_th, nu, Lambda, **time**

**Metadata** (stored in `/metadata` group):
- `timestamp`: When checkpoint was created
- `step`: Step number
- `eta`, `nu`, `hyper_r`, `hyper_n`: Dissipation parameters
- `force_amplitude`, `v_A`, `dt`: Forcing and timestep
- `description`: Human-readable description
- Any custom metadata you provide

## What Parameters Can/Cannot Change on Resume

**✅ Can change (command-line override):**
- `eta` (dissipation coefficient)
- `nu` (collision frequency)
- `hyper_r`, `hyper_n` (hyper-dissipation orders)
- `force_amplitude` (forcing strength)
- `v_A` (Alfvén velocity)
- `dt` (timestep - computed from CFL or manual)

**❌ Cannot change (locked by checkpoint data):**
- Grid resolution: Nx, Ny, Nz
- Domain size: Lx, Ly, Lz
- Number of Hermite moments: M

## Best Practices

1. **Always use `--checkpoint-interval-time`** for long runs (20-50 τ_A intervals)
2. **Enable `--save-diagnostics`** when experimenting with parameters (Issue #82)
3. **Keep original checkpoints**: Don't delete stable checkpoints until run completes
4. **Document parameter changes**: When resuming with overrides, keep notes on why
5. **Check metadata**: Use `h5dump -A checkpoint.h5` to inspect saved parameters

## Inspecting Checkpoints

### Using Python

```python
from krmhd.io import load_checkpoint

state, grid, metadata = load_checkpoint("checkpoint_t160.0.h5")

print(f"Checkpoint time: {state.time} τ_A")
print(f"Grid: {grid.Nx}×{grid.Ny}×{grid.Nz}")
print(f"Original eta: {metadata['eta']}")
print(f"Saved at: {metadata['timestamp']}")

# Extract energy
from krmhd import energy as compute_energy
E = compute_energy(state)['total']
print(f"Total energy: {E:.6e}")
```

### Using HDF5 Tools

```bash
# View checkpoint structure
h5dump -n checkpoint_t160.0.h5

# View metadata only
h5dump -A checkpoint_t160.0.h5 | grep "eta\|force_amplitude\|time"
```

## Troubleshooting

**"Checkpoint file not found"**
- Check path: Use absolute path or ensure working directory is correct
- Check spelling: Filenames use format `checkpoint_t160.0.h5` (not `t160`)

**"Grid dimensions don't match"**
- Cannot resume 64³ checkpoint on 128³ grid (or vice versa)
- Use same `--resolution` as original run

**"Parameters not overriding"**
- Ensure you specify `--eta`, `--force-amplitude`, etc. AFTER `--resume-from`
- Check printed output for "Parameter overrides:" confirmation

**"Simulation still unstable after resume"**
- Try stronger dissipation: `--eta 20.0` (for 64³)
- Try weaker forcing: `--force-amplitude 0.01`
- Resume from earlier checkpoint (before instability developed)
- See `CLAUDE.md` "Forced Turbulence: Parameter Selection Guide" for details
