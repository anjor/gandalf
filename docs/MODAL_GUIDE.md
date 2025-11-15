# Running GANDALF on Modal Cloud

This guide explains how to run GANDALF KRMHD simulations on [Modal](https://modal.com), a serverless cloud computing platform that provides on-demand CPU and GPU resources.

## Why Use Modal?

- **Scalability**: Run high-resolution simulations (256³, 512³+) without local hardware constraints
- **GPU Acceleration**: Access NVIDIA GPUs (T4, A10G, A100) for JAX-accelerated computations
- **Parallel Sweeps**: Run parameter sweeps with automatic parallelization
- **Cost-Effective**: Pay only for compute time used (no idle costs)
- **Easy Setup**: No Docker/Kubernetes knowledge required

## Prerequisites

1. **Modal Account** (free tier available):
   ```bash
   # Create account at https://modal.com
   ```

2. **Install Modal CLI**:
   ```bash
   pip install modal
   ```

3. **Authenticate**:
   ```bash
   modal token new
   ```
   This will open a browser for authentication.

4. **Verify Installation**:
   ```bash
   modal token list
   ```

## Quick Start

### 1. Deploy the GANDALF Modal App

From the GANDALF repository root:

```bash
modal deploy modal_app.py
```

This creates a Modal app named `gandalf-krmhd` and sets up a persistent volume for results.

### 2. Submit Your First Simulation

Using the helper script:

```bash
python scripts/modal_submit.py submit configs/driven_turbulence.yaml
```

Or directly with Modal CLI:

```bash
modal run modal_app.py --config-path configs/driven_turbulence.yaml
```

### 3. Monitor Progress

The job will stream logs to your terminal. You'll see:
- JAX device information (CPU/GPU)
- Simulation progress (timesteps, energy, etc.)
- Final results and output location

### 4. Download Results

```bash
# List available results
python scripts/modal_submit.py list

# Download specific run
python scripts/modal_submit.py download driven_turbulence_20240115_120000 ./results
```

Or use Modal CLI directly:

```bash
modal volume get gandalf-results driven_turbulence_20240115_120000 ./results
```

## Migrating Local Configs to Modal

When adapting existing local configuration files for Modal cloud execution, consider these key differences and adjustments:

### Config Compatibility

**Good news**: Modal uses the same YAML config format as local runs! You can use your existing configs directly.

**However**, you may want to adjust parameters for cloud execution:

### Resolution and Resource Scaling

**Local Development** (typical laptop/desktop):
- **Recommended**: 32³ to 64³ (development, debugging)
- **Maximum**: 128³ (M1 Pro/M2 with 16-32 GB RAM)
- **Limitation**: Memory constraints

**Modal Cloud** (scalable resources):
- **Small**: 64³ (fast testing, ~5-10 min, ~$0.10-0.20)
- **Medium**: 128³ (production, ~15-30 min, ~$0.50-1.00)
- **Large**: 256³+ (high-resolution, ~1-4 hours, ~$5-20)
- **Limitation**: Cost (but no hard memory limits)

**Migration tip**: Start with your local resolution for validation, then scale up gradually.

### Forcing Parameters (Issue #97 Fix)

**Important**: The config uses **physical wavenumbers** (k_min, k_max), but these are automatically converted to **integer mode numbers** (n_min, n_max) at runtime.

```yaml
forcing:
  enabled: true
  k_min: 2.0    # Physical wavenumber (user-friendly)
  k_max: 5.0    # Converted to mode numbers internally
  amplitude: 0.1
```

**Conversion formula**: `n = round(k * L_min / (2π))`

**Example** (for domain L=1.0):
- k_min=2.0 → n_min=0 (clamped to 1, fundamental mode)
- k_min=6.28 (2π) → n_min=1 (fundamental mode)
- k_min=12.57 (4π) → n_min=2 (second harmonic)

**No action needed** - conversion is automatic! Just be aware that small k values may be clamped to n≥1.

### Hyper-dissipation for High Resolution

For 256³+ simulations, use stronger hyper-dissipation to prevent spectral pile-up:

```yaml
# Local config (64³-128³)
physics:
  eta: 1.0
  nu: 1.0
  hyper_r: 2
  hyper_n: 2

# Modal high-res config (256³+)
physics:
  eta: 5.0      # Increased from 1.0
  nu: 5.0       # Matched to eta
  hyper_r: 2    # Keep r=2 for stability
  hyper_n: 2
```

**Why?** Higher resolution requires stronger dissipation at small scales. See [CLAUDE.md](../CLAUDE.md) "Forced Turbulence: Parameter Selection Guide" for details.

### Time Integration

**Local testing** (short runs):
```yaml
time_integration:
  n_steps: 100        # Quick validation
  dt_fixed: 0.005
```

**Modal production** (long runs):
```yaml
time_integration:
  n_steps: 10000      # 50 τ_A at dt=0.005
  dt_fixed: null      # Use CFL timestep for safety
  cfl_safety: 0.3
```

**Migration tip**: Use `dt_fixed` for reproducibility in validation, CFL for safety in long cloud runs.

### I/O Configuration

**Local**:
```yaml
io:
  checkpoint_interval: 1000   # Frequent for interactive work
  output_dir: "./output"      # Local filesystem
```

**Modal**:
```yaml
io:
  checkpoint_interval: 500    # Balance recovery time vs I/O overhead
  output_dir: "./output"      # Mapped to Modal volume automatically
```

**Note**: Modal saves to `/results` in the volume. The `output_dir` config is relative to this path.

### GPU vs CPU Selection

**When to use GPU** (via `--gpu` flag):
- Resolution ≥ 256³
- Long runs (>1 hour)
- Parameter sweeps with many jobs

**When to use CPU**:
- Resolution ≤ 128³
- Short validation runs (<10 min)
- Development/debugging

**Migration tip**: Test with CPU first, then switch to GPU for production.

### Common Migration Workflow

1. **Start with local config as-is**:
   ```bash
   python scripts/modal_submit.py submit configs/my_local_config.yaml
   ```

2. **Validate** (compare first few timesteps with local run):
   ```bash
   python scripts/modal_submit.py download my_local_config_<timestamp>
   # Compare output/history.h5 with local run
   ```

3. **Scale up** (create high-res variant):
   ```bash
   # Copy and modify
   cp configs/my_local_config.yaml configs/my_modal_highres.yaml
   # Edit: Increase grid.Nx/Ny/Nz, adjust physics.eta, increase n_steps
   ```

4. **Submit production run**:
   ```bash
   python scripts/modal_submit.py submit configs/my_modal_highres.yaml --gpu
   ```

### Parameter Sweep Migration

**Local** (sequential or manual parallelization):
```bash
# Run multiple configs manually
for eta in 0.5 1.0 2.0; do
  sed "s/eta: .*/eta: $eta/" config.yaml > config_eta${eta}.yaml
  python scripts/run_simulation.py config_eta${eta}.yaml
done
```

**Modal** (automatic parallelization):
```bash
# Single command runs all combinations in parallel!
python scripts/modal_submit.py sweep configs/base_config.yaml \
  --param physics.eta=0.5,1.0,2.0 \
  --param physics.nu=0.5,1.0
# Launches 3×2=6 parallel jobs
```

### Troubleshooting Config Issues

**Problem**: "Parameter 'eta' must use dot notation"

**Solution**: In parameter sweeps, use `physics.eta` not `eta`:
```bash
# Wrong
--param eta=0.5,1.0

# Correct
--param physics.eta=0.5,1.0
```

**Problem**: Energy grows exponentially in forced turbulence

**Solution**: Increase dissipation or reduce forcing amplitude:
```yaml
# If unstable
physics:
  eta: 2.0          # Try 5.0 or 10.0
forcing:
  amplitude: 0.05   # Try 0.01 or 0.02
```

See [Issue #82](https://github.com/anjor/gandalf/issues/82) for detailed stability guidelines.

## Usage Examples

### Example 1: CPU Simulation (Low-Cost)

```bash
# Submit 64³ simulation on CPU
python scripts/modal_submit.py submit configs/driven_turbulence.yaml

# Runtime: ~5-10 minutes
# Cost: ~$0.10-0.20 (8 CPU cores, 32 GB RAM)
```

### Example 2: GPU Simulation (High-Performance)

```bash
# Submit 256³ simulation on GPU
python scripts/modal_submit.py submit configs/modal_high_res.yaml --gpu

# Runtime: ~30-60 minutes
# Cost: ~$2-4 (NVIDIA T4 GPU, 4 CPU cores, 16 GB RAM)
```

### Example 3: Parameter Sweep (Parallel)

```bash
# Sweep over resistivity and resolution
python scripts/modal_submit.py sweep configs/driven_turbulence.yaml \
    --param physics.eta 0.01 0.02 0.05 0.1 \
    --param grid.Nx 64 128 \
    --name eta_resolution_sweep

# Runs 4×2=8 simulations in parallel
# Runtime: ~20 minutes (limited by longest job)
# Cost: ~$1-2 total (parallel execution)
```

### Example 4: Custom Output Directory

```bash
# Specify custom output location
python scripts/modal_submit.py submit configs/driven_turbulence.yaml \
    --output-subdir my_experiment/run_001
```

## Resource Configuration

### CPU vs GPU

**Use CPU when:**
- Resolution ≤ 128³
- Quick tests and debugging
- Cost-sensitive applications
- Minimal JAX GPU overhead

**Use GPU when:**
- Resolution ≥ 256³
- Long time integrations (>1000 steps)
- Multiple parameter sweeps
- Speedup justifies cost (~2-5× faster)

### Customizing Resources

Edit `modal_app.py` to adjust resources:

```python
@app.function(
    image=image,
    cpu=16.0,        # CPU cores (default: 8.0)
    memory=65536,    # RAM in MB (default: 32768)
    timeout=7200,    # Timeout in seconds (default: 14400)
)
def run_simulation_remote(...):
    ...

@app.function(
    image=gpu_image,
    gpu="A100",      # GPU type: T4, A10G, A100 (default: T4)
    cpu=8.0,         # CPU cores (default: 4.0)
    memory=32768,    # RAM in MB (default: 16384)
    timeout=7200,
)
def run_simulation_gpu(...):
    ...
```

Redeploy after changes:
```bash
modal deploy modal_app.py
```

## Parameter Sweeps

### Basic Sweep

```python
# In Python script
from pathlib import Path
import modal_app

# Read base config
with open("configs/driven_turbulence.yaml") as f:
    base_config = f.read()

# Define parameter space
parameters = {
    'physics.eta': [0.01, 0.02, 0.05],
    'physics.nu': [0.01, 0.02],
    'grid.Nx': [64, 128],
}

# Submit sweep (runs 3×2×2=12 jobs in parallel)
results = modal_app.run_parameter_sweep.remote(
    base_config,
    parameters,
    sweep_name="eta_nu_resolution",
    use_gpu=False
)
```

### Advanced Sweep with Analysis

```python
import modal_app
import numpy as np
import matplotlib.pyplot as plt

# Run sweep
with open("configs/driven_turbulence.yaml") as f:
    base_config = f.read()

parameters = {'physics.eta': np.logspace(-3, -1, 10).tolist()}

results = modal_app.run_parameter_sweep.remote(
    base_config,
    parameters,
    sweep_name="resistivity_scan",
    use_gpu=False
)

# Analyze results
etas = parameters['physics.eta']
energies = [r['final_energy_total'] for r in results]

plt.loglog(etas, energies, 'o-')
plt.xlabel('Resistivity η')
plt.ylabel('Final Energy')
plt.savefig('resistivity_scan.png')
```

## Data Management

### Modal Volume Structure

Results are stored in a persistent Modal volume:

```
/results/
├── driven_turbulence_20240115_120000/
│   ├── config.yaml
│   ├── energy_history.h5
│   ├── spectra.npz
│   ├── final_state.h5
│   └── *.png
├── sweep_20240115_130000/
│   ├── run_0000_eta=0.01/
│   ├── run_0001_eta=0.02/
│   └── sweep_metadata.json
└── ...
```

### Downloading Results

**Method 1: Helper script**
```bash
python scripts/modal_submit.py download driven_turbulence_20240115_120000 ./results
```

**Method 2: Modal CLI**
```bash
# Download entire directory
modal volume get gandalf-results driven_turbulence_20240115_120000 ./results

# Download single file
modal volume get gandalf-results driven_turbulence_20240115_120000/energy_history.h5 ./energy.h5
```

**Method 3: Python API**
```python
import modal

volume = modal.Volume.lookup("gandalf-results")

# List files
for path in volume.listdir("/driven_turbulence_20240115_120000"):
    print(path)

# Read file
data = volume.read_file("/driven_turbulence_20240115_120000/energy_history.h5")
with open("local_energy.h5", "wb") as f:
    f.write(data)
```

### Cleanup

Delete old results to save storage costs:

```bash
# List volumes
modal volume list

# Delete specific directory
modal volume rm gandalf-results driven_turbulence_20240115_120000

# Delete entire volume (WARNING: irreversible!)
modal volume delete gandalf-results
```

## Cost Estimation

> **Pricing as of January 2025**. Costs may vary based on Modal's current pricing and regional availability. Always check [modal.com/pricing](https://modal.com/pricing) for up-to-date rates.

### CPU Instances (8 cores, 32 GB RAM)

| Resolution | Steps | Runtime | Estimated Cost |
|------------|-------|---------|----------------|
| 64³        | 200   | ~5 min  | ~$0.10         |
| 128³       | 500   | ~20 min | ~$0.40         |
| 256³       | 1000  | ~2 hrs  | ~$4.00         |

### GPU Instances (T4, 4 cores, 16 GB RAM)

| Resolution | Steps | Runtime | Estimated Cost |
|------------|-------|---------|----------------|
| 128³       | 500   | ~5 min  | ~$0.50         |
| 256³       | 1000  | ~30 min | ~$3.00         |
| 512³       | 2000  | ~4 hrs  | ~$40.00        |

*Costs assume on-demand pricing. Modal also offers spot instances (up to 90% cheaper) but with potential interruptions.*

### Optimization Tips

1. **Test locally first**: Debug with small configs before submitting to Modal
2. **Use checkpoints**: Enable checkpointing for long runs (>1 hour) to resume if interrupted (see Checkpointing section below)
3. **Batch similar jobs**: Parameter sweeps share startup overhead
4. **Monitor costs**: Use `modal app list` to track spending
5. **Delete old results**: Modal charges for persistent storage ($0.10/GB-month)
6. **Consider spot instances**: For non-urgent jobs, spot pricing can reduce costs significantly

## Checkpointing and Disaster Recovery

For long-running simulations (>1 hour), checkpointing is essential to protect against:
- Job timeouts (default: 4 hours)
- Hardware failures or preemption
- Network interruptions
- Budget overruns requiring manual cancellation

### Enabling Checkpoints

Add to your YAML configuration:

```yaml
time_integration:
  n_steps: 10000
  checkpoint_interval: 500  # Save checkpoint every 500 steps
  save_interval: 50          # Save diagnostics every 50 steps
```

This will create checkpoints at:
- `/results/<output_dir>/checkpoint_step000500.h5`
- `/results/<output_dir>/checkpoint_step001000.h5`
- etc.

### Resuming from Checkpoint

If a simulation is interrupted:

1. **Download the latest checkpoint**:
   ```bash
   # Find the latest checkpoint
   modal volume ls gandalf-results/<your_run_dir>

   # Download it
   modal volume get gandalf-results/<your_run_dir>/checkpoint_step002000.h5 ./checkpoint.h5
   ```

2. **Create a resume configuration**:
   ```yaml
   # resume_config.yaml
   name: resumed_simulation
   initial_condition:
     type: checkpoint  # Load from checkpoint instead of fresh initialization
     checkpoint_path: ./checkpoint.h5

   time_integration:
     n_steps: 8000  # Continue for remaining 8000 steps (10000 - 2000)
     # ... same parameters as original
   ```

3. **Submit resume job**:
   ```bash
   python scripts/modal_submit.py submit resume_config.yaml --gpu
   ```

### Best Practices

- **Checkpoint frequency**: Balance between recovery time and I/O overhead
  - High-resolution (256³+): Every 250-500 steps (~15-30 min of compute)
  - Low-resolution (64³): Every 1000-2000 steps
- **Monitor job progress**: Check logs periodically for long runs
- **Set realistic timeouts**: Estimate 2× expected runtime as safety margin
- **Test checkpoint/resume**: Verify it works on a short test run before multi-hour production runs

**Note**: Checkpoint I/O functions (`save_checkpoint()`, `load_checkpoint()`) are available in `krmhd.io`. However, config-based `type: checkpoint` for automatic checkpoint resume is not yet implemented. To resume from checkpoints, you currently need to:
1. Use `save_checkpoint()` to periodically save state during simulation (already supported in Modal runs)
2. Manually modify your simulation script to call `load_checkpoint()` to resume from a specific checkpoint file
3. Re-submit the modified script to Modal

Future enhancement: Add `initial_condition.type: checkpoint` with `checkpoint_path` parameter for automatic resume support.

## Troubleshooting

### Issue: "modal: command not found"

**Solution**: Install Modal CLI
```bash
pip install modal
```

### Issue: "Authentication required"

**Solution**: Authenticate with Modal
```bash
modal token new
```

### Issue: "Config file not found"

**Solution**: Use absolute or relative paths from repository root
```bash
# From gandalf/ directory
python scripts/modal_submit.py submit configs/driven_turbulence.yaml

# Or use absolute path
python scripts/modal_submit.py submit /path/to/gandalf/configs/driven_turbulence.yaml
```

### Issue: "Out of memory" on GPU

**Solution**: Reduce resolution or increase GPU memory
```python
# In modal_app.py, change GPU type
@app.function(
    gpu="A100",  # Larger GPU with more memory
    ...
)
```

### Issue: Job timeout

**Solution**: Increase timeout in `modal_app.py`
```python
@app.function(
    timeout=7200,  # 2 hours (default: 4 hours)
    ...
)
```

### Issue: Slow FFT performance on GPU

**Solution**: Ensure JAX is using GPU backend
```python
# In modal_app.py, verify GPU detection
import jax
print(f"JAX backend: {jax.default_backend()}")  # Should be 'gpu'
print(f"Devices: {jax.devices()}")  # Should show GPU
```

### Issue: Cannot download results

**Solution**: Check volume name and path
```bash
# List volumes
modal volume list

# List contents
modal volume ls gandalf-results

# Verify path exists
modal volume ls gandalf-results/driven_turbulence_20240115_120000
```

## Best Practices

### 1. Version Control Your Configs

```bash
# Track simulation configs in git
git add configs/modal_high_res.yaml
git commit -m "Add high-res config for Modal runs"
```

### 2. Tag Experiments

```bash
# Use descriptive names for sweeps
python scripts/modal_submit.py sweep configs/driven_turbulence.yaml \
    --name paper_fig3_resolution_study \
    --param grid.Nx 64 128 256
```

### 3. Save Metadata

```yaml
# Add descriptions to configs
description: "Figure 3: Resolution convergence test for critical balance validation"
```

### 4. Test Locally First

```bash
# Quick local test (low resolution)
python scripts/run_simulation.py configs/test_quick.yaml

# Then scale up on Modal
python scripts/modal_submit.py submit configs/modal_high_res.yaml --gpu
```

### 5. Monitor Long Runs

```bash
# For runs >1 hour, use screen/tmux or run in background
nohup python scripts/modal_submit.py submit configs/modal_high_res.yaml --gpu > modal.log 2>&1 &

# Check logs
tail -f modal.log
```

## Advanced Usage

### Custom Modal Functions

Add new functions to `modal_app.py`:

```python
@app.function(image=image, volumes={"/results": volume})
def analyze_sweep(sweep_dir: str) -> dict:
    """Analyze parameter sweep results."""
    import json
    from pathlib import Path

    # Load metadata
    metadata_path = Path(f"/results/{sweep_dir}/sweep_metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Analyze results
    results = metadata['results']
    best_run = max(results, key=lambda r: r['final_energy_total'])

    return {
        'best_config': best_run['output_dir'],
        'best_energy': best_run['final_energy_total'],
        'n_runs': len(results),
    }
```

### Integration with Jupyter

```python
# In Jupyter notebook
import modal

# Connect to Modal
stub = modal.Stub.lookup("gandalf-krmhd")

# Submit job from notebook
with open("configs/driven_turbulence.yaml") as f:
    config = f.read()

result = stub.run_simulation_remote.remote(config, verbose=False)

print(f"Final energy: {result['final_energy_total']:.6e}")
```

### Cloud Storage Integration

Mount S3/GCS buckets for large-scale data:

```python
# In modal_app.py
from modal import CloudBucketMount

bucket = CloudBucketMount(
    "my-bucket",
    secret=modal.Secret.from_name("my-aws-secret")
)

@app.function(
    image=image,
    volumes={"/results": volume},
    cloud_bucket_mounts={"/data": bucket},
)
def run_with_external_data(...):
    # Access data from /data
    ...
```

## Support

- **Modal Documentation**: https://modal.com/docs
- **GANDALF Issues**: https://github.com/anjor/gandalf/issues
- **Modal Community**: https://modal.com/slack

## Next Steps

1. **Run the examples**: Try the quick start guide above
2. **Customize configs**: Create your own YAML files for specific studies
3. **Scale up**: Move to high-resolution GPU runs for production
4. **Automate**: Integrate Modal submissions into your workflow scripts

Happy computing! 🚀
