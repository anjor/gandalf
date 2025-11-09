# Parameter Scans

Systematic parameter studies are essential for understanding turbulent plasma behavior and validating theoretical predictions. This guide shows how to design, execute, and analyze parameter scans with GANDALF.

## Why Parameter Scans?

**Scientific questions that require parameter scans:**

1. **Resolution convergence:** Is my simulation resolved? (Scan Nx = 32, 64, 128, 256)
2. **Dissipation sensitivity:** How does physics depend on viscosity? (Scan η)
3. **Plasma beta effects:** How does β affect turbulent cascade? (Scan β = 0.01-100)
4. **Temperature ratio:** Role of kinetic effects? (Scan τ = Tᵢ/Tₑ = 1-10)
5. **Forcing strength:** Energy injection vs dissipation balance? (Scan amplitude)
6. **Collisionality:** Collisionless vs fluid limit? (Scan ν)

**Good parameter scan characteristics:**
- **Logarithmic spacing** for wide ranges (β = 0.01, 0.1, 1, 10, 100)
- **Linear spacing** for narrow ranges (Nx = 32, 64, 96, 128)
- **Multiple realizations** for stochastic forcing (same parameters, different seeds)
- **Fixed diagnostics** across runs (same output times for comparison)

## Planning a Parameter Study

### Step 1: Define the Question

**Bad:** "Let's scan β and see what happens."

**Good:** "Does the spectral slope E(k⊥) ~ k⊥^α depend on plasma β? Theory predicts α = -5/3 for all β in KRMHD regime."

**Actionable plan:**
- Scan β = [0.1, 1.0, 10.0, 100.0]
- Fixed resolution: 128³
- Fixed dissipation: η=2.0, r=2
- Run to steady state (50 τ_A)
- Time-average last 20 τ_A
- Measure spectral slope in inertial range (k=2-10)
- Compare slopes across β values

### Step 2: Choose Parameter Ranges

**Resolution scan:**
```
Nx ∈ [32, 64, 128, 256]
```
- Factor of 2 spacing (standard for convergence studies)
- Computational cost scales as (Nx)³ log(Nx)
- 256³ may require HPC resources

**Plasma beta scan:**
```
β ∈ [0.01, 0.1, 1.0, 10.0, 100.0]
```
- Logarithmic spacing covers 4 orders of magnitude
- β ~ 1 is typical astrophysical regime
- Very high β (>100) may violate RMHD ordering

**Dissipation scan:**
```
η ∈ [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
```
- Linear or logarithmic spacing
- Must satisfy η·dt < 50 (overflow constraint)
- Lower η → longer spin-up time, may need longer runs

**Temperature ratio scan:**
```
τ = Tᵢ/Tₑ ∈ [1.0, 2.0, 5.0, 10.0]
```
- Affects Landau damping rates
- τ = 1 is baseline (equal temperatures)

**Forcing amplitude scan:**
```
amplitude ∈ [0.01, 0.02, 0.05, 0.1]
```
- Must balance with dissipation (see Issue #82)
- Higher amplitude → stronger turbulence, but may become unstable

### Step 3: Estimate Resources

**Time per run:**
- 32³: minutes
- 64³: ~20 minutes
- 128³: ~1-2 hours
- 256³: ~8-16 hours (estimated)

**For a β scan (4 values at 128³):**
- 4 runs × 1.5 hours = 6 hours total (serial)
- 1.5 hours (parallel, 4 cores/GPUs)

**Storage:**
- Checkpoints: ~200 MB/checkpoint × 5 checkpoints = 1 GB/run
- Diagnostics: ~10 MB/run
- Total: ~5 GB for scan (manageable)

**Recommendation:** Start with coarse scan (32³ or 64³) to verify parameters, then run production at 128³.

## Generating Configurations

### Method 1: Script to Generate YAML Files

Create `generate_beta_scan.py`:

```python
#!/usr/bin/env python3
"""Generate YAML configs for plasma beta scan."""

import yaml
import numpy as np
from pathlib import Path

# Parameter values
beta_values = [0.1, 1.0, 10.0, 100.0]
resolution = 128
eta = 2.0
hyper_r = 2
t_final = 50.0
averaging_start = 30.0

# Template config
template = {
    "simulation": {
        "name": None,  # Will be filled
        "output_dir": None,
    },
    "grid": {
        "Nx": resolution,
        "Ny": resolution,
        "Nz": resolution,
        "Lx": 2 * np.pi,
        "Ly": 2 * np.pi,
        "Lz": 2 * np.pi,
    },
    "initial_condition": {
        "type": "random_spectrum",
        "energy": 0.01,
        "alpha": 5/3,
    },
    "physics": {
        "beta": None,  # Will be filled
        "temperature_ratio": 1.0,
    },
    "timestepping": {
        "dt": 0.005,
        "t_final": t_final,
        "cfl_safety_factor": 0.5,
    },
    "dissipation": {
        "eta": eta,
        "nu": eta,
        "hyper_r": hyper_r,
        "hyper_n": 2,
    },
    "forcing": {
        "enabled": True,
        "amplitude": 0.05,
        "k_min": 1,
        "k_max": 2,
    },
    "diagnostics": {
        "energy_interval": 0.5,
        "spectrum_interval": 2.0,
        "checkpoint_interval": 10.0,
        "save_diagnostics": True,
        "diagnostic_interval": 5,
    },
}

# Generate configs
output_dir = Path("configs/beta_scan")
output_dir.mkdir(exist_ok=True)

for beta in beta_values:
    config = template.copy()
    config["simulation"]["name"] = f"beta_scan_beta{beta:.2f}"
    config["simulation"]["output_dir"] = f"output/beta_scan/beta{beta:.2f}"
    config["physics"]["beta"] = float(beta)

    filename = output_dir / f"beta{beta:.2f}.yaml"
    with open(filename, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Generated {filename}")

print(f"\nCreated {len(beta_values)} configs in {output_dir}/")
print(f"Run with: uv run python run_simulation.py --config {output_dir}/beta0.10.yaml")
```

**Run:**
```bash
uv run python generate_beta_scan.py
```

**Output:**
```
configs/beta_scan/
├── beta0.10.yaml
├── beta1.00.yaml
├── beta10.00.yaml
└── beta100.00.yaml
```

### Method 2: Template Modification

Start with an existing config:
```bash
cp configs/driven_turbulence.yaml configs/beta_scan/beta1.00.yaml
```

Edit manually or with `sed`:
```bash
sed 's/beta: 1.0/beta: 10.0/' configs/beta_scan/beta1.00.yaml > configs/beta_scan/beta10.00.yaml
```

**Advantage:** Quick for small scans (few values)
**Disadvantage:** Error-prone, hard to maintain for large scans

## Running Batch Jobs

### Method 1: Shell Script (Serial)

Create `run_beta_scan.sh`:

```bash
#!/bin/bash
# Run beta scan serially

set -e  # Exit on error

CONFIGS=(
    configs/beta_scan/beta0.10.yaml
    configs/beta_scan/beta1.00.yaml
    configs/beta_scan/beta10.00.yaml
    configs/beta_scan/beta100.00.yaml
)

for config in "${CONFIGS[@]}"; do
    echo "========================================="
    echo "Running: $config"
    echo "========================================="
    uv run python run_simulation.py --config "$config"
    echo "Completed: $config"
    echo ""
done

echo "All simulations completed!"
```

**Run:**
```bash
chmod +x run_beta_scan.sh
./run_beta_scan.sh
```

**Advantages:**
- Simple, no dependencies
- Easy to monitor progress

**Disadvantages:**
- Serial execution (slow)
- No parallelization

### Method 2: GNU Parallel

```bash
# Install GNU parallel (if needed)
brew install parallel  # macOS
# or: apt install parallel  # Linux

# Run all configs in parallel (4 at a time)
ls configs/beta_scan/*.yaml | \
    parallel -j 4 uv run python run_simulation.py --config {}
```

**Advantages:**
- Automatic parallelization
- Progress monitoring
- Resume failed jobs

**Disadvantages:**
- Requires multiple GPUs/cores
- Memory usage multiplied

### Method 3: Python Multiprocessing

Create `run_beta_scan_parallel.py`:

```python
#!/usr/bin/env python3
"""Run beta scan in parallel."""

import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

configs = list(Path("configs/beta_scan").glob("*.yaml"))

def run_simulation(config_path):
    """Run a single simulation."""
    print(f"Starting: {config_path.name}")
    result = subprocess.run(
        ["uv", "run", "python", "run_simulation.py", "--config", str(config_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"FAILED: {config_path.name}")
        print(result.stderr)
        return None
    print(f"Completed: {config_path.name}")
    return config_path

# Run with 4 workers (adjust based on available resources)
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(run_simulation, config) for config in configs]
    for future in as_completed(futures):
        result = future.result()

print("All simulations completed!")
```

**Run:**
```bash
uv run python run_beta_scan_parallel.py
```

### Method 4: HPC Job Arrays (SLURM)

For large scans on clusters, use job arrays:

```bash
#!/bin/bash
#SBATCH --job-name=beta_scan
#SBATCH --array=0-3  # 4 jobs (indices 0, 1, 2, 3)
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/beta_scan_%a.out

# Array of config files
CONFIGS=(
    configs/beta_scan/beta0.10.yaml
    configs/beta_scan/beta1.00.yaml
    configs/beta_scan/beta10.00.yaml
    configs/beta_scan/beta100.00.yaml
)

# Get config for this array task
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}

echo "Running: $CONFIG"
uv run python run_simulation.py --config "$CONFIG"
```

**Submit:**
```bash
sbatch run_beta_scan_slurm.sh
```

**Advantages:**
- Automatic parallelization on cluster
- Resource management by scheduler
- Fault tolerance (resubmit failed jobs)

## Data Organization

### Directory Structure

Organize output by scan parameter:

```
output/
├── beta_scan/
│   ├── beta0.10/
│   │   ├── checkpoint_t00.h5
│   │   ├── checkpoint_t10.h5
│   │   ├── energy_history.h5
│   │   └── diagnostics.h5
│   ├── beta1.00/
│   ├── beta10.00/
│   └── beta100.00/
├── resolution_scan/
│   ├── R032/
│   ├── R064/
│   ├── R128/
│   └── R256/
└── eta_scan/
    ├── eta0.5/
    ├── eta1.0/
    ├── eta2.0/
    └── eta5.0/
```

**Naming convention:**
- Use parameter value in directory name: `beta0.10`, `eta2.0`
- Pad numbers for sorting: `R032`, `R064`, `R128` (not `R32`, `R64`)
- Be consistent across scans

### Metadata Tracking

Create a `scan_metadata.yaml` file:

```yaml
scan_type: "plasma_beta"
date_created: "2025-11-09"
description: "Effect of plasma beta on turbulent cascade spectral slope"

parameters:
  varied:
    - name: "beta"
      values: [0.1, 1.0, 10.0, 100.0]
  fixed:
    - resolution: 128
    - eta: 2.0
    - hyper_r: 2
    - t_final: 50.0

runs:
  - beta: 0.1
    status: "completed"
    output_dir: "output/beta_scan/beta0.10"
    runtime_hours: 1.2
  - beta: 1.0
    status: "completed"
    output_dir: "output/beta_scan/beta1.00"
    runtime_hours: 1.3
  # ... etc
```

**Use for:**
- Tracking completion status
- Recording runtime
- Documenting fixed parameters
- Sharing with collaborators

## Analysis and Comparison

### Loading Results

Create `analyze_beta_scan.py`:

```python
#!/usr/bin/env python3
"""Analyze beta scan results."""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from krmhd.io import load_timeseries, load_checkpoint
from krmhd.diagnostics import energy_spectrum_perpendicular

# Beta values
beta_values = [0.1, 1.0, 10.0, 100.0]
base_dir = Path("output/beta_scan")

# Collect time-averaged spectra
spectra = {}

for beta in beta_values:
    output_dir = base_dir / f"beta{beta:.2f}"

    # Load final checkpoint
    checkpoint_file = output_dir / "checkpoint_t50.h5"
    if not checkpoint_file.exists():
        print(f"Warning: {checkpoint_file} not found, skipping beta={beta}")
        continue

    state, metadata = load_checkpoint(checkpoint_file)

    # Compute spectrum
    E_perp, k_perp = energy_spectrum_perpendicular(state)
    spectra[beta] = {"k": k_perp, "E": E_perp}

    print(f"Loaded beta={beta}: E_total={state.energy():.4f}")

# Plot comparison
fig, ax = plt.subplots(figsize=(8, 6))

for beta in beta_values:
    if beta not in spectra:
        continue

    k = spectra[beta]["k"]
    E = spectra[beta]["E"]

    ax.loglog(k, E, 'o-', label=f"β = {beta}")

# Reference slopes
k_ref = np.logspace(0, 1, 10)
ax.loglog(k_ref, 0.1 * k_ref**(-5/3), 'k--', alpha=0.5, label=r"$k^{-5/3}$")

ax.set_xlabel(r"$k_\perp$")
ax.set_ylabel(r"$E(k_\perp)$")
ax.set_title("Perpendicular Energy Spectrum vs Plasma Beta")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig("beta_scan_spectra.png", dpi=150)
print("Saved: beta_scan_spectra.png")
```

**Run:**
```bash
uv run python analyze_beta_scan.py
```

### Measuring Spectral Slopes

```python
def fit_power_law(k, E, k_min=2, k_max=10):
    """Fit E ~ k^α in specified range."""
    mask = (k >= k_min) & (k <= k_max)
    if mask.sum() < 2:
        return np.nan

    # Log-linear fit: log(E) = α·log(k) + const
    coeffs = np.polyfit(np.log(k[mask]), np.log(E[mask]), deg=1)
    alpha = coeffs[0]  # Slope
    return alpha

# Measure slopes
slopes = {}
for beta in beta_values:
    if beta not in spectra:
        continue
    k = spectra[beta]["k"]
    E = spectra[beta]["E"]
    alpha = fit_power_law(k, E, k_min=2, k_max=10)
    slopes[beta] = alpha
    print(f"β = {beta:6.2f}: α = {alpha:.3f}")

# Expected: α ≈ -5/3 ≈ -1.667 for all β
```

### Energy History Comparison

```python
fig, ax = plt.subplots(figsize=(8, 6))

for beta in beta_values:
    output_dir = base_dir / f"beta{beta:.2f}"
    history_file = output_dir / "energy_history.h5"

    if not history_file.exists():
        continue

    history = load_timeseries(history_file)
    ax.plot(history.times, history.total_energy, label=f"β = {beta}")

ax.set_xlabel("Time (τ_A)")
ax.set_ylabel("Total Energy")
ax.set_title("Energy Evolution vs Plasma Beta")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig("beta_scan_energy_history.png", dpi=150)
```

### Statistical Analysis

For multiple realizations (different random seeds):

```python
# Run 5 realizations for each beta
realizations = 5

# Collect all slopes
slopes_by_beta = {beta: [] for beta in beta_values}

for beta in beta_values:
    for realization in range(realizations):
        output_dir = base_dir / f"beta{beta:.2f}_run{realization}"
        # ... load and compute slope ...
        slopes_by_beta[beta].append(alpha)

# Compute statistics
for beta in beta_values:
    slopes = np.array(slopes_by_beta[beta])
    mean_slope = np.mean(slopes)
    std_slope = np.std(slopes)
    print(f"β = {beta}: α = {mean_slope:.3f} ± {std_slope:.3f}")
```

## Example Workflows

### Workflow 1: Resolution Convergence Study

**Question:** Is 128³ sufficient resolution, or do I need 256³?

**Parameters:**
```python
resolutions = [32, 64, 128, 256]
eta = 2.0  # Same dissipation for all
hyper_r = 2
t_final = 20.0  # Shorter run for testing
```

**Generate configs:**
```python
for Nx in resolutions:
    config["grid"]["Nx"] = Nx
    config["grid"]["Ny"] = Nx
    config["grid"]["Nz"] = Nx
    # Adjust timestep (finer grid → smaller dt)
    config["timestepping"]["dt"] = 0.02 / (Nx / 32)
    # Save to file...
```

**Analysis:**
- Compute E(k⊥) for each resolution
- Check if 128³ and 256³ agree in inertial range (k=2-10)
- If yes → 128³ is converged
- If no → need 256³ or higher

**Expected:**
- 32³: Too coarse, spectrum deviates at k > 5
- 64³: Marginal, some deviation at k > 8
- 128³: Well-resolved for k < 16
- 256³: Overkill for most studies (unless studying k > 20)

### Workflow 2: Dissipation Parameter Scan

**Question:** What η value balances forcing and dissipation?

**Parameters:**
```python
eta_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
resolution = 64  # Fixed
force_amplitude = 0.05  # Fixed
```

**Diagnostics to monitor:**
- Energy evolution: Should reach steady plateau
- Spectral slope: Should maintain k^(-5/3)
- High-k energy fraction: Should be <1% (no pile-up)

**Analysis:**
```python
for eta in eta_values:
    history = load_timeseries(f"output/eta_scan/eta{eta}/energy_history.h5")

    # Check for steady state (last 10 τ_A)
    E_late = history.total_energy[-200:]  # Last 10 τ_A at dt=0.005
    dE = np.std(E_late) / np.mean(E_late)

    if dE < 0.05:
        print(f"η = {eta}: Steady state reached (ΔE/⟨E⟩ = {dE:.2%})")
    else:
        print(f"η = {eta}: NOT steady (ΔE/⟨E⟩ = {dE:.2%})")
```

**Interpretation:**
- If ΔE/⟨E⟩ < 5%: Good steady state
- If ΔE/⟨E⟩ > 10%: Either not equilibrated or unstable
- If energy still growing: Increase η or reduce forcing

### Workflow 3: Plasma Beta Scan

**Question:** How does kinetic/magnetic pressure ratio affect cascade?

**Parameters:**
```python
beta_values = [0.1, 1.0, 10.0]
resolution = 128
t_final = 50.0
```

**Physical expectation:**
- KRMHD prediction: Spectral slope independent of β (as long as ε << 1)
- But: Dissipation scales and Landau damping rates may change

**Analysis:**
- Measure spectral slopes (should all be -5/3)
- Measure energy partition (magnetic vs kinetic)
- Check Hermite moment distribution (kinetic effects stronger at high β?)

### Workflow 4: Temperature Ratio Scan

**Question:** Effect of τ = Tᵢ/Tₑ on Landau damping?

**Parameters:**
```python
tau_values = [1.0, 2.0, 5.0, 10.0]
```

**Analysis:**
- Compute Hermite moment energy: E_m vs m
- Higher τ → different velocity-space structure
- Measure phase mixing vs phase unmixing energy

## Visualization Best Practices

### Multi-Panel Comparison

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Spectra
ax = axes[0, 0]
for beta in beta_values:
    ax.loglog(k, E, label=f"β={beta}")
ax.set_xlabel(r"$k_\perp$")
ax.set_ylabel(r"$E(k_\perp)$")
ax.legend()

# Panel 2: Energy history
ax = axes[0, 1]
for beta in beta_values:
    ax.plot(times, energy, label=f"β={beta}")
ax.set_xlabel("Time")
ax.set_ylabel("Energy")

# Panel 3: Spectral slopes
ax = axes[1, 0]
ax.plot(beta_values, slopes, 'o-')
ax.axhline(-5/3, color='k', linestyle='--', label=r"$-5/3$")
ax.set_xlabel("β")
ax.set_ylabel("Spectral slope α")

# Panel 4: Magnetic fraction
ax = axes[1, 1]
ax.plot(beta_values, mag_fractions, 'o-')
ax.set_xlabel("β")
ax.set_ylabel("Magnetic energy fraction")

plt.tight_layout()
plt.savefig("beta_scan_summary.png", dpi=150)
```

### Heatmaps for 2D Scans

If scanning two parameters (e.g., β and τ):

```python
beta_values = [0.1, 1.0, 10.0]
tau_values = [1.0, 2.0, 5.0, 10.0]

# Create 2D grid of slopes
slopes_2d = np.zeros((len(tau_values), len(beta_values)))

for i, tau in enumerate(tau_values):
    for j, beta in enumerate(beta_values):
        # Load and compute slope...
        slopes_2d[i, j] = alpha

# Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(slopes_2d, aspect='auto', origin='lower', cmap='RdBu_r')
ax.set_xticks(range(len(beta_values)))
ax.set_xticklabels([f"{b}" for b in beta_values])
ax.set_yticks(range(len(tau_values)))
ax.set_yticklabels([f"{t}" for t in tau_values])
ax.set_xlabel("β")
ax.set_ylabel("τ = Tᵢ/Tₑ")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Spectral slope α")
plt.title("Spectral Slope vs β and τ")
plt.savefig("beta_tau_scan_heatmap.png", dpi=150)
```

## Tips and Tricks

### 1. Start Small

Before running 256³ × 10 realizations × 5 parameters (expensive!):
- Test at 32³ with 1 realization
- Verify analysis scripts work
- Check that parameters give sensible results
- Then scale up to production resolution

### 2. Checkpoint Everything

Always enable checkpointing:
```yaml
diagnostics:
  checkpoint_interval: 10.0
```

If a run crashes, you can resume instead of starting over.

### 3. Monitor Progress

Add logging to batch scripts:
```bash
echo "Started: $(date)" >> logs/progress.txt
uv run python run_simulation.py --config $CONFIG
echo "Finished: $(date)" >> logs/progress.txt
```

Or use `tee` to save output:
```bash
uv run python run_simulation.py --config $CONFIG 2>&1 | tee logs/run_beta0.10.log
```

### 4. Sanity Check Early

After first few runs complete:
- Load results and plot
- Check if trends make sense
- If something looks wrong, fix before continuing

Don't wait until all 100 runs finish to discover a bug in your config!

### 5. Document Everything

Keep a research notebook (Markdown or Jupyter):
```markdown
# Beta Scan - 2025-11-09

## Goal
Measure dependence of spectral slope on plasma beta.

## Parameters
- β ∈ [0.1, 1.0, 10.0, 100.0]
- Resolution: 128³
- Dissipation: η=2.0, r=2

## Results
- All runs completed successfully
- Spectral slopes: α = -1.67 ± 0.05 (consistent with theory)
- No significant β dependence (as expected for KRMHD)

## Next Steps
- Test higher β (β=1000) to find breakdown
- Repeat at 256³ for convergence check
```

### 6. Archive Raw Data

After analysis, compress and archive:
```bash
# Compress output directories
tar -czf beta_scan_output.tar.gz output/beta_scan/

# Archive to external storage
rsync -av beta_scan_output.tar.gz /Volumes/ExternalDrive/GANDALF_data/
```

HDF5 files are already compressed, but `tar.gz` adds metadata preservation.

## Common Pitfalls

### 1. Not Allowing Spin-Up Time

**Problem:** Measuring diagnostics before steady state reached.

**Solution:** Always discard early times (first 10-20 τ_A) and average over late times.

### 2. Inconsistent Diagnostics

**Problem:** Different output times across runs → can't compare.

**Solution:** Use same `spectrum_interval` and `t_final` in all configs.

### 3. Forgetting Random Seeds

**Problem:** "Multiple realizations" but using same random seed → identical results!

**Solution:** Explicitly vary seed:
```yaml
random_seed: 12345  # Change for each realization
```

### 4. Overwriting Output

**Problem:** Running two configs with same `output_dir` → data overwritten.

**Solution:** Unique output directory per run (include parameter in name).

### 5. Ignoring Failed Runs

**Problem:** One run crashes silently, results incomplete.

**Solution:** Check metadata/logs for completion status before analyzing.

## Summary

**Parameter scan workflow:**

1. **Plan:** Define question, choose parameter ranges
2. **Generate:** Create configs (scripted or manual)
3. **Run:** Execute serially, parallel, or on HPC
4. **Organize:** Consistent directory structure, metadata
5. **Analyze:** Load results, compute diagnostics, compare
6. **Visualize:** Multi-panel plots, heatmaps, trends
7. **Document:** Keep notebook, archive data

**Key tools:**
- `generate_configs.py`: Scripted config generation
- `run_batch.sh`: Serial execution
- `GNU parallel`: Parallel execution
- `analyze_scan.py`: Automated analysis
- `scan_metadata.yaml`: Tracking and documentation

**Best practices:**
- Start small (32³ test before 256³ production)
- Checkpoint frequently (resume on failure)
- Monitor progress (logs, intermediate plots)
- Sanity check early (fix bugs before completing scan)
- Document everything (future you will thank present you)

## Further Reading

- [Running Simulations](running_simulations.md) - Individual run workflows
- [Physics Validity](physics_validity.md) - Valid parameter ranges
- [Numerical Methods](numerical_methods.md) - Resolution requirements and scaling

For HPC cluster usage, consult your institution's documentation on job submission (SLURM, PBS, etc.).
