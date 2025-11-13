#!/usr/bin/env python3
"""
Modal App for Running GANDALF KRMHD Simulations

This module provides a Modal-based cloud computing interface for running
GANDALF simulations. It supports:

- GPU-accelerated JAX computations
- Configurable compute resources (CPU, GPU, memory)
- YAML configuration upload
- Automatic result download to Modal volumes
- Parallel parameter sweeps

Usage:
    # Deploy the app
    modal deploy modal_app.py

    # Run a simulation locally (for testing)
    modal run modal_app.py --config configs/driven_turbulence.yaml

    # Run on Modal cloud
    modal run modal_app.py::run_simulation --config-path configs/driven_turbulence.yaml

    # Run parameter sweep
    modal run modal_app.py::run_parameter_sweep --base-config configs/driven_turbulence.yaml

Requirements:
    pip install modal

Setup:
    1. Create Modal account: https://modal.com
    2. Install Modal CLI: pip install modal
    3. Authenticate: modal token new
    4. Deploy: modal deploy modal_app.py
"""

import io
import time
from pathlib import Path
from typing import Any, Optional

import modal

# =============================================================================
# Modal Configuration
# =============================================================================

# Modal app definition
app = modal.App("gandalf-krmhd")

# Create a persistent volume for storing results
volume = modal.Volume.from_name(
    "gandalf-results",
    create_if_missing=True
)

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "jax[cpu]>=0.4.20",  # Start with CPU, can upgrade to GPU
        "h5py>=3.11.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0.0",
        "scipy>=1.10.0",
    )
    # Copy the GANDALF source code into the image
    .copy_local_dir("src", "/root/gandalf/src")
    .copy_local_file("pyproject.toml", "/root/gandalf/pyproject.toml")
    .workdir("/root/gandalf")
    .pip_install("-e .")  # Install GANDALF in editable mode
)

# GPU image (optional - use for large simulations)
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "jax[cuda12_pip]>=0.4.20",  # JAX with CUDA support
        "h5py>=3.11.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0.0",
        "scipy>=1.10.0",
    )
    .copy_local_dir("src", "/root/gandalf/src")
    .copy_local_file("pyproject.toml", "/root/gandalf/pyproject.toml")
    .workdir("/root/gandalf")
    .pip_install("-e .")
)


# =============================================================================
# Core Simulation Function
# =============================================================================

@app.function(
    image=image,
    volumes={"/results": volume},
    timeout=3600 * 4,  # 4 hour timeout
    cpu=8.0,  # 8 CPU cores
    memory=32768,  # 32 GB RAM
)
def run_simulation_remote(
    config_yaml: str,
    output_subdir: Optional[str] = None,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Run GANDALF simulation on Modal with CPU.

    This function is executed in the Modal cloud environment.

    Args:
        config_yaml: YAML configuration as string
        output_subdir: Optional subdirectory within /results for outputs
        verbose: Print progress information

    Returns:
        Dictionary with simulation results and metadata
    """
    import tempfile
    import yaml
    from pathlib import Path
    import numpy as np

    from krmhd.config import SimulationConfig
    from krmhd.io import save_checkpoint, save_timeseries

    # Import the run_simulation function from scripts
    import sys
    sys.path.insert(0, '/root/gandalf')
    from scripts.run_simulation import run_simulation

    # Parse configuration
    config_dict = yaml.safe_load(config_yaml)
    config = SimulationConfig(**config_dict)

    # Override output directory to use Modal volume
    if output_subdir:
        output_path = f"/results/{output_subdir}"
    else:
        # Use timestamp-based directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"/results/{config.name}_{timestamp}"

    # Update config with Modal output path
    config = config.model_copy(
        update={'io': config.io.model_copy(update={'output_dir': output_path, 'overwrite': True})}
    )

    if verbose:
        print(f"\n{'='*70}")
        print(f"Running on Modal Cloud")
        print(f"Output directory: {output_path}")
        print(f"{'='*70}\n")

    # Run simulation
    start_time = time.time()
    state, energy_history, grid = run_simulation(config, verbose=verbose)
    elapsed = time.time() - start_time

    # Commit volume changes
    volume.commit()

    if verbose:
        print(f"\n{'='*70}")
        print(f"Simulation complete!")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Results saved to: {output_path}")
        print(f"{'='*70}\n")

    # Return metadata
    from krmhd import energy as compute_energy
    final_energy = compute_energy(state)

    return {
        'status': 'success',
        'output_dir': output_path,
        'elapsed_time': elapsed,
        'config_name': config.name,
        'resolution': (config.grid.Nx, config.grid.Ny, config.grid.Nz),
        'n_steps': config.time_integration.n_steps,
        'final_time': float(state.time),
        'final_energy_total': float(final_energy['total']),
        'final_energy_magnetic': float(final_energy['magnetic']),
        'final_energy_kinetic': float(final_energy['kinetic']),
    }


@app.function(
    image=gpu_image,
    volumes={"/results": volume},
    timeout=3600 * 4,  # 4 hour timeout
    gpu="T4",  # NVIDIA T4 GPU (can use "A10G", "A100", etc.)
    cpu=4.0,
    memory=16384,  # 16 GB RAM
)
def run_simulation_gpu(
    config_yaml: str,
    output_subdir: Optional[str] = None,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Run GANDALF simulation on Modal with GPU acceleration.

    This function is identical to run_simulation_remote but uses GPU resources.

    Args:
        config_yaml: YAML configuration as string
        output_subdir: Optional subdirectory within /results for outputs
        verbose: Print progress information

    Returns:
        Dictionary with simulation results and metadata
    """
    import tempfile
    import yaml
    from pathlib import Path
    import numpy as np
    import jax

    # Verify GPU is available
    if verbose:
        print(f"\nJAX devices: {jax.devices()}")
        print(f"Default backend: {jax.default_backend()}\n")

    from krmhd.config import SimulationConfig
    from krmhd.io import save_checkpoint, save_timeseries

    # Import the run_simulation function from scripts
    import sys
    sys.path.insert(0, '/root/gandalf')
    from scripts.run_simulation import run_simulation

    # Parse configuration
    config_dict = yaml.safe_load(config_yaml)
    config = SimulationConfig(**config_dict)

    # Override output directory to use Modal volume
    if output_subdir:
        output_path = f"/results/{output_subdir}"
    else:
        # Use timestamp-based directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"/results/{config.name}_{timestamp}"

    # Update config with Modal output path
    config = config.model_copy(
        update={'io': config.io.model_copy(update={'output_dir': output_path, 'overwrite': True})}
    )

    if verbose:
        print(f"\n{'='*70}")
        print(f"Running on Modal Cloud with GPU")
        print(f"Output directory: {output_path}")
        print(f"{'='*70}\n")

    # Run simulation
    start_time = time.time()
    state, energy_history, grid = run_simulation(config, verbose=verbose)
    elapsed = time.time() - start_time

    # Commit volume changes
    volume.commit()

    if verbose:
        print(f"\n{'='*70}")
        print(f"Simulation complete!")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Results saved to: {output_path}")
        print(f"{'='*70}\n")

    # Return metadata
    from krmhd import energy as compute_energy
    final_energy = compute_energy(state)

    return {
        'status': 'success',
        'output_dir': output_path,
        'elapsed_time': elapsed,
        'config_name': config.name,
        'resolution': (config.grid.Nx, config.grid.Ny, config.grid.Nz),
        'n_steps': config.time_integration.n_steps,
        'final_time': float(state.time),
        'final_energy_total': float(final_energy['total']),
        'final_energy_magnetic': float(final_energy['magnetic']),
        'final_energy_kinetic': float(final_energy['kinetic']),
        'backend': 'gpu',
    }


# =============================================================================
# Parameter Sweep Functions
# =============================================================================

@app.function(
    image=image,
    volumes={"/results": volume},
    timeout=3600 * 8,  # 8 hour timeout for sweeps
)
def run_parameter_sweep(
    base_config_yaml: str,
    parameters: dict[str, list],
    sweep_name: Optional[str] = None,
    use_gpu: bool = False,
) -> list[dict[str, Any]]:
    """
    Run a parameter sweep in parallel on Modal.

    Args:
        base_config_yaml: Base YAML configuration as string
        parameters: Dictionary mapping parameter paths to lists of values
                   e.g., {'physics.eta': [0.01, 0.02, 0.05], 'grid.Nx': [64, 128]}
        sweep_name: Optional name for the sweep (used in output paths)
        use_gpu: Whether to use GPU instances for each run

    Returns:
        List of result dictionaries from each simulation
    """
    import yaml
    import itertools

    # Generate all parameter combinations
    param_names = list(parameters.keys())
    param_values = list(parameters.values())
    combinations = list(itertools.product(*param_values))

    print(f"\nParameter sweep: {len(combinations)} combinations")
    print(f"Parameters: {param_names}")
    print(f"Using GPU: {use_gpu}\n")

    # Create sweep directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sweep_dir = sweep_name or f"sweep_{timestamp}"

    # Submit all jobs in parallel
    results = []
    func = run_simulation_gpu if use_gpu else run_simulation_remote

    for i, combo in enumerate(combinations):
        # Update config with parameter values
        config_dict = yaml.safe_load(base_config_yaml)

        for param_name, param_value in zip(param_names, combo):
            # Navigate nested dict structure (e.g., 'physics.eta')
            keys = param_name.split('.')
            d = config_dict
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = param_value

        # Convert back to YAML
        updated_yaml = yaml.safe_dump(config_dict)

        # Create unique output directory for this run
        param_str = "_".join(f"{k.split('.')[-1]}={v}" for k, v in zip(param_names, combo))
        output_subdir = f"{sweep_dir}/run_{i:04d}_{param_str}"

        # Submit job (Modal will parallelize automatically)
        result = func.remote(updated_yaml, output_subdir=output_subdir, verbose=False)
        results.append(result)

    # Wait for all jobs to complete and collect results
    completed_results = [r.get() for r in results]

    # Save sweep metadata
    import json
    metadata = {
        'sweep_name': sweep_dir,
        'timestamp': timestamp,
        'n_runs': len(combinations),
        'parameters': {k: list(v) for k, v in parameters.items()},
        'use_gpu': use_gpu,
        'results': completed_results,
    }

    metadata_path = Path(f"/results/{sweep_dir}/sweep_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    volume.commit()

    print(f"\nSweep complete! Results saved to /results/{sweep_dir}")
    return completed_results


# =============================================================================
# Utility Functions
# =============================================================================

@app.function(
    image=image,
    volumes={"/results": volume},
)
def list_results() -> list[str]:
    """List all result directories in the Modal volume."""
    from pathlib import Path

    results_dir = Path("/results")
    if not results_dir.exists():
        return []

    subdirs = [str(p.relative_to(results_dir)) for p in results_dir.iterdir() if p.is_dir()]
    return sorted(subdirs)


@app.function(
    image=image,
    volumes={"/results": volume},
)
def download_results(result_dir: str, local_path: str) -> None:
    """
    Download results from Modal volume to local machine.

    Args:
        result_dir: Directory name within /results to download
        local_path: Local path to save results
    """
    import shutil
    from pathlib import Path

    source = Path(f"/results/{result_dir}")
    if not source.exists():
        raise FileNotFoundError(f"Result directory not found: {source}")

    # This will be executed on Modal, so we need to return the data
    # In practice, you'd use modal.CloudBucketMount or download via CLI
    print(f"To download results, use:")
    print(f"  modal volume get gandalf-results {result_dir} {local_path}")


# =============================================================================
# Local Entry Points (for testing)
# =============================================================================

@app.local_entrypoint()
def main(
    config_path: str = "configs/driven_turbulence.yaml",
    use_gpu: bool = False,
    output_subdir: Optional[str] = None,
):
    """
    Local entry point for running simulations from the CLI.

    Usage:
        modal run modal_app.py --config-path configs/driven_turbulence.yaml
        modal run modal_app.py --config-path configs/driven_turbulence.yaml --use-gpu
    """
    from pathlib import Path

    # Read config file
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config_yaml = f.read()

    # Run on Modal
    print(f"\nSubmitting simulation to Modal...")
    print(f"Config: {config_path}")
    print(f"GPU: {use_gpu}\n")

    func = run_simulation_gpu if use_gpu else run_simulation_remote
    result = func.remote(config_yaml, output_subdir=output_subdir, verbose=True)

    print("\n" + "="*70)
    print("Simulation complete!")
    print("="*70)
    print(f"Status: {result['status']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Elapsed time: {result['elapsed_time']:.1f}s")
    print(f"Final time: {result['final_time']:.4f}")
    print(f"Final energy: {result['final_energy_total']:.6e}")
    print("\nTo download results:")
    print(f"  modal volume get gandalf-results {result['output_dir'].replace('/results/', '')} ./local_results")
    print("="*70)


if __name__ == "__main__":
    # This allows testing locally without Modal
    import sys
    print("This script is designed to run on Modal.")
    print("Usage:")
    print("  modal deploy modal_app.py")
    print("  modal run modal_app.py --config-path configs/driven_turbulence.yaml")
    sys.exit(0)
