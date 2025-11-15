#!/usr/bin/env python3
"""
Modal App for Running GANDALF KRMHD Simulations

This module provides a Modal-based cloud computing interface for running
GANDALF simulations. It supports:

- GPU-accelerated JAX computations
- Configurable compute resources (CPU, GPU, memory)
- YAML configuration upload
- Automatic result download to Modal volumes
- Parallel parameter sweeps with fault tolerance

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

import os
import time
from pathlib import Path
from typing import Any, Optional

import modal

# =============================================================================
# Configuration Constants
# =============================================================================

# Timeout values in seconds
TIMEOUT_SINGLE_SIMULATION = int(os.environ.get("MODAL_TIMEOUT_SINGLE", 3600 * 4))  # 4 hours
TIMEOUT_PARAMETER_SWEEP = int(os.environ.get("MODAL_TIMEOUT_SWEEP", 3600 * 8))  # 8 hours
TIMEOUT_SWEEP_JOB_GET = int(os.environ.get("MODAL_TIMEOUT_JOB_GET", TIMEOUT_SINGLE_SIMULATION + 600))  # Job timeout + 10 min buffer

# Resource defaults
DEFAULT_CPU_CORES = 8.0
DEFAULT_CPU_MEMORY_MB = 32768  # 32 GB
DEFAULT_GPU_CORES = 4.0
DEFAULT_GPU_MEMORY_MB = 16384  # 16 GB
DEFAULT_GPU_TYPE = "T4"

# Volume name (can be overridden via environment variable)
VOLUME_NAME = os.environ.get("MODAL_VOLUME_NAME", "gandalf-results")

# JAX CUDA version (configurable for different Modal GPU types)
# CUDA 12 is the default, but CUDA 11.8 may be needed for older GPUs
JAX_CUDA_VERSION = os.environ.get("JAX_CUDA_VERSION", "cuda12_pip")

# =============================================================================
# Modal Configuration
# =============================================================================

# Modal app definition
app = modal.App("gandalf-krmhd")

# Create a persistent volume for storing results
volume = modal.Volume.from_name(
    VOLUME_NAME,
    create_if_missing=True
)

# Define the container image with all dependencies
# Pin JAX to 0.4.x to avoid breaking changes in 0.5.x
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "jax[cpu]>=0.4.20,<0.5.0",  # Pin to 0.4.x series
        "h5py>=3.11.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0.0",
        "scipy>=1.10.0",
    )
    # Copy the GANDALF source code into the image
    .copy_local_dir("src", "/root/gandalf/src")
    .copy_local_dir("scripts", "/root/gandalf/scripts")  # Copy scripts directory
    .copy_local_file("pyproject.toml", "/root/gandalf/pyproject.toml")
    .workdir("/root/gandalf")
    .pip_install("-e .")  # Install GANDALF in editable mode
)

# GPU image (optional - use for large simulations)
# Note: JAX CUDA version is configurable via JAX_CUDA_VERSION environment variable
# Pin JAX to 0.4.x to avoid breaking changes in 0.5.x
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        f"jax[{JAX_CUDA_VERSION}]>=0.4.20,<0.5.0",  # Pin to 0.4.x series (version configurable)
        "h5py>=3.11.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0.0",
        "scipy>=1.10.0",
    )
    .copy_local_dir("src", "/root/gandalf/src")
    .copy_local_dir("scripts", "/root/gandalf/scripts")
    .copy_local_file("pyproject.toml", "/root/gandalf/pyproject.toml")
    .workdir("/root/gandalf")
    .pip_install("-e .")
)


# =============================================================================
# Simulation Runner (extracted from scripts/run_simulation.py)
# =============================================================================

def _run_gandalf_simulation(config, verbose: bool = True):
    """
    Internal function to run GANDALF simulation (Modal-compatible).

    This is a simplified version of scripts/run_simulation.py that avoids
    import path hacks and works within Modal's containerized environment.

    Args:
        config: SimulationConfig instance
        verbose: Print progress information

    Returns:
        tuple: (final_state, energy_history, grid)

    Raises:
        Exception: If simulation fails
    """
    import jax.random as jr
    import numpy as np
    from krmhd.config import SimulationConfig
    from krmhd import (
        gandalf_step,
        compute_cfl_timestep,
        energy as compute_energy,
        force_alfven_modes,
        KRMHDState,
        SpectralGrid3D,
    )
    from krmhd.diagnostics import (
        EnergyHistory,
        energy_spectrum_1d,
        energy_spectrum_perpendicular,
        energy_spectrum_parallel,
        plot_energy_history,
        plot_energy_spectrum,
    )
    from krmhd.io import save_checkpoint, save_timeseries

    if verbose:
        print(config.summary())

    # Initialize
    if verbose:
        print("\nInitializing...")

    grid = config.create_grid()
    state = config.create_initial_state(grid)

    if verbose:
        print(f"✓ Created {config.grid.Nx}×{config.grid.Ny}×{config.grid.Nz} grid")
        print(f"✓ Initialized {config.initial_condition.type} state")

    # Initialize forcing if enabled
    if config.forcing.enabled:
        key = jr.PRNGKey(config.forcing.seed or 42)
        if verbose:
            print(f"✓ Forcing enabled: k ∈ [{config.forcing.k_min}, {config.forcing.k_max}]")
    else:
        key = None

    # Initialize diagnostics
    energy_history = EnergyHistory()
    energy_history.append(state)
    E_init = compute_energy(state)

    if verbose:
        print(f"✓ Initial energy: {E_init['total']:.6e}")

    # Prepare output directory
    output_dir = config.get_output_dir()
    if verbose:
        print(f"✓ Output directory: {output_dir}")

    # Save initial configuration
    config.to_yaml(output_dir / "config.yaml")

    # Time Evolution
    if verbose:
        print("\n" + "-" * 70)
        print("Time evolution:")
        print("-" * 70)

    # Determine timestep
    if config.time_integration.dt_fixed is not None:
        dt = config.time_integration.dt_fixed
        if verbose:
            print(f"Using fixed timestep: dt = {dt:.6e}")
    else:
        dt = compute_cfl_timestep(
            state,
            v_A=config.physics.v_A,
            cfl_safety=config.time_integration.cfl_safety
        )
        if verbose:
            print(f"CFL timestep: dt = {dt:.6e}")

    # Main loop
    t = 0.0
    start_time = time.time()

    for step in range(config.time_integration.n_steps):
        # Apply forcing if enabled
        if config.forcing.enabled:
            # Convert physical wavenumbers to integer mode numbers (Issue #97)
            # k = 2πn/L_min, so n = k*L_min/(2π)
            import numpy as np
            L_min = min(grid.Lx, grid.Ly, grid.Lz)
            n_min = int(np.round(config.forcing.k_min * L_min / (2 * np.pi)))
            n_max = int(np.round(config.forcing.k_max * L_min / (2 * np.pi)))

            # Ensure at least one mode
            n_min = max(1, n_min)
            n_max = max(n_min, n_max)

            state, key = force_alfven_modes(
                state,
                amplitude=config.forcing.amplitude,
                n_min=n_min,
                n_max=n_max,
                dt=dt,
                key=key
            )

        # Time step
        state = gandalf_step(
            state,
            dt=dt,
            eta=config.physics.eta,
            v_A=config.physics.v_A,
            nu=config.physics.nu,
            hyper_r=config.physics.hyper_r,
            hyper_n=config.physics.hyper_n
        )
        t += dt

        # Save diagnostics
        if (step + 1) % config.time_integration.save_interval == 0:
            energy_history.append(state)
            E = compute_energy(state)

            if verbose:
                mag_frac = E['magnetic'] / E['total'] if E['total'] > 0 else 0
                print(
                    f"  Step {step+1:5d}/{config.time_integration.n_steps}: "
                    f"t = {t:.4f}, E = {E['total']:.6e}, "
                    f"E_mag/E_tot = {mag_frac:.3f}"
                )

        # Save checkpoint if enabled
        if (config.time_integration.checkpoint_interval is not None and
            (step + 1) % config.time_integration.checkpoint_interval == 0):
            checkpoint_file = output_dir / f"checkpoint_step{step+1:06d}.h5"
            save_checkpoint(
                state,
                str(checkpoint_file),
                metadata={
                    'step': step + 1,
                    'time': t,
                    'config_name': config.name,
                    'eta': config.physics.eta,
                    'nu': config.physics.nu,
                }
            )
            if verbose:
                print(f"    ✓ Saved checkpoint: {checkpoint_file.name}")

    elapsed = time.time() - start_time

    if verbose:
        print("-" * 70)
        print(f"✓ Evolution complete: {elapsed:.1f}s")
        print(f"  Performance: {config.time_integration.n_steps / elapsed:.1f} steps/s")

    # Final Diagnostics and Output
    if verbose:
        print("\n" + "-" * 70)
        print("Computing diagnostics...")

    # Final energy
    E_final = compute_energy(state)
    if verbose:
        print(f"✓ Final energy: {E_final['total']:.6e}")
        print(f"  Energy change: {(E_final['total'] - E_init['total']) / E_init['total'] * 100:.2f}%")

    # Compute spectra
    if config.io.save_spectra:
        k_bins, E_k = energy_spectrum_1d(state)
        k_perp_bins, E_kperp = energy_spectrum_perpendicular(state)
        k_par_bins, E_kpar = energy_spectrum_parallel(state)

        spectra = {
            'k_1d': k_bins,
            'E_1d': E_k,
            'k_perp': k_perp_bins,
            'E_perp': E_kperp,
            'k_par': k_par_bins,
            'E_par': E_kpar,
        }

        if verbose:
            print(f"✓ Computed energy spectra")
    else:
        spectra = {}

    # Save outputs
    if config.io.save_energy_history:
        save_timeseries(
            energy_history,
            str(output_dir / "energy_history.h5"),
            metadata={
                'config_name': config.name,
                'n_steps': config.time_integration.n_steps,
                'save_interval': config.time_integration.save_interval,
            },
            overwrite=config.io.overwrite
        )
        if verbose:
            print(f"✓ Saved energy history (HDF5)")

    if config.io.save_spectra:
        np.savez(
            output_dir / "spectra.npz",
            **spectra
        )
        if verbose:
            print(f"✓ Saved energy spectra")

    if config.io.save_final_state:
        save_checkpoint(
            state,
            str(output_dir / "final_state.h5"),
            metadata={
                'config_name': config.name,
                'description': 'Final simulation state',
                'n_steps': config.time_integration.n_steps,
                'total_time': state.time,
            },
            overwrite=config.io.overwrite
        )
        if verbose:
            print(f"✓ Saved final state (HDF5 checkpoint)")

    # Create plots
    if config.io.save_spectra or config.io.save_energy_history:
        if verbose:
            print("\nGenerating plots...")

        if config.io.save_energy_history:
            plot_energy_history(
                energy_history,
                filename=str(output_dir / "energy_history.png"),
                show=False
            )
            if verbose:
                print(f"✓ Saved energy_history.png")

        if config.io.save_spectra:
            plot_energy_spectrum(
                spectra['k_1d'],
                spectra['E_1d'],
                spectrum_type='1D',
                filename=str(output_dir / "spectrum_1d.png"),
                show=False
            )

            plot_energy_spectrum(
                spectra['k_perp'],
                spectra['E_perp'],
                spectrum_type='Perpendicular',
                filename=str(output_dir / "spectrum_perp.png"),
                show=False
            )

            if verbose:
                print(f"✓ Saved spectrum plots")

    if verbose:
        print("\n" + "=" * 70)
        print("Simulation complete!")
        print(f"Results saved to: {output_dir}")
        print("=" * 70)

    return state, energy_history, grid


# =============================================================================
# Core Simulation Function (Shared Implementation)
# =============================================================================

def _run_simulation_impl(
    config_yaml: str,
    output_subdir: Optional[str],
    verbose: bool,
    backend: str = 'cpu'
) -> dict[str, Any]:
    """
    Shared implementation for running GANDALF simulations on Modal.

    This function contains the common logic for both CPU and GPU execution,
    eliminating code duplication.

    Args:
        config_yaml: YAML configuration as string
        output_subdir: Optional subdirectory within /results for outputs
        verbose: Print progress information
        backend: 'cpu' or 'gpu' for logging and metadata

    Returns:
        Dictionary with simulation results and metadata on success,
        or error information on failure
    """
    import yaml
    import traceback
    import jax
    from krmhd.config import SimulationConfig
    from krmhd import energy as compute_energy

    try:
        # Verify device availability
        devices = jax.devices()
        if verbose:
            print(f"\nJAX devices: {devices}")
            print(f"Default backend: {jax.default_backend()}")

        # Validate GPU is actually available when requested
        if backend == 'gpu':
            gpu_available = any(d.platform == 'gpu' or 'gpu' in str(d).lower() for d in devices)
            if not gpu_available:
                raise RuntimeError(
                    f"GPU requested but not available. Detected devices: {devices}. "
                    "This may indicate a Modal GPU allocation issue or JAX configuration problem."
                )
            if verbose:
                print(f"✓ GPU device verified\n")
        else:
            if verbose:
                print()

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
            print(f"{'='*70}")
            backend_str = "GPU" if backend == 'gpu' else "CPU"
            print(f"Running on Modal Cloud ({backend_str})")
            print(f"Output directory: {output_path}")
            print(f"{'='*70}\n")

        # Run simulation
        start_time = time.time()
        state, energy_history, grid = _run_gandalf_simulation(config, verbose=verbose)
        elapsed = time.time() - start_time

        # Commit volume changes with error logging
        commit_success = True
        commit_error_msg = None
        try:
            volume.commit()
            if verbose:
                print(f"\n✓ Volume changes committed successfully")
        except Exception as commit_error:
            commit_success = False
            commit_error_msg = str(commit_error)
            error_msg = f"WARNING: Failed to commit volume changes: {commit_error}"
            print(f"\n{error_msg}")
            # Don't fail the entire job, but flag as partial failure
            # Results are computed correctly but not persisted to volume

        if verbose:
            print(f"\n{'='*70}")
            status_str = "Simulation complete!" if commit_success else "Simulation complete (volume commit failed)!"
            print(status_str)
            print(f"Total time: {elapsed:.1f}s")
            print(f"Results saved to: {output_path}")
            if not commit_success:
                print(f"WARNING: Results may not be persisted (commit failed)")
            print(f"{'='*70}\n")

        # Return metadata
        final_energy = compute_energy(state)

        result = {
            'status': 'success' if commit_success else 'partial_failure',
            'output_dir': output_path,
            'elapsed_time': elapsed,
            'config_name': config.name,
            'resolution': (config.grid.Nx, config.grid.Ny, config.grid.Nz),
            'n_steps': config.time_integration.n_steps,
            'final_time': float(state.time),
            'final_energy_total': float(final_energy['total']),
            'final_energy_magnetic': float(final_energy['magnetic']),
            'final_energy_kinetic': float(final_energy['kinetic']),
            'backend': backend,
        }

        # Add commit error details if commit failed
        if not commit_success:
            result['volume_commit_error'] = commit_error_msg

        return result

    except Exception as e:
        # Handle errors gracefully and return diagnostics
        error_msg = str(e)
        error_trace = traceback.format_exc()

        if verbose:
            print(f"\n{'='*70}")
            print(f"ERROR: Simulation failed!")
            print(f"{'='*70}")
            print(error_trace)

        # Try to commit any partial results with explicit error handling
        try:
            volume.commit()
            if verbose:
                print("\n⚠ Partial results committed to volume despite error")
        except Exception as commit_error:
            if verbose:
                print(f"\n⚠ Volume commit also failed: {commit_error}")
            # Don't mask the original error

        return {
            'status': 'failed',
            'error': error_msg,
            'traceback': error_trace,
            'output_dir': output_subdir or 'unknown',
            'backend': backend,
        }


@app.function(
    image=image,
    volumes={"/results": volume},
    timeout=TIMEOUT_SINGLE_SIMULATION,
    cpu=DEFAULT_CPU_CORES,
    memory=DEFAULT_CPU_MEMORY_MB,
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
        Dictionary with simulation results and metadata on success,
        or error information on failure
    """
    return _run_simulation_impl(config_yaml, output_subdir, verbose, backend='cpu')


@app.function(
    image=gpu_image,
    volumes={"/results": volume},
    timeout=TIMEOUT_SINGLE_SIMULATION,
    gpu=DEFAULT_GPU_TYPE,
    cpu=DEFAULT_GPU_CORES,
    memory=DEFAULT_GPU_MEMORY_MB,
)
def run_simulation_gpu(
    config_yaml: str,
    output_subdir: Optional[str] = None,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Run GANDALF simulation on Modal with GPU acceleration.

    This function uses the same implementation as run_simulation_remote
    but executes on GPU-enabled hardware.

    Args:
        config_yaml: YAML configuration as string
        output_subdir: Optional subdirectory within /results for outputs
        verbose: Print progress information

    Returns:
        Dictionary with simulation results and metadata on success,
        or error information on failure
    """
    return _run_simulation_impl(config_yaml, output_subdir, verbose, backend='gpu')


# =============================================================================
# Parameter Sweep Functions
# =============================================================================

@app.function(
    image=image,
    volumes={"/results": volume},
    timeout=TIMEOUT_PARAMETER_SWEEP,
)
def run_parameter_sweep(
    base_config_yaml: str,
    parameters: dict[str, list],
    sweep_name: Optional[str] = None,
    use_gpu: bool = False,
) -> dict[str, Any]:
    """
    Run a parameter sweep in parallel on Modal with fault tolerance.

    Args:
        base_config_yaml: Base YAML configuration as string
        parameters: Dictionary mapping parameter paths to lists of values
                   e.g., {'physics.eta': [0.01, 0.02, 0.05], 'grid.Nx': [64, 128]}
        sweep_name: Optional name for the sweep (used in output paths)
        use_gpu: Whether to use GPU instances for each run

    Returns:
        Dictionary with sweep metadata, successful results, and failed jobs
    """
    import yaml
    import itertools
    import json

    # Validate parameters dictionary
    if not parameters:
        raise ValueError("Parameters dictionary is empty. Must specify at least one parameter to sweep.")

    for param_name, param_list in parameters.items():
        if not isinstance(param_list, list):
            raise TypeError(f"Parameter '{param_name}' must be a list of values, got {type(param_list).__name__}")
        if not param_list:
            raise ValueError(f"Parameter '{param_name}' has empty value list")
        if '.' not in param_name:
            raise ValueError(
                f"Parameter '{param_name}' must use dot notation (e.g., 'physics.eta', 'grid.Nx'). "
                f"Simple keys are not supported."
            )

    # Generate all parameter combinations
    param_names = list(parameters.keys())
    param_values = list(parameters.values())
    combinations = list(itertools.product(*param_values))

    print(f"\nParameter sweep: {len(combinations)} combinations")
    print(f"Parameters: {param_names}")
    print(f"Using GPU: {use_gpu}")

    # Validate first combination to catch config errors early
    print("\nValidating first parameter combination...")
    try:
        from krmhd.config import SimulationConfig

        test_config_dict = yaml.safe_load(base_config_yaml)
        first_combo = combinations[0]

        for param_name, param_value in zip(param_names, first_combo):
            keys = param_name.split('.')
            d = test_config_dict
            for key in keys[:-1]:
                if key not in d:
                    raise KeyError(f"Parameter path '{param_name}' is invalid: '{key}' not found in config")
                d = d[key]
            if keys[-1] not in d:
                raise KeyError(f"Parameter path '{param_name}' is invalid: '{keys[-1]}' not found")
            d[keys[-1]] = param_value

        # Test Pydantic validation
        _ = SimulationConfig(**test_config_dict)
        print("✓ Parameter validation passed\n")

    except Exception as e:
        raise ValueError(f"Parameter sweep validation failed: {e}") from e

    # Create sweep directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sweep_dir = sweep_name or f"sweep_{timestamp}"

    # Submit all jobs in parallel
    futures = []
    job_metadata = []
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
        future = func.spawn(updated_yaml, output_subdir=output_subdir, verbose=False)
        futures.append(future)
        job_metadata.append({
            'job_id': i,
            'parameters': dict(zip(param_names, combo)),
            'output_subdir': output_subdir,
        })

    # Wait for all jobs to complete and collect results
    print(f"Waiting for {len(futures)} jobs to complete (timeout: {TIMEOUT_SWEEP_JOB_GET}s per job)...")

    successful_results = []
    partial_failures = []
    failed_jobs = []

    for i, (future, metadata) in enumerate(zip(futures, job_metadata)):
        try:
            # Add explicit timeout to prevent indefinite blocking
            result = future.get(timeout=TIMEOUT_SWEEP_JOB_GET)

            if result['status'] == 'success':
                successful_results.append({
                    **result,
                    'job_metadata': metadata,
                })
                print(f"✓ Job {i+1}/{len(futures)} completed successfully")
            elif result['status'] == 'partial_failure':
                # Job completed but volume commit failed
                partial_failures.append({
                    **result,
                    'job_metadata': metadata,
                })
                print(f"⚠ Job {i+1}/{len(futures)} partial failure: {result.get('volume_commit_error', 'volume commit failed')}")
            else:
                # Failed status
                failed_jobs.append({
                    **result,
                    'job_metadata': metadata,
                })
                print(f"✗ Job {i+1}/{len(futures)} failed: {result.get('error', 'Unknown error')}")

        except TimeoutError:
            # Job exceeded timeout
            failed_jobs.append({
                'status': 'failed',
                'error': f'Job timed out after {TIMEOUT_SWEEP_JOB_GET}s',
                'job_metadata': metadata,
            })
            print(f"✗ Job {i+1}/{len(futures)} timed out after {TIMEOUT_SWEEP_JOB_GET}s")

        except Exception as e:
            # Handle catastrophic job failures (e.g., OOM, network errors)
            failed_jobs.append({
                'status': 'failed',
                'error': str(e),
                'job_metadata': metadata,
            })
            print(f"✗ Job {i+1}/{len(futures)} crashed: {str(e)}")

    # Save sweep metadata
    sweep_metadata = {
        'sweep_name': sweep_dir,
        'timestamp': timestamp,
        'n_runs_total': len(combinations),
        'n_runs_successful': len(successful_results),
        'n_runs_partial_failure': len(partial_failures),
        'n_runs_failed': len(failed_jobs),
        'parameters': {k: list(v) for k, v in parameters.items()},
        'use_gpu': use_gpu,
        'successful_results': successful_results,
        'partial_failures': partial_failures,
        'failed_jobs': failed_jobs,
    }

    metadata_path = Path(f"/results/{sweep_dir}/sweep_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(sweep_metadata, f, indent=2)

    volume.commit()

    print(f"\nSweep complete!")
    print(f"  Successful: {len(successful_results)}/{len(combinations)}")
    if partial_failures:
        print(f"  Partial failures: {len(partial_failures)}/{len(combinations)} (computed but not persisted)")
    print(f"  Failed: {len(failed_jobs)}/{len(combinations)}")
    print(f"  Results saved to /results/{sweep_dir}")

    return sweep_metadata


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
    from pathlib import Path

    source = Path(f"/results/{result_dir}")
    if not source.exists():
        raise FileNotFoundError(f"Result directory not found: {source}")

    # This will be executed on Modal, so we need to return the data
    # In practice, you'd use modal.CloudBucketMount or download via CLI
    print(f"To download results, use:")
    print(f"  modal volume get {VOLUME_NAME} {result_dir} {local_path}")


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

    if result['status'] == 'success':
        print("Simulation complete!")
        print("="*70)
        print(f"Status: {result['status']}")
        print(f"Output directory: {result['output_dir']}")
        print(f"Elapsed time: {result['elapsed_time']:.1f}s")
        print(f"Final time: {result['final_time']:.4f}")
        print(f"Final energy: {result['final_energy_total']:.6e}")
        print("\nTo download results:")
        print(f"  modal volume get {VOLUME_NAME} {result['output_dir'].replace('/results/', '')} ./local_results")
    else:
        print("Simulation FAILED!")
        print("="*70)
        print(f"Error: {result['error']}")
        print("\nTraceback:")
        print(result['traceback'])

    print("="*70)


if __name__ == "__main__":
    # This allows testing locally without Modal
    import sys
    print("This script is designed to run on Modal.")
    print("Usage:")
    print("  modal deploy modal_app.py")
    print("  modal run modal_app.py --config-path configs/driven_turbulence.yaml")
    sys.exit(0)
