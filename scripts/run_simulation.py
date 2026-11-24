#!/usr/bin/env python3
"""
KRMHD Simulation Runner

This script runs KRMHD simulations from YAML configuration files.
It provides a unified interface for all simulation types (decaying turbulence,
driven turbulence, Orszag-Tang vortex, etc.).

Usage:
    # Run with config file
    python scripts/run_simulation.py configs/decaying_turbulence.yaml

    # Generate template config
    python scripts/run_simulation.py --template decaying > config.yaml

    # Run with custom output directory
    python scripts/run_simulation.py config.yaml --output-dir my_results

    # Stop after 23 h of wall time (exit status 3), then continue from the latest
    # checkpoint with the same command
    python scripts/run_simulation.py config.yaml --resume --max-wall-hours 23

Example:
    # Generate and customize config
    python scripts/run_simulation.py --template driven > my_config.yaml
    # Edit my_config.yaml as needed
    python scripts/run_simulation.py my_config.yaml
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import jax.numpy as jnp
import jax.random as jr
import numpy as np

# CFL validation thresholds for fixed timestep
# 2.0× CFL limit may cause numerical instability in explicit timestepping
CFL_WARNING_THRESHOLD = 2.0
# 10.0× CFL limit will almost certainly cause catastrophic failure
CFL_ERROR_THRESHOLD = 10.0

from krmhd.config import (
    SimulationConfig,
    decaying_turbulence_config,
    driven_turbulence_config,
    orszag_tang_config,
)
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
from krmhd.io import load_checkpoint, load_timeseries, save_checkpoint, save_timeseries

# Exit status after --max-wall-hours: the state is checkpointed, continue with --resume
PAUSED_EXIT = 3
_CHECKPOINT_NAME = re.compile(r"checkpoint_step(\d+)\.h5")


class SimulationPaused(Exception):
    """run_simulation stopped at max_wall_hours; its state is in the latest checkpoint."""

    def __init__(self, t: float, step: int):
        super().__init__(f"paused at t = {t:.6f}, step {step}")
        self.t = t
        self.step = step


def latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """The checkpoint_step<N>.h5 in `output_dir` with the largest N, if any."""
    found = [(int(m.group(1)), p) for p in output_dir.glob("checkpoint_step*.h5")
             if (m := _CHECKPOINT_NAME.fullmatch(p.name))]
    return max(found)[1] if found else None


def _save_resume_point(state: KRMHDState, history: EnergyHistory, config: SimulationConfig,
                       output_dir: Path, step: int, t: float, dt: float, key: Any) -> Path:
    """Write the checkpoint of `step`, and the energy history so far, for --resume.

    Each file is written under a temporary name and renamed, so an interrupted write
    never leaves a truncated checkpoint as the latest one.
    """
    path = output_dir / f"checkpoint_step{step:06d}.h5"
    metadata = {'step': step, 'time': t, 'dt': dt, 'config_name': config.name,
                'eta': config.physics.eta, 'nu': config.physics.nu}
    if key is not None:
        metadata['forcing_key'] = np.asarray(key)
    tmp = output_dir / f".{path.name}.tmp"
    save_checkpoint(state, str(tmp), metadata=metadata, overwrite=True)
    os.replace(tmp, path)
    if config.io.save_energy_history:
        tmp = output_dir / ".energy_history.h5.tmp"
        save_timeseries(history, str(tmp), metadata={'config_name': config.name}, overwrite=True)
        os.replace(tmp, output_dir / "energy_history.h5")
    return path


def run_simulation(
    config: SimulationConfig,
    verbose: bool = True,
    resume: bool = False,
    max_wall_hours: Optional[float] = None,
) -> tuple[KRMHDState, EnergyHistory, SpectralGrid3D]:
    """
    Run KRMHD simulation from configuration.

    Args:
        config: SimulationConfig instance
        verbose: Print progress information
        resume: Continue from the latest checkpoint_step<N>.h5 in the output directory
            (state, step, time, dt, forcing key and energy history); start a new run if
            there is none. The output directory may then be non-empty.
        max_wall_hours: Once this call has run this long, checkpoint the state and raise
            SimulationPaused, so that the run can be continued with resume=True.

    Returns:
        tuple: (final_state, energy_history, grid)

    Raises:
        SimulationPaused: max_wall_hours ran out before the last step
    """
    if verbose:
        print(config.summary())

    # ==========================================================================
    # Initialize
    # ==========================================================================

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
            # Show both physical wavenumbers and mode numbers for clarity
            L_min = min(grid.Lx, grid.Ly, grid.Lz)
            n_min = int(np.round(config.forcing.k_min * L_min / (2 * np.pi)))
            n_max = int(np.round(config.forcing.k_max * L_min / (2 * np.pi)))
            n_min = max(1, n_min)
            n_max = max(n_min, n_max)
            print(f"✓ Forcing enabled: k ∈ [{config.forcing.k_min:.2f}, {config.forcing.k_max:.2f}], modes n ∈ [{n_min}, {n_max}]")
    else:
        key = None

    # Initialize diagnostics
    energy_history = EnergyHistory()
    energy_history.append(state)
    E_init = compute_energy(state)

    if verbose:
        print(f"✓ Initial energy: {E_init['total']:.6e}")

    # Prepare output directory
    if resume:
        output_dir = Path(config.io.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = config.get_output_dir()
    if verbose:
        print(f"✓ Output directory: {output_dir}")

    # Save initial configuration
    config.to_yaml(output_dir / "config.yaml")

    # Continue from the latest checkpoint
    step0, t, dt_resumed = 0, 0.0, None
    checkpoint = latest_checkpoint(output_dir) if resume else None
    if checkpoint is not None:
        # The arrays as saved, so that the resumed run continues the same trajectory
        loaded, _, meta = load_checkpoint(str(checkpoint), enforce_hermitian=False)
        if (loaded.z_plus.shape, loaded.g.shape) != (state.z_plus.shape, state.g.shape):
            raise ValueError(f"{checkpoint} does not match the config: z± {loaded.z_plus.shape}, "
                             f"g {loaded.g.shape} vs {state.z_plus.shape}, {state.g.shape}")
        state = loaded
        step0, t, dt_resumed = int(meta['step']), float(meta['time']), meta.get('dt')
        if key is not None and 'forcing_key' in meta:
            key = jnp.asarray(meta['forcing_key'], dtype=jnp.uint32)
        history_file = output_dir / "energy_history.h5"
        if history_file.exists():
            saved, _ = load_timeseries(str(history_file))
            # history times are the states' own times, like that of the checkpoint
            n = sum(1 for ti in saved.times if ti <= float(state.time))
            energy_history = EnergyHistory(times=saved.times[:n], E_magnetic=saved.E_magnetic[:n],
                                           E_kinetic=saved.E_kinetic[:n], E_total=saved.E_total[:n])
        else:
            energy_history = EnergyHistory()
            energy_history.append(state)
        if verbose:
            print(f"✓ Resuming from {checkpoint.name}: step {step0}, t = {t:.4f}")

    # ==========================================================================
    # Time Evolution
    # ==========================================================================

    if verbose:
        print("\n" + "-" * 70)
        print("Time evolution:")
        print("-" * 70)

    # Determine timestep
    if config.time_integration.dt_fixed is not None:
        dt = config.time_integration.dt_fixed

        # Validate fixed timestep against CFL
        dt_cfl = compute_cfl_timestep(
            state,
            v_A=config.physics.v_A,
            cfl_safety=config.time_integration.cfl_safety,
            closure=config.physics.closure,
        )

        # Error on severe CFL violations (>10× CFL limit)
        if dt > CFL_ERROR_THRESHOLD * dt_cfl:
            raise ValueError(
                f"Fixed timestep dt={dt:.6e} is {dt/dt_cfl:.1f}x larger than CFL "
                f"timestep ({dt_cfl:.6e}). This will cause catastrophic numerical failure. "
                f"Reduce dt to < {CFL_ERROR_THRESHOLD * dt_cfl:.6e} or use CFL-based timestep (remove dt_fixed)."
            )

        # Warn on moderate CFL violations (2-10× CFL limit)
        if dt > CFL_WARNING_THRESHOLD * dt_cfl:
            import warnings
            warnings.warn(
                f"Fixed timestep dt={dt:.6e} is {dt/dt_cfl:.1f}x larger than CFL "
                f"timestep ({dt_cfl:.6e}). This may cause numerical instability. "
                f"Consider reducing dt or using CFL-based timestep.",
                UserWarning
            )

        if verbose:
            print(f"Using fixed timestep: dt = {dt:.6e} (CFL suggests {dt_cfl:.6e})")
    else:
        dt = compute_cfl_timestep(
            state,
            v_A=config.physics.v_A,
            cfl_safety=config.time_integration.cfl_safety,
            closure=config.physics.closure,
        )
        if verbose:
            print(f"CFL timestep: dt = {dt:.6e}")
    if dt_resumed is not None:
        dt = float(dt_resumed)  # the run's own timestep, not the CFL step of the resumed state

    # Main loop
    start_time = time.time()
    n_steps = config.time_integration.n_steps

    for step in range(step0, n_steps):
        # Apply forcing if enabled
        if config.forcing.enabled:
            # Convert physical wavenumbers to integer mode numbers (Issue #97)
            # k = 2πn/L_min, so n = k*L_min/(2π)
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
        # Note: config.physics.nu provides runtime control of collision frequency
        # gandalf_step uses config.physics.nu if provided, otherwise defaults to state.nu
        # This avoids needing to recreate state when changing collision parameters
        state = gandalf_step(
            state,
            dt=dt,
            eta=config.physics.eta,
            v_A=config.physics.v_A,
            nu=config.physics.nu,
            hyper_r=config.physics.hyper_r,
            hyper_n=config.physics.hyper_n,
            eta_z=config.physics.eta_z,
            hyper_rz=config.physics.hyper_rz,
            closure=config.physics.closure,
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
        saved = (config.time_integration.checkpoint_interval is not None and
                 (step + 1) % config.time_integration.checkpoint_interval == 0)
        if saved:
            checkpoint_file = _save_resume_point(state, energy_history, config, output_dir,
                                                 step + 1, t, dt, key)
            if verbose:
                print(f"    ✓ Saved checkpoint: {checkpoint_file.name}")

        # Pause at the wall-time budget; the same command with --resume continues
        if (max_wall_hours is not None and step + 1 < n_steps
                and time.time() - start_time >= 3600 * max_wall_hours):
            if not saved:
                _save_resume_point(state, energy_history, config, output_dir, step + 1, t, dt, key)
            print(f"paused at t = {t:.6f}, step {step + 1}, after {max_wall_hours:g} h of wall time; "
                  "continue with --resume", flush=True)
            raise SimulationPaused(t, step + 1)

    elapsed = time.time() - start_time

    if verbose:
        print("-" * 70)
        print(f"✓ Evolution complete: {elapsed:.1f}s")
        print(f"  Performance: {(n_steps - step0) / elapsed:.1f} steps/s")

    # ==========================================================================
    # Final Diagnostics and Output
    # ==========================================================================

    if verbose:
        print("\n" + "-" * 70)
        print("Computing diagnostics...")

    # Final energy
    E_final = compute_energy(state)
    if verbose:
        print(f"✓ Final energy: {E_final['total']:.6e}")
        print(f"  Energy change: {(E_final['total'] - E_init['total']) / E_init['total'] * 100:.2f}%")

    # Compute spectra
    spectra: dict[str, Any] = {}
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

    # Save outputs. overwrite=True: the folder was checked at the start (get_output_dir),
    # or belongs to this run (resume), so existing files are this run's own checkpoints.
    if config.io.save_energy_history:
        # Use HDF5 format for energy history (Issue #13)
        save_timeseries(
            energy_history,
            str(output_dir / "energy_history.h5"),
            metadata={
                'config_name': config.name,
                'n_steps': config.time_integration.n_steps,
                'save_interval': config.time_integration.save_interval,
            },
            overwrite=True
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
        # Use HDF5 checkpoint format for final state (Issue #13)
        save_checkpoint(
            state,
            str(output_dir / "final_state.h5"),
            metadata={
                'config_name': config.name,
                'description': 'Final simulation state',
                'n_steps': config.time_integration.n_steps,
                'total_time': state.time,
            },
            overwrite=True
        )
        if verbose:
            print(f"✓ Saved final state (HDF5 checkpoint)")

    # Create plots
    if config.io.save_spectra or config.io.save_energy_history:
        if verbose:
            print("\nGenerating plots...")

        # Energy history plot
        if config.io.save_energy_history:
            plot_energy_history(
                energy_history,
                filename=str(output_dir / "energy_history.png"),
                show=False
            )
            if verbose:
                print(f"✓ Saved energy_history.png")

        # Spectrum plots
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


def generate_template(template_type: str) -> None:
    """Generate template configuration file."""
    templates = {
        'decaying': decaying_turbulence_config,
        'driven': driven_turbulence_config,
        'orszag': orszag_tang_config,
    }

    if template_type not in templates:
        print(f"Error: Unknown template type '{template_type}'", file=sys.stderr)
        print(f"Available templates: {', '.join(templates.keys())}", file=sys.stderr)
        sys.exit(1)

    config = templates[template_type]()
    print(f"# KRMHD Configuration: {config.name}")
    print(f"# {config.description}")
    print("#")
    print("# Edit this file and run:")
    print(f"#   python scripts/run_simulation.py config.yaml")
    print()

    # Output as YAML to stdout
    import yaml
    data = config.model_dump(mode='python', exclude_none=True)
    print(yaml.safe_dump(data, default_flow_style=False, sort_keys=False, indent=2))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run KRMHD simulations from configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate template config
  python scripts/run_simulation.py --template decaying > decaying.yaml

  # Run simulation
  python scripts/run_simulation.py decaying.yaml

  # Run with custom output
  python scripts/run_simulation.py config.yaml --output-dir results_v2

Available templates:
  decaying - Decaying turbulence with k^(-5/3) spectrum
  driven   - Driven turbulence with forcing
  orszag   - Orszag-Tang vortex benchmark
        """
    )

    parser.add_argument(
        'config',
        nargs='?',
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--template',
        choices=['decaying', 'driven', 'orszag'],
        help='Generate template configuration (outputs to stdout)'
    )

    parser.add_argument(
        '--output-dir',
        help='Override output directory from config'
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Continue from the latest checkpoint in the output directory (a new run if there is none)'
    )

    parser.add_argument(
        '--max-wall-hours',
        type=float,
        help=f'Checkpoint and exit with status {PAUSED_EXIT} after this many hours; continue with --resume'
    )

    args = parser.parse_args()

    # Template generation mode
    if args.template:
        generate_template(args.template)
        return

    # Simulation mode
    if not args.config:
        parser.print_help()
        print("\nError: Must specify config file or --template", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Load configuration
    try:
        config = SimulationConfig.from_yaml(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        sys.exit(1)

    # Override output directory if specified
    if args.output_dir:
        # Use model_copy to create updated config while preserving immutability and validation
        config = config.model_copy(
            update={'io': config.io.model_copy(update={'output_dir': args.output_dir})}
        )

    # Run simulation
    try:
        run_simulation(config, verbose=not args.quiet, resume=args.resume,
                       max_wall_hours=args.max_wall_hours)
    except SimulationPaused:
        sys.exit(PAUSED_EXIT)
    except Exception as e:
        print(f"Error running simulation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
