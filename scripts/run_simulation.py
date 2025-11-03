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

Example:
    # Generate and customize config
    python scripts/run_simulation.py --template driven > my_config.yaml
    # Edit my_config.yaml as needed
    python scripts/run_simulation.py my_config.yaml
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Tuple

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


def run_simulation(
    config: SimulationConfig, verbose: bool = True
) -> Tuple[KRMHDState, EnergyHistory, SpectralGrid3D]:
    """
    Run KRMHD simulation from configuration.

    Args:
        config: SimulationConfig instance
        verbose: Print progress information

    Returns:
        tuple: (final_state, energy_history, grid)
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
            cfl_safety=config.time_integration.cfl_safety
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
            cfl_safety=config.time_integration.cfl_safety
        )
        if verbose:
            print(f"CFL timestep: dt = {dt:.6e}")

    # Main loop
    t = 0.0
    start_time = time.time()

    # TODO: Implement checkpoint saving when checkpoint_interval is set
    # GitHub Issue: https://github.com/anjor/gandalf/issues/13 (HDF5 I/O)
    # if config.time_integration.checkpoint_interval is not None:
    #     # Save state every checkpoint_interval steps to HDF5

    for step in range(config.time_integration.n_steps):
        # Apply forcing if enabled
        if config.forcing.enabled:
            state, key = force_alfven_modes(
                state,
                amplitude=config.forcing.amplitude,
                k_min=config.forcing.k_min,
                k_max=config.forcing.k_max,
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

    elapsed = time.time() - start_time

    if verbose:
        print("-" * 70)
        print(f"✓ Evolution complete: {elapsed:.1f}s")
        print(f"  Performance: {config.time_integration.n_steps / elapsed:.1f} steps/s")

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

    # Save outputs
    if config.io.save_energy_history:
        energy_dict = energy_history.to_dict()
        np.savez(
            output_dir / "energy_history.npz",
            **energy_dict
        )
        if verbose:
            print(f"✓ Saved energy history")

    if config.io.save_spectra:
        np.savez(
            output_dir / "spectra.npz",
            **spectra
        )
        if verbose:
            print(f"✓ Saved energy spectra")

    if config.io.save_final_state:
        # Save final state as NPZ with all fields and metadata for exact restart
        # NOTE: Using NPZ for now; will migrate to HDF5 for better compression,
        # metadata support, and partial reads once Issue #13 is implemented
        np.savez(
            output_dir / "final_state.npz",
            # Primary state fields (Fourier space)
            z_plus=np.array(state.z_plus),
            z_minus=np.array(state.z_minus),
            B_parallel=np.array(state.B_parallel),
            g=np.array(state.g),
            # Metadata for state reconstruction
            M=state.M,
            beta_i=state.beta_i,
            v_th=state.v_th,
            nu=state.nu,
            Lambda=state.Lambda,
            time=state.time,
            # Grid parameters for validation
            Nx=state.grid.Nx,
            Ny=state.grid.Ny,
            Nz=state.grid.Nz,
            Lx=state.grid.Lx,
            Ly=state.grid.Ly,
            Lz=state.grid.Lz,
        )
        if verbose:
            print(f"✓ Saved final state (z±, B∥, g, metadata)")

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
        # Use model_copy to avoid mutating Pydantic model (bypasses validation)
        config = config.model_copy(
            update={'io': config.io.model_copy(update={'output_dir': args.output_dir})}
        )

    # Run simulation
    try:
        run_simulation(config, verbose=not args.quiet)
    except Exception as e:
        print(f"Error running simulation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
