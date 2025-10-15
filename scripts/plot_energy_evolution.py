#!/usr/bin/env python3
"""
Plot energy evolution for Orszag-Tang vortex simulation.

This script loads energy history data from a simulation run and creates
publication-quality plots showing the time evolution of total, kinetic,
and magnetic energies.

Usage:
    uv run python scripts/plot_energy_evolution.py [--run-simulation]

Options:
    --run-simulation : Run the Orszag-Tang simulation first before plotting
    --output FILE    : Output filename (default: orszag_tang_energy.png)
    --normalize-time : Normalize time by Alfvén time τ_A = L/v_A

Note:
    This script must be run with 'uv run' to ensure the krmhd package is
    available in the Python path.
"""

import sys
from pathlib import Path
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt


def run_orszag_tang_simulation(save_file: str = "orszag_tang_history.pkl"):
    """Run Orszag-Tang simulation and save energy history."""
    print("Running Orszag-Tang simulation...")

    from krmhd import (
        SpectralGrid3D,
        initialize_orszag_tang,
        gandalf_step,
        compute_cfl_timestep,
    )
    from krmhd.diagnostics import EnergyHistory

    # Grid and parameters (matching examples/orszag_tang.py)
    Nx, Ny, Nz = 64, 64, 2
    Lx = Ly = 2 * np.pi
    Lz = 2 * np.pi
    B0 = 1.0 / np.sqrt(4 * np.pi)
    v_A = 1.0
    eta = 0.001
    cfl_safety = 0.3
    t_final = 1.0
    save_interval = 0.1

    # Initialize grid and state using shared function
    grid = SpectralGrid3D.create(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz)
    state = initialize_orszag_tang(
        grid=grid,
        M=10,
        B0=B0,
        v_th=1.0,
        beta_i=1.0,
        nu=0.01,
        Lambda=1.0,
    )

    # Time evolution
    history = EnergyHistory()
    next_save_time = 0.0

    print(f"  Evolving to t={t_final}...")
    while state.time < t_final:
        if state.time >= next_save_time:
            history.append(state)
            print(f"    t = {state.time:.3f}")
            next_save_time += save_interval

        dt = compute_cfl_timestep(state, v_A, cfl_safety)
        dt = min(dt, t_final - state.time, next_save_time - state.time)
        state = gandalf_step(state, dt, eta, v_A)

    history.append(state)
    print(f"  ✓ Completed: t = {state.time:.3f}")

    # Save history to file
    history_data = {
        'times': np.array(history.times),
        'E_total': np.array(history.E_total),
        'E_magnetic': np.array(history.E_magnetic),
        'E_kinetic': np.array(history.E_kinetic),
        'E_compressive': np.array(history.E_compressive),
        'v_A': v_A,
        'Lx': Lx,
    }

    with open(save_file, 'wb') as f:
        pickle.dump(history_data, f)

    print(f"  ✓ Saved energy history to {save_file}")
    return history_data


def plot_energy_evolution(
    history_file: str = "orszag_tang_history.pkl",
    output_file: str = "orszag_tang_energy.png",
    normalize_time: bool = True,
):
    """
    Create energy evolution plot from saved history data.

    Args:
        history_file: Path to pickled energy history data
        output_file: Output filename for plot
        normalize_time: If True, normalize time by Alfvén time τ_A = L/v_A
    """
    print(f"\nLoading energy history from {history_file}...")

    # Load data
    with open(history_file, 'rb') as f:
        data = pickle.load(f)

    times = data['times']
    E_total = data['E_total']
    E_magnetic = data['E_magnetic']
    E_kinetic = data['E_kinetic']

    # Normalize time if requested
    if normalize_time and 'v_A' in data and 'Lx' in data:
        tau_A = data['Lx'] / data['v_A']  # Alfvén crossing time
        times = times / tau_A
        time_label = r'$t/\tau_A$'
    else:
        time_label = 't'

    # Create figure matching thesis style
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot energy components
    ax.plot(times, E_total, 'k-', linewidth=2, label='Total Energy')
    ax.plot(times, E_kinetic, 'r--', linewidth=2, label='Kinetic Energy')
    ax.plot(times, E_magnetic, 'g:', linewidth=2, label='Magnetic Energy')

    # Formatting
    ax.set_xlabel(time_label, fontsize=14)
    ax.set_ylabel('Energy', fontsize=14)
    ax.set_xlim(times[0], times[-1])
    ax.set_ylim(0, max(E_total.max(), E_magnetic.max(), E_kinetic.max()) * 1.1)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)

    # Add info text
    E_ratio_initial = E_magnetic[0] / E_kinetic[0]
    E_ratio_final = E_magnetic[-1] / E_kinetic[-1]
    textstr = f'Initial $E_{{mag}}/E_{{kin}}$ = {E_ratio_initial:.3f}\n'
    textstr += f'Final $E_{{mag}}/E_{{kin}}$ = {E_ratio_final:.3f}\n'
    textstr += f'Energy conserved: {100*E_total[-1]/E_total[0]:.2f}%'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_file}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot energy evolution for Orszag-Tang vortex'
    )
    parser.add_argument(
        '--run-simulation',
        action='store_true',
        help='Run simulation before plotting'
    )
    parser.add_argument(
        '--output',
        default='orszag_tang_energy.png',
        help='Output filename (default: orszag_tang_energy.png)'
    )
    parser.add_argument(
        '--no-normalize-time',
        action='store_true',
        help='Do not normalize time by Alfvén time'
    )
    parser.add_argument(
        '--history-file',
        default='orszag_tang_history.pkl',
        help='Energy history file (default: orszag_tang_history.pkl)'
    )

    args = parser.parse_args()

    # Run simulation if requested
    if args.run_simulation:
        run_orszag_tang_simulation(args.history_file)

    # Check if history file exists
    if not Path(args.history_file).exists():
        print(f"Error: History file '{args.history_file}' not found!")
        print("Run with --run-simulation to generate it first.")
        sys.exit(1)

    # Create plot
    plot_energy_evolution(
        history_file=args.history_file,
        output_file=args.output,
        normalize_time=not args.no_normalize_time,
    )

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
