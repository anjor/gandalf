#!/usr/bin/env python3
"""
Analyze Turbulence Diagnostics for Issue #82 Investigation

This script loads and visualizes turbulence diagnostics saved by
alfvenic_cascade_benchmark.py with --save-diagnostics flag.

Usage:
    # Analyze single resolution
    python analyze_issue82_diagnostics.py --resolution 64

    # Compare multiple resolutions
    python analyze_issue82_diagnostics.py --compare 32 64 128

    # Load specific diagnostic files
    python analyze_issue82_diagnostics.py --files diag1.h5 diag2.h5 diag3.h5

The script generates time series plots of key diagnostics:
- max_velocity(t): Detects field amplitude blow-up
- CFL number(t): Identifies timestep stability violations
- max_nonlinear(t): Tracks cascade rate extremes
- energy_highk(t): Monitors spectral pile-up near dealiasing boundary
- critical_balance_ratio(t): Validates RMHD cascade assumptions

Physics Interpretation:
- **CFL > 1.0**: Timestep too large for explicit time integration (instability)
- **max_velocity growth**: Exponential indicates numerical instability
- **High-k energy accumulation**: Insufficient dissipation or dealiasing failure
- **Critical balance ≠ 1**: RMHD ordering violated (τ_nl should ~ τ_A)

Goal (Issue #82):
Identify WHEN and WHERE the 64³ instability develops to determine root cause.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from krmhd.io import load_turbulence_diagnostics


def plot_diagnostics_comparison(
    diagnostics_dict,
    labels,
    output_dir='examples/output',
    save_prefix='issue82_diagnostics'
):
    """
    Plot comprehensive comparison of turbulence diagnostics.

    Args:
        diagnostics_dict: Dictionary {label: diagnostics_list}
        labels: List of labels for plot legend
        output_dir: Directory to save plots
        save_prefix: Prefix for output filenames
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract time series for each diagnostic
    data = {}
    for label in labels:
        diag_list = diagnostics_dict[label]
        data[label] = {
            'times': [d.time for d in diag_list],
            'max_velocity': [d.max_velocity for d in diag_list],
            'cfl_number': [d.cfl_number for d in diag_list],
            'max_nonlinear': [d.max_nonlinear for d in diag_list],
            'energy_highk': [d.energy_highk for d in diag_list],
            'critical_balance_ratio': [d.critical_balance_ratio for d in diag_list],
            'energy_total': [d.energy_total for d in diag_list],
        }

    # Create comprehensive diagnostic plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Turbulence Diagnostics Comparison (Issue #82)', fontsize=14, fontweight='bold')

    # Define colors for different cases
    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # 1. Max velocity
    ax = axes[0, 0]
    for i, label in enumerate(labels):
        ax.plot(data[label]['times'], data[label]['max_velocity'],
                label=label, color=colors[i % len(colors)], linewidth=1.5)
    ax.set_xlabel('Time (τ_A)')
    ax.set_ylabel('Max velocity |v⊥|')
    ax.set_title('Maximum Perpendicular Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 2. CFL number
    ax = axes[0, 1]
    for i, label in enumerate(labels):
        ax.plot(data[label]['times'], data[label]['cfl_number'],
                label=label, color=colors[i % len(colors)], linewidth=1.5)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='CFL = 1 (critical)')
    ax.axhline(0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='CFL = 0.5 (safe)')
    ax.set_xlabel('Time (τ_A)')
    ax.set_ylabel('CFL number')
    ax.set_title('Courant-Friedrichs-Lewy Number')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Max nonlinear term
    ax = axes[1, 0]
    for i, label in enumerate(labels):
        ax.plot(data[label]['times'], data[label]['max_nonlinear'],
                label=label, color=colors[i % len(colors)], linewidth=1.5)
    ax.set_xlabel('Time (τ_A)')
    ax.set_ylabel('Max |{z∓, ∇²z±}|')
    ax.set_title('Maximum Nonlinear Term (Cascade Rate)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 4. High-k energy fraction
    ax = axes[1, 1]
    for i, label in enumerate(labels):
        ax.plot(data[label]['times'], data[label]['energy_highk'],
                label=label, color=colors[i % len(colors)], linewidth=1.5)
    ax.set_xlabel('Time (τ_A)')
    ax.set_ylabel('E(k > 0.9k_max) / E_total')
    ax.set_title('High-k Energy Fraction (Pile-up Detector)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 5. Critical balance ratio
    ax = axes[2, 0]
    for i, label in enumerate(labels):
        ax.plot(data[label]['times'], data[label]['critical_balance_ratio'],
                label=label, color=colors[i % len(colors)], linewidth=1.5)
    ax.axhline(1.0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='τ_nl/τ_A = 1 (ideal)')
    ax.set_xlabel('Time (τ_A)')
    ax.set_ylabel('⟨τ_nl/τ_A⟩ (k ~ 5-10)')
    ax.set_title('Critical Balance Ratio (Goldreich-Sridhar)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # 6. Total energy
    ax = axes[2, 1]
    for i, label in enumerate(labels):
        ax.plot(data[label]['times'], data[label]['energy_total'],
                label=label, color=colors[i % len(colors)], linewidth=1.5)
    ax.set_xlabel('Time (τ_A)')
    ax.set_ylabel('E_total')
    ax.set_title('Total Energy (Normalization Check)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    # Save figure
    filename = f"{save_prefix}_comparison.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {filepath}")


def print_summary_statistics(diagnostics_list, label, metadata):
    """Print summary statistics for a single diagnostic run."""
    print(f"\n{'='*70}")
    print(f"Summary Statistics: {label}")
    print(f"{'='*70}")

    # Print metadata
    print(f"\nMetadata:")
    for key, value in metadata.items():
        if key not in ['created_at', 'format_version', 'description']:
            print(f"  {key}: {value}")

    # Extract arrays
    times = np.array([d.time for d in diagnostics_list])
    max_velocities = np.array([d.max_velocity for d in diagnostics_list])
    cfl_numbers = np.array([d.cfl_number for d in diagnostics_list])
    max_nonlinear = np.array([d.max_nonlinear for d in diagnostics_list])
    energy_highk = np.array([d.energy_highk for d in diagnostics_list])
    critical_balance = np.array([d.critical_balance_ratio for d in diagnostics_list])

    print(f"\nTime range: {times[0]:.2f} to {times[-1]:.2f} τ_A ({len(times)} samples)")

    print(f"\nDiagnostics:")
    print(f"  max_velocity:")
    print(f"    min={np.min(max_velocities):.3f}, max={np.max(max_velocities):.3f}, mean={np.mean(max_velocities):.3f}")
    print(f"  CFL number:")
    print(f"    min={np.min(cfl_numbers):.3f}, max={np.max(cfl_numbers):.3f}, mean={np.mean(cfl_numbers):.3f}")
    print(f"  max_nonlinear:")
    print(f"    min={np.min(max_nonlinear):.2e}, max={np.max(max_nonlinear):.2e}, mean={np.mean(max_nonlinear):.2e}")
    print(f"  energy_highk:")
    print(f"    min={np.min(energy_highk):.4f}, max={np.max(energy_highk):.4f}, mean={np.mean(energy_highk):.4f}")
    print(f"  critical_balance_ratio:")
    print(f"    min={np.min(critical_balance):.3f}, max={np.max(critical_balance):.3f}, mean={np.mean(critical_balance):.3f}")

    # Check for warnings
    print(f"\nWarnings:")
    if np.max(cfl_numbers) > 1.0:
        print(f"  ⚠️  CFL > 1.0 detected (max={np.max(cfl_numbers):.3f})")
        cfl_violation_idx = np.where(cfl_numbers > 1.0)[0]
        print(f"      First violation at t={times[cfl_violation_idx[0]]:.2f} τ_A (sample {cfl_violation_idx[0]})")
        print(f"      Total violations: {len(cfl_violation_idx)} / {len(times)} samples")
    else:
        print(f"  ✓ No CFL violations (max CFL = {np.max(cfl_numbers):.3f})")

    if np.max(max_velocities) > 100.0:
        print(f"  ⚠️  Very high velocities detected (max={np.max(max_velocities):.2e})")
        high_vel_idx = np.where(max_velocities > 100.0)[0]
        print(f"      First occurrence at t={times[high_vel_idx[0]]:.2f} τ_A (sample {high_vel_idx[0]})")
    else:
        print(f"  ✓ Velocities within normal range (max = {np.max(max_velocities):.3f})")

    # Check for exponential growth (instability signature)
    # Fit log(max_velocity) ~ a*t + b to last 50% of data
    if len(times) > 10:
        mid_idx = len(times) // 2
        t_fit = times[mid_idx:]
        vel_fit = max_velocities[mid_idx:]

        # Only fit if velocities are increasing
        if vel_fit[-1] > vel_fit[0]:
            log_vel = np.log(vel_fit + 1e-10)  # Avoid log(0)
            coeffs = np.polyfit(t_fit, log_vel, 1)
            growth_rate = coeffs[0]

            if growth_rate > 0.1:  # Growing faster than 10% per τ_A
                print(f"  ⚠️  EXPONENTIAL GROWTH detected: γ = {growth_rate:.3f} τ_A⁻¹")
                print(f"      Doubling time: {np.log(2)/growth_rate:.2f} τ_A")
                print(f"      This indicates NUMERICAL INSTABILITY")
            else:
                print(f"  ✓ No exponential growth detected (γ = {growth_rate:.3f} τ_A⁻¹)")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze turbulence diagnostics for Issue #82 investigation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--resolution', type=int, choices=[32, 64, 128],
                       help='Load diagnostics for single resolution')
    group.add_argument('--compare', type=int, nargs='+', choices=[32, 64, 128],
                       help='Compare multiple resolutions (e.g., --compare 32 64 128)')
    group.add_argument('--files', type=str, nargs='+',
                       help='Load specific diagnostic files')

    # Output options
    parser.add_argument('--output-dir', type=str, default='examples/output',
                        help='Output directory for plots (default: examples/output)')
    parser.add_argument('--input-dir', type=str, default='examples/output',
                        help='Input directory for diagnostic files (default: examples/output)')
    parser.add_argument('--save-prefix', type=str, default='issue82_diagnostics',
                        help='Prefix for output filenames (default: issue82_diagnostics)')

    args = parser.parse_args()

    print("="*70)
    print("Turbulence Diagnostics Analysis (Issue #82)")
    print("="*70)

    # Load diagnostic files
    diagnostics_dict = {}
    labels = []

    if args.resolution:
        # Single resolution
        input_dir = Path(args.input_dir)
        filename = f"turbulence_diagnostics_{args.resolution}cubed.h5"
        filepath = input_dir / filename

        print(f"\nLoading: {filepath}")
        diag_list, metadata = load_turbulence_diagnostics(str(filepath))

        label = f"{args.resolution}³"
        diagnostics_dict[label] = diag_list
        labels.append(label)

        # Print statistics
        print_summary_statistics(diag_list, label, metadata)

    elif args.compare:
        # Multiple resolutions
        input_dir = Path(args.input_dir)

        for res in args.compare:
            filename = f"turbulence_diagnostics_{res}cubed.h5"
            filepath = input_dir / filename

            print(f"\nLoading: {filepath}")
            try:
                diag_list, metadata = load_turbulence_diagnostics(str(filepath))

                label = f"{res}³"
                diagnostics_dict[label] = diag_list
                labels.append(label)

                # Print statistics
                print_summary_statistics(diag_list, label, metadata)

            except FileNotFoundError:
                print(f"  ⚠️  File not found, skipping: {filepath}")
                continue

    elif args.files:
        # Custom files
        for i, filepath in enumerate(args.files):
            filepath = Path(filepath)
            print(f"\nLoading: {filepath}")

            diag_list, metadata = load_turbulence_diagnostics(str(filepath))

            # Use resolution from metadata if available, otherwise use filename
            if 'resolution' in metadata:
                label = f"{metadata['resolution']}³"
            else:
                label = filepath.stem  # Filename without extension

            diagnostics_dict[label] = diag_list
            labels.append(label)

            # Print statistics
            print_summary_statistics(diag_list, label, metadata)

    # Generate comparison plots
    if len(labels) > 0:
        print(f"\n{'='*70}")
        print("Generating comparison plots...")
        print(f"{'='*70}")

        plot_diagnostics_comparison(
            diagnostics_dict,
            labels,
            output_dir=args.output_dir,
            save_prefix=args.save_prefix
        )

        print(f"\n{'='*70}")
        print("Analysis complete!")
        print(f"{'='*70}")
    else:
        print("\n⚠️  No diagnostic files loaded. Check file paths and try again.")


if __name__ == "__main__":
    main()
