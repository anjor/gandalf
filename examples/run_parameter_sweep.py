#!/usr/bin/env python3
"""
Automated parameter sweep for turbulence parameter identification.

This script runs multiple configurations of alfvenic_cascade_benchmark.py
to systematically identify good parameters for achieving steady-state
turbulence with clean -5/3 spectrum.

Usage:
    # 32³ sweep (4-5 runs × ~10 min = 40-50 minutes)
    python run_parameter_sweep.py --resolution 32 --total-time 100

    # 64³ sweep (with diagnostics)
    python run_parameter_sweep.py --resolution 64 --total-time 100 --save-diagnostics

    # Custom parameter grid
    python run_parameter_sweep.py --resolution 32 --eta-values 0.5 1.0 1.5 --amp-values 0.005 0.01
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import h5py

# Add examples directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def run_single_configuration(
    resolution: int,
    eta: float,
    nu: float,
    amplitude: float,
    hyper_r: int,
    hyper_n: int,
    total_time: float,
    averaging_start: float,
    save_diagnostics: bool,
    output_subdir: str
) -> Dict:
    """
    Run a single parameter configuration.

    Returns:
        dict: Contains success status, runtime, output files
    """
    print(f"\n{'='*70}")
    print(f"Running: N={resolution}³, η={eta:.4f}, ν={nu:.4f}, amplitude={amplitude:.4f}, r={hyper_r}, n={hyper_n}")
    print(f"{'='*70}")

    # Construct output subdirectory
    output_dir = Path('output') / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command with parameter overrides
    cmd = [
        'uv', 'run', 'python', 'examples/alfvenic_cascade_benchmark.py',
        '--resolution', str(resolution),
        '--total-time', str(total_time),
        '--averaging-start', str(averaging_start),
        '--eta', str(eta),
        '--nu', str(nu),
        '--force-amplitude', str(amplitude),
        '--hyper-r', str(hyper_r),
        '--hyper-n', str(hyper_n),
        '--save-spectra'
    ]

    if save_diagnostics:
        cmd.append('--save-diagnostics')

    print(f"Command: {' '.join(cmd)}")
    print(f"Output: {output_dir}")

    start_time = datetime.now()

    try:
        # Run the benchmark
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )

        end_time = datetime.now()
        runtime = (end_time - start_time).total_seconds()

        if result.returncode == 0:
            print(f"✓ Success! Runtime: {runtime:.1f}s ({runtime/60:.1f} min)")

            # Move output files to subdirectory
            output_files = {}
            patterns = [
                f'spectral_data_{resolution}cubed.h5',
                f'energy_history_{resolution}cubed.h5',
                f'alfvenic_cascade_{resolution}cubed.png',
                f'alfvenic_cascade_thesis_style_{resolution}cubed.png'
            ]

            if save_diagnostics:
                patterns.append(f'turbulence_diagnostics_{resolution}cubed.h5')

            for pattern in patterns:
                src = Path('examples/output') / pattern
                if src.exists():
                    dst = output_dir / pattern
                    src.rename(dst)
                    output_files[pattern] = str(dst)

            return {
                'success': True,
                'runtime': runtime,
                'output_dir': str(output_dir),
                'output_files': output_files,
                'stdout': result.stdout[-1000:],  # Last 1000 chars
                'stderr': ''
            }
        else:
            print(f"✗ Failed with return code {result.returncode}")
            print(f"STDERR:\n{result.stderr[-1000:]}")

            return {
                'success': False,
                'runtime': runtime,
                'output_dir': str(output_dir),
                'output_files': {},
                'stdout': result.stdout[-1000:],
                'stderr': result.stderr[-1000:]
            }

    except subprocess.TimeoutExpired:
        print(f"✗ Timeout after 2 hours")
        return {
            'success': False,
            'runtime': 7200,
            'output_dir': str(output_dir),
            'output_files': {},
            'stdout': '',
            'stderr': 'Timeout after 2 hours'
        }
    except Exception as e:
        print(f"✗ Exception: {e}")
        return {
            'success': False,
            'runtime': 0,
            'output_dir': str(output_dir),
            'output_files': {},
            'stdout': '',
            'stderr': str(e)
        }


def run_parameter_sweep(
    resolution: int,
    eta_values: List[float],
    amplitude_values: List[float],
    total_time: float,
    averaging_start: float,
    save_diagnostics: bool,
    hyper_r: int = 2,
    hyper_n: int = 2
) -> List[Dict]:
    """
    Run parameter sweep over eta and amplitude grids.

    Returns:
        list: Results for each configuration
    """
    results = []
    sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*70}")
    print(f"PARAMETER SWEEP: {resolution}³")
    print(f"{'='*70}")
    print(f"Sweep ID: {sweep_id}")
    print(f"Grid: {len(eta_values)} eta × {len(amplitude_values)} amplitude = {len(eta_values) * len(amplitude_values)} runs")
    print(f"Estimated time: ~{len(eta_values) * len(amplitude_values) * (5 if resolution == 32 else 15)} minutes")
    print(f"{'='*70}\n")

    for eta in eta_values:
        for amplitude in amplitude_values:
            # Create unique subdirectory for this configuration
            config_name = f"sweep_{resolution}cubed_eta{eta:.4f}_amp{amplitude:.4f}"
            output_subdir = f"{sweep_id}/{config_name}"

            # Run configuration (now uses command-line arguments)
            result = run_single_configuration(
                resolution=resolution,
                eta=eta,
                nu=eta,  # Keep nu = eta
                amplitude=amplitude,
                hyper_r=hyper_r,
                hyper_n=hyper_n,
                total_time=total_time,
                averaging_start=averaging_start,
                save_diagnostics=save_diagnostics,
                output_subdir=output_subdir
            )

            # Store configuration info
            result['config'] = {
                'resolution': resolution,
                'eta': eta,
                'nu': eta,
                'amplitude': amplitude,
                'hyper_r': hyper_r,
                'hyper_n': hyper_n,
                'total_time': total_time,
                'averaging_start': averaging_start
            }

            results.append(result)

            # Save intermediate results
            results_file = Path('output') / sweep_id / 'sweep_results.json'
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

    return results, sweep_id


def analyze_sweep_results(results: List[Dict], sweep_id: str):
    """
    Analyze sweep results and generate summary plots.
    """
    print(f"\n{'='*70}")
    print("ANALYZING SWEEP RESULTS")
    print(f"{'='*70}\n")

    # Load quality metrics for each configuration
    successful_results = [r for r in results if r['success']]

    if len(successful_results) == 0:
        print("✗ No successful runs to analyze!")
        return

    print(f"Successful runs: {len(successful_results)} / {len(results)}")

    # For each successful run, run analyze_spectrum_quality.py
    quality_metrics = []

    for result in successful_results:
        config = result['config']
        output_dir = Path(result['output_dir'])

        spectral_file = output_dir / f"spectral_data_{config['resolution']}cubed.h5"

        if not spectral_file.exists():
            print(f"⚠ Missing spectral data: {spectral_file}")
            continue

        # Run analysis script
        try:
            cmd = [
                'uv', 'run', 'python', 'examples/analyze_spectrum_quality.py',
                str(spectral_file),
                '--output-dir', str(output_dir)
            ]

            subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            # Load the HDF5 data to extract quality metrics
            from analyze_spectrum_quality import (
                load_spectral_data, load_energy_history,
                compute_power_law_fit, assess_steady_state
            )

            data = load_spectral_data(spectral_file)
            history = load_energy_history(spectral_file)

            k = data['k_perp']
            E_total = data['E_kinetic_avg'] + data['E_magnetic_avg']

            fit = compute_power_law_fit(k, E_total, (2.0, 10.0))
            steady = assess_steady_state(history, data['averaging_start'])

            quality_metrics.append({
                'eta': config['eta'],
                'amplitude': config['amplitude'],
                'steady_state_quality': steady['quality'],
                'steady_state_variation': steady['variation'],
                'spectrum_quality': fit['quality'],
                'spectrum_slope': fit['slope'],
                'spectrum_R2': fit['R2'],
                'runtime': result['runtime'],
                'output_dir': result['output_dir']
            })

        except Exception as e:
            print(f"⚠ Analysis failed for {spectral_file}: {e}")
            continue

    if len(quality_metrics) == 0:
        print("✗ No quality metrics could be extracted!")
        return

    # Generate comparison plot
    plot_sweep_comparison(quality_metrics, sweep_id, results[0]['config']['resolution'])

    # Generate ranking table
    generate_ranking_table(quality_metrics, sweep_id)


def plot_sweep_comparison(metrics: List[Dict], sweep_id: str, resolution: int):
    """Generate comparison plot for parameter sweep."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    eta_vals = np.array([m['eta'] for m in metrics])
    amp_vals = np.array([m['amplitude'] for m in metrics])
    steady_var = np.array([m['steady_state_variation'] for m in metrics])
    spectrum_slope = np.array([m['spectrum_slope'] for m in metrics])
    spectrum_R2 = np.array([m['spectrum_R2'] for m in metrics])
    runtime = np.array([m['runtime'] / 60 for m in metrics])  # Convert to minutes

    # Panel 1: Steady-state variation vs parameters
    ax = axes[0, 0]
    scatter = ax.scatter(eta_vals, amp_vals, c=steady_var, s=200, cmap='RdYlGn_r',
                        edgecolors='black', linewidths=1.5, vmin=0, vmax=10)
    ax.set_xlabel('η (dissipation)', fontsize=11)
    ax.set_ylabel('Forcing amplitude', fontsize=11)
    ax.set_title('Steady-State Quality (ΔE/⟨E⟩ %)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='ΔE/⟨E⟩ (%)')

    # Add quality thresholds
    for m in metrics:
        if m['steady_state_variation'] < 2.0:
            ax.plot(m['eta'], m['amplitude'], 'g*', markersize=15, markeredgecolor='darkgreen', markeredgewidth=1.5)

    # Panel 2: Spectrum slope vs parameters
    ax = axes[0, 1]
    scatter = ax.scatter(eta_vals, amp_vals, c=spectrum_slope, s=200, cmap='coolwarm',
                        edgecolors='black', linewidths=1.5, vmin=-2.0, vmax=-1.3)
    ax.set_xlabel('η (dissipation)', fontsize=11)
    ax.set_ylabel('Forcing amplitude', fontsize=11)
    ax.set_title('Spectrum Slope α (target: -1.667)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='α')

    # Mark good slopes
    for m in metrics:
        if abs(m['spectrum_slope'] - (-5/3)) < 0.2:
            ax.plot(m['eta'], m['amplitude'], 'g*', markersize=15, markeredgecolor='darkgreen', markeredgewidth=1.5)

    # Panel 3: Spectrum R² vs parameters
    ax = axes[1, 0]
    scatter = ax.scatter(eta_vals, amp_vals, c=spectrum_R2, s=200, cmap='RdYlGn',
                        edgecolors='black', linewidths=1.5, vmin=0.8, vmax=1.0)
    ax.set_xlabel('η (dissipation)', fontsize=11)
    ax.set_ylabel('Forcing amplitude', fontsize=11)
    ax.set_title('Spectrum Fit Quality (R²)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='R²')

    # Mark excellent fits
    for m in metrics:
        if m['spectrum_R2'] > 0.95:
            ax.plot(m['eta'], m['amplitude'], 'g*', markersize=15, markeredgecolor='darkgreen', markeredgewidth=1.5)

    # Panel 4: Runtime vs parameters
    ax = axes[1, 1]
    scatter = ax.scatter(eta_vals, amp_vals, c=runtime, s=200, cmap='viridis',
                        edgecolors='black', linewidths=1.5)
    ax.set_xlabel('η (dissipation)', fontsize=11)
    ax.set_ylabel('Forcing amplitude', fontsize=11)
    ax.set_title('Runtime (minutes)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Time (min)')

    fig.suptitle(f'Parameter Sweep Results - {resolution}³\n' +
                f'Green stars (★) indicate EXCELLENT quality (ΔE<2%, |α+5/3|<0.2, R²>0.95)',
                fontsize=13, fontweight='bold')

    plt.tight_layout()

    output_file = Path('output') / sweep_id / f'sweep_comparison_{resolution}cubed.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {output_file}")
    plt.close()


def generate_ranking_table(metrics: List[Dict], sweep_id: str):
    """Generate ranking table sorted by quality."""

    # Define scoring function (lower is better)
    def compute_score(m):
        steady_penalty = m['steady_state_variation']  # Want < 2%
        slope_penalty = abs(m['spectrum_slope'] - (-5/3)) * 10  # Want close to -5/3
        r2_penalty = (1 - m['spectrum_R2']) * 100  # Want close to 1

        return steady_penalty + slope_penalty + r2_penalty

    # Sort by score
    ranked = sorted(metrics, key=compute_score)

    # Print table
    print(f"\n{'='*70}")
    print("PARAMETER RANKING (Best to Worst)")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'η':<8} {'Amp':<8} {'ΔE/⟨E⟩':<10} {'α':<8} {'R²':<8} {'Time':<8} {'Quality'}")
    print(f"{'-'*70}")

    for i, m in enumerate(ranked, 1):
        # Overall assessment
        if (m['steady_state_variation'] < 2.0 and
            abs(m['spectrum_slope'] - (-5/3)) < 0.2 and
            m['spectrum_R2'] > 0.95):
            quality = "★ EXCELLENT"
        elif (m['steady_state_variation'] < 5.0 and
              abs(m['spectrum_slope'] - (-5/3)) < 0.3 and
              m['spectrum_R2'] > 0.90):
            quality = "✓ GOOD"
        elif m['steady_state_variation'] < 10.0:
            quality = "○ FAIR"
        else:
            quality = "✗ POOR"

        print(f"{i:<5} {m['eta']:<8.4f} {m['amplitude']:<8.4f} "
              f"{m['steady_state_variation']:<10.2f} {m['spectrum_slope']:<8.3f} "
              f"{m['spectrum_R2']:<8.4f} {m['runtime']/60:<8.1f} {quality}")

    print(f"{'-'*70}\n")

    # Save to file
    output_file = Path('output') / sweep_id / 'ranking_table.txt'
    with open(output_file, 'w') as f:
        f.write(f"PARAMETER RANKING\n")
        f.write(f"{'='*70}\n")
        f.write(f"{'Rank':<5} {'η':<8} {'Amp':<8} {'ΔE/⟨E⟩':<10} {'α':<8} {'R²':<8} {'Time':<8} {'Quality'}\n")
        f.write(f"{'-'*70}\n")

        for i, m in enumerate(ranked, 1):
            if (m['steady_state_variation'] < 2.0 and
                abs(m['spectrum_slope'] - (-5/3)) < 0.2 and
                m['spectrum_R2'] > 0.95):
                quality = "EXCELLENT"
            elif (m['steady_state_variation'] < 5.0 and
                  abs(m['spectrum_slope'] - (-5/3)) < 0.3 and
                  m['spectrum_R2'] > 0.90):
                quality = "GOOD"
            elif m['steady_state_variation'] < 10.0:
                quality = "FAIR"
            else:
                quality = "POOR"

            f.write(f"{i:<5} {m['eta']:<8.4f} {m['amplitude']:<8.4f} "
                   f"{m['steady_state_variation']:<10.2f} {m['spectrum_slope']:<8.3f} "
                   f"{m['spectrum_R2']:<8.4f} {m['runtime']/60:<8.1f} {quality}\n")

    print(f"✓ Saved ranking table: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Run turbulence parameter sweep')

    # Resolution
    parser.add_argument('--resolution', type=int, required=True, choices=[32, 64, 128],
                       help='Grid resolution (32, 64, or 128)')

    # Parameter ranges
    parser.add_argument('--eta-values', type=float, nargs='+',
                       help='List of eta values to test (default: resolution-dependent)')
    parser.add_argument('--amp-values', type=float, nargs='+',
                       help='List of amplitude values to test (default: resolution-dependent)')

    # Runtime
    parser.add_argument('--total-time', type=float, default=100.0,
                       help='Total simulation time in τ_A (default: 100)')
    parser.add_argument('--averaging-start', type=float, default=70.0,
                       help='When to start averaging in τ_A (default: 70)')

    # Options
    parser.add_argument('--save-diagnostics', action='store_true',
                       help='Save turbulence diagnostics (Issue #82)')
    parser.add_argument('--hyper-r', type=int, default=2,
                       help='Hyper-dissipation order (default: 2)')
    parser.add_argument('--hyper-n', type=int, default=2,
                       help='Hyper-collision order (default: 2)')

    args = parser.parse_args()

    # Set default parameter ranges if not specified
    if args.eta_values is None:
        if args.resolution == 32:
            args.eta_values = [0.5, 1.0, 1.5, 2.0]
        elif args.resolution == 64:
            args.eta_values = [2.0, 3.0, 5.0, 10.0]
        else:  # 128
            args.eta_values = [1.5, 2.0, 2.5]

    if args.amp_values is None:
        if args.resolution == 32:
            args.amp_values = [0.005, 0.01, 0.02]
        elif args.resolution == 64:
            args.amp_values = [0.002, 0.005, 0.01]
        else:  # 128
            args.amp_values = [0.005, 0.01, 0.015]

    # Run sweep
    results, sweep_id = run_parameter_sweep(
        resolution=args.resolution,
        eta_values=args.eta_values,
        amplitude_values=args.amp_values,
        total_time=args.total_time,
        averaging_start=args.averaging_start,
        save_diagnostics=args.save_diagnostics,
        hyper_r=args.hyper_r,
        hyper_n=args.hyper_n
    )

    # Analyze results
    analyze_sweep_results(results, sweep_id)

    print(f"\n{'='*70}")
    print("PARAMETER SWEEP COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: output/{sweep_id}/")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
