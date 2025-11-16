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
import concurrent.futures as futures
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
    amplitude: float,
    hyper_r: int,
    hyper_n: int,
    total_time: float,
    averaging_start: float,
    save_diagnostics: bool,
    output_subdir: str,
    balanced_elsasser: bool,
    max_nz: int,
    include_nz0: bool,
    correlation: float,
    checkpoint_interval_time: float = None,
    checkpoint_interval_steps: int = None,
    checkpoint_final: bool = True,
) -> Dict:
    """
    Run a single parameter configuration.

    Returns:
        dict: Contains success status, runtime, output files
    """
    print(f"\n{'='*70}")
    print(f"Running: N={resolution}³, η={eta:.4f}, amplitude={amplitude:.4f}, r={hyper_r}, n={hyper_n}")
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
        '--force-amplitude', str(amplitude),
        '--hyper-r', str(hyper_r),
        '--hyper-n', str(hyper_n),
        '--save-spectra',
        '--output-dir', str(output_dir)
    ]

    # Use balanced Elsasser forcing with low-|nz| restriction by default
    if balanced_elsasser:
        cmd.append('--balanced-elsasser')
        cmd += ['--max-nz', str(max_nz)]
        if include_nz0:
            cmd.append('--include-nz0')
        if correlation > 0.0:
            cmd += ['--correlation', str(correlation)]

    if save_diagnostics:
        cmd.append('--save-diagnostics')

    # Checkpoint configuration
    if checkpoint_interval_time is not None:
        cmd += ['--checkpoint-interval-time', str(checkpoint_interval_time)]
    if checkpoint_interval_steps is not None:
        cmd += ['--checkpoint-interval-steps', str(checkpoint_interval_steps)]
    if not checkpoint_final:
        cmd.append('--no-checkpoint-final')

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

        # Persist full logs for post-mortem
        (output_dir / 'stdout.txt').write_text(result.stdout or '')
        (output_dir / 'stderr.txt').write_text(result.stderr or '')

        if result.returncode == 0:
            print(f"✓ Success! Runtime: {runtime:.1f}s ({runtime/60:.1f} min)")
            output_files = {}
            # Files should already exist in output_dir since we passed --output-dir
            for pattern in [
                f'spectral_data_{resolution}cubed.h5',
                f'energy_history_{resolution}cubed.h5',
                f'alfvenic_cascade_{resolution}cubed.png',
                f'alfvenic_cascade_thesis_style_{resolution}cubed.png',
                f'turbulence_diagnostics_{resolution}cubed.h5'
            ]:
                candidate = output_dir / pattern
                if candidate.exists():
                    output_files[pattern] = str(candidate)

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
            tail = (result.stderr or '')[-1000:]
            print(f"STDERR tail:\n{tail}")
            print(f"  Full logs: {output_dir}/stdout.txt, {output_dir}/stderr.txt")

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
    hyper_r_values: List[int],
    hyper_n: int = 2,
    balanced_elsasser: bool = True,
    max_nz: int = 1,
    include_nz0: bool = False,
    correlation: float = 0.0,
    checkpoint_interval_time: float = None,
    checkpoint_interval_steps: int = None,
    checkpoint_final: bool = True,
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
    print(f"Grid: {len(eta_values)} eta × {len(amplitude_values)} amplitude × {len(hyper_r_values)} r = {len(eta_values) * len(amplitude_values) * len(hyper_r_values)} runs")
    print(f"Estimated time per run: ~{5 if resolution == 32 else 15} minutes")
    print(f"{'='*70}\n")

    # Build task list
    tasks = []
    for r in hyper_r_values:
        for eta in eta_values:
            for amplitude in amplitude_values:
                config_name = f"sweep_{resolution}cubed_r{r}_eta{eta:.4f}_amp{amplitude:.4f}"
                output_subdir = f"{sweep_id}/{config_name}"
                tasks.append((r, eta, amplitude, output_subdir))

    # Parallel execution using thread pool (spawns separate processes)
    # Control concurrency via --jobs
    return results, sweep_id, tasks


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

            # Fit in mode-number space, resolution-aware inertial range
            if 'n_perp' in data:
                n = data['n_perp']
            else:
                # fallback compute n from domain if needed
                Lx = float(data.get('domain_size', (1.0, 1.0, 1.0))[0]) if 'domain_size' in data else 1.0
                n = data['k_perp'] * Lx / (2.0 * np.pi)

            if config['resolution'] == 32:
                n_range = (2.0, 8.0)
            elif config['resolution'] == 64:
                n_range = (3.0, 12.0)
            else:
                n_range = (3.0, 20.0)

            E_total = data['E_kinetic_avg'] + data['E_magnetic_avg']
            fit = compute_power_law_fit(n, E_total, n_range)
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
    parser.add_argument('--hyper-r-values', type=int, nargs='+', default=[2],
                       help='List of hyper-dissipation orders to test (e.g., 2 3 4)')
    parser.add_argument('--hyper-n', type=int, default=2,
                       help='Hyper-collision order (default: 2)')
    parser.add_argument('--balanced-elsasser', action='store_true',
                       help='Use balanced Elsasser forcing (recommended)')
    parser.add_argument('--max-nz', type=int, default=1,
                       help='Max |nz| for forcing (default: 1)')
    parser.add_argument('--include-nz0', action='store_true',
                       help='Include kz=0 plane in forcing')
    parser.add_argument('--correlation', type=float, default=0.0,
                       help='Correlation between z+ and z- forcing [0,1)')

    # Checkpoint configuration
    parser.add_argument('--checkpoint-interval-time', type=float, default=None,
                       help='Save checkpoint every N τ_A (passed to benchmark script)')
    parser.add_argument('--checkpoint-interval-steps', type=int, default=None,
                       help='Save checkpoint every N steps (passed to benchmark script)')
    parser.add_argument('--checkpoint-final', action='store_true', default=True,
                       help='Save final checkpoint (default: enabled)')
    parser.add_argument('--no-checkpoint-final', dest='checkpoint_final', action='store_false',
                       help='Disable final checkpoint')

    parser.add_argument('--jobs', type=int, default=1,
                       help='Number of parallel runs (default: 1)')

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
    results, sweep_id, tasks = run_parameter_sweep(
        resolution=args.resolution,
        eta_values=args.eta_values,
        amplitude_values=args.amp_values,
        total_time=args.total_time,
        averaging_start=args.averaging_start,
        save_diagnostics=args.save_diagnostics,
        hyper_r_values=args.hyper_r_values,
        hyper_n=args.hyper_n,
        balanced_elsasser=args.balanced_elsasser or True,
        max_nz=args.max_nz,
        include_nz0=args.include_nz0,
        correlation=args.correlation,
        checkpoint_interval_time=args.checkpoint_interval_time,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        checkpoint_final=args.checkpoint_final,
    )

    # Execute tasks in parallel (each task spawns an external process)
    print(f"\nLaunching {len(tasks)} runs with jobs={args.jobs}...\n")
    Path('output', sweep_id).mkdir(parents=True, exist_ok=True)

    def _run(task):
        r, eta, amplitude, output_subdir = task
        res = run_single_configuration(
            resolution=args.resolution,
            eta=eta,
            amplitude=amplitude,
            hyper_r=r,
            hyper_n=args.hyper_n,
            total_time=args.total_time,
            averaging_start=args.averaging_start,
            save_diagnostics=args.save_diagnostics,
            output_subdir=output_subdir,
            balanced_elsasser=(args.balanced_elsasser or True),
            max_nz=args.max_nz,
            include_nz0=args.include_nz0,
            correlation=args.correlation,
            checkpoint_interval_time=args.checkpoint_interval_time,
            checkpoint_interval_steps=args.checkpoint_interval_steps,
            checkpoint_final=args.checkpoint_final,
        )
        res['config'] = {
            'resolution': args.resolution,
            'eta': eta,
            'amplitude': amplitude,
            'hyper_r': r,
            'hyper_n': args.hyper_n,
            'total_time': args.total_time,
            'averaging_start': args.averaging_start,
            'balanced_elsasser': (args.balanced_elsasser or True),
            'max_nz': args.max_nz,
            'include_nz0': args.include_nz0,
            'correlation': args.correlation,
        }
        return res

    with futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        for res in ex.map(_run, tasks):
            results.append(res)
            # Save intermediate results incrementally
            results_file = Path('output') / sweep_id / 'sweep_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)

    # Analyze results once all runs complete
    analyze_sweep_results(results, sweep_id)

    print(f"\n{'='*70}")
    print("PARAMETER SWEEP COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: output/{sweep_id}/")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
