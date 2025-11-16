#!/usr/bin/env python3
"""
Enhanced spectral analysis tool for turbulence parameter identification.

This script:
1. Loads spectral data from alfvenic_cascade_benchmark.py output
2. Computes -5/3 power law fit quality in the inertial range
3. Generates detailed diagnostic plots
4. Quantifies steady-state quality and spectrum cleanliness

Usage:
    python analyze_spectrum_quality.py output/spectral_data_32cubed.h5
    python analyze_spectrum_quality.py --compare output/spectral_data_*.h5
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple


def load_spectral_data(filepath: Path) -> Dict:
    """Load spectral data from HDF5 file."""
    data = {}
    with h5py.File(filepath, 'r') as f:
        # Load attributes
        data['resolution'] = f.attrs['resolution']
        data['eta'] = f.attrs['eta']
        data['nu'] = f.attrs['nu']
        data['hyper_r'] = f.attrs['hyper_r']
        data['hyper_n'] = f.attrs['hyper_n']
        data['force_amplitude'] = f.attrs['force_amplitude']
        data['force_n_min'] = f.attrs['force_n_min']
        data['force_n_max'] = f.attrs['force_n_max']
        data['dt'] = f.attrs['dt']
        data['total_time'] = f.attrs['total_time']
        data['averaging_start'] = f.attrs['averaging_start']
        data['averaging_duration'] = f.attrs['averaging_duration']
        data['v_A'] = f.attrs['v_A']
        # Domain size (if present)
        if 'domain_size' in f.attrs:
            data['domain_size'] = f.attrs['domain_size']

        # Load spectral data
        data['k_perp'] = f['k_perp'][:]
        # Prefer mode-number axis if present (no 2π factors for user-facing analysis)
        if 'n_perp' in f:
            data['n_perp'] = f['n_perp'][:]
        data['E_kinetic_avg'] = f['E_kinetic_avg'][:]
        data['E_magnetic_avg'] = f['E_magnetic_avg'][:]
        data['E_kinetic_std'] = f['E_kinetic_std'][:]
        data['E_magnetic_std'] = f['E_magnetic_std'][:]

        # Load time series if available
        if 'E_kinetic_all' in f:
            data['E_kinetic_all'] = f['E_kinetic_all'][:]
            data['E_magnetic_all'] = f['E_magnetic_all'][:]

        # Embedded history fallback (benchmark writes these datasets)
        if 'times' in f and 'energy_total' in f:
            data['times'] = f['times'][:]
            data['energy_total'] = f['energy_total'][:]

    return data


def load_energy_history(filepath: Path) -> Dict:
    """Load energy history from HDF5 file."""
    history_filepath = filepath.parent / filepath.name.replace('spectral_data', 'energy_history')

    if not history_filepath.exists():
        # Fallback to embedded history inside spectral file
        try:
            with h5py.File(filepath, 'r') as g:
                if 'times' in g and 'energy_total' in g:
                    return {
                        'times': g['times'][:],
                        'E_total': g['energy_total'][:],
                        'E_kinetic': None,
                        'E_magnetic': None,
                        'E_compressive': None,
                    }
        except Exception:
            pass
        print(f"Warning: Energy history file not found: {history_filepath}")
        return None

    history = {}
    with h5py.File(history_filepath, 'r') as f:
        history['times'] = f['times'][:]
        history['E_total'] = f['E_total'][:]
        history['E_kinetic'] = f['E_kinetic'][:]
        history['E_magnetic'] = f['E_magnetic'][:]
        history['E_compressive'] = f['E_compressive'][:]

    return history


def compute_power_law_fit(x: np.ndarray, E: np.ndarray, x_range: Tuple[float, float],
                          expected_slope: float = -5/3) -> Dict:
    """
    Compute power law fit E(k) ~ k^α in specified k range.

    Returns:
        dict: Contains slope, intercept, R², residuals, and quality metrics
    """
    # Select inertial range
    mask = (x >= x_range[0]) & (x <= x_range[1]) & (E > 0)
    x_fit = x[mask]
    E_fit = E[mask]

    if len(x_fit) < 3:
        return {
            'slope': np.nan,
            'intercept': np.nan,
            'R2': 0.0,
            'rmse': np.inf,
            'n_points': 0,
            'slope_error': np.nan,
            'quality': 'INSUFFICIENT_DATA'
        }

    # Fit in log-log space: log(E) = slope * log(k) + intercept
    log_x = np.log10(x_fit)
    log_E = np.log10(E_fit)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_E)

    # Compute residuals
    log_E_fit = slope * log_x + intercept
    residuals = log_E - log_E_fit
    rmse = np.sqrt(np.mean(residuals**2))

    # Quality assessment
    R2 = r_value**2
    slope_error = abs(slope - expected_slope)

    if R2 > 0.95 and slope_error < 0.2:
        quality = 'EXCELLENT'
    elif R2 > 0.90 and slope_error < 0.3:
        quality = 'GOOD'
    elif R2 > 0.80:
        quality = 'FAIR'
    else:
        quality = 'POOR'

    return {
        'slope': slope,
        'intercept': intercept,
        'R2': R2,
        'rmse': rmse,
        'n_points': len(x_fit),
        'slope_error': slope_error,
        'residuals': residuals,
        'x_fit': x_fit,
        'E_fit': E_fit,
        'quality': quality
    }


def assess_steady_state(history: Dict, averaging_start: float) -> Dict:
    """Assess steady-state quality from energy history."""
    if history is None:
        return {
            'quality': 'NO_DATA',
            'variation': np.nan,
            'mean_energy': np.nan,
            'std_energy': np.nan
        }

    # Select averaging window
    mask = history['times'] >= averaging_start
    if history.get('E_total') is None:
        return {
            'quality': 'NO_DATA',
            'variation': np.nan,
            'mean_energy': np.nan,
            'std_energy': np.nan
        }
    E_window = history['E_total'][mask]

    if len(E_window) < 2:
        return {
            'quality': 'INSUFFICIENT_DATA',
            'variation': np.nan,
            'mean_energy': np.nan,
            'std_energy': np.nan
        }

    mean_E = np.mean(E_window)
    std_E = np.std(E_window)
    variation = std_E / mean_E * 100  # Percent variation

    # Classify steady-state quality
    if variation < 2.0:
        quality = 'EXCELLENT'
    elif variation < 5.0:
        quality = 'GOOD'
    elif variation < 10.0:
        quality = 'FAIR'
    else:
        quality = 'POOR'

    return {
        'quality': quality,
        'variation': variation,
        'mean_energy': mean_E,
        'std_energy': std_E,
        'n_samples': len(E_window)
    }


def plot_detailed_analysis(data: Dict, history: Dict, save_path: Path):
    """Create comprehensive 6-panel diagnostic plot."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Use saved mode-number axis if available; otherwise convert
    if 'n_perp' in data:
        n = data['n_perp']
    else:
        k = data['k_perp']
        Lx = float(data.get('domain_size', (2*np.pi, 2*np.pi, 2*np.pi))[0]) if 'domain_size' in data else 2*np.pi
        n = k * Lx / (2.0 * np.pi)
    E_kin = data['E_kinetic_avg']
    E_mag = data['E_magnetic_avg']
    E_kin_std = data['E_kinetic_std']
    E_mag_std = data['E_magnetic_std']
    E_total = E_kin + E_mag

    # Define inertial range in mode number
    res = int(data.get('resolution', 64))
    if res == 32:
        n_inertial = (2.0, 8.0)
    elif res == 64:
        n_inertial = (3.0, 12.0)
    else:
        n_inertial = (3.0, 20.0)

    # Compute fits
    fit_kin = compute_power_law_fit(n, E_kin, n_inertial)
    fit_mag = compute_power_law_fit(n, E_mag, n_inertial)
    fit_total = compute_power_law_fit(n, E_total, n_inertial)

    # Assess steady state
    steady_state = assess_steady_state(history, data['averaging_start'])

    # Panel 1: Energy history
    ax1 = fig.add_subplot(gs[0, 0])
    if history is not None:
        if history.get('times') is not None and history.get('E_total') is not None:
            ax1.plot(history['times'], history['E_total'], 'k-', linewidth=1.5, label='Total')
        if history.get('times') is not None and history.get('E_kinetic') is not None:
            ax1.plot(history['times'], history['E_kinetic'], 'b-', alpha=0.7, label='Kinetic')
        if history.get('times') is not None and history.get('E_magnetic') is not None:
            ax1.plot(history['times'], history['E_magnetic'], 'r-', alpha=0.7, label='Magnetic')
        ax1.axvline(data['averaging_start'], color='gray', linestyle='--',
                   label=f'Averaging start')
        ax1.set_xlabel('Time (τ_A)', fontsize=11)
        ax1.set_ylabel('Energy', fontsize=11)
        ax1.set_title(f'Energy Evolution | Steady-State: {steady_state["quality"]} (ΔE/⟨E⟩ = {steady_state["variation"]:.2f}%)',
                     fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No energy history data', ha='center', va='center')
        ax1.set_title('Energy Evolution', fontsize=12, fontweight='bold')

    # Panel 2: Energy history (log scale)
    ax2 = fig.add_subplot(gs[0, 1])
    if history is not None and history.get('times') is not None and history.get('E_total') is not None:
        ax2.semilogy(history['times'], history['E_total'], 'k-', linewidth=1.5, label='Total')
        ax2.axvline(data['averaging_start'], color='gray', linestyle='--')
        ax2.set_xlabel('Time (τ_A)', fontsize=11)
        ax2.set_ylabel('Energy (log scale)', fontsize=11)
        ax2.set_title('Energy Evolution (Detect Exponential Growth)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, which='both')
    else:
        ax2.text(0.5, 0.5, 'No energy history data', ha='center', va='center')
        ax2.set_title('Energy Evolution (Log Scale)', fontsize=12, fontweight='bold')

    # Panel 3: Combined spectrum
    ax3 = fig.add_subplot(gs[1, 0])
    mask_pos = n > 0
    ax3.loglog(n[mask_pos], E_total[mask_pos], 'ko-', markersize=4, linewidth=1, label='Total (K+M)')

    # Plot -5/3 reference line
    if fit_total['n_points'] > 0:
        n_ref = np.logspace(np.log10(n_inertial[0]), np.log10(n_inertial[1]), 50)
        E_ref = 10**fit_total['intercept'] * n_ref**(-5/3)
        ax3.loglog(n_ref, E_ref, 'g--', linewidth=2, label=f'n^(-5/3) reference')

        # Highlight inertial range
        ax3.axvspan(n_inertial[0], n_inertial[1], alpha=0.1, color='green', label='Inertial range')

    ax3.set_xlabel('Mode number n', fontsize=11)
    ax3.set_ylabel('E(k⊥)', fontsize=11)
    ax3.set_title(f'Total Spectrum | Fit: {fit_total["quality"]} (α = {fit_total["slope"]:.3f}, R² = {fit_total["R2"]:.3f})',
                 fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.set_ylim(1e-3, None)

    # Panel 4: Kinetic vs Magnetic spectra
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.loglog(n[mask_pos], E_kin[mask_pos], 'bo-', markersize=4, linewidth=1, label=f'Kinetic (α={fit_kin["slope"]:.3f})')
    ax4.loglog(n[mask_pos], E_mag[mask_pos], 'ro-', markersize=4, linewidth=1, label=f'Magnetic (α={fit_mag["slope"]:.3f})')

    # Plot -5/3 reference
    if fit_total['n_points'] > 0:
        n_ref = np.logspace(np.log10(n_inertial[0]), np.log10(n_inertial[1]), 50)
        E_ref_kin = 10**fit_kin['intercept'] * n_ref**(-5/3)
        E_ref_mag = 10**fit_mag['intercept'] * n_ref**(-5/3)
        ax4.loglog(n_ref, E_ref_kin, 'b--', linewidth=1.5, alpha=0.7)
        ax4.loglog(n_ref, E_ref_mag, 'r--', linewidth=1.5, alpha=0.7)
        ax4.axvspan(n_inertial[0], n_inertial[1], alpha=0.1, color='green')

    ax4.set_xlabel('Mode number n', fontsize=11)
    ax4.set_ylabel('E(k⊥)', fontsize=11)
    ax4.set_title(f'Kinetic vs Magnetic | K: {fit_kin["quality"]}, M: {fit_mag["quality"]}',
                 fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_ylim(1e-3, None)

    # Panel 5: Fit residuals
    ax5 = fig.add_subplot(gs[2, 0])
    if fit_total['n_points'] > 0:
        ax5.semilogx(fit_total['x_fit'], fit_total['residuals'], 'ko-', markersize=4, linewidth=1)
        ax5.axhline(0, color='g', linestyle='--', linewidth=2)
        ax5.fill_between(fit_total['x_fit'], -0.1, 0.1, alpha=0.2, color='green', label='±0.1 dex')
        ax5.set_xlabel('Mode number n', fontsize=11)
        ax5.set_ylabel('Residuals (log₁₀)', fontsize=11)
        ax5.set_title(f'Total Spectrum Fit Residuals | RMSE = {fit_total["rmse"]:.4f}',
                     fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Insufficient data for fit', ha='center', va='center')
        ax5.set_title('Fit Residuals', fontsize=12, fontweight='bold')

    # Panel 6: Parameter summary
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    # Compose domain string from attributes if available (no explicit 2π mention)
    if 'domain_size' in data:
        Lx, Ly, Lz = data['domain_size']
        domain_str = f"L = ({Lx:.3g}, {Ly:.3g}, {Lz:.3g})"
    else:
        domain_str = "(not specified)"

    summary_text = f"""
PARAMETER SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Resolution:  {data['resolution']}³
Domain:      {domain_str}
v_A:         {data['v_A']:.2f}

DISSIPATION
η (resistivity):      {data['eta']:.4f}
ν (collisions):       {data['nu']:.4f}
r (hyper-order):      {data['hyper_r']}
n (collision order):  {data['hyper_n']}

FORCING
Amplitude:            {data['force_amplitude']:.4f}
Mode range:           n ∈ [{data['force_n_min']}, {data['force_n_max']}]

RUNTIME
Total time:           {data['total_time']:.1f} τ_A
Averaging window:     {data['averaging_start']:.1f} - {data['total_time']:.1f} τ_A
Timestep:             {data['dt']:.5f}

QUALITY ASSESSMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Steady State:         {steady_state['quality']}
  ΔE/⟨E⟩:             {steady_state['variation']:.2f}%

Total Spectrum:       {fit_total['quality']}
  Slope α:            {fit_total['slope']:.3f} (target: -1.667)
  R²:                 {fit_total['R2']:.4f}
  RMSE:               {fit_total['rmse']:.4f}

Kinetic Spectrum:     {fit_kin['quality']}
  Slope α:            {fit_kin['slope']:.3f}
  R²:                 {fit_kin['R2']:.4f}

Magnetic Spectrum:    {fit_mag['quality']}
  Slope α:            {fit_mag['slope']:.3f}
  R²:                 {fit_mag['R2']:.4f}
"""

    ax6.text(0.05, 0.95, summary_text, fontsize=9, family='monospace',
             verticalalignment='top', transform=ax6.transAxes)

    # Overall title
    fig.suptitle(f'Turbulence Spectrum Quality Analysis - N={data["resolution"]}³',
                fontsize=14, fontweight='bold')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved detailed analysis plot: {save_path}")
    plt.close()


def generate_summary_report(data: Dict, history: Dict) -> str:
    """Generate text summary report."""
    # Use saved mode-number axis if available; otherwise convert
    if 'n_perp' in data:
        n = data['n_perp']
    else:
        k = data['k_perp']
        Lx = float(data.get('domain_size', (2*np.pi, 2*np.pi, 2*np.pi))[0])
        n = k * Lx / (2.0 * np.pi)
    res = int(data.get('resolution', 64))
    if res == 32:
        n_inertial = (2.0, 8.0)
    elif res == 64:
        n_inertial = (3.0, 12.0)
    else:
        n_inertial = (3.0, 20.0)
    E_total = data['E_kinetic_avg'] + data['E_magnetic_avg']

    fit_total = compute_power_law_fit(n, E_total, n_inertial)
    steady_state = assess_steady_state(history, data['averaging_start'])

    report = f"""
{'='*70}
TURBULENCE PARAMETER QUALITY REPORT
{'='*70}

Resolution: {data['resolution']}³
Parameters: η={data['eta']:.4f}, ν={data['nu']:.4f}, r={data['hyper_r']}, n={data['hyper_n']}
Forcing:    amplitude={data['force_amplitude']:.4f}, modes=[{data['force_n_min']}, {data['force_n_max']}]
Runtime:    {data['total_time']:.1f} τ_A (averaging: {data['averaging_start']:.1f}-{data['total_time']:.1f})

RESULTS
───────────────────────────────────────────────────────────────────────
Steady State:   {steady_state['quality']:12s}  ΔE/⟨E⟩ = {steady_state['variation']:6.2f}%
Spectrum Fit:   {fit_total['quality']:12s}  α = {fit_total['slope']:6.3f}, R² = {fit_total['R2']:.4f}

OVERALL ASSESSMENT
───────────────────────────────────────────────────────────────────────
"""

    # Overall assessment
    if steady_state['quality'] in ['EXCELLENT', 'GOOD'] and fit_total['quality'] in ['EXCELLENT', 'GOOD']:
        report += "✓ RECOMMENDED: This parameter set meets quality criteria.\n"
    elif steady_state['quality'] == 'FAIR' or fit_total['quality'] == 'FAIR':
        report += "⚠ MARGINAL: Parameter set is usable but not ideal.\n"
    else:
        report += "✗ NOT RECOMMENDED: Parameter set does not meet quality criteria.\n"

    report += f"{'='*70}\n"

    return report


def main():
    parser = argparse.ArgumentParser(description='Analyze turbulence spectrum quality')
    parser.add_argument('files', nargs='+', help='HDF5 spectral data files')
    parser.add_argument('--compare', action='store_true',
                       help='Generate comparison plot for multiple files')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for plots')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Process each file
    for filepath in args.files:
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            continue

        print(f"\nAnalyzing: {filepath}")
        print("=" * 70)

        # Load data
        data = load_spectral_data(filepath)
        history = load_energy_history(filepath)

        # Generate report
        report = generate_summary_report(data, history)
        print(report)

        # Generate detailed plot
        plot_name = filepath.stem.replace('spectral_data', 'quality_analysis') + '.png'
        plot_path = output_dir / plot_name
        plot_detailed_analysis(data, history, plot_path)

    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
