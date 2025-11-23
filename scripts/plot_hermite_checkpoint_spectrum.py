#!/usr/bin/env python3
"""
Plot Hermite Moment Spectrum from KRMHD Checkpoint

Loads a checkpoint file and plots the velocity-space cascade showing the
Hermite moment energy distribution E_m vs m. Useful for analyzing saved
states from hermite_cascade_benchmark.py without re-running simulations.

Physics:
- Hermite moment m represents velocity-space structure (m=0: density, m=1: velocity, m≥2: higher)
- Expected spectrum: E_m ~ m^(-1/2) from phase mixing/collisional balance
- Steep spectra (m^(-2) or worse) indicate collisions dominate phase mixing

Usage:
    # Basic usage
    python plot_hermite_checkpoint_spectrum.py checkpoint.h5

    # Custom output filename
    python plot_hermite_checkpoint_spectrum.py --output fig.png checkpoint.h5

    # Interactive display
    python plot_hermite_checkpoint_spectrum.py --show checkpoint.h5

    # Batch process
    for f in examples/output/hermite_cascade_checkpoint_*.h5; do
        python plot_hermite_checkpoint_spectrum.py "$f"
    done

Output:
- PNG file with Hermite moment spectrum
- Console summary with power law fit and interpretation

Author: KRMHD Project
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Add src to path for krmhd imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from krmhd.io import load_checkpoint
from krmhd.diagnostics import hermite_moment_energy


def fit_power_law(m_values, E_m, m_min=2, m_max=16):
    """
    Fit power law E_m ~ m^α to Hermite spectrum.

    Args:
        m_values: Array of Hermite moment indices
        E_m: Array of Hermite moment energies
        m_min: Minimum moment for fit range (default: 2, skip m=0,1)
        m_max: Maximum moment for fit range

    Returns:
        slope: Power law exponent α
        intercept: Log10 of normalization constant
        r_squared: Coefficient of determination (quality of fit)
        m_fit: Moment values in fit range
        E_fit_predicted: Fitted power law values
    """
    # Select fit range
    mask = (m_values >= m_min) & (m_values <= m_max) & (E_m > 0)
    m_fit = m_values[mask]
    E_fit = E_m[mask]

    if len(m_fit) < 3:
        # Not enough points for meaningful fit
        return np.nan, np.nan, np.nan, m_fit, np.full_like(m_fit, np.nan, dtype=float)

    # Log-log linear regression
    log_m = np.log10(m_fit)
    log_E = np.log10(E_fit)

    slope, intercept, r_value, _, _ = stats.linregress(log_m, log_E)
    r_squared = r_value**2

    # Predicted values for plotting
    E_fit_predicted = 10**(intercept + slope * log_m)

    return slope, intercept, r_squared, m_fit, E_fit_predicted


def interpret_slope(slope):
    """
    Provide interpretation of fitted power law slope.

    Args:
        slope: Fitted power law exponent

    Returns:
        status: "good", "too_steep", or "too_shallow"
        message: Interpretation message
    """
    if np.isnan(slope):
        return "unknown", "Unable to fit power law"

    expected = -0.5
    error = abs(slope - expected) / abs(expected)

    if error < 0.2:  # Within 20%
        return "good", "✓ Good phase mixing/collision balance"
    elif slope < -1.5:
        return "too_steep", "✗ Spectrum too steep: Collisions dominate phase mixing\n  → Try reducing nu or increasing forcing amplitude"
    elif slope < expected - 0.2:
        return "too_steep", "⚠ Spectrum somewhat too steep: Consider reducing nu or increasing amplitude"
    elif slope > -0.2:
        return "too_shallow", "✗ Spectrum too shallow: Phase mixing dominates collisions\n  → Try increasing nu or reducing forcing amplitude"
    else:
        return "too_shallow", "⚠ Spectrum somewhat too shallow: Consider increasing nu or reducing amplitude"


def main():
    """Main function to plot Hermite spectrum from checkpoint."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Plot Hermite moment spectrum from KRMHD checkpoint file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s checkpoint.h5
  %(prog)s --output fig.png checkpoint.h5
  %(prog)s --show checkpoint.h5
        """
    )
    parser.add_argument('checkpoint', type=Path,
                        help='Path to checkpoint HDF5 file')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output PNG filename (default: auto-generated)')
    parser.add_argument('--show', action='store_true',
                        help='Display plot interactively')

    args = parser.parse_args()

    # Validate checkpoint file exists
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    print(f"Loading checkpoint: {args.checkpoint.name}")

    # Load checkpoint
    try:
        state, grid, metadata = load_checkpoint(str(args.checkpoint))
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    # Compute Hermite moment energy
    E_m = np.array(hermite_moment_energy(state))
    M = state.M
    m_values = np.arange(M + 1)

    # Total energy
    E_total = np.sum(E_m)

    # Print summary
    print(f"\nCheckpoint: {args.checkpoint.name}")
    print(f"Time: t = {state.time:.2f} τ_A")
    print(f"Grid: {grid.Nx}×{grid.Ny}×{grid.Nz}")
    print(f"Hermite moments: M = {M}")
    print(f"Total Hermite energy: E_total = {E_total:.6e}")

    # Fit power law
    m_fit_min = 2  # Skip forced modes m=0,1
    m_fit_max = min(16, M // 2)  # Avoid truncation artifacts

    slope, intercept, r_squared, m_fit, E_fit_predicted = fit_power_law(
        m_values, E_m, m_min=m_fit_min, m_max=m_fit_max
    )

    print(f"\nPower law fit (m ∈ [{m_fit_min}, {m_fit_max}]):")
    if not np.isnan(slope):
        print(f"  E_m ~ m^({slope:.3f})  (expected: -0.5)")
        print(f"  R² = {r_squared:.4f}")
        print(f"  Relative error: {abs(slope + 0.5) / 0.5 * 100:.1f}%")
    else:
        print(f"  Unable to fit power law (insufficient data)")

    # Interpret slope
    status, message = interpret_slope(slope)
    print(f"\n{message}\n")

    # Generate output filename if not specified
    if args.output is None:
        output_name = f"hermite_spectrum_checkpoint_t{state.time:.0f}.png"
        args.output = args.checkpoint.parent / output_name

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Filter out zero values for log-log plot
    mask_nonzero = E_m > 0
    m_plot = m_values[mask_nonzero]
    E_m_plot = E_m[mask_nonzero]

    # Main spectrum plot
    ax.loglog(m_plot, E_m_plot, 'bo-', linewidth=2, markersize=6,
              label='Checkpoint data', zorder=3)

    # Reference slope m^(-1/2)
    m_ref = np.array([2.0, float(m_fit_max)])
    # Normalize to E_m at m=3 (index 2 in fit range, away from forcing)
    if len(m_fit) > 2 and 3 in m_fit:
        E_ref = m_ref**(-0.5) * E_m[3] / (3.0**(-0.5))
    else:
        E_ref = m_ref**(-0.5)
    ax.loglog(m_ref, E_ref, 'k--', linewidth=2, label='m$^{-1/2}$ reference', zorder=2)

    # Fitted power law
    if not np.isnan(slope) and len(m_fit) >= 3:
        m_fit_plot = np.logspace(np.log10(m_fit_min), np.log10(m_fit_max), 50)
        E_fit_plot = 10**(intercept) * m_fit_plot**slope
        ax.loglog(m_fit_plot, E_fit_plot, 'r-', linewidth=1.5, alpha=0.7,
                  label=f'Fit: m$^{{{slope:.2f}}}$, R²={r_squared:.3f}', zorder=1)

    # Highlight forcing range (m=0,1)
    ax.axvspan(0.5, 1.5, color='green', alpha=0.15, label='Forced: g₀, g₁', zorder=0)

    # Highlight fit range
    ax.axvspan(m_fit_min, m_fit_max, color='orange', alpha=0.1,
               label=f'Fit range [{m_fit_min},{m_fit_max}]', zorder=0)

    # Axis labels and title
    ax.set_xlabel('Hermite moment m', fontsize=12)
    ax.set_ylabel('Energy E$_m$', fontsize=12)
    ax.set_title(f'Hermite Moment Spectrum (t={state.time:.1f} τ$_A$)',
                 fontsize=14, fontweight='bold')

    # Grid and legend
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10, loc='best')

    # Axis limits
    ax.set_xlim(0.8, M)
    ax.set_ylim(bottom=max(1e-5, np.min(E_m_plot[E_m_plot > 0]) * 0.1))

    plt.tight_layout()

    # Save figure
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.output}")

    # Show interactively if requested
    if args.show:
        plt.show()

    plt.close()


if __name__ == "__main__":
    main()
