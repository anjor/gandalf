#!/usr/bin/env python3
"""Plot energy spectra from a KRMHD checkpoint file.

This script loads a checkpoint file and computes/plots the perpendicular energy
spectra E(k⊥) without needing to rerun the simulation. It separates kinetic and
magnetic contributions and compares them to the k⊥^(-5/3) Kolmogorov spectrum
expected in MHD turbulence.

Physical Context:
-----------------
In RMHD turbulence, the perpendicular cascade follows critical balance with
energy distributed as E(k⊥) ~ k⊥^(-5/3) in the inertial range. The spectrum
is split into:
- Kinetic energy: From perpendicular velocity (stream function φ)
- Magnetic energy: From perpendicular magnetic field (vector potential A∥)

The magnetic fraction f_mag = E_mag / E_total typically increases over time
as magnetic energy is preferentially preserved (selective decay).

Output Formats:
--------------
1. Standard style (default):
   - Left panel: Total energy spectrum with reference line
   - Right panel: Kinetic vs Magnetic spectra comparison
   - Mode number axis: n⊥ (integers, no 2π factors)

2. Thesis style (--thesis-style):
   - Left panel: Kinetic energy spectrum
   - Right panel: Magnetic energy spectrum
   - Physical wavenumber axis: k⊥ (matches published figures)
   - Clean formatting for papers/presentations

Usage Examples:
--------------
Basic usage (saves to auto-generated filename):
    python plot_checkpoint_spectrum.py checkpoint_t0300.0.h5

Custom output filename:
    python plot_checkpoint_spectrum.py --output fig_spectrum.png checkpoint.h5

Thesis-style formatting:
    python plot_checkpoint_spectrum.py --thesis-style checkpoint.h5

Interactive display:
    python plot_checkpoint_spectrum.py --show checkpoint.h5

Batch processing multiple checkpoints:
    for f in examples/output/checkpoints/checkpoint_*.h5; do
        python plot_checkpoint_spectrum.py "$f"
    done

Output:
-------
- PNG image file with log-log plots of energy spectra
- Console output with energy summary statistics
- Reference lines showing k⊥^(-5/3) or n⊥^(-5/3) power law
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from krmhd.io import load_checkpoint
from krmhd.diagnostics import (
    energy_spectrum_perpendicular_kinetic,
    energy_spectrum_perpendicular_magnetic,
)
from krmhd.physics import energy


def main():
    """Load checkpoint, compute spectra, and generate publication-quality plots.

    This function:
    1. Loads checkpoint data (state, grid, metadata) from HDF5 file
    2. Computes total energy and magnetic fraction
    3. Computes perpendicular energy spectra: E_kin(k⊥) and E_mag(k⊥)
    4. Generates plots with k⊥^(-5/3) reference lines
    5. Saves output figure to PNG file

    The perpendicular spectrum E(k⊥) is computed by binning Fourier modes by
    their perpendicular wavenumber magnitude k⊥ = sqrt(kx² + ky²), then
    summing energy contributions from all kz modes at each k⊥.

    Reference Line Normalization:
    The k⊥^(-5/3) power law reference line is normalized to match the spectrum
    at mode n=3 (index 2). This provides a visual guide for assessing whether
    the inertial range follows the expected Kolmogorov scaling.
    """
    parser = argparse.ArgumentParser(description="Plot energy spectra from checkpoint")
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint HDF5 file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output plot filename (default: auto-generated from checkpoint name)",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display plot interactively"
    )
    parser.add_argument(
        "--thesis-style",
        action="store_true",
        help="Use thesis-style formatting (k⊥ axis, clean labels)",
    )
    args = parser.parse_args()

    # ===================================================================
    # Step 1: Load checkpoint and extract state
    # ===================================================================
    print(f"Loading checkpoint: {args.checkpoint}")
    state, grid, metadata = load_checkpoint(args.checkpoint)

    # ===================================================================
    # Step 2: Compute integrated energy diagnostics
    # ===================================================================
    # The energy() function computes volume integrals of:
    # - Kinetic: E_kin = ∫ |∇⊥φ|²/2 dV (perpendicular flow energy)
    # - Magnetic: E_mag = ∫ |∇⊥A∥|²/2 dV (perpendicular field energy)
    # - Compressive: E_comp = ∫ |δB∥|²/2 dV (parallel field perturbations)
    energy_dict = energy(state)
    E_mag = energy_dict["magnetic"]
    E_kin = energy_dict["kinetic"]
    E_comp = energy_dict["compressive"]
    E_total = energy_dict["total"]
    f_mag = E_mag / E_total if E_total > 0 else 0.0

    print(f"  Time:       t = {state.time:.2f} τ_A")
    print(f"  Grid:       {grid.Nx}×{grid.Ny}×{grid.Nz}")
    print(f"  Total energy: {E_total:.4e}")
    print(f"  Magnetic fraction: {f_mag:.3f}")

    # ===================================================================
    # Step 3: Compute perpendicular energy spectra E(k⊥)
    # ===================================================================
    # These functions bin Fourier modes by k⊥ = sqrt(kx² + ky²) and sum
    # energy contributions from all kz modes at each k⊥. This gives the
    # perpendicular cascade spectrum, which should follow k⊥^(-5/3) in
    # the inertial range for RMHD turbulence.
    print("Computing energy spectra...")
    k_perp, E_kin_spec = energy_spectrum_perpendicular_kinetic(state)
    _, E_mag_spec = energy_spectrum_perpendicular_magnetic(state)
    E_total_spec = E_kin_spec + E_mag_spec

    # ===================================================================
    # Step 4: Convert to mode numbers for user-friendly axis
    # ===================================================================
    # Mode numbers (n=1, 2, 3, ...) are more intuitive than wavenumbers
    # with 2π factors. For a domain of size L, mode n has k = 2πn/L.
    # We use L_min to ensure n corresponds to the fundamental mode.
    L_min = min(grid.Lx, grid.Ly)
    n_perp = k_perp * L_min / (2 * np.pi)

    # ===================================================================
    # Step 5: Generate output filename
    # ===================================================================
    if args.output is None:
        # Auto-generate filename from checkpoint time and style
        style_suffix = "_thesis_style" if args.thesis_style else ""
        output_name = f"spectrum_checkpoint{style_suffix}_t{state.time:.0f}.png"
        args.output = args.checkpoint.parent / output_name

    # ===================================================================
    # Step 6: Choose axis variable and labels based on style
    # ===================================================================
    # Thesis style: Use physical wavenumbers k⊥ (matches publications)
    # Standard style: Use integer mode numbers n⊥ (easier to interpret)
    if args.thesis_style:
        x_axis = k_perp
        x_label = "k⊥"
        x_ref_label = "k⊥^(-5/3)"
    else:
        x_axis = n_perp
        x_label = "Mode number n⊥"
        x_ref_label = "n⊥^(-5/3)"

    # ===================================================================
    # Step 7: Create two-panel figure
    # ===================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reference line normalization point (mode n=3, which is index 2)
    # We normalize the k^(-5/3) power law to match the spectrum at this
    # point, chosen to be in the inertial range for typical resolutions.
    idx_ref = 2  # n=3 or k⊥=3 (0-indexed)

    # ===================================================================
    # Step 8: Plot spectra with reference lines
    # ===================================================================
    # The k⊥^(-5/3) reference line is the expected Kolmogorov-like spectrum
    # for MHD turbulence in the inertial range. Deviations indicate:
    # - Steeper slope (< -5/3): Over-dissipation or insufficient forcing
    # - Shallower slope (> -5/3): Under-resolved, energy piling up
    # - Good match: Proper balance of forcing, cascade, and dissipation

    if args.thesis_style:
        # ===============================================================
        # Thesis Style: Side-by-side kinetic and magnetic spectra
        # ===============================================================
        # Left panel: Kinetic
        ax1.loglog(x_axis, E_kin_spec, "-", linewidth=3, color='#1f77b4', label=f"{grid.Nx}³, r=2")
        if len(E_kin_spec) > idx_ref:
            ref_line = (x_axis ** (-5 / 3)) * (E_kin_spec[idx_ref] / (x_axis[idx_ref] ** (-5 / 3)))
            ax1.loglog(x_axis, ref_line, "k--", label=x_ref_label, alpha=0.7, linewidth=1.5)
        ax1.set_xlabel(x_label, fontsize=14)
        ax1.set_ylabel("Kinetic Energy", fontsize=14)
        ax1.set_title("Kinetic Energy Spectrum", fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12, frameon=False)
        ax1.grid(alpha=0.3, which="both")

        # Right panel: Magnetic
        ax2.loglog(x_axis, E_mag_spec, "-", linewidth=3, color='#ff7f0e', label=f"{grid.Nx}³, r=2")
        if len(E_mag_spec) > idx_ref:
            ref_line = (x_axis ** (-5 / 3)) * (E_mag_spec[idx_ref] / (x_axis[idx_ref] ** (-5 / 3)))
            ax2.loglog(x_axis, ref_line, "k--", label=x_ref_label, alpha=0.7, linewidth=1.5)
        ax2.set_xlabel(x_label, fontsize=14)
        ax2.set_ylabel("Magnetic Energy", fontsize=14)
        ax2.set_title("Magnetic Energy Spectrum", fontsize=16, fontweight='bold')
        ax2.legend(fontsize=12, frameon=False)
        ax2.grid(alpha=0.3, which="both")
    else:
        # ===============================================================
        # Standard Style: Total spectrum + kinetic/magnetic comparison
        # ===============================================================
        # Left panel: Total spectrum
        ax1.loglog(x_axis, E_total_spec, "o-", label="Total", linewidth=2, markersize=5)
        if len(E_total_spec) > idx_ref:
            ref_line = (x_axis ** (-5 / 3)) * (E_total_spec[idx_ref] / (x_axis[idx_ref] ** (-5 / 3)))
            ax1.loglog(x_axis, ref_line, "k--", label=x_ref_label, alpha=0.5, linewidth=1.5)
        ax1.set_xlabel(x_label, fontsize=12)
        ax1.set_ylabel("E(n⊥)", fontsize=12)
        ax1.set_title(f"Total Energy Spectrum (t={state.time:.1f} τ_A)", fontsize=13)
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3, which="both")

        # Right panel: Kinetic vs Magnetic
        ax2.loglog(x_axis, E_kin_spec, "o-", label="Kinetic", linewidth=2, markersize=5, alpha=0.8)
        ax2.loglog(x_axis, E_mag_spec, "s-", label="Magnetic", linewidth=2, markersize=5, alpha=0.8)
        if len(E_kin_spec) > idx_ref:
            ref_line_kin = (x_axis ** (-5 / 3)) * (E_kin_spec[idx_ref] / (x_axis[idx_ref] ** (-5 / 3)))
            ax2.loglog(x_axis, ref_line_kin, "k--", label=x_ref_label, alpha=0.5, linewidth=1.5)
        ax2.set_xlabel(x_label, fontsize=12)
        ax2.set_ylabel("E(n⊥)", fontsize=12)
        ax2.set_title(f"Kinetic vs Magnetic (f_mag={f_mag:.3f})", fontsize=13)
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3, which="both")

        # Overall title for standard style only
        fig.suptitle(f"Energy Spectra from Checkpoint ({grid.Nx}³ grid)", fontsize=14, y=0.98)

    # ===================================================================
    # Step 9: Finalize layout and save figure
    # ===================================================================
    # Adjust spacing to prevent label overlap
    if args.thesis_style:
        plt.tight_layout()
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save high-resolution PNG (150 dpi suitable for papers/presentations)
    print(f"Saving plot: {args.output}")
    plt.savefig(args.output, dpi=150, bbox_inches="tight")

    # Optionally display interactively (useful for quick inspection)
    if args.show:
        plt.show()

    print("Done!")
    print("\nInterpretation Guide:")
    print("  - Good k⊥^(-5/3) match in inertial range (n ~ 3-10): Healthy turbulence")
    print("  - High magnetic fraction (f_mag > 0.5): Selective decay underway")
    print("  - Exponential cutoff at high-k: Hyper-dissipation working correctly")
    print("  - Flat or rising spectrum at high-k: Under-dissipated, increase η")


if __name__ == "__main__":
    main()
