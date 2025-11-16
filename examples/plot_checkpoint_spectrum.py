#!/usr/bin/env python3
"""Plot energy spectra from a KRMHD checkpoint file.

This script loads a checkpoint file and computes/plots the energy spectra
without needing to rerun the simulation.

Usage:
    python plot_checkpoint_spectrum.py checkpoint_final_t0200.0.h5
    python plot_checkpoint_spectrum.py --output custom_name.png checkpoint.h5
    python plot_checkpoint_spectrum.py --thesis-style checkpoint.h5
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

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    state, grid, metadata = load_checkpoint(args.checkpoint)

    # Compute total energy
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

    # Compute spectra
    print("Computing energy spectra...")
    k_perp, E_kin_spec = energy_spectrum_perpendicular_kinetic(state)
    _, E_mag_spec = energy_spectrum_perpendicular_magnetic(state)
    E_total_spec = E_kin_spec + E_mag_spec

    # Mode numbers (more intuitive than k with 2π factors)
    # k_perp = 2π n / L_min, so n = k_perp L_min / 2π
    L_min = min(grid.Lx, grid.Ly)
    n_perp = k_perp * L_min / (2 * np.pi)

    # Generate output filename
    if args.output is None:
        # Extract time from checkpoint name or use state.time
        style_suffix = "_thesis_style" if args.thesis_style else ""
        output_name = f"spectrum_checkpoint{style_suffix}_t{state.time:.0f}.png"
        args.output = args.checkpoint.parent / output_name

    # Determine axis variable (k_perp for thesis style, n_perp for standard)
    if args.thesis_style:
        x_axis = k_perp
        x_label = "k⊥"
        x_ref_label = "k⊥^(-5/3)"
    else:
        x_axis = n_perp
        x_label = "Mode number n⊥"
        x_ref_label = "n⊥^(-5/3)"

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reference line normalization point
    idx_ref = 2  # n=3 or k⊥=3 (0-indexed)

    if args.thesis_style:
        # Thesis style: Kinetic on left, Magnetic on right, clean titles
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
        # Standard style: Total on left, Kinetic vs Magnetic on right
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

    # Adjust layout
    if args.thesis_style:
        plt.tight_layout()
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    print(f"Saving plot: {args.output}")
    plt.savefig(args.output, dpi=150, bbox_inches="tight")

    if args.show:
        plt.show()

    print("Done!")


if __name__ == "__main__":
    main()
