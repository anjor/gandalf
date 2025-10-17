#!/usr/bin/env python3
"""
Hyper-dissipation Demonstration

This example demonstrates the use of hyper-dissipation operators to selectively
damp small-scale modes while preserving large-scale dynamics in turbulent simulations.

Physics:
    Standard dissipation (r=1):   -η∇²⊥ ~ -ηk⊥²
    Hyper-dissipation (r=2):      -η∇⁴⊥ ~ -ηk⊥⁴

    For r>1, dissipation is concentrated at high-k, creating a steep exponential
    cutoff in the spectrum while minimally affecting the inertial range.

Key concepts:
    - Hyper-resistivity: -ηk⊥^(2r) for Alfvén modes (z±)
    - Hyper-collisions: -νm^(2n) for Hermite moments (gₘ)
    - Parameter selection: r=2, n=2 recommended for typical grids
    - Overflow safety: Validation prevents exp() underflow

Expected results:
    - r=1 (standard): Gradual spectral decay across all scales
    - r=2 (hyper): Sharp exponential cutoff at high-k, flat inertial range
    - Energy decay rate: Faster for r=1 (dissipates all scales)
    - Large-scale preservation: Better for r=2

Runtime: ~20-30 seconds on M1 Pro
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

from krmhd import (
    SpectralGrid3D,
    initialize_random_spectrum,
    gandalf_step,
    energy as compute_energy,
)
from krmhd.diagnostics import (
    energy_spectrum_perpendicular,
    plot_energy_spectrum,
)


def run_simulation(state, dt, eta, n_steps, hyper_r=1, label=""):
    """Run decaying turbulence with specified dissipation order.

    Parameters
    ----------
    state : KRMHDState
        Initial state
    dt : float
        Timestep
    eta : float
        Dissipation coefficient (resistivity)
    n_steps : int
        Number of timesteps
    hyper_r : int, default=1
        Hyper-dissipation order (1=standard, 2=hyper-viscosity)
    label : str
        Label for progress printing

    Returns
    -------
    state : KRMHDState
        Final state
    energy_history : list
        Energy at each timestep
    """
    energy_history = []

    print(f"\n{label}")
    print(f"  Parameters: eta={eta}, r={hyper_r}, dt={dt}")

    E0 = compute_energy(state)['total']
    energy_history.append(E0)
    print(f"  Initial energy: {E0:.4e}")

    # Time evolution
    for i in range(n_steps):
        state = gandalf_step(state, dt=dt, eta=eta, v_A=1.0, hyper_r=hyper_r)
        E = compute_energy(state)['total']
        energy_history.append(E)

        if (i + 1) % 10 == 0:
            decay_fraction = (E0 - E) / E0 * 100
            print(f"  Step {i+1:3d}: E = {E:.4e} ({decay_fraction:5.1f}% decay)")

    E_final = compute_energy(state)['total']
    total_decay = (E0 - E_final) / E0 * 100
    print(f"  Final energy: {E_final:.4e} ({total_decay:.1f}% total decay)")

    return state, energy_history


def main():
    """Compare standard vs hyper-dissipation in decaying turbulence."""

    print("=" * 70)
    print("Hyper-dissipation Demonstration")
    print("=" * 70)

    # ==========================================================================
    # Simulation Parameters
    # ==========================================================================

    # Grid resolution (modest for fast demo)
    Nx, Ny, Nz = 64, 64, 32
    Lx = Ly = Lz = 2 * np.pi

    # Physics parameters
    v_A = 1.0          # Alfvén velocity
    beta_i = 1.0       # Ion plasma beta
    nu = 0.01          # Collision frequency

    # Turbulence initialization
    alpha = 5.0 / 3.0  # Kolmogorov spectrum k^(-5/3)
    amplitude = 1.0    # Initial amplitude
    k_min = 1.0        # Large scales
    k_max = 10.0       # Inertial range cutoff

    # Time integration
    n_steps = 50       # Number of timesteps
    dt = 0.01          # Timestep (small for stability)

    # Dissipation comparison
    # For r=1: Use larger η to see clear dissipation effect
    # For r=2: Use smaller η (k⊥⁴ is much stronger than k⊥²)
    eta_standard = 0.05   # Standard resistivity (r=1)
    eta_hyper = 0.0001    # Hyper-resistivity (r=2)

    print("\nSimulation setup:")
    print(f"  Grid: {Nx}×{Ny}×{Nz}")
    print(f"  Domain: ({Lx:.2f}, {Ly:.2f}, {Lz:.2f})")
    print(f"  Timesteps: {n_steps}, dt={dt}")
    print(f"  Initial spectrum: k^(-{alpha:.2f}), k ∈ [{k_min}, {k_max}]")

    # ==========================================================================
    # Initialize Grid and State
    # ==========================================================================

    print("\nInitializing turbulent spectrum...")
    grid = SpectralGrid3D.create(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz)
    state_init = initialize_random_spectrum(
        grid,
        M=10,
        alpha=alpha,
        amplitude=amplitude,
        k_min=k_min,
        k_max=k_max,
        v_th=1.0,
        beta_i=beta_i,
        seed=42
    )

    E_init = compute_energy(state_init)
    print(f"  E_magnetic = {E_init['magnetic']:.4e}")
    print(f"  E_kinetic  = {E_init['kinetic']:.4e}")
    print(f"  E_total    = {E_init['total']:.4e}")

    # Compute initial spectrum for reference
    k_perp_init, E_k_perp_init = energy_spectrum_perpendicular(state_init)

    # ==========================================================================
    # Run Comparison: Standard (r=1) vs Hyper-dissipation (r=2)
    # ==========================================================================

    print("\n" + "=" * 70)
    print("Running simulations...")
    print("=" * 70)

    # Standard dissipation (r=1)
    state_standard, energy_standard = run_simulation(
        state_init,
        dt=dt,
        eta=eta_standard,
        n_steps=n_steps,
        hyper_r=1,
        label="[1] Standard Dissipation (r=1)"
    )

    # Hyper-dissipation (r=2)
    state_hyper, energy_hyper = run_simulation(
        state_init,
        dt=dt,
        eta=eta_hyper,
        n_steps=n_steps,
        hyper_r=2,
        label="[2] Hyper-dissipation (r=2)"
    )

    # ==========================================================================
    # Compute Final Spectra
    # ==========================================================================

    print("\nComputing final spectra...")
    k_perp_std, E_k_perp_std = energy_spectrum_perpendicular(state_standard)
    k_perp_hyp, E_k_perp_hyp = energy_spectrum_perpendicular(state_hyper)

    # ==========================================================================
    # Visualization
    # ==========================================================================

    print("\nGenerating plots...")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left panel: Energy evolution ---
    ax = axes[0]
    times = np.arange(len(energy_standard)) * dt

    ax.semilogy(times, energy_standard, 'b-', linewidth=2,
                label=f'Standard (r=1, η={eta_standard})')
    ax.semilogy(times, energy_hyper, 'r-', linewidth=2,
                label=f'Hyper (r=2, η={eta_hyper})')

    ax.set_xlabel('Time [v_A⁻¹]', fontsize=12)
    ax.set_ylabel('Total Energy', fontsize=12)
    ax.set_title('Energy Decay Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Right panel: Spectral comparison ---
    ax = axes[1]

    # Plot initial spectrum
    mask_init = k_perp_init > 0
    ax.loglog(k_perp_init[mask_init], E_k_perp_init[mask_init],
              'k--', linewidth=1.5, alpha=0.5, label='Initial')

    # Plot final spectra
    mask_std = k_perp_std > 0
    ax.loglog(k_perp_std[mask_std], E_k_perp_std[mask_std],
              'b-', linewidth=2, label='Standard (r=1)')

    mask_hyp = k_perp_hyp > 0
    ax.loglog(k_perp_hyp[mask_hyp], E_k_perp_hyp[mask_hyp],
              'r-', linewidth=2, label='Hyper (r=2)')

    # Add reference slopes
    k_ref = np.array([2.0, 8.0])
    E_ref = 1e-3 * (k_ref / k_ref[0]) ** (-5.0/3.0)
    ax.loglog(k_ref, E_ref, 'k:', linewidth=1, alpha=0.5, label='k⁻⁵ᐟ³')

    ax.set_xlabel('k⊥', fontsize=12)
    ax.set_ylabel('E(k⊥)', fontsize=12)
    ax.set_title('Perpendicular Energy Spectrum', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, Nx//2)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / "hyper_dissipation_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    E0 = compute_energy(state_init)['total']
    E_std = compute_energy(state_standard)['total']
    E_hyp = compute_energy(state_hyper)['total']

    decay_std = (E0 - E_std) / E0 * 100
    decay_hyp = (E0 - E_hyp) / E0 * 100

    print(f"\nEnergy decay:")
    print(f"  Standard (r=1):      {decay_std:.1f}%")
    print(f"  Hyper (r=2):         {decay_hyp:.1f}%")
    print(f"  Difference:          {decay_std - decay_hyp:.1f}%")

    print(f"\nKey observations:")
    print(f"  - Standard dissipation removes energy from all scales")
    print(f"  - Hyper-dissipation concentrates removal at high-k")
    print(f"  - Large scales (k⊥ ~ 2-5) better preserved with r=2")
    print(f"  - Steep exponential cutoff at k⊥ > 10 for r=2")

    print(f"\nParameter selection guidelines:")
    print(f"  - Use r=2, n=2 for typical turbulence studies")
    print(f"  - Choose η, ν to satisfy overflow validation (rate·dt < 50)")
    print(f"  - For k_max ~ 30: η < 1e-5 for r=2, η < 0.01 for r=1")
    print(f"  - Hyper-dissipation requires smaller coefficients due to k^(2r) scaling")

    print(f"\nOutput saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
