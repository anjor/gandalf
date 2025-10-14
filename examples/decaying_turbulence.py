#!/usr/bin/env python3
"""
Decaying Turbulence Simulation with Diagnostics

This example demonstrates:
1. Initializing turbulent spectrum with k^(-5/3) power law
2. Time evolution using GANDALF integrator
3. Tracking energy history E(t)
4. Computing and visualizing energy spectra E(k), E(k⊥), E(k∥)
5. Analyzing energy decay and selective dissipation

Physics:
    - Decaying (unforced) turbulence shows exponential energy decay
    - Magnetic energy dominates at late times (selective decay)
    - Perpendicular cascade transfers energy to small scales
    - Resistive dissipation removes energy at high k⊥

Expected results:
    - E_total decays exponentially with rate ~ η⟨k²⟩
    - E_magnetic/E_kinetic ratio increases over time
    - Spectrum develops k⊥^(-3/2) inertial range (Goldreich-Sridhar)

Runtime: ~1-2 minutes on M1 Pro for 64³ resolution
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

from krmhd import (
    SpectralGrid3D,
    initialize_random_spectrum,
    gandalf_step,
    compute_cfl_timestep,
    energy as compute_energy,
)
from krmhd.diagnostics import (
    EnergyHistory,
    energy_spectrum_1d,
    energy_spectrum_perpendicular,
    energy_spectrum_parallel,
    plot_state,
    plot_energy_history,
    plot_energy_spectrum,
)


def main():
    """Run decaying turbulence simulation with diagnostics."""

    print("=" * 70)
    print("KRMHD Decaying Turbulence Simulation")
    print("=" * 70)

    # ==========================================================================
    # Simulation Parameters
    # ==========================================================================

    # Grid resolution
    Nx, Ny, Nz = 64, 64, 32
    Lx = Ly = Lz = 2 * np.pi

    # Physics parameters
    v_A = 1.0          # Alfvén velocity
    eta = 0.01         # Resistivity
    beta_i = 1.0       # Ion plasma beta
    nu = 0.01          # Collision frequency

    # Turbulence initialization
    alpha = 5.0 / 3.0  # Spectral index (Kolmogorov)
    amplitude = 1.0    # Initial amplitude
    k_min = 1.0        # Minimum wavenumber
    k_max = 10.0       # Maximum wavenumber

    # Time integration
    n_steps = 100      # Number of timesteps
    cfl_safety = 0.3   # CFL safety factor
    save_interval = 10 # Save diagnostics every N steps

    print(f"\nGrid: {Nx} × {Ny} × {Nz}")
    print(f"Domain: [{Lx:.1f}, {Ly:.1f}, {Lz:.1f}]")
    print(f"Physics: v_A={v_A}, η={eta}, β_i={beta_i}, ν={nu}")
    print(f"Initial spectrum: k^(-{alpha:.2f}), k ∈ [{k_min}, {k_max}]")
    print(f"Evolution: {n_steps} steps with CFL={cfl_safety}")

    # ==========================================================================
    # Initialize Grid and State
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Initializing...")

    grid = SpectralGrid3D.create(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz)
    print(f"✓ Created {Nx}×{Ny}×{Nz} spectral grid")

    state = initialize_random_spectrum(
        grid,
        M=20,              # Number of Hermite moments
        alpha=alpha,
        amplitude=amplitude,
        k_min=k_min,
        k_max=k_max,
        v_th=1.0,
        beta_i=beta_i,
        nu=nu,
        Lambda=1.0,
        seed=42,
    )
    print(f"✓ Initialized random k^(-{alpha:.2f}) spectrum")

    # Compute initial energy
    initial_energies = compute_energy(state)
    print(f"  Initial energy: E_total = {initial_energies['total']:.6e}")

    # ==========================================================================
    # Time Evolution
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Time evolution...")

    history = EnergyHistory()

    for step in range(n_steps + 1):
        # Record diagnostics
        if step % save_interval == 0:
            history.append(state)
            print(f"  Step {step:3d}: t = {state.time:6.3f}, "
                  f"E_total = {history.E_total[-1]:.6e}, "
                  f"E_mag/E_kin = {history.E_magnetic[-1]/max(history.E_kinetic[-1], 1e-10):.3f}")

        # Time step
        if step < n_steps:
            dt = compute_cfl_timestep(state, v_A, cfl_safety)
            state = gandalf_step(state, dt, eta, v_A)

    print(f"✓ Completed {n_steps} timesteps")
    print(f"  Final time: t = {state.time:.3f}")
    print(f"  Energy decay: E(t)/E(0) = {history.E_total[-1]/history.E_total[0]:.3f}")

    # ==========================================================================
    # Compute Spectra
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Computing spectra...")

    k, E_k = energy_spectrum_1d(state, n_bins=32)
    print(f"✓ 1D spectrum E(k)")

    k_perp, E_perp = energy_spectrum_perpendicular(state, n_bins=32)
    print(f"✓ Perpendicular spectrum E(k⊥)")

    kz, E_parallel = energy_spectrum_parallel(state)
    print(f"✓ Parallel spectrum E(k∥)")

    # ==========================================================================
    # Visualization
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Creating visualizations...")

    # Create output directory relative to this script's location
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Plot 1: Energy history
    plot_energy_history(
        history,
        log_scale=True,
        filename=str(output_dir / "energy_history.png"),
        show=False
    )
    print(f"✓ Saved energy_history.png")

    # Plot 2: Final state
    plot_state(
        state,
        z_slice=0,
        filename=str(output_dir / "final_state.png"),
        show=False
    )
    print(f"✓ Saved final_state.png")

    # Plot 3: Energy spectra (combined figure)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1D spectrum with Kolmogorov reference
    ax = axes[0]
    valid = (k > 0) & (E_k > 0)
    ax.loglog(k[valid], E_k[valid], 'b-', linewidth=2, label='E(k)')
    # Add k^(-5/3) reference line
    k_ref = k[len(k) // 3]
    E_ref = E_k[len(E_k) // 3]
    k_line = np.logspace(np.log10(k[valid].min()), np.log10(k[valid].max()), 50)
    E_line = E_ref * (k_line / k_ref)**(-5/3)
    ax.loglog(k_line, E_line, 'k--', linewidth=1, label='k^(-5/3)')
    ax.set_xlabel('|k|', fontsize=11)
    ax.set_ylabel('E(k)', fontsize=11)
    ax.set_title('1D Spectrum', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    # Perpendicular spectrum with Goldreich-Sridhar reference
    ax = axes[1]
    valid = (k_perp > 0) & (E_perp > 0)
    ax.loglog(k_perp[valid], E_perp[valid], 'r-', linewidth=2, label='E(k⊥)')
    # Add k⊥^(-3/2) reference line
    k_ref = k_perp[len(k_perp) // 3]
    E_ref = E_perp[len(E_perp) // 3]
    k_line = np.logspace(np.log10(k_perp[valid].min()), np.log10(k_perp[valid].max()), 50)
    E_line = E_ref * (k_line / k_ref)**(-3/2)
    ax.loglog(k_line, E_line, 'k--', linewidth=1, label='k⊥^(-3/2)')
    ax.set_xlabel('k⊥', fontsize=11)
    ax.set_ylabel('E(k⊥)', fontsize=11)
    ax.set_title('Perpendicular Spectrum', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    # Parallel spectrum
    ax = axes[2]
    valid = (kz != 0) & (E_parallel > 0)
    ax.semilogy(kz[valid], E_parallel[valid], 'g-', linewidth=2, label='E(k∥)')
    ax.set_xlabel('k∥ (kz)', fontsize=11)
    ax.set_ylabel('E(k∥)', fontsize=11)
    ax.set_title('Parallel Spectrum', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "energy_spectra.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved energy_spectra.png")

    # ==========================================================================
    # Summary Statistics
    # ==========================================================================

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    # Energy decay
    E_initial = history.E_total[0]
    E_final = history.E_total[-1]
    decay_ratio = E_final / E_initial
    print(f"\nEnergy decay:")
    print(f"  E(t=0)    = {E_initial:.6e}")
    print(f"  E(t={state.time:.2f}) = {E_final:.6e}")
    print(f"  E_final / E_initial = {decay_ratio:.3f}")

    # Selective decay
    mag_frac_initial = history.E_magnetic[0] / E_initial
    mag_frac_final = history.E_magnetic[-1] / E_final
    print(f"\nSelective decay:")
    print(f"  E_mag/E_total (t=0)    = {mag_frac_initial:.3f}")
    print(f"  E_mag/E_total (t={state.time:.2f}) = {mag_frac_final:.3f}")
    print(f"  Increase = {mag_frac_final/mag_frac_initial:.2f}×")

    # Spectrum properties
    total_from_1d = np.sum(E_k) * (k[1] - k[0])
    total_from_perp = np.sum(E_perp) * (k_perp[1] - k_perp[0])
    print(f"\nSpectrum normalization:")
    print(f"  E_total (direct)     = {E_final:.6e}")
    print(f"  ∫E(k)dk             = {total_from_1d:.6e} (error: {abs(total_from_1d - E_final)/E_final:.1%})")
    print(f"  ∫E(k⊥)dk⊥           = {total_from_perp:.6e} (error: {abs(total_from_perp - E_final)/E_final:.1%})")

    print(f"\n✓ All outputs saved to {output_dir}/")
    print("\nFiles created:")
    print("  - energy_history.png : E(t) for all components")
    print("  - final_state.png    : φ and A∥ at final time")
    print("  - energy_spectra.png : E(k), E(k⊥), E(k∥)")

    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
