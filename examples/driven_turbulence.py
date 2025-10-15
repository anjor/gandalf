#!/usr/bin/env python3
"""
Driven Turbulence Simulation with Forcing

This example demonstrates:
1. Gaussian white noise forcing at large scales (k ~ 2-5)
2. Energy injection and steady-state energy balance
3. Perpendicular cascade transferring energy to small scales
4. Energy spectra showing inertial range scaling

Physics:
    - Driven (forced) turbulence reaches steady state when ε_inj = ε_diss
    - Forcing drives perpendicular velocity (u⊥) only, not magnetic field
    - Perpendicular cascade develops k⊥^(-3/2) or k^(-5/3) spectrum
    - Steady-state energy fluctuates around mean value

Expected results:
    - Energy oscillates around steady-state value (ε_inj ≈ ε_diss)
    - Magnetic and kinetic energy reach equipartition
    - Inertial range spectrum between forcing and dissipation scales
    - Energy injection rate measurable and positive

Runtime: ~20s on M1 Pro for 64×64×32 resolution @ 200 steps
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

from krmhd import (
    SpectralGrid3D,
    initialize_random_spectrum,
    gandalf_step,
    compute_cfl_timestep,
    energy as compute_energy,
    force_alfven_modes,
    compute_energy_injection_rate,
)
from krmhd.diagnostics import (
    EnergyHistory,
    energy_spectrum_1d,
    energy_spectrum_perpendicular,
    energy_spectrum_parallel,
    plot_energy_history,
    plot_energy_spectrum,
)


def main():
    """Run driven turbulence simulation with forcing."""

    print("=" * 70)
    print("KRMHD Driven Turbulence Simulation with Forcing")
    print("=" * 70)

    # ==========================================================================
    # Simulation Parameters
    # ==========================================================================

    # Grid resolution
    Nx, Ny, Nz = 64, 64, 32
    Lx = Ly = Lz = 2 * np.pi

    # Physics parameters
    v_A = 1.0          # Alfvén velocity
    eta = 0.02         # Resistivity
    beta_i = 1.0       # Ion plasma beta
    nu = 0.02          # Collision frequency

    # Forcing parameters
    force_amplitude = 0.3   # Forcing strength (ε_inj ~ amplitude²)
    k_force_min = 2.0       # Minimum forcing wavenumber
    k_force_max = 5.0       # Maximum forcing wavenumber

    # Initial turbulence (weak)
    alpha = 5.0 / 3.0       # Spectral index
    amplitude = 0.1         # Initial amplitude (weak to see forcing effect)
    k_min = 1.0
    k_max = 10.0

    # Time integration
    n_steps = 200           # Number of timesteps
    cfl_safety = 0.3        # CFL safety factor
    save_interval = 5       # Save diagnostics every N steps

    print(f"\nGrid: {Nx} × {Ny} × {Nz}")
    print(f"Domain: [{Lx:.1f}, {Ly:.1f}, {Lz:.1f}]")
    print(f"Physics: v_A={v_A}, η={eta}, β_i={beta_i}, ν={nu}")
    print(f"Forcing: amplitude={force_amplitude}, k ∈ [{k_force_min}, {k_force_max}]")
    print(f"Evolution: {n_steps} steps with CFL={cfl_safety}")

    # ==========================================================================
    # Initialize Grid and State
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Initializing...")

    start_time = time.time()

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
    print(f"✓ Initialized weak k^(-{alpha:.2f}) spectrum")

    # Initialize JAX random key for forcing
    key = jax.random.PRNGKey(42)

    # Compute initial energy
    initial_energies = compute_energy(state)
    print(f"  Initial energy: E_total = {initial_energies['total']:.6e}")

    # ==========================================================================
    # Time Evolution with Forcing
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Running forced evolution...")

    # Initialize energy history
    history = EnergyHistory()
    history.append(state)

    # Compute CFL-limited timestep
    dt = compute_cfl_timestep(state, v_A=v_A, cfl_safety=cfl_safety)
    print(f"  Using dt = {dt:.4f} (CFL-limited)")

    # Track energy injection
    total_injection = 0.0

    # Main timestepping loop
    for i in range(n_steps):
        # Apply forcing (add energy at large scales)
        state_before_forcing = state
        state, key = force_alfven_modes(
            state,
            amplitude=force_amplitude,
            k_min=k_force_min,
            k_max=k_force_max,
            dt=dt,
            key=key
        )

        # Compute energy injection rate
        eps_inj = compute_energy_injection_rate(state_before_forcing, state, dt)
        total_injection += eps_inj * dt

        # Evolve dynamics (cascade + dissipation)
        state = gandalf_step(state, dt=dt, eta=eta, v_A=v_A)

        # Save diagnostics
        if (i + 1) % save_interval == 0:
            history.append(state)
            E = compute_energy(state)["total"]
            print(f"  Step {i+1:3d}/{n_steps}: t={state.time:.2f}, E={E:.4e}, ε_inj={eps_inj:.3e}")

    elapsed_time = time.time() - start_time

    print(f"\n✓ Completed {n_steps} timesteps")
    print(f"  Runtime: {elapsed_time:.1f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"  (Estimated ~20s on M1 Pro for 64×64×32 @ 200 steps)")

    # ==========================================================================
    # Final Diagnostics
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Computing final diagnostics...")

    final_energies = compute_energy(state)
    print(f"  Final energy: E_total = {final_energies['total']:.6e}")
    print(f"  Energy change: ΔE = {final_energies['total'] - initial_energies['total']:.6e}")
    print(f"  Total injection: ∫ε_inj dt = {total_injection:.6e}")

    # Magnetic vs kinetic energy
    E_mag = final_energies["magnetic"]
    E_kin = final_energies["kinetic"]
    print(f"  Magnetic fraction: {E_mag/(E_mag+E_kin):.3f}")

    # Estimate dissipation rate
    # For driven turbulence: ε_diss ≈ ε_inj in steady state
    avg_injection_rate = total_injection / state.time
    print(f"  Average injection rate: ⟨ε_inj⟩ = {avg_injection_rate:.3e}")

    # Compute energy spectra
    spec_1d = energy_spectrum_1d(state)
    spec_perp = energy_spectrum_perpendicular(state)
    spec_parallel = energy_spectrum_parallel(state)

    # ==========================================================================
    # Visualization
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Creating visualizations...")

    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Plot 1: Energy history
    fig1, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Total energy vs time
    times = jnp.array(history.times)
    energies = jnp.array(history.E_total)

    axes[0].plot(times, energies, 'b-', linewidth=2, label="Total energy")
    axes[0].axhline(energies[-1], color='r', linestyle='--', alpha=0.5, label="Final value")
    axes[0].set_ylabel("Energy")
    axes[0].set_title("Driven Turbulence: Energy Evolution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Magnetic fraction vs time
    mag_frac = jnp.array(history.E_magnetic) / (jnp.array(history.E_magnetic) + jnp.array(history.E_kinetic))
    axes[1].plot(times, mag_frac, 'g-', linewidth=2)
    axes[1].axhline(0.5, color='k', linestyle='--', alpha=0.5, label="Equipartition")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Magnetic Fraction")
    axes[1].set_title("Energy Partition")
    axes[1].set_ylim([0, 1])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "driven_energy_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved driven_energy_history.png")

    # Plot 2: Energy spectra (combined figure)
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1D spherically-averaged spectrum
    k_bins, E_k = spec_1d
    axes[0].loglog(k_bins, E_k, 'o-', markersize=4, linewidth=2)

    # Reference slopes for inertial range
    k_ref = k_bins[(k_bins > 5) & (k_bins < 15)]
    if len(k_ref) > 0:
        k_ref_mid = k_ref[len(k_ref)//2]
        E_ref = E_k[(k_bins > 5) & (k_bins < 15)][len(k_ref)//2]
        axes[0].loglog(k_ref, E_ref * (k_ref/k_ref_mid)**(-5/3),
                       'k--', alpha=0.5, linewidth=1, label='k⁻⁵ᐟ³')

    # Mark forcing band
    axes[0].axvspan(k_force_min, k_force_max, alpha=0.2, color='green', label='Forcing band')

    axes[0].set_xlabel("Wavenumber |k|")
    axes[0].set_ylabel("E(|k|)")
    axes[0].set_title("1D Spectrum")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which="both")

    # Perpendicular spectrum
    k_perp_bins, E_k_perp = spec_perp
    axes[1].loglog(k_perp_bins, E_k_perp, 'o-', markersize=4, linewidth=2, color='orange')
    axes[1].axvspan(k_force_min, k_force_max, alpha=0.2, color='green', label='Forcing band')
    axes[1].set_xlabel("k⊥")
    axes[1].set_ylabel("E(k⊥)")
    axes[1].set_title("Perpendicular Spectrum")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which="both")

    # Parallel spectrum
    k_par_bins, E_k_par = spec_parallel
    axes[2].semilogy(k_par_bins, E_k_par, 'o-', markersize=6, linewidth=2, color='purple')
    axes[2].axvspan(k_force_min, k_force_max, alpha=0.2, color='green', label='Forcing band')
    axes[2].set_xlabel("k∥")
    axes[2].set_ylabel("E(k∥)")
    axes[2].set_title("Parallel Spectrum")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "driven_energy_spectra.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved driven_energy_spectra.png")

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print(f"\nKey Results:")
    print(f"  • Initial energy: {initial_energies['total']:.4e}")
    print(f"  • Final energy: {final_energies['total']:.4e}")
    print(f"  • Energy change: {final_energies['total']/initial_energies['total']:.2f}x")
    print(f"  • Average injection: ⟨ε_inj⟩ = {avg_injection_rate:.3e}")
    print(f"  • Magnetic fraction: {E_mag/(E_mag+E_kin):.3f}")
    print(f"  • Runtime: {elapsed_time:.1f}s ({elapsed_time/60:.2f} min)")
    print(f"\nOutput saved to: {output_dir}/")
    print(f"  - driven_energy_history.png (energy evolution)")
    print(f"  - driven_energy_spectra.png (E(k), E(k⊥), E(k∥))")
    print("")


if __name__ == "__main__":
    main()
