#!/usr/bin/env python3
"""
High-Reynolds Number Driven Turbulence
Optimized to show k^(-5/3) inertial range

Key parameters:
- 128³ resolution for scale separation
- Hyper-dissipation (r=2) with small η = 1e-4
- Narrow forcing band k ∈ [2, 4]
- Longer integration (500 steps) for cascade development

Expected runtime: ~2-5 minutes on M1 Pro (CPU or GPU)
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
    """Run high-Re driven turbulence to observe inertial range."""

    print("=" * 70)
    print("High-Reynolds Number Driven Turbulence (k^(-5/3) Inertial Range)")
    print("=" * 70)

    # ==========================================================================
    # OPTIMIZED PARAMETERS FOR INERTIAL RANGE
    # ==========================================================================

    # Higher resolution for scale separation
    Nx, Ny, Nz = 128, 128, 64
    Lx = Ly = Lz = 2 * np.pi

    # Physics parameters
    v_A = 1.0          # Alfvén velocity
    eta = 1e-4         # Very small resistivity (hyper-dissipation!)
    beta_i = 1.0       # Ion plasma beta
    nu = 1e-4          # Very small collision frequency
    hyper_r = 2        # Hyper-resistivity order (k⊥⁴ damping)
    hyper_n = 2        # Hyper-collision order

    # Narrow forcing band for clearer injection scale
    force_amplitude = 0.5   # Stronger forcing
    k_force_min = 2.0
    k_force_max = 4.0

    # Weak initial condition (let forcing drive the turbulence)
    alpha = 5.0 / 3.0
    amplitude = 0.05        # Very weak initial spectrum
    k_min = 1.0
    k_max = 15.0

    # Longer integration for cascade development
    n_steps = 500
    cfl_safety = 0.3
    save_interval = 10

    print(f"\nGrid: {Nx} × {Ny} × {Nz}")
    print(f"Domain: [{Lx:.1f}, {Ly:.1f}, {Lz:.1f}]")
    print(f"Physics: v_A={v_A}, η={eta:.1e}, ν={nu:.1e}")
    print(f"Hyper-dissipation: r={hyper_r}, n={hyper_n}")
    print(f"Forcing: amplitude={force_amplitude}, k ∈ [{k_force_min}, {k_force_max}]")
    print(f"Evolution: {n_steps} steps with CFL={cfl_safety}")
    print(f"\nEstimated runtime: ~2-5 minutes")

    # ==========================================================================
    # Initialize Grid and State
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Initializing...")

    grid = SpectralGrid3D.create(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz)
    print(f"✓ Created {Nx}×{Ny}×{Nz} spectral grid")

    # Weak initial spectrum
    M = 10  # Number of Hermite moments
    state = initialize_random_spectrum(
        grid,
        M=M,
        alpha=alpha,
        amplitude=amplitude,
        k_min=k_min,
        k_max=k_max,
        v_th=1.0,
        beta_i=beta_i,
        seed=42
    )

    E0 = compute_energy(state)['total']
    print(f"✓ Initialized weak k^(-{alpha:.2f}) spectrum")
    print(f"  Initial energy: E_total = {E0:.6e}")

    # ==========================================================================
    # Compute Timestep
    # ==========================================================================

    dt = compute_cfl_timestep(
        state=state,
        v_A=v_A,
        cfl_safety=cfl_safety
    )

    print(f"\n  Using dt = {dt:.4f} (CFL-limited)")

    # ==========================================================================
    # Time Evolution with Forcing
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Running forced evolution...")

    # Energy history tracking
    history = EnergyHistory()

    # Store energy injection rates
    injection_rates = []

    # Random key for forcing
    key = jax.random.PRNGKey(42)

    start_time = time.time()

    for step in range(n_steps + 1):
        # Compute current energy
        E_dict = compute_energy(state)
        E_total = E_dict['total']
        E_mag = E_dict['magnetic']

        # Track history
        if step % save_interval == 0:
            history.append(state)

            # Print progress
            if step % 50 == 0:
                mag_frac = E_mag / E_total if E_total > 0 else 0
                avg_inj = np.mean(injection_rates[-50:]) if len(injection_rates) >= 50 else 0
                print(f"  Step {step:3d}/{n_steps}: t={state.time:.2f}, "
                      f"E={E_total:.4e}, f_mag={mag_frac:.3f}, "
                      f"⟨ε_inj⟩={avg_inj:.2e}")

        if step == n_steps:
            break

        # Apply forcing
        state_before = state
        key, subkey = jax.random.split(key)
        state, key = force_alfven_modes(
            state,
            amplitude=force_amplitude,
            k_min=k_force_min,
            k_max=k_force_max,
            dt=dt,
            key=subkey
        )

        # Compute energy injection rate
        eps_inj = compute_energy_injection_rate(state_before, state, dt)
        injection_rates.append(float(eps_inj))

        # Time step with hyper-dissipation
        state = gandalf_step(
            state,
            dt=dt,
            eta=eta,
            nu=nu,
            v_A=v_A,
            hyper_r=hyper_r,
            hyper_n=hyper_n
        )

    elapsed = time.time() - start_time

    print(f"✓ Completed {n_steps} timesteps")
    print(f"  Runtime: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")

    # ==========================================================================
    # Final Diagnostics
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Computing final diagnostics...")

    E_final_dict = compute_energy(state)
    E_final = E_final_dict['total']
    E_mag_final = E_final_dict['magnetic']

    total_injection = np.trapz(injection_rates, dx=dt)
    avg_injection = np.mean(injection_rates[n_steps//2:])  # Average over second half

    print(f"  Final energy: E_total = {E_final:.6e}")
    print(f"  Energy change: ΔE = {E_final - E0:.6e}")
    print(f"  Total injection: ∫ε_inj dt = {total_injection:.6e}")
    print(f"  Magnetic fraction: {E_mag_final / E_final:.3f}")
    print(f"  Average injection rate (t > {n_steps//2*dt:.1f}): ⟨ε_inj⟩ = {avg_injection:.3e}")

    # Compute spectra
    k_bins, E_k = energy_spectrum_1d(state)
    kperp_bins, E_kperp = energy_spectrum_perpendicular(state)
    kpar_bins, E_kpar = energy_spectrum_parallel(state)

    # ==========================================================================
    # Visualization
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Creating visualizations...")

    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Energy history plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Energy evolution
    ax = axes[0]
    times = np.array(history.times)
    energies = np.array(history.E_total)
    ax.plot(times, energies, 'k-', linewidth=2)
    ax.set_xlabel('Time [v_A⁻¹]', fontsize=12)
    ax.set_ylabel('Total Energy', fontsize=12)
    ax.set_title('Energy Evolution with Forcing', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Energy injection rate
    ax = axes[1]
    time_inj = np.linspace(0, n_steps * dt, len(injection_rates))
    ax.plot(time_inj, injection_rates, 'b-', alpha=0.5, linewidth=1, label='ε_inj(t)')
    # Moving average
    window = 50
    if len(injection_rates) >= window:
        moving_avg = np.convolve(injection_rates, np.ones(window)/window, mode='valid')
        time_avg = time_inj[window-1:]
        ax.plot(time_avg, moving_avg, 'r-', linewidth=2, label=f'Moving avg ({window} steps)')
    ax.axhline(avg_injection, color='green', linestyle='--', linewidth=2,
               label=f'⟨ε_inj⟩ = {avg_injection:.2e}')
    ax.set_xlabel('Time [v_A⁻¹]', fontsize=12)
    ax.set_ylabel('Energy Injection Rate', fontsize=12)
    ax.set_title('Energy Injection Rate', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "high_re_energy_history.png", dpi=150, bbox_inches='tight')
    print("✓ Saved high_re_energy_history.png")
    plt.close()

    # Energy spectra plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1D spectrum
    ax = axes[0]
    ax.loglog(k_bins, E_k, 'o-', linewidth=2, markersize=4, label='E(k)')
    # Reference k^(-5/3)
    k_ref = k_bins[(k_bins > 5) & (k_bins < 20)]
    if len(k_ref) > 0:
        E_ref = 0.1 * k_ref**(-5/3)
        ax.loglog(k_ref, E_ref, 'k--', linewidth=2, alpha=0.7, label='k^(-5/3)')
    # Forcing band
    ax.axvspan(k_force_min, k_force_max, alpha=0.2, color='green', label='Forcing band')
    ax.set_xlabel('Wavenumber |k|', fontsize=12)
    ax.set_ylabel('E(k)', fontsize=12)
    ax.set_title('1D Spectrum', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Perpendicular spectrum
    ax = axes[1]
    ax.loglog(kperp_bins, E_kperp, 'o-', linewidth=2, markersize=4, color='orange', label='E(k⊥)')
    # Reference k⊥^(-5/3)
    kp_ref = kperp_bins[(kperp_bins > 5) & (kperp_bins < 30)]
    if len(kp_ref) > 0:
        E_ref = 0.1 * kp_ref**(-5/3)
        ax.loglog(kp_ref, E_ref, 'k--', linewidth=2, alpha=0.7, label='k⊥^(-5/3)')
    # Forcing band
    ax.axvspan(k_force_min, k_force_max, alpha=0.2, color='green', label='Forcing band')
    ax.set_xlabel('k⊥', fontsize=12)
    ax.set_ylabel('E(k⊥)', fontsize=12)
    ax.set_title('Perpendicular Spectrum', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    # Parallel spectrum
    ax = axes[2]
    ax.semilogy(kpar_bins, E_kpar, 'o-', linewidth=2, markersize=4, color='purple', label='E(k∥)')
    # Forcing band
    ax.axvspan(-k_force_max, -k_force_min, alpha=0.2, color='green')
    ax.axvspan(k_force_min, k_force_max, alpha=0.2, color='green', label='Forcing band')
    ax.set_xlabel('k∥ (kz)', fontsize=12)
    ax.set_ylabel('E(k∥)', fontsize=12)
    ax.set_title('Parallel Spectrum', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "high_re_energy_spectra.png", dpi=150, bbox_inches='tight')
    print("✓ Saved high_re_energy_spectra.png")
    plt.close()

    # ==========================================================================
    # Summary
    # ==========================================================================

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    print("\nKey Results:")
    print(f"  • Initial energy: {E0:.4e}")
    print(f"  • Final energy: {E_final:.4e}")
    print(f"  • Energy change: {(E_final/E0):.2f}x")
    print(f"  • Average injection: ⟨ε_inj⟩ = {avg_injection:.3e}")
    print(f"  • Magnetic fraction: {E_mag_final / E_final:.3f}")
    print(f"  • Runtime: {elapsed:.1f}s ({elapsed/60:.2f} min)")

    print(f"\nParameters (for inertial range):")
    print(f"  • Resolution: {Nx}×{Ny}×{Nz}")
    print(f"  • Hyper-dissipation: r={hyper_r}, η={eta:.1e}")
    print(f"  • Forcing scale: k ∈ [{k_force_min}, {k_force_max}]")
    print(f"  • Dissipation scale: k_diss ~ (1/η)^(1/2r) ~ {(1/eta)**(1/(2*hyper_r)):.1f}")
    print(f"  • Reynolds number: Re ~ k_diss/k_force ~ {(1/eta)**(1/(2*hyper_r))/k_force_max:.1f}")

    print(f"\nOutput saved to: {output_dir}/")
    print(f"  - high_re_energy_history.png (energy evolution)")
    print(f"  - high_re_energy_spectra.png (E(k), E(k⊥), E(k∥))")

    print("\n" + "=" * 70)
    print("Check the perpendicular spectrum for k^(-5/3) inertial range!")
    print("Expected: E(k⊥) ∝ k⊥^(-5/3) for k ∈ [5, 30]")
    print("=" * 70)


if __name__ == "__main__":
    main()
