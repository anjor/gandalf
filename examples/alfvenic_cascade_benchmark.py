#!/usr/bin/env python3
"""
Alfvénic Turbulent Cascade Benchmark (Thesis Section 2.6.3, Figure 2.2)

Reproduces the thesis benchmark showing k⊥^(-5/3) critical-balance spectrum
for kinetic and magnetic energy. Runs to steady state and time-averages
spectra over the final window.

Thesis parameters:
- Resolutions: 64³ and 128³
- Hyper-diffusion: r=4 and r=8 (thesis)
- This implementation:
  - 32³: r=4 (k⊥⁸ damping, matches thesis)
  - 64³: r=2 (r=4 yields instability despite no overflow: η·dt=0.1<<50)
  - 128³: r=2 (r=4 also unstable, not overflow-limited)
- Run to saturation (steady state)
- Time-averaged spectra over final window (default: 30-50 τ_A)

Expected runtime (default 50 τ_A):
- 32³:  ~5-10 minutes
- 64³:  ~15-25 minutes
- 128³: ~1-2 hours

Steady-State Considerations:
- True steady state requires energy injection = dissipation (plateau in E(t))
- Default runtime (50 τ_A) should achieve steady state for 32³ and 64³
- For 128³ or publication quality: use --total-time 100 for cleaner results
- Check "Steady-state check" output during run: target ΔE/⟨E⟩ < 2%
- Averaging over final 20 τ_A (30-50) provides better statistics than shorter windows

Acceptable Energy Variation During Averaging:
- Ideal: ΔE/⟨E⟩ < 2% (truly steady state, recommended for publication)
- Good: ΔE/⟨E⟩ < 5% (weak growth acceptable, spectral slopes reliable)
- Marginal: ΔE/⟨E⟩ < 10% (spectral shape qualitatively correct, quantitative error ~5-10%)
- Unacceptable: ΔE/⟨E⟩ > 10% (non-stationary, biased spectra, extend runtime)
"""

import argparse
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
    energy_spectrum_perpendicular_kinetic,
    energy_spectrum_perpendicular_magnetic,
    plot_energy_history,
)


def detect_steady_state(energy_history, window=100, threshold=0.02):
    """
    Detect if system has reached steady state (energy plateau).

    Checks if energy has stopped growing by looking at the trend over
    a long window. True steady state requires energy to plateau, not
    just have small fluctuations.

    Args:
        energy_history: List of total energy values
        window: Number of recent points to check (default 100)
        threshold: Relative energy change threshold (default 2%)

    Returns:
        True if steady state detected (energy plateau), False otherwise
    """
    if len(energy_history) < window:
        return False

    recent = energy_history[-window:]
    # Average over 10 points to smooth out high-frequency fluctuations
    # while preserving low-frequency trends. This corresponds to ~0.5 τ_A
    # for typical save_interval=10 and dt~0.005.
    n_smooth = 10
    E_start = np.mean(recent[:n_smooth])   # Average of first n_smooth points in window
    E_end = np.mean(recent[-n_smooth:])    # Average of last n_smooth points in window

    if E_start == 0:
        return False

    # Check if energy has stopped growing (plateau)
    relative_change = abs(E_end - E_start) / E_start
    return relative_change < threshold


def main():
    """Run Alfvénic cascade benchmark to steady state."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Alfvénic Turbulent Cascade Benchmark')
    parser.add_argument('--resolution', type=int, default=64, choices=[32, 64, 128],
                        help='Grid resolution (32, 64, or 128)')
    parser.add_argument('--total-time', type=float, default=50.0,
                        help='Total simulation time in Alfvén times (default: 50)')
    parser.add_argument('--averaging-start', type=float, default=30.0,
                        help='When to start averaging in Alfvén times (default: 30)')
    parser.add_argument('--output-dir', type=str, default='examples/output',
                        help='Output directory for plots (default: examples/output)')
    args = parser.parse_args()

    print("=" * 70)
    print("Alfvénic Turbulent Cascade Benchmark (Thesis Section 2.6.3)")
    print("=" * 70)

    # ==========================================================================
    # PARAMETERS
    # ==========================================================================

    # Resolution-dependent parameters
    Nx = Ny = args.resolution
    Nz = args.resolution // 2  # Elongated box (standard RMHD)

    # With NORMALIZED hyper-dissipation (matching original GANDALF):
    # - Constraint is eta·dt < 50 (independent of resolution!)
    # - Using r=2 for practical stability with clean inertial range
    # - r=2 provides moderate dissipation range (broader than r=4, narrower than r=1)
    if args.resolution == 32:
        eta = 1.0         # Moderate dissipation for r=2
        nu = 1.0          # Hyper-collision coefficient
        hyper_r = 2       # Practical choice: stable with clean inertial range
        hyper_n = 2
    elif args.resolution == 64:
        # ANOMALY: 64³ requires 10× stronger dissipation than expected
        # Expected η ~ 1.5 (scaling from 32³), but needs η = 20.0 for stability
        # Root cause unclear - may be specific wavenumber resonance or
        # critical balance violation at this intermediate resolution.
        # See PR #81 for full parameter search history (7 failed attempts).
        # TODO: Investigate why 64³ is anomalously unstable
        eta = 20.0        # Anomalously strong dissipation required for stability
        nu = 1.0          # Doesn't matter (Hermite moments not evolved)
        hyper_r = 2       # Stable and practical for turbulence studies
        hyper_n = 2
    else:  # 128³
        eta = 2.0         # Stronger for high resolution
        nu = 2.0
        hyper_r = 2       # Maintains stability at high resolution
        hyper_n = 2

    Lx = Ly = Lz = 1.0    # Unit box (thesis convention)

    # Physics parameters
    v_A = 1.0             # Alfvén velocity
    beta_i = 1.0          # Ion plasma beta

    # Forcing parameters (inject energy at large scales)
    # Force modes 1 and 2 (largest scales, k = 2π/L for L=1)
    # With r=2 hyper-dissipation, need gentle forcing to avoid numerical instability
    if args.resolution == 32:
        force_amplitude = 0.05   # Gentle forcing for stability
    elif args.resolution == 64:
        force_amplitude = 0.01   # Extremely gentle forcing to prevent NaN blowup
    else:  # 128³
        force_amplitude = 0.05   # Conservative forcing for high resolution

    force_modes = [1, 2]    # Mode numbers to force
    k_force_min = (1 - 0.5) * 2 * np.pi / Lx  # Lower bound for mode 1
    k_force_max = (2 + 0.5) * 2 * np.pi / Lx  # Upper bound for mode 2

    # Initial condition (weak, let forcing drive the turbulence)
    alpha = 5.0 / 3.0     # k^(-5/3) initial spectrum
    amplitude = 0.05      # Weak initial amplitude
    k_min = 1.0
    k_max = 15.0

    # Time integration
    tau_A = Lz / v_A  # Alfvén crossing time (= 1.0 for unit box)
    total_time = args.total_time * tau_A  # Total simulation time
    averaging_start = args.averaging_start * tau_A  # When to start averaging
    cfl_safety = 0.3
    save_interval = 10

    # Steady-state logging (for informational purposes only, doesn't affect runtime)
    steady_state_check_interval = 50  # Check every N steps during averaging
    steady_state_window = 100
    steady_state_threshold = 0.02  # 2% relative change

    # Warn user if runtime may be insufficient for steady state
    if args.total_time < 30.0:
        print("\n" + "!" * 70)
        print("⚠️  WARNING: Runtime may be insufficient for true steady state!")
        print(f"    Current: {args.total_time} τ_A")
        print("    Recommended: ≥50 τ_A (default) or ≥100 τ_A for publication quality")
        print("    Use --total-time 50 or higher for reliable results")
        print("    Monitor 'Steady-state check' output during run (target: ΔE/⟨E⟩ < 2%)")
        print("!" * 70)

    print(f"\nGrid: {Nx} × {Ny} × {Nz}")
    print(f"Domain: [{Lx:.1f}, {Ly:.1f}, {Lz:.1f}]")
    print(f"Physics: v_A={v_A}, η={eta:.1e}, ν={nu:.1e}")
    print(f"Hyper-dissipation: r={hyper_r}, n={hyper_n}")
    print(f"Forcing: amplitude={force_amplitude}, modes {force_modes}")
    print(f"Evolution: Run for {args.total_time:.1f} τ_A total, average last {args.total_time - args.averaging_start:.1f} τ_A")
    print(f"CFL safety: {cfl_safety}")

    if args.resolution == 32:
        print(f"\nEstimated runtime: ~2-5 minutes")
    elif args.resolution == 64:
        print(f"\nEstimated runtime: ~5-10 minutes")
    else:
        print(f"\nEstimated runtime: ~30-60 minutes")

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
    print(f"  Total runtime: {total_time:.1f} time units = {int(total_time/dt)} timesteps")
    print(f"  Averaging starts at: {averaging_start:.1f} time units ({args.averaging_start:.1f} τ_A)")
    print(f"  Averaging duration: {total_time - averaging_start:.1f} time units ({args.total_time - args.averaging_start:.1f} τ_A)")

    # ==========================================================================
    # Time Evolution with Forcing
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Running forced evolution...")

    # Energy history tracking
    history = EnergyHistory()
    energy_values = []  # For steady-state logging

    # Store energy injection rates
    injection_rates = []

    # Time-averaged spectra (accumulated during averaging window)
    spectrum_kinetic_list = []
    spectrum_magnetic_list = []
    averaging_started = False
    averaging_start_step = None

    # Random key for forcing
    key = jax.random.PRNGKey(42)

    start_time = time.time()
    step = 0

    # Main loop: run until we reach total_time
    while state.time < total_time:
        # Compute current energy
        E_dict = compute_energy(state)
        E_total = E_dict['total']
        E_mag = E_dict['magnetic']
        E_kin = E_dict['kinetic']

        # Track history
        energy_values.append(E_total)

        if step % save_interval == 0:
            history.append(state)

            # Start averaging when we reach averaging_start time
            if not averaging_started and state.time >= averaging_start:
                averaging_started = True
                averaging_start_step = step
                print(f"\n  *** AVERAGING STARTED at step {step}, t={state.time:.2f} τ_A ***\n")

            # Collect spectra during averaging window
            if averaging_started:
                k_perp, E_kin_perp = energy_spectrum_perpendicular_kinetic(state)
                k_perp, E_mag_perp = energy_spectrum_perpendicular_magnetic(state)
                spectrum_kinetic_list.append(E_kin_perp)
                spectrum_magnetic_list.append(E_mag_perp)

                # Periodically check and log steady-state status
                if step % steady_state_check_interval == 0 and len(energy_values) >= averaging_start_step + steady_state_window:
                    recent_energy = energy_values[averaging_start_step:]
                    is_steady = detect_steady_state(
                        energy_values,
                        window=min(steady_state_window, len(recent_energy)),
                        threshold=steady_state_threshold
                    )
                    energy_variation = (max(recent_energy) - min(recent_energy)) / np.mean(recent_energy) * 100
                    status_symbol = "✓" if is_steady else "✗"
                    print(f"  {status_symbol} Steady-state check: ΔE/⟨E⟩ = {energy_variation:.1f}% ({'PASS' if is_steady else 'FAIL'})")

            # Print progress
            if step % 50 == 0:
                mag_frac = E_mag / E_total if E_total > 0 else 0
                avg_inj = np.mean(injection_rates[-50:]) if len(injection_rates) >= 50 else 0
                phase = "[AVERAGING]" if averaging_started else "[SPIN-UP]"
                n_spectra = len(spectrum_kinetic_list)
                print(f"  Step {step:5d}: t={state.time:.2f} τ_A, "
                      f"E={E_total:.4e}, f_mag={mag_frac:.3f}, "
                      f"⟨ε_inj⟩={avg_inj:.2e} {phase} (spectra: {n_spectra})")

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

        step += 1

    # Final summary
    if averaging_started:
        recent_energy = energy_values[averaging_start_step:]
        energy_variation = (max(recent_energy) - min(recent_energy)) / np.mean(recent_energy) * 100
        print(f"\n  *** EVOLUTION COMPLETE: {len(spectrum_kinetic_list)} spectra collected ***")
        print(f"  *** Energy range during averaging: [{min(recent_energy):.2e}, {max(recent_energy):.2e}] ***")
        print(f"  *** Relative variation: {energy_variation:.1f}% (target: <10% for good statistics) ***\n")

    elapsed = time.time() - start_time

    print(f"\n✓ Completed evolution")
    print(f"  Runtime: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")

    # ==========================================================================
    # Compute Time-Averaged Spectra
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Computing time-averaged spectra...")

    if len(spectrum_kinetic_list) == 0:
        print("ERROR: No spectra collected! Steady state not reached.")
        return

    # Time-average
    E_kin_avg = np.mean(spectrum_kinetic_list, axis=0)
    E_mag_avg = np.mean(spectrum_magnetic_list, axis=0)

    # Standard deviation for error bars
    E_kin_std = np.std(spectrum_kinetic_list, axis=0)
    E_mag_std = np.std(spectrum_magnetic_list, axis=0)

    averaging_duration = total_time - averaging_start
    print(f"✓ Averaged {len(spectrum_kinetic_list)} spectra over {averaging_duration:.1f} τ_A")

    # ==========================================================================
    # Final Diagnostics
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Final diagnostics...")

    E_final_dict = compute_energy(state)
    E_final = E_final_dict['total']
    E_mag_final = E_final_dict['magnetic']
    E_kin_final = E_final_dict['kinetic']

    total_injection = np.trapz(injection_rates, dx=dt)
    avg_injection = np.mean(injection_rates[len(injection_rates)//2:])  # Second half

    print(f"  Final energy: E_total = {E_final:.6e}")
    print(f"  Energy change: ΔE = {E_final - E0:.6e}")
    print(f"  Total injection: ∫ε_inj dt = {total_injection:.6e}")
    print(f"  Kinetic fraction: {E_kin_final / E_final:.3f}")
    print(f"  Magnetic fraction: {E_mag_final / E_final:.3f}")
    print(f"  Average injection rate (late time): ⟨ε_inj⟩ = {avg_injection:.3e}")

    # Analyze spectrum slope
    # Fit k⊥^(-5/3) in inertial range (k⊥ ~ 5-20)
    mask = (k_perp >= 5) & (k_perp <= 20)
    if np.sum(mask) > 5:
        k_fit = k_perp[mask]
        E_kin_fit = E_kin_avg[mask]
        E_mag_fit = E_mag_avg[mask]

        # Log-log linear fit
        log_k = np.log10(k_fit)
        log_E_kin = np.log10(E_kin_fit + 1e-20)
        log_E_mag = np.log10(E_mag_fit + 1e-20)

        slope_kin = np.polyfit(log_k, log_E_kin, 1)[0]
        slope_mag = np.polyfit(log_k, log_E_mag, 1)[0]

        print(f"\n  Inertial range slopes (k⊥ ∈ [5, 20]):")
        print(f"    Kinetic:  k⊥^({slope_kin:.2f})  (expected: -5/3 = -1.67)")
        print(f"    Magnetic: k⊥^({slope_mag:.2f})  (expected: -5/3 = -1.67)")

    # ==========================================================================
    # Visualization
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Creating visualizations...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Convert k⊥ to mode numbers: n = k⊥ L / (2π)
    n_perp = k_perp * Lx / (2 * np.pi)
    n_force_min = k_force_min * Lx / (2 * np.pi)
    n_force_max = k_force_max * Lx / (2 * np.pi)

    # Create figure with three panels
    fig = plt.figure(figsize=(15, 5))

    # -------------------------------------------------------------------------
    # Panel 1: Energy Evolution
    # -------------------------------------------------------------------------
    ax1 = plt.subplot(131)
    times = np.array(history.times)
    energies = np.array(history.E_total)
    ax1.plot(times, energies, 'k-', linewidth=2)

    # Mark averaging window
    if averaging_started:
        ax1.axvline(averaging_start, color='r', linestyle='--', linewidth=1.5,
                   label=f'Averaging starts (t={averaging_start:.1f})')
        ax1.axvspan(averaging_start, total_time,
                   color='red', alpha=0.1, label=f'Averaging window ({total_time - averaging_start:.1f} time units)')

    ax1.set_xlabel('Time [τ_A]', fontsize=12)
    ax1.set_ylabel('Total Energy', fontsize=12)
    ax1.set_title('Energy Evolution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # -------------------------------------------------------------------------
    # Panel 2: Time-Averaged Kinetic Spectrum
    # -------------------------------------------------------------------------
    ax2 = plt.subplot(132)

    # Kinetic spectrum with error bars
    ax2.loglog(n_perp, E_kin_avg, 'b-', linewidth=2, label='Kinetic')
    ax2.fill_between(n_perp, E_kin_avg - E_kin_std, E_kin_avg + E_kin_std,
                     color='b', alpha=0.2)

    # Reference slope n^(-5/3)
    n_ref = np.array([1.0, 5.0])
    E_ref = 0.5 * n_ref**(-5.0/3.0)
    ax2.loglog(n_ref, E_ref, 'k--', linewidth=1.5, label='n^(-5/3)')

    # Forcing range
    ax2.axvspan(n_force_min, n_force_max, color='green', alpha=0.1,
               label='Forcing modes 1-2')

    ax2.set_xlabel('Mode number n', fontsize=12)
    ax2.set_ylabel('E_kin(n)', fontsize=12)
    ax2.set_title('Kinetic Energy Spectrum (Time-Averaged)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(fontsize=10)
    ax2.set_xlim(0.5, n_perp[-1])

    # -------------------------------------------------------------------------
    # Panel 3: Time-Averaged Magnetic Spectrum
    # -------------------------------------------------------------------------
    ax3 = plt.subplot(133)

    # Magnetic spectrum with error bars
    ax3.loglog(n_perp, E_mag_avg, 'r-', linewidth=2, label='Magnetic')
    ax3.fill_between(n_perp, E_mag_avg - E_mag_std, E_mag_avg + E_mag_std,
                     color='r', alpha=0.2)

    # Reference slope n^(-5/3)
    ax3.loglog(n_ref, E_ref, 'k--', linewidth=1.5, label='n^(-5/3)')

    # Forcing range
    ax3.axvspan(n_force_min, n_force_max, color='green', alpha=0.1,
               label='Forcing modes 1-2')

    ax3.set_xlabel('Mode number n', fontsize=12)
    ax3.set_ylabel('E_mag(n)', fontsize=12)
    ax3.set_title('Magnetic Energy Spectrum (Time-Averaged)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend(fontsize=10)
    ax3.set_xlim(0.5, n_perp[-1])

    plt.tight_layout()

    filename = f"alfvenic_cascade_{args.resolution}cubed.png"
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Saved figure: {filepath}")

    # -------------------------------------------------------------------------
    # Combined Kinetic + Magnetic Plot (like thesis Figure 2.2)
    # -------------------------------------------------------------------------
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))

    # Kinetic
    ax_left.loglog(n_perp, E_kin_avg, 'b-', linewidth=2.5,
                   label=f'{args.resolution}³, r={hyper_r}')
    ax_left.fill_between(n_perp, E_kin_avg - E_kin_std, E_kin_avg + E_kin_std,
                        color='b', alpha=0.2)
    ax_left.loglog(n_ref, E_ref, 'k--', linewidth=2, label='k⊥^(-5/3)')
    ax_left.set_xlabel('k⊥', fontsize=14)
    ax_left.set_ylabel('Kinetic Energy', fontsize=14)
    ax_left.set_title('Kinetic Energy Spectrum', fontsize=16, fontweight='bold')
    ax_left.grid(True, alpha=0.3, which='both')
    ax_left.legend(fontsize=12)
    ax_left.set_xlim(0.5, n_perp[-1])

    # Magnetic
    ax_right.loglog(n_perp, E_mag_avg, 'r-', linewidth=2.5,
                    label=f'{args.resolution}³, r={hyper_r}')
    ax_right.fill_between(n_perp, E_mag_avg - E_mag_std, E_mag_avg + E_mag_std,
                         color='r', alpha=0.2)
    ax_right.loglog(n_ref, E_ref, 'k--', linewidth=2, label='k⊥^(-5/3)')
    ax_right.set_xlabel('k⊥', fontsize=14)
    ax_right.set_ylabel('Magnetic Energy', fontsize=14)
    ax_right.set_title('Magnetic Energy Spectrum', fontsize=16, fontweight='bold')
    ax_right.grid(True, alpha=0.3, which='both')
    ax_right.legend(fontsize=12)
    ax_right.set_xlim(0.5, n_perp[-1])

    plt.tight_layout()

    filename_thesis = f"alfvenic_cascade_thesis_style_{args.resolution}cubed.png"
    filepath_thesis = output_dir / filename_thesis
    plt.savefig(filepath_thesis, dpi=150, bbox_inches='tight')
    print(f"✓ Saved thesis-style figure: {filepath_thesis}")

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
