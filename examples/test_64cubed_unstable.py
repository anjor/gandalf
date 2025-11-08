#!/usr/bin/env python3
"""
Test Script for 64³ Unstable Parameters (Issue #82)

This script runs the 64³ resolution with parameters that are EXPECTED TO FAIL
based on the parameter search history in PR #81. The goal is to capture detailed
diagnostics during the instability development to identify the root cause.

Unstable parameters (from PR #81):
- η = 2.0 (10× weaker than stable η=20.0)
- force_amplitude = 0.05 (5× stronger than stable 0.01)
- hyper_r = 2

Expected behavior:
- Instability develops around t ~ 15-20 τ_A (based on PR #81 attempt #5)
- NaN or exponential blow-up terminates run early

This script is instrumented with comprehensive diagnostics to capture:
- WHEN: At what time does instability start?
- WHERE: Which field diverges first (velocity, CFL, nonlinear term)?
- HOW: Is it exponential growth or sudden jump?

Output:
- turbulence_diagnostics_64cubed_unstable.h5
- alfvenic_cascade_64cubed_unstable.png
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
    energy_spectrum_perpendicular_kinetic,
    energy_spectrum_perpendicular_magnetic,
    compute_turbulence_diagnostics,
)
from krmhd.io import save_turbulence_diagnostics


def main():
    """Run 64³ with UNSTABLE parameters to capture instability development."""

    print("=" * 70)
    print("Issue #82: Testing 64³ UNSTABLE Parameters")
    print("=" * 70)
    print("\n⚠️  WARNING: This run is EXPECTED TO FAIL")
    print("   Purpose: Capture diagnostics during instability development")
    print("   Based on PR #81 parameter search (attempt #5: η=2.5, failed at t~14.8)\n")

    # ==========================================================================
    # UNSTABLE PARAMETERS (from PR #81)
    # ==========================================================================

    # Grid
    Nx = Ny = 64
    Nz = 32
    Lx = Ly = Lz = 1.0

    # Physics
    v_A = 1.0
    beta_i = 1.0

    # UNSTABLE dissipation (expected to fail)
    eta = 2.0           # 10× weaker than stable 20.0
    nu = 1.0
    hyper_r = 2
    hyper_n = 2

    # UNSTABLE forcing (5× stronger than stable 0.01)
    force_amplitude = 0.05
    force_modes = [1, 2]
    k_force_min = (1 - 0.5) * 2 * np.pi / Lx
    k_force_max = (2 + 0.5) * 2 * np.pi / Lx

    # Initial condition
    alpha = 5.0 / 3.0
    amplitude = 0.05
    k_min = 1.0
    k_max = 15.0

    # Time integration
    tau_A = Lz / v_A
    total_time = 30.0 * tau_A  # Run up to 30 τ_A (will likely fail before this)
    cfl_safety = 0.3
    save_interval = 10

    # Diagnostic settings - FREQUENT sampling to capture instability
    diagnostic_interval = 5  # Every 5 steps
    progress_print_interval = 50

    # Output directory
    output_dir = Path("examples/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGrid: {Nx} × {Ny} × {Nz}")
    print(f"Domain: [{Lx}, {Ly}, {Lz}]")
    print(f"Physics: v_A={v_A}, η={eta:.1e}, ν={nu:.1e}")
    print(f"Hyper-dissipation: r={hyper_r}, n={hyper_n}")
    print(f"Forcing: amplitude={force_amplitude}, modes {force_modes}")
    print(f"Evolution: Run up to {total_time/tau_A:.1f} τ_A (will terminate on failure)")
    print(f"CFL safety: {cfl_safety}")

    # ==========================================================================
    # Initialization
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Initializing...")

    # Create grid
    grid = SpectralGrid3D.create(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz)
    print(f"✓ Created {Nx}×{Ny}×{Nz} spectral grid")

    # Initialize state
    state = initialize_random_spectrum(
        grid,
        M=10,
        alpha=alpha,
        amplitude=amplitude,
        k_min=k_min,
        k_max=k_max,
    )
    E_init = compute_energy(state)['total']
    print(f"✓ Initialized weak k^({-alpha:.2f}) spectrum")
    print(f"  Initial energy: E_total = {E_init:.6e}")

    # Compute timestep
    dt = compute_cfl_timestep(state, v_A=v_A, cfl_safety=cfl_safety)
    n_steps = int(total_time / dt)
    print(f"\n  Using dt = {dt:.4f} (CFL-limited)")
    print(f"  Total runtime: {total_time:.1f} time units = {n_steps} timesteps")

    # Validation
    eta_rate = eta * dt
    nu_rate = nu * dt
    print(f"\n  Dissipation rate at k_max: η·dt = {eta_rate:.4f} (constraint: < 50)")
    print(f"  Collision rate at M=10: ν·dt = {nu_rate:.4f} (constraint: < 50)")

    # ==========================================================================
    # Time Evolution with Diagnostics
    # ==========================================================================

    print("\n" + "-" * 70)
    print("Running evolution with UNSTABLE parameters...")
    print("DIAGNOSTIC MODE: Tracking instability development")
    print(f"  Computing diagnostics every {diagnostic_interval} steps")
    print(f"  Terminating on NaN/Inf or excessive growth (max_vel > 1000)")
    print("-" * 70 + "\n")

    # Data structures
    history = EnergyHistory()
    energy_values = []
    injection_rates = []
    diagnostics_list = []

    # Random key
    key = jax.random.PRNGKey(42)

    start_time = time.time()
    step = 0
    termination_reason = "completed"  # Will be updated if early termination

    # Main loop
    while state.time < total_time:
        # Compute energy
        E_dict = compute_energy(state)
        E_total = E_dict['total']
        energy_values.append(E_total)

        # Compute diagnostics (frequent sampling)
        if step % diagnostic_interval == 0:
            diag = compute_turbulence_diagnostics(state, dt=dt, v_A=v_A)
            diagnostics_list.append(diag)

            # Check for instability signatures
            if not np.isfinite(diag.max_velocity):
                print(f"\n  ❌ NaN/Inf in max_velocity at step {step}, t={state.time:.2f} τ_A")
                termination_reason = "NaN_velocity"
                break

            if diag.cfl_number > 5.0:  # Way beyond stability limit
                print(f"\n  ❌ EXTREME CFL violation at step {step}, t={state.time:.2f} τ_A")
                print(f"     CFL = {diag.cfl_number:.3f} (critical: 1.0)")
                termination_reason = "extreme_CFL"
                break

            if diag.max_velocity > 1000.0:  # Clearly blown up
                print(f"\n  ❌ VELOCITY BLOW-UP at step {step}, t={state.time:.2f} τ_A")
                print(f"     max_vel = {diag.max_velocity:.2e}")
                termination_reason = "velocity_blowup"
                break

            # Warnings (don't terminate, just log)
            if diag.cfl_number > 1.0 and step % progress_print_interval == 0:
                print(f"  ⚠️  CFL > 1.0 at step {step}, t={state.time:.2f}: CFL = {diag.cfl_number:.3f}")

        # Save to history
        if step % save_interval == 0:
            history.append(state)

            # Progress report
            if step % progress_print_interval == 0:
                # Check for NaN in total energy
                if not np.isfinite(E_total):
                    E_mag = E_dict['magnetic']
                    E_kin = E_dict['kinetic']
                    print(f"\n  ❌ NaN/Inf in energy at step {step}, t={state.time:.2f} τ_A")
                    print(f"     E_total={E_total}, E_kin={E_kin}, E_mag={E_mag}")
                    termination_reason = "NaN_energy"
                    break

                E_mag = E_dict['magnetic']
                mag_frac = E_mag / E_total if E_total > 0 else 0
                avg_inj = np.mean(injection_rates[-50:]) if len(injection_rates) >= 50 else 0

                # Get last diagnostic if available
                if diagnostics_list:
                    last_diag = diagnostics_list[-1]
                    print(f"  Step {step:5d}: t={state.time:.2f} τ_A, E={E_total:.4e}, "
                          f"f_mag={mag_frac:.3f}, ⟨ε_inj⟩={avg_inj:.2e}, "
                          f"max_vel={last_diag.max_velocity:.3f}, CFL={last_diag.cfl_number:.3f}")
                else:
                    print(f"  Step {step:5d}: t={state.time:.2f} τ_A, E={E_total:.4e}, "
                          f"f_mag={mag_frac:.3f}, ⟨ε_inj⟩={avg_inj:.2e}")

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

        # Compute injection rate
        eps_inj = compute_energy_injection_rate(state_before, state, dt)
        injection_rates.append(float(eps_inj))

        # Time step
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

    # ==========================================================================
    # Post-Processing and Analysis
    # ==========================================================================

    elapsed = time.time() - start_time
    final_time = state.time

    print("\n" + "=" * 70)
    print("Evolution terminated")
    print("=" * 70)
    print(f"  Reason: {termination_reason}")
    print(f"  Final time: t = {final_time:.2f} τ_A ({step} steps)")
    print(f"  Wall time: {elapsed:.1f} seconds")

    if termination_reason != "completed":
        print(f"\n  ✓ SUCCESS: Captured instability development!")
        print(f"    Time-to-failure: {final_time:.2f} τ_A")
        print(f"    Diagnostic samples: {len(diagnostics_list)}")

    # ==========================================================================
    # Save Diagnostics
    # ==========================================================================

    if diagnostics_list:
        print("\n" + "=" * 70)
        print("Saving turbulence diagnostics")
        print("=" * 70)

        diag_filename = "turbulence_diagnostics_64cubed_unstable.h5"
        diag_filepath = output_dir / diag_filename

        metadata = {
            'resolution': 64,
            'eta': float(eta),
            'nu': float(nu),
            'hyper_r': int(hyper_r),
            'hyper_n': int(hyper_n),
            'force_amplitude': float(force_amplitude),
            'dt': float(dt),
            'total_time': float(final_time),
            'n_steps': int(step),
            'termination_reason': termination_reason,
            'description': '64³ UNSTABLE parameters (Issue #82 investigation)'
        }

        save_turbulence_diagnostics(diagnostics_list, str(diag_filepath), metadata=metadata)
        print(f"✓ Saved {len(diagnostics_list)} diagnostic samples to: {diag_filepath}")
        print(f"  Time range: t={diagnostics_list[0].time:.2f} to {diagnostics_list[-1].time:.2f} τ_A")

        # Summary statistics
        max_velocities = [d.max_velocity for d in diagnostics_list]
        cfl_numbers = [d.cfl_number for d in diagnostics_list]
        energy_highk = [d.energy_highk for d in diagnostics_list]

        print(f"\n  Summary Statistics:")
        print(f"    max_velocity:  min={min(max_velocities):.3f}, max={max(max_velocities):.3f}")
        print(f"    CFL number:    min={min(cfl_numbers):.3f}, max={max(cfl_numbers):.3f}")
        print(f"    High-k energy: min={min(energy_highk):.4f}, max={max(energy_highk):.4f}")

        # Check for exponential growth
        if len(diagnostics_list) > 10:
            times = np.array([d.time for d in diagnostics_list])
            velocities = np.array(max_velocities)

            # Fit last 50% of data
            mid_idx = len(times) // 2
            t_fit = times[mid_idx:]
            vel_fit = velocities[mid_idx:]

            if vel_fit[-1] > vel_fit[0] * 2:  # At least doubled
                log_vel = np.log(vel_fit + 1e-10)
                coeffs = np.polyfit(t_fit, log_vel, 1)
                growth_rate = coeffs[0]

                print(f"\n  📈 Exponential Growth Analysis:")
                print(f"    Growth rate: γ = {growth_rate:.3f} τ_A⁻¹")
                print(f"    Doubling time: {np.log(2)/growth_rate:.2f} τ_A")

    print("\n" + "=" * 70)
    print("Test complete! Analyze with:")
    print("  uv run python examples/analyze_issue82_diagnostics.py \\")
    print("    --files examples/output/turbulence_diagnostics_64cubed_unstable.h5")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
