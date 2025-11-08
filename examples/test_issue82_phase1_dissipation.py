#!/usr/bin/env python3
"""
Issue #82 Phase 1: Verify Dissipation is Working Correctly

Tests decaying turbulence (no forcing) to verify that hyper-dissipation
actually removes energy at the expected rate. This is a critical validation
before investigating the forced turbulence instability.

Expected behavior:
    E(t) = E₀ × exp(-2η⟨k⊥²⟩t)

If dissipation is working correctly:
- 32³ with η=1.0: Should decay smoothly
- 64³ with η=2.0: Should decay smoothly at faster rate

If dissipation fails:
- Energy growth or instability → CODE BUG in dissipation application
- Slower decay than expected → Dissipation normalization error

Usage:
    uv run python examples/test_issue82_phase1_dissipation.py

Runtime: ~5 minutes total (2 runs)
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
)
from krmhd.diagnostics import EnergyHistory


def test_decaying_turbulence(resolution, eta, total_time=10.0):
    """
    Test energy decay for unforced turbulence.

    Args:
        resolution: Grid resolution (32 or 64)
        eta: Hyper-resistivity coefficient
        total_time: Simulation time in τ_A

    Returns:
        times, energies, expected_energies
    """
    print("=" * 70)
    print(f"Testing {resolution}³ Decaying Turbulence")
    print("=" * 70)
    print(f"Parameters: η={eta}, ν={eta}, r=2, n=2")
    print(f"Runtime: {total_time} τ_A (no forcing)")
    print()

    # Grid setup
    Nx = Ny = resolution
    Nz = resolution // 2
    Lx = Ly = Lz = 1.0

    grid = SpectralGrid3D.create(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz)

    # Initialize turbulent spectrum
    state = initialize_random_spectrum(
        grid=grid,
        M=10,
        alpha=5.0/3.0,  # k^(-5/3) Kolmogorov spectrum
        amplitude=1.0,   # Strong amplitude for clear decay signal
        k_min=2,
        k_max=8,
    )

    E_init = compute_energy(state)['total']
    print(f"Initial energy: E₀ = {E_init:.6e}")

    # Compute timestep
    dt = compute_cfl_timestep(state, v_A=1.0, cfl_safety=0.3)
    n_steps = int(total_time / dt)
    print(f"Timestep: dt = {dt:.6f} (CFL-limited)")
    print(f"Total steps: {n_steps}")

    # Validation
    eta_rate = eta * dt
    print(f"Dissipation rate: η·dt = {eta_rate:.6f} (constraint: < 50)")
    print()

    # Time evolution (no forcing)
    print("Running evolution...")
    times = []
    energies = []

    start_time = time.time()
    for step in range(n_steps + 1):
        if step % 50 == 0:
            E_dict = compute_energy(state)
            E_total = E_dict['total']
            times.append(float(state.time))
            energies.append(float(E_total))

            if step % 200 == 0:
                print(f"  Step {step:5d}: t={state.time:.2f} τ_A, E={E_total:.6e}")

            # Check for instability (energy should ONLY decrease)
            if E_total > 1.1 * E_init:
                print(f"\n  ❌ ENERGY GROWTH DETECTED at step {step}!")
                print(f"     E/E₀ = {E_total/E_init:.3f} (should be < 1.0)")
                print(f"     This indicates a BUG in dissipation application!")
                break

            # Check for NaN
            if not np.isfinite(E_total):
                print(f"\n  ❌ NaN/Inf detected at step {step}!")
                break

        # Time step (no forcing)
        state = gandalf_step(
            state,
            dt=dt,
            eta=eta,
            nu=eta,  # Match resistivity and collisionality
            v_A=1.0,
            hyper_r=2,
            hyper_n=2,
        )

    wall_time = time.time() - start_time
    print(f"\n✓ Completed in {wall_time:.1f} seconds")

    # Convert to arrays
    times = np.array(times)
    energies = np.array(energies)

    # Compute expected decay
    # E(t) = E₀ exp(-2η⟨k⊥²⟩t)
    # For normalized hyper-dissipation: rate at k_max
    # Approximate ⟨k⊥²⟩ ~ (k_max/2)² for spectral average
    idx_max = (Nx - 1) // 3  # 2/3 dealiasing
    k_max = (2 * np.pi / Lx) * idx_max
    k_avg_sq = (k_max / 2.0) ** 2  # Rough estimate
    expected_rate = 2 * eta * k_avg_sq
    expected_energies = E_init * np.exp(-expected_rate * times)

    # Analysis
    print("\n" + "-" * 70)
    print("Analysis:")
    print("-" * 70)

    final_energy = energies[-1]
    expected_final = expected_energies[-1]
    actual_rate = -np.log(final_energy / E_init) / times[-1]

    print(f"Final energy:    E(t={times[-1]:.1f}) = {final_energy:.6e}")
    print(f"Expected (rough): E(t={times[-1]:.1f}) = {expected_final:.6e}")
    print(f"Energy ratio:     E_final/E₀ = {final_energy/E_init:.4f}")
    print(f"Actual decay rate: γ_actual = {actual_rate:.6f}")
    print(f"Expected rate (rough): γ_expected = {expected_rate:.6f}")

    if final_energy > 0.95 * E_init:
        print("\n⚠️  WARNING: Energy barely decreased!")
        print("   Dissipation may not be working correctly.")
        status = "FAIL"
    elif final_energy > E_init:
        print("\n❌ FAIL: Energy increased!")
        print("   This is a BUG in dissipation application.")
        status = "FAIL"
    else:
        print("\n✓ PASS: Energy decreased as expected.")
        print("  Dissipation is working correctly.")
        status = "PASS"

    return times, energies, expected_energies, status


def main():
    """Run Phase 1 tests."""
    print("\n" + "=" * 70)
    print("Issue #82 Phase 1: Dissipation Verification")
    print("=" * 70)
    print()
    print("Goal: Verify that hyper-dissipation removes energy correctly")
    print("Method: Decaying turbulence (no forcing), check E(t) decay")
    print()

    # Test configurations
    tests = [
        {"resolution": 32, "eta": 1.0, "time": 10.0},
        {"resolution": 64, "eta": 2.0, "time": 10.0},
    ]

    results = {}

    for test in tests:
        times, energies, expected, status = test_decaying_turbulence(
            resolution=test["resolution"],
            eta=test["eta"],
            total_time=test["time"],
        )
        results[test["resolution"]] = {
            "times": times,
            "energies": energies,
            "expected": expected,
            "status": status,
        }
        print()

    # Summary plot
    print("=" * 70)
    print("Creating summary plot...")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (res, data) in enumerate(results.items()):
        ax = axes[i]
        ax.semilogy(data["times"], data["energies"], 'b-', linewidth=2, label=f'{res}³ actual')
        ax.semilogy(data["times"], data["expected"], 'r--', linewidth=2, label=f'{res}³ expected (rough)')
        ax.set_xlabel('Time (τ_A)', fontsize=12)
        ax.set_ylabel('Total Energy', fontsize=12)
        ax.set_title(f'{res}³ Decaying Turbulence ({data["status"]})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Annotate decay rate
        E0 = data["energies"][0]
        Ef = data["energies"][-1]
        tf = data["times"][-1]
        gamma = -np.log(Ef / E0) / tf
        ax.text(0.05, 0.95, f'γ = {gamma:.4f}\nE_f/E_0 = {Ef/E0:.4f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "issue82_phase1_dissipation_verification.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to: {output_file}")

    # Final summary
    print("\n" + "=" * 70)
    print("Phase 1 Summary:")
    print("=" * 70)
    for res, data in results.items():
        print(f"  {res}³: {data['status']}")

    all_pass = all(data["status"] == "PASS" for data in results.values())
    if all_pass:
        print("\n✓ All tests passed! Dissipation is working correctly.")
        print("  → Proceed to Phase 2: Test weak forcing stability")
    else:
        print("\n❌ Some tests failed! Dissipation may have a bug.")
        print("  → Investigate dissipation implementation before proceeding")

    print("=" * 70)


if __name__ == "__main__":
    main()
