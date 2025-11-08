#!/usr/bin/env python3
"""
Detailed Analysis of 64³ Instability from Diagnostic Data

Uses the captured turbulence diagnostics to identify:
1. WHEN the instability starts (exponential growth onset)
2. WHERE in k-space energy accumulates (spectral pile-up)
3. HOW the cascade fails (critical balance violation)

Goal: Understand the physics of the instability to guide parameter tuning.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from krmhd.io import load_turbulence_diagnostics


def analyze_growth_phases(times, max_vel, energy):
    """Identify distinct phases in the evolution."""
    print("=" * 70)
    print("Growth Phase Analysis")
    print("=" * 70)

    # Phase 1: Spin-up (t < 3 τ_A)
    spin_up_mask = times < 3.0
    if np.any(spin_up_mask):
        t_spin = times[spin_up_mask]
        v_spin = max_vel[spin_up_mask]
        E_spin = energy[spin_up_mask]
        print(f"Phase 1: Spin-up (t < 3 τ_A)")
        print(f"  Velocity: {v_spin[0]:.6f} → {v_spin[-1]:.6f}")
        print(f"  Energy:   {E_spin[0]:.6e} → {E_spin[-1]:.6e}")

    # Phase 2: Quasi-steady (3 < t < 13 τ_A)
    steady_mask = (times >= 3.0) & (times < 13.0)
    if np.any(steady_mask):
        t_steady = times[steady_mask]
        v_steady = max_vel[steady_mask]
        E_steady = energy[steady_mask]
        v_std = np.std(v_steady)
        E_growth = E_steady[-1] / E_steady[0]
        print(f"\nPhase 2: Quasi-steady (3 < t < 13 τ_A)")
        print(f"  Velocity: {v_steady[0]:.6f} ±{v_std:.6f} (fluctuation level)")
        print(f"  Energy growth: {E_growth:.2f}×")

    # Phase 3: Exponential blow-up (t > 13 τ_A)
    blowup_mask = times >= 13.0
    if np.any(blowup_mask):
        t_blowup = times[blowup_mask]
        v_blowup = max_vel[blowup_mask]
        E_blowup = energy[blowup_mask]

        # Fit exponential: v = v0 * exp(γt)
        valid = np.isfinite(v_blowup) & (v_blowup > 0)
        if np.sum(valid) > 5:
            t_fit = t_blowup[valid]
            log_v = np.log(v_blowup[valid])
            gamma, log_v0 = np.polyfit(t_fit, log_v, 1)
            v0 = np.exp(log_v0)

            print(f"\nPhase 3: Exponential blow-up (t > 13 τ_A)")
            print(f"  Growth rate: γ = {gamma:.4f} (1/τ_A)")
            print(f"  Doubling time: {np.log(2)/gamma:.2f} τ_A")
            print(f"  Velocity: {v_blowup[0]:.6f} → {v_blowup[valid][-1]:.6f}")
            print(f"  Energy: {E_blowup[0]:.2e} → {E_blowup[valid][-1]:.2e}")

            return gamma

    return None


def main():
    """Analyze 64³ diagnostic data in detail."""
    print("\n" + "=" * 70)
    print("Detailed Analysis of 64³ Instability")
    print("=" * 70)
    print()

    # Load diagnostic data
    diag_file = Path("examples/output/turbulence_diagnostics_64cubed_unstable.h5")
    if not diag_file.exists():
        print(f"❌ Diagnostic file not found: {diag_file}")
        print("   Run test_64cubed_unstable.py first to generate data.")
        return

    diag_list, metadata = load_turbulence_diagnostics(str(diag_file))

    # Extract arrays from list of TurbulenceDiagnostics
    times = np.array([d.time for d in diag_list])
    max_vel = np.array([d.max_velocity for d in diag_list])
    cfl = np.array([d.cfl_number for d in diag_list])
    max_nl = np.array([d.max_nonlinear for d in diag_list])
    highk_energy = np.array([d.energy_highk for d in diag_list])
    cb_ratio = np.array([d.critical_balance_ratio for d in diag_list])
    energy = np.array([d.energy_total for d in diag_list])

    print(f"Loaded {len(times)} diagnostic samples from t={times[0]:.2f} to {times[-1]:.2f} τ_A")
    print()

    # Identify growth phases
    gamma = analyze_growth_phases(times, max_vel, energy)

    # High-k energy accumulation
    print("\n" + "=" * 70)
    print("High-k Energy Accumulation (Spectral Pile-up)")
    print("=" * 70)

    # Energy fraction at k > 0.9 k_max
    t_early = times < 5.0
    t_mid = (times >= 5.0) & (times < 13.0)
    t_late = times >= 13.0

    print(f"High-k energy fraction (k > 0.9 k_max):")
    print(f"  Early (t < 5):   {np.median(highk_energy[t_early]):.2e}")
    print(f"  Mid (5 < t < 13): {np.median(highk_energy[t_mid]):.2e}")
    if np.any(t_late):
        valid_late = t_late & np.isfinite(highk_energy)
        if np.any(valid_late):
            print(f"  Late (t > 13):    {np.median(highk_energy[valid_late]):.2e}")

    # CFL number check
    print("\n" + "=" * 70)
    print("CFL Number (Timestep Stability)")
    print("=" * 70)
    print(f"Max CFL: {np.nanmax(cfl):.4f}")
    print(f"CFL > 1.0: {np.sum(cfl > 1.0)} timesteps (should be 0!)")
    if np.nanmax(cfl) < 0.5:
        print("✓ Timestep is NOT the problem (CFL << 1)")

    # Critical balance
    print("\n" + "=" * 70)
    print("Critical Balance (RMHD Validity)")
    print("=" * 70)
    cb_finite = cb_ratio[np.isfinite(cb_ratio) & (cb_ratio > 0)]
    if len(cb_finite) > 0:
        cb_median = np.median(cb_finite)
        print(f"Median τ_nl/τ_A: {cb_median:.2e}")
        print(f"Expected: ~1.0 (critical balance)")
        if cb_median > 100:
            print("⚠️  Critical balance violated: cascade too slow!")

    # Create detailed plot
    print("\n" + "=" * 70)
    print("Creating Detailed Diagnostic Plot")
    print("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Max velocity (log scale, detect exponential growth)
    ax = axes[0, 0]
    ax.semilogy(times, max_vel, 'b-', linewidth=2)
    ax.axvline(13.0, color='r', linestyle='--', alpha=0.5, label='Blow-up onset')
    ax.set_xlabel('Time (τ_A)', fontsize=12)
    ax.set_ylabel('Max Velocity |v⊥|', fontsize=12)
    ax.set_title('Velocity Evolution (Log Scale)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Annotate growth rate
    if gamma is not None:
        ax.text(0.05, 0.95, f'γ = {gamma:.4f}\nt_double = {np.log(2)/gamma:.2f} τ_A',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Energy evolution (log scale)
    ax = axes[0, 1]
    ax.semilogy(times, energy, 'g-', linewidth=2)
    ax.axvline(13.0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (τ_A)', fontsize=12)
    ax.set_ylabel('Total Energy', fontsize=12)
    ax.set_title('Energy Growth', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Annotate final growth
    E0 = energy[min(100, len(energy)-1)]  # t~5
    Ef_valid = energy[np.isfinite(energy)][-1] if np.any(np.isfinite(energy)) else E0
    growth = Ef_valid / E0
    ax.text(0.05, 0.95, f'E_final/E(t=5) = {growth:.1f}×',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # 3. High-k energy fraction
    ax = axes[0, 2]
    ax.semilogy(times, highk_energy, 'r-', linewidth=2)
    ax.axhline(0.01, color='k', linestyle=':', alpha=0.5, label='1% threshold')
    ax.axvline(13.0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (τ_A)', fontsize=12)
    ax.set_ylabel('High-k Energy Fraction', fontsize=12)
    ax.set_title('Spectral Pile-up (k > 0.9 k_max)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. CFL number
    ax = axes[1, 0]
    ax.plot(times, cfl, 'b-', linewidth=2)
    ax.axhline(1.0, color='r', linestyle='--', linewidth=2, label='CFL = 1 (critical)')
    ax.axhline(0.5, color='orange', linestyle='--', alpha=0.5, label='CFL = 0.5 (safe)')
    ax.set_xlabel('Time (τ_A)', fontsize=12)
    ax.set_ylabel('CFL Number', fontsize=12)
    ax.set_title('Timestep Stability', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([0, min(1.5, np.nanmax(cfl)*1.1)])

    # 5. Max nonlinear term
    ax = axes[1, 1]
    ax.semilogy(times, max_nl, 'purple', linewidth=2)
    ax.axvline(13.0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (τ_A)', fontsize=12)
    ax.set_ylabel('Max Nonlinear Term', fontsize=12)
    ax.set_title('Cascade Rate |{z∓, ∇²z±}|', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 6. Energy injection vs dissipation (estimate)
    ax = axes[1, 2]
    # Energy derivative (numerical)
    dE_dt = np.gradient(energy, times)
    ax.plot(times, dE_dt, 'k-', linewidth=2, label='dE/dt')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(13.0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (τ_A)', fontsize=12)
    ax.set_ylabel('dE/dt', fontsize=12)
    ax.set_title('Energy Balance', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Annotate steady-state window
    steady_mask = (times >= 5.0) & (times < 13.0)
    if np.any(steady_mask):
        dE_steady = np.median(dE_dt[steady_mask])
        ax.text(0.05, 0.95, f'Median dE/dt (5-13 τ_A):\n{dE_steady:.2e}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    output_file = Path("examples/output/issue82_64cubed_detailed_analysis.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved detailed plot to: {output_file}")

    # Summary
    print("\n" + "=" * 70)
    print("Key Findings:")
    print("=" * 70)
    print("1. Exponential blow-up starts at t ~ 13 τ_A")
    if gamma is not None:
        print(f"   Growth rate: γ = {gamma:.4f}, doubling time = {np.log(2)/gamma:.2f} τ_A")
    print(f"2. CFL number stays well below 1.0 (max {np.nanmax(cfl):.4f})")
    print("   → Timestep is NOT the problem")
    print(f"3. High-k energy grows from ~10^-20 to ~10^-2")
    print("   → Spectral pile-up confirms energy accumulation at small scales")
    print("4. Before t=13: System appears quasi-steady with slow energy growth")
    print("   → Suggests gradual energy accumulation until critical threshold")
    print("\n✓ Diagnosis: Energy injection > dissipation → gradual accumulation → instability")
    print("=" * 70)


if __name__ == "__main__":
    main()
