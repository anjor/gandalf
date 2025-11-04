#!/usr/bin/env python3
"""
Pure Fluid Orszag-Tang Vortex: Energy Conservation Benchmark

Tests the GANDALF energy-conserving formulation for pure fluid (non-kinetic) RMHD.
With η=0, ν=0, M=0, energy should be conserved to ~0.01% (machine precision).

This benchmark verifies:
- Exact energy conservation in inviscid limit (GANDALF integrating factor method)
- Kinetic ↔ magnetic energy exchange (Alfvénic dynamics)
- Nonlinear MHD dynamics without kinetic effects

Original Orszag-Tang (Compressible MHD):
    - Velocity: Vx = -sin(y), Vy = sin(x)
    - Magnetic: Bx = -B0·sin(y), By = B0·sin(2x), B0 = 1/√(4π)

This Pure Fluid RMHD Version:
    - Stream function: φ = -(cos(x) + cos(y)) → v⊥ = ẑ × ∇φ
    - Vector potential: A∥ = 2B0·(cos(2x) + 2cos(y)) → B⊥ = ẑ × ∇A∥
    - Hermite moments: M=0 (pure fluid, no kinetic physics)
    - Dissipation: η=0, ν=0 (inviscid/collisionless for energy conservation test)

Reference: GANDALF energy-conserving formulation (Issue #44)
Expected: |ΔE/E₀| < 0.01% over t ∈ [0, 2.0]
Runtime: ~5-10 seconds on M1 Pro for 32² resolution
"""

import numpy as np
from krmhd import (
    SpectralGrid3D,
    initialize_orszag_tang,
    gandalf_step,
    compute_cfl_timestep,
    energy as compute_energy,
)
from krmhd.diagnostics import EnergyHistory

print("=" * 70)
print("Pure Fluid Orszag-Tang: Energy Conservation Benchmark")
print("=" * 70)

# Grid resolution (2D problem - minimal for pure perpendicular physics)
Nx, Ny, Nz = 32, 32, 2  # 32² × 2 (Nz=2 minimum for 3D code, but physics is pure 2D)
Lx = Ly = 2 * np.pi      # Match original Orszag-Tang domain
Lz = 2 * np.pi           # Arbitrary for 2D problem

# Physics parameters (INVISCID for energy conservation test)
B0 = 1.0 / np.sqrt(4 * np.pi)  # Magnetic field amplitude (~0.282)
v_A = 1.0           # Alfvén velocity
eta = 0.0           # NO resistivity (inviscid test)
nu = 0.0            # NO collisions (collisionless test)
cfl_safety = 0.3    # CFL safety factor

# Time evolution (reference runs to t = 2.0 τ_A)
# Note: Start with t=1.0 for testing, increase to t=2.0 for full benchmark
t_final = 1.0       # One Alfvén time (for testing)
save_interval = 0.1  # Save diagnostics every 0.1 time units

print(f"\nGrid: {Nx} × {Ny} (2D)")
print(f"Domain: {Lx:.2f} × {Ly:.2f}")
print(f"Physics: B0={B0:.3f}, v_A={v_A}, η={eta}")
print(f"Evolution: t ∈ [0, {t_final}]")

# ============================================================================
# Initialize Grid and State
# ============================================================================

print("\nInitializing...")
grid = SpectralGrid3D.create(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz)

# Initialize Orszag-Tang vortex: PURE FLUID (M=0, nu=0.0 defaults)
state = initialize_orszag_tang(grid=grid, B0=B0)

E0 = compute_energy(state)

print(f"✓ Initialized Pure Fluid Orszag-Tang")
print(f"  Mode: M=0 (pure fluid), η=0, ν=0 (inviscid)")
print(f"  Initial energy: E_total = {E0['total']:.6e}")
print(f"  E_mag/E_kin = {E0['magnetic']/E0['kinetic']:.3f}")
print(f"  Grid: {Nx} × {Ny} × {Nz}, domain: {Lx:.2f} × {Ly:.2f}")
print(f"  Grid spacing: dx = {Lx/Nx:.4f}, dy = {Ly/Ny:.4f}")

# ============================================================================
# Time Evolution
# ============================================================================

print("\nTime evolution...")
print("  (First timestep may take ~30s for JAX compilation)")

history = EnergyHistory()
next_save_time = 0.0

step_count = 0
while state.time < t_final:
    if state.time >= next_save_time:
        history.append(state)
        E = history.E_total[-1]
        mag_frac = history.E_magnetic[-1] / max(history.E_kinetic[-1], 1e-10)
        print(f"  t = {state.time:5.3f}, E = {E:.6e}, E_mag/E_kin = {mag_frac:.3f}, steps = {step_count}")
        next_save_time += save_interval

    dt = compute_cfl_timestep(state, v_A, cfl_safety)
    dt = min(dt, t_final - state.time, next_save_time - state.time)

    # Debug: print if timestep becomes very small
    if dt < 1e-4:
        print(f"  WARNING: Small timestep dt = {dt:.2e} at t = {state.time:.3f}")

    state = gandalf_step(state, dt, eta, v_A)
    step_count += 1

    # Safety: abort if too many steps
    if step_count > 10000:
        print(f"  ERROR: Too many steps ({step_count}), aborting!")
        break

# Final snapshot
history.append(state)

print(f"✓ Completed evolution to t = {state.time:.3f}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

E_initial = history.E_total[0]
E_final = history.E_total[-1]
mag_frac_initial = history.E_magnetic[0] / E_initial
mag_frac_final = history.E_magnetic[-1] / E_final

# Energy conservation check
energy_error = abs(E_final - E_initial) / E_initial
print(f"\nEnergy Conservation (Inviscid Test):")
print(f"  E(t=0)   = {E_initial:.6e}")
print(f"  E(t={t_final}) = {E_final:.6e}")
print(f"  ΔE/E₀    = {energy_error:.2%}")
print(f"  Expected: < 0.01% (GANDALF guarantee from Issue #44)")

if energy_error < 0.0001:  # < 0.01%
    print(f"  ✓ PASS: Energy conserved to {energy_error:.4%}")
elif energy_error < 0.001:  # < 0.1%
    print(f"  ~ ACCEPTABLE: Energy conserved to {energy_error:.3%}")
else:
    print(f"  ✗ FAIL: Energy error {energy_error:.2%} exceeds tolerance")

# Kinetic/magnetic energy exchange
print(f"\nKinetic ↔ Magnetic Energy Exchange:")
print(f"  E_mag/E_total (t=0)     = {mag_frac_initial:.3f}")
print(f"  E_mag/E_total (t={t_final}) = {mag_frac_final:.3f}")

if len(history.E_magnetic) > 1:
    E_mag_array = np.array(history.E_magnetic)
    E_mag_mean = np.mean(E_mag_array)
    E_mag_std = np.std(E_mag_array)
    oscillation_amplitude = E_mag_std / E_mag_mean
    print(f"  Magnetic oscillation amplitude: {oscillation_amplitude:.1%} of mean")
    if oscillation_amplitude > 0.05:
        print(f"  ✓ Energy exchange detected (Alfvénic dynamics)")

print(f"\nTime evolution:")
print(f"  Number of timesteps: {len(history.times)}")
print(f"  Final time: t = {state.time:.3f}")

print("\n" + "=" * 70)
print("✓ Pure Fluid Orszag-Tang Benchmark Complete")
print("=" * 70)

# ============================================================================
# Visualization
# ============================================================================

print("\nGenerating plots...")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot 1: Energy components vs time
ax1 = axes[0]
times = np.array(history.times)
E_total = np.array(history.E_total)
E_kinetic = np.array(history.E_kinetic)
E_magnetic = np.array(history.E_magnetic)

ax1.plot(times, E_total, 'k-', linewidth=2, label='Total Energy')
ax1.plot(times, E_kinetic, 'r--', linewidth=2, label='Kinetic Energy')
ax1.plot(times, E_magnetic, 'b:', linewidth=2, label='Magnetic Energy')
ax1.set_ylabel('Energy', fontsize=12)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title('Pure Fluid Orszag-Tang: Energy Conservation Test', fontsize=14, fontweight='bold')

# Plot 2: Energy conservation error
ax2 = axes[1]
E_error = np.abs(E_total - E_total[0]) / E_total[0] * 100  # Percentage
ax2.plot(times, E_error, 'k-', linewidth=2)
ax2.axhline(y=0.01, color='g', linestyle='--', linewidth=1, label='Target: 0.01%')
ax2.axhline(y=0.1, color='orange', linestyle='--', linewidth=1, label='Acceptable: 0.1%')
ax2.set_xlabel('Time (Alfvén times)', fontsize=12)
ax2.set_ylabel('|ΔE/E₀| (%)', fontsize=12)
ax2.set_yscale('log')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_title('Energy Conservation Error', fontsize=12)

plt.tight_layout()
plt.savefig('orszag_tang_energy_conservation.png', dpi=150, bbox_inches='tight')
print("  ✓ Saved: orszag_tang_energy_conservation.png")

plt.show()
print("\nPlots displayed. Close window to exit.")
