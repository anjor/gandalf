#!/usr/bin/env python3
"""
Incompressible Orszag-Tang Vortex for RMHD

Adaptation of the classic compressible Orszag-Tang vortex test to incompressible
Reduced MHD (RMHD). Tests nonlinear dynamics, energy cascade, and current sheet
formation in the incompressible limit.

Original Orszag-Tang (Compressible MHD):
    - Velocity: Vx = -sin(y), Vy = sin(x)
    - Magnetic: Bx = -B0·sin(y), By = B0·sin(2x), B0 = 1/√(4π)
    - Shows shock formation and complex dynamics

This Adaptation (Incompressible RMHD):
    - Stream function: φ = [cos(x) + cos(y)]/(2π) → v⊥ = ẑ × ∇φ
    - Vector potential: A∥ = B0·[cos(2x)/(4π) + cos(y)]/(2π) → B⊥ = ẑ × ∇A∥
    - Incompressible: no shocks, but current sheets and nonlinear cascade develop

Runtime: ~1-2 minutes on M1 Pro for 64² resolution
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
print("Incompressible Orszag-Tang Vortex (RMHD)")
print("=" * 70)

# Grid resolution (2D problem - use Nz=2 minimum for 3D code)
# Use 64² for reasonable speed, 128² for better resolution
Nx, Ny, Nz = 64, 64, 2
Lx = Ly = 2 * np.pi  # Match original Orszag-Tang domain
Lz = 2 * np.pi       # Arbitrary for 2D problem

# Physics parameters
B0 = 1.0 / np.sqrt(4 * np.pi)  # Magnetic field amplitude (~0.282)
v_A = 1.0           # Alfvén velocity
eta = 0.001         # Resistivity (small for quasi-ideal evolution)
cfl_safety = 0.3    # CFL safety factor

# Time evolution
t_final = 1.0       # Standard benchmark time
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

# Initialize Orszag-Tang vortex using shared function
state = initialize_orszag_tang(
    grid=grid,
    M=10,  # Number of Hermite moments (sufficient for fluid limit)
    B0=B0,
    v_th=1.0,
    beta_i=1.0,
    nu=0.01,
    Lambda=1.0,
)

E0 = compute_energy(state)
print(f"✓ Initialized Orszag-Tang vortex")
print(f"  Initial energy: E_total = {E0['total']:.6e}")
print(f"  E_mag/E_kin = {E0['magnetic']/E0['kinetic']:.3f}")
print(f"  Grid: {Nx} × {Ny}, domain: {Lx:.2f} × {Ly:.2f}")
print(f"  Grid spacing: dx = {Lx/Nx:.4f}, dy = {Ly/Ny:.4f}")

# ============================================================================
# Time Evolution
# ============================================================================

print("\nTime evolution...")
print("  (First timestep may take ~30s for JAX compilation)")

history = EnergyHistory()
next_save_time = 0.0

while state.time < t_final:
    if state.time >= next_save_time:
        history.append(state)
        E = history.E_total[-1]
        mag_frac = history.E_magnetic[-1] / max(history.E_kinetic[-1], 1e-10)
        print(f"  t = {state.time:5.3f}, E = {E:.6e}, E_mag/E_kin = {mag_frac:.3f}")
        next_save_time += save_interval

    dt = compute_cfl_timestep(state, v_A, cfl_safety)
    dt = min(dt, t_final - state.time, next_save_time - state.time)
    state = gandalf_step(state, dt, eta, v_A)

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

print(f"\nEnergy evolution:")
print(f"  E(t=0)   = {E_initial:.6e}")
print(f"  E(t={t_final}) = {E_final:.6e}")
print(f"  E_final / E_initial = {E_final/E_initial:.4f}")

print(f"\nMagnetic fraction:")
print(f"  E_mag/E_total (t=0)   = {mag_frac_initial:.3f}")
print(f"  E_mag/E_total (t={t_final}) = {mag_frac_final:.3f}")

if len(history.E_total) > 1:
    dE = np.abs(np.diff(history.E_total))
    peak_idx = np.argmax(dE)
    print(f"\nDynamics:")
    print(f"  Peak energy change at t ≈ {history.times[peak_idx]:.3f}")
    print(f"  Number of time snapshots: {len(history.times)}")

print("\n" + "=" * 70)
print("✓ Incompressible Orszag-Tang simulation complete!")
print("=" * 70)
print("\nNote: This is an INCOMPRESSIBLE adaptation. No shocks form,")
print("but current sheets and nonlinear energy cascade develop similarly.")
print("=" * 70)
