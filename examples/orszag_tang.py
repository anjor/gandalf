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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import jax.numpy as jnp
from krmhd import (
    SpectralGrid3D,
    KRMHDState,
    gandalf_step,
    compute_cfl_timestep,
    energy as compute_energy,
)
from krmhd.spectral import rfftn_forward
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

# Create coordinate arrays (shape: Nz, Ny, Nx)
x = jnp.linspace(0, grid.Lx, grid.Nx, endpoint=False)
y = jnp.linspace(0, grid.Ly, grid.Ny, endpoint=False)
z = jnp.linspace(-grid.Lz / 2, grid.Lz / 2, grid.Nz, endpoint=False)
Z, Y, X = jnp.meshgrid(z, y, x, indexing="ij")

# Orszag-Tang initial conditions (incompressible version)
# φ generates v⊥ = (-sin(y), sin(x))/(2π) via v⊥ = ẑ × ∇φ
phi_real = (jnp.cos(2 * np.pi * X / Lx) + jnp.cos(2 * np.pi * Y / Ly)) / (2 * np.pi)

# A∥ generates B⊥ = B0·(-sin(y), sin(2x))/(2π) via B⊥ = ẑ × ∇A∥
# Note: factor of 2 difference in x-wavenumber (sin(2x) vs sin(x))
A_parallel_real = B0 * (
    jnp.cos(4 * np.pi * X / Lx) / (4 * np.pi) +
    jnp.cos(2 * np.pi * Y / Ly) / (2 * np.pi)
)

# Transform to Fourier space
phi_k = rfftn_forward(phi_real)
A_parallel_k = rfftn_forward(A_parallel_real)

# Create KRMHD state with Elsasser variables z± = φ ± A∥
state = KRMHDState(
    z_plus=phi_k + A_parallel_k,
    z_minus=phi_k - A_parallel_k,
    B_parallel=jnp.zeros_like(phi_k),  # Pure Alfvén mode
    g=jnp.zeros((Nz, Ny, Nx // 2 + 1, 11), dtype=complex),  # Minimal kinetics
    M=10,
    beta_i=1.0,
    v_th=1.0,
    nu=0.01,
    Lambda=1.0,
    time=0.0,
    grid=grid,
)

E0 = compute_energy(state)
print(f"✓ Initialized Orszag-Tang vortex")
print(f"  Initial energy: E_total = {E0['total']:.6e}")
print(f"  E_mag/E_kin = {E0['magnetic']/E0['kinetic']:.3f}")

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
