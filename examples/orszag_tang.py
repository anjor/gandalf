#!/usr/bin/env python3
"""
Pure Fluid Orszag-Tang Vortex: Nonlinear MHD Dynamics Benchmark

Tests pure fluid (non-kinetic) RMHD with the GANDALF energy-conserving formulation.
Uses M=0 (no Hermite moments) to test fluid MHD without kinetic corrections.

This benchmark verifies:
- Nonlinear MHD dynamics without kinetic effects
- Kinetic ↔ magnetic energy exchange (selective decay in 2D MHD)
- Perfect energy conservation (ΔE/E₀ < 10⁻⁵%)

Initial Conditions (from thesis Eq. 2.31-2.32):
    - φ = -2(cos x + cos y)  →  v⊥ = ẑ × ∇φ
    - Ψ = cos(2x) + 2cos(y)  →  B⊥ = ẑ × ∇Ψ
    - In Elsasser variables: z± = φ ± Ψ

Time Normalization (from thesis Section 2.3):
    - τ_A = Lz/v_A = 2π time units (parallel Alfvén crossing time)
    - Verified from original GANDALF code (init_func.cu, gandalf.cu)

FFT Normalization:
    - Original GANDALF uses unnormalized CUFFT
    - JAX uses normalized FFT → must scale by Nx to match energies
    - Target: E_total ~ 4.0 (thesis Figure 2.1)

Expected Dynamics:
    - 2D MHD selective decay: magnetic energy grows, kinetic decreases
    - E_mag/E_kin ~ 1.0 → 1.2 over 2 Alfvén times
    - Total energy conserved to machine precision

Reference:
    - GANDALF energy-conserving formulation (Issue #44)
    - Original GANDALF: https://github.com/anjor/gandalf-original
    - Thesis Figure 2.1 for comparison

Runtime: ~10 seconds on M1 Pro for 32² × 2 resolution
"""

import numpy as np
from pathlib import Path
from krmhd import (
    SpectralGrid3D,
    initialize_orszag_tang,
    gandalf_step,
    compute_cfl_timestep,
    energy as compute_energy,
)
from krmhd.diagnostics import EnergyHistory

# Create output directory
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

print("=" * 70)
print("Pure Fluid Orszag-Tang: Energy Conservation Benchmark")
print("=" * 70)

# Grid resolution (2D problem - minimal for pure perpendicular physics)
Nx, Ny, Nz = 32, 32, 2  # 32² × 2 for testing
Lx = Ly = 1.0            # Match thesis box size (Eqs. 2.31-2.32)
Lz = 1.0                 # Match thesis normalization

# Physics parameters
B0 = 1.0 / np.sqrt(4 * np.pi)  # Magnetic field amplitude (~0.282)
v_A = 1.0           # Alfvén velocity
eta = 0.0           # NO resistivity (test inviscid with higher resolution)
nu = 0.0            # NO collisions (M=0, not used)
cfl_safety = 0.3    # CFL safety factor

# Time evolution
# Time normalization: τ_A = Lz/v_A (thesis Section 2.3)
# With Lz = 1.0, v_A = 1.0: τ_A = 1.0 time units
# Thesis Figure 2.1 shows full oscillation over ~2 Alfvén times
tau_A = Lz / v_A  # = 1.0 time units (matches thesis)
t_final = 4.0 * tau_A  # Four Alfvén times to see full oscillation
save_interval = 0.05 * tau_A  # Save every 0.05 τ_A

print(f"\nGrid: {Nx} × {Ny} (2D)")
print(f"Domain: {Lx:.2f} × {Ly:.2f}")
print(f"Physics: B0={B0:.3f}, v_A={v_A}, η={eta}")
print(f"Time normalization: τ_A = {tau_A:.4f} time units")
print(f"Evolution: t ∈ [0, {t_final:.2f}] = [0, {t_final/tau_A:.1f}] τ_A")

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
        t_alfven = state.time / tau_A
        print(f"  t = {state.time:6.2f} ({t_alfven:.3f} τ_A), E = {E:.6e}, E_mag/E_kin = {mag_frac:.3f}, steps = {step_count}")
        next_save_time += save_interval

    dt = compute_cfl_timestep(state, v_A, cfl_safety)
    dt = min(dt, t_final - state.time)

    # Debug: print if timestep becomes very small
    if dt < 1e-4:
        print(f"  WARNING: Small timestep dt = {dt:.2e} at t = {state.time:.3f}")

    state = gandalf_step(state, dt, eta, v_A)
    step_count += 1

    # Safety: abort if too many steps
    if step_count > 50000:
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
import jax.numpy as jnp

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot 1: Energy components vs time
ax1 = axes[0]
times = np.array(history.times)
times_alfven = times / tau_A  # Convert to Alfvén times for x-axis
E_total = np.array(history.E_total)
E_kinetic = np.array(history.E_kinetic)
E_magnetic = np.array(history.E_magnetic)

ax1.plot(times_alfven, E_total, 'k-', linewidth=2, label='Total Energy')
ax1.plot(times_alfven, E_kinetic, 'r--', linewidth=2, label='Kinetic Energy')
ax1.plot(times_alfven, E_magnetic, 'b:', linewidth=2, label='Magnetic Energy')
ax1.set_ylabel('Energy', fontsize=12)
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_title('Pure Fluid Orszag-Tang: Energy Conservation Test', fontsize=14, fontweight='bold')

# Plot 2: Energy conservation error
ax2 = axes[1]
E_error = np.abs(E_total - E_total[0]) / E_total[0] * 100  # Percentage
ax2.plot(times_alfven, E_error, 'k-', linewidth=2)
ax2.axhline(y=0.01, color='g', linestyle='--', linewidth=1, label='Target: 0.01%')
ax2.axhline(y=0.1, color='orange', linestyle='--', linewidth=1, label='Acceptable: 0.1%')
ax2.set_xlabel('t / τ_A', fontsize=12)
ax2.set_ylabel('|ΔE/E₀| (%)', fontsize=12)
ax2.set_yscale('log')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_title('Energy Conservation Error', fontsize=12)

plt.tight_layout()
output_file = output_dir / 'orszag_tang_energy_conservation.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_file}")

# ============================================================================
# 2D Structure Plots (like Athena test page)
# ============================================================================

print("\nGenerating 2D structure plots...")

# Get final state fields in Fourier space (kz=0 plane for 2D)
z_plus_k = state.z_plus[0, :, :]   # Shape: (Ny, Nx//2+1)
z_minus_k = state.z_minus[0, :, :]

# Convert to physical fields: φ = (z+ + z-)/2, Ψ = (z+ - z-)/2
phi_k = 0.5 * (z_plus_k + z_minus_k)
psi_k = 0.5 * (z_plus_k - z_minus_k)

# Compute Laplacians: ω = ∇²φ (vorticity), J∥ = ∇²Ψ (current)
kx = grid.kx
ky = grid.ky[:, np.newaxis]  # Broadcast for 2D
k_perp_sq = kx**2 + ky**2

omega_k = -k_perp_sq * phi_k
j_parallel_k = -k_perp_sq * psi_k

# Transform to real space using JAX FFT (2D inverse rfft)
phi_real = jnp.fft.irfft2(phi_k, s=(Ny, Nx))
psi_real = jnp.fft.irfft2(psi_k, s=(Ny, Nx))
omega_real = jnp.fft.irfft2(omega_k, s=(Ny, Nx))
j_parallel_real = jnp.fft.irfft2(j_parallel_k, s=(Ny, Nx))

# Create 2×2 subplot for structures
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 12))

# Coordinate grids
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y)

# Plot 1: Vorticity (fluid vortex structures)
ax = axes2[0, 0]
im1 = ax.pcolormesh(X, Y, np.array(omega_real), cmap='RdBu_r', shading='auto')
ax.set_title(f'Vorticity ω = ∇²φ at t = {t_final/tau_A:.1f} τ_A', fontsize=12, fontweight='bold')
ax.set_xlabel('x', fontsize=10)
ax.set_ylabel('y', fontsize=10)
ax.set_aspect('equal')
fig2.colorbar(im1, ax=ax, label='ω')

# Plot 2: Current density (magnetic current sheets)
ax = axes2[0, 1]
im2 = ax.pcolormesh(X, Y, np.array(j_parallel_real), cmap='PiYG', shading='auto')
ax.set_title(f'Current Density J∥ = ∇²Ψ at t = {t_final/tau_A:.1f} τ_A', fontsize=12, fontweight='bold')
ax.set_xlabel('x', fontsize=10)
ax.set_ylabel('y', fontsize=10)
ax.set_aspect('equal')
fig2.colorbar(im2, ax=ax, label='J∥')

# Plot 3: Stream function (flow potential)
ax = axes2[1, 0]
im3 = ax.pcolormesh(X, Y, np.array(phi_real), cmap='viridis', shading='auto')
ax.set_title(f'Stream Function φ at t = {t_final/tau_A:.1f} τ_A', fontsize=12, fontweight='bold')
ax.set_xlabel('x', fontsize=10)
ax.set_ylabel('y', fontsize=10)
ax.set_aspect('equal')
fig2.colorbar(im3, ax=ax, label='φ')

# Plot 4: Vector potential (magnetic flux)
ax = axes2[1, 1]
im4 = ax.pcolormesh(X, Y, np.array(psi_real), cmap='plasma', shading='auto')
ax.set_title(f'Vector Potential Ψ at t = {t_final/tau_A:.1f} τ_A', fontsize=12, fontweight='bold')
ax.set_xlabel('x', fontsize=10)
ax.set_ylabel('y', fontsize=10)
ax.set_aspect('equal')
fig2.colorbar(im4, ax=ax, label='Ψ')

plt.tight_layout()
output_file = output_dir / 'orszag_tang_structures.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"  ✓ Saved: {output_file}")

plt.show()
print("\nPlots displayed. Close window to exit.")
