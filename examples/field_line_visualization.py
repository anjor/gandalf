#!/usr/bin/env python3
"""
Field Line Following and Visualization Example

This example demonstrates:
1. Initializing a turbulent RMHD state
2. Tracing magnetic field lines through 3D domain
3. Visualizing field line wandering due to turbulent δB⊥
4. Comparing E(k∥) spectrum along field lines vs simple E(kz)

Physics:
    Field lines in turbulent plasmas wander perpendicular to the mean field
    due to perpendicular magnetic perturbations δB⊥. The wandering amplitude
    scales as δr⊥ ~ (δB⊥/B₀) × Lz.

    For passive scalars (slow modes, Hermite moments), structures that are
    constant along curved field lines have k∥ = 0 in field-line coordinates,
    but appear to have kz ≠ 0 in Cartesian coordinates due to field line
    curvature.

Expected results:
    - Weak turbulence (δB⊥/B₀ << 1): Nearly straight field lines
    - Strong turbulence (δB⊥/B₀ ~ 1): Significant wandering
    - E(k∥) ≠ E(kz) when field lines are curved

Runtime: ~30 seconds on M1 Pro for 64³ resolution with 10 field lines
"""

import numpy as np
import jax.numpy as jnp
from pathlib import Path

from krmhd.spectral import SpectralGrid3D
from krmhd.physics import initialize_random_spectrum
from krmhd.diagnostics import (
    follow_field_line,
    plot_field_lines,
    plot_parallel_spectrum_comparison,
    energy_spectrum_parallel,
    compute_energy,
)

# ==========================================================================
# Configuration
# ==========================================================================

# Grid setup
Nx, Ny, Nz = 64, 64, 64  # Resolution (use 64³ for fast demo, 128³ for better quality)
Lx, Ly, Lz = 2 * np.pi, 2 * np.pi, 2 * np.pi  # Domain size

# Physics parameters
alpha = 5.0 / 3.0  # Spectral slope (Kolmogorov)
M = 20  # Number of Hermite moments
beta_i = 1.0  # Ion plasma beta
nu = 0.01  # Collision frequency

# Field line parameters
n_fieldlines = 10  # Number of field lines to trace (10 for fast, 20 for detailed)
padding_factor = 2  # Spectral interpolation resolution (2 is good balance)

print("=" * 70)
print("Field Line Following and Visualization Example")
print("=" * 70)
print(f"Grid: {Nx} × {Ny} × {Nz}")
print(f"Domain: {Lx:.2f} × {Ly:.2f} × {Lz:.2f}")
print(f"Field lines: {n_fieldlines}")
print(f"Padding factor: {padding_factor}×")
print("=" * 70)

# ==========================================================================
# Initialize State
# ==========================================================================

print("\nInitializing turbulent state...")

# Create spectral grid
grid = SpectralGrid3D.create(
    Nx=Nx,
    Ny=Ny,
    Nz=Nz,
    Lx=Lx,
    Ly=Ly,
    Lz=Lz,
)
print(f"✓ Created {Nx}×{Ny}×{Nz} spectral grid")

# Initialize random turbulent spectrum
state = initialize_random_spectrum(
    grid=grid,
    M=M,
    alpha=alpha,
    amplitude=1.0,
    k_min=1.0,
    k_max=grid.Nx // 3,  # 2/3 dealiasing limit
    beta_i=beta_i,
    nu=nu,
    Lambda=1.0,
    seed=42,
)
print(f"✓ Initialized random k^(-{alpha:.2f}) spectrum")

# Compute initial energy
initial_energies = compute_energy(state)
print(f"  Initial energy: E_total = {initial_energies['total']:.6e}")

# Compute field strength for turbulence assessment
phi = (state.z_plus + state.z_minus) / 2.0
A_parallel = (state.z_plus - state.z_minus) / 2.0

# Estimate δB⊥/B₀ from A_parallel (rough approximation)
# In normalized units, B₀ ~ 1, and δB⊥ ~ k⊥ |A∥|
# For rough estimate: δB⊥/B₀ ~ sqrt(<|A∥|²>)
A_rms = float(jnp.sqrt(jnp.mean(jnp.abs(A_parallel) ** 2)))
print(f"  Estimated δB⊥/B₀ ~ {A_rms:.3f} (turbulence strength)")
if A_rms < 0.1:
    print(f"  → Weak turbulence: field lines should be nearly straight")
elif A_rms < 0.5:
    print(f"  → Moderate turbulence: some field line wandering expected")
else:
    print(f"  → Strong turbulence: significant field line wandering expected")

# ==========================================================================
# Field Line Tracing
# ==========================================================================

print("\n" + "-" * 70)
print("Tracing field lines...")

# Sample a few field lines and print statistics
x_starts = np.linspace(0, Lx, n_fieldlines, endpoint=False)
y_starts = np.linspace(0, Ly, n_fieldlines, endpoint=False)

print(f"  Tracing {n_fieldlines} field lines from z = {-Lz/2:.2f} to z = {Lz/2:.2f}")

wandering_distances = []
for i in range(min(5, n_fieldlines)):  # Just show first 5
    x0, y0 = x_starts[i], y_starts[i]
    trajectory = follow_field_line(state, x0, y0, padding_factor=padding_factor)

    # Compute wandering distance
    x_traj = np.array(trajectory[:, 0])
    y_traj = np.array(trajectory[:, 1])

    # Perpendicular displacement from start
    dx = x_traj - x0
    dy = y_traj - y0
    wandering = np.sqrt(dx**2 + dy**2)
    max_wandering = np.max(wandering)
    wandering_distances.append(max_wandering)

    print(f"    Field line {i+1}: max wandering = {max_wandering:.4f} (in units of Lx)")

avg_wandering = np.mean(wandering_distances)
print(f"  Average wandering: {avg_wandering:.4f}")
print(f"  Theory estimate: δr⊥ ~ {A_rms * Lz:.4f}")

# ==========================================================================
# Visualization
# ==========================================================================

print("\n" + "-" * 70)
print("Creating visualizations...")

# Create output directory
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# 1. Field line trajectories
print("  Plotting field line trajectories...")
plot_field_lines(
    state,
    n_lines=n_fieldlines,
    padding_factor=padding_factor,
    filename=str(output_dir / "field_line_trajectories.png"),
    show=False,
)
print(f"    ✓ Saved to {output_dir / 'field_line_trajectories.png'}")

# 2. Parallel spectrum comparison
print("  Plotting parallel spectrum comparison...")
plot_parallel_spectrum_comparison(
    state,
    n_fieldlines=n_fieldlines,
    padding_factor=padding_factor,
    filename=str(output_dir / "parallel_spectrum_comparison.png"),
    show=False,
)
print(f"    ✓ Saved to {output_dir / 'parallel_spectrum_comparison.png'}")

# 3. Print spectrum statistics
print("\n" + "-" * 70)
print("Parallel spectrum statistics:")

kz, E_kz = energy_spectrum_parallel(state)
kz_np = np.array(kz)
E_kz_np = np.array(E_kz)

# Find peak and compute spectrum width
valid = (kz_np > 0) & (E_kz_np > 0)
if np.any(valid):
    k_peak_idx = np.argmax(E_kz_np[valid])
    k_peak = kz_np[valid][k_peak_idx]
    E_peak = E_kz_np[valid][k_peak_idx]
    print(f"  Peak wavenumber: kz = {k_peak:.3f}")
    print(f"  Peak energy: E(kz) = {E_peak:.3e}")

    # Compute effective width (where E > 0.1 * E_peak)
    width_mask = E_kz_np[valid] > 0.1 * E_peak
    if np.any(width_mask):
        k_width = np.max(kz_np[valid][width_mask]) - np.min(kz_np[valid][width_mask])
        print(f"  Spectrum width (E > 0.1 E_peak): Δkz = {k_width:.3f}")

# ==========================================================================
# Summary
# ==========================================================================

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print(f"✓ Traced {n_fieldlines} field lines through turbulent state")
print(f"✓ Average perpendicular wandering: {avg_wandering:.4f} Lx")
print(f"✓ Turbulence strength: δB⊥/B₀ ~ {A_rms:.3f}")
print(f"✓ Generated visualizations in {output_dir}/")
print("=" * 70)

print("\nFiles created:")
print(f"  - {output_dir / 'field_line_trajectories.png'}")
print(f"  - {output_dir / 'parallel_spectrum_comparison.png'}")

print("\nNext steps:")
print("  - Adjust turbulence strength by changing amplitude or alpha")
print("  - Increase n_fieldlines for better statistics")
print("  - Compare different time snapshots to see field line evolution")
print("  - Use true k∥ spectrum diagnostic when implemented (Issue #25, #26, #27)")

print("\n" + "=" * 70)
