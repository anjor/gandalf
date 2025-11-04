#!/usr/bin/env python3
"""Debug Orszag-Tang to find the bug causing CFL collapse."""

import numpy as np
import jax.numpy as jnp
from krmhd import SpectralGrid3D, initialize_orszag_tang, gandalf_step, compute_cfl_timestep, energy as compute_energy

print("Debugging Orszag-Tang CFL collapse...")

# Minimal setup
grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=2, Lx=2*np.pi, Ly=2*np.pi, Lz=2*np.pi)
B0 = 1.0 / np.sqrt(4 * np.pi)
state = initialize_orszag_tang(grid=grid, B0=B0)

v_A = 1.0
eta = 0.0

# Track diagnostics
t_vals = []
E_vals = []
dt_vals = []
vmax_vals = []

from krmhd.spectral import rfftn_inverse, derivative_x, derivative_y
from krmhd.physics import elsasser_to_physical

t_target = 0.8
while state.time < t_target:
    # Diagnostics
    E = compute_energy(state)
    dt = compute_cfl_timestep(state, v_A, 0.3)

    # Check velocity
    phi, _ = elsasser_to_physical(state.z_plus, state.z_minus)
    dphi_dx = derivative_x(phi, grid.kx)
    dphi_dy = derivative_y(phi, grid.ky)
    dphi_dx_real = rfftn_inverse(dphi_dx, grid.Nz, grid.Ny, grid.Nx)
    dphi_dy_real = rfftn_inverse(dphi_dy, grid.Nz, grid.Ny, grid.Nx)
    v_max = float(jnp.sqrt(dphi_dx_real**2 + dphi_dy_real**2).max())

    t_vals.append(float(state.time))
    E_vals.append(E['total'])
    dt_vals.append(dt)
    vmax_vals.append(v_max)

    if state.time > 0.6:
        print(f"t={state.time:.4f}: E={E['total']:.4e}, dt={dt:.2e}, v_max={v_max:.4e}")

    if dt < 1e-6:
        print(f"\n**CFL COLLAPSE at t={state.time:.4f}**")
        print(f"  v_max = {v_max:.4e}")
        print(f"  dt = {dt:.2e}")
        print(f"  Energy growth: {(E['total']/E_vals[0] - 1)*100:.2f}%")
        break

    state = gandalf_step(state, dt, eta, v_A)

# Plot
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

axes[0].semilogy(t_vals, E_vals, 'k-')
axes[0].set_ylabel('Energy')
axes[0].grid(True)

axes[1].semilogy(t_vals, vmax_vals, 'r-')
axes[1].set_ylabel('v_max')
axes[1].grid(True)

axes[2].semilogy(t_vals, dt_vals, 'b-')
axes[2].set_ylabel('dt')
axes[2].set_xlabel('Time')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('debug_orszag.png', dpi=150)
print("\n✓ Saved debug_orszag.png")
