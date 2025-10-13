#!/usr/bin/env python3
"""
Quick performance benchmark for timestepping.

Tests JIT compilation performance on 128³ grid.
"""

import time
import jax
import jax.numpy as jnp

from krmhd.spectral import SpectralGrid3D
from krmhd.physics import initialize_alfven_wave
from krmhd.timestepping import rk4_step

# Force compilation before timing
jax.clear_caches()

# Create 128³ grid
print("Creating 128³ grid...")
grid = SpectralGrid3D.create(Nx=128, Ny=128, Nz=128)
state = initialize_alfven_wave(grid, M=20, kz_mode=1, amplitude=0.1)

print(f"Grid shape: {state.z_plus.shape}")
print(f"Memory per field: {state.z_plus.nbytes / 1e6:.1f} MB")

# Warm-up: trigger JIT compilation
print("\nWarming up (JIT compilation)...")
t0 = time.time()
state_warm = rk4_step(state, dt=0.01, eta=0.01, v_A=1.0)
t_warmup = time.time() - t0
print(f"First call (includes compilation): {t_warmup:.3f} s")

# Now time 10 compiled steps
print("\nTiming 10 timesteps (JIT-compiled)...")
t0 = time.time()
for i in range(10):
    state = rk4_step(state, dt=0.01, eta=0.01, v_A=1.0)
t_total = time.time() - t0

print(f"\nResults:")
print(f"Total time for 10 steps: {t_total:.3f} s")
print(f"Time per step: {t_total/10:.3f} s")
print(f"Steps per second: {10/t_total:.2f}")
print(f"\nFor 1000 steps: ~{t_total*100:.1f} s = ~{t_total*100/60:.1f} min")
print(f"For 100K steps: ~{t_total*10000/3600:.1f} hours")
