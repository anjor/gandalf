#!/usr/bin/env python3
"""
Minimal Forcing Example

This is a minimal (~50 line) example showing basic forcing usage.
For comprehensive demonstration with diagnostics and visualization,
see driven_turbulence.py.

Runtime: ~2 seconds on M1 Pro
"""

import jax
from krmhd import (
    SpectralGrid3D,
    initialize_alfven_wave,
    gandalf_step,
    force_alfven_modes,
    compute_energy_injection_rate,
    energy as compute_energy,
)


def main():
    """Minimal forcing example."""

    # Create grid and initial state
    grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
    state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.01)

    # Initialize random key for forcing
    key = jax.random.PRNGKey(42)

    # Physics parameters
    dt = 0.01
    eta = 0.01
    v_A = 1.0

    print(f"Initial energy: {compute_energy(state)['total']:.4e}")

    # Run 10 steps with forcing
    for i in range(10):
        # Apply forcing at large scales (modes n=1-2)
        state_before = state
        state, key = force_alfven_modes(
            state,
            amplitude=0.1,
            n_min=1,
            n_max=2,
            dt=dt,
            key=key
        )

        # Measure energy injection
        eps_inj = compute_energy_injection_rate(state_before, state, dt)

        # Evolve dynamics (cascade + dissipation)
        state = gandalf_step(state, dt=dt, eta=eta, v_A=v_A)

        # Print progress
        E = compute_energy(state)['total']
        print(f"Step {i+1:2d}: E = {E:.4e}, ε_inj = {eps_inj:.3e}")

    print(f"\nFinal energy: {compute_energy(state)['total']:.4e}")


if __name__ == "__main__":
    main()
