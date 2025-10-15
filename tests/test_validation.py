"""
Validation tests for KRMHD physics benchmarks.

These tests validate the physical correctness of the implementation
against known analytical solutions and standard benchmarks:
- Orszag-Tang vortex: nonlinear dynamics and energy conservation
- Linear wave dispersion: Alfvén waves, kinetic Alfvén waves
- Landau damping: kinetic response validation
"""

import numpy as np
import jax.numpy as jnp
import pytest

from krmhd import (
    SpectralGrid3D,
    initialize_orszag_tang,
    gandalf_step,
    compute_cfl_timestep,
    energy as compute_energy,
)


class TestOrszagTangVortex:
    """
    Tests for Orszag-Tang vortex benchmark.

    The Orszag-Tang vortex is a standard test for nonlinear MHD codes,
    testing complex dynamics, energy cascade, and current sheet formation.
    """

    def test_energy_conservation_short_time(self):
        """
        Test that Orszag-Tang conserves energy over short time with small dissipation.

        Runs a short simulation (t=0.1) and verifies:
        - Energy decay < 1% (with η=0.001 small dissipation)
        - Magnetic fraction increases (selective decay)
        - No NaN/Inf in fields
        """
        # Small grid for fast testing
        Nx, Ny, Nz = 32, 32, 2
        Lx = Ly = 2 * np.pi
        Lz = 2 * np.pi
        B0 = 1.0 / np.sqrt(4 * np.pi)
        v_A = 1.0
        eta = 0.001  # Small dissipation
        cfl_safety = 0.3
        t_final = 0.1  # Short time for CI

        # Initialize grid and state using shared function
        grid = SpectralGrid3D.create(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz)
        state = initialize_orszag_tang(
            grid=grid,
            M=10,
            B0=B0,
            v_th=1.0,
            beta_i=1.0,
            nu=0.01,
            Lambda=1.0,
        )

        # Record initial energy
        E_initial = compute_energy(state)
        mag_frac_initial = E_initial['magnetic'] / E_initial['kinetic']

        # Evolve to t_final
        while state.time < t_final:
            dt = compute_cfl_timestep(state, v_A, cfl_safety)
            dt = min(dt, t_final - state.time)
            state = gandalf_step(state, dt, eta, v_A)

        # Record final energy
        E_final = compute_energy(state)
        mag_frac_final = E_final['magnetic'] / E_final['kinetic']

        # Assertions
        # 1. Energy should be conserved to within 1% (small dissipation)
        energy_ratio = E_final['total'] / E_initial['total']
        assert 0.99 < energy_ratio <= 1.0, \
            f"Energy not conserved: E_final/E_initial = {energy_ratio:.4f}"

        # 2. Magnetic fraction should increase (selective decay)
        assert mag_frac_final > mag_frac_initial, \
            f"Magnetic fraction should increase, got {mag_frac_initial:.3f} → {mag_frac_final:.3f}"

        # 3. No NaN or Inf in fields
        assert jnp.all(jnp.isfinite(state.z_plus)), "NaN/Inf in z_plus"
        assert jnp.all(jnp.isfinite(state.z_minus)), "NaN/Inf in z_minus"
        assert jnp.all(jnp.isfinite(state.g)), "NaN/Inf in Hermite moments"

        # 4. Total energy should be positive
        assert E_final['total'] > 0, "Total energy should be positive"
        assert E_final['magnetic'] > 0, "Magnetic energy should be positive"
        assert E_final['kinetic'] > 0, "Kinetic energy should be positive"

    def test_initial_conditions_reality(self):
        """
        Test that Orszag-Tang initial conditions satisfy reality condition.

        Fourier coefficients must satisfy f(-k) = f*(k) for real fields.
        """
        # Minimal grid
        Nx, Ny, Nz = 16, 16, 2
        Lx = Ly = Lz = 2 * np.pi
        B0 = 1.0 / np.sqrt(4 * np.pi)

        grid = SpectralGrid3D.create(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz)
        state = initialize_orszag_tang(grid=grid, M=10, B0=B0)

        # Extract Elsasser fields
        phi_k = (state.z_plus + state.z_minus) / 2
        A_parallel_k = (state.z_plus - state.z_minus) / 2

        # Check reality condition for ky=0 modes (should be real)
        # For rfft, the ky=0 plane should have f(kz, 0, kx) real when kx=0
        assert jnp.abs(jnp.imag(phi_k[:, 0, 0])).max() < 1e-10, \
            "k=0 mode should be real for phi"
        assert jnp.abs(jnp.imag(A_parallel_k[:, 0, 0])).max() < 1e-10, \
            "k=0 mode should be real for A_parallel"

    def test_initial_energy_components(self):
        """Test that initial energy components have reasonable magnitudes."""
        Nx, Ny, Nz = 32, 32, 2
        Lx = Ly = Lz = 2 * np.pi
        B0 = 1.0 / np.sqrt(4 * np.pi)

        grid = SpectralGrid3D.create(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz)
        state = initialize_orszag_tang(
            grid=grid,
            M=10,
            B0=B0,
            v_th=1.0,
            beta_i=1.0,
            nu=0.01,
            Lambda=1.0,
        )

        E = compute_energy(state)

        # All energy components should be positive
        assert E['total'] > 0, "Total energy must be positive"
        assert E['magnetic'] > 0, "Magnetic energy must be positive"
        assert E['kinetic'] > 0, "Kinetic energy must be positive"

        # Total energy should equal sum of components
        E_sum = E['magnetic'] + E['kinetic'] + E['compressive']
        assert abs(E['total'] - E_sum) / E['total'] < 1e-10, \
            "Total energy should equal sum of components"

        # Kinetic energy should dominate initially (typical for Orszag-Tang)
        assert E['kinetic'] > E['magnetic'], \
            "Kinetic energy should dominate initially"
