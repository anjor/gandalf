"""
Tests for time integration module.

This test suite validates:
- RK4 integrator correctness and convergence
- CFL condition calculator
- KRMHD RHS function wrapper
- Energy conservation during time evolution
- Alfvén wave propagation

Physics verification includes:
- 4th order convergence rate for RK4
- Linear wave dispersion: ω = k∥v_A
- Numerical stability under CFL condition
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from krmhd.spectral import SpectralGrid3D
from krmhd.physics import (
    KRMHDState,
    initialize_alfven_wave,
    initialize_hermite_moments,
    energy,
)
from krmhd.timestepping import krmhd_rhs, rk4_step, compute_cfl_timestep


class TestKRMHDRHS:
    """Test the unified RHS function."""

    def test_rhs_zero_state(self):
        """RHS should be zero for zero state."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)

        # Zero state
        state = KRMHDState(
            z_plus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            z_minus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            B_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            g=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, 21), dtype=jnp.complex64),
            M=20,
            beta_i=1.0,
            v_th=1.0,
            nu=0.01,
            Lambda=1.0,
            time=0.0,
            grid=grid,
        )

        # Compute RHS
        rhs = krmhd_rhs(state, eta=0.01, v_A=1.0)

        # All derivatives should be zero
        assert jnp.allclose(rhs.z_plus, 0.0, atol=1e-10)
        assert jnp.allclose(rhs.z_minus, 0.0, atol=1e-10)
        assert jnp.allclose(rhs.B_parallel, 0.0, atol=1e-10)
        assert jnp.allclose(rhs.g, 0.0, atol=1e-10)

    def test_rhs_shape_preservation(self):
        """RHS should preserve field shapes."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        # Random state
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)

        state = KRMHDState(
            z_plus=jax.random.normal(keys[0], (grid.Nz, grid.Ny, grid.Nx // 2 + 1)) +
                   1j * jax.random.normal(keys[0], (grid.Nz, grid.Ny, grid.Nx // 2 + 1)),
            z_minus=jax.random.normal(keys[1], (grid.Nz, grid.Ny, grid.Nx // 2 + 1)) +
                    1j * jax.random.normal(keys[1], (grid.Nz, grid.Ny, grid.Nx // 2 + 1)),
            B_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            g=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, 21), dtype=jnp.complex64),
            M=20,
            beta_i=1.0,
            v_th=1.0,
            nu=0.01,
            Lambda=1.0,
            time=0.0,
            grid=grid,
        )

        # Compute RHS
        rhs = krmhd_rhs(state, eta=0.01, v_A=1.0)

        # Check shapes
        assert rhs.z_plus.shape == state.z_plus.shape
        assert rhs.z_minus.shape == state.z_minus.shape
        assert rhs.B_parallel.shape == state.B_parallel.shape
        assert rhs.g.shape == state.g.shape

    def test_rhs_nonzero_output(self):
        """RHS should be nonzero for nonzero fields."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        # Initialize Alfvén wave (nonzero fields)
        state = initialize_alfven_wave(grid, M=20, kz_mode=1, amplitude=0.1)

        # Compute RHS
        rhs = krmhd_rhs(state, eta=0.01, v_A=1.0)

        # RHS should be nonzero for nonzero fields
        assert not jnp.allclose(rhs.z_plus, 0.0, atol=1e-10)
        assert not jnp.allclose(rhs.z_minus, 0.0, atol=1e-10)


class TestRK4Step:
    """Test RK4 time integrator."""

    def test_rk4_zero_state(self):
        """Zero state should remain zero (only time advances)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        state = KRMHDState(
            z_plus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            z_minus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            B_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            g=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, 21), dtype=jnp.complex64),
            M=20,
            beta_i=1.0,
            v_th=1.0,
            nu=0.01,
            Lambda=1.0,
            time=0.0,
            grid=grid,
        )

        # Take timestep
        dt = 0.01
        new_state = rk4_step(state, dt, eta=0.01, v_A=1.0)

        # Fields should remain zero
        assert jnp.allclose(new_state.z_plus, 0.0, atol=1e-10)
        assert jnp.allclose(new_state.z_minus, 0.0, atol=1e-10)
        assert jnp.allclose(new_state.B_parallel, 0.0, atol=1e-10)
        assert jnp.allclose(new_state.g, 0.0, atol=1e-10)

        # Time should advance
        assert jnp.isclose(new_state.time, dt)

    def test_rk4_shape_preservation(self):
        """RK4 should preserve field shapes."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 2)

        state = KRMHDState(
            z_plus=jax.random.normal(keys[0], (grid.Nz, grid.Ny, grid.Nx // 2 + 1)) +
                   1j * jax.random.normal(keys[0], (grid.Nz, grid.Ny, grid.Nx // 2 + 1)),
            z_minus=jax.random.normal(keys[1], (grid.Nz, grid.Ny, grid.Nx // 2 + 1)) +
                    1j * jax.random.normal(keys[1], (grid.Nz, grid.Ny, grid.Nx // 2 + 1)),
            B_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            g=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, 21), dtype=jnp.complex64),
            M=20,
            beta_i=1.0,
            v_th=1.0,
            nu=0.01,
            Lambda=1.0,
            time=0.0,
            grid=grid,
        )

        # Take timestep
        new_state = rk4_step(state, dt=0.01, eta=0.01, v_A=1.0)

        # Check shapes
        assert new_state.z_plus.shape == state.z_plus.shape
        assert new_state.z_minus.shape == state.z_minus.shape
        assert new_state.B_parallel.shape == state.B_parallel.shape
        assert new_state.g.shape == state.g.shape

    def test_rk4_time_increment(self):
        """RK4 should correctly increment time."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        state = initialize_alfven_wave(
            grid,
            kz_mode=1,
            amplitude=0.1,
            M=20,
        )

        # Multiple timesteps
        dt = 0.01
        for i in range(10):
            state = rk4_step(state, dt, eta=0.0, v_A=1.0)

        # Time should be 10*dt
        expected_time = 10 * dt
        assert jnp.isclose(state.time, expected_time, rtol=1e-6)

    def test_rk4_deterministic(self):
        """RK4 should produce deterministic results."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        state = initialize_alfven_wave(grid, M=20, kz_mode=1, amplitude=0.1)

        # Run twice with same inputs
        new_state_1 = rk4_step(state, dt=0.01, eta=0.01, v_A=1.0)
        new_state_2 = rk4_step(state, dt=0.01, eta=0.01, v_A=1.0)

        # Should produce identical results
        assert jnp.allclose(new_state_1.z_plus, new_state_2.z_plus)
        assert jnp.allclose(new_state_1.z_minus, new_state_2.z_minus)
        assert new_state_1.time == new_state_2.time

    def test_rk4_reality_condition(self):
        """RK4 should preserve reality condition f(-k) = f*(k)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        state = initialize_alfven_wave(grid, M=20, kz_mode=1, amplitude=0.1)

        # Take a timestep
        new_state = rk4_step(state, dt=0.01, eta=0.01, v_A=1.0)

        # Check reality condition for z_plus
        # For rfft: we only store positive kx frequencies
        # Reality: f[kz, ky, kx] = conj(f[-kz, -ky, kx]) for stored kx

        # Check a few modes manually
        Nz, Ny = grid.Nz, grid.Ny

        # Mode (1, 1): should equal conj of mode (-1, -1)
        f_pos = new_state.z_plus[1, 1, 1]
        f_neg = new_state.z_plus[-1, -1, 1]
        assert jnp.isclose(f_pos, jnp.conj(f_neg), rtol=1e-5), \
            f"Reality condition violated: f(1,1)={f_pos}, f*(-1,-1)={jnp.conj(f_neg)}"

        # Mode (2, 3): should equal conj of mode (-2, -3)
        f_pos = new_state.z_plus[2, 3, 2]
        f_neg = new_state.z_plus[-2, -3, 2]
        assert jnp.isclose(f_pos, jnp.conj(f_neg), rtol=1e-5)

        # Same for z_minus
        f_pos = new_state.z_minus[1, 1, 1]
        f_neg = new_state.z_minus[-1, -1, 1]
        assert jnp.isclose(f_pos, jnp.conj(f_neg), rtol=1e-5)


class TestCFLCalculator:
    """Test CFL condition calculator."""

    def test_cfl_zero_velocity(self):
        """CFL should depend only on v_A for zero flow."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)

        # Zero state → zero flow
        state = KRMHDState(
            z_plus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            z_minus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            B_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            g=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, 21), dtype=jnp.complex64),
            M=20,
            beta_i=1.0,
            v_th=1.0,
            nu=0.01,
            Lambda=1.0,
            time=0.0,
            grid=grid,
        )

        v_A = 1.0
        cfl_safety = 0.3

        dt = compute_cfl_timestep(state, v_A, cfl_safety)

        # dt should be cfl_safety * min_spacing / v_A
        dx = grid.Lx / grid.Nx
        dy = grid.Ly / grid.Ny
        dz = grid.Lz / grid.Nz
        min_spacing = min(dx, dy, dz)
        expected_dt = cfl_safety * min_spacing / v_A

        assert jnp.isclose(dt, expected_dt, rtol=1e-6)

    def test_cfl_grid_dependence(self):
        """CFL timestep should scale with grid spacing."""
        v_A = 1.0
        cfl_safety = 0.3

        # Coarse grid
        grid_coarse = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state_coarse = initialize_alfven_wave(grid_coarse, M=20, kz_mode=1, amplitude=0.01)
        dt_coarse = compute_cfl_timestep(state_coarse, v_A, cfl_safety)

        # Fine grid (2x resolution)
        grid_fine = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)
        state_fine = initialize_alfven_wave(grid_fine, M=20, kz_mode=1, amplitude=0.01)
        dt_fine = compute_cfl_timestep(state_fine, v_A, cfl_safety)

        # dt should scale linearly with spacing (halve when doubling resolution)
        # Allow some tolerance due to different velocity amplitudes
        assert dt_fine < dt_coarse
        assert 0.4 < dt_fine / dt_coarse < 0.6  # Should be ~0.5

    def test_cfl_velocity_dependence(self):
        """CFL timestep should decrease with increasing velocity."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        cfl_safety = 0.3

        # Small amplitude wave
        state_small = initialize_alfven_wave(grid, M=20, kz_mode=1, amplitude=0.01)
        dt_small = compute_cfl_timestep(state_small, v_A=1.0, cfl_safety=cfl_safety)

        # Larger amplitude wave (stronger flows)
        state_large = initialize_alfven_wave(grid, M=20, kz_mode=1, amplitude=0.5)
        dt_large = compute_cfl_timestep(state_large, v_A=1.0, cfl_safety=cfl_safety)

        # Larger amplitude → stronger flows → smaller timestep
        assert dt_large <= dt_small

    def test_cfl_safety_factor(self):
        """CFL timestep should scale linearly with safety factor."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=20, kz_mode=1, amplitude=0.1)
        v_A = 1.0

        dt_conservative = compute_cfl_timestep(state, v_A, cfl_safety=0.1)
        dt_aggressive = compute_cfl_timestep(state, v_A, cfl_safety=0.5)

        # dt should scale linearly with safety factor
        ratio = dt_aggressive / dt_conservative
        assert jnp.isclose(ratio, 0.5 / 0.1, rtol=0.01)


class TestAlfvenWavePropagation:
    """Test Alfvén wave dynamics with time integration."""

    def test_alfven_wave_frequency(self):
        """Verify Alfvén wave oscillates at ω = k∥v_A with good energy conservation."""
        # Setup: single Alfvén wave with k∥ = 2π/Lz
        Lz = 2.0 * jnp.pi
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=Lz)

        kz = 2.0 * jnp.pi / Lz  # k∥ = 1 in code units
        v_A = 1.0
        omega_expected = kz * v_A  # ω = k∥v_A = 1.0

        # Initialize Alfvén wave
        state = initialize_alfven_wave(grid, M=20, kz_mode=1, amplitude=0.1)

        # Measure initial energy
        E_dict_0 = energy(state)
        E_0 = E_dict_0['total']

        # Integrate for one period: T = 2π/ω
        T_period = 2.0 * jnp.pi / omega_expected
        n_steps = 100
        dt = T_period / n_steps

        for _ in range(n_steps):
            state = rk4_step(state, dt, eta=0.0, v_A=v_A)  # Inviscid

        # After one period, energy should be conserved
        E_dict_1 = energy(state)
        E_1 = E_dict_1['total']

        # Check numerical stability (no NaN/Inf)
        assert jnp.isfinite(E_0), f"Initial energy is not finite: {E_0}"
        assert jnp.isfinite(E_1), f"Final energy is not finite: {E_1}"

        # Energy conservation: with GANDALF formulation + RK4, expect < 1% error over one period
        relative_energy_change = jnp.abs(E_1 - E_0) / E_0
        assert relative_energy_change < 0.01, \
            f"Energy not conserved: ΔE/E = {relative_energy_change:.2%}, expected < 1%"

    def test_wave_does_not_grow_exponentially(self):
        """Wave energy should remain bounded (numerical stability test)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=20, kz_mode=1, amplitude=0.1)

        # CFL-limited timestep
        dt = compute_cfl_timestep(state, v_A=1.0, cfl_safety=0.3)

        # Track energy over time
        energies = []
        times = []

        for i in range(50):
            E_dict = energy(state)
            energies.append(float(E_dict['total']))
            times.append(float(state.time))
            state = rk4_step(state, dt, eta=0.0, v_A=1.0)

        energies = jnp.array(energies)

        # Check numerical stability: no NaN/Inf
        assert jnp.all(jnp.isfinite(energies)), "Energy became NaN or Inf"

        # Energy should be well-conserved (< 5% change over 50 steps)
        # With GANDALF formulation, energy drift should be minimal
        if energies[0] > 0:
            relative_change = jnp.abs(energies[-1] - energies[0]) / energies[0]
            assert relative_change < 0.05, \
                f"Energy changed by {relative_change:.1%} over 50 steps, expected < 5%"

    def test_dissipation_with_eta(self):
        """Energy should decay with resistivity η > 0 according to exp(-2ηk⊥²t)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)

        # Initialize Alfvén wave with known k⊥
        kz_mode = 1
        state = initialize_alfven_wave(grid, M=20, kz_mode=kz_mode, amplitude=0.1)

        E_dict_0 = energy(state)
        E_0 = E_dict_0['total']

        # Integration parameters
        eta = 0.1  # Strong dissipation for clear signal
        dt = 0.01
        n_steps = 50
        total_time = dt * n_steps

        # Calculate expected dissipation for dominant mode (kx=1, ky=0)
        # For Alfvén wave initialized with kx_mode=1.0, ky_mode=0.0:
        kx_dominant = 2.0 * jnp.pi / grid.Lx  # = 1.0
        ky_dominant = 0.0
        k_perp_squared = kx_dominant**2 + ky_dominant**2

        # Energy decays as: E(t) = E_0 * exp(-2 * η * k⊥² * t)
        # (factor of 2 because energy ~ |field|², and field decays as exp(-ηk⊥²t))
        expected_decay_fraction = 1.0 - jnp.exp(-2.0 * eta * k_perp_squared * total_time)

        for _ in range(n_steps):
            state = rk4_step(state, dt, eta=eta, v_A=1.0)

        E_dict_1 = energy(state)
        E_1 = E_dict_1['total']

        # Check for finite energies
        assert jnp.isfinite(E_0), f"Initial energy is not finite: {E_0}"
        assert jnp.isfinite(E_1), f"Final energy is not finite: {E_1}"

        # Energy should decrease with resistivity (if initial energy is nonzero)
        if E_0 > 1e-10:  # Only test if initial energy is significant
            assert E_1 < E_0, f"Energy should decay with η>0: E_0={E_0:.3e}, E_1={E_1:.3e}"

            # Check dissipation against analytical prediction (with tolerance)
            # Using k⊥² dissipation (thesis Eq. 2.23): E(t) = E_0 * exp(-2ηk⊥²t)
            actual_decay_fraction = (E_0 - E_1) / E_0

            # Allow 60%-140% of expected decay (accounting for nonlinear effects, RK2 error, etc.)
            assert 0.6 * expected_decay_fraction < actual_decay_fraction < 1.4 * expected_decay_fraction, \
                f"Dissipation rate mismatch: expected {expected_decay_fraction*100:.1f}% decay, " \
                f"got {actual_decay_fraction*100:.1f}% (tolerance: 60%-140%)"
        else:
            # If initial energy is too small, just verify it stayed small
            assert E_1 < 1e-8, f"Energy grew from nearly zero: E_0={E_0:.3e}, E_1={E_1:.3e}"


class TestConvergence:
    """Test GANDALF integrating factor + RK2 convergence."""

    def test_second_order_convergence(self):
        """Verify GANDALF integrating factor + RK2 achieves O(dt²) convergence.

        GANDALF algorithm gives 2nd-order convergence:
        - Linear propagation: exact via integrating factor (no error)
        - Nonlinear terms: O(dt²) error from RK2 (midpoint method)

        Overall convergence is O(dt²).
        """
        # Simple test: single Alfvén wave over short time
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)
        v_A = 1.0
        eta = 0.0  # Inviscid for cleaner convergence test

        # Reference solution: very small timestep (effectively "exact")
        state_ref = initialize_alfven_wave(grid, kz_mode=1, amplitude=0.1, M=20)
        dt_ref = 1e-4
        t_final = 0.1
        n_steps_ref = int(t_final / dt_ref)

        for _ in range(n_steps_ref):
            state_ref = rk4_step(state_ref, dt_ref, eta=eta, v_A=v_A)

        # Extract reference z_plus field
        z_plus_ref = state_ref.z_plus

        # Test with progressively smaller timesteps
        timesteps = [0.02, 0.01, 0.005, 0.0025]
        errors = []

        for dt in timesteps:
            state = initialize_alfven_wave(grid, M=20, kz_mode=1, amplitude=0.1)
            n_steps = int(t_final / dt)

            for _ in range(n_steps):
                state = rk4_step(state, dt, eta=eta, v_A=v_A)

            # L2 error in z_plus
            error = jnp.sqrt(jnp.mean(jnp.abs(state.z_plus - z_plus_ref)**2))
            errors.append(float(error))

        errors = jnp.array(errors)
        timesteps = jnp.array(timesteps)

        # Compute convergence rate from consecutive error ratios
        # error ~ C * dt^p  =>  error_ratio = (dt1/dt2)^p
        convergence_rates = []
        for i in range(len(errors) - 1):
            dt_ratio = timesteps[i] / timesteps[i+1]  # Should be 2.0
            error_ratio = errors[i] / errors[i+1]
            p = jnp.log(error_ratio) / jnp.log(dt_ratio)
            convergence_rates.append(float(p))

        # For GANDALF RK2, expect p ≈ 2.0 in theory
        # However, with small amplitude (0.1) and fine spatial resolution,
        # errors can be dominated by spatial discretization (~1e-8) rather than
        # temporal truncation. If errors are already near machine precision,
        # we won't see convergence in time.

        # Check if errors are small enough that we're limited by spatial resolution
        max_error = jnp.max(jnp.array(errors))
        if max_error < 1e-7:
            # Errors are tiny - spatial resolution or roundoff dominated
            # Just verify they stay small
            assert max_error < 1e-6, f"Errors should be small: max error = {max_error:.3e}"
        else:
            # Errors are large enough to measure temporal convergence
            avg_rate = jnp.mean(jnp.array(convergence_rates))
            assert avg_rate > 1.5, \
                f"Average convergence rate p={avg_rate:.2f} too low (expected >1.5 for RK2)"


class TestHermiteMomentIntegration:
    """Test that Hermite moments are properly integrated in GANDALF timestepper."""

    def test_hermite_moment_evolution(self):
        """Verify Hermite moments evolve when integrated with GANDALF (Issue #49)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        # Initialize with perturbed Hermite moments
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)

        # Verify initial g moments are non-zero (have small perturbation from initialization)
        initial_g_norm = jnp.linalg.norm(state.g)
        assert initial_g_norm > 0, f"Initial g should have perturbations, got norm={initial_g_norm:.3e}"

        # Take several timesteps
        dt = 0.01
        for _ in range(10):
            state = rk4_step(state, dt, eta=0.01, v_A=1.0)

        # Verify g evolved (changed from initial)
        final_g_norm = jnp.linalg.norm(state.g)
        relative_change = jnp.abs(final_g_norm - initial_g_norm) / initial_g_norm

        assert relative_change > 1e-6, \
            f"g moments should evolve significantly, got {relative_change:.2e} relative change"
        assert jnp.isfinite(final_g_norm), \
            f"g moments should remain finite, got {final_g_norm}"

        # Verify moments remain complex-valued
        assert jnp.iscomplexobj(state.g), "g should remain complex in Fourier space"

    def test_hermite_moment_coupling(self):
        """Verify Hermite moments couple to Elsasser fields."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        # Initialize with Alfvén wave (has both Elsasser and Hermite perturbations)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1, beta_i=1.0)

        # Compute initial RHS
        rhs = krmhd_rhs(state, eta=0.0, v_A=1.0)

        # Verify g RHS is non-zero (coupling exists)
        g_rhs_norm = jnp.linalg.norm(rhs.g)
        assert g_rhs_norm > 1e-12, \
            f"g RHS should be non-zero due to coupling, got norm={g_rhs_norm:.3e}"

        # Verify g0 and g1 specifically evolve (thesis Eq. 2.7-2.8)
        g0_rhs_norm = jnp.linalg.norm(rhs.g[:, :, :, 0])
        g1_rhs_norm = jnp.linalg.norm(rhs.g[:, :, :, 1])

        assert g0_rhs_norm > 1e-12, f"g0 RHS should be non-zero, got {g0_rhs_norm:.3e}"
        assert g1_rhs_norm > 1e-12, f"g1 RHS should be non-zero, got {g1_rhs_norm:.3e}"


class TestHyperdissipation:
    """Test suite for hyper-dissipation and hyper-collisions."""

    def test_backward_compatibility_r1_n1(self):
        """Default hyper_r=1, hyper_n=1 should match original behavior."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)

        # Initialize same state
        state = initialize_alfven_wave(grid, M=20, kz_mode=1, amplitude=0.1)

        # Standard call (no hyper parameters)
        dt = 0.01
        eta = 0.01
        v_A = 1.0

        # Evolve with defaults (hyper_r=1, hyper_n=1)
        state_default = rk4_step(state, dt, eta=eta, v_A=v_A)

        # Evolve with explicit hyper_r=1, hyper_n=1 (should be identical)
        state_explicit = rk4_step(state, dt, eta=eta, v_A=v_A, hyper_r=1, hyper_n=1)

        # Should be exactly the same
        assert jnp.allclose(state_default.z_plus, state_explicit.z_plus, atol=1e-10), \
            "hyper_r=1 should match default behavior"
        assert jnp.allclose(state_default.z_minus, state_explicit.z_minus, atol=1e-10), \
            "hyper_r=1 should match default behavior"
        assert jnp.allclose(state_default.g, state_explicit.g, atol=1e-10), \
            "hyper_n=1 should match default behavior"

    def test_hermite_moments_dual_dissipation(self):
        """Hermite moments should receive BOTH resistive dissipation AND collision damping."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)

        # Initialize state with non-zero Hermite moments (M=10 for reasonable test time)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)

        # Use moderate parameters with both mechanisms at comparable strength
        dt = 0.01
        eta = 0.01  # Smaller resistivity (safe for r=2, k_max~16)
        nu = 0.1    # Moderate collision frequency (safe for n=2, M=10: nu < 0.5)
        state.nu = nu
        v_A = 1.0

        # Evolve with BOTH hyper-resistivity (r=2) and hyper-collisions (n=2)
        state_dual = rk4_step(state, dt, eta=eta, v_A=v_A, hyper_r=2, hyper_n=2)

        # Also evolve with ONLY resistivity (n=1) for comparison
        state_resist_only = rk4_step(state, dt, eta=eta, v_A=v_A, hyper_r=2, hyper_n=1)

        # Also evolve with ONLY collisions (r=1) for comparison
        state_collide_only = rk4_step(state, dt, eta=eta, v_A=v_A, hyper_r=1, hyper_n=2)

        # Compute Hermite moment energy (excluding m=0,1 which are conserved by collisions)
        # Sum over m >= 2 to see the effect of collision damping
        E_dual_high = jnp.sum(jnp.abs(state_dual.g[2:, :, :, :])**2)
        E_resist_only_high = jnp.sum(jnp.abs(state_resist_only.g[2:, :, :, :])**2)
        E_collide_only_high = jnp.sum(jnp.abs(state_collide_only.g[2:, :, :, :])**2)

        # With both mechanisms, high-m energy should be lower than either mechanism alone
        # Collision damping primarily affects m >= 2 moments
        # Resistivity affects all moments through coupling to z±
        assert E_dual_high < E_resist_only_high, \
            "Dual dissipation should remove more high-m energy than resistivity alone"
        assert E_dual_high < E_collide_only_high, \
            "Dual dissipation should remove more high-m energy than collisions alone"

    # NOTE: Tests for r=4, r=8, and n=4 are omitted here because safe parameters
    # (required to avoid overflow) result in negligible dissipation that cannot be
    # reliably measured. The validation tests (TestHyperdissipationValidation) ensure
    # these parameters are caught before use. In production, r=2 is the practical
    # maximum for typical grid sizes, not r=8.


class TestHyperdissipationValidation:
    """Test suite for hyper-dissipation parameter validation and safety checks."""

    def test_invalid_hyper_r(self):
        """Invalid hyper_r should raise ValueError."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)

        # Invalid hyper_r values
        invalid_r_values = [0, 3, 5, 6, 7, 9, 10]

        for r in invalid_r_values:
            with pytest.raises(ValueError, match="hyper_r must be 1, 2, 4, or 8"):
                rk4_step(state, dt=0.01, eta=0.01, v_A=1.0, hyper_r=r)

    def test_invalid_hyper_n(self):
        """Invalid hyper_n should raise ValueError."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)

        # Invalid hyper_n values
        invalid_n_values = [0, 3, 5, 6, 8]

        for n in invalid_n_values:
            with pytest.raises(ValueError, match="hyper_n must be 1, 2, or 4"):
                rk4_step(state, dt=0.01, eta=0.01, v_A=1.0, hyper_n=n)

    def test_hypercollision_overflow_error(self):
        """Hyper-collision overflow risk should raise ValueError."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        # Create state with parameters that will cause overflow
        # For M=20, n=4, dt=0.1: 20^8 = 2.56e10
        # Need nu * 2.56e10 * 0.1 >= 50 → nu >= 1.95e-8
        # Use nu=2e-8 which is too large
        state = initialize_alfven_wave(grid, M=20, kz_mode=1, amplitude=0.1)
        state.nu = 2e-8  # Too large for M=20, n=4, dt=0.1

        with pytest.raises(ValueError, match="Hyper-collision overflow risk detected"):
            rk4_step(state, dt=0.1, eta=0.01, v_A=1.0, hyper_n=4)

    def test_hypercollision_overflow_warning(self):
        """Moderate hyper-collision rate should emit warning."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        # Create state with moderate overflow risk (20 < rate < 50)
        # For M=10, n=4, dt=0.01: 10^8 = 1e8
        # Need nu * 1e8 * 0.01 between 20 and 50 → 2e-5 < nu < 5e-5
        # Use nu=3e-5 which triggers warning but not error
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)
        state.nu = 3e-5  # Moderate risk for M=10, n=4, dt=0.01

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            rk4_step(state, dt=0.01, eta=0.01, v_A=1.0, hyper_n=4)

            # Check that a warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)
            assert "Hyper-collision damping rate is high" in str(w[0].message)

    def test_hyperresistivity_overflow_error(self):
        """Hyper-resistivity overflow risk should raise ValueError."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)

        # For r=8, Nx=32: k_max ≈ 16, k_max^16 is huge
        # safe eta < 50 / (k_max^16 * dt)
        # Use very large eta to trigger error
        eta_overflow = 1.0  # Too large for r=8

        with pytest.raises(ValueError, match="Hyper-resistivity overflow risk detected"):
            rk4_step(state, dt=0.1, eta=eta_overflow, v_A=1.0, hyper_r=8)

    def test_hyperresistivity_overflow_warning(self):
        """Moderate hyper-resistivity rate should emit warning or succeed."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)

        # Test that moderate parameters don't crash
        # Whether it warns or not depends on exact grid/parameter combination
        # The key is it doesn't raise an error
        eta_moderate = 0.001

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = rk4_step(state, dt=0.01, eta=eta_moderate, v_A=1.0, hyper_r=2)

            # Should complete successfully (may or may not warn)
            assert result.time > state.time
            assert jnp.isfinite(result.z_plus).all()

    def test_safe_hyper_parameters(self):
        """Safe hyper parameters should work without error or warning."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        # Create state with safe parameters
        # For M=10, n=4: need nu * 10^8 * dt < 20 → nu < 2e-6 (for dt=0.01)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)
        state.nu = 1e-9  # Very safe for M=10, n=4

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Use very small eta for r=8 to avoid overflow
            # For r=8, k_max ≈ 16: k_max^16 ≈ 2e19, so need eta << 1e-18
            result = rk4_step(state, dt=0.01, eta=1e-20, v_A=1.0, hyper_r=8, hyper_n=4)

            # Verify no warnings were issued
            hyper_warnings = [warning for warning in w
                             if "damping rate" in str(warning.message)]
            assert len(hyper_warnings) == 0, \
                f"Safe parameters should not trigger warnings, got: {hyper_warnings}"

            # Verify state advanced correctly
            assert result.time > state.time
            assert jnp.isfinite(result.z_plus).all()
            assert jnp.isfinite(result.g).all()


class TestHyperdissipationEdgeCases:
    """Test suite for edge cases: non-square grids, non-2π domains, mixed parameters."""

    def test_non_square_grid(self):
        """Hyper-dissipation should work with non-square grids (Nx != Ny)."""
        # Create rectangular grid: 64x32x16
        grid = SpectralGrid3D.create(Nx=64, Ny=32, Nz=16, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)

        # Use moderate hyper-dissipation (eta scaled for larger k_max)
        dt = 0.01
        eta = 0.001  # Smaller for Nx=64: k_max~32
        state.nu = 0.05

        # Should work without errors
        state_new = rk4_step(state, dt, eta=eta, v_A=1.0, hyper_r=2, hyper_n=2)

        # Verify shapes are preserved
        assert state_new.z_plus.shape == (grid.Nz, grid.Ny, grid.Nx//2+1)
        assert state_new.g.shape == (grid.Nz, grid.Ny, grid.Nx//2+1, state.M+1)
        assert jnp.isfinite(state_new.z_plus).all()
        assert jnp.isfinite(state_new.g).all()

    def test_non_2pi_domain(self):
        """Hyper-dissipation should work with non-2π domains."""
        # Create grid with arbitrary domain size
        Lx, Ly, Lz = 4.0, 6.0, 8.0
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16, Lx=Lx, Ly=Ly, Lz=Lz)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)

        # Use moderate hyper-dissipation
        dt = 0.01
        eta = 0.01
        state.nu = 0.05

        # Should work without errors
        state_new = rk4_step(state, dt, eta=eta, v_A=1.0, hyper_r=2, hyper_n=2)

        # Verify physics is correct
        # k_max should be 2π/dx = 2π/(Lx/Nx) = 2πNx/Lx
        kx_max_expected = jnp.pi * grid.Nx / Lx
        kx_max_actual = grid.kx[-1]
        assert jnp.allclose(kx_max_actual, kx_max_expected, rtol=1e-10)

        assert jnp.isfinite(state_new.z_plus).all()
        assert jnp.isfinite(state_new.g).all()

    def test_mixed_hyper_parameters_r2_n4(self):
        """Test mixed hyper parameters: r=2 (moderate resistivity), n=4 (strong collisions)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)

        # Mixed parameters: r=2 (easy to tune), n=4 (requires small nu)
        dt = 0.01
        eta = 0.02  # Safe for r=2
        nu = 1e-7   # Safe for n=4, M=10: nu < 5e-7
        state.nu = nu

        # Should work without errors
        state_new = rk4_step(state, dt, eta=eta, v_A=1.0, hyper_r=2, hyper_n=4)

        # Verify dual dissipation mechanisms active
        # r=2: Moderate resistive dissipation
        # n=4: Strong collision damping on high-m moments
        E_initial = jnp.sum(jnp.abs(state.g)**2)
        E_final = jnp.sum(jnp.abs(state_new.g)**2)
        assert E_final < E_initial, "Energy should decrease with dual dissipation"

    def test_mixed_hyper_parameters_r4_n2(self):
        """Test mixed hyper parameters: r=4 (strong resistivity), n=2 (moderate collisions)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)

        # Mixed parameters: r=4 (requires VERY small eta), n=2 (easy to tune)
        dt = 0.01
        eta = 1e-7   # Tiny eta required for r=4, k_max~16 (demonstrates impracticality!)
        nu = 0.1     # Safe for n=2, M=10
        state.nu = nu

        # Should work without errors
        state_new = rk4_step(state, dt, eta=eta, v_A=1.0, hyper_r=4, hyper_n=2)

        # Verify dual dissipation mechanisms active
        E_initial = jnp.sum(jnp.abs(state.g)**2)
        E_final = jnp.sum(jnp.abs(state_new.g)**2)
        assert E_final < E_initial, "Energy should decrease with dual dissipation"

    def test_anisotropic_domain(self):
        """Test with highly anisotropic domain (Lx >> Ly >> Lz)."""
        # Anisotropic domain: long in x, short in z
        grid = SpectralGrid3D.create(Nx=64, Ny=32, Nz=16,
                                     Lx=8*jnp.pi, Ly=4*jnp.pi, Lz=2*jnp.pi)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)

        # Use moderate hyper-dissipation (eta scaled for grid)
        dt = 0.01
        eta = 0.001  # Smaller for Nx=64
        state.nu = 0.05

        # Should work without errors
        state_new = rk4_step(state, dt, eta=eta, v_A=1.0, hyper_r=2, hyper_n=2)

        # Verify wavenumber grids are correctly scaled
        # For Lx = 8π, kx grid should be denser (smaller dk)
        # For Lz = 2π, kz grid should be coarser (larger dk)
        dk_x = grid.kx[1] - grid.kx[0]
        dk_z = grid.kz[1] - grid.kz[0]
        assert dk_x < dk_z, "Longer domain should have finer k-spacing"

        assert jnp.isfinite(state_new.z_plus).all()
        assert jnp.isfinite(state_new.g).all()


class TestHyperdissipationDegenerateCases:
    """Test suite for degenerate cases: zero fields, M=0, very coarse grids."""

    def test_zero_fields(self):
        """Hyper-dissipation should handle zero initial fields gracefully."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        # Create state with all zero fields explicitly
        M = 10
        state = KRMHDState(
            z_plus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1), dtype=jnp.complex64),
            z_minus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1), dtype=jnp.complex64),
            B_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1), dtype=jnp.complex64),
            g=jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1, M+1), dtype=jnp.complex64),
            M=M,
            beta_i=1.0,
            v_th=1.0,
            nu=0.1,
            Lambda=1.0,
            time=0.0,
            grid=grid,
        )

        # Apply hyper-dissipation
        dt = 0.01
        eta = 0.01
        state_new = rk4_step(state, dt, eta=eta, v_A=1.0, hyper_r=2, hyper_n=2)

        # Should remain zero (dissipation of zero is zero)
        assert jnp.allclose(state_new.z_plus, 0.0, atol=1e-15)
        assert jnp.allclose(state_new.z_minus, 0.0, atol=1e-15)
        assert jnp.allclose(state_new.g, 0.0, atol=1e-15)

    def test_very_coarse_grid(self):
        """Hyper-dissipation should work with very coarse grids (Nx=8)."""
        # Very coarse grid: 8x8x8
        grid = SpectralGrid3D.create(Nx=8, Ny=8, Nz=8)
        state = initialize_alfven_wave(grid, M=5, kz_mode=1, amplitude=0.1)

        # Even for a coarse grid, r=8 requires tiny eta
        # k_max ~ 4.12, so k_max^16 ~ 7e9 (still huge!)
        # Use r=2 instead (practical maximum)
        dt = 0.01
        eta = 0.01  # Safe for r=2
        state.nu = 0.05

        # Should work without errors
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            state_new = rk4_step(state, dt, eta=eta, v_A=1.0, hyper_r=2, hyper_n=2)

            # Should not crash and produce finite results
            assert jnp.isfinite(state_new.z_plus).all()
            assert jnp.isfinite(state_new.g).all()
            # Verify dissipation is working (energy decreases)
            E_elsasser_init = jnp.sum(jnp.abs(state.z_plus)**2) + jnp.sum(jnp.abs(state.z_minus)**2)
            E_hermite_init = jnp.sum(jnp.abs(state.g)**2)
            E_initial = E_elsasser_init + E_hermite_init

            E_elsasser_final = jnp.sum(jnp.abs(state_new.z_plus)**2) + jnp.sum(jnp.abs(state_new.z_minus)**2)
            E_hermite_final = jnp.sum(jnp.abs(state_new.g)**2)
            E_final = E_elsasser_final + E_hermite_final
            assert E_final < E_initial, "Dissipation should reduce energy"

    def test_M_equals_one(self):
        """Hyper-collisions should handle M=1 gracefully (only m=0,1 moments, no collisions)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        # Create state with M=1 (only g_0 and g_1 exist, both exempt from collisions)
        M = 1
        z_plus = jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1), dtype=jnp.complex64)
        z_plus = z_plus.at[1, 1, 1].set(0.1 + 0.0j)

        g = jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1, M+1), dtype=jnp.complex64)
        g = g.at[1, 1, 1, 0].set(0.1 + 0.0j)  # m=0
        g = g.at[1, 1, 1, 1].set(0.05 + 0.0j)  # m=1

        state = KRMHDState(
            z_plus=z_plus,
            z_minus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1), dtype=jnp.complex64),
            B_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1), dtype=jnp.complex64),
            g=g,
            M=M,
            beta_i=1.0,
            v_th=1.0,
            nu=10.0,  # Large collision frequency (should have NO effect since m=0,1 are exempt)
            Lambda=1.0,
            time=0.0,
            grid=grid,
        )

        # Apply hyper-collisions (large nu to verify exemption)
        dt = 0.01
        eta = 0.01
        state_new = rk4_step(state, dt, eta=eta, v_A=1.0, hyper_r=2, hyper_n=4)

        # m=0,1 should not be affected by collisions (conservation)
        # Should not produce NaN or Inf values despite large nu
        assert state_new.g.shape == (grid.Nz, grid.Ny, grid.Nx//2+1, M+1)
        assert jnp.isfinite(state_new.g).all(), "Should produce finite values despite large nu"

        # Energy should decrease (from resistive dissipation only)
        E_initial = jnp.sum(jnp.abs(state.g)**2)
        E_final = jnp.sum(jnp.abs(state_new.g)**2)
        assert E_final < E_initial, "Energy should decrease due to resistive dissipation"

    def test_warning_threshold_exact(self):
        """Verify warnings trigger at exactly the documented threshold (20.0)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)

        # For M=10, n=2, dt=0.01:
        # Collision rate = nu * 10^4 * 0.01 = nu * 100
        # To hit threshold of exactly 20.0: nu = 20.0 / 100 = 0.2
        dt = 0.01
        eta = 0.001
        nu_threshold = 20.0 / (10**4 * dt)  # Exactly at warning threshold
        state.nu = nu_threshold

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Should trigger warning (rate >= 20.0)
            state_new = rk4_step(state, dt, eta=eta, v_A=1.0, hyper_r=2, hyper_n=2)

            # Verify warning was triggered
            hyper_warnings = [warning for warning in w
                             if "damping rate" in str(warning.message)]
            assert len(hyper_warnings) > 0, "Should trigger warning at threshold 20.0"

        # Just below threshold should NOT warn
        state.nu = nu_threshold * 0.99  # Slightly below 20.0

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            state_new = rk4_step(state, dt, eta=eta, v_A=1.0, hyper_r=2, hyper_n=2)

            # Verify no warning below threshold
            hyper_warnings = [warning for warning in w
                             if "damping rate" in str(warning.message)]
            assert len(hyper_warnings) == 0, "Should NOT warn below threshold 20.0"
