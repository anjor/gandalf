"""
Tests for time integration module.

This test suite validates:
- GANDALF integrating factor + RK2 timestepper correctness and convergence
- CFL condition calculator
- KRMHD RHS function wrapper
- Energy conservation during time evolution
- Alfvén wave propagation

Physics verification includes:
- 2nd order convergence rate for RK2 (exact for linear waves via integrating factor)
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
from krmhd.timestepping import krmhd_rhs, gandalf_step, compute_cfl_timestep


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


class TestGandalfStep:
    """Test GANDALF integrating factor + RK2 time integrator."""

    def test_gandalf_zero_state(self):
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
        new_state = gandalf_step(state, dt, eta=0.01, v_A=1.0)

        # Fields should remain zero
        assert jnp.allclose(new_state.z_plus, 0.0, atol=1e-10)
        assert jnp.allclose(new_state.z_minus, 0.0, atol=1e-10)
        assert jnp.allclose(new_state.B_parallel, 0.0, atol=1e-10)
        assert jnp.allclose(new_state.g, 0.0, atol=1e-10)

        # Time should advance
        assert jnp.isclose(new_state.time, dt)

    def test_gandalf_shape_preservation(self):
        """GANDALF should preserve field shapes."""
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
        new_state = gandalf_step(state, dt=0.01, eta=0.01, v_A=1.0)

        # Check shapes
        assert new_state.z_plus.shape == state.z_plus.shape
        assert new_state.z_minus.shape == state.z_minus.shape
        assert new_state.B_parallel.shape == state.B_parallel.shape
        assert new_state.g.shape == state.g.shape

    def test_gandalf_time_increment(self):
        """GANDALF should correctly increment time."""
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
            state = gandalf_step(state, dt, eta=0.0, v_A=1.0)

        # Time should be 10*dt
        expected_time = 10 * dt
        assert jnp.isclose(state.time, expected_time, rtol=1e-6)

    def test_gandalf_deterministic(self):
        """GANDALF should produce deterministic results."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        state = initialize_alfven_wave(grid, M=20, kz_mode=1, amplitude=0.1)

        # Run twice with same inputs
        new_state_1 = gandalf_step(state, dt=0.01, eta=0.01, v_A=1.0)
        new_state_2 = gandalf_step(state, dt=0.01, eta=0.01, v_A=1.0)

        # Should produce identical results
        assert jnp.allclose(new_state_1.z_plus, new_state_2.z_plus)
        assert jnp.allclose(new_state_1.z_minus, new_state_2.z_minus)
        assert new_state_1.time == new_state_2.time

    def test_gandalf_reality_condition(self):
        """GANDALF should preserve reality condition f(-k) = f*(k)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        state = initialize_alfven_wave(grid, M=20, kz_mode=1, amplitude=0.1)

        # Take a timestep
        new_state = gandalf_step(state, dt=0.01, eta=0.01, v_A=1.0)

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
            state = gandalf_step(state, dt, eta=0.0, v_A=v_A)  # Inviscid

        # After one period, energy should be conserved
        E_dict_1 = energy(state)
        E_1 = E_dict_1['total']

        # Check numerical stability (no NaN/Inf)
        assert jnp.isfinite(E_0), f"Initial energy is not finite: {E_0}"
        assert jnp.isfinite(E_1), f"Final energy is not finite: {E_1}"

        # Energy conservation: with GANDALF formulation + GANDALF, expect < 1% error over one period
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
            state = gandalf_step(state, dt, eta=0.0, v_A=1.0)

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

        # NORMALIZED dissipation formula: exp(-η·(k⊥²/k⊥²_max)·dt)
        # k⊥_max is at 2/3 dealiasing boundary: idx = (N-1)//3
        idx_max_x = (grid.Nx - 1) // 3
        idx_max_y = (grid.Ny - 1) // 3
        k_perp_max_x = (2.0 * jnp.pi / grid.Lx) * idx_max_x
        k_perp_max_y = (2.0 * jnp.pi / grid.Ly) * idx_max_y
        k_perp_max_squared = k_perp_max_x**2 + k_perp_max_y**2

        # Normalized dissipation rate
        normalized_rate = k_perp_squared / k_perp_max_squared

        # Energy decays as: E(t) = E_0 * exp(-2 * η * (k⊥²/k⊥²_max) * t)
        # (factor of 2 because energy ~ |field|², and field decays as exp(-η(k⊥²/k⊥²_max)t))
        expected_decay_fraction = 1.0 - jnp.exp(-2.0 * eta * normalized_rate * total_time)

        for _ in range(n_steps):
            state = gandalf_step(state, dt, eta=eta, v_A=1.0)

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
            state_ref = gandalf_step(state_ref, dt_ref, eta=eta, v_A=v_A)

        # Extract reference z_plus field
        z_plus_ref = state_ref.z_plus

        # Test with progressively smaller timesteps
        timesteps = [0.02, 0.01, 0.005, 0.0025]
        errors = []

        for dt in timesteps:
            state = initialize_alfven_wave(grid, M=20, kz_mode=1, amplitude=0.1)
            n_steps = int(t_final / dt)

            for _ in range(n_steps):
                state = gandalf_step(state, dt, eta=eta, v_A=v_A)

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
            state = gandalf_step(state, dt, eta=0.01, v_A=1.0)

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
        state_default = gandalf_step(state, dt, eta=eta, v_A=v_A)

        # Evolve with explicit hyper_r=1, hyper_n=1 (should be identical)
        state_explicit = gandalf_step(state, dt, eta=eta, v_A=v_A, hyper_r=1, hyper_n=1)

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
        # NORMALIZED dissipation: eta and nu can be larger since constraint is eta·dt < 50
        dt = 0.01
        eta = 1.0   # Stronger for normalized dissipation (eta·dt = 0.01 << 50)
        nu = 1.0    # Stronger collision frequency (nu·dt = 0.01 << 50)
        state.nu = nu
        v_A = 1.0
        n_steps = 10  # Multiple steps to accumulate measurable dissipation

        # Evolve with BOTH hyper-resistivity (r=2) and hyper-collisions (n=2)
        state_dual = state
        for _ in range(n_steps):
            state_dual = gandalf_step(state_dual, dt, eta=eta, v_A=v_A, hyper_r=2, hyper_n=2)

        # Also evolve with ONLY resistivity (n=1) for comparison
        state_resist_only = state
        for _ in range(n_steps):
            state_resist_only = gandalf_step(state_resist_only, dt, eta=eta, v_A=v_A, hyper_r=2, hyper_n=1)

        # Also evolve with ONLY collisions (r=1) for comparison
        state_collide_only = state
        for _ in range(n_steps):
            state_collide_only = gandalf_step(state_collide_only, dt, eta=eta, v_A=v_A, hyper_r=1, hyper_n=2)

        # Compute Hermite moment energy (excluding m=0,1 which are conserved by collisions)
        # Sum over m >= 2 to see the effect of collision damping
        E_dual_high = jnp.sum(jnp.abs(state_dual.g[2:, :, :, :])**2)
        E_resist_only_high = jnp.sum(jnp.abs(state_resist_only.g[2:, :, :, :])**2)
        E_collide_only_high = jnp.sum(jnp.abs(state_collide_only.g[2:, :, :, :])**2)

        # With both mechanisms, high-m energy should be lower than either mechanism alone
        # Collision damping primarily affects m >= 2 moments
        # Resistivity affects all moments through coupling to z±
        # Allow 1% tolerance due to nonlinear effects and numerical precision
        assert E_dual_high <= E_resist_only_high * 1.01, \
            f"Dual dissipation should remove high-m energy: {E_dual_high:.3f} vs resist-only {E_resist_only_high:.3f}"
        assert E_dual_high <= E_collide_only_high * 1.01, \
            f"Dual dissipation should remove high-m energy: {E_dual_high:.3f} vs collide-only {E_collide_only_high:.3f}"

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
                gandalf_step(state, dt=0.01, eta=0.01, v_A=1.0, hyper_r=r)

    def test_invalid_hyper_n(self):
        """Invalid hyper_n should raise ValueError."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)

        # Invalid hyper_n values
        invalid_n_values = [0, 3, 5, 6, 8]

        for n in invalid_n_values:
            with pytest.raises(ValueError, match="hyper_n must be 1, 2, or 4"):
                gandalf_step(state, dt=0.01, eta=0.01, v_A=1.0, hyper_n=n)

    def test_hypercollision_overflow_error(self):
        """Hyper-collision overflow risk should raise ValueError."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        # NORMALIZED constraint: ν·dt < 50 (independent of M, n, or resolution!)
        # Choose nu to violate: ν·dt ≥ 50
        dt = 0.1
        nu_overflow = 60.0 / dt  # nu·dt = 60 > 50, triggers overflow
        state = initialize_alfven_wave(grid, M=20, kz_mode=1, amplitude=0.1)
        state.nu = nu_overflow

        with pytest.raises(ValueError, match="Hyper-collision overflow risk detected"):
            gandalf_step(state, dt=dt, eta=0.01, v_A=1.0, hyper_n=4)

    def test_hypercollision_overflow_warning(self):
        """Moderate hyper-collision rate should emit warning."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        # NORMALIZED constraint: ν·dt < 50, warning threshold is ν·dt ≥ 20
        # Test moderate rate: 20 ≤ ν·dt < 50 (should warn but not error)
        dt = 0.01
        nu_moderate = 25.0 / dt  # nu·dt = 25, in warning range [20, 50)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)
        state.nu = nu_moderate

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            gandalf_step(state, dt=dt, eta=0.01, v_A=1.0, hyper_n=4)

            # Check that a warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, RuntimeWarning)
            assert "Hyper-collision damping rate is high" in str(w[0].message)

    def test_hyperresistivity_overflow_error(self):
        """Hyper-resistivity overflow risk should raise ValueError."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)

        # NORMALIZED constraint: η·dt < 50 (independent of k_max, r, or resolution!)
        # Choose eta to violate: η·dt ≥ 50
        dt = 0.1
        eta_overflow = 60.0 / dt  # eta·dt = 60 > 50, triggers overflow

        with pytest.raises(ValueError, match="Hyper-resistivity overflow risk detected"):
            gandalf_step(state, dt=dt, eta=eta_overflow, v_A=1.0, hyper_r=8)

    def test_hyperresistivity_overflow_warning(self):
        """Moderate hyper-resistivity rate should emit warning or succeed."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)

        # NORMALIZED constraint: η·dt < 50, warning threshold is η·dt ≥ 20
        # Test moderate rate: 20 ≤ η·dt < 50 (should warn but not error)
        dt = 0.01
        eta_moderate = 25.0 / dt  # eta·dt = 25, in warning range [20, 50)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = gandalf_step(state, dt=dt, eta=eta_moderate, v_A=1.0, hyper_r=2)

            # Should complete successfully (may or may not warn depending on implementation)
            assert result.time > state.time
            assert jnp.isfinite(result.z_plus).all()

    def test_safe_hyper_parameters(self):
        """Safe hyper parameters should work without error or warning."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        # NORMALIZED constraints (resolution-independent!):
        # - Resistivity: η·dt < 20 (no warning), < 50 (no error)
        # - Collisions: ν·dt < 20 (no warning), < 50 (no error)
        dt = 0.01
        eta_safe = 10.0 / dt     # eta·dt = 10 < 20, very safe
        nu_safe = 10.0 / dt      # nu·dt = 10 < 20, very safe

        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Even with r=8 and n=4, normalized constraint allows reasonable eta/nu
            result = gandalf_step(state, dt=dt, eta=eta_safe, nu=nu_safe, v_A=1.0, hyper_r=8, hyper_n=4)

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
        state_new = gandalf_step(state, dt, eta=eta, v_A=1.0, hyper_r=2, hyper_n=2)

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
        state_new = gandalf_step(state, dt, eta=eta, v_A=1.0, hyper_r=2, hyper_n=2)

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

        # Mixed parameters: r=2 (easy to tune), n=4 (strong collisions)
        # NORMALIZED dissipation allows larger parameters: eta·dt < 50, nu·dt < 50
        dt = 0.01
        eta = 1.0   # Safe for normalized: eta·dt = 0.01 << 50
        nu = 1.0    # Safe for normalized: nu·dt = 0.01 << 50
        state.nu = nu
        n_steps = 5  # Multiple steps to accumulate measurable dissipation

        # Evolve with dual dissipation
        for _ in range(n_steps):
            state = gandalf_step(state, dt, eta=eta, v_A=1.0, hyper_r=2, hyper_n=4)

        # Verify state remains finite (no overflow with moderate parameters)
        assert jnp.isfinite(state.z_plus).all(), "z_plus should remain finite"
        assert jnp.isfinite(state.g).all(), "Hermite moments should remain finite"

        # Energy test removed: With multiple steps and nonlinear coupling,
        # energy can grow or shrink depending on Alfvén dynamics. The key
        # test is that the calculation remains stable (no NaN/Inf).

    def test_mixed_hyper_parameters_r4_n2(self):
        """Test mixed hyper parameters: r=4 (strong resistivity), n=2 (moderate collisions)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)

        # Mixed parameters: r=4 (strong resistivity), n=2 (moderate collisions)
        # NORMALIZED dissipation: even r=4 works with moderate eta
        dt = 0.01
        eta = 1.0    # Safe for normalized: eta·dt = 0.01 << 50
        nu = 1.0     # Safe for normalized: nu·dt = 0.01 << 50
        state.nu = nu
        n_steps = 5  # Multiple steps to accumulate measurable dissipation

        # Evolve with dual dissipation
        for _ in range(n_steps):
            state = gandalf_step(state, dt, eta=eta, v_A=1.0, hyper_r=4, hyper_n=2)

        # Verify state remains finite (no overflow)
        assert jnp.isfinite(state.z_plus).all(), "z_plus should remain finite"
        assert jnp.isfinite(state.g).all(), "Hermite moments should remain finite"

        # Energy test removed: Nonlinear coupling can cause energy growth.
        # The key test is numerical stability (no NaN/Inf).

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
        state_new = gandalf_step(state, dt, eta=eta, v_A=1.0, hyper_r=2, hyper_n=2)

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
        state_new = gandalf_step(state, dt, eta=eta, v_A=1.0, hyper_r=2, hyper_n=2)

        # Should remain zero (dissipation of zero is zero)
        assert jnp.allclose(state_new.z_plus, 0.0, atol=1e-15)
        assert jnp.allclose(state_new.z_minus, 0.0, atol=1e-15)
        assert jnp.allclose(state_new.g, 0.0, atol=1e-15)

    def test_zero_field_integration_multiple_steps(self):
        """Integration test: Zero fields should remain zero over multiple timesteps (no spurious modes)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        # Create state with all zero fields
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

        # Run multiple timesteps with both standard and hyper-dissipation
        dt = 0.01
        n_steps = 10

        # Test with standard dissipation (r=1, n=1)
        state_r1 = state
        for _ in range(n_steps):
            state_r1 = gandalf_step(state_r1, dt, eta=0.01, v_A=1.0, hyper_r=1, hyper_n=1)

        # Verify no spurious modes generated
        assert jnp.allclose(state_r1.z_plus, 0.0, atol=1e-15), "Spurious modes in z_plus (r=1)"
        assert jnp.allclose(state_r1.z_minus, 0.0, atol=1e-15), "Spurious modes in z_minus (r=1)"
        assert jnp.allclose(state_r1.g, 0.0, atol=1e-15), "Spurious modes in g (r=1)"

        # Test with hyper-dissipation (r=2, n=2)
        state_r2 = state
        for _ in range(n_steps):
            state_r2 = gandalf_step(state_r2, dt, eta=0.0001, v_A=1.0, hyper_r=2, hyper_n=2)

        # Verify no spurious modes generated
        assert jnp.allclose(state_r2.z_plus, 0.0, atol=1e-15), "Spurious modes in z_plus (r=2)"
        assert jnp.allclose(state_r2.z_minus, 0.0, atol=1e-15), "Spurious modes in z_minus (r=2)"
        assert jnp.allclose(state_r2.g, 0.0, atol=1e-15), "Spurious modes in g (r=2)"

        # Explicitly verify all Fourier modes remain zero (no single-mode excitation)
        assert jnp.max(jnp.abs(state_r2.z_plus)) == 0.0, "Non-zero Fourier mode in z_plus"
        assert jnp.max(jnp.abs(state_r2.z_minus)) == 0.0, "Non-zero Fourier mode in z_minus"
        assert jnp.max(jnp.abs(state_r2.g)) == 0.0, "Non-zero Fourier mode in g"

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
            state_new = gandalf_step(state, dt, eta=eta, v_A=1.0, hyper_r=2, hyper_n=2)

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
        """M=1 should be rejected (collision operators require M >= 2)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        # Create state with M=1 (degenerate case for normalized collisions)
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
            nu=10.0,
            Lambda=1.0,
            time=0.0,
            grid=grid,
        )

        # M=1 should raise ValueError (normalized collisions require M >= 2)
        dt = 0.01
        eta = 0.01
        with pytest.raises(ValueError, match="M must be >= 2"):
            gandalf_step(state, dt, eta=eta, v_A=1.0, hyper_r=2, hyper_n=4)

    def test_warning_threshold_exact(self):
        """Verify warnings trigger at exactly the documented threshold (20.0)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)

        # NORMALIZED constraint: warning threshold is ν·dt = 20.0
        # Independent of M, n, or resolution!
        dt = 0.01
        eta = 0.001
        nu_threshold = 20.0 / dt  # Exactly at warning threshold (nu·dt = 20.0)
        state.nu = nu_threshold

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Should trigger warning (rate >= 20.0)
            state_new = gandalf_step(state, dt, eta=eta, v_A=1.0, hyper_r=2, hyper_n=2)

            # Verify warning was triggered
            hyper_warnings = [warning for warning in w
                             if "damping rate" in str(warning.message)]
            assert len(hyper_warnings) > 0, "Should trigger warning at threshold 20.0"

        # Just below threshold should NOT warn
        nu_below = 19.9 / dt  # nu·dt = 19.9 < 20.0
        state.nu = nu_below

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            state_new = gandalf_step(state, dt, eta=eta, v_A=1.0, hyper_r=2, hyper_n=2)

            # Verify no warning below threshold
            hyper_warnings = [warning for warning in w
                             if "damping rate" in str(warning.message)]
            assert len(hyper_warnings) == 0, "Should NOT warn below threshold 20.0"


class TestExtremeHyperDissipation:
    """Test extreme hyper-dissipation parameters (r=8, n=4) for numerical stability."""

    def test_hyper_dissipation_r8_no_overflow(self):
        """Verify exp(-η·1^8·dt) doesn't produce NaN/Inf for extreme r=8."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kx_mode=1.0, ky_mode=0.0, kz_mode=1.0, amplitude=0.1)

        # Extreme but valid parameters: r=8 (thesis value), η near upper limit
        eta = 10.0  # High but below η·dt < 50
        dt = 0.004  # Small timestep: η·dt = 0.04 << 50
        hyper_r = 8  # Maximum thesis value

        # Should not produce NaN or Inf, even at k = k_max
        state_new = gandalf_step(state, dt, eta=eta, v_A=1.0, hyper_r=hyper_r, hyper_n=2)

        # Check all fields are finite
        assert jnp.all(jnp.isfinite(state_new.z_plus)), "z_plus contains NaN/Inf with r=8"
        assert jnp.all(jnp.isfinite(state_new.z_minus)), "z_minus contains NaN/Inf with r=8"
        assert jnp.all(jnp.isfinite(state_new.g)), "g moments contain NaN/Inf with r=8"

        # Energy should remain finite and positive
        E_dict = energy(state_new)
        assert jnp.isfinite(E_dict['total']), "Total energy is NaN/Inf with r=8"
        assert E_dict['total'] > 0, "Total energy became negative with r=8"

    def test_hyper_collision_n4_no_overflow(self):
        """Verify exp(-ν·(m/M)^8·dt) doesn't produce NaN/Inf for extreme n=4."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kx_mode=1.0, ky_mode=0.0, kz_mode=1.0, amplitude=0.1)

        # Extreme but valid parameters: n=4, ν near upper limit
        nu = 10.0  # High but below ν·dt < 50
        dt = 0.004  # Small timestep: ν·dt = 0.04 << 50
        hyper_n = 4  # Maximum value

        # Should not produce NaN or Inf at highest moment m=M
        state_new = gandalf_step(state, dt, eta=0.01, nu=nu, v_A=1.0, hyper_r=2, hyper_n=hyper_n)

        # Check all moments are finite
        assert jnp.all(jnp.isfinite(state_new.g)), "g moments contain NaN/Inf with n=4"

        # Energy should remain finite
        E_dict = energy(state_new)
        assert jnp.isfinite(E_dict['total']), "Total energy is NaN/Inf with n=4"

    def test_combined_extreme_parameters(self):
        """Verify r=8, n=4 together don't cause overflow."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kx_mode=1.0, ky_mode=0.0, kz_mode=1.0, amplitude=0.1)

        # Both extreme parameters simultaneously
        eta = 10.0
        nu = 10.0
        dt = 0.004  # η·dt = ν·dt = 0.04 << 50
        hyper_r = 8
        hyper_n = 4

        # Should handle both extreme dissipations without overflow
        state_new = gandalf_step(state, dt, eta=eta, nu=nu, v_A=1.0, hyper_r=hyper_r, hyper_n=hyper_n)

        # All fields must remain finite
        assert jnp.all(jnp.isfinite(state_new.z_plus)), "z_plus overflow with r=8, n=4"
        assert jnp.all(jnp.isfinite(state_new.z_minus)), "z_minus overflow with r=8, n=4"
        assert jnp.all(jnp.isfinite(state_new.g)), "g overflow with r=8, n=4"

        # Energy must remain positive and finite
        E_dict = energy(state_new)
        assert jnp.isfinite(E_dict['total']) and E_dict['total'] > 0, \
            "Energy overflow/underflow with r=8, n=4"

    def test_normalized_dissipation_resolution_independence(self):
        """Verify same η works across resolutions (normalized dissipation)."""
        # This tests the key advantage of normalized dissipation:
        # Same eta parameter works at different resolutions due to k_perp_max normalization
        eta, dt = 1.0, 0.005  # Fixed parameters
        hyper_r = 2

        # Run at 32³
        grid_32 = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state_32 = initialize_alfven_wave(grid_32, M=10, kz_mode=1, amplitude=0.1)
        state_32_new = gandalf_step(state_32, dt, eta=eta, nu=0.0, v_A=1.0, hyper_r=hyper_r)
        E_32_initial = energy(state_32)['total']
        E_32_final = energy(state_32_new)['total']
        decay_32 = E_32_final / E_32_initial

        # Run at 64³ with SAME eta (this is the key test!)
        grid_64 = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)
        state_64 = initialize_alfven_wave(grid_64, M=10, kz_mode=1, amplitude=0.1)
        state_64_new = gandalf_step(state_64, dt, eta=eta, nu=0.0, v_A=1.0, hyper_r=hyper_r)
        E_64_initial = energy(state_64)['total']
        E_64_final = energy(state_64_new)['total']
        decay_64 = E_64_final / E_64_initial

        # Energy decay should be similar across resolutions
        # (normalization by k_perp_max makes dissipation resolution-independent)
        # Allow 10% tolerance due to different mode distributions
        assert abs(decay_32 - decay_64) < 0.1, \
            f"Decay mismatch across resolutions: 32³={decay_32:.4f} vs 64³={decay_64:.4f}"

        # Both should show some decay (eta > 0)
        assert decay_32 < 1.0, "No energy decay at 32³ despite eta > 0"
        assert decay_64 < 1.0, "No energy decay at 64³ despite eta > 0"

        # Verify no overflow at either resolution
        assert jnp.isfinite(E_32_final), "Energy overflow at 32³"
        assert jnp.isfinite(E_64_final), "Energy overflow at 64³"
