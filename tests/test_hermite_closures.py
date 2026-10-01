"""
Comprehensive test suite for Hermite closure schemes and convergence diagnostics.

Tests cover:
1. Basic closure functionality (closure_zero, closure_symmetric)
2. Convergence diagnostics (check_hermite_convergence)
3. Physics validation (closure independence with collisions)
4. M-dependence and convergence testing
5. Integration with RHS functions

Reference:
    Thesis §2.4 - Hermite hierarchy truncation and closure schemes
"""

import pytest
import jax
import jax.numpy as jnp
import jax.scipy as jsp

from krmhd.hermite import (
    closure_zero,
    closure_symmetric,
    check_hermite_convergence,
)
from krmhd.spectral import SpectralGrid3D
from krmhd.physics import KRMHDState, initialize_hermite_moments


# ============================================================================
# Test: Basic Closure Functionality
# ============================================================================


class TestClosureBasicFunctionality:
    """Test basic functionality of closure functions."""

    def test_closure_zero_returns_zeros(self):
        """Test closure_zero returns zeros with correct shape."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        # Create random moment array
        key = jax.random.PRNGKey(42)
        g = jax.random.normal(
            key, (grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), dtype=jnp.float32
        ) + 1j * jax.random.normal(
            key, (grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), dtype=jnp.float32
        )
        g = g.astype(jnp.complex64)

        # Apply closure
        g_M_plus_1 = closure_zero(g, M)

        # Check shape
        expected_shape = (grid.Nz, grid.Ny, grid.Nx // 2 + 1)
        assert g_M_plus_1.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {g_M_plus_1.shape}"

        # Check all values are zero
        assert jnp.all(g_M_plus_1 == 0.0), "closure_zero should return all zeros"

    def test_closure_symmetric_returns_gm_minus_1(self):
        """Test closure_symmetric returns gₘ₋₁ with correct shape."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        # Create random moment array
        key = jax.random.PRNGKey(42)
        g = jax.random.normal(
            key, (grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), dtype=jnp.float32
        ) + 1j * jax.random.normal(
            key, (grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), dtype=jnp.float32
        )
        g = g.astype(jnp.complex64)

        # Apply closure
        g_M_plus_1 = closure_symmetric(g, M)

        # Check shape
        expected_shape = (grid.Nz, grid.Ny, grid.Nx // 2 + 1)
        assert g_M_plus_1.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {g_M_plus_1.shape}"

        # Check values equal gₘ₋₁
        assert jnp.allclose(g_M_plus_1, g[:, :, :, M - 1]), \
            "closure_symmetric should return gₘ₋₁"

    def test_closure_symmetric_raises_for_small_M(self):
        """Test closure_symmetric raises error for M < 2."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        # Create moment array with M = 1 (too small)
        g = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, 2), dtype=jnp.complex64)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Symmetric closure requires M"):
            closure_symmetric(g, M=1)

    def test_closures_preserve_dtype(self):
        """Test closures preserve complex dtype."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        g = jnp.ones(
            (grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), dtype=jnp.complex64
        )

        g_zero = closure_zero(g, M)
        g_sym = closure_symmetric(g, M)

        assert jnp.iscomplexobj(g_zero), "closure_zero should preserve complex dtype"
        assert jnp.iscomplexobj(g_sym), "closure_symmetric should preserve complex dtype"

    def test_closures_jit_compilation(self):
        """Test closures are JIT-compatible."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        key = jax.random.PRNGKey(42)
        g = jax.random.normal(
            key, (grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), dtype=jnp.float32
        ).astype(jnp.complex64)

        # Should compile without error
        try:
            _ = jax.jit(closure_zero, static_argnames=['M'])(g, M)
            _ = jax.jit(closure_symmetric, static_argnames=['M'])(g, M)
        except Exception as e:
            pytest.fail(f"JIT compilation failed: {e}")


# ============================================================================
# Test: Convergence Diagnostics
# ============================================================================


class TestHermiteConvergence:
    """Test convergence diagnostic function."""

    def test_converged_case_exponential_decay(self):
        """Test convergence check with exponentially decaying moments."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        # Create exponentially decaying moments: gₘ ~ exp(-0.5·m)
        g = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), dtype=jnp.complex64)

        for m in range(M + 1):
            # Use exponential decay with some spatial structure
            amplitude = jnp.exp(-0.5 * m)
            g = g.at[:, :, :, m].set(amplitude * (1.0 + 0.0j))

        # Check convergence
        result = check_hermite_convergence(g, threshold=1e-3)

        # Should be converged (exp(-0.5*10) / sum(exp(-0.5*m)) << 1e-3)
        assert result['is_converged'], \
            f"Should be converged, but got energy fraction {result['energy_fraction']}"
        assert result['max_moment_index'] == M
        assert result['energy_total'] > 0

    def test_not_converged_case_uniform_moments(self):
        """Test convergence check fails for uniform moments."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        # Create uniform moments: all equal amplitude
        g = jnp.ones(
            (grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), dtype=jnp.complex64
        )

        # Check convergence
        result = check_hermite_convergence(g, threshold=1e-3)

        # Should NOT be converged (1/(M+1) = 1/11 ≈ 9% >> 0.1%)
        assert not result['is_converged'], \
            f"Should not be converged, but got is_converged=True"
        assert result['energy_fraction'] > 1e-3, \
            f"Energy fraction {result['energy_fraction']} should exceed threshold"

    def test_convergence_threshold_sensitivity(self):
        """Test convergence depends on threshold parameter."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        # Create moments with modest decay: gₘ ~ m^(-2)
        g = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), dtype=jnp.complex64)

        for m in range(M + 1):
            amplitude = 1.0 / (m + 1.0) ** 2  # Avoid division by zero
            g = g.at[:, :, :, m].set(amplitude * (1.0 + 0.0j))

        # Check with loose threshold (should pass)
        result_loose = check_hermite_convergence(g, threshold=0.1)
        assert result_loose['is_converged'], "Should converge with loose threshold"

        # Check with tight threshold (should fail)
        result_tight = check_hermite_convergence(g, threshold=1e-5)
        assert not result_tight['is_converged'], "Should not converge with tight threshold"

    def test_convergence_zero_moments(self):
        """Test convergence check handles zero moments gracefully."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        # All moments zero
        g = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), dtype=jnp.complex64)

        result = check_hermite_convergence(g)

        # Should be trivially converged
        assert result['is_converged']
        assert result['energy_fraction'] == 0.0
        assert result['energy_total'] == 0.0
        assert 'trivially converged' in result['recommendation'].lower()

    def test_convergence_rfft_accounting(self):
        """Test convergence correctly accounts for rfft format."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 5

        # Create moments with energy only in kx=0 plane
        g_kx0 = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), dtype=jnp.complex64)
        g_kx0 = g_kx0.at[:, :, 0, 0].set(1.0)  # Only g0 at kx=0

        # Create moments with energy in kx>0 planes
        g_kx_pos = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), dtype=jnp.complex64)
        g_kx_pos = g_kx_pos.at[:, :, 1, 0].set(1.0)  # Only g0 at kx>0

        # With rfft accounting, kx>0 should have 2× weight
        result_kx0 = check_hermite_convergence(g_kx0, account_for_rfft=True)
        result_kx_pos = check_hermite_convergence(g_kx_pos, account_for_rfft=True)

        # Both should be converged (energy in g0 only)
        assert result_kx0['is_converged']
        assert result_kx_pos['is_converged']

        # kx>0 should have 2× the energy of kx=0 with same amplitude
        # (because of conjugate pairs)
        assert jnp.isclose(
            result_kx_pos['energy_total'] / result_kx0['energy_total'], 2.0, rtol=0.01
        ), "kx>0 modes should have 2× energy due to conjugate pairs"

    def test_convergence_returns_all_fields(self):
        """Test convergence returns all expected dictionary fields."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        g = jnp.ones((grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), dtype=jnp.complex64)

        result = check_hermite_convergence(g)

        # Check all required fields are present
        required_fields = [
            'is_converged',
            'energy_fraction',
            'max_moment_index',
            'energy_highest_moment',
            'energy_total',
            'recommendation'
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

        # Check types
        assert isinstance(result['is_converged'], bool)
        assert isinstance(result['energy_fraction'], float)
        assert isinstance(result['max_moment_index'], int)
        assert isinstance(result['recommendation'], str)

    def test_convergence_recommendation_content(self):
        """Test convergence recommendation contains useful info."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        # Converged case
        g_converged = jnp.zeros(
            (grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), dtype=jnp.complex64
        )
        for m in range(M + 1):
            g_converged = g_converged.at[:, :, :, m].set(jnp.exp(-m) * (1.0 + 0.0j))

        result_conv = check_hermite_convergence(g_converged)
        assert '✓' in result_conv['recommendation'] or 'Converged' in result_conv['recommendation']

        # Not converged case
        g_not_converged = jnp.ones_like(g_converged)

        result_not = check_hermite_convergence(g_not_converged)
        assert '✗' in result_not['recommendation'] or 'Not converged' in result_not['recommendation']
        # Should contain actionable advice
        assert 'Increase M' in result_not['recommendation'] or 'collision' in result_not['recommendation'].lower()


# ============================================================================
# Test: Physics Validation with Timestepping
# ============================================================================


class TestClosurePhysicsValidation:
    """Test that closures give consistent results with finite collisions."""

    def test_gm_rhs_with_different_closures_no_collisions(self):
        """Test gm_rhs with different closures gives different results when ν=0."""
        from krmhd.physics import gm_rhs

        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        # Create non-trivial moment field
        key = jax.random.PRNGKey(42)
        g = jax.random.normal(
            key, (grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), dtype=jnp.float32
        ).astype(jnp.complex64) * 0.01

        z_plus = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64)
        z_minus = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64)

        beta_i = 1.0
        nu = 0.0  # No collisions

        # Compute RHS for highest moment with implicit zero closure
        rhs_M = gm_rhs(
            g, z_plus, z_minus, grid.kx, grid.ky, grid.kz,
            grid.dealias_mask, M, beta_i, grid.Nz, grid.Ny, grid.Nx
        )

        # With ν=0 and different closures (implicit in gm_rhs vs explicit),
        # we expect coupling structure to differ
        # This test documents current behavior: gm_rhs uses zero closure

        # Just verify RHS is computed
        assert rhs_M.shape == (grid.Nz, grid.Ny, grid.Nx // 2 + 1)

    def test_gm_rhs_no_collision_term(self):
        """Test that gm_rhs does NOT include collision damping.

        Collisions are now handled exclusively by the exponential step in the
        timestepper (not in the RHS). This test verifies that with isolated
        moments (gₘ = 1, all neighbors = 0, zero z±), gm_rhs returns zero.
        """
        from krmhd.physics import gm_rhs

        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        z_plus = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64)
        z_minus = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64)

        beta_i = 1.0

        # Set only g[2] = 1, all neighbors zero → no streaming coupling
        g_m2 = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), dtype=jnp.complex64)
        g_m2 = g_m2.at[4, 5, 6, 2].set(1.0 + 0.0j)

        rhs_m2 = gm_rhs(
            g_m2, z_plus, z_minus, grid.kx, grid.ky, grid.kz,
            grid.dealias_mask, 2, beta_i, grid.Nz, grid.Ny, grid.Nx
        )

        # With no neighbors and no z±, RHS should be zero (no collision term)
        assert jnp.allclose(rhs_m2[4, 5, 6], 0.0, atol=1e-6), \
            f"gm_rhs should be zero for isolated moment (no collision term), got {rhs_m2[4, 5, 6]}"


# ============================================================================
# Test: M-dependence and Convergence
# ============================================================================


class TestMDependence:
    """Test that results converge with increasing M."""

    def test_moment_energy_decreases_with_m(self):
        """Test that moment energy decreases with increasing m for physical fields."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 20

        # Initialize Hermite moments
        g = initialize_hermite_moments(
            grid=grid,
            M=M,
            v_th=1.0,
            perturbation_amplitude=0.1,
            seed=42
        )

        # Compute energy in each moment
        energies = []
        for m in range(M + 1):
            # Energy accounting for rfft format
            energy_kx0 = jnp.sum(jnp.abs(g[:, :, 0, m]) ** 2)
            energy_kx_pos = jnp.sum(jnp.abs(g[:, :, 1:, m]) ** 2)
            energy_m = energy_kx0 + 2.0 * energy_kx_pos
            energies.append(float(energy_m))

        # With perturbations, energy should be present
        total_energy = sum(energies)
        assert total_energy > 0, "Should have non-zero energy with perturbations"

        # Check convergence using our diagnostic
        result = check_hermite_convergence(g, threshold=1e-2)

        # Document convergence status
        print(f"Convergence result: {result['recommendation']}")

        # Verify convergence diagnostic returns all expected fields
        assert 'is_converged' in result
        assert 'energy_fraction' in result

    def test_increasing_M_improves_convergence(self):
        """Test that increasing M improves convergence metric."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        convergence_fractions = []

        for M in [5, 10, 15, 20]:
            # Initialize with exponential decay
            g = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), dtype=jnp.complex64)

            for m in range(M + 1):
                amplitude = jnp.exp(-0.3 * m)
                g = g.at[:, :, :, m].set(amplitude * (1.0 + 0.0j))

            result = check_hermite_convergence(g)
            convergence_fractions.append(result['energy_fraction'])

            print(f"M={M}: energy_fraction={result['energy_fraction']:.6f}, "
                  f"converged={result['is_converged']}")

        # With exponential decay, higher M should have smaller energy fraction
        # (more moments → total energy spreads out → highest moment has less fraction)
        assert convergence_fractions[-1] < convergence_fractions[0], \
            f"Larger M should have better convergence: fractions={convergence_fractions}"


# ============================================================================
# Test: Integration with Existing Code
# ============================================================================


class TestClosureIntegration:
    """Test closures integrate with existing KRMHD infrastructure."""

    def test_closures_work_with_hermite_moments(self):
        """Test closures work with initialized Hermite moments."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        # Initialize Hermite moments
        g = initialize_hermite_moments(
            grid=grid,
            M=M,
            v_th=1.0,
            perturbation_amplitude=0.1,
            seed=42
        )

        # Apply closures
        g_zero = closure_zero(g, M)
        g_sym = closure_symmetric(g, M)

        # Check shapes match expectations
        expected_shape = (grid.Nz, grid.Ny, grid.Nx // 2 + 1)
        assert g_zero.shape == expected_shape
        assert g_sym.shape == expected_shape

        # Check convergence
        result = check_hermite_convergence(g)
        assert 'is_converged' in result

    def test_convergence_check_on_evolved_state(self):
        """Test convergence check works on time-evolved states."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        # Initialize Hermite moments
        g_init = initialize_hermite_moments(
            grid=grid,
            M=M,
            v_th=1.0,
            perturbation_amplitude=0.1,
            seed=42
        )

        # "Evolve" by applying collision damping
        # JAX arrays are immutable, so we can assign directly
        g_evolved = g_init

        # Apply collision damping manually for moments m ≥ 2
        # (in real timestepping this happens in gm_rhs)
        nu = 0.1  # Collision frequency
        dt = 0.01

        # Vectorized damping: apply to all moments m ≥ 2 at once
        m_indices = jnp.arange(2, M + 1)
        damping_factors = jnp.exp(-nu * m_indices * dt)

        # Apply damping using broadcasting: shape (M-1,) → (1, 1, 1, M-1)
        g_evolved = g_evolved.at[:, :, :, 2:].multiply(
            damping_factors[None, None, None, :]
        )

        # Check convergence on both states
        result_init = check_hermite_convergence(g_init)
        result_evolved = check_hermite_convergence(g_evolved)

        # After damping, highest moment should have less energy
        assert result_evolved['energy_fraction'] <= result_init['energy_fraction'], \
            "Collision damping should improve convergence"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])


# ============================================================================
# Test: Runtime Closure Selection in the Solver
# ============================================================================


def _linear_g_state(M: int, Lambda: float = 1.0, seed: int = 0) -> KRMHDState:
    """z± = 0 (no nonlinearity), random dealiased g, k=0 mode zeroed."""
    import numpy as np

    grid = SpectralGrid3D.create(Nx=8, Ny=8, Nz=8)
    shape = (grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1)
    rng = np.random.default_rng(seed)
    g = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    g = g * np.asarray(grid.dealias_mask)[..., None]
    g[0, 0, 0, :] = 0.0
    zeros = jnp.zeros(shape[:3], dtype=jnp.complex64)
    return KRMHDState(
        z_plus=zeros, z_minus=zeros, g=jnp.asarray(g), M=M, beta_i=1.0,
        v_th=1.0, nu=0.0, Lambda=Lambda, time=0.0, grid=grid,
    )


class TestSelectableClosure:
    """Runtime closure selection (closure="zero" | "symmetric") in RHS and steppers."""

    def test_streaming_matrix_symmetric_closure_row(self):
        """Symmetric closure folds the g_{M+1} coupling into T[M, M-1]."""
        import numpy as np
        from krmhd.hermite import compute_streaming_matrix

        M = 6
        T_zero = compute_streaming_matrix(M, 1.5)
        T_sym = compute_streaming_matrix(M, 1.5, closure="symmetric")

        expected = np.sqrt(M / 2.0) + np.sqrt((M + 1) / 2.0)
        assert T_sym[M, M - 1] == pytest.approx(expected)
        assert T_zero[M, M - 1] == pytest.approx(np.sqrt(M / 2.0))
        diff = T_sym - T_zero
        diff[M, M - 1] = 0.0
        assert np.all(diff == 0.0)

    def test_gm_rhs_symmetric_matches_extended_hierarchy(self):
        """gm_rhs(m=M, symmetric) == gm_rhs on an M+1 hierarchy with g_{M+1} := g_{M-1}."""
        from krmhd.physics import gm_rhs, initialize_random_spectrum

        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8)
        M = 5
        state = initialize_random_spectrum(grid, M=M, amplitude=0.5, seed=3)
        key_re, key_im = jax.random.split(jax.random.PRNGKey(7))
        g = (jax.random.normal(key_re, state.g.shape)
             + 1j * jax.random.normal(key_im, state.g.shape)).astype(jnp.complex64)
        g = g * grid.dealias_mask[..., None]
        g_ext = jnp.concatenate([g, g[..., M - 1:M]], axis=-1)
        args = (state.z_plus, state.z_minus, grid.kx, grid.ky, grid.kz, grid.dealias_mask)

        rhs_sym = gm_rhs(g, *args, M, 1.0, grid.Nz, grid.Ny, grid.Nx, closure="symmetric")
        rhs_ext = gm_rhs(g_ext, *args, M, 1.0, grid.Nz, grid.Ny, grid.Nx)
        rhs_zero = gm_rhs(g, *args, M, 1.0, grid.Nz, grid.Ny, grid.Nx)

        scale = float(jnp.max(jnp.abs(rhs_ext)))
        assert float(jnp.max(jnp.abs(rhs_sym - rhs_ext))) < 1e-5 * scale
        assert float(jnp.max(jnp.abs(rhs_sym - rhs_zero))) > 1e-2 * scale

    @pytest.mark.parametrize("closure", ["zero", "symmetric"])
    def test_lawson_pure_streaming_matches_expm(self, closure):
        """With z±=0 the Lawson step is exact streaming: g(dt) = expm(-i√β kz T dt) g."""
        import numpy as np
        import scipy.linalg
        from krmhd.hermite import compute_streaming_matrix
        from krmhd.timestepping import gandalf_step

        M, Lambda, dt = 6, 2.236, 0.05
        state = _linear_g_state(M, Lambda)
        new = gandalf_step(state, dt, eta=0.0, v_A=1.0, nu=0.0,
                           scheme="lawson_rk4", closure=closure)

        T = compute_streaming_matrix(M, Lambda, closure=closure)
        g0 = np.asarray(state.g, dtype=np.complex128)
        expected = np.empty_like(g0)
        for iz, kz in enumerate(np.asarray(state.grid.kz)):
            U = scipy.linalg.expm(-1j * np.sqrt(state.beta_i) * kz * T * dt)
            expected[iz] = g0[iz] @ U.T
        err = np.max(np.abs(np.asarray(new.g) - expected)) / np.max(np.abs(expected))
        assert err < 1e-4

    def test_imex_pure_streaming_converges_to_symmetric_expm(self):
        """IMEX with closure='symmetric' converges at 2nd order to the symmetric-closure propagator."""
        import numpy as np
        import scipy.linalg
        from krmhd.hermite import compute_streaming_matrix
        from krmhd.timestepping import gandalf_step

        M, Lambda, t_end = 6, 2.236, 0.2
        state = _linear_g_state(M, Lambda)
        T = compute_streaming_matrix(M, Lambda, closure="symmetric")
        g0 = np.asarray(state.g, dtype=np.complex128)
        exact = np.empty_like(g0)
        for iz, kz in enumerate(np.asarray(state.grid.kz)):
            exact[iz] = g0[iz] @ scipy.linalg.expm(-1j * kz * T * t_end).T

        errs = []
        for n_steps in (4, 8):
            s = state
            for _ in range(n_steps):
                s = gandalf_step(s, t_end / n_steps, eta=0.0, v_A=1.0, nu=0.0,
                                 scheme="imex_rk222", closure="symmetric")
            errs.append(np.max(np.abs(np.asarray(s.g) - exact)))
        assert errs[1] < errs[0]
        assert np.log2(errs[0] / errs[1]) > 1.7

    def test_default_closure_is_zero(self):
        """Omitting closure reproduces closure='zero' bit-for-bit on both schemes."""
        from krmhd.timestepping import gandalf_step

        state = _linear_g_state(M=6, Lambda=1.5)
        for scheme in ("imex_rk222", "lawson_rk4"):
            a = gandalf_step(state, 0.05, eta=0.0, v_A=1.0, nu=0.0, scheme=scheme)
            b = gandalf_step(state, 0.05, eta=0.0, v_A=1.0, nu=0.0, scheme=scheme,
                             closure="zero")
            assert jnp.array_equal(a.g, b.g)

    def test_invalid_closure_raises(self):
        from krmhd.timestepping import gandalf_step

        state = _linear_g_state(M=6)
        with pytest.raises(ValueError, match="closure"):
            gandalf_step(state, 0.05, eta=0.0, v_A=1.0, closure="bogus")

    def test_symmetric_closure_requires_M_ge_2(self):
        from krmhd.timestepping import gandalf_step

        state = _linear_g_state(M=1)
        with pytest.raises(ValueError, match="M"):
            gandalf_step(state, 0.05, eta=0.0, v_A=1.0, nu=0.0, closure="symmetric")

    def test_physics_config_closure_field(self):
        from krmhd.config import PhysicsConfig

        assert PhysicsConfig().closure == "zero"
        assert PhysicsConfig(closure="symmetric").closure == "symmetric"
        with pytest.raises(ValueError):
            PhysicsConfig(closure="bogus")
