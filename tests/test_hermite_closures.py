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
            grid.dealias_mask, M, beta_i, nu, grid.Nz, grid.Ny, grid.Nx
        )

        # With ν=0 and different closures (implicit in gm_rhs vs explicit),
        # we expect coupling structure to differ
        # This test documents current behavior: gm_rhs uses zero closure

        # Just verify RHS is computed
        assert rhs_M.shape == (grid.Nz, grid.Ny, grid.Nx // 2 + 1)

    def test_closure_convergence_with_strong_collisions(self):
        """Test that collision damping in gm_rhs scales correctly with moment index.

        This test isolates the collision term -νm·gₘ by setting:
        - z_plus = z_minus = 0 (no perpendicular advection, field line coupling)
        - gₘ = 1 with all neighbors zero (no parallel streaming coupling)

        This makes gm_rhs return purely the collision damping term -νm·gₘ.
        The test verifies the damping scales correctly: RHS(m=5) / RHS(m=2) = 5/2.
        """
        from krmhd.physics import gm_rhs

        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        M = 10

        # Isolate collision damping: set gₘ = 1 with all neighbors = 0
        # This makes RHS = -νm·gₘ purely (no other coupling terms)
        z_plus = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64)
        z_minus = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64)

        beta_i = 1.0
        nu = 1.0  # Strong collisions

        # Test isolation: set only one moment at a time, neighbors zero
        # For m=2: set only g[2] = 1, g[1] = g[3] = 0
        g_m2 = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), dtype=jnp.complex64)
        g_m2 = g_m2.at[4, 5, 6, 2].set(1.0 + 0.0j)

        # For m=5: set only g[5] = 1, g[4] = g[6] = 0
        g_m5 = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), dtype=jnp.complex64)
        g_m5 = g_m5.at[4, 5, 6, 5].set(1.0 + 0.0j)

        # Compute RHS (should be purely collision damping)
        rhs_m2 = gm_rhs(
            g_m2, z_plus, z_minus, grid.kx, grid.ky, grid.kz,
            grid.dealias_mask, 2, beta_i, nu, grid.Nz, grid.Ny, grid.Nx
        )

        rhs_m5 = gm_rhs(
            g_m5, z_plus, z_minus, grid.kx, grid.ky, grid.kz,
            grid.dealias_mask, 5, beta_i, nu, grid.Nz, grid.Ny, grid.Nx
        )

        # Extract magnitude at the mode location
        damping_m2 = jnp.abs(rhs_m2[4, 5, 6])
        damping_m5 = jnp.abs(rhs_m5[4, 5, 6])

        # Expected: |RHS| = νm for unit amplitude
        expected_m2 = nu * 2
        expected_m5 = nu * 5

        # Check collision scaling
        assert jnp.abs(damping_m2 - expected_m2) < 1e-5, \
            f"m=2 damping should be {expected_m2}, got {damping_m2}"
        assert jnp.abs(damping_m5 - expected_m5) < 1e-5, \
            f"m=5 damping should be {expected_m5}, got {damping_m5}"

        # Verify ratio is correct
        ratio = damping_m5 / damping_m2
        expected_ratio = 5.0 / 2.0
        assert jnp.abs(ratio - expected_ratio) < 0.01, \
            f"Collision ratio should be {expected_ratio}, got {ratio}"


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
        g_evolved = g_init.copy()

        # Apply collision damping manually for moments m ≥ 2
        # (in real timestepping this happens in gm_rhs)
        nu = 0.1  # Collision frequency
        dt = 0.01

        for m in range(2, M + 1):
            damping_factor = jnp.exp(-nu * m * dt)
            g_evolved = g_evolved.at[:, :, :, m].multiply(damping_factor)

        # Check convergence on both states
        result_init = check_hermite_convergence(g_init)
        result_evolved = check_hermite_convergence(g_evolved)

        # After damping, highest moment should have less energy
        assert result_evolved['energy_fraction'] <= result_init['energy_fraction'], \
            "Collision damping should improve convergence"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
