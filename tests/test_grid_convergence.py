"""
Unit tests for grid convergence validation infrastructure.

Tests basic functionality of grid convergence testing tools without
running expensive full convergence sweeps.
"""

import pytest
import jax.numpy as jnp
import sys
from pathlib import Path

# Add examples directory to path to import grid_convergence_tests
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from grid_convergence_tests import (
    compute_l2_error,
    compute_l2_norm,
    analytical_alfven_solution,
    test_alfven_wave_convergence,
    test_orszag_tang_convergence,
)
from krmhd import SpectralGrid3D, initialize_alfven_wave


class TestL2ErrorComputation:
    """Test L2 error and norm calculations."""

    def test_l2_error_identity(self):
        """L2 error of identical states should be zero."""
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=4, Lx=1.0, Ly=1.0, Lz=1.0)
        state = initialize_alfven_wave(grid, M=2, amplitude=0.1)

        error = compute_l2_error(state, state)

        assert error < 1e-14, f"Expected zero error for identical states, got {error}"

    def test_l2_error_symmetry(self):
        """L2 error should be symmetric: error(a, b) == error(b, a)."""
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=4, Lx=1.0, Ly=1.0, Lz=1.0)
        state1 = initialize_alfven_wave(grid, M=2, kz_mode=1, amplitude=0.1)
        state2 = initialize_alfven_wave(grid, M=2, kz_mode=2, amplitude=0.15)

        error_12 = compute_l2_error(state1, state2)
        error_21 = compute_l2_error(state2, state1)

        assert abs(error_12 - error_21) < 1e-14, \
            f"L2 error should be symmetric: {error_12} != {error_21}"

    def test_l2_norm_positive(self):
        """L2 norm should be positive for non-zero states."""
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=4, Lx=1.0, Ly=1.0, Lz=1.0)
        state = initialize_alfven_wave(grid, M=2, amplitude=0.1)

        norm = compute_l2_norm(state)

        assert norm > 0, f"L2 norm should be positive for non-zero state, got {norm}"

    def test_l2_norm_scaling(self):
        """L2 norm should scale linearly with amplitude."""
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=4, Lx=1.0, Ly=1.0, Lz=1.0)
        state1 = initialize_alfven_wave(grid, M=2, amplitude=0.1)
        state2 = initialize_alfven_wave(grid, M=2, amplitude=0.2)

        norm1 = compute_l2_norm(state1)
        norm2 = compute_l2_norm(state2)

        # Should be approximately 2× (within floating point tolerance)
        ratio = norm2 / norm1
        assert abs(ratio - 2.0) < 1e-10, \
            f"L2 norm should scale linearly with amplitude: {ratio} != 2.0"


class TestAnalyticalSolution:
    """Test analytical Alfvén wave solution."""

    def test_analytical_solution_at_t0(self):
        """Analytical solution at t=0 should match initial condition."""
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=4, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)
        kz_mode = 1
        amplitude = 0.1
        v_A = 1.0
        M = 2

        # Get initial condition
        state0 = initialize_alfven_wave(grid, M=M, kz_mode=kz_mode, amplitude=amplitude)

        # Get analytical solution at t=0
        state_analytical = analytical_alfven_solution(grid, t=0.0, kz_mode=kz_mode,
                                                     amplitude=amplitude, v_A=v_A, M=M)

        # Should be identical (or very close due to phase factor exp(0) = 1)
        error = compute_l2_error(state0, state_analytical)

        assert error < 1e-10, \
            f"Analytical solution at t=0 should match initial condition, error = {error}"

    def test_analytical_solution_preserves_energy(self):
        """Analytical solution should preserve L2 norm (energy)."""
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=4, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)
        kz_mode = 1
        amplitude = 0.1
        v_A = 1.0
        M = 2

        # Get analytical solutions at different times
        state_t0 = analytical_alfven_solution(grid, t=0.0, kz_mode=kz_mode,
                                             amplitude=amplitude, v_A=v_A, M=M)
        state_t1 = analytical_alfven_solution(grid, t=1.0, kz_mode=kz_mode,
                                             amplitude=amplitude, v_A=v_A, M=M)

        norm_t0 = compute_l2_norm(state_t0)
        norm_t1 = compute_l2_norm(state_t1)

        # Energy (L2 norm squared) should be conserved
        # Note: float32 precision limits this to ~1e-7
        assert abs(norm_t1 - norm_t0) / norm_t0 < 1e-6, \
            f"Analytical solution should preserve energy, ΔE/E = {abs(norm_t1 - norm_t0) / norm_t0}"


class TestConvergence:
    """Test convergence testing infrastructure (smoke tests)."""

    @pytest.mark.slow
    def test_alfven_convergence_runs(self):
        """Smoke test: Alfvén convergence test should complete without errors."""
        # Run with minimal resolutions for speed
        results = test_alfven_wave_convergence(
            resolutions=[16, 32],
            t_final=0.5,
            M=2,
        )

        # Check that results have expected keys
        assert 'N' in results
        assert 'errors' in results
        assert 'rel_errors' in results
        assert 'times' in results

        # Check that we got results for both resolutions
        assert len(results['N']) == 2
        assert len(results['errors']) == 2

    @pytest.mark.slow
    def test_orszag_tang_convergence_runs(self):
        """Smoke test: Orszag-Tang convergence test should complete without errors."""
        # Run with minimal resolutions for speed
        results = test_orszag_tang_convergence(
            resolutions=[16, 24],
            reference_resolution=32,
            t_final=0.05,
        )

        # Check that results have expected keys
        assert 'N' in results
        assert 'errors' in results
        assert 'rel_errors' in results
        assert 'times' in results
        assert 'reference_N' in results

        # Check that we got results for both resolutions
        assert len(results['N']) == 2
        assert len(results['errors']) == 2
        assert results['reference_N'] == 32

    @pytest.mark.slow
    def test_convergence_error_decreases(self):
        """Higher resolution should have lower error (for smooth problems)."""
        # Use very short time and simple wave to ensure smoothness
        results = test_alfven_wave_convergence(
            resolutions=[16, 24, 32],
            t_final=0.1,  # Very short time
            M=2,
        )

        errors = results['rel_errors']

        # Errors should generally decrease (though may not be monotonic for Alfvén)
        # At minimum, highest resolution should have lower error than lowest
        assert errors[-1] <= errors[0], \
            f"Higher resolution should have lower error: {errors[-1]} > {errors[0]}"
