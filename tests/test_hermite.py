"""
Comprehensive test suite for Hermite polynomial basis functions.

Tests cover:
1. Hermite polynomial recurrence relations
2. Orthogonality with Maxwellian weight: ∫ H_m·H_n·F_0 dv = δ_{mn}·2^m·m!
3. Normalization constants
4. Round-trip consistency: distribution → moments → distribution
5. Edge cases and numerical stability

All tests use high-resolution velocity grids for accurate integration.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.scipy as jsp

from krmhd.hermite import (
    hermite_polynomial,
    hermite_polynomials_all,
    hermite_normalization,
    hermite_basis_function,
    distribution_to_moments,
    moments_to_distribution,
    check_orthogonality,
)


# Test fixtures
@pytest.fixture
def velocity_grid_fine():
    """Fine velocity grid for accurate integration: [-10, 10] with 2001 points."""
    return jnp.linspace(-10.0, 10.0, 2001)


@pytest.fixture
def velocity_grid_standard():
    """Standard velocity grid: [-10, 10] with 1000 points."""
    return jnp.linspace(-10.0, 10.0, 1000)


@pytest.fixture
def thermal_velocity():
    """Standard thermal velocity."""
    return 1.0


# ============================================================================
# Test: Hermite Polynomial Recurrence Relations
# ============================================================================


def test_hermite_base_cases():
    """Test H_0(v) = 1 and H_1(v) = 2v."""
    v = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    H_0 = hermite_polynomial(0, v)
    assert jnp.allclose(H_0, jnp.ones_like(v)), "H_0 should be all ones"

    H_1 = hermite_polynomial(1, v)
    assert jnp.allclose(H_1, 2.0 * v), "H_1 should equal 2v"


def test_hermite_recurrence_relation():
    """Verify H_{m+1} = 2v·H_m - 2m·H_{m-1} explicitly."""
    v = jnp.linspace(-3.0, 3.0, 50)

    for m in range(1, 10):
        H_m_minus_1 = hermite_polynomial(m - 1, v)
        H_m = hermite_polynomial(m, v)
        H_m_plus_1_computed = hermite_polynomial(m + 1, v)

        # Apply recurrence relation
        H_m_plus_1_recurrence = 2.0 * v * H_m - 2.0 * m * H_m_minus_1

        assert jnp.allclose(
            H_m_plus_1_computed, H_m_plus_1_recurrence, rtol=1e-5, atol=1e-6
        ), f"Recurrence relation failed for m={m}"


def test_hermite_known_values():
    """Test against known Hermite polynomial values."""
    v = jnp.array([0.0, 1.0, -1.0])

    # H_2(v) = 4v² - 2
    H_2 = hermite_polynomial(2, v)
    expected_H_2 = 4.0 * v**2 - 2.0
    assert jnp.allclose(H_2, expected_H_2), "H_2 incorrect"

    # H_3(v) = 8v³ - 12v
    H_3 = hermite_polynomial(3, v)
    expected_H_3 = 8.0 * v**3 - 12.0 * v
    assert jnp.allclose(H_3, expected_H_3), "H_3 incorrect"

    # H_4(v) = 16v⁴ - 48v² + 12
    H_4 = hermite_polynomial(4, v)
    expected_H_4 = 16.0 * v**4 - 48.0 * v**2 + 12.0
    assert jnp.allclose(H_4, expected_H_4), "H_4 incorrect"


def test_hermite_negative_order_raises():
    """Ensure negative orders raise ValueError."""
    v = jnp.array([0.0, 1.0])

    with pytest.raises(ValueError, match="Hermite order must be non-negative"):
        hermite_polynomial(-1, v)


# ============================================================================
# Test: Normalization Constants
# ============================================================================


def test_hermite_normalization_values():
    """Check normalization constants 1/√(2^m · m! · √π)."""
    # N_m = 1 / √(2^m · m! · √π)
    pi_fourth = jnp.sqrt(jnp.sqrt(jnp.pi))

    assert jnp.isclose(
        hermite_normalization(0), 1.0 / pi_fourth, rtol=1e-6
    ), f"N_0 should be 1/π^(1/4) ≈ 0.751"

    assert jnp.isclose(
        hermite_normalization(1), 1.0 / (jnp.sqrt(2.0) * pi_fourth), rtol=1e-6
    ), f"N_1 should be 1/(√2·π^(1/4)) ≈ 0.531"

    assert jnp.isclose(
        hermite_normalization(2), 1.0 / (jnp.sqrt(8.0) * pi_fourth), rtol=1e-6
    ), f"N_2 should be 1/(√8·π^(1/4)) ≈ 0.266"

    assert jnp.isclose(
        hermite_normalization(3), 1.0 / (jnp.sqrt(48.0) * pi_fourth), rtol=1e-6
    ), f"N_3 should be 1/(√48·π^(1/4)) ≈ 0.108"


def test_hermite_normalization_scaling():
    """Verify normalization decreases with m."""
    norms = [hermite_normalization(m) for m in range(10)]

    # Should be monotonically decreasing
    for i in range(len(norms) - 1):
        assert norms[i] > norms[i + 1], f"Normalization not decreasing at m={i}"


# ============================================================================
# Test: Orthogonality with Maxwellian Weight
# ============================================================================


def test_orthogonality_diagonal(velocity_grid_fine):
    """Test ∫ ψ_m² dv = 1 for orthonormal Hermite functions."""
    v = velocity_grid_fine
    v_th = 1.0

    for m in range(8):
        result = check_orthogonality(m, m, v, v_th, rtol=1e-3)

        assert result['is_orthogonal'], (
            f"Orthonormality failed for m=n={m}:\n"
            f"  Computed: {result['integral']}\n"
            f"  Expected: {result['expected']}\n"
            f"  Relative error: {result['relative_error']}"
        )


def test_orthogonality_off_diagonal(velocity_grid_fine):
    """Test ∫ ψ_m · ψ_n dv = 0 for m ≠ n."""
    v = velocity_grid_fine
    v_th = 1.0

    # Test several pairs
    pairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5)]

    for m, n in pairs:
        # For off-diagonal, we use absolute tolerance since expected is 0
        result = check_orthogonality(m, n, v, v_th, rtol=1e-6)

        # Check absolute error is small
        abs_error = abs(result['integral'])
        assert abs_error < 1e-5, (
            f"Orthogonality failed for m={m}, n={n}:\n"
            f"  Computed: {result['integral']}\n"
            f"  Expected: 0\n"
            f"  Absolute error: {abs_error}"
        )


def test_orthogonality_matrix(velocity_grid_standard):
    """Compute full orthonormality matrix for M=0..6."""
    v = velocity_grid_standard
    v_th = 1.0
    M_max = 6

    ortho_matrix = jnp.zeros((M_max + 1, M_max + 1))

    for m in range(M_max + 1):
        for n in range(M_max + 1):
            result = check_orthogonality(m, n, v, v_th)
            ortho_matrix = ortho_matrix.at[m, n].set(result['integral'])

    # Diagonal should be 1 (orthonormal)
    for m in range(M_max + 1):
        expected_diag = 1.0
        assert jnp.isclose(
            ortho_matrix[m, m], expected_diag, rtol=1e-3
        ), f"Diagonal element ({m},{m}) = {ortho_matrix[m, m]}, expected 1.0"

    # Off-diagonal should be near zero
    for m in range(M_max + 1):
        for n in range(M_max + 1):
            if m != n:
                assert jnp.isclose(
                    ortho_matrix[m, n], 0.0, atol=1e-4
                ), f"Off-diagonal element ({m},{n}) = {ortho_matrix[m,n]} not zero"


# ============================================================================
# Test: Basis Functions with Maxwellian Weight
# ============================================================================


def test_hermite_basis_normalization(velocity_grid_fine):
    """Test that basis functions integrate to correct normalization."""
    v = velocity_grid_fine
    v_th = 1.0

    # ψ_0 should have norm 1: ∫ ψ_0² dv = 1
    psi_0 = hermite_basis_function(0, v, v_th)
    norm_sq_0 = jnp.trapezoid(psi_0**2, v)
    assert jnp.isclose(
        norm_sq_0, 1.0, rtol=1e-3
    ), f"||ψ_0||² = {norm_sq_0}, expected 1.0"

    # ψ_0 integrates to ∫ ψ_0 dv = √2 · π^(1/4)
    # Because ∫ exp(-v²/(2v_th²)) dv = √(2πv_th²)
    # And ψ_0 = N_0 · exp(-v²/(2v_th²)) where N_0 = 1/(π^(1/4)·√v_th)
    # So ∫ ψ_0 = √(2πv_th²) / (π^(1/4)·√v_th) = √2 · π^(1/4)
    integral_0 = jnp.trapezoid(psi_0, v)
    expected_int_0 = jnp.sqrt(2.0) * jnp.sqrt(jnp.sqrt(jnp.pi))
    assert jnp.isclose(
        integral_0, expected_int_0, rtol=1e-3
    ), f"∫ ψ_0 dv = {integral_0}, expected {expected_int_0}"

    # Odd basis functions should integrate to 0 (H_m is odd function)
    for m in [1, 3, 5]:
        psi_m = hermite_basis_function(m, v, v_th)
        integral_m = jnp.trapezoid(psi_m, v)
        assert jnp.isclose(
            integral_m, 0.0, atol=1e-5
        ), f"∫ ψ_{m} dv = {integral_m}, expected 0 (odd function)"


def test_hermite_basis_orthonormality(velocity_grid_fine):
    """Test ∫ ψ_m · ψ_n dv = δ_{mn}."""
    v = velocity_grid_fine
    v_th = 1.0
    M_max = 5

    for m in range(M_max + 1):
        psi_m = hermite_basis_function(m, v, v_th)
        for n in range(M_max + 1):
            psi_n = hermite_basis_function(n, v, v_th)

            integral = jnp.trapezoid(psi_m * psi_n, v)

            if m == n:
                expected = 1.0
            else:
                expected = 0.0

            assert jnp.isclose(integral, expected, atol=1e-5), (
                f"Basis orthonormality failed for m={m}, n={n}:\n"
                f"  Computed: {integral}\n"
                f"  Expected: {expected}"
            )


# ============================================================================
# Test: Distribution ↔ Moments Transformations
# ============================================================================


def test_maxwellian_moments(velocity_grid_fine):
    """Pure Maxwellian should have g_0 = π^(1/4)/√(2π), others = 0."""
    v = velocity_grid_fine
    v_th = 1.0
    M = 10

    # Normalized Maxwellian distribution
    g_v = jnp.exp(-0.5 * v**2) / jnp.sqrt(2.0 * jnp.pi)

    moments = distribution_to_moments(g_v, v, M, v_th)

    # g_0 should be π^(1/4) / √(2π) ≈ 0.531
    expected_g0 = jnp.sqrt(jnp.sqrt(jnp.pi)) / jnp.sqrt(2.0 * jnp.pi)
    assert jnp.isclose(moments[0], expected_g0, rtol=1e-3), (
        f"g_0 = {moments[0]}, expected {expected_g0}"
    )

    # All higher moments should be ~0
    for m in range(1, M + 1):
        assert jnp.isclose(
            moments[m], 0.0, atol=1e-5
        ), f"g_{m} = {moments[m]}, expected 0"


def test_shifted_maxwellian_moments(velocity_grid_fine):
    """Shifted Maxwellian should have non-zero g_1."""
    v = velocity_grid_fine
    v_th = 1.0
    v_drift = 0.5
    M = 10

    # Shifted Maxwellian: exp(-(v - v_drift)²/2) / √(2π)
    g_v = jnp.exp(-0.5 * (v - v_drift) ** 2) / jnp.sqrt(2.0 * jnp.pi)

    moments = distribution_to_moments(g_v, v, M, v_th)

    # g_0 should be positive (but not same as centered Maxwellian)
    assert moments[0] > 0.1, f"g_0 = {moments[0]} should be positive"

    # g_1 should be non-zero (proportional to drift)
    # For small drift, g_1 should have same sign as drift
    if v_drift > 0:
        assert moments[1] > 0.05, f"g_1 = {moments[1]} should be positive for positive drift"
    else:
        assert moments[1] < -0.05, f"g_1 = {moments[1]} should be negative for negative drift"

    # Total "energy" in moments should be reasonable
    moment_norm = jnp.sqrt(jnp.sum(moments**2))
    assert 0.3 < moment_norm < 1.0, f"Moment norm = {moment_norm} out of expected range"


def test_round_trip_consistency(velocity_grid_fine):
    """Test distribution → moments → distribution recovers original."""
    v = velocity_grid_fine
    v_th = 1.0
    M = 20

    # Test with several distributions
    test_distributions = [
        # Pure Maxwellian
        jnp.exp(-0.5 * v**2) / jnp.sqrt(2.0 * jnp.pi),
        # Shifted Maxwellian
        jnp.exp(-0.5 * (v - 0.5) ** 2) / jnp.sqrt(2.0 * jnp.pi),
        # Broader distribution
        jnp.exp(-0.125 * v**2) / jnp.sqrt(2.0 * jnp.pi * 4.0),
    ]

    for i, g_v_original in enumerate(test_distributions):
        # Forward: distribution → moments
        moments = distribution_to_moments(g_v_original, v, M, v_th)

        # Backward: moments → distribution
        g_v_reconstructed = moments_to_distribution(moments, v, v_th, M)

        # Compare
        max_error = jnp.max(jnp.abs(g_v_original - g_v_reconstructed))
        rel_error = max_error / jnp.max(jnp.abs(g_v_original))

        assert rel_error < 0.01, (
            f"Round-trip test {i} failed:\n"
            f"  Max absolute error: {max_error}\n"
            f"  Relative error: {rel_error}"
        )


def test_round_trip_convergence_with_M(velocity_grid_standard):
    """Verify reconstruction improves with increasing M."""
    v = velocity_grid_standard
    v_th = 1.0

    # Slightly perturbed Maxwellian
    g_v_original = jnp.exp(-0.5 * v**2) * (1.0 + 0.1 * jnp.sin(v))
    g_v_original /= jnp.trapezoid(g_v_original, v)  # Normalize

    errors = []
    M_values = [5, 10, 15, 20, 25]

    for M in M_values:
        moments = distribution_to_moments(g_v_original, v, M, v_th)
        g_v_reconstructed = moments_to_distribution(moments, v, v_th, M)

        error = jnp.sqrt(jnp.trapezoid((g_v_original - g_v_reconstructed) ** 2, v))
        errors.append(error)

    # Error should generally decrease with M (allow small fluctuations)
    # Check that final error is significantly smaller than initial error
    assert errors[-1] < errors[0] * 0.5, (
        f"Error not decreasing sufficiently: "
        f"M={M_values[0]} error={errors[0]}, M={M_values[-1]} error={errors[-1]}"
    )

    # Check that most steps show improvement (allow 1 non-improvement)
    improvements = sum(1 for i in range(len(errors) - 1) if errors[i] > errors[i + 1])
    assert improvements >= len(errors) - 2, (
        f"Too many non-improvements in convergence: {improvements}/{len(errors)-1}"
    )


# ============================================================================
# Test: Batched Operations
# ============================================================================


def test_distribution_to_moments_batched(velocity_grid_standard):
    """Test batched distribution → moments transformation."""
    v = velocity_grid_standard
    v_th = 1.0
    M = 10

    # Create batch of distributions (shape: [batch_size, Nv])
    batch_size = 5
    v_drifts = jnp.linspace(-1.0, 1.0, batch_size)

    # Batched distributions (normalized Maxwellians with different drifts)
    g_v_batch = jax.vmap(
        lambda v_drift: jnp.exp(-0.5 * (v - v_drift) ** 2) / jnp.sqrt(2.0 * jnp.pi)
    )(v_drifts)

    # Compute moments
    moments_batch = distribution_to_moments(g_v_batch, v, M, v_th)

    assert moments_batch.shape == (
        batch_size,
        M + 1,
    ), f"Expected shape {(batch_size, M+1)}, got {moments_batch.shape}"

    # All g_0 should be positive
    assert jnp.all(moments_batch[:, 0] > 0.1), "All g_0 should be positive"

    # Middle distribution (zero drift) should have g_0 ≈ π^(1/4) / √(2π)
    expected_g0_centered = jnp.sqrt(jnp.sqrt(jnp.pi)) / jnp.sqrt(2.0 * jnp.pi)
    mid_idx = batch_size // 2
    assert jnp.isclose(
        moments_batch[mid_idx, 0], expected_g0_centered, rtol=1e-2
    ), f"Centered g_0 = {moments_batch[mid_idx, 0]}, expected {expected_g0_centered}"

    # g_1 should correlate with drift: positive drift → positive g_1
    assert moments_batch[0, 1] < 0, "Negative drift should give negative g_1"
    assert moments_batch[-1, 1] > 0, "Positive drift should give positive g_1"
    assert jnp.abs(moments_batch[mid_idx, 1]) < 0.05, "Zero drift should give ~zero g_1"


def test_moments_to_distribution_batched(velocity_grid_standard):
    """Test batched moments → distribution reconstruction."""
    v = velocity_grid_standard
    v_th = 1.0
    M = 15

    # Create batch of moment sets
    batch_size = 3
    moments_batch = jnp.zeros((batch_size, M + 1))

    # First: pure ψ_0 with g_0 = π^(1/4)/√(2π) → gives Maxwellian
    expected_g0 = jnp.sqrt(jnp.sqrt(jnp.pi)) / jnp.sqrt(2.0 * jnp.pi)
    moments_batch = moments_batch.at[0, 0].set(expected_g0)

    # Second: same as first plus some g_1 (drift)
    moments_batch = moments_batch.at[1, 0].set(expected_g0)
    moments_batch = moments_batch.at[1, 1].set(0.1)

    # Third: same as first plus some g_2
    moments_batch = moments_batch.at[2, 0].set(expected_g0)
    moments_batch = moments_batch.at[2, 2].set(0.1)

    # Reconstruct distributions
    g_v_batch = moments_to_distribution(moments_batch, v, v_th, M)

    assert g_v_batch.shape == (
        batch_size,
        len(v),
    ), f"Expected shape {(batch_size, len(v))}, got {g_v_batch.shape}"

    # First should be Maxwellian
    expected_maxwellian = jnp.exp(-0.5 * v**2) / jnp.sqrt(2.0 * jnp.pi)
    rel_error = jnp.max(jnp.abs(g_v_batch[0] - expected_maxwellian)) / jnp.max(jnp.abs(expected_maxwellian))
    assert rel_error < 1e-2, f"First distribution not Maxwellian, rel_error = {rel_error}"


# ============================================================================
# Test: Edge Cases and Numerical Stability
# ============================================================================


def test_hermite_large_order():
    """Test Hermite polynomials remain stable at high orders."""
    v = jnp.array([0.0, 0.5, 1.0])

    # Should compute without overflow for reasonable orders
    for m in [10, 15, 20]:
        H_m = hermite_polynomial(m, v)
        assert jnp.all(jnp.isfinite(H_m)), f"H_{m} contains non-finite values"


def test_hermite_extreme_velocities():
    """Test behavior at large |v| values."""
    v_extreme = jnp.array([-50.0, -10.0, 0.0, 10.0, 50.0])

    # Should compute without overflow
    for m in range(5):
        H_m = hermite_polynomial(m, v_extreme)
        assert jnp.all(jnp.isfinite(H_m)), f"H_{m} fails at extreme velocities"


def test_zero_velocity():
    """Test all functions handle v=0 correctly."""
    v = jnp.array([0.0])
    v_th = 1.0

    # Hermite polynomials at v=0
    for m in range(10):
        H_m = hermite_polynomial(m, v)
        assert jnp.isfinite(H_m[0]), f"H_{m}(0) not finite"

    # Basis function at v=0
    psi_0 = hermite_basis_function(0, v, v_th)
    assert jnp.isfinite(psi_0[0]), "ψ_0(0) not finite"


def test_thermal_velocity_scaling(velocity_grid_standard):
    """Test that v_th scaling works correctly."""
    v = velocity_grid_standard

    # Maxwellian with v_th = 2.0
    v_th = 2.0
    g_v = jnp.exp(-0.5 * (v / v_th) ** 2) / jnp.sqrt(2.0 * jnp.pi * v_th**2)

    moments = distribution_to_moments(g_v, v, M=15, v_th=v_th)

    # g_0 should be positive and reasonable
    expected_g0_approx = jnp.sqrt(jnp.sqrt(jnp.pi)) / jnp.sqrt(2.0 * jnp.pi)
    assert 0.3 < moments[0] < 0.7, f"g_0 = {moments[0]} with v_th={v_th}, expected ~{expected_g0_approx}"

    # Reconstruct
    g_v_reconstructed = moments_to_distribution(moments, v, v_th=v_th, M=15)

    # Should match original reasonably well
    rel_error = jnp.max(jnp.abs(g_v - g_v_reconstructed)) / jnp.max(jnp.abs(g_v))
    assert rel_error < 0.02, f"v_th scaling test failed with rel_error {rel_error}"


# ============================================================================
# Test: JIT Compilation
# ============================================================================


def test_jit_compilation():
    """Verify all functions are JIT-compatible."""
    v = jnp.linspace(-5.0, 5.0, 100)
    v_th = 1.0
    M = 10

    # Test each function can be JIT-compiled and executed
    try:
        _ = jax.jit(hermite_polynomial, static_argnames=['m'])(5, v)
        _ = jax.jit(hermite_normalization)(5)
        _ = jax.jit(hermite_basis_function, static_argnames=['m'])(5, v, v_th)
        _ = jax.jit(hermite_polynomials_all, static_argnames=['M'])(M, v)

        g_v = jnp.exp(-0.5 * v**2) / jnp.sqrt(2.0 * jnp.pi)
        _ = jax.jit(distribution_to_moments, static_argnames=['M'])(g_v, v, M, v_th)

        moments = jnp.ones(M + 1) * 0.1
        _ = jax.jit(moments_to_distribution, static_argnames=['M'])(
            moments, v, v_th, M
        )

        # Note: check_orthogonality is not JIT-compatible (uses runtime m, n values)

    except Exception as e:
        pytest.fail(f"JIT compilation failed: {e}")


# ============================================================================
# Test: Physical Interpretations
# ============================================================================


def test_moment_physical_meaning(velocity_grid_fine):
    """Verify moments correspond to physical quantities."""
    v = velocity_grid_fine
    v_th = 1.0

    # For a Maxwellian centered at v_drift with temperature T_th
    v_drift = 0.3
    g_v = jnp.exp(-0.5 * (v - v_drift) ** 2 / v_th**2) / jnp.sqrt(
        2.0 * jnp.pi * v_th**2
    )

    moments = distribution_to_moments(g_v, v, M=10, v_th=v_th)

    # g_0 should be positive (shifted Maxwellians have different g_0)
    assert moments[0] > 0.1, "g_0 should be positive"

    # g_1 should correlate with v_drift (positive drift → positive g_1)
    if v_drift > 0:
        assert moments[1] > 0.05, f"g_1 = {moments[1]} should be positive for positive drift"

    # Total moment energy should be reasonable
    moment_energy = jnp.sum(moments**2)
    assert 0.1 < moment_energy < 1.0, f"Moment energy = {moment_energy} out of range"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
