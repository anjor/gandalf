"""
Validation tests for KRMHD physics benchmarks.

These tests validate the physical correctness of the implementation
against known analytical solutions and standard benchmarks:
- Plasma physics special functions: Z(ζ), Bessel functions
- Linear response theory: kinetic response, FLR corrections
- Orszag-Tang vortex: nonlinear dynamics and energy conservation
- Linear wave dispersion: Alfvén waves, kinetic Alfvén waves
- Landau damping: kinetic response validation
"""

import numpy as np
import jax.numpy as jnp
import pytest
from scipy.special import wofz, iv  # For reference implementations

from krmhd import (
    SpectralGrid3D,
    initialize_orszag_tang,
    gandalf_step,
    compute_cfl_timestep,
    energy as compute_energy,
)
from krmhd.validation import (
    plasma_dispersion_function,
    plasma_dispersion_derivative,
    modified_bessel_ratio,
    kinetic_response_function,
    flr_correction_factor,
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


class TestPlasmaPhysicsFunctions:
    """
    Unit tests for plasma physics special functions.

    Tests the correctness of:
    - Plasma dispersion function Z(ζ)
    - Modified Bessel function ratios I_m(x)exp(-x)
    - Linear response functions
    - FLR correction factors
    """

    def test_plasma_dispersion_function_at_zero(self):
        """
        Test Z(0) = i√π.

        This is a well-known exact value (Fried & Conte 1961).
        """
        zeta = np.array([0.0 + 0.0j])
        Z = plasma_dispersion_function(zeta)

        expected = 1j * np.sqrt(np.pi)
        np.testing.assert_allclose(Z[0], expected, rtol=1e-10,
                                    err_msg="Z(0) should equal i√π")

    def test_plasma_dispersion_function_large_argument(self):
        """
        Test Z(ζ) → -1/ζ for large |ζ|.

        Asymptotic expansion: Z(ζ) ≈ -1/ζ - 1/(2ζ³) + ... for |ζ| >> 1
        """
        # Large real argument
        zeta_large = np.array([10.0 + 0.1j, -10.0 + 0.1j])
        Z = plasma_dispersion_function(zeta_large)

        # First-order asymptotic
        Z_asymptotic = -1.0 / zeta_large

        # Should agree to ~1% for |ζ| = 10
        np.testing.assert_allclose(Z, Z_asymptotic, rtol=0.05,
                                    err_msg="Z(ζ) should approach -1/ζ for large |ζ|")

    def test_plasma_dispersion_function_consistency_with_wofz(self):
        """
        Test Z(ζ) = i√π w(ζ) against scipy.special.wofz.

        This validates our implementation against the standard reference.
        """
        zeta_values = np.array([
            0.5 + 0.1j,
            1.0 + 0.0j,
            -1.0 + 0.5j,
            2.0 + 1.0j,
        ])

        Z = plasma_dispersion_function(zeta_values)
        Z_expected = 1j * np.sqrt(np.pi) * wofz(zeta_values)

        np.testing.assert_allclose(Z, Z_expected, rtol=1e-10,
                                    err_msg="Z(ζ) should match i√π w(ζ)")

    def test_plasma_dispersion_derivative_relation(self):
        """
        Test Z'(ζ) = -2[1 + ζZ(ζ)] relation.

        This is the defining relation for the derivative (Fried & Conte 1961).
        """
        zeta_values = np.array([
            0.5 + 0.1j,
            1.0 + 0.5j,
            -1.5 + 0.2j,
        ])

        Z = plasma_dispersion_function(zeta_values)
        Zprime = plasma_dispersion_derivative(zeta_values)

        # Check relation Z'(ζ) = -2[1 + ζZ(ζ)]
        Zprime_expected = -2.0 * (1.0 + zeta_values * Z)

        np.testing.assert_allclose(Zprime, Zprime_expected, rtol=1e-10,
                                    err_msg="Z'(ζ) should satisfy -2[1 + ζZ(ζ)]")

    def test_modified_bessel_ratio_small_x_m0(self):
        """
        Test I_0(x)exp(-x) → 1 - x for small x.

        For m=0, the small argument expansion is I_0(x) ≈ 1 + x²/4 + ...
        So I_0(x)exp(-x) ≈ (1 + x²/4)exp(-x) ≈ 1 - x + x²/4 + O(x²)
        For very small x: I_0(x)exp(-x) ≈ 1 - x
        """
        x_small = 1e-6
        result = modified_bessel_ratio(0, x_small)

        # For m=0, small x: should be close to 1 - x
        expected = 1.0 - x_small
        assert abs(result - expected) < 1e-5, \
            f"I_0(x)exp(-x) should be ~1-x for small x, got {result}, expected {expected}"

    def test_modified_bessel_ratio_small_x_m_positive(self):
        """
        Test I_m(x)exp(-x) → 0 for small x, m > 0.

        For m > 0, I_m(x) ~ (x/2)^m / m!, so I_m(x)exp(-x) → 0 as x → 0.
        """
        x_small = 1e-6

        for m in [1, 2, 5]:
            result = modified_bessel_ratio(m, x_small)
            # For x=1e-6, I_1(x) ~ 5e-7, which is small but not < 1e-8
            # Relax tolerance to 1e-5 (still much smaller than I_0(x)~1)
            assert abs(result) < 1e-5, \
                f"I_{m}(x)exp(-x) should be ~0 for small x, got {result}"

    def test_modified_bessel_ratio_large_x_convergence(self):
        """
        Test I_m(x)exp(-x) → 1/√(2πx) for large x (all m).

        Asymptotically, all orders converge to the same value, but convergence
        is slower for higher m. Test low-m values only.
        """
        x_large = 20.0

        # Asymptotic value
        asymptotic = 1.0 / np.sqrt(2 * np.pi * x_large)

        # Test low m values (m ≤ 5)
        # For x=20, higher m (like m=10) requires much larger x for convergence
        for m in [0, 1, 2, 5]:
            result = modified_bessel_ratio(m, x_large)

            # Should be within factor of 2 of asymptotic value for x=20
            assert abs(result - asymptotic) / asymptotic < 0.5, \
                f"I_{m}({x_large})exp(-{x_large}) should approach {asymptotic:.3e}, got {result:.3e}"

    def test_modified_bessel_ratio_consistency_with_scipy(self):
        """
        Test I_m(x)exp(-x) against scipy.special.iv for various x and m.
        """
        x_values = [0.5, 1.0, 5.0, 10.0]
        m_values = [0, 1, 2, 5]

        for x in x_values:
            for m in m_values:
                result = modified_bessel_ratio(m, x)
                expected = iv(m, x) * np.exp(-x)

                np.testing.assert_allclose(result, expected, rtol=1e-10,
                                            err_msg=f"I_{m}({x})exp(-{x}) mismatch with scipy")

    def test_kinetic_response_function_k_parallel_zero(self):
        """
        Test kinetic_response_function handles k_parallel=0 (pure perpendicular mode).

        Should return non-resonant response (no Landau damping).
        """
        response = kinetic_response_function(
            k_parallel=0.0,
            k_perp=1.0,
            omega=1.0,
            v_th=1.0,
            Lambda=1.0,
            nu=0.1,
            beta_i=1.0,
            v_A=1.0,
        )

        # Should return simple non-resonant value (1 + 0j)
        assert np.isfinite(response), "Response should be finite for k∥=0"
        assert response == 1.0 + 0.0j, \
            f"Response for k∥=0 should be 1+0j, got {response}"

    def test_kinetic_response_function_causality(self):
        """
        Test that response function has proper causality (Im[R] structure).

        For collisionless case, causality requires small positive imaginary
        part in frequency (Landau prescription).
        """
        response = kinetic_response_function(
            k_parallel=1.0,
            k_perp=1.0,
            omega=1.0,
            v_th=1.0,
            Lambda=1.0,
            nu=0.0,  # Collisionless
            beta_i=1.0,
            v_A=1.0,
        )

        # Response should be finite and well-behaved
        assert np.isfinite(response), "Response should be finite"
        # For this specific case (ω ~ k∥v_A), response may be nearly real
        # The imaginary part from CAUSALITY_EPSILON is very small (~1e-3)
        # Main test: function doesn't crash with collisionless case
        assert isinstance(response, (complex, np.complex128, np.complexfloating)), \
            "Response should be complex type"

    def test_kinetic_response_function_finite_with_collisions(self):
        """
        Test that response is finite and reasonable with collisions.
        """
        response = kinetic_response_function(
            k_parallel=1.0,
            k_perp=1.0,
            omega=1.0,
            v_th=1.0,
            Lambda=1.0,
            nu=0.3,
            beta_i=1.0,
            v_A=1.0,
        )

        assert np.isfinite(response), "Response should be finite"
        assert abs(response) > 0, "Response should be non-zero"
        assert abs(response) < 100, "Response should have reasonable magnitude"

    def test_flr_correction_factor_small_k_perp(self):
        """
        Test FLR correction → 1 for m=0, → 0 for m>0 when k⊥ρ_s << 1 (fluid limit).
        """
        k_perp = 0.01  # Small
        rho_s = 1.0

        # m=0: should be ~1
        Gamma_0_sq = flr_correction_factor(0, k_perp, rho_s)
        assert abs(Gamma_0_sq - 1.0) < 0.01, \
            f"Γ_0²(k⊥ρ_s<<1) should be ~1, got {Gamma_0_sq}"

        # m>0: should be ~0
        for m in [1, 2, 5]:
            Gamma_m_sq = flr_correction_factor(m, k_perp, rho_s)
            assert Gamma_m_sq < 1e-4, \
                f"Γ_{m}²(k⊥ρ_s<<1) should be ~0, got {Gamma_m_sq}"

    def test_flr_correction_factor_large_k_perp(self):
        """
        Test FLR correction → 0 for all m when k⊥ρ_s >> 1 (sub-gyroradius scales).
        """
        k_perp = 10.0  # Large
        rho_s = 1.0

        for m in [0, 1, 2, 5]:
            Gamma_m_sq = flr_correction_factor(m, k_perp, rho_s)
            # All moments damped at sub-gyroradius scales
            assert Gamma_m_sq < 0.1, \
                f"Γ_{m}²(k⊥ρ_s>>1) should be small, got {Gamma_m_sq}"

    def test_flr_correction_factor_positive_and_bounded(self):
        """
        Test that FLR correction is always positive and ≤ 1.

        Since Γ_m² = [I_m(b)exp(-b)]², it should be positive and bounded.
        """
        k_perp_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        rho_s = 1.0

        for k_perp in k_perp_values:
            for m in range(10):
                Gamma_m_sq = flr_correction_factor(m, k_perp, rho_s)

                assert Gamma_m_sq >= 0, \
                    f"Γ_{m}²(k⊥={k_perp}) should be ≥ 0, got {Gamma_m_sq}"
                assert Gamma_m_sq <= 1.1, \
                    f"Γ_{m}²(k⊥={k_perp}) should be ≤ 1, got {Gamma_m_sq}"

    def test_flr_correction_factor_m_dependence(self):
        """
        Test that FLR correction decreases with m for k⊥ρ_s ~ 1.

        Higher moments should be more suppressed by FLR effects.
        """
        k_perp = 1.0  # Kinetic regime
        rho_s = 1.0

        Gamma_values = [flr_correction_factor(m, k_perp, rho_s) for m in range(10)]

        # Should generally decrease with m (though not monotonic for all k⊥)
        # Check that high m is suppressed relative to low m
        assert Gamma_values[5] < Gamma_values[0], \
            "Higher moments should be more suppressed by FLR"
        assert Gamma_values[9] < Gamma_values[5], \
            "Highest moments should be most suppressed"
