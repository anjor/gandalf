"""
Tests for forcing mechanisms (Issue #29).

This module tests Gaussian white noise forcing for driven turbulence simulations,
including Alfvén mode forcing, slow mode forcing, and energy injection diagnostics.
"""

import pytest
import jax
import jax.numpy as jnp
from krmhd import (
    SpectralGrid3D,
    initialize_alfven_wave,
    initialize_random_spectrum,
    gaussian_white_noise_fourier,
    force_alfven_modes,
    force_slow_modes,
    compute_energy_injection_rate,
    gandalf_step,
    energy,
)


class TestGaussianWhiteNoise:
    """Test Gaussian white noise generation in Fourier space.

    Note on stochastic test design:
    - These tests use single seeds for determinism (CI/CD reproducibility)
    - Tolerances (±10-30%) are empirically chosen to bound typical variance
    - For more robust validation, consider seed-averaged tests:
      * Generate 5-10 realizations with different seeds
      * Check that mean converges to expected value
      * Example: mean_ratio = np.mean([test_single_seed(i) for i in range(10)])
    - Production ensemble averaging demonstrated in driven_turbulence.py
    """

    def test_input_validation_n_min_n_max(self):
        """Should raise ValueError if n_min >= n_max."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        key = jax.random.PRNGKey(42)

        # n_min > n_max should fail
        with pytest.raises(ValueError, match="n_min must be < n_max"):
            gaussian_white_noise_fourier(
                grid, amplitude=0.1, n_min=2, n_max=1, dt=0.01, key=key
            )

        # n_min == n_max should fail
        with pytest.raises(ValueError, match="n_min must be < n_max"):
            gaussian_white_noise_fourier(
                grid, amplitude=0.1, n_min=2, n_max=2, dt=0.01, key=key
            )

    def test_input_validation_amplitude(self):
        """Should raise ValueError if amplitude <= 0."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        key = jax.random.PRNGKey(42)

        with pytest.raises(ValueError, match="amplitude must be positive"):
            gaussian_white_noise_fourier(
                grid, amplitude=0.0, n_min=1, n_max=2, dt=0.01, key=key
            )

        with pytest.raises(ValueError, match="amplitude must be positive"):
            gaussian_white_noise_fourier(
                grid, amplitude=-0.1, n_min=1, n_max=2, dt=0.01, key=key
            )

    def test_input_validation_dt(self):
        """Should raise ValueError if dt <= 0."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        key = jax.random.PRNGKey(42)

        with pytest.raises(ValueError, match="dt must be positive"):
            gaussian_white_noise_fourier(
                grid, amplitude=0.1, n_min=1, n_max=2, dt=0.0, key=key
            )

        with pytest.raises(ValueError, match="dt must be positive"):
            gaussian_white_noise_fourier(
                grid, amplitude=0.1, n_min=1, n_max=2, dt=-0.01, key=key
            )

    def test_noise_shape(self):
        """Noise field should have correct Fourier space shape."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        key = jax.random.PRNGKey(42)

        noise, key = gaussian_white_noise_fourier(
            grid, amplitude=0.1, n_min=1, n_max=2, dt=0.01, key=key
        )

        expected_shape = (grid.Nz, grid.Ny, grid.Nx // 2 + 1)
        assert noise.shape == expected_shape
        assert jnp.iscomplexobj(noise)

    def test_spectral_localization(self):
        """Noise should be concentrated in [n_min, n_max] mode number band."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        key = jax.random.PRNGKey(42)
        n_min, n_max = 1, 2

        noise, key = gaussian_white_noise_fourier(
            grid, amplitude=0.1, n_min=n_min, n_max=n_max, dt=0.01, key=key
        )

        # Convert mode numbers to physical wavenumbers for validation
        L_min = min(grid.Lx, grid.Ly, grid.Lz)
        k_min = n_min * 2 * jnp.pi / L_min
        k_max = n_max * 2 * jnp.pi / L_min

        # Compute |k| for each mode with proper broadcasting
        kx_3d = grid.kx[jnp.newaxis, jnp.newaxis, :]
        ky_3d = grid.ky[jnp.newaxis, :, jnp.newaxis]
        kz_3d = grid.kz[:, jnp.newaxis, jnp.newaxis]
        k_mag = jnp.sqrt(kx_3d**2 + ky_3d**2 + kz_3d**2)

        # Modes outside [k_min, k_max] should be zero
        outside_band = (k_mag < k_min) | (k_mag > k_max)
        assert jnp.allclose(noise[outside_band], 0.0)

        # Modes inside band should be non-zero (with high probability)
        inside_band = (k_mag >= k_min) & (k_mag <= k_max)
        assert jnp.sum(jnp.abs(noise[inside_band]) > 0) > 0

    def test_zero_mean(self):
        """Noise should have approximately zero mean (k=0 mode is zero)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        key = jax.random.PRNGKey(42)

        noise, key = gaussian_white_noise_fourier(
            grid, amplitude=0.1, n_min=1, n_max=2, dt=0.01, key=key
        )

        # k=0 mode should be exactly zero
        assert noise[0, 0, 0] == 0.0 + 0.0j

    def test_reality_condition_kx_zero_plane(self):
        """kx=0 plane must be real-valued (Hermitian symmetry requirement).

        This is a DIRECT test of Hermitian symmetry enforcement in Fourier space,
        not relying on irfftn to fix violations. Critical because forcing is applied
        directly in Fourier space without inverse transform.
        """
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        key = jax.random.PRNGKey(42)

        noise, key = gaussian_white_noise_fourier(
            grid, amplitude=0.1, n_min=1, n_max=2, dt=0.01, key=key
        )

        # kx=0 plane (index 0 in rfft format) must have zero imaginary part
        max_imag_kx0 = jnp.max(jnp.abs(noise[:, :, 0].imag))
        assert max_imag_kx0 < 1e-10, (
            f"kx=0 plane not real: max(imag) = {max_imag_kx0}. "
            "Violates Hermitian symmetry required for real fields."
        )

        # kx=Nyquist plane (if exists) must also have zero imaginary part
        if grid.Nx % 2 == 0:  # Nyquist frequency exists
            nyquist_idx = grid.Nx // 2
            max_imag_nyquist = jnp.max(jnp.abs(noise[:, :, nyquist_idx].imag))
            assert max_imag_nyquist < 1e-10, (
                f"kx=Nyquist plane not real: max(imag) = {max_imag_nyquist}. "
                "Violates Hermitian symmetry."
            )

    def test_reality_condition_inverse_transform(self):
        """Inverse transform should give real-valued field (end-to-end test)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        key = jax.random.PRNGKey(42)

        noise, key = gaussian_white_noise_fourier(
            grid, amplitude=0.1, n_min=1, n_max=2, dt=0.01, key=key
        )

        # Shape should be correct for rfft (Nx//2+1 in x-direction)
        assert noise.shape[2] == grid.Nx // 2 + 1

        # Inverse transform should give real-valued field
        from krmhd.spectral import rfftn_inverse
        noise_real = rfftn_inverse(noise, grid.Nz, grid.Ny, grid.Nx)

        # Check that imaginary part is negligible (< 1e-10)
        max_imag = jnp.max(jnp.abs(jnp.imag(noise_real)))
        assert max_imag < 1e-10, f"Inverse transform not real: max(imag) = {max_imag}"

    def test_key_threading(self):
        """Different keys should produce different noise realizations."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(43)

        noise1, _ = gaussian_white_noise_fourier(
            grid, amplitude=0.1, n_min=1, n_max=2, dt=0.01, key=key1
        )
        noise2, _ = gaussian_white_noise_fourier(
            grid, amplitude=0.1, n_min=1, n_max=2, dt=0.01, key=key2
        )

        # Different seeds → different noise
        assert not jnp.allclose(noise1, noise2)

    def test_reproducibility(self):
        """Same key should produce identical noise."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        key = jax.random.PRNGKey(42)

        noise1, _ = gaussian_white_noise_fourier(
            grid, amplitude=0.1, n_min=1, n_max=2, dt=0.01, key=key
        )
        noise2, _ = gaussian_white_noise_fourier(
            grid, amplitude=0.1, n_min=1, n_max=2, dt=0.01, key=key
        )

        # Same seed → identical noise
        assert jnp.allclose(noise1, noise2)

    def test_amplitude_scaling(self):
        """Larger amplitude should produce larger noise.

        Note: Uses single seed for determinism. For more robust validation of
        stochastic scaling, average over multiple seeds (see driven_turbulence.py
        for ensemble-averaged steady-state validation).
        """
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        key = jax.random.PRNGKey(42)

        noise_weak, key = gaussian_white_noise_fourier(
            grid, amplitude=0.1, n_min=1, n_max=2, dt=0.01, key=key
        )

        key = jax.random.PRNGKey(42)  # Reset for fair comparison
        noise_strong, key = gaussian_white_noise_fourier(
            grid, amplitude=1.0, n_min=1, n_max=2, dt=0.01, key=key
        )

        # Ratio of magnitudes should be ~10x (amplitude ratio)
        ratio = jnp.mean(jnp.abs(noise_strong)) / jnp.mean(jnp.abs(noise_weak))
        # Tighter tolerance: ±10% (was ±20%)
        assert jnp.abs(ratio - 10.0) < 1.0, f"Amplitude scaling failed: ratio={ratio:.2f}, expected=10.0"

    def test_timestep_scaling(self):
        """White noise scaling: amplitude/√dt → energy injection independent of dt.

        Note: Uses single seed for determinism. Statistical fluctuations are real
        but bounded by ±10% tolerance. For production use, time-average over many
        forcing realizations.
        """
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        key = jax.random.PRNGKey(42)

        noise_small_dt, key = gaussian_white_noise_fourier(
            grid, amplitude=0.1, n_min=1, n_max=2, dt=0.001, key=key
        )

        key = jax.random.PRNGKey(42)  # Reset
        noise_large_dt, key = gaussian_white_noise_fourier(
            grid, amplitude=0.1, n_min=1, n_max=2, dt=0.01, key=key
        )

        # Ratio should be ~√10 ≈ 3.16 (√(dt_large/dt_small))
        ratio = jnp.mean(jnp.abs(noise_small_dt)) / jnp.mean(jnp.abs(noise_large_dt))
        expected_ratio = jnp.sqrt(0.01 / 0.001)
        # Tighter tolerance: ±0.3 instead of ±0.5 (~10% instead of ~16%)
        assert jnp.abs(ratio - expected_ratio) < 0.3, f"Timestep scaling failed: ratio={ratio:.2f}, expected={expected_ratio:.2f}"


class TestAlfvenForcing:
    """Test Alfvén mode forcing (z⁺ and z⁻)."""

    def test_alfven_forcing_identical(self):
        """z⁺ and z⁻ should receive IDENTICAL forcing."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)
        key = jax.random.PRNGKey(42)

        state_forced, key = force_alfven_modes(
            state, amplitude=0.1, n_min=1, n_max=2, dt=0.01, key=key
        )

        # Compute forcing applied: Δz = z_new - z_old
        delta_z_plus = state_forced.z_plus - state.z_plus
        delta_z_minus = state_forced.z_minus - state.z_minus

        # Forcing should be identical
        assert jnp.allclose(delta_z_plus, delta_z_minus)

    def test_phi_forced_A_parallel_unchanged(self):
        """Forcing should drive φ only, leaving A∥ unchanged."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)
        key = jax.random.PRNGKey(42)

        # Compute φ and A∥ before forcing
        phi_old = (state.z_plus + state.z_minus) / 2.0
        A_old = (state.z_plus - state.z_minus) / 2.0

        state_forced, key = force_alfven_modes(
            state, amplitude=0.1, n_min=1, n_max=2, dt=0.01, key=key
        )

        # Compute φ and A∥ after forcing
        phi_new = (state_forced.z_plus + state_forced.z_minus) / 2.0
        A_new = (state_forced.z_plus - state_forced.z_minus) / 2.0

        # φ should change (driven)
        assert not jnp.allclose(phi_old, phi_new)

        # A∥ should be UNCHANGED (not driven)
        assert jnp.allclose(A_old, A_new)

    def test_energy_injection_positive(self):
        """Forcing should inject energy (E_after > E_before)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.01)
        key = jax.random.PRNGKey(42)

        E_before = energy(state)["total"]

        state_forced, key = force_alfven_modes(
            state, amplitude=0.5, n_min=1, n_max=2, dt=0.01, key=key
        )

        E_after = energy(state_forced)["total"]

        # Energy should increase
        assert E_after > E_before

    def test_forcing_concentrated_at_k_band(self):
        """Forcing should be concentrated at specified mode number band."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.0)  # Zero initial
        key = jax.random.PRNGKey(42)
        n_min, n_max = 1, 2

        state_forced, key = force_alfven_modes(
            state, amplitude=1.0, n_min=n_min, n_max=n_max, dt=0.01, key=key
        )

        # Convert mode numbers to physical wavenumbers for validation
        L_min = min(grid.Lx, grid.Ly, grid.Lz)
        k_min = n_min * 2 * jnp.pi / L_min
        k_max = n_max * 2 * jnp.pi / L_min

        # Compute |k| with proper broadcasting
        kx_3d = grid.kx[jnp.newaxis, jnp.newaxis, :]
        ky_3d = grid.ky[jnp.newaxis, :, jnp.newaxis]
        kz_3d = grid.kz[:, jnp.newaxis, jnp.newaxis]
        k_mag = jnp.sqrt(kx_3d**2 + ky_3d**2 + kz_3d**2)

        # Check forcing is concentrated in band
        inside_band = (k_mag >= k_min) & (k_mag <= k_max)
        outside_band = (k_mag < k_min) | (k_mag > k_max)

        # Inside band should have large amplitudes
        energy_inside = jnp.sum(jnp.abs(state_forced.z_plus[inside_band])**2)

        # Outside band should be zero (or very small if initial state had energy there)
        energy_outside = jnp.sum(jnp.abs(state_forced.z_plus[outside_band])**2)

        # Most energy should be in the forcing band
        assert energy_inside > 10 * energy_outside

    def test_deterministic_with_fixed_seed(self):
        """Fixed seed should produce deterministic forcing."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)

        key1 = jax.random.PRNGKey(42)
        state_forced1, _ = force_alfven_modes(
            state, amplitude=0.1, n_min=1, n_max=2, dt=0.01, key=key1
        )

        key2 = jax.random.PRNGKey(42)  # Same seed
        state_forced2, _ = force_alfven_modes(
            state, amplitude=0.1, n_min=1, n_max=2, dt=0.01, key=key2
        )

        # Should be identical
        assert jnp.allclose(state_forced1.z_plus, state_forced2.z_plus)
        assert jnp.allclose(state_forced1.z_minus, state_forced2.z_minus)

    def test_hermite_moments_unchanged(self):
        """Alfvén forcing should not affect Hermite moments."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_random_spectrum(
            grid, M=10, alpha=5/3, amplitude=0.1, seed=42
        )
        key = jax.random.PRNGKey(42)

        g_old = state.g.copy()

        state_forced, key = force_alfven_modes(
            state, amplitude=0.1, n_min=1, n_max=2, dt=0.01, key=key
        )

        # Hermite moments should be unchanged
        assert jnp.allclose(state_forced.g, g_old)


class TestSlowModeForcing:
    """Test slow mode forcing (δB∥)."""

    def test_B_parallel_increases(self):
        """Slow mode forcing should increase |B∥|."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)
        key = jax.random.PRNGKey(42)

        B_parallel_magnitude_old = jnp.sum(jnp.abs(state.B_parallel)**2)

        state_forced, key = force_slow_modes(
            state, amplitude=0.5, n_min=1, n_max=2, dt=0.01, key=key
        )

        B_parallel_magnitude_new = jnp.sum(jnp.abs(state_forced.B_parallel)**2)

        # B∥ magnitude should increase
        assert B_parallel_magnitude_new > B_parallel_magnitude_old

    def test_independent_from_alfven_forcing(self):
        """Slow mode forcing should not affect Elsasser variables."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)
        key = jax.random.PRNGKey(42)

        z_plus_old = state.z_plus.copy()
        z_minus_old = state.z_minus.copy()

        state_forced, key = force_slow_modes(
            state, amplitude=0.1, n_min=1, n_max=2, dt=0.01, key=key
        )

        # Elsasser variables should be unchanged
        assert jnp.allclose(state_forced.z_plus, z_plus_old)
        assert jnp.allclose(state_forced.z_minus, z_minus_old)

    def test_slow_mode_energy_injection(self):
        """Slow mode forcing should inject energy."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.01)
        key = jax.random.PRNGKey(42)

        E_before = energy(state)["total"]

        state_forced, key = force_slow_modes(
            state, amplitude=0.5, n_min=1, n_max=2, dt=0.01, key=key
        )

        E_after = energy(state_forced)["total"]

        # Energy should increase
        assert E_after > E_before


class TestEnergyInjection:
    """Test energy injection rate computation."""

    def test_energy_injection_rate_positive(self):
        """Energy injection rate should be positive for forcing."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.01)
        key = jax.random.PRNGKey(42)
        dt = 0.01

        state_forced, key = force_alfven_modes(
            state, amplitude=0.5, n_min=1, n_max=2, dt=dt, key=key
        )

        eps_inj = compute_energy_injection_rate(state, state_forced, dt)

        # Should be positive (energy added)
        assert eps_inj > 0

    def test_energy_injection_scales_with_amplitude(self):
        """Energy injection should scale with amplitude².

        Note: Single-seed test with ±30% tolerance accounts for stochastic
        fluctuations. The ε ∝ A² scaling is a time-averaged property, and
        individual realizations show O(1) variance. For steady-state validation,
        see driven_turbulence.py example (~200 steps, ensemble averaging).
        """
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.01)
        dt = 0.01

        # Weak forcing
        key1 = jax.random.PRNGKey(42)
        state_forced_weak, _ = force_alfven_modes(
            state, amplitude=0.1, n_min=1, n_max=2, dt=dt, key=key1
        )
        eps_weak = compute_energy_injection_rate(state, state_forced_weak, dt)

        # Strong forcing (10x amplitude)
        key2 = jax.random.PRNGKey(42)  # Same seed for fair comparison
        state_forced_strong, _ = force_alfven_modes(
            state, amplitude=1.0, n_min=1, n_max=2, dt=dt, key=key2
        )
        eps_strong = compute_energy_injection_rate(state, state_forced_strong, dt)

        # Energy injection should scale as amplitude² → ratio ~ 100
        ratio = eps_strong / eps_weak
        # Tighter tolerance: 70-130 (±30%) instead of 50-150 (±50%)
        # Statistical variation is real but should be bounded
        assert 70 < ratio < 130, f"Energy scaling failed: ratio={ratio:.1f}, expected~100"

    def test_cumulative_energy_over_multiple_steps(self):
        """Cumulative energy injection over multiple steps."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.01)
        key = jax.random.PRNGKey(42)
        dt = 0.01

        E_initial = energy(state)["total"]
        total_injection = 0.0

        # Apply forcing multiple times
        for _ in range(5):
            state_old = state
            state, key = force_alfven_modes(
                state, amplitude=0.1, n_min=1, n_max=2, dt=dt, key=key
            )
            eps_inj = compute_energy_injection_rate(state_old, state, dt)
            total_injection += eps_inj * dt

        E_final = energy(state)["total"]

        # Energy increase should equal cumulative injection
        energy_increase = E_final - E_initial
        assert jnp.abs(energy_increase - total_injection) < 1e-5


class TestIntegrationWithTimestepping:
    """Test forcing integrated with GANDALF timestepper."""

    def test_forcing_plus_timestepping(self):
        """Forcing followed by timestepping should work correctly."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)
        key = jax.random.PRNGKey(42)
        dt = 0.01

        # Apply forcing
        state_forced, key = force_alfven_modes(
            state, amplitude=0.1, n_min=1, n_max=2, dt=dt, key=key
        )

        # Evolve with timestepper
        state_evolved = gandalf_step(state_forced, dt=dt, eta=0.01, v_A=1.0)

        # Should complete without error
        assert state_evolved is not None
        assert energy(state_evolved)["total"] > 0

    def test_forced_evolution_loop(self):
        """Run forced evolution loop (forcing + evolution repeatedly)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, kz_mode=1, amplitude=0.1)
        key = jax.random.PRNGKey(42)
        dt = 0.01

        # Run several steps with forcing
        for i in range(10):
            # Apply forcing
            state, key = force_alfven_modes(
                state, amplitude=0.1, n_min=1, n_max=2, dt=dt, key=key
            )

            # Evolve
            state = gandalf_step(state, dt=dt, eta=0.01, v_A=1.0)

        # Should complete successfully
        assert state.time > 0
        assert energy(state)["total"] > 0

    def test_forced_evolution_stability(self):
        """Forced turbulence evolution should remain stable over many steps.

        Note: This is a weak integration test that verifies the forcing+timestepping
        loop runs without errors. It does NOT validate steady-state energy balance
        (ε_inj ≈ ε_diss), which would require much longer integration (~1000+ steps)
        and careful dissipation rate measurement.

        For true steady-state validation, see driven_turbulence.py example.
        """
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_random_spectrum(
            grid, M=10, alpha=5/3, amplitude=0.01, seed=42
        )
        key = jax.random.PRNGKey(42)
        dt = 0.01
        eta = 0.1  # Moderate dissipation

        # Run forced evolution loop for 50 steps
        energies = []
        for i in range(50):
            # Force
            state, key = force_alfven_modes(
                state, amplitude=0.5, n_min=1, n_max=2, dt=dt, key=key
            )

            # Evolve
            state = gandalf_step(state, dt=dt, eta=eta, v_A=1.0)

            # Track energy
            energies.append(energy(state)["total"])

        # Basic stability checks
        assert len(energies) == 50  # Completed all steps
        assert all(e > 0 for e in energies)  # Energy always positive
        assert all(jnp.isfinite(e) for e in energies)  # No NaN/Inf

        # Energy should increase initially (forcing dominates dissipation)
        assert energies[-1] > energies[0], "Energy should increase with forcing"

    def test_forced_vs_unforced_evolution(self):
        """Forced evolution should maintain higher energy than unforced."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        # Two identical initial states
        state_forced = initialize_random_spectrum(
            grid, M=10, alpha=5/3, amplitude=0.1, seed=42
        )
        state_unforced = initialize_random_spectrum(
            grid, M=10, alpha=5/3, amplitude=0.1, seed=42
        )

        key = jax.random.PRNGKey(42)
        dt = 0.01
        eta = 0.05

        # Evolve both for several steps
        for i in range(20):
            # Forced evolution
            state_forced, key = force_alfven_modes(
                state_forced, amplitude=0.3, n_min=1, n_max=2, dt=dt, key=key
            )
            state_forced = gandalf_step(state_forced, dt=dt, eta=eta, v_A=1.0)

            # Unforced (decaying) evolution
            state_unforced = gandalf_step(state_unforced, dt=dt, eta=eta, v_A=1.0)

        E_forced = energy(state_forced)["total"]
        E_unforced = energy(state_unforced)["total"]

        # Forced should have much higher energy
        assert E_forced > E_unforced
        assert E_forced > 2 * E_unforced  # At least 2x higher
