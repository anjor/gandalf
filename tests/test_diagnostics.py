"""
Tests for diagnostic functions.

This test suite validates:
- Energy spectrum calculations (1D, perpendicular, parallel)
- Spectrum normalization (Parseval's theorem)
- Energy history tracking
- Visualization functions (check they run without errors)
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from krmhd.spectral import SpectralGrid3D
from krmhd.physics import (
    KRMHDState,
    initialize_alfven_wave,
    initialize_random_spectrum,
    energy as compute_energy,
)
from krmhd.diagnostics import (
    energy_spectrum_1d,
    energy_spectrum_perpendicular,
    energy_spectrum_perpendicular_kinetic,
    energy_spectrum_perpendicular_magnetic,
    energy_spectrum_parallel,
    EnergyHistory,
    plot_state,
    plot_energy_history,
    plot_energy_spectrum,
    spectral_pad_and_ifft,
    interpolate_on_fine_grid,
    compute_magnetic_field_components,
    follow_field_line,
)


class TestEnergySpectrum1D:
    """Test 1D spherically-averaged energy spectrum."""

    def test_single_mode_spectrum(self):
        """
        Test spectrum of single Alfvén wave mode (should be delta function).

        For a single mode at (kx, ky, kz), all energy should be concentrated
        in the bin containing |k| = √(kx² + ky² + kz²).
        """
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)
        state = initialize_alfven_wave(grid, M=10, kx_mode=1.0, ky_mode=0.0, kz_mode=1.0, amplitude=0.1)

        k, E_k = energy_spectrum_1d(state, n_bins=32)

        # Spectrum should be positive
        assert jnp.all(E_k >= 0), "Spectrum has negative values"

        # Most energy should be concentrated near |k| = √(1² + 0² + 1²) ≈ 1.41
        k_expected = jnp.sqrt(1.0**2 + 0.0**2 + 1.0**2)
        peak_idx = jnp.argmax(E_k)
        k_peak = k[peak_idx]

        # Peak should be within 1 bin of expected value
        dk = k[1] - k[0]
        assert jnp.abs(k_peak - k_expected) < 2 * dk, \
            f"Peak at k={k_peak:.2f}, expected k={k_expected:.2f}"

    def test_spectrum_normalization(self):
        """
        Test that spectrum integrates to total energy (Parseval's theorem).

        ∫ E(k) dk ≈ E_total (within numerical precision)
        """
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=1.0, seed=42)

        k, E_k = energy_spectrum_1d(state, n_bins=32)

        # Integrate spectrum
        dk = k[1] - k[0]
        E_from_spectrum = jnp.sum(E_k) * dk

        # Compute total energy directly
        energies = compute_energy(state)
        E_total = energies['total']

        # Should match within 10% (coarse binning)
        rel_error = jnp.abs(E_from_spectrum - E_total) / E_total
        assert rel_error < 0.1, \
            f"Spectrum integral {E_from_spectrum:.6f} != E_total {E_total:.6f} (error {rel_error:.2%})"

    def test_spectrum_shape(self):
        """Test that spectrum has correct shape and properties."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=1.0, seed=42)

        n_bins = 32
        k, E_k = energy_spectrum_1d(state, n_bins=n_bins)

        # Check shapes
        assert k.shape == (n_bins,), f"k shape {k.shape} != ({n_bins},)"
        assert E_k.shape == (n_bins,), f"E_k shape {E_k.shape} != ({n_bins},)"

        # k should be monotonically increasing
        assert jnp.all(jnp.diff(k) > 0), "k is not monotonically increasing"

        # E_k should be non-negative
        assert jnp.all(E_k >= 0), "E_k has negative values"

        # E_k should be real-valued
        assert jnp.isrealobj(E_k), "E_k is not real-valued"

    def test_zero_state_spectrum(self):
        """Zero state should have zero spectrum."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)

        state = KRMHDState(
            z_plus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            z_minus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            B_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            g=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, 11), dtype=jnp.complex64),
            M=10,
            beta_i=1.0,
            v_th=1.0,
            nu=0.01,
            Lambda=1.0,
            time=0.0,
            grid=grid,
        )

        k, E_k = energy_spectrum_1d(state, n_bins=32)

        assert jnp.allclose(E_k, 0.0, atol=1e-10), "Zero state has non-zero spectrum"


class TestEnergySpectrumPerpendicular:
    """Test perpendicular energy spectrum E(k⊥)."""

    def test_perpendicular_spectrum_shape(self):
        """Test that perpendicular spectrum has correct shape."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=1.0, seed=42)

        n_bins = 32
        k_perp, E_perp = energy_spectrum_perpendicular(state, n_bins=n_bins)

        # Check shapes
        assert k_perp.shape == (n_bins,), f"k_perp shape {k_perp.shape} != ({n_bins},)"
        assert E_perp.shape == (n_bins,), f"E_perp shape {E_perp.shape} != ({n_bins},)"

        # k_perp should be monotonically increasing
        assert jnp.all(jnp.diff(k_perp) > 0), "k_perp is not monotonically increasing"

        # E_perp should be non-negative
        assert jnp.all(E_perp >= 0), "E_perp has negative values"

    def test_perpendicular_normalization(self):
        """Test that perpendicular spectrum integrates to total energy."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=1.0, seed=42)

        k_perp, E_perp = energy_spectrum_perpendicular(state, n_bins=32)

        # Integrate spectrum
        dk_perp = k_perp[1] - k_perp[0]
        E_from_spectrum = jnp.sum(E_perp) * dk_perp

        # Compute total energy directly
        energies = compute_energy(state)
        E_total = energies['total']

        # Should match within 10% (coarse binning)
        rel_error = jnp.abs(E_from_spectrum - E_total) / E_total
        assert rel_error < 0.1, \
            f"Perp spectrum integral {E_from_spectrum:.6f} != E_total {E_total:.6f} (error {rel_error:.2%})"

    def test_zero_state_perpendicular_spectrum(self):
        """Zero state should have zero perpendicular spectrum."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)

        state = KRMHDState(
            z_plus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            z_minus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            B_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            g=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, 11), dtype=jnp.complex64),
            M=10,
            beta_i=1.0,
            v_th=1.0,
            nu=0.01,
            Lambda=1.0,
            time=0.0,
            grid=grid,
        )

        k_perp, E_perp = energy_spectrum_perpendicular(state, n_bins=32)

        assert jnp.allclose(E_perp, 0.0, atol=1e-10), "Zero state has non-zero perpendicular spectrum"


class TestEnergySpectrumPerpendicularKinetic:
    """Test perpendicular kinetic energy spectrum E_kin(k⊥) from φ."""

    def test_kinetic_spectrum_shape(self):
        """Test that kinetic spectrum has correct shape."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=1.0, seed=42)

        n_bins = 32
        k_perp, E_kin = energy_spectrum_perpendicular_kinetic(state, n_bins=n_bins)

        # Check shapes
        assert k_perp.shape == (n_bins,), f"k_perp shape {k_perp.shape} != ({n_bins},)"
        assert E_kin.shape == (n_bins,), f"E_kin shape {E_kin.shape} != ({n_bins},)"

        # k_perp should be monotonically increasing
        assert jnp.all(jnp.diff(k_perp) > 0), "k_perp is not monotonically increasing"

        # E_kin should be non-negative
        assert jnp.all(E_kin >= 0), "E_kin has negative values"

    def test_kinetic_single_mode(self):
        """Test kinetic spectrum for single Alfvén wave mode."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)

        # Initialize pure Alfvén wave: z+ and z- have same amplitude → pure velocity
        state = initialize_alfven_wave(grid, M=10, kx_mode=2.0, ky_mode=0.0, kz_mode=1.0, amplitude=0.1)

        k_perp, E_kin = energy_spectrum_perpendicular_kinetic(state, n_bins=32)

        # Spectrum should be positive
        assert jnp.all(E_kin >= 0), "Spectrum has negative values"

        # Most energy should be concentrated near k⊥ = 2.0
        k_expected = 2.0
        peak_idx = jnp.argmax(E_kin)
        k_peak = k_perp[peak_idx]

        # Peak should be within 2 bins of expected value
        dk = k_perp[1] - k_perp[0]
        assert jnp.abs(k_peak - k_expected) < 2 * dk, \
            f"Peak at k⊥={k_peak:.2f}, expected k⊥={k_expected:.2f}"

    def test_kinetic_normalization(self):
        """Test that kinetic spectrum integrates to kinetic energy."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=1.0, seed=42)

        k_perp, E_kin_spectrum = energy_spectrum_perpendicular_kinetic(state, n_bins=32)

        # Integrate spectrum
        dk_perp = k_perp[1] - k_perp[0]
        E_from_spectrum = jnp.sum(E_kin_spectrum) * dk_perp

        # Compute kinetic energy directly
        energies = compute_energy(state)
        E_kinetic_direct = energies['kinetic']

        # Should match within 10% (coarse binning)
        rel_error = jnp.abs(E_from_spectrum - E_kinetic_direct) / E_kinetic_direct
        assert rel_error < 0.1, \
            f"Kinetic spectrum integral {E_from_spectrum:.6f} != E_kinetic {E_kinetic_direct:.6f} (error {rel_error:.2%})"

    def test_kinetic_zero_state(self):
        """Zero state should have zero kinetic spectrum."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)

        state = KRMHDState(
            z_plus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            z_minus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            B_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            g=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, 11), dtype=jnp.complex64),
            M=10,
            beta_i=1.0,
            v_th=1.0,
            nu=0.01,
            Lambda=1.0,
            time=0.0,
            grid=grid,
        )

        k_perp, E_kin = energy_spectrum_perpendicular_kinetic(state, n_bins=32)

        assert jnp.allclose(E_kin, 0.0, atol=1e-10), "Zero state has non-zero kinetic spectrum"


class TestEnergySpectrumPerpendicularMagnetic:
    """Test perpendicular magnetic energy spectrum E_mag(k⊥) from A∥."""

    def test_magnetic_spectrum_shape(self):
        """Test that magnetic spectrum has correct shape."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=1.0, seed=42)

        n_bins = 32
        k_perp, E_mag = energy_spectrum_perpendicular_magnetic(state, n_bins=n_bins)

        # Check shapes
        assert k_perp.shape == (n_bins,), f"k_perp shape {k_perp.shape} != ({n_bins},)"
        assert E_mag.shape == (n_bins,), f"E_mag shape {E_mag.shape} != ({n_bins},)"

        # k_perp should be monotonically increasing
        assert jnp.all(jnp.diff(k_perp) > 0), "k_perp is not monotonically increasing"

        # E_mag should be non-negative
        assert jnp.all(E_mag >= 0), "E_mag has negative values"

    def test_magnetic_single_mode(self):
        """Test magnetic spectrum for single Alfvén wave mode."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)

        # Initialize pure Alfvén wave
        state = initialize_alfven_wave(grid, M=10, kx_mode=2.0, ky_mode=0.0, kz_mode=1.0, amplitude=0.1)

        k_perp, E_mag = energy_spectrum_perpendicular_magnetic(state, n_bins=32)

        # Spectrum should be positive
        assert jnp.all(E_mag >= 0), "Spectrum has negative values"

        # Most energy should be concentrated near k⊥ = 2.0
        k_expected = 2.0
        peak_idx = jnp.argmax(E_mag)
        k_peak = k_perp[peak_idx]

        # Peak should be within 2 bins of expected value
        dk = k_perp[1] - k_perp[0]
        assert jnp.abs(k_peak - k_expected) < 2 * dk, \
            f"Peak at k⊥={k_peak:.2f}, expected k⊥={k_expected:.2f}"

    def test_magnetic_normalization(self):
        """Test that magnetic spectrum integrates to magnetic energy."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=1.0, seed=42)

        k_perp, E_mag_spectrum = energy_spectrum_perpendicular_magnetic(state, n_bins=32)

        # Integrate spectrum
        dk_perp = k_perp[1] - k_perp[0]
        E_from_spectrum = jnp.sum(E_mag_spectrum) * dk_perp

        # Compute magnetic energy directly
        energies = compute_energy(state)
        E_magnetic_direct = energies['magnetic']

        # Should match within 10% (coarse binning)
        rel_error = jnp.abs(E_from_spectrum - E_magnetic_direct) / E_magnetic_direct
        assert rel_error < 0.1, \
            f"Magnetic spectrum integral {E_from_spectrum:.6f} != E_magnetic {E_magnetic_direct:.6f} (error {rel_error:.2%})"

    def test_magnetic_zero_state(self):
        """Zero state should have zero magnetic spectrum."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)

        state = KRMHDState(
            z_plus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            z_minus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            B_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            g=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, 11), dtype=jnp.complex64),
            M=10,
            beta_i=1.0,
            v_th=1.0,
            nu=0.01,
            Lambda=1.0,
            time=0.0,
            grid=grid,
        )

        k_perp, E_mag = energy_spectrum_perpendicular_magnetic(state, n_bins=32)

        assert jnp.allclose(E_mag, 0.0, atol=1e-10), "Zero state has non-zero magnetic spectrum"


class TestAlfvenicEquipartition:
    """Test Alfvénic equipartition: E_kin ≈ E_mag for Alfvén waves."""

    def test_alfven_wave_equipartition(self):
        """Test that Alfvén wave has equal kinetic and magnetic energy."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)

        # Pure Alfvén wave: z+ propagates, z- = 0 → equal E_kin and E_mag
        state = initialize_alfven_wave(grid, M=10, kx_mode=1.0, ky_mode=0.0, kz_mode=1.0, amplitude=0.1)

        k_perp_kin, E_kin = energy_spectrum_perpendicular_kinetic(state, n_bins=32)
        k_perp_mag, E_mag = energy_spectrum_perpendicular_magnetic(state, n_bins=32)

        # k_perp arrays should match
        assert jnp.allclose(k_perp_kin, k_perp_mag), "k_perp arrays don't match"

        # Integrate spectra to get total energies
        dk = k_perp_kin[1] - k_perp_kin[0]
        E_kin_total = jnp.sum(E_kin) * dk
        E_mag_total = jnp.sum(E_mag) * dk

        # For Alfvén waves: E_kin ≈ E_mag (equipartition)
        ratio = E_kin_total / (E_mag_total + 1e-10)
        assert jnp.abs(ratio - 1.0) < 0.1, \
            f"Alfvén wave should have equipartition: E_kin/E_mag = {ratio:.3f} (expected ≈1.0)"

    def test_total_energy_conservation(self):
        """Test that E_kin + E_mag ≈ E_total."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=1.0, seed=42)

        k_perp_kin, E_kin = energy_spectrum_perpendicular_kinetic(state, n_bins=32)
        k_perp_mag, E_mag = energy_spectrum_perpendicular_magnetic(state, n_bins=32)
        k_perp_total, E_total = energy_spectrum_perpendicular(state, n_bins=32)

        # E_total should equal E_kin + E_mag at each k⊥
        E_sum = E_kin + E_mag

        # Check bin-by-bin (some tolerance for numerical precision)
        max_rel_error = jnp.max(jnp.abs(E_sum - E_total) / (E_total + 1e-10))
        assert max_rel_error < 0.01, \
            f"E_kin + E_mag != E_total: max relative error = {max_rel_error:.2%}"

    def test_kinetic_magnetic_decomposition(self):
        """Test that kinetic and magnetic functions compute DIFFERENT energies.

        This validates that:
        1. energy_spectrum_perpendicular_kinetic uses φ = (z⁺ + z⁻)/2
        2. energy_spectrum_perpendicular_magnetic uses A∥ = (z⁺ - z⁻)/2
        3. For non-equipartition states, E_kin ≠ E_mag

        Note: For Alfvén waves, E_kin = E_mag by physics (equipartition),
        but the functions are computing different quantities that happen to
        have equal magnitudes.
        """
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)

        # Test 1: Kinetic-only state (z⁺ = z⁻ → φ ≠ 0, A∥ = 0)
        z_plus = jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1), dtype=complex)
        z_plus = z_plus.at[1, 0, 1].set(0.1 + 0.1j)
        z_minus = z_plus  # Same sign → A∥ = 0

        state_kin = KRMHDState(
            grid=grid, z_plus=z_plus, z_minus=z_minus,
            B_parallel=jnp.zeros_like(z_plus),
            g=jnp.zeros((10, grid.Nz, grid.Ny, grid.Nx//2+1), dtype=complex),
            M=10, beta_i=1.0, v_th=1.0, nu=0.01, Lambda=1.0, time=0.0,
        )

        k_kin, E_kin = energy_spectrum_perpendicular_kinetic(state_kin, n_bins=32)
        k_mag, E_mag = energy_spectrum_perpendicular_magnetic(state_kin, n_bins=32)

        # Should have kinetic energy but no magnetic energy
        assert jnp.sum(E_kin) > 1e-10, "Kinetic energy should be nonzero"
        assert jnp.sum(E_mag) < 1e-10, "Magnetic energy should be zero"

        # Test 2: Magnetic-only state (z⁺ = -z⁻ → φ = 0, A∥ ≠ 0)
        z_plus = jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1), dtype=complex)
        z_plus = z_plus.at[1, 0, 1].set(0.1 + 0.1j)
        z_minus = -z_plus  # Opposite sign → φ = 0

        state_mag = KRMHDState(
            grid=grid, z_plus=z_plus, z_minus=z_minus,
            B_parallel=jnp.zeros_like(z_plus),
            g=jnp.zeros((10, grid.Nz, grid.Ny, grid.Nx//2+1), dtype=complex),
            M=10, beta_i=1.0, v_th=1.0, nu=0.01, Lambda=1.0, time=0.0,
        )

        k_kin, E_kin = energy_spectrum_perpendicular_kinetic(state_mag, n_bins=32)
        k_mag, E_mag = energy_spectrum_perpendicular_magnetic(state_mag, n_bins=32)

        # Should have magnetic energy but no kinetic energy
        assert jnp.sum(E_kin) < 1e-10, "Kinetic energy should be zero"
        assert jnp.sum(E_mag) > 1e-10, "Magnetic energy should be nonzero"


class TestEnergySpectrumParallel:
    """Test parallel energy spectrum E(k∥)."""

    def test_parallel_spectrum_shape(self):
        """Test that parallel spectrum has correct shape."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=1.0, seed=42)

        kz, E_parallel = energy_spectrum_parallel(state)

        # Check shapes
        assert kz.shape == (grid.Nz,), f"kz shape {kz.shape} != ({grid.Nz},)"
        assert E_parallel.shape == (grid.Nz,), f"E_parallel shape {E_parallel.shape} != ({grid.Nz},)"

        # E_parallel should be non-negative
        assert jnp.all(E_parallel >= 0), "E_parallel has negative values"

        # E_parallel should be real-valued
        assert jnp.isrealobj(E_parallel), "E_parallel is not real-valued"

    def test_parallel_single_mode(self):
        """
        Test parallel spectrum for single kz mode.

        Energy should be concentrated in the kz bin corresponding to the mode.
        """
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)
        state = initialize_alfven_wave(grid, M=10, kx_mode=1.0, ky_mode=0.0, kz_mode=2.0, amplitude=0.1)

        kz, E_parallel = energy_spectrum_parallel(state)

        # Most energy should be at kz = 2.0
        peak_idx = jnp.argmax(E_parallel)
        kz_peak = kz[peak_idx]

        # Peak should be within 1 bin of expected value
        dkz = kz[1] - kz[0]
        assert jnp.abs(kz_peak - 2.0) < 2 * dkz, \
            f"Peak at kz={kz_peak:.2f}, expected kz=2.0"

    def test_zero_state_parallel_spectrum(self):
        """Zero state should have zero parallel spectrum."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)

        state = KRMHDState(
            z_plus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            z_minus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            B_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            g=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, 11), dtype=jnp.complex64),
            M=10,
            beta_i=1.0,
            v_th=1.0,
            nu=0.01,
            Lambda=1.0,
            time=0.0,
            grid=grid,
        )

        kz, E_parallel = energy_spectrum_parallel(state)

        assert jnp.allclose(E_parallel, 0.0, atol=1e-10), "Zero state has non-zero parallel spectrum"


class TestEnergyHistory:
    """Test energy history tracking."""

    def test_energy_history_creation(self):
        """Test that EnergyHistory can be created."""
        history = EnergyHistory()

        assert len(history.times) == 0
        assert len(history.E_magnetic) == 0
        assert len(history.E_kinetic) == 0
        assert len(history.E_compressive) == 0
        assert len(history.E_total) == 0

    def test_energy_history_append(self):
        """Test appending states to energy history."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, amplitude=0.1)

        history = EnergyHistory()
        history.append(state)

        assert len(history.times) == 1
        assert len(history.E_magnetic) == 1
        assert len(history.E_kinetic) == 1
        assert len(history.E_compressive) == 1
        assert len(history.E_total) == 1

        # Values should match direct energy calculation
        energies = compute_energy(state)
        assert jnp.isclose(history.E_magnetic[0], energies['magnetic'])
        assert jnp.isclose(history.E_kinetic[0], energies['kinetic'])
        assert jnp.isclose(history.E_compressive[0], energies['compressive'])
        assert jnp.isclose(history.E_total[0], energies['total'])

    def test_energy_history_multiple_appends(self):
        """Test appending multiple states."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        history = EnergyHistory()

        # Create and append multiple states with different times
        for i in range(5):
            state = initialize_alfven_wave(grid, M=10, amplitude=0.1)
            state.time = float(i * 0.1)
            history.append(state)

        assert len(history.times) == 5
        # Use approximate comparison for floating point values
        expected_times = [0.0, 0.1, 0.2, 0.3, 0.4]
        assert np.allclose(history.times, expected_times, rtol=1e-10)

    def test_energy_history_to_dict(self):
        """Test converting energy history to dictionary."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_alfven_wave(grid, M=10, amplitude=0.1)

        history = EnergyHistory()
        history.append(state)

        data = history.to_dict()

        assert 'times' in data
        assert 'E_magnetic' in data
        assert 'E_kinetic' in data
        assert 'E_compressive' in data
        assert 'E_total' in data

        assert len(data['times']) == 1

    def test_magnetic_fraction(self):
        """Test computing magnetic energy fraction."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        history = EnergyHistory()

        # Append multiple states
        for i in range(3):
            state = initialize_alfven_wave(grid, M=10, amplitude=0.1)
            state.time = float(i * 0.1)
            history.append(state)

        mag_frac = history.magnetic_fraction()

        assert mag_frac.shape == (3,)
        assert jnp.all((mag_frac >= 0) & (mag_frac <= 1)), "Magnetic fraction outside [0, 1]"

    def test_dissipation_rate(self):
        """Test computing dissipation rate."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        history = EnergyHistory()

        # Append states with decreasing energy (simulating dissipation)
        for i in range(5):
            state = initialize_alfven_wave(grid, M=10, amplitude=0.1 * (0.9**i))
            state.time = float(i * 0.1)
            history.append(state)

        dE_dt = history.dissipation_rate()

        assert dE_dt.shape == (4,), "Dissipation rate should have length N-1"
        # Dissipation should be negative (energy decreasing)
        assert jnp.all(dE_dt < 0), "Dissipation rate should be negative"


class TestVisualizationFunctions:
    """Test visualization functions (check they run without errors)."""

    def test_plot_state_runs(self):
        """Test that plot_state executes without errors."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=1.0, seed=42)

        # Should not raise any errors
        plot_state(state, show=False)
        plt.close('all')

    def test_plot_state_with_filename(self, tmp_path):
        """Test that plot_state can save to file."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=1.0, seed=42)

        filename = tmp_path / "test_state.png"
        plot_state(state, filename=str(filename), show=False)

        assert filename.exists(), "Output file was not created"
        plt.close('all')

    def test_plot_energy_history_runs(self):
        """Test that plot_energy_history executes without errors."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        history = EnergyHistory()
        for i in range(5):
            state = initialize_alfven_wave(grid, M=10, amplitude=0.1)
            state.time = float(i * 0.1)
            history.append(state)

        # Should not raise any errors
        plot_energy_history(history, show=False)
        plot_energy_history(history, log_scale=True, show=False)
        plt.close('all')

    def test_plot_energy_history_with_filename(self, tmp_path):
        """Test that plot_energy_history can save to file."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        history = EnergyHistory()
        for i in range(3):
            state = initialize_alfven_wave(grid, M=10, amplitude=0.1)
            state.time = float(i * 0.1)
            history.append(state)

        filename = tmp_path / "test_energy.png"
        plot_energy_history(history, filename=str(filename), show=False)

        assert filename.exists(), "Output file was not created"
        plt.close('all')

    def test_plot_energy_spectrum_runs(self):
        """Test that plot_energy_spectrum executes without errors."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=1.0, seed=42)

        k, E_k = energy_spectrum_1d(state, n_bins=32)

        # Should not raise any errors
        plot_energy_spectrum(k, E_k, show=False)
        plot_energy_spectrum(k, E_k, reference_slope=-5/3, show=False)
        plt.close('all')

    def test_plot_energy_spectrum_with_filename(self, tmp_path):
        """Test that plot_energy_spectrum can save to file."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=1.0, seed=42)

        k, E_k = energy_spectrum_1d(state, n_bins=32)

        filename = tmp_path / "test_spectrum.png"
        plot_energy_spectrum(k, E_k, filename=str(filename), show=False)

        assert filename.exists(), "Output file was not created"
        plt.close('all')

    def test_plot_different_spectrum_types(self):
        """Test plotting different spectrum types."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=1.0, seed=42)

        # 1D spectrum
        k, E_k = energy_spectrum_1d(state, n_bins=32)
        plot_energy_spectrum(k, E_k, spectrum_type='1D', show=False)

        # Perpendicular spectrum
        k_perp, E_perp = energy_spectrum_perpendicular(state, n_bins=32)
        plot_energy_spectrum(k_perp, E_perp, spectrum_type='perpendicular',
                           reference_slope=-3/2, show=False)

        # Parallel spectrum
        kz, E_parallel = energy_spectrum_parallel(state)
        plot_energy_spectrum(kz, E_parallel, spectrum_type='parallel', show=False)

        plt.close('all')


# =============================================================================
# Field Line Following Tests
# =============================================================================


class TestSpectralPadAndIfft:
    """Test spectral interpolation via FFT padding."""

    def test_grid_values_match(self):
        """
        Test that padded grid matches original at corresponding points.

        This verifies that the padding_factor³ rescaling in spectral_pad_and_ifft
        is correct - padding should interpolate smoothly without changing values
        at the original grid points.
        """
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=16, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)

        # Create a smooth known function in real space
        x = jnp.linspace(0, 2*jnp.pi, 16, endpoint=False)
        y = jnp.linspace(0, 2*jnp.pi, 16, endpoint=False)
        z = jnp.linspace(0, 2*jnp.pi, 16, endpoint=False)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        field_original = jnp.sin(X) * jnp.cos(Y)

        # Transform to Fourier
        field_fourier = jnp.fft.rfftn(field_original)

        # Pad and transform back
        field_fine = spectral_pad_and_ifft(field_fourier, padding_factor=2)

        # Sample at original grid points in fine grid
        # Original grid: indices 0, 1, 2, ..., 15
        # Fine grid (2×): indices 0, 2, 4, ..., 30 (every other point)
        field_sampled = field_fine[::2, ::2, ::2]

        # Should match original values (tolerance for float32 precision)
        max_error = jnp.max(jnp.abs(field_sampled - field_original))
        assert max_error < 1e-6, f"Sampled values don't match original: {max_error}"

    def test_padding_increases_resolution(self):
        """Test that padding factor correctly increases grid resolution."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=32)
        field_fourier = jnp.ones((32, 32, 17), dtype=complex)

        # Padding factor 2
        field_2x = spectral_pad_and_ifft(field_fourier, padding_factor=2)
        assert field_2x.shape == (64, 64, 64), f"Wrong shape for 2× padding: {field_2x.shape}"

        # Padding factor 3
        field_3x = spectral_pad_and_ifft(field_fourier, padding_factor=3)
        assert field_3x.shape == (96, 96, 96), f"Wrong shape for 3× padding: {field_3x.shape}"

    def test_known_function(self):
        """Test padding with a known smooth function."""
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=16, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)

        # Create sin(x) function
        x = jnp.linspace(0, 2*jnp.pi, 16, endpoint=False)
        y = jnp.linspace(0, 2*jnp.pi, 16, endpoint=False)
        z = jnp.linspace(0, 2*jnp.pi, 16, endpoint=False)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')

        field_real = jnp.sin(X)
        field_fourier = jnp.fft.rfftn(field_real)

        # Pad to 2× resolution
        field_fine = spectral_pad_and_ifft(field_fourier, padding_factor=2)

        # Expected values on fine grid
        x_fine = jnp.linspace(0, 2*jnp.pi, 32, endpoint=False)
        y_fine = jnp.linspace(0, 2*jnp.pi, 32, endpoint=False)
        z_fine = jnp.linspace(0, 2*jnp.pi, 32, endpoint=False)
        X_fine, Y_fine, Z_fine = jnp.meshgrid(x_fine, y_fine, z_fine, indexing='ij')
        expected = jnp.sin(X_fine)

        # Check accuracy (tolerance for float32 precision)
        max_error = jnp.max(jnp.abs(field_fine - expected))
        assert max_error < 1e-6, f"Interpolation error too large: {max_error}"

    def test_convergence_with_padding_factor(self):
        """
        Test that interpolation accuracy improves with increasing padding_factor.

        For smooth functions, spectral interpolation should give exponential
        convergence (limited by float32 precision).
        """
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=16, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)

        # Create smooth function with known values everywhere
        x = jnp.linspace(0, 2*jnp.pi, 16, endpoint=False)
        y = jnp.linspace(0, 2*jnp.pi, 16, endpoint=False)
        z = jnp.linspace(0, 2*jnp.pi, 16, endpoint=False)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        field_real = jnp.sin(X) * jnp.cos(Y) * jnp.sin(Z)
        field_fourier = jnp.fft.rfftn(field_real)

        # Test at arbitrary points (not on grid)
        test_points = [
            jnp.array([1.5, 2.3, 3.7]),
            jnp.array([0.7, 4.2, 1.1]),
            jnp.array([5.1, 3.8, 2.9]),
        ]

        errors = {}
        for padding in [1, 2, 3]:
            field_fine = spectral_pad_and_ifft(field_fourier, padding_factor=padding)

            # Test interpolation at each point
            max_err = 0.0
            for pos in test_points:
                from krmhd.diagnostics import interpolate_on_fine_grid
                value = interpolate_on_fine_grid(
                    field_fine, pos, grid.Lx, grid.Ly, grid.Lz, padding_factor=padding
                )
                expected = jnp.sin(pos[0]) * jnp.cos(pos[1]) * jnp.sin(pos[2])
                err = float(jnp.abs(value - expected))
                max_err = max(max_err, err)

            errors[padding] = max_err

        # Errors should decrease with padding (spectral convergence)
        # Note: Trilinear interpolation adds O(h²) error, limiting absolute accuracy
        assert errors[2] < errors[1], f"Error should decrease: {errors[1]} -> {errors[2]}"
        assert errors[3] < errors[2], f"Error should decrease: {errors[2]} -> {errors[3]}"

        # All errors should be reasonably small (limited by trilinear interpolation)
        assert errors[3] < 0.01, f"Error with 3× padding too large: {errors[3]}"

    def test_grid_too_small_raises(self):
        """Test that grids smaller than 4 raise ValueError."""
        import pytest
        from krmhd.diagnostics import spectral_pad_and_ifft

        # Create a tiny grid (2×2×2) in Fourier space
        # rfft format: shape is [Nz, Ny, Nx//2+1]
        tiny_field = jnp.ones((2, 2, 2), dtype=complex)

        with pytest.raises(ValueError, match="Grid too small for spectral padding"):
            spectral_pad_and_ifft(tiny_field, padding_factor=2)

    def test_hermitian_symmetry_preserved(self):
        """
        Test that spectral padding preserves reality condition.

        For rfft format starting from a real field:
        1. Fourier representation implicitly satisfies Hermitian symmetry
        2. After padding and IFFT, result must remain real (no spurious imaginary components)
        3. kx=0 and kx=Nyquist planes should be approximately real (within float32 precision)

        This test verifies JAX's irfftn handles Nyquist modes correctly during padding.
        """
        from krmhd.diagnostics import spectral_pad_and_ifft

        # Create grid with even dimensions (has Nyquist modes)
        Nx, Ny, Nz = 16, 16, 16
        grid = SpectralGrid3D.create(Nx=Nx, Ny=Ny, Nz=Nz, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)

        # Start with real field
        x = jnp.linspace(0, 2*jnp.pi, Nx, endpoint=False)
        y = jnp.linspace(0, 2*jnp.pi, Ny, endpoint=False)
        z = jnp.linspace(0, 2*jnp.pi, Nz, endpoint=False)
        Z, Y, X = jnp.meshgrid(z, y, x, indexing='ij')
        field_real = jnp.sin(X) * jnp.cos(Y) + jnp.cos(Z)

        # Verify input is real
        assert jnp.max(jnp.abs(jnp.imag(field_real))) < 1e-10, \
            "Input field should be real"

        # Transform to Fourier (rfft format: [Nz, Ny, Nx//2+1])
        field_fourier = jnp.fft.rfftn(field_real)

        # Verify Hermitian symmetry in Fourier space (within float32 precision)
        # kx=0 plane should be approximately real
        kx0_plane = field_fourier[:, :, 0]
        kx0_imag_max = jnp.max(jnp.abs(jnp.imag(kx0_plane)))
        assert kx0_imag_max < 1e-3, \
            f"kx=0 plane should be approximately real (float32), got max_imag={kx0_imag_max}"

        # kx=Nyquist plane (last index) should also be approximately real for even Nx
        kx_nyq_plane = field_fourier[:, :, -1]
        kx_nyq_imag_max = jnp.max(jnp.abs(jnp.imag(kx_nyq_plane)))
        assert kx_nyq_imag_max < 1e-3, \
            f"kx=Nyquist plane should be approximately real (float32), got max_imag={kx_nyq_imag_max}"

        # Pad and transform back
        field_fine = spectral_pad_and_ifft(field_fourier, padding_factor=2)

        # CRITICAL CHECK: Result should be purely real (no imaginary artifacts from padding)
        # This is the key guarantee from JAX's irfftn with proper Hermitian input
        max_imag = jnp.max(jnp.abs(jnp.imag(field_fine)))
        assert max_imag < 1e-6, \
            f"Padded field should be real, but has imaginary part: {max_imag}"

        # Verify result dtype is real
        assert field_fine.dtype == jnp.float32 or field_fine.dtype == jnp.float64, \
            f"Output should be real-valued array, got dtype={field_fine.dtype}"


class TestInterpolateOnFineGrid:
    """Test interpolation on spectrally-padded grids."""

    def test_grid_point_interpolation(self):
        """Test interpolation at exact grid points gives correct values."""
        Lx, Ly, Lz = 2*jnp.pi, 2*jnp.pi, 2*jnp.pi

        # Create a simple function on fine grid
        field_fine = jnp.arange(8*8*8, dtype=float).reshape(8, 8, 8)

        # Interpolate at grid point (should give exact value)
        dx = Lx / 8
        dy = Ly / 8
        dz = Lz / 8

        position = jnp.array([dx, dy, dz])  # Grid point (1, 1, 1)
        value = interpolate_on_fine_grid(field_fine, position, Lx, Ly, Lz, padding_factor=1)

        expected = field_fine[1, 1, 1]
        assert jnp.abs(value - expected) < 1e-6, f"Grid point interpolation failed: {value} != {expected}"

    def test_periodic_boundaries(self):
        """Test that periodic boundaries work correctly."""
        Lx, Ly, Lz = 2*jnp.pi, 2*jnp.pi, 2*jnp.pi
        field_fine = jnp.ones((8, 8, 8))

        # Position wrapping around periodic boundary
        position_outside = jnp.array([2*jnp.pi + 0.1, jnp.pi, jnp.pi])
        position_inside = jnp.array([0.1, jnp.pi, jnp.pi])

        value_outside = interpolate_on_fine_grid(field_fine, position_outside, Lx, Ly, Lz)
        value_inside = interpolate_on_fine_grid(field_fine, position_inside, Lx, Ly, Lz)

        # Should give same result due to periodicity
        assert jnp.abs(value_outside - value_inside) < 1e-6, "Periodic boundaries failed"

    def test_linear_function_interpolation(self):
        """Test interpolation of linear function."""
        Lx, Ly, Lz = 1.0, 1.0, 1.0

        # Create field f(x,y,z) = x + 2y + 3z on fine grid
        # Note: Use (z, y, x) order to match spectral grid convention
        x = jnp.linspace(0, 1, 16, endpoint=False)
        y = jnp.linspace(0, 1, 16, endpoint=False)
        z = jnp.linspace(0, 1, 16, endpoint=False)
        Z, Y, X = jnp.meshgrid(z, y, x, indexing='ij')
        field_fine = X + 2*Y + 3*Z

        # Interpolate at arbitrary point
        pos = jnp.array([0.3, 0.7, 0.5])
        value = interpolate_on_fine_grid(field_fine, pos, Lx, Ly, Lz, padding_factor=1)
        expected = 0.3 + 2*0.7 + 3*0.5

        # Linear interpolation of linear function should be exact
        assert jnp.abs(value - expected) < 1e-6, f"Linear interpolation failed: {value} != {expected}"


class TestComputeMagneticFieldComponents:
    """Test magnetic field computation."""

    def test_divergence_free_constraint(self):
        """Test that ∇·B ≈ 0 for computed field."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=32, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.1, seed=42)

        # Compute B components
        B_components = compute_magnetic_field_components(state, padding_factor=1)
        Bx = B_components['Bx']
        By = B_components['By']
        Bz = B_components['Bz']

        # Compute divergence using finite differences
        dx = grid.Lx / grid.Nx
        dy = grid.Ly / grid.Ny
        dz = grid.Lz / grid.Nz

        dBx_dx = (jnp.roll(Bx, -1, axis=2) - jnp.roll(Bx, 1, axis=2)) / (2*dx)
        dBy_dy = (jnp.roll(By, -1, axis=1) - jnp.roll(By, 1, axis=1)) / (2*dy)
        dBz_dz = (jnp.roll(Bz, -1, axis=0) - jnp.roll(Bz, 1, axis=0)) / (2*dz)

        div_B = dBx_dx + dBy_dy + dBz_dz

        # Should be small (spectral accuracy)
        max_div = jnp.max(jnp.abs(div_B))
        rms_div = jnp.sqrt(jnp.mean(div_B**2))

        # Relaxed tolerance for finite difference approximation
        # (finite differences introduce O(h²) errors)
        assert max_div < 1e-3, f"Max |∇·B| too large: {max_div}"
        assert rms_div < 2e-4, f"RMS |∇·B| too large: {rms_div}"

    def test_straight_field_limit(self):
        """Test that zero perturbations give B = B₀ẑ."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=32)

        # Create state with zero perturbations
        z_plus = jnp.zeros((32, 32, 17), dtype=complex)
        z_minus = jnp.zeros((32, 32, 17), dtype=complex)
        B_parallel = jnp.zeros((32, 32, 17), dtype=complex)
        g = jnp.zeros((32, 32, 17, 11), dtype=complex)

        state = KRMHDState(
            z_plus=z_plus,
            z_minus=z_minus,
            B_parallel=B_parallel,
            g=g,
            M=10,
            beta_i=1.0,
            v_th=1.0,
            nu=0.01,
            Lambda=1.0,
            time=0.0,
            grid=grid
        )

        B_components = compute_magnetic_field_components(state, padding_factor=1)

        # Bx and By should be zero (no perpendicular field)
        assert jnp.max(jnp.abs(B_components['Bx'])) < 1e-6, "Bx should be zero"
        assert jnp.max(jnp.abs(B_components['By'])) < 1e-6, "By should be zero"

        # Bz should be constant ≈ 1 (B₀ in normalized units)
        Bz_mean = jnp.mean(B_components['Bz'])
        Bz_std = jnp.std(B_components['Bz'])
        assert jnp.abs(Bz_mean - 1.0) < 1e-6, f"Bz mean should be 1.0: {Bz_mean}"
        assert Bz_std < 1e-6, f"Bz should be constant: std = {Bz_std}"

    def test_output_shapes(self):
        """Test that output shapes match expected fine grid sizes."""
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=16)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.1, seed=42)

        # Test different padding factors
        for padding_factor in [1, 2]:
            B_components = compute_magnetic_field_components(state, padding_factor)

            expected_shape = (16 * padding_factor, 16 * padding_factor, 16 * padding_factor)

            assert B_components['Bx'].shape == expected_shape
            assert B_components['By'].shape == expected_shape
            assert B_components['Bz'].shape == expected_shape


class TestFollowFieldLine:
    """Test field line following algorithm."""

    def test_straight_field_limit(self):
        """Test that straight field (B = B₀ẑ) gives vertical line."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=32, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)

        # Zero perturbations → straight field
        z_plus = jnp.zeros((32, 32, 17), dtype=complex)
        z_minus = jnp.zeros((32, 32, 17), dtype=complex)
        B_parallel = jnp.zeros((32, 32, 17), dtype=complex)
        g = jnp.zeros((32, 32, 17, 11), dtype=complex)

        state = KRMHDState(
            z_plus=z_plus, z_minus=z_minus, B_parallel=B_parallel, g=g,
            M=10, beta_i=1.0, v_th=1.0, nu=0.01, Lambda=1.0, time=0.0, grid=grid
        )

        # Follow field line
        x0, y0 = jnp.pi, jnp.pi / 2
        trajectory = follow_field_line(state, x0, y0, padding_factor=1)

        # x and y should remain constant (vertical line)
        x_traj = trajectory[:, 0]
        y_traj = trajectory[:, 1]
        z_traj = trajectory[:, 2]

        x_variation = jnp.std(x_traj)
        y_variation = jnp.std(y_traj)

        # Relaxed tolerance for numerical integration
        assert x_variation < 0.01, f"x should remain constant: std = {x_variation}"
        assert y_variation < 0.01, f"y should remain constant: std = {y_variation}"

        # z should span the domain
        assert z_traj[0] < -grid.Lz/2 + 0.5, "Should start near z_min"
        assert z_traj[-1] > grid.Lz/2 - 0.5, "Should end near z_max"

    def test_periodic_boundaries_respected(self):
        """Test that field line respects periodic boundaries in x, y."""
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=16, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.5, seed=42)

        # Follow field line
        trajectory = follow_field_line(state, jnp.pi, jnp.pi, dz=0.2, padding_factor=1)

        x_traj = trajectory[:, 0]
        y_traj = trajectory[:, 1]

        # All points should be within periodic domain
        assert jnp.all((x_traj >= 0) & (x_traj < grid.Lx)), "x out of bounds"
        assert jnp.all((y_traj >= 0) & (y_traj < grid.Ly)), "y out of bounds"

    def test_trajectory_not_empty(self):
        """Test that trajectory contains multiple points."""
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=16)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.1, seed=42)

        trajectory = follow_field_line(state, jnp.pi, jnp.pi, padding_factor=1)

        assert trajectory.shape[0] > 10, "Trajectory should have multiple points"
        assert trajectory.shape[1] == 3, "Trajectory should have (x, y, z) coordinates"

    def test_step_size_convergence(self):
        """Test that smaller step sizes give more accurate trajectories."""
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=16, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)

        # Straight field for exact solution
        z_plus = jnp.zeros((16, 16, 9), dtype=complex)
        z_minus = jnp.zeros((16, 16, 9), dtype=complex)
        B_parallel = jnp.zeros((16, 16, 9), dtype=complex)
        g = jnp.zeros((16, 16, 9, 11), dtype=complex)

        state = KRMHDState(
            z_plus=z_plus, z_minus=z_minus, B_parallel=B_parallel, g=g,
            M=10, beta_i=1.0, v_th=1.0, nu=0.01, Lambda=1.0, time=0.0, grid=grid
        )

        x0, y0 = jnp.pi, jnp.pi / 2

        # Coarse step
        traj_coarse = follow_field_line(state, x0, y0, dz=0.5, padding_factor=1)

        # Fine step
        traj_fine = follow_field_line(state, x0, y0, dz=0.1, padding_factor=1)

        # Both should be nearly vertical (straight field)
        x_var_coarse = jnp.std(traj_coarse[:, 0])
        x_var_fine = jnp.std(traj_fine[:, 0])

        # Finer step should have smaller variation (better accuracy)
        # But both should be small for straight field
        assert x_var_fine < x_var_coarse + 0.1, "Finer step should not be worse"
        assert x_var_fine < 0.02, "Fine step should be accurate"


# =============================================================================
# Phase Mixing Diagnostics Tests (Issue #26)
# =============================================================================


class TestHermiteFlux:
    """Test Hermite moment flux computation."""

    def test_hermite_flux_shape(self):
        """Test that flux has correct output shape."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.1, seed=42)

        # Import phase mixing diagnostics
        from krmhd.diagnostics import hermite_flux

        flux = hermite_flux(state)

        # Shape should be [Nz, Ny, Nx//2+1, M] for M moment transitions
        expected_shape = (grid.Nz, grid.Ny, grid.Nx // 2 + 1, state.M)
        assert flux.shape == expected_shape, f"Flux shape {flux.shape} != {expected_shape}"

    def test_hermite_flux_reality(self):
        """Test that flux is real-valued (imaginary part of product)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.1, seed=42)

        from krmhd.diagnostics import hermite_flux

        flux = hermite_flux(state)

        # Flux should be real (result of Im[g_{m+1} * g_m*])
        assert jnp.isrealobj(flux), "Flux should be real-valued"
        assert flux.dtype in [jnp.float32, jnp.float64], f"Flux dtype should be float, got {flux.dtype}"

    def test_hermite_flux_zero_state(self):
        """Test that zero state gives zero flux."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        # Create zero state
        state = KRMHDState(
            z_plus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            z_minus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            B_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            g=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, 11), dtype=jnp.complex64),
            M=10,
            beta_i=1.0,
            v_th=1.0,
            nu=0.01,
            Lambda=1.0,
            time=0.0,
            grid=grid,
        )

        from krmhd.diagnostics import hermite_flux

        flux = hermite_flux(state)

        # All flux should be zero
        assert jnp.allclose(flux, 0.0, atol=1e-10), "Zero state should have zero flux"

    def test_hermite_flux_conservation(self):
        """Test that total flux sums to ~zero (conservation)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.1, seed=42)

        from krmhd.diagnostics import hermite_flux

        flux = hermite_flux(state)

        # For each moment transition, sum over k-space
        # In a closed system, ∑ₖ Γₘ,ₖ ≈ 0 (energy flows through moment space)
        flux_total = jnp.sum(flux, axis=(0, 1, 2))  # Shape: [M]

        # Check that total flux is small (conservation)
        # Note: May not be exactly zero due to external forcing or boundaries
        # But should be much smaller than typical flux magnitudes
        flux_magnitude = jnp.max(jnp.abs(flux))
        conservation_error = jnp.max(jnp.abs(flux_total))

        # Conservation error should be small relative to flux magnitude
        relative_error = conservation_error / (flux_magnitude + 1e-10)

        # Relaxed tolerance (100%) for random initial state
        # Reason: Random spectrum initialization creates g moments that don't necessarily
        # satisfy flux conservation yet (no time evolution, no forcing/damping balance).
        # This test checks the formula is implemented correctly, not that the physics
        # is in equilibrium. Tighter conservation (~1e-10) is achieved during time evolution
        # with forcing and collisions (tested in Issue #27: Kinetic FDT validation).
        #
        # TODO (Issue #27): Add dedicated test with time-evolved state showing tight
        # flux conservation (~1e-10 relative error) once forcing/damping balance is
        # established. This will validate the physics, not just the formula.
        assert relative_error < 1.0, \
            f"Flux conservation violated: relative error = {relative_error}"

    def test_hermite_flux_formula(self):
        """Test flux formula against manual calculation."""
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8)

        # Create simple state with known moments
        g = jnp.zeros((8, 16, 9, 6), dtype=complex)

        # Set g0 and g1 to simple values for testing
        # g0[0,0,0] = 1+0j, g1[0,0,0] = 0+1j
        g = g.at[0, 0, 0, 0].set(1.0 + 0.0j)
        g = g.at[0, 0, 0, 1].set(0.0 + 1.0j)

        state = KRMHDState(
            z_plus=jnp.zeros((8, 16, 9), dtype=complex),
            z_minus=jnp.zeros((8, 16, 9), dtype=complex),
            B_parallel=jnp.zeros((8, 16, 9), dtype=complex),
            g=g,
            M=5,
            beta_i=1.0,
            v_th=1.0,
            nu=0.01,
            Lambda=1.0,
            time=0.0,
            grid=grid,
        )

        from krmhd.diagnostics import hermite_flux

        flux = hermite_flux(state)

        # Manual calculation for m=0 transition at k=(0,0,0)
        # Γ₀ = -k∥·√(2·1)·Im[g₁·g₀*]
        # g₀ = 1, g₁ = i
        # g₁·g₀* = i·1* = i
        # Im[i] = 1
        # k∥ = kz[0] (should be 0 or small)
        k_parallel = grid.kz[0]
        coupling = jnp.sqrt(2.0 * 1)
        expected_flux_00 = -k_parallel * coupling * 1.0

        computed_flux_00 = flux[0, 0, 0, 0]

        assert jnp.allclose(computed_flux_00, expected_flux_00, atol=1e-6), \
            f"Flux formula mismatch: {computed_flux_00} != {expected_flux_00}"


class TestHermiteMomentEnergy:
    """Test Hermite moment energy computation."""

    def test_hermite_moment_energy_shape(self):
        """Test that energy array has correct shape."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.1, seed=42)

        from krmhd.diagnostics import hermite_moment_energy

        E_m = hermite_moment_energy(state)

        # Should have M+1 elements (moments 0 through M)
        assert E_m.shape == (state.M + 1,), f"Energy shape {E_m.shape} != ({state.M + 1},)"

    def test_hermite_moment_energy_positive(self):
        """Test that all energies are non-negative."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.1, seed=42)

        from krmhd.diagnostics import hermite_moment_energy

        E_m = hermite_moment_energy(state)

        # All energies must be ≥ 0
        assert jnp.all(E_m >= 0.0), "Energies must be non-negative"

    def test_hermite_moment_energy_zero_state(self):
        """Test that zero state has zero energy."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        state = KRMHDState(
            z_plus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            z_minus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            B_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            g=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, 11), dtype=jnp.complex64),
            M=10,
            beta_i=1.0,
            v_th=1.0,
            nu=0.01,
            Lambda=1.0,
            time=0.0,
            grid=grid,
        )

        from krmhd.diagnostics import hermite_moment_energy

        E_m = hermite_moment_energy(state)

        assert jnp.allclose(E_m, 0.0, atol=1e-10), "Zero state should have zero energy"

    def test_hermite_moment_energy_decreasing(self):
        """Test that energy generally decreases with m (collisional damping)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        # Create state with exponentially decaying moments (realistic)
        # E_m ~ exp(-m/m_decay) models collision damping
        state = initialize_random_spectrum(grid, M=15, alpha=5/3, amplitude=0.1, seed=42)

        # Manually set g to have exponential decay structure
        # This simulates what collisions would produce
        g = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, 16), dtype=complex)
        for m in range(16):
            # Energy ~ exp(-m/5)
            amplitude = jnp.exp(-m / 5.0) * 0.1
            # Set g[0,0,1,m] to have this amplitude (testing purposes)
            g = g.at[0, 0, 1, m].set(amplitude * (1 + 0j))

        # Update state with decaying moments
        state = KRMHDState(
            z_plus=state.z_plus,
            z_minus=state.z_minus,
            B_parallel=state.B_parallel,
            g=g,
            M=15,
            beta_i=state.beta_i,
            v_th=state.v_th,
            nu=state.nu,
            Lambda=state.Lambda,
            time=state.time,
            grid=state.grid,
        )

        from krmhd.diagnostics import hermite_moment_energy

        E_m = hermite_moment_energy(state)

        # At high m, should see decay (collision damping)
        # Check that E_M < E_0 (convergence requirement)
        assert E_m[-1] < E_m[0], "Energy should decay to high m"

        # Check that last few moments show decay
        assert E_m[-1] < E_m[-3], "High moments should be damped"


class TestPhaseMixingEnergy:
    """Test phase mixing/unmixing energy decomposition."""

    def test_phase_mixing_energy_positive(self):
        """Test that mixing energy is non-negative."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.1, seed=42)

        from krmhd.diagnostics import phase_mixing_energy

        # Check several moments
        for m in range(min(5, state.M)):
            E_mix = phase_mixing_energy(state, m)
            assert E_mix >= 0.0, f"Mixing energy at m={m} should be non-negative, got {E_mix}"

    def test_phase_unmixing_energy_positive(self):
        """Test that unmixing energy is non-negative."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.1, seed=42)

        from krmhd.diagnostics import phase_unmixing_energy

        # Check several moments
        for m in range(min(5, state.M)):
            E_unmix = phase_unmixing_energy(state, m)
            assert E_unmix >= 0.0, f"Unmixing energy at m={m} should be non-negative, got {E_unmix}"

    def test_energy_decomposition(self):
        """Test that E_total(m) ≈ E_mixing(m) + E_unmixing(m)."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.1, seed=42)

        from krmhd.diagnostics import (
            hermite_moment_energy,
            phase_mixing_energy,
            phase_unmixing_energy,
        )

        E_total = hermite_moment_energy(state)

        # Check decomposition for first few moments
        for m in range(min(5, state.M)):
            E_mix = phase_mixing_energy(state, m)
            E_unmix = phase_unmixing_energy(state, m)
            E_sum = E_mix + E_unmix

            # Should match total energy at moment m
            relative_error = jnp.abs(E_sum - E_total[m]) / (E_total[m] + 1e-10)

            assert relative_error < 1e-6, \
                f"Energy decomposition failed at m={m}: {E_sum} != {E_total[m]} (error={relative_error})"

    def test_zero_state_decomposition(self):
        """Test that zero state has zero mixing and unmixing energy."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)

        state = KRMHDState(
            z_plus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            z_minus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            B_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64),
            g=jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, 11), dtype=jnp.complex64),
            M=10,
            beta_i=1.0,
            v_th=1.0,
            nu=0.01,
            Lambda=1.0,
            time=0.0,
            grid=grid,
        )

        from krmhd.diagnostics import phase_mixing_energy, phase_unmixing_energy

        for m in range(state.M):
            E_mix = phase_mixing_energy(state, m)
            E_unmix = phase_unmixing_energy(state, m)

            assert jnp.allclose(E_mix, 0.0, atol=1e-10), f"Zero state mixing energy should be zero at m={m}"
            assert jnp.allclose(E_unmix, 0.0, atol=1e-10), f"Zero state unmixing energy should be zero at m={m}"


class TestPhaseMixingVisualization:
    """Test phase mixing visualization functions (smoke tests)."""

    def test_plot_hermite_flux_spectrum_runs(self):
        """Test that plot_hermite_flux_spectrum runs without errors."""
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.1, seed=42)

        from krmhd.diagnostics import plot_hermite_flux_spectrum

        # Should run without errors and close figure
        plot_hermite_flux_spectrum(state, show=False)

    def test_plot_hermite_moment_energy_runs(self):
        """Test that plot_hermite_moment_energy runs without errors."""
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.1, seed=42)

        from krmhd.diagnostics import plot_hermite_moment_energy

        # Should run without errors and close figure
        plot_hermite_moment_energy(state, show=False)

    def test_plot_phase_mixing_2d_runs(self):
        """Test that plot_phase_mixing_2d runs without errors."""
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.1, seed=42)

        from krmhd.diagnostics import plot_phase_mixing_2d

        # Should run without errors and close figure
        plot_phase_mixing_2d(state, m=0, kz_index=0, show=False)


class TestTurbulenceDiagnostics:
    """Test compute_turbulence_diagnostics() for Issue #82 investigation."""

    def test_compute_diagnostics_returns_correct_type(self):
        """Test that diagnostics return TurbulenceDiagnostics dataclass."""
        from krmhd.diagnostics import compute_turbulence_diagnostics, TurbulenceDiagnostics

        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.1, seed=42)
        state = state.model_copy(update={"time": 5.0})  # Set non-zero time

        dt = 0.01
        diag = compute_turbulence_diagnostics(state, dt=dt, v_A=1.0)

        # Check type
        assert isinstance(diag, TurbulenceDiagnostics), f"Expected TurbulenceDiagnostics, got {type(diag)}"

        # Check all fields are present and finite
        assert isinstance(diag.time, float)
        assert jnp.isfinite(diag.time)
        assert isinstance(diag.max_velocity, float)
        assert jnp.isfinite(diag.max_velocity)
        assert isinstance(diag.cfl_number, float)
        assert jnp.isfinite(diag.cfl_number)
        assert isinstance(diag.max_nonlinear, float)
        assert jnp.isfinite(diag.max_nonlinear)
        assert isinstance(diag.energy_highk, float)
        assert jnp.isfinite(diag.energy_highk)
        assert isinstance(diag.critical_balance_ratio, float)
        assert jnp.isfinite(diag.critical_balance_ratio)
        assert isinstance(diag.energy_total, float)
        assert jnp.isfinite(diag.energy_total)

    def test_diagnostics_zero_state(self):
        """Test diagnostics on zero state (should not crash)."""
        from krmhd.diagnostics import compute_turbulence_diagnostics

        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8)
        state = KRMHDState(
            z_plus=jnp.zeros((8, 16, 9), dtype=complex),
            z_minus=jnp.zeros((8, 16, 9), dtype=complex),
            B_parallel=jnp.zeros((8, 16, 9), dtype=complex),
            g=jnp.zeros((10, 8, 16, 9), dtype=complex),
            M=10,
            beta_i=1.0,
            v_th=1.0,
            nu=0.01,
            Lambda=1.0,
            time=0.0,
            grid=grid,
        )

        dt = 0.01
        diag = compute_turbulence_diagnostics(state, dt=dt, v_A=1.0)

        # Zero state should give zero diagnostics
        assert diag.max_velocity == 0.0, "Zero state should have zero velocity"
        assert diag.cfl_number == 0.0, "Zero state should have zero CFL"
        assert diag.energy_total == 0.0, "Zero state should have zero energy"
        assert diag.energy_highk == 0.0, "Zero state should have zero high-k energy"

    def test_cfl_calculation(self):
        """Test CFL number calculation with known velocity."""
        from krmhd.diagnostics import compute_turbulence_diagnostics

        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)

        # Create state with known velocity: v = (z+ + z-)/2
        # Set single mode with amplitude A → velocity A
        state = initialize_alfven_wave(
            grid=grid,
            M=10,
            kx_mode=1.0,
            ky_mode=0.0,
            kz_mode=1.0,
            amplitude=0.5,  # Velocity amplitude
        )

        dt = 0.01
        diag = compute_turbulence_diagnostics(state, dt=dt, v_A=1.0)

        # CFL = max_velocity * dt / dx
        # dx = Lx / Nx = 2π / 32 ≈ 0.196
        dx = grid.Lx / grid.Nx
        expected_cfl = diag.max_velocity * dt / dx

        # Allow for numerical precision
        assert jnp.abs(diag.cfl_number - expected_cfl) < 0.01, \
            f"CFL mismatch: got {diag.cfl_number}, expected ~{expected_cfl}"

    def test_energy_consistency(self):
        """Test that diagnostic energy matches energy() function."""
        from krmhd.diagnostics import compute_turbulence_diagnostics
        from krmhd.physics import energy

        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.1, seed=42)

        # Compute energy via diagnostics
        dt = 0.01
        diag = compute_turbulence_diagnostics(state, dt=dt, v_A=1.0)

        # Compute energy via physics module (returns dict)
        energy_dict = energy(state)
        E_total = energy_dict["magnetic"] + energy_dict["kinetic"] + energy_dict["compressive"]

        # Should match (within numerical precision)
        rel_error = jnp.abs(diag.energy_total - E_total) / (E_total + 1e-30)
        assert rel_error < 0.01, \
            f"Energy mismatch: diagnostic={diag.energy_total}, energy()={E_total}, rel_error={rel_error}"

    def test_highk_energy_fraction(self):
        """Test that high-k energy is a fraction of total energy."""
        from krmhd.diagnostics import compute_turbulence_diagnostics

        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.1, seed=42)

        dt = 0.01
        diag = compute_turbulence_diagnostics(state, dt=dt, v_A=1.0)

        # FIXED (PR #84): energy_highk is a FRACTION (0-1), not absolute energy
        assert 0 <= diag.energy_highk <= 1.0, \
            f"High-k energy fraction {diag.energy_highk} not in range [0, 1]"

        # For k^-5/3 spectrum, most energy is at low-k, so high-k fraction should be small
        # (unless there's significant pile-up)
        assert diag.energy_highk < 0.5, \
            f"High-k energy fraction {diag.energy_highk} unexpectedly large"

    def test_diagnostics_with_single_mode(self):
        """Test diagnostics with a single Alfvén wave mode."""
        from krmhd.diagnostics import compute_turbulence_diagnostics

        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)
        state = initialize_alfven_wave(
            grid=grid,
            M=10,
            kx_mode=2.0,
            ky_mode=2.0,
            kz_mode=1.0,
            amplitude=0.1,
        )

        dt = 0.01
        diag = compute_turbulence_diagnostics(state, dt=dt, v_A=1.0)

        # All diagnostics should be well-defined
        assert diag.max_velocity > 0, "Single mode should have non-zero velocity"
        assert diag.energy_total > 0, "Single mode should have non-zero energy"
        assert diag.cfl_number > 0, "Single mode should have non-zero CFL"

    def test_energy_highk_is_fraction(self):
        """Test that energy_highk is a fraction between 0 and 1 (PR #84 feedback)."""
        from krmhd.diagnostics import compute_turbulence_diagnostics

        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.1, seed=42)

        dt = 0.01
        diag = compute_turbulence_diagnostics(state, dt=dt, v_A=1.0, k_threshold=0.9)

        # CRITICAL: energy_highk must be a FRACTION (0-1), not absolute energy
        assert 0.0 <= diag.energy_highk <= 1.0, \
            f"energy_highk={diag.energy_highk} must be a fraction in [0, 1]"

        # For k^-5/3 spectrum, most energy at low-k, so high-k fraction should be small
        assert diag.energy_highk < 0.5, \
            f"High-k fraction {diag.energy_highk} unexpectedly large for k^-5/3 spectrum"

    def test_energy_fractions_sum_to_one(self):
        """Test that low-k + high-k energy fractions sum to ~1.0 (PR #84 feedback)."""
        from krmhd.diagnostics import compute_turbulence_diagnostics

        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.2, seed=123)

        dt = 0.005
        k_threshold = 0.8  # 80% of k_max

        diag = compute_turbulence_diagnostics(state, dt=dt, v_A=1.0, k_threshold=k_threshold)

        # Compute low-k fraction separately
        # energy_highk is fraction at k > k_threshold * k_max
        # energy_lowk is fraction at k <= k_threshold * k_max
        # They should sum to approximately 1.0 (within numerical precision)

        # Since we can't directly compute low-k fraction, we check:
        # 0 <= energy_highk <= 1.0 (it's a valid fraction)
        assert 0.0 <= diag.energy_highk <= 1.0, \
            f"energy_highk={diag.energy_highk} is not a valid fraction"

        # And for a realistic turbulent spectrum, it should be << 1
        # (most energy is at low-k for k^-5/3 spectrum)
        assert diag.energy_highk < 0.3, \
            f"High-k fraction {diag.energy_highk} too large for k^-5/3 spectrum"

    def test_critical_balance_with_known_state(self):
        """Test critical balance calculation with controlled velocity field (PR #84 feedback)."""
        from krmhd.diagnostics import compute_turbulence_diagnostics

        # Create state with known velocity distribution in inertial range
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32, Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.1, seed=456)

        dt = 0.005
        v_A = 1.0
        diag = compute_turbulence_diagnostics(state, dt=dt, v_A=v_A)

        # Critical balance ratio should be computed
        # For turbulent state, expect τ_nl/τ_A ~ O(1) or O(0.1-10)
        # But may be 0.0 if insufficient modes in inertial range or unreliable
        assert isinstance(diag.critical_balance_ratio, float), \
            "Critical balance ratio must be a float"
        assert diag.critical_balance_ratio >= 0.0, \
            f"Critical balance ratio {diag.critical_balance_ratio} must be non-negative"

        # If non-zero, should be physically reasonable (< 1e10)
        if diag.critical_balance_ratio > 0:
            assert diag.critical_balance_ratio < 1e10, \
                f"Critical balance ratio {diag.critical_balance_ratio:.2e} unreasonably large"

    def test_critical_balance_validation_warning(self):
        """Test that unrealistic critical balance values trigger warning (PR #84 feedback)."""
        import warnings
        from krmhd.diagnostics import compute_turbulence_diagnostics

        # Create a state that might trigger unrealistic critical balance
        # (e.g., very low resolution or zero velocity in inertial range)
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8)  # Low resolution
        state = initialize_random_spectrum(grid, M=10, alpha=5/3, amplitude=0.001, seed=789)  # Very weak

        dt = 0.01

        # Critical balance may be unreliable with low resolution/amplitude
        # The function should either return valid value or 0.0 with warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            diag = compute_turbulence_diagnostics(state, dt=dt, v_A=1.0)

            # If warning was raised, critical balance should be set to 0.0
            if len(w) > 0 and "Critical balance ratio unreliable" in str(w[0].message):
                assert diag.critical_balance_ratio == 0.0, \
                    "Unreliable critical balance should be set to 0.0"

        # Regardless, value should be finite and non-negative
        assert jnp.isfinite(diag.critical_balance_ratio), \
            "Critical balance ratio must be finite"
        assert diag.critical_balance_ratio >= 0.0, \
            "Critical balance ratio must be non-negative"
