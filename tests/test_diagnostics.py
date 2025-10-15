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
