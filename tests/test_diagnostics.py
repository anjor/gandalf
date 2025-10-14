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
