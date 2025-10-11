"""
Tests for core spectral infrastructure.

Validates:
- FFT forward/inverse operations
- Spectral derivatives (∂x, ∂y)
- Dealiasing (2/3 rule)
- Pydantic model validation
- Laplacian operator composition
"""

import pytest
import jax
import jax.numpy as jnp
from pydantic import ValidationError

from krmhd.spectral import (
    SpectralGrid2D,
    SpectralField2D,
    rfft2_forward,
    rfft2_inverse,
    derivative_x,
    derivative_y,
    laplacian,
    dealias,
)


class TestSpectralGrid2D:
    """Test suite for SpectralGrid2D Pydantic model."""

    def test_create_basic_grid(self):
        """Test basic grid creation with default parameters."""
        grid = SpectralGrid2D.create(Nx=64, Ny=64)

        assert grid.Nx == 64
        assert grid.Ny == 64
        assert jnp.isclose(grid.Lx, 2 * jnp.pi)
        assert jnp.isclose(grid.Ly, 2 * jnp.pi)

        # Check wavenumber array shapes
        assert grid.kx.shape == (64 // 2 + 1,)  # rfft2 convention
        assert grid.ky.shape == (64,)
        assert grid.dealias_mask.shape == (64, 64 // 2 + 1)

    def test_create_custom_domain(self):
        """Test grid creation with custom domain sizes."""
        grid = SpectralGrid2D.create(Nx=128, Ny=256, Lx=4 * jnp.pi, Ly=8 * jnp.pi)

        assert grid.Nx == 128
        assert grid.Ny == 256
        assert jnp.isclose(grid.Lx, 4 * jnp.pi)
        assert jnp.isclose(grid.Ly, 8 * jnp.pi)

    def test_grid_immutability(self):
        """Test that SpectralGrid2D is immutable (frozen=True)."""
        grid = SpectralGrid2D.create(Nx=64, Ny=64)

        with pytest.raises(ValidationError):
            grid.Nx = 128

    def test_validation_positive_dimensions(self):
        """Test that factory method validates positive dimensions."""
        # Negative dimension
        with pytest.raises(ValueError, match="must be positive"):
            SpectralGrid2D.create(Nx=-64, Ny=64)

        # Zero dimension
        with pytest.raises(ValueError, match="must be positive"):
            SpectralGrid2D.create(Nx=64, Ny=0)

        # Negative domain size
        with pytest.raises(ValueError, match="must be positive"):
            SpectralGrid2D.create(Nx=64, Ny=64, Lx=-1.0)

    def test_validation_even_dimensions(self):
        """Test that grid dimensions must be even."""
        with pytest.raises(ValueError, match="must be even"):
            SpectralGrid2D.create(Nx=63, Ny=64)

        with pytest.raises(ValueError, match="must be even"):
            SpectralGrid2D.create(Nx=64, Ny=65)

    def test_wavenumber_arrays(self):
        """Test that wavenumber arrays follow correct FFT conventions."""
        grid = SpectralGrid2D.create(Nx=64, Ny=64, Lx=2 * jnp.pi, Ly=2 * jnp.pi)

        # kx should be non-negative (rfft2 convention)
        assert jnp.all(grid.kx >= 0)
        assert grid.kx[0] == 0.0
        assert grid.kx[-1] == 32  # Nyquist: Nx//2

        # ky should include negative frequencies (standard FFT ordering)
        assert grid.ky[0] == 0.0
        assert grid.ky[grid.Ny // 2] == -32  # Nyquist wraps to negative

    def test_dealias_mask(self):
        """Test that dealiasing mask correctly implements 2/3 rule."""
        grid = SpectralGrid2D.create(Nx=64, Ny=64)

        # Low-k modes should be kept
        assert grid.dealias_mask[0, 0]  # k=0 mode
        assert grid.dealias_mask[1, 1]  # Low-k mode

        # High-k modes should be zeroed
        # Max kx is 32, so 2/3 * 32 ≈ 21.3
        # Max ky is 32, so modes beyond ~21 should be masked
        assert not grid.dealias_mask[0, -1]  # kx = Nyquist
        assert not grid.dealias_mask[32, 0]  # ky = Nyquist (index Ny//2)


class TestFFTOperations:
    """Test suite for FFT forward/inverse operations."""

    def test_fft_roundtrip(self):
        """Test that FFT -> IFFT recovers original field."""
        # Create random real field
        key = jax.random.PRNGKey(0)
        field_real = jax.random.normal(key, (64, 64))

        # Forward and inverse transform
        field_fourier = rfft2_forward(field_real)
        field_recovered = rfft2_inverse(field_fourier, 64, 64)

        # Should match to float32 precision (~1e-6)
        assert jnp.allclose(field_real, field_recovered, atol=1e-6)

    def test_fft_shapes(self):
        """Test that FFT operations produce correct output shapes."""
        Nx, Ny = 128, 256
        field_real = jnp.zeros((Ny, Nx))

        field_fourier = rfft2_forward(field_real)
        assert field_fourier.shape == (Ny, Nx // 2 + 1)

        field_back = rfft2_inverse(field_fourier, Ny, Nx)
        assert field_back.shape == (Ny, Nx)

    def test_reality_condition(self):
        """Test that rfft2 preserves reality condition: F(-k) = F*(k)."""
        # Create real field
        x = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        field_real = jnp.sin(3 * X.T) + jnp.cos(2 * Y.T)

        field_fourier = rfft2_forward(field_real)
        field_back = rfft2_inverse(field_fourier, 64, 64)

        # Inverse should be purely real
        assert jnp.allclose(field_back.imag, 0.0, atol=1e-12)


class TestDerivatives:
    """Test suite for spectral derivative operators."""

    def test_derivative_sine_x(self):
        """Test ∂x(sin(kx)) = k·cos(kx) [REQUIRED by issue #2]."""
        grid = SpectralGrid2D.create(Nx=128, Ny=128, Lx=2 * jnp.pi, Ly=2 * jnp.pi)

        # Create coordinate arrays
        x = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        X = X.T  # Shape [Ny, Nx]

        # Test multiple wavenumbers
        for n in [1, 2, 3, 5]:
            # f(x) = sin(n*x)
            field_real = jnp.sin(n * X)
            field_fourier = rfft2_forward(field_real)

            # Compute spectral derivative
            dfdx_fourier = derivative_x(field_fourier, grid.kx)
            dfdx_real = rfft2_inverse(dfdx_fourier, 128, 128)

            # Analytical derivative: df/dx = n·cos(n*x)
            dfdx_exact = n * jnp.cos(n * X)

            # Should match to float32 precision (~1e-5 for derivatives)
            assert jnp.allclose(dfdx_real, dfdx_exact, atol=1e-5)

    def test_derivative_sine_y(self):
        """Test ∂y(sin(ky)) = k·cos(ky)."""
        grid = SpectralGrid2D.create(Nx=128, Ny=128, Lx=2 * jnp.pi, Ly=2 * jnp.pi)

        # Create coordinate arrays
        x = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        Y = Y.T  # Shape [Ny, Nx]

        # Test multiple wavenumbers
        for n in [1, 2, 3, 5]:
            # f(y) = sin(n*y)
            field_real = jnp.sin(n * Y)
            field_fourier = rfft2_forward(field_real)

            # Compute spectral derivative
            dfdy_fourier = derivative_y(field_fourier, grid.ky)
            dfdy_real = rfft2_inverse(dfdy_fourier, 128, 128)

            # Analytical derivative: df/dy = n·cos(n*y)
            dfdy_exact = n * jnp.cos(n * Y)

            # Should match to float32 precision (~1e-5 for derivatives)
            assert jnp.allclose(dfdy_real, dfdy_exact, atol=1e-5)

    def test_derivative_2d_function(self):
        """Test derivatives on 2D function: f(x,y) = sin(kx*x) * cos(ky*y)."""
        grid = SpectralGrid2D.create(Nx=128, Ny=128, Lx=2 * jnp.pi, Ly=2 * jnp.pi)

        # Create coordinate arrays
        x = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        X, Y = X.T, Y.T

        kx, ky = 3, 2
        field_real = jnp.sin(kx * X) * jnp.cos(ky * Y)
        field_fourier = rfft2_forward(field_real)

        # ∂f/∂x = kx·cos(kx*x)·cos(ky*y)
        dfdx_fourier = derivative_x(field_fourier, grid.kx)
        dfdx_real = rfft2_inverse(dfdx_fourier, 128, 128)
        dfdx_exact = kx * jnp.cos(kx * X) * jnp.cos(ky * Y)
        assert jnp.allclose(dfdx_real, dfdx_exact, atol=1e-5)

        # ∂f/∂y = -ky·sin(kx*x)·sin(ky*y)
        dfdy_fourier = derivative_y(field_fourier, grid.ky)
        dfdy_real = rfft2_inverse(dfdy_fourier, 128, 128)
        dfdy_exact = -ky * jnp.sin(kx * X) * jnp.sin(ky * Y)
        assert jnp.allclose(dfdy_real, dfdy_exact, atol=1e-5)

    def test_laplacian(self):
        """Test Laplacian: ∇²f = ∂²f/∂x² + ∂²f/∂y² = -(kx² + ky²)·f̂(k)."""
        grid = SpectralGrid2D.create(Nx=128, Ny=128, Lx=2 * jnp.pi, Ly=2 * jnp.pi)

        # Create coordinate arrays
        x = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        X, Y = X.T, Y.T

        kx, ky = 4, 3
        # f(x,y) = sin(kx*x)·sin(ky*y)
        field_real = jnp.sin(kx * X) * jnp.sin(ky * Y)
        field_fourier = rfft2_forward(field_real)

        # Compute ∂²f/∂x²
        d2fdx2_fourier = derivative_x(derivative_x(field_fourier, grid.kx), grid.kx)
        d2fdx2_real = rfft2_inverse(d2fdx2_fourier, 128, 128)

        # Compute ∂²f/∂y²
        d2fdy2_fourier = derivative_y(derivative_y(field_fourier, grid.ky), grid.ky)
        d2fdy2_real = rfft2_inverse(d2fdy2_fourier, 128, 128)

        # Laplacian
        laplacian_real = d2fdx2_real + d2fdy2_real

        # Analytical: ∇²f = -(kx² + ky²)·sin(kx*x)·sin(ky*y)
        laplacian_exact = -(kx**2 + ky**2) * field_real

        # Float32 precision with accumulated error from two sequential derivative operations
        # Looser tolerance (rtol=1e-4, atol=1e-2) accounts for error propagation:
        # ∇² = ∂²/∂x² + ∂²/∂y² involves 4 FFT operations total (2 forward, 2 inverse)
        assert jnp.allclose(laplacian_real, laplacian_exact, rtol=1e-4, atol=1e-2)

    def test_laplacian_helper(self):
        """Test dedicated laplacian() function (more efficient than sequential derivatives)."""
        grid = SpectralGrid2D.create(Nx=128, Ny=128, Lx=2 * jnp.pi, Ly=2 * jnp.pi)

        # Create coordinate arrays
        x = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        X, Y = X.T, Y.T

        kx, ky = 4, 3
        # f(x,y) = sin(kx*x)·sin(ky*y)
        field_real = jnp.sin(kx * X) * jnp.sin(ky * Y)
        field_fourier = rfft2_forward(field_real)

        # Compute Laplacian using helper function
        lap_fourier = laplacian(field_fourier, grid.kx, grid.ky)
        lap_real = rfft2_inverse(lap_fourier, 128, 128)

        # Analytical: ∇²f = -(kx² + ky²)·sin(kx*x)·sin(ky*y)
        lap_exact = -(kx**2 + ky**2) * field_real

        # Should match with similar tolerance to sequential method
        assert jnp.allclose(lap_real, lap_exact, rtol=1e-4, atol=1e-2)

    def test_derivatives_non_square_grid(self):
        """Test derivatives on non-square grid (Nx != Ny) to verify broadcasting."""
        grid = SpectralGrid2D.create(Nx=256, Ny=128, Lx=2 * jnp.pi, Ly=2 * jnp.pi)

        # Create coordinate arrays (non-square)
        x = jnp.linspace(0, 2 * jnp.pi, 256, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        X, Y = X.T, Y.T  # Shape: [Ny=128, Nx=256]

        # Test ∂x on non-square grid
        n = 3
        field_real = jnp.sin(n * X)
        field_fourier = rfft2_forward(field_real)

        dfdx_fourier = derivative_x(field_fourier, grid.kx)
        dfdx_real = rfft2_inverse(dfdx_fourier, 128, 256)
        dfdx_exact = n * jnp.cos(n * X)

        # Slightly looser tolerance for non-square grids (different aspect ratio)
        assert jnp.allclose(dfdx_real, dfdx_exact, atol=1e-4)

        # Test ∂y on non-square grid
        field_real = jnp.sin(n * Y)
        field_fourier = rfft2_forward(field_real)

        dfdy_fourier = derivative_y(field_fourier, grid.ky)
        dfdy_real = rfft2_inverse(dfdy_fourier, 128, 256)
        dfdy_exact = n * jnp.cos(n * Y)

        assert jnp.allclose(dfdy_real, dfdy_exact, atol=1e-4)


class TestDealiasing:
    """Test suite for 2/3 rule dealiasing."""

    def test_dealias_zeros_high_k(self):
        """Test that dealiasing zeros high-wavenumber modes."""
        grid = SpectralGrid2D.create(Nx=64, Ny=64)

        # Create field with all modes set to 1
        field_fourier = jnp.ones((64, 64 // 2 + 1), dtype=jnp.complex64)

        # Apply dealiasing
        field_dealiased = dealias(field_fourier, grid.dealias_mask)

        # High-k modes should be zero
        assert field_dealiased[0, -1] == 0.0  # kx = Nyquist
        assert field_dealiased[32, 0] == 0.0  # ky = Nyquist (index Ny//2)

        # Low-k modes should be unchanged
        assert field_dealiased[0, 0] == 1.0  # k=0 mode
        assert field_dealiased[1, 1] == 1.0  # Low-k mode

    def test_dealias_preserves_low_k(self):
        """Test that dealiasing preserves low-wavenumber physics."""
        grid = SpectralGrid2D.create(Nx=128, Ny=128, Lx=2 * jnp.pi, Ly=2 * jnp.pi)

        # Create low-k field: f = sin(2x) + cos(3y)
        x = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        X, Y = X.T, Y.T

        field_real = jnp.sin(2 * X) + jnp.cos(3 * Y)
        field_fourier = rfft2_forward(field_real)

        # Dealias
        field_dealiased = dealias(field_fourier, grid.dealias_mask)
        field_back = rfft2_inverse(field_dealiased, 128, 128)

        # Low-k field should be nearly unchanged
        assert jnp.allclose(field_real, field_back, atol=1e-6)


class TestSpectralField2D:
    """Test suite for SpectralField2D lazy evaluation."""

    def test_from_real(self):
        """Test creating SpectralField2D from real-space data."""
        grid = SpectralGrid2D.create(Nx=64, Ny=64)
        field_real = jnp.ones((64, 64))

        field = SpectralField2D.from_real(field_real, grid)

        # Real should be cached
        assert jnp.allclose(field.real, field_real)

        # Fourier should be computed lazily
        field_fourier = field.fourier
        assert field_fourier.shape == (64, 64 // 2 + 1)

    def test_from_fourier(self):
        """Test creating SpectralField2D from Fourier-space data."""
        grid = SpectralGrid2D.create(Nx=64, Ny=64)
        field_fourier = jnp.ones((64, 64 // 2 + 1), dtype=jnp.complex64)

        field = SpectralField2D.from_fourier(field_fourier, grid)

        # Fourier should be cached
        assert jnp.allclose(field.fourier, field_fourier)

        # Real should be computed lazily
        field_real = field.real
        assert field_real.shape == (64, 64)

    def test_lazy_evaluation(self):
        """Test that transformations are only done once (cached)."""
        grid = SpectralGrid2D.create(Nx=64, Ny=64)
        field_real = jnp.ones((64, 64))

        field = SpectralField2D.from_real(field_real, grid)

        # First access computes Fourier transform
        fourier1 = field.fourier

        # Second access should return cached value (same object)
        fourier2 = field.fourier
        assert fourier1 is fourier2

    def test_roundtrip_consistency(self):
        """Test that real -> Fourier -> real preserves data."""
        grid = SpectralGrid2D.create(Nx=64, Ny=64)

        # Create field from real data
        x = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        field_real_original = jnp.sin(3 * X.T) + jnp.cos(2 * Y.T)

        field = SpectralField2D.from_real(field_real_original, grid)

        # Access Fourier, then real again
        _ = field.fourier
        field_real_recovered = field.real

        assert jnp.allclose(field_real_original, field_real_recovered, atol=1e-12)

    def test_shape_validation_real(self):
        """Test that from_real validates field shape matches grid."""
        grid = SpectralGrid2D.create(Nx=64, Ny=64)

        # Wrong shape should raise ValueError
        wrong_shape_field = jnp.ones((32, 64))  # Should be (64, 64)

        with pytest.raises(ValueError, match="does not match grid shape"):
            SpectralField2D.from_real(wrong_shape_field, grid)

    def test_shape_validation_fourier(self):
        """Test that from_fourier validates field shape matches grid."""
        grid = SpectralGrid2D.create(Nx=64, Ny=64)

        # Wrong shape should raise ValueError
        wrong_shape_field = jnp.ones(
            (64, 32), dtype=jnp.complex64
        )  # Should be (64, 33)

        with pytest.raises(ValueError, match="does not match expected Fourier shape"):
            SpectralField2D.from_fourier(wrong_shape_field, grid)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
