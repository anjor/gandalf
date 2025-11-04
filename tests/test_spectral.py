"""
Tests for core spectral infrastructure.

Validates:
- FFT forward/inverse operations
- Spectral derivatives (∂x, ∂y)
- Dealiasing (2/3 rule)
- Pydantic model validation
- Laplacian operator composition
- Poisson solver (∇²φ = ω)
"""

import pytest
import jax
import jax.numpy as jnp
from pydantic import ValidationError

from krmhd.spectral import (
    SpectralGrid2D,
    SpectralGrid3D,
    SpectralField2D,
    SpectralField3D,
    rfft2_forward,
    rfft2_inverse,
    rfftn_forward,
    rfftn_inverse,
    derivative_x,
    derivative_y,
    derivative_z,
    laplacian,
    dealias,
    poisson_solve_2d,
    poisson_solve_3d,
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


class TestSpectralGrid3D:
    """Test suite for SpectralGrid3D Pydantic model."""

    def test_create_basic_grid(self):
        """Test basic 3D grid creation with default parameters."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)

        assert grid.Nx == 64
        assert grid.Ny == 64
        assert grid.Nz == 64
        assert jnp.isclose(grid.Lx, 2 * jnp.pi)
        assert jnp.isclose(grid.Ly, 2 * jnp.pi)
        assert jnp.isclose(grid.Lz, 2 * jnp.pi)

        # Check wavenumber array shapes
        assert grid.kx.shape == (64 // 2 + 1,)  # rfftn convention
        assert grid.ky.shape == (64,)
        assert grid.kz.shape == (64,)
        assert grid.dealias_mask.shape == (64, 64, 64 // 2 + 1)

    def test_create_custom_domain(self):
        """Test 3D grid creation with custom domain sizes."""
        grid = SpectralGrid3D.create(
            Nx=128, Ny=64, Nz=32, Lx=4 * jnp.pi, Ly=2 * jnp.pi, Lz=jnp.pi
        )

        assert grid.Nx == 128
        assert grid.Ny == 64
        assert grid.Nz == 32
        assert jnp.isclose(grid.Lx, 4 * jnp.pi)
        assert jnp.isclose(grid.Ly, 2 * jnp.pi)
        assert jnp.isclose(grid.Lz, jnp.pi)

    def test_grid_immutability(self):
        """Test that SpectralGrid3D is immutable (frozen=True)."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)

        with pytest.raises(ValidationError):
            grid.Nx = 128

    def test_validation_positive_dimensions(self):
        """Test that factory method validates positive dimensions."""
        with pytest.raises(ValueError, match="must be positive"):
            SpectralGrid3D.create(Nx=-64, Ny=64, Nz=64)

        with pytest.raises(ValueError, match="must be positive"):
            SpectralGrid3D.create(Nx=64, Ny=64, Nz=0)

    def test_validation_even_dimensions(self):
        """Test that grid dimensions must be even."""
        with pytest.raises(ValueError, match="must be even"):
            SpectralGrid3D.create(Nx=63, Ny=64, Nz=64)

        with pytest.raises(ValueError, match="must be even"):
            SpectralGrid3D.create(Nx=64, Ny=64, Nz=65)

    def test_wavenumber_arrays(self):
        """Test that wavenumber arrays follow correct FFT conventions."""
        grid = SpectralGrid3D.create(
            Nx=64, Ny=64, Nz=64, Lx=2 * jnp.pi, Ly=2 * jnp.pi, Lz=2 * jnp.pi
        )

        # kx should be non-negative (rfftn convention)
        assert jnp.all(grid.kx >= 0)
        assert grid.kx[0] == 0.0
        assert grid.kx[-1] == 32  # Nyquist: Nx//2

        # ky and kz should include negative frequencies (standard FFT ordering)
        assert grid.ky[0] == 0.0
        assert grid.ky[grid.Ny // 2] == -32  # Nyquist wraps to negative
        assert grid.kz[0] == 0.0
        assert grid.kz[grid.Nz // 2] == -32  # Nyquist wraps to negative

    def test_dealias_mask_3d(self):
        """Test that 3D dealiasing mask correctly implements 2/3 rule."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)

        # Low-k modes should be kept
        assert grid.dealias_mask[0, 0, 0]  # k=0 mode
        assert grid.dealias_mask[1, 1, 1]  # Low-k mode

        # High-k modes should be zeroed
        # Max kx, ky, kz is 32, so 2/3 * 32 ≈ 21.3
        assert not grid.dealias_mask[0, 0, -1]  # kx = Nyquist
        assert not grid.dealias_mask[0, 32, 0]  # ky = Nyquist
        assert not grid.dealias_mask[32, 0, 0]  # kz = Nyquist


class TestFFTOperations3D:
    """Test suite for 3D FFT forward/inverse operations."""

    def test_fft_roundtrip_3d(self):
        """Test that 3D FFT -> IFFT recovers original field."""
        # Create random real field
        key = jax.random.PRNGKey(0)
        field_real = jax.random.normal(key, (64, 64, 64))

        # Forward and inverse transform
        field_fourier = rfftn_forward(field_real)
        field_recovered = rfftn_inverse(field_fourier, 64, 64, 64)

        # Should match to float32 precision (~1e-6)
        assert jnp.allclose(field_real, field_recovered, atol=1e-6)

    def test_fft_shapes_3d(self):
        """Test that 3D FFT operations produce correct output shapes."""
        Nx, Ny, Nz = 128, 64, 32
        field_real = jnp.zeros((Nz, Ny, Nx))

        field_fourier = rfftn_forward(field_real)
        assert field_fourier.shape == (Nz, Ny, Nx // 2 + 1)

        field_back = rfftn_inverse(field_fourier, Nz, Ny, Nx)
        assert field_back.shape == (Nz, Ny, Nx)

    def test_reality_condition_3d(self):
        """Test that rfftn preserves reality condition."""
        # Create real field
        x = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        z = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        # Transpose to [Nz, Ny, Nx]
        X, Y, Z = X.transpose(2, 1, 0), Y.transpose(2, 1, 0), Z.transpose(2, 1, 0)
        field_real = jnp.sin(3 * X) + jnp.cos(2 * Y) + jnp.sin(Z)

        field_fourier = rfftn_forward(field_real)
        field_back = rfftn_inverse(field_fourier, 64, 64, 64)

        # Inverse should be purely real
        assert jnp.allclose(field_back.imag, 0.0, atol=1e-12)


class TestDerivatives3D:
    """Test suite for 3D spectral derivative operators."""

    def test_derivative_z_sine(self):
        """Test ∂z(sin(kz)) = k·cos(kz)."""
        grid = SpectralGrid3D.create(
            Nx=128, Ny=128, Nz=128, Lx=2 * jnp.pi, Ly=2 * jnp.pi, Lz=2 * jnp.pi
        )

        # Create coordinate arrays
        x = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        z = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        Z = Z.transpose(2, 1, 0)  # Shape [Nz, Ny, Nx]

        # Test multiple wavenumbers
        for n in [1, 2, 3, 5]:
            # f(z) = sin(n*z)
            field_real = jnp.sin(n * Z)
            field_fourier = rfftn_forward(field_real)

            # Compute spectral derivative
            dfdz_fourier = derivative_z(field_fourier, grid.kz)
            dfdz_real = rfftn_inverse(dfdz_fourier, 128, 128, 128)

            # Analytical derivative: df/dz = n·cos(n*z)
            dfdz_exact = n * jnp.cos(n * Z)

            # Should match to float32 precision
            assert jnp.allclose(dfdz_real, dfdz_exact, atol=1e-5)

    def test_derivative_3d_function(self):
        """Test all three derivatives on 3D function."""
        grid = SpectralGrid3D.create(
            Nx=128, Ny=128, Nz=128, Lx=2 * jnp.pi, Ly=2 * jnp.pi, Lz=2 * jnp.pi
        )

        # Create coordinate arrays
        x = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        z = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        X, Y, Z = X.transpose(2, 1, 0), Y.transpose(2, 1, 0), Z.transpose(2, 1, 0)

        kx, ky, kz = 3, 2, 4
        field_real = jnp.sin(kx * X) * jnp.cos(ky * Y) * jnp.sin(kz * Z)
        field_fourier = rfftn_forward(field_real)

        # Test ∂x
        dfdx_fourier = derivative_x(field_fourier, grid.kx)
        dfdx_real = rfftn_inverse(dfdx_fourier, 128, 128, 128)
        dfdx_exact = kx * jnp.cos(kx * X) * jnp.cos(ky * Y) * jnp.sin(kz * Z)
        # Looser tolerance for 3D: 128³ grid → more accumulated rounding error
        # from 6 FFT operations (3 forward + 3 inverse) compared to 2D (4 ops)
        assert jnp.allclose(dfdx_real, dfdx_exact, rtol=1e-4, atol=1e-4)

        # Test ∂y
        dfdy_fourier = derivative_y(field_fourier, grid.ky)
        dfdy_real = rfftn_inverse(dfdy_fourier, 128, 128, 128)
        dfdy_exact = -ky * jnp.sin(kx * X) * jnp.sin(ky * Y) * jnp.sin(kz * Z)
        assert jnp.allclose(dfdy_real, dfdy_exact, rtol=1e-4, atol=1e-4)

        # Test ∂z
        dfdz_fourier = derivative_z(field_fourier, grid.kz)
        dfdz_real = rfftn_inverse(dfdz_fourier, 128, 128, 128)
        dfdz_exact = kz * jnp.sin(kx * X) * jnp.cos(ky * Y) * jnp.cos(kz * Z)
        assert jnp.allclose(dfdz_real, dfdz_exact, rtol=1e-4, atol=1e-4)

    def test_laplacian_3d_full(self):
        """Test full 3D Laplacian with all three directions."""
        grid = SpectralGrid3D.create(
            Nx=128, Ny=128, Nz=128, Lx=2 * jnp.pi, Ly=2 * jnp.pi, Lz=2 * jnp.pi
        )

        # Create coordinate arrays
        x = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        z = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        X, Y, Z = X.transpose(2, 1, 0), Y.transpose(2, 1, 0), Z.transpose(2, 1, 0)

        kx, ky, kz = 4, 3, 2
        # f(x,y,z) = sin(kx*x)·sin(ky*y)·sin(kz*z)
        field_real = jnp.sin(kx * X) * jnp.sin(ky * Y) * jnp.sin(kz * Z)
        field_fourier = rfftn_forward(field_real)

        # Compute 3D Laplacian
        lap_fourier = laplacian(field_fourier, grid.kx, grid.ky, grid.kz)
        lap_real = rfftn_inverse(lap_fourier, 128, 128, 128)

        # Analytical: ∇²f = -(kx² + ky² + kz²)·f
        lap_exact = -(kx**2 + ky**2 + kz**2) * field_real

        # Looser tolerance for 3D Laplacian: accumulated error from multiple FFT operations
        # rtol=1e-4 accounts for float32 precision on 128³ grid
        assert jnp.allclose(lap_real, lap_exact, rtol=1e-4, atol=1e-2)

    def test_laplacian_3d_perpendicular_only(self):
        """Test perpendicular Laplacian (kz=None) on 3D field."""
        grid = SpectralGrid3D.create(
            Nx=128, Ny=128, Nz=128, Lx=2 * jnp.pi, Ly=2 * jnp.pi, Lz=2 * jnp.pi
        )

        # Create coordinate arrays
        x = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        z = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        X, Y, Z = X.transpose(2, 1, 0), Y.transpose(2, 1, 0), Z.transpose(2, 1, 0)

        kx, ky, kz = 4, 3, 2
        field_real = jnp.sin(kx * X) * jnp.sin(ky * Y) * jnp.sin(kz * Z)
        field_fourier = rfftn_forward(field_real)

        # Compute perpendicular Laplacian only (kz=None)
        lap_perp_fourier = laplacian(field_fourier, grid.kx, grid.ky, kz=None)
        lap_perp_real = rfftn_inverse(lap_perp_fourier, 128, 128, 128)

        # Analytical: ∇²⊥f = -(kx² + ky²)·f (no kz contribution)
        lap_perp_exact = -(kx**2 + ky**2) * field_real

        # Perpendicular Laplacian should have similar accuracy to full 3D
        assert jnp.allclose(lap_perp_real, lap_perp_exact, rtol=1e-4, atol=1e-2)


class TestNegativeCases:
    """Test suite for error handling and edge cases."""

    def test_derivative_z_on_2d_field_raises(self):
        """Test that derivative_z fails gracefully on 2D fields."""
        grid2d = SpectralGrid2D.create(64, 64)
        field_2d = jnp.ones((64, 33), dtype=jnp.complex64)

        with pytest.raises(ValueError, match="Unsupported field dimensionality: 2"):
            derivative_z(field_2d, jnp.ones(64))

    def test_derivative_x_invalid_ndim(self):
        """Test that derivative_x fails on 1D or 4D fields."""
        # 1D field should fail
        field_1d = jnp.ones(64, dtype=jnp.complex64)
        with pytest.raises(ValueError, match="Unsupported field dimensionality"):
            derivative_x(field_1d, jnp.ones(33))

        # 4D field should fail
        field_4d = jnp.ones((32, 32, 32, 17), dtype=jnp.complex64)
        with pytest.raises(ValueError, match="Unsupported field dimensionality"):
            derivative_x(field_4d, jnp.ones(33))

    def test_laplacian_3d_without_kz(self):
        """Test that perpendicular Laplacian works correctly (kz=None on 3D field)."""
        grid = SpectralGrid3D.create(64, 64, 64)
        field_fourier = jnp.ones((64, 64, 33), dtype=jnp.complex64)

        # Should work without error (perpendicular Laplacian only)
        lap_perp = laplacian(field_fourier, grid.kx, grid.ky, kz=None)
        assert lap_perp.shape == field_fourier.shape


class TestSpectralField3D:
    """Test suite for SpectralField3D lazy evaluation."""

    def test_from_real_3d(self):
        """Test creating SpectralField3D from real-space data."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)
        field_real = jnp.ones((64, 64, 64))

        field = SpectralField3D.from_real(field_real, grid)

        # Real should be cached
        assert jnp.allclose(field.real, field_real)

        # Fourier should be computed lazily
        field_fourier = field.fourier
        assert field_fourier.shape == (64, 64, 64 // 2 + 1)

    def test_from_fourier_3d(self):
        """Test creating SpectralField3D from Fourier-space data."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)
        field_fourier = jnp.ones((64, 64, 64 // 2 + 1), dtype=jnp.complex64)

        field = SpectralField3D.from_fourier(field_fourier, grid)

        # Fourier should be cached
        assert jnp.allclose(field.fourier, field_fourier)

        # Real should be computed lazily
        field_real = field.real
        assert field_real.shape == (64, 64, 64)

    def test_roundtrip_consistency_3d(self):
        """Test that real -> Fourier -> real preserves data (3D)."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)

        # Create field from real data
        x = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        z = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        X, Y, Z = X.transpose(2, 1, 0), Y.transpose(2, 1, 0), Z.transpose(2, 1, 0)
        field_real_original = jnp.sin(3 * X) + jnp.cos(2 * Y) + jnp.sin(Z)

        field = SpectralField3D.from_real(field_real_original, grid)

        # Access Fourier, then real again
        _ = field.fourier
        field_real_recovered = field.real

        assert jnp.allclose(field_real_original, field_real_recovered, atol=1e-12)


class TestPoissonSolver2D:
    """Test suite for 2D Poisson solver."""

    def test_manufactured_solution_sin_cos(self):
        """Test 2D Poisson solver with manufactured solution φ = sin(x)·cos(y)."""
        grid = SpectralGrid2D.create(Nx=128, Ny=128, Lx=2 * jnp.pi, Ly=2 * jnp.pi)

        # Create coordinate arrays
        x = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        X, Y = X.T, Y.T  # Match [Ny, Nx] convention

        # Manufactured solution: φ = sin(x)·cos(y)
        phi_exact = jnp.sin(X) * jnp.cos(Y)

        # Compute exact vorticity: ω = ∇²φ = -2·sin(x)·cos(y)
        omega_exact = -2.0 * jnp.sin(X) * jnp.cos(Y)

        # Transform to Fourier space
        omega_fourier = rfft2_forward(omega_exact)

        # Solve Poisson equation
        phi_fourier = poisson_solve_2d(omega_fourier, grid.kx, grid.ky)

        # Transform back to real space
        phi_recovered = rfft2_inverse(phi_fourier, grid.Ny, grid.Nx)

        # Remove mean from both (k=0 mode is arbitrary)
        phi_exact_zero_mean = phi_exact - jnp.mean(phi_exact)
        phi_recovered_zero_mean = phi_recovered - jnp.mean(phi_recovered)

        # Check that recovered solution matches exact solution (up to constant)
        # Note: Using atol=1e-6 for float32 precision
        assert jnp.allclose(phi_recovered_zero_mean, phi_exact_zero_mean, atol=1e-6)

    def test_manufactured_solution_complex_trig(self):
        """Test 2D Poisson solver with more complex trigonometric solution."""
        grid = SpectralGrid2D.create(Nx=128, Ny=128, Lx=2 * jnp.pi, Ly=2 * jnp.pi)

        x = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        X, Y = X.T, Y.T

        # More complex solution: φ = sin(3x)·cos(2y) + cos(x)·sin(4y)
        phi_exact = jnp.sin(3 * X) * jnp.cos(2 * Y) + jnp.cos(X) * jnp.sin(4 * Y)

        # Compute Laplacian in Fourier space
        phi_fourier_exact = rfft2_forward(phi_exact)
        omega_fourier = laplacian(phi_fourier_exact, grid.kx, grid.ky)

        # Solve Poisson equation
        phi_fourier_recovered = poisson_solve_2d(omega_fourier, grid.kx, grid.ky)

        # Transform back
        phi_recovered = rfft2_inverse(phi_fourier_recovered, grid.Ny, grid.Nx)

        # Remove means
        phi_exact_zero_mean = phi_exact - jnp.mean(phi_exact)
        phi_recovered_zero_mean = phi_recovered - jnp.mean(phi_recovered)

        # Note: Using atol=1e-6 for float32 precision
        assert jnp.allclose(phi_recovered_zero_mean, phi_exact_zero_mean, atol=1e-6)

    def test_k0_mode_handling(self):
        """Test that k=0 mode is properly set to zero."""
        grid = SpectralGrid2D.create(Nx=64, Ny=64)

        # Create omega with non-zero mean (k=0 mode)
        omega_fourier = jnp.ones((64, 64 // 2 + 1), dtype=jnp.complex64)
        omega_fourier = omega_fourier.at[0, 0].set(100.0 + 0.0j)  # Large k=0 mode

        # Solve Poisson equation
        phi_fourier = poisson_solve_2d(omega_fourier, grid.kx, grid.ky)

        # k=0 mode should be zero
        assert jnp.isclose(phi_fourier[0, 0], 0.0 + 0.0j, atol=1e-12)

    def test_output_shape_2d(self):
        """Test that output shape matches input shape."""
        grid = SpectralGrid2D.create(Nx=128, Ny=256)
        omega_fourier = jnp.ones((256, 128 // 2 + 1), dtype=jnp.complex64)

        phi_fourier = poisson_solve_2d(omega_fourier, grid.kx, grid.ky)

        assert phi_fourier.shape == omega_fourier.shape

    def test_laplacian_roundtrip_2d(self):
        """Test that Laplacian -> Poisson solver gives identity (up to k=0)."""
        grid = SpectralGrid2D.create(Nx=128, Ny=128)

        # Create arbitrary field (with zero mean)
        x = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 128, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        X, Y = X.T, Y.T
        phi_original = jnp.sin(2 * X) * jnp.cos(3 * Y)

        # Laplacian
        phi_fourier = rfft2_forward(phi_original)
        omega_fourier = laplacian(phi_fourier, grid.kx, grid.ky)

        # Poisson solve
        phi_fourier_recovered = poisson_solve_2d(omega_fourier, grid.kx, grid.ky)
        phi_recovered = rfft2_inverse(phi_fourier_recovered, grid.Ny, grid.Nx)

        # Remove means
        phi_original_zero_mean = phi_original - jnp.mean(phi_original)
        phi_recovered_zero_mean = phi_recovered - jnp.mean(phi_recovered)

        # Note: Using atol=1e-6 for float32 precision
        assert jnp.allclose(phi_original_zero_mean, phi_recovered_zero_mean, atol=1e-6)

    def test_invalid_dimensionality(self):
        """Test that 3D field raises error."""
        grid = SpectralGrid2D.create(Nx=64, Ny=64)

        # Try to pass 3D field to 2D solver
        omega_fourier_3d = jnp.ones((64, 64, 64 // 2 + 1), dtype=jnp.complex64)

        with pytest.raises(ValueError, match="Expected 2D field"):
            poisson_solve_2d(omega_fourier_3d, grid.kx, grid.ky)


class TestPoissonSolver3D:
    """Test suite for 3D Poisson solver."""

    def test_manufactured_solution_perpendicular_3d(self):
        """Test 3D perpendicular Poisson solver (kz=None) with manufactured solution."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)

        # Create coordinate arrays
        x = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        z = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        X, Y, Z = X.transpose(2, 1, 0), Y.transpose(2, 1, 0), Z.transpose(2, 1, 0)

        # Solution with z-dependence: φ = sin(x)·cos(y)·cos(2z)
        phi_exact = jnp.sin(X) * jnp.cos(Y) * jnp.cos(2 * Z)

        # Perpendicular Laplacian: ∇²⊥φ = -2·sin(x)·cos(y)·cos(2z)
        omega_exact = -2.0 * jnp.sin(X) * jnp.cos(Y) * jnp.cos(2 * Z)

        # Transform to Fourier space
        omega_fourier = rfftn_forward(omega_exact)

        # Solve perpendicular Poisson equation (kz=None)
        phi_fourier = poisson_solve_3d(omega_fourier, grid.kx, grid.ky, kz=None)

        # Transform back
        phi_recovered = rfftn_inverse(phi_fourier, grid.Nz, grid.Ny, grid.Nx)

        # Remove means
        phi_exact_zero_mean = phi_exact - jnp.mean(phi_exact)
        phi_recovered_zero_mean = phi_recovered - jnp.mean(phi_recovered)

        # Note: Using atol=1e-6 for float32 precision
        assert jnp.allclose(phi_recovered_zero_mean, phi_exact_zero_mean, atol=1e-6)

    def test_manufactured_solution_full_3d(self):
        """Test 3D full Poisson solver with kz included."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)

        x = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        z = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        X, Y, Z = X.transpose(2, 1, 0), Y.transpose(2, 1, 0), Z.transpose(2, 1, 0)

        # Solution: φ = sin(x)·cos(y)·sin(z)
        phi_exact = jnp.sin(X) * jnp.cos(Y) * jnp.sin(Z)

        # Full 3D Laplacian: ∇²φ = -3·sin(x)·cos(y)·sin(z)
        omega_exact = -3.0 * jnp.sin(X) * jnp.cos(Y) * jnp.sin(Z)

        # Transform to Fourier space
        omega_fourier = rfftn_forward(omega_exact)

        # Solve full 3D Poisson equation
        phi_fourier = poisson_solve_3d(omega_fourier, grid.kx, grid.ky, kz=grid.kz)

        # Transform back
        phi_recovered = rfftn_inverse(phi_fourier, grid.Nz, grid.Ny, grid.Nx)

        # Remove means
        phi_exact_zero_mean = phi_exact - jnp.mean(phi_exact)
        phi_recovered_zero_mean = phi_recovered - jnp.mean(phi_recovered)

        # Note: Using atol=1e-6 for float32 precision
        assert jnp.allclose(phi_recovered_zero_mean, phi_exact_zero_mean, atol=1e-6)

    def test_k0_mode_handling_3d(self):
        """Test that k=0 mode is properly handled in 3D."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)

        omega_fourier = jnp.ones((64, 64, 64 // 2 + 1), dtype=jnp.complex64)
        omega_fourier = omega_fourier.at[0, 0, 0].set(1000.0 + 0.0j)

        # Solve with perpendicular Laplacian
        phi_fourier = poisson_solve_3d(omega_fourier, grid.kx, grid.ky, kz=None)

        # k=0 mode should be zero
        assert jnp.isclose(phi_fourier[0, 0, 0], 0.0 + 0.0j, atol=1e-12)

    def test_output_shape_3d(self):
        """Test that output shape matches input shape."""
        grid = SpectralGrid3D.create(Nx=128, Ny=64, Nz=32)
        omega_fourier = jnp.ones((32, 64, 128 // 2 + 1), dtype=jnp.complex64)

        phi_fourier = poisson_solve_3d(omega_fourier, grid.kx, grid.ky, kz=None)

        assert phi_fourier.shape == omega_fourier.shape

    def test_perpendicular_vs_full_laplacian(self):
        """Test difference between perpendicular and full 3D Laplacian solves."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)

        x = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        z = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        X, Y, Z = X.transpose(2, 1, 0), Y.transpose(2, 1, 0), Z.transpose(2, 1, 0)

        # Create field with z-dependence
        phi = jnp.sin(X) * jnp.cos(Y) * jnp.sin(2 * Z)
        phi_fourier = rfftn_forward(phi)

        # Perpendicular Laplacian
        omega_perp_fourier = laplacian(phi_fourier, grid.kx, grid.ky, kz=None)

        # Full 3D Laplacian
        omega_full_fourier = laplacian(phi_fourier, grid.kx, grid.ky, kz=grid.kz)

        # These should be different
        omega_perp = rfftn_inverse(omega_perp_fourier, grid.Nz, grid.Ny, grid.Nx)
        omega_full = rfftn_inverse(omega_full_fourier, grid.Nz, grid.Ny, grid.Nx)

        # They should differ by the z-derivative term
        assert not jnp.allclose(omega_perp, omega_full)

    def test_laplacian_roundtrip_3d(self):
        """Test that Laplacian -> Poisson solver gives identity (up to k=0)."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)

        x = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        z = jnp.linspace(0, 2 * jnp.pi, 64, endpoint=False)
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        X, Y, Z = X.transpose(2, 1, 0), Y.transpose(2, 1, 0), Z.transpose(2, 1, 0)

        phi_original = jnp.sin(2 * X) * jnp.cos(3 * Y) * jnp.sin(Z)

        # Perpendicular Laplacian
        phi_fourier = rfftn_forward(phi_original)
        omega_fourier = laplacian(phi_fourier, grid.kx, grid.ky, kz=None)

        # Poisson solve
        phi_fourier_recovered = poisson_solve_3d(omega_fourier, grid.kx, grid.ky, kz=None)
        phi_recovered = rfftn_inverse(phi_fourier_recovered, grid.Nz, grid.Ny, grid.Nx)

        # Remove means
        phi_original_zero_mean = phi_original - jnp.mean(phi_original)
        phi_recovered_zero_mean = phi_recovered - jnp.mean(phi_recovered)

        # Note: Using atol=1e-6 for float32 precision
        assert jnp.allclose(phi_original_zero_mean, phi_recovered_zero_mean, atol=1e-6)

    def test_invalid_dimensionality(self):
        """Test that 2D field raises error."""
        grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)

        # Try to pass 2D field to 3D solver
        omega_fourier_2d = jnp.ones((64, 64 // 2 + 1), dtype=jnp.complex64)

        with pytest.raises(ValueError, match="Expected 3D field"):
            poisson_solve_3d(omega_fourier_2d, grid.kx, grid.ky)


class TestPytreeRegistration:
    """Test suite for JAX pytree registration of spectral grids."""

    def test_spectral_grid_2d_tree_flatten_unflatten(self):
        """Test that SpectralGrid2D can be flattened and unflattened."""
        grid = SpectralGrid2D.create(Nx=64, Ny=64, Lx=1.0, Ly=2.0)

        # Flatten using our custom flatten function directly
        from krmhd.spectral import _spectral_grid_2d_flatten, _spectral_grid_2d_unflatten

        children, aux_data = _spectral_grid_2d_flatten(grid)

        # Check children are arrays
        assert len(children) == 3  # kx, ky, dealias_mask
        assert all(isinstance(c, jax.Array) for c in children)

        # Check aux_data contains static fields
        Nx, Ny, Lx, Ly = aux_data
        assert Nx == 64
        assert Ny == 64
        assert Lx == 1.0
        assert Ly == 2.0

        # Unflatten and verify roundtrip
        grid_reconstructed = _spectral_grid_2d_unflatten(aux_data, children)
        assert grid_reconstructed.Nx == grid.Nx
        assert grid_reconstructed.Ny == grid.Ny
        assert jnp.allclose(grid_reconstructed.kx, grid.kx)
        assert jnp.allclose(grid_reconstructed.ky, grid.ky)
        assert jnp.allclose(grid_reconstructed.dealias_mask, grid.dealias_mask)

    def test_spectral_grid_3d_tree_flatten_unflatten(self):
        """Test that SpectralGrid3D can be flattened and unflattened."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=32, Lx=1.0, Ly=2.0, Lz=3.0)

        # Flatten using our custom flatten function directly
        from krmhd.spectral import _spectral_grid_3d_flatten, _spectral_grid_3d_unflatten

        children, aux_data = _spectral_grid_3d_flatten(grid)

        # Check children are arrays
        assert len(children) == 4  # kx, ky, kz, dealias_mask
        assert all(isinstance(c, jax.Array) for c in children)

        # Check aux_data contains static fields
        Nx, Ny, Nz, Lx, Ly, Lz = aux_data
        assert Nx == 32
        assert Ny == 32
        assert Nz == 32
        assert Lx == 1.0
        assert Ly == 2.0
        assert Lz == 3.0

        # Unflatten and verify roundtrip
        grid_reconstructed = _spectral_grid_3d_unflatten(aux_data, children)
        assert grid_reconstructed.Nx == grid.Nx
        assert grid_reconstructed.Ny == grid.Ny
        assert grid_reconstructed.Nz == grid.Nz
        assert jnp.allclose(grid_reconstructed.kx, grid.kx)
        assert jnp.allclose(grid_reconstructed.ky, grid.ky)
        assert jnp.allclose(grid_reconstructed.kz, grid.kz)
        assert jnp.allclose(grid_reconstructed.dealias_mask, grid.dealias_mask)

    def test_spectral_grid_2d_tree_map(self):
        """Test that jax.tree.map works on SpectralGrid2D."""
        grid = SpectralGrid2D.create(Nx=64, Ny=64)

        # Apply tree.map to scale all arrays by 2
        grid_scaled = jax.tree.map(lambda x: x * 2, grid)

        # Check that arrays were scaled
        assert jnp.allclose(grid_scaled.kx, grid.kx * 2)
        assert jnp.allclose(grid_scaled.ky, grid.ky * 2)
        assert jnp.allclose(grid_scaled.dealias_mask, grid.dealias_mask * 2)

        # Check that static fields are preserved
        assert grid_scaled.Nx == grid.Nx
        assert grid_scaled.Ny == grid.Ny
        assert grid_scaled.Lx == grid.Lx
        assert grid_scaled.Ly == grid.Ly

    def test_spectral_grid_3d_tree_map(self):
        """Test that jax.tree.map works on SpectralGrid3D."""
        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=32)

        # Apply tree.map to scale all arrays by 3
        grid_scaled = jax.tree.map(lambda x: x * 3, grid)

        # Check that arrays were scaled
        assert jnp.allclose(grid_scaled.kx, grid.kx * 3)
        assert jnp.allclose(grid_scaled.ky, grid.ky * 3)
        assert jnp.allclose(grid_scaled.kz, grid.kz * 3)
        assert jnp.allclose(grid_scaled.dealias_mask, grid.dealias_mask * 3)

        # Check that static fields are preserved
        assert grid_scaled.Nx == grid.Nx
        assert grid_scaled.Ny == grid.Ny
        assert grid_scaled.Nz == grid.Nz

    def test_spectral_grid_2d_jit_acceptance(self):
        """Test that SpectralGrid2D can be passed to JIT-compiled functions."""

        @jax.jit
        def compute_kx_sum(grid: SpectralGrid2D) -> float:
            return jnp.sum(grid.kx)

        grid = SpectralGrid2D.create(Nx=64, Ny=64)
        result = compute_kx_sum(grid)

        # Should not raise an error and should give expected result
        expected = jnp.sum(grid.kx)
        assert jnp.isclose(result, expected)

    def test_spectral_grid_3d_jit_acceptance(self):
        """Test that SpectralGrid3D can be passed to JIT-compiled functions."""

        @jax.jit
        def compute_grid_volume(grid: SpectralGrid3D) -> float:
            return grid.Lx * grid.Ly * grid.Lz

        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=32, Lx=1.0, Ly=2.0, Lz=3.0)
        result = compute_grid_volume(grid)

        # Should not raise an error
        expected = 1.0 * 2.0 * 3.0
        assert jnp.isclose(result, expected)

    def test_spectral_grid_3d_jit_with_arrays(self):
        """Test that SpectralGrid3D arrays can be accessed in JIT functions."""

        @jax.jit
        def compute_k_max(grid: SpectralGrid3D) -> float:
            kx_max = jnp.max(jnp.abs(grid.kx))
            ky_max = jnp.max(jnp.abs(grid.ky))
            kz_max = jnp.max(jnp.abs(grid.kz))
            return jnp.sqrt(kx_max**2 + ky_max**2 + kz_max**2)

        grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=32)
        result = compute_k_max(grid)

        # Should not raise an error
        assert result > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
