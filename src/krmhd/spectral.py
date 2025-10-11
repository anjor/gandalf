"""
Core spectral infrastructure for KRMHD solver.

This module provides 2D spectral operations using FFT methods:
- SpectralGrid2D: Pydantic model defining grid parameters and wavenumbers
- SpectralField2D: Manages real-space and Fourier-space representations
- FFT operations using rfft2 for real-to-complex transforms
- Spectral derivatives (∂x, ∂y) computed as multiplication in Fourier space
- Dealiasing using the 2/3 rule to prevent aliasing errors

All performance-critical functions are JIT-compiled with JAX.
"""

from typing import Optional
import jax
import jax.numpy as jnp
from jax import Array
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr, field_validator


class SpectralGrid2D(BaseModel):
    """
    Immutable 2D spectral grid specification with wavenumber arrays.

    Defines a rectangular grid in real space (Nx × Ny) with periodic boundary
    conditions. Wavenumber arrays are pre-computed for spectral derivatives,
    and a dealiasing mask implements the 2/3 rule for nonlinear operations.

    Attributes:
        Nx: Number of grid points in x direction (must be > 0)
        Ny: Number of grid points in y direction (must be > 0)
        Lx: Physical domain size in x direction (must be > 0)
        Ly: Physical domain size in y direction (must be > 0)
        kx: Wavenumber array in x (shape: [Nx//2+1], non-negative for rfft2)
        ky: Wavenumber array in y (shape: [Ny], includes negative frequencies)
        dealias_mask: Boolean mask for 2/3 rule dealiasing (shape: [Ny, Nx//2+1])

    Example:
        >>> grid = SpectralGrid2D.create(Nx=256, Ny=256, Lx=2*jnp.pi, Ly=2*jnp.pi)
        >>> grid.Nx
        256
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    Nx: int = Field(gt=0, description="Number of grid points in x")
    Ny: int = Field(gt=0, description="Number of grid points in y")
    Lx: float = Field(gt=0.0, description="Domain size in x")
    Ly: float = Field(gt=0.0, description="Domain size in y")
    kx: Array = Field(description="Wavenumber array in x (non-negative)")
    ky: Array = Field(description="Wavenumber array in y (includes negative)")
    dealias_mask: Array = Field(description="Boolean mask for 2/3 rule dealiasing")

    @field_validator("Nx", "Ny")
    @classmethod
    def validate_even_dimensions(cls, v: int) -> int:
        """Ensure grid dimensions are even for proper FFT symmetry."""
        if v % 2 != 0:
            raise ValueError(f"Grid dimension must be even, got {v}")
        return v

    @classmethod
    def create(
        cls,
        Nx: int,
        Ny: int,
        Lx: float = 2 * jnp.pi,
        Ly: float = 2 * jnp.pi,
    ) -> "SpectralGrid2D":
        """
        Factory method to create a SpectralGrid2D with computed wavenumbers.

        Wavenumber arrays follow the rfft2 convention:
        - kx: [0, 1, 2, ..., Nx//2] (non-negative only, exploits reality)
        - ky: [0, 1, ..., Ny//2-1, -Ny//2, ..., -1] (standard FFT ordering)

        The 2/3 dealiasing mask zeros modes where max(|kx|/kx_max, |ky|/ky_max) > 2/3
        to prevent aliasing errors in nonlinear terms.

        Args:
            Nx: Number of grid points in x (must be even and > 0)
            Ny: Number of grid points in y (must be even and > 0)
            Lx: Physical domain size in x (default: 2π)
            Ly: Physical domain size in y (default: 2π)

        Returns:
            SpectralGrid2D instance with pre-computed wavenumbers and mask

        Raises:
            ValueError: If Nx or Ny are not positive even integers, or if Lx, Ly ≤ 0
        """
        # Early validation before JAX operations
        if Nx <= 0 or Ny <= 0:
            raise ValueError(f"Grid dimensions must be positive, got Nx={Nx}, Ny={Ny}")
        if Nx % 2 != 0 or Ny % 2 != 0:
            raise ValueError(f"Grid dimensions must be even, got Nx={Nx}, Ny={Ny}")
        if Lx <= 0 or Ly <= 0:
            raise ValueError(f"Domain sizes must be positive, got Lx={Lx}, Ly={Ly}")

        # Wavenumber arrays (frequency space)
        kx = jnp.fft.rfftfreq(Nx, d=Lx / (2 * jnp.pi * Nx))
        ky = jnp.fft.fftfreq(Ny, d=Ly / (2 * jnp.pi * Ny))

        # 2/3 rule dealiasing: zero modes beyond |k| > (2/3) * k_max
        kx_max = jnp.max(jnp.abs(kx))
        ky_max = jnp.max(jnp.abs(ky))

        # Create 2D mesh for mask computation
        kx_2d, ky_2d = jnp.meshgrid(kx, ky, indexing="ij")
        kx_2d = kx_2d.T  # Transpose to [Ny, Nx//2+1]
        ky_2d = ky_2d.T

        # Mask: True where mode should be kept, False where it should be zeroed
        dealias_mask = (jnp.abs(kx_2d) <= 2 / 3 * kx_max) & (
            jnp.abs(ky_2d) <= 2 / 3 * ky_max
        )

        return cls(
            Nx=Nx,
            Ny=Ny,
            Lx=Lx,
            Ly=Ly,
            kx=kx,
            ky=ky,
            dealias_mask=dealias_mask,
        )


class SpectralField2D(BaseModel):
    """
    Manages a 2D field in both real and Fourier space with lazy evaluation.

    Stores a field in ONE of {real, Fourier} representation and transforms
    on-demand when the other is accessed. This avoids redundant FFTs.

    Attributes:
        grid: The spectral grid defining the field's domain
        _real: Private cache for real-space representation (Ny × Nx)
        _fourier: Private cache for Fourier-space representation (Ny × Nx//2+1)

    Example:
        >>> grid = SpectralGrid2D.create(64, 64)
        >>> field_real = jnp.sin(jnp.linspace(0, 2*jnp.pi, 64))
        >>> field = SpectralField2D.from_real(field_real, grid)
        >>> field_k = field.fourier  # Lazy FFT on first access
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid: SpectralGrid2D
    _real: Optional[Array] = PrivateAttr(default=None)
    _fourier: Optional[Array] = PrivateAttr(default=None)

    @classmethod
    def from_real(cls, field_real: Array, grid: SpectralGrid2D) -> "SpectralField2D":
        """
        Create a SpectralField2D from real-space data.

        Args:
            field_real: Real-space field array (shape: [Ny, Nx])
            grid: SpectralGrid2D defining the domain

        Returns:
            SpectralField2D with real data cached

        Raises:
            ValueError: If field_real shape does not match grid dimensions
        """
        expected_shape = (grid.Ny, grid.Nx)
        if field_real.shape != expected_shape:
            raise ValueError(
                f"Field shape {field_real.shape} does not match grid shape {expected_shape}"
            )
        instance = cls(grid=grid)
        instance._real = field_real
        return instance

    @classmethod
    def from_fourier(
        cls, field_fourier: Array, grid: SpectralGrid2D
    ) -> "SpectralField2D":
        """
        Create a SpectralField2D from Fourier-space data.

        Args:
            field_fourier: Fourier-space field array (shape: [Ny, Nx//2+1])
            grid: SpectralGrid2D defining the domain

        Returns:
            SpectralField2D with Fourier data cached

        Raises:
            ValueError: If field_fourier shape does not match grid dimensions
        """
        expected_shape = (grid.Ny, grid.Nx // 2 + 1)
        if field_fourier.shape != expected_shape:
            raise ValueError(
                f"Field shape {field_fourier.shape} does not match expected Fourier shape {expected_shape}"
            )
        instance = cls(grid=grid)
        instance._fourier = field_fourier
        return instance

    @property
    def real(self) -> Array:
        """
        Get real-space representation, transforming from Fourier if needed.

        Returns:
            Real-space array (shape: [Ny, Nx])

        Warning:
            The returned array is cached. While JAX arrays are conceptually immutable,
            do not modify the returned array as it may lead to inconsistent state.
            Create a copy if modifications are needed: `field.real.copy()`
        """
        if self._real is None:
            if self._fourier is None:
                raise ValueError("Field has neither real nor Fourier data")
            self._real = rfft2_inverse(self._fourier, self.grid.Ny, self.grid.Nx)
        return self._real

    @property
    def fourier(self) -> Array:
        """
        Get Fourier-space representation, transforming from real if needed.

        Returns:
            Fourier-space array (shape: [Ny, Nx//2+1])

        Warning:
            The returned array is cached. While JAX arrays are conceptually immutable,
            do not modify the returned array as it may lead to inconsistent state.
            Create a copy if modifications are needed: `field.fourier.copy()`
        """
        if self._fourier is None:
            if self._real is None:
                raise ValueError("Field has neither real nor Fourier data")
            self._fourier = rfft2_forward(self._real)
        return self._fourier


# =============================================================================
# FFT Operations (JIT-compiled)
# =============================================================================


@jax.jit
def rfft2_forward(field_real: Array) -> Array:
    """
    Forward 2D real-to-complex FFT.

    Uses rfft2 to exploit reality condition: F(-k) = F*(k), saving 50% memory.

    Args:
        field_real: Real-space field (shape: [Ny, Nx])

    Returns:
        Fourier-space field (shape: [Ny, Nx//2+1], complex)

    Note:
        No normalization on forward transform (normalization is on inverse).
    """
    return jnp.fft.rfft2(field_real)


def rfft2_inverse(field_fourier: Array, Ny: int, Nx: int) -> Array:
    """
    Inverse 2D complex-to-real FFT.

    Args:
        field_fourier: Fourier-space field (shape: [Ny, Nx//2+1], complex)
        Ny: Output shape in y direction
        Nx: Output shape in x direction

    Returns:
        Real-space field (shape: [Ny, Nx], real)

    Note:
        Normalization factor 1/(Nx*Ny) is automatically applied by irfft2.
        Not JIT-compiled because shape arguments cannot be traced.
        The underlying jnp.fft.irfft2 is already highly optimized.
    """
    return jnp.fft.irfft2(field_fourier, s=(Ny, Nx))


# =============================================================================
# Spectral Derivatives (JIT-compiled)
# =============================================================================


@jax.jit
def derivative_x(field_fourier: Array, kx: Array) -> Array:
    """
    Compute ∂f/∂x in Fourier space: ∂f/∂x → i·kx·f̂(k).

    This is exact for band-limited functions (no truncation error).

    Args:
        field_fourier: Fourier-space field (shape: [Ny, Nx//2+1])
        kx: Wavenumber array in x (shape: [Nx//2+1])

    Returns:
        Fourier-space derivative ∂f/∂x (shape: [Ny, Nx//2+1])

    Example:
        >>> # For f(x) = sin(kx), ∂f/∂x = k·cos(kx)
        >>> df_dx_fourier = derivative_x(f_fourier, grid.kx)
        >>> df_dx_real = rfft2_inverse(df_dx_fourier, grid.Ny, grid.Nx)
    """
    # Broadcast kx to shape [Ny, Nx//2+1]
    kx_2d = kx[jnp.newaxis, :]  # Shape: [1, Nx//2+1]
    return 1j * kx_2d * field_fourier


@jax.jit
def derivative_y(field_fourier: Array, ky: Array) -> Array:
    """
    Compute ∂f/∂y in Fourier space: ∂f/∂y → i·ky·f̂(k).

    This is exact for band-limited functions (no truncation error).

    Args:
        field_fourier: Fourier-space field (shape: [Ny, Nx//2+1])
        ky: Wavenumber array in y (shape: [Ny])

    Returns:
        Fourier-space derivative ∂f/∂y (shape: [Ny, Nx//2+1])

    Example:
        >>> # For f(y) = sin(ky), ∂f/∂y = k·cos(ky)
        >>> df_dy_fourier = derivative_y(f_fourier, grid.ky)
        >>> df_dy_real = rfft2_inverse(df_dy_fourier, grid.Ny, grid.Nx)
    """
    # Broadcast ky to shape [Ny, Nx//2+1]
    ky_2d = ky[:, jnp.newaxis]  # Shape: [Ny, 1]
    return 1j * ky_2d * field_fourier


@jax.jit
def laplacian(field_fourier: Array, kx: Array, ky: Array) -> Array:
    """
    Compute Laplacian ∇²f in Fourier space: ∇²f → -(kx² + ky²)·f̂(k).

    This is a fundamental operation in KRMHD for the Poisson solver and
    dissipation terms. More efficient than computing ∂²/∂x² and ∂²/∂y² separately.

    Args:
        field_fourier: Fourier-space field (shape: [Ny, Nx//2+1])
        kx: Wavenumber array in x (shape: [Nx//2+1])
        ky: Wavenumber array in y (shape: [Ny])

    Returns:
        Fourier-space Laplacian ∇²f (shape: [Ny, Nx//2+1])

    Example:
        >>> # For f(x,y) = sin(kx*x)·sin(ky*y), ∇²f = -(kx² + ky²)·f
        >>> lap_fourier = laplacian(f_fourier, grid.kx, grid.ky)
        >>> lap_real = rfft2_inverse(lap_fourier, grid.Ny, grid.Nx)

    Note:
        This will be essential for the Poisson solver in Step 3:
        k²φ = ∇²⊥A∥ → φ = ∇²⊥A∥ / k²
    """
    # Broadcast wavenumbers to 2D
    kx_2d = kx[jnp.newaxis, :]  # Shape: [1, Nx//2+1]
    ky_2d = ky[:, jnp.newaxis]  # Shape: [Ny, 1]

    # Compute k² = kx² + ky²
    k_squared = kx_2d**2 + ky_2d**2

    # ∇²f = -(kx² + ky²)·f̂(k)
    return -k_squared * field_fourier


# =============================================================================
# Dealiasing (JIT-compiled)
# =============================================================================


@jax.jit
def dealias(field_fourier: Array, dealias_mask: Array) -> Array:
    """
    Apply 2/3 rule dealiasing by zeroing high-wavenumber modes.

    Prevents aliasing errors in nonlinear terms by ensuring products of
    dealiased fields remain properly resolved. Modes where
    max(|kx|/kx_max, |ky|/ky_max) > 2/3 are set to zero.

    This is CRITICAL for stability in nonlinear spectral codes.

    Args:
        field_fourier: Fourier-space field (shape: [Ny, Nx//2+1])
        dealias_mask: Boolean mask for dealiasing (shape: [Ny, Nx//2+1])

    Returns:
        Dealiased Fourier-space field (shape: [Ny, Nx//2+1])

    Example:
        >>> # After computing nonlinear term: f * g
        >>> fg_fourier = rfft2_forward(f_real * g_real)
        >>> fg_fourier = dealias(fg_fourier, grid.dealias_mask)  # MUST dealias!
    """
    return field_fourier * dealias_mask
