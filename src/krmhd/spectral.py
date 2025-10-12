"""
Core spectral infrastructure for KRMHD solver.

This module provides 2D and 3D spectral operations using FFT methods:
- SpectralGrid2D: Pydantic model defining 2D grid parameters and wavenumbers
- SpectralGrid3D: Pydantic model defining 3D grid parameters and wavenumbers
- SpectralField2D/3D: Manages real-space and Fourier-space representations
- FFT operations using rfft2/rfftn for real-to-complex transforms
- Spectral derivatives (∂x, ∂y, ∂z) computed as multiplication in Fourier space
- Laplacian operators (∇², ∇²⊥) for 2D and 3D fields
- Poisson solvers for ∇²φ = ω and ∇²⊥φ = ω
- Dealiasing using the 2/3 rule to prevent aliasing errors

All performance-critical functions are JIT-compiled with JAX.
"""

from functools import partial
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
        # meshgrid with indexing="ij" produces [len(kx), len(ky)]
        kx_2d, ky_2d = jnp.meshgrid(kx, ky, indexing="ij")

        # Transpose to match rfft2 output shape [Ny, Nx//2+1]
        # meshgrid order: [len(kx), len(ky)] = [Nx//2+1, Ny]
        # After .T: [len(ky), len(kx)] = [Ny, Nx//2+1] ✓
        kx_2d = kx_2d.T
        ky_2d = ky_2d.T

        # Mask: True where mode should be kept, False where it should be zeroed
        # Use explicit float32 for 2/3 threshold to match JAX default precision
        threshold = jnp.float32(2.0 / 3.0)
        dealias_mask = (jnp.abs(kx_2d) <= threshold * kx_max) & (
            jnp.abs(ky_2d) <= threshold * ky_max
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


class SpectralGrid3D(BaseModel):
    """
    Immutable 3D spectral grid specification with wavenumber arrays.

    Defines a rectangular grid in real space (Nx × Ny × Nz) with periodic boundary
    conditions. Wavenumber arrays are pre-computed for spectral derivatives,
    and a dealiasing mask implements the 2/3 rule for nonlinear operations.

    The z-direction is parallel to the background magnetic field B₀.

    Attributes:
        Nx: Number of grid points in x direction (must be > 0, even)
        Ny: Number of grid points in y direction (must be > 0, even)
        Nz: Number of grid points in z direction (must be > 0, even)
        Lx: Physical domain size in x direction (must be > 0)
        Ly: Physical domain size in y direction (must be > 0)
        Lz: Physical domain size in z direction (must be > 0)
        kx: Wavenumber array in x (shape: [Nx//2+1], non-negative for rfftn)
        ky: Wavenumber array in y (shape: [Ny], includes negative frequencies)
        kz: Wavenumber array in z (shape: [Nz], includes negative frequencies)
        dealias_mask: Boolean mask for 2/3 rule dealiasing (shape: [Nz, Ny, Nx//2+1])

    Example:
        >>> grid = SpectralGrid3D.create(Nx=128, Ny=128, Nz=128,
        ...                               Lx=2*jnp.pi, Ly=2*jnp.pi, Lz=2*jnp.pi)
        >>> grid.Nx
        128
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    Nx: int = Field(gt=0, description="Number of grid points in x")
    Ny: int = Field(gt=0, description="Number of grid points in y")
    Nz: int = Field(gt=0, description="Number of grid points in z")
    Lx: float = Field(gt=0.0, description="Domain size in x")
    Ly: float = Field(gt=0.0, description="Domain size in y")
    Lz: float = Field(gt=0.0, description="Domain size in z")
    kx: Array = Field(description="Wavenumber array in x (non-negative)")
    ky: Array = Field(description="Wavenumber array in y (includes negative)")
    kz: Array = Field(description="Wavenumber array in z (includes negative)")
    dealias_mask: Array = Field(description="Boolean mask for 2/3 rule dealiasing")

    @field_validator("Nx", "Ny", "Nz")
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
        Nz: int,
        Lx: float = 2 * jnp.pi,
        Ly: float = 2 * jnp.pi,
        Lz: float = 2 * jnp.pi,
    ) -> "SpectralGrid3D":
        """
        Factory method to create a SpectralGrid3D with computed wavenumbers.

        Wavenumber arrays follow the rfftn convention:
        - kx: [0, 1, 2, ..., Nx//2] (non-negative only, exploits reality)
        - ky: [0, 1, ..., Ny//2-1, -Ny//2, ..., -1] (standard FFT ordering)
        - kz: [0, 1, ..., Nz//2-1, -Nz//2, ..., -1] (standard FFT ordering)

        The 2/3 dealiasing mask zeros modes where max(|kx|/kx_max, |ky|/ky_max, |kz|/kz_max) > 2/3
        to prevent aliasing errors in nonlinear terms.

        Args:
            Nx: Number of grid points in x (must be even and > 0)
            Ny: Number of grid points in y (must be even and > 0)
            Nz: Number of grid points in z (must be even and > 0)
            Lx: Physical domain size in x (default: 2π)
            Ly: Physical domain size in y (default: 2π)
            Lz: Physical domain size in z (default: 2π)

        Returns:
            SpectralGrid3D instance with pre-computed wavenumbers and mask

        Raises:
            ValueError: If Nx, Ny, or Nz are not positive even integers, or if Lx, Ly, Lz ≤ 0
        """
        # Early validation before JAX operations
        if Nx <= 0 or Ny <= 0 or Nz <= 0:
            raise ValueError(
                f"Grid dimensions must be positive, got Nx={Nx}, Ny={Ny}, Nz={Nz}"
            )
        if Nx % 2 != 0 or Ny % 2 != 0 or Nz % 2 != 0:
            raise ValueError(
                f"Grid dimensions must be even, got Nx={Nx}, Ny={Ny}, Nz={Nz}"
            )
        if Lx <= 0 or Ly <= 0 or Lz <= 0:
            raise ValueError(
                f"Domain sizes must be positive, got Lx={Lx}, Ly={Ly}, Lz={Lz}"
            )

        # Wavenumber arrays (frequency space)
        kx = jnp.fft.rfftfreq(Nx, d=Lx / (2 * jnp.pi * Nx))
        ky = jnp.fft.fftfreq(Ny, d=Ly / (2 * jnp.pi * Ny))
        kz = jnp.fft.fftfreq(Nz, d=Lz / (2 * jnp.pi * Nz))

        # 2/3 rule dealiasing: zero modes beyond |k| > (2/3) * k_max
        kx_max = jnp.max(jnp.abs(kx))
        ky_max = jnp.max(jnp.abs(ky))
        kz_max = jnp.max(jnp.abs(kz))

        # Create 3D mesh for mask computation
        # meshgrid with indexing="ij" produces [len(kx), len(ky), len(kz)]
        kx_3d, ky_3d, kz_3d = jnp.meshgrid(kx, ky, kz, indexing="ij")

        # Transpose to match rfftn output shape [Nz, Ny, Nx//2+1]
        # meshgrid order: [len(kx), len(ky), len(kz)] = [Nx//2+1, Ny, Nz]
        # After transpose(2,1,0): [len(kz), len(ky), len(kx)] = [Nz, Ny, Nx//2+1] ✓
        kx_3d = kx_3d.transpose(2, 1, 0)
        ky_3d = ky_3d.transpose(2, 1, 0)
        kz_3d = kz_3d.transpose(2, 1, 0)

        # Mask: True where mode should be kept, False where it should be zeroed
        # Use explicit float32 for 2/3 threshold to match JAX default precision
        threshold = jnp.float32(2.0 / 3.0)
        dealias_mask = (
            (jnp.abs(kx_3d) <= threshold * kx_max)
            & (jnp.abs(ky_3d) <= threshold * ky_max)
            & (jnp.abs(kz_3d) <= threshold * kz_max)
        )

        return cls(
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            kx=kx,
            ky=ky,
            kz=kz,
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


class SpectralField3D(BaseModel):
    """
    Manages a 3D field in both real and Fourier space with lazy evaluation.

    Stores a field in ONE of {real, Fourier} representation and transforms
    on-demand when the other is accessed. This avoids redundant FFTs.

    Attributes:
        grid: The spectral grid defining the field's domain
        _real: Private cache for real-space representation (Nz × Ny × Nx)
        _fourier: Private cache for Fourier-space representation (Nz × Ny × Nx//2+1)

    Example:
        >>> grid = SpectralGrid3D.create(64, 64, 64)
        >>> field_real = jnp.sin(jnp.linspace(0, 2*jnp.pi, 64*64*64).reshape(64,64,64))
        >>> field = SpectralField3D.from_real(field_real, grid)
        >>> field_k = field.fourier  # Lazy FFT on first access
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid: SpectralGrid3D
    _real: Optional[Array] = PrivateAttr(default=None)
    _fourier: Optional[Array] = PrivateAttr(default=None)

    @classmethod
    def from_real(cls, field_real: Array, grid: SpectralGrid3D) -> "SpectralField3D":
        """
        Create a SpectralField3D from real-space data.

        Args:
            field_real: Real-space field array (shape: [Nz, Ny, Nx])
            grid: SpectralGrid3D defining the domain

        Returns:
            SpectralField3D with real data cached

        Raises:
            ValueError: If field_real shape does not match grid dimensions
        """
        expected_shape = (grid.Nz, grid.Ny, grid.Nx)
        if field_real.shape != expected_shape:
            raise ValueError(
                f"Field shape {field_real.shape} does not match grid shape {expected_shape}"
            )
        instance = cls(grid=grid)
        instance._real = field_real
        return instance

    @classmethod
    def from_fourier(
        cls, field_fourier: Array, grid: SpectralGrid3D
    ) -> "SpectralField3D":
        """
        Create a SpectralField3D from Fourier-space data.

        Args:
            field_fourier: Fourier-space field array (shape: [Nz, Ny, Nx//2+1])
            grid: SpectralGrid3D defining the domain

        Returns:
            SpectralField3D with Fourier data cached

        Raises:
            ValueError: If field_fourier shape does not match grid dimensions
        """
        expected_shape = (grid.Nz, grid.Ny, grid.Nx // 2 + 1)
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
            Real-space array (shape: [Nz, Ny, Nx])

        Warning:
            The returned array is cached. While JAX arrays are conceptually immutable,
            do not modify the returned array as it may lead to inconsistent state.
            Create a copy if modifications are needed: `field.real.copy()`
        """
        if self._real is None:
            if self._fourier is None:
                raise ValueError("Field has neither real nor Fourier data")
            self._real = rfftn_inverse(
                self._fourier, self.grid.Nz, self.grid.Ny, self.grid.Nx
            )
        return self._real

    @property
    def fourier(self) -> Array:
        """
        Get Fourier-space representation, transforming from real if needed.

        Returns:
            Fourier-space array (shape: [Nz, Ny, Nx//2+1])

        Warning:
            The returned array is cached. While JAX arrays are conceptually immutable,
            do not modify the returned array as it may lead to inconsistent state.
            Create a copy if modifications are needed: `field.fourier.copy()`
        """
        if self._fourier is None:
            if self._real is None:
                raise ValueError("Field has neither real nor Fourier data")
            self._fourier = rfftn_forward(self._real)
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


@jax.jit
def rfftn_forward(field_real: Array) -> Array:
    """
    Forward 3D real-to-complex FFT.

    Uses rfftn to exploit reality condition: F(-k) = F*(k), saving ~50% memory.
    The real FFT is performed only in the x-direction (last axis).

    Args:
        field_real: Real-space field (shape: [Nz, Ny, Nx])

    Returns:
        Fourier-space field (shape: [Nz, Ny, Nx//2+1], complex)

    Note:
        No normalization on forward transform (normalization is on inverse).
        Uses axes=(0, 1, 2) for full 3D transform with rfft in x-direction.
    """
    return jnp.fft.rfftn(field_real, axes=(0, 1, 2))


def rfftn_inverse(field_fourier: Array, Nz: int, Ny: int, Nx: int) -> Array:
    """
    Inverse 3D complex-to-real FFT.

    Args:
        field_fourier: Fourier-space field (shape: [Nz, Ny, Nx//2+1], complex)
        Nz: Output shape in z direction
        Ny: Output shape in y direction
        Nx: Output shape in x direction

    Returns:
        Real-space field (shape: [Nz, Ny, Nx], real)

    Note:
        Normalization factor 1/(Nx*Ny*Nz) is automatically applied by irfftn.
        Not JIT-compiled because shape arguments cannot be traced.
        The underlying jnp.fft.irfftn is already highly optimized.
    """
    return jnp.fft.irfftn(field_fourier, s=(Nz, Ny, Nx), axes=(0, 1, 2))


# =============================================================================
# Spectral Derivatives (JIT-compiled)
# =============================================================================


@partial(jax.jit, static_argnames=['ndim'])
def _broadcast_wavenumber_x(k: Array, ndim: int) -> Array:
    """
    Broadcast 1D wavenumber array for x-direction to match field dimensionality.

    Args:
        k: 1D wavenumber array
        ndim: Target dimensionality (2 or 3) - static argument for JIT compilation

    Returns:
        Broadcasted array with shape appropriate for element-wise multiplication
        - 2D: [1, len(k)] for shape [Ny, Nx//2+1]
        - 3D: [1, 1, len(k)] for shape [Nz, Ny, Nx//2+1]

    Note:
        JIT-compiled with ndim as a static argument, allowing JAX to specialize
        the function for each dimensionality without recompilation overhead.
    """
    if ndim == 2:
        return k[jnp.newaxis, :]  # [1, Nx//2+1]
    elif ndim == 3:
        return k[jnp.newaxis, jnp.newaxis, :]  # [1, 1, Nx//2+1]
    else:
        raise ValueError(f"Unsupported field dimensionality: {ndim}")


@partial(jax.jit, static_argnames=['ndim'])
def _broadcast_wavenumber_y(k: Array, ndim: int) -> Array:
    """
    Broadcast 1D wavenumber array for y-direction to match field dimensionality.

    Args:
        k: 1D wavenumber array
        ndim: Target dimensionality (2 or 3) - static argument for JIT compilation

    Returns:
        Broadcasted array with shape appropriate for element-wise multiplication
        - 2D: [len(k), 1] for shape [Ny, Nx//2+1]
        - 3D: [1, len(k), 1] for shape [Nz, Ny, Nx//2+1]

    Note:
        JIT-compiled with ndim as a static argument, allowing JAX to specialize
        the function for each dimensionality without recompilation overhead.
    """
    if ndim == 2:
        return k[:, jnp.newaxis]  # [Ny, 1]
    elif ndim == 3:
        return k[jnp.newaxis, :, jnp.newaxis]  # [1, Ny, 1]
    else:
        raise ValueError(f"Unsupported field dimensionality: {ndim}")


@jax.jit
def derivative_x(field_fourier: Array, kx: Array) -> Array:
    """
    Compute ∂f/∂x in Fourier space: ∂f/∂x → i·kx·f̂(k).

    This is exact for band-limited functions (no truncation error).

    Args:
        field_fourier: Fourier-space field (shape: [Ny, Nx//2+1] or [Nz, Ny, Nx//2+1])
        kx: Wavenumber array in x (shape: [Nx//2+1])

    Returns:
        Fourier-space derivative ∂f/∂x (same shape as input)

    Example:
        >>> # For f(x) = sin(kx), ∂f/∂x = k·cos(kx)
        >>> df_dx_fourier = derivative_x(f_fourier, grid.kx)
        >>> df_dx_real = rfft2_inverse(df_dx_fourier, grid.Ny, grid.Nx)
    """
    kx_nd = _broadcast_wavenumber_x(kx, field_fourier.ndim)
    return 1j * kx_nd * field_fourier


@jax.jit
def derivative_y(field_fourier: Array, ky: Array) -> Array:
    """
    Compute ∂f/∂y in Fourier space: ∂f/∂y → i·ky·f̂(k).

    This is exact for band-limited functions (no truncation error).

    Args:
        field_fourier: Fourier-space field (shape: [Ny, Nx//2+1] or [Nz, Ny, Nx//2+1])
        ky: Wavenumber array in y (shape: [Ny])

    Returns:
        Fourier-space derivative ∂f/∂y (same shape as input)

    Example:
        >>> # For f(y) = sin(ky), ∂f/∂y = k·cos(ky)
        >>> df_dy_fourier = derivative_y(f_fourier, grid.ky)
        >>> df_dy_real = rfft2_inverse(df_dy_fourier, grid.Ny, grid.Nx)
    """
    ky_nd = _broadcast_wavenumber_y(ky, field_fourier.ndim)
    return 1j * ky_nd * field_fourier


@jax.jit
def derivative_z(field_fourier: Array, kz: Array) -> Array:
    """
    Compute ∂f/∂z in Fourier space: ∂f/∂z → i·kz·f̂(k).

    This is the parallel gradient operator ∇∥ for KRMHD with B₀ = B₀ẑ.
    Exact for band-limited functions (no truncation error).

    Args:
        field_fourier: Fourier-space field (shape: [Nz, Ny, Nx//2+1])
        kz: Wavenumber array in z (shape: [Nz])

    Returns:
        Fourier-space derivative ∂f/∂z (shape: [Nz, Ny, Nx//2+1])

    Raises:
        ValueError: If field_fourier is not 3D

    Example:
        >>> # For f(z) = sin(kz), ∂f/∂z = k·cos(kz)
        >>> df_dz_fourier = derivative_z(f_fourier, grid.kz)
        >>> df_dz_real = rfftn_inverse(df_dz_fourier, grid.Nz, grid.Ny, grid.Nx)
    """
    if field_fourier.ndim != 3:
        raise ValueError(f"Unsupported field dimensionality: {field_fourier.ndim}")
    # Broadcast kz to shape [Nz, Ny, Nx//2+1]
    kz_3d = kz[:, jnp.newaxis, jnp.newaxis]  # Shape: [Nz, 1, 1]
    return 1j * kz_3d * field_fourier


@jax.jit
def laplacian(field_fourier: Array, kx: Array, ky: Array, kz: Optional[Array] = None) -> Array:
    """
    Compute Laplacian ∇²f in Fourier space.

    For 2D: ∇²f → -(kx² + ky²)·f̂(k)
    For 3D: ∇²f → -(kx² + ky² + kz²)·f̂(k)

    This is a fundamental operation in KRMHD for the Poisson solver and
    dissipation terms. More efficient than computing individual second derivatives.

    Args:
        field_fourier: Fourier-space field (shape: [Ny, Nx//2+1] or [Nz, Ny, Nx//2+1])
        kx: Wavenumber array in x (shape: [Nx//2+1])
        ky: Wavenumber array in y (shape: [Ny])
        kz: Wavenumber array in z (shape: [Nz], optional for 3D)

    Returns:
        Fourier-space Laplacian ∇²f (same shape as input)

    Example:
        >>> # 2D: For f(x,y) = sin(kx*x)·sin(ky*y), ∇²f = -(kx² + ky²)·f
        >>> lap_fourier = laplacian(f_fourier, grid.kx, grid.ky)
        >>> lap_real = rfft2_inverse(lap_fourier, grid.Ny, grid.Nx)

        >>> # 3D: Include kz for full 3D Laplacian
        >>> lap_fourier_3d = laplacian(f_fourier_3d, grid.kx, grid.ky, grid.kz)

    Note:
        **When to use kz=None:**
        - KRMHD Poisson solver: k²φ = ∇²⊥A∥ requires perpendicular Laplacian only
        - Perpendicular dynamics: Use kz=None for ∇²⊥ = ∂²/∂x² + ∂²/∂y²
        - Dissipation in 3D: Use kz=grid.kz for full 3D Laplacian ∇² = ∇²⊥ + ∂²/∂z²

        For 3D fields, kz=None computes perpendicular Laplacian at each z-plane,
        which is the standard form for KRMHD turbulent cascade operators.
    """
    ndim = field_fourier.ndim
    kx_nd = _broadcast_wavenumber_x(kx, ndim)
    ky_nd = _broadcast_wavenumber_y(ky, ndim)
    k_squared = kx_nd**2 + ky_nd**2

    # Add z-component for full 3D Laplacian if requested
    if ndim == 3 and kz is not None:
        kz_nd = kz[:, jnp.newaxis, jnp.newaxis]  # Shape: [Nz, 1, 1]
        k_squared = k_squared + kz_nd**2

    # ∇²f = -k²·f̂(k)
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


# =============================================================================
# Poisson Solver (JIT-compiled)
# =============================================================================


@jax.jit
def poisson_solve_2d(omega_fourier: Array, kx: Array, ky: Array) -> Array:
    """
    Solve 2D Poisson equation ∇²φ = ω in Fourier space.

    In Fourier space, using the convention that laplacian() returns -k²·φ̂:
        F{∇²φ} = -k²·φ̂ = F{ω}
        Therefore: φ̂ = -ω̂/k²

    The k=0 mode (constant component) is set to zero since the Poisson
    equation only determines φ up to an additive constant.

    Args:
        omega_fourier: Fourier-space vorticity field (shape: [Ny, Nx//2+1])
        kx: Wavenumber array in x (shape: [Nx//2+1])
        ky: Wavenumber array in y (shape: [Ny])

    Returns:
        Fourier-space solution φ̂ (shape: [Ny, Nx//2+1])

    Raises:
        ValueError: If omega_fourier is not 2D

    Physics context:
        In KRMHD, this solves for the stream function φ from vorticity:
            ω = ∇²φ  (in 2D incompressible flow)
        The stream function generates velocity via: v = ẑ × ∇φ

    Example:
        >>> # Manufactured solution: φ = sin(x)·cos(y)
        >>> # Then: ω = ∇²φ = -2·sin(x)·cos(y)
        >>> omega_real = -2 * jnp.sin(x) * jnp.cos(y)
        >>> omega_fourier = rfft2_forward(omega_real)
        >>> phi_fourier = poisson_solve_2d(omega_fourier, grid.kx, grid.ky)
        >>> phi_real = rfft2_inverse(phi_fourier, grid.Ny, grid.Nx)

    Note:
        - k=0 mode is fixed at zero (arbitrary constant choice)
        - No boundary conditions needed (periodic domain)
        - Division by zero at k=0 is handled via jnp.where
        - This is spectrally accurate (no truncation error)
    """
    # Validate input dimensions
    if omega_fourier.ndim != 2:
        raise ValueError(
            f"Expected 2D field, got {omega_fourier.ndim}D with shape {omega_fourier.shape}"
        )
    # Broadcast wavenumbers to match field shape
    kx_2d = _broadcast_wavenumber_x(kx, ndim=2)
    ky_2d = _broadcast_wavenumber_y(ky, ndim=2)

    # Compute k² = kx² + ky²
    k_squared = kx_2d**2 + ky_2d**2

    # Solve: φ̂ = -ω̂/k²
    # Since laplacian() returns F{∇²φ} = -k²·φ̂, we have -k²·φ̂ = ω̂, thus φ̂ = -ω̂/k²
    # Use jnp.where to handle k=0 mode (set to zero)
    phi_fourier = jnp.where(
        k_squared > 0.0,
        -omega_fourier / k_squared,
        0.0 + 0.0j  # k=0 mode set to zero
    )

    return phi_fourier


@jax.jit
def poisson_solve_3d(
    omega_fourier: Array,
    kx: Array,
    ky: Array,
    kz: Optional[Array] = None
) -> Array:
    """
    Solve 3D Poisson equation in Fourier space.

    Solves either:
    - Perpendicular Laplacian (kz=None): ∇²⊥φ = ω  where ∇²⊥ = ∂²/∂x² + ∂²/∂y²
    - Full 3D Laplacian (kz provided): ∇²φ = ω  where ∇² = ∂²/∂x² + ∂²/∂y² + ∂²/∂z²

    In Fourier space, using the convention that laplacian() returns -k²·φ̂:
        F{∇²⊥φ} = -(kx² + ky²)·φ̂ = F{ω}  →  φ̂ = -ω̂/(kx² + ky²)
        F{∇²φ} = -(kx² + ky² + kz²)·φ̂ = F{ω}  →  φ̂ = -ω̂/k²

    The k=0 mode is set to zero for each case.

    Args:
        omega_fourier: Fourier-space source field (shape: [Nz, Ny, Nx//2+1])
        kx: Wavenumber array in x (shape: [Nx//2+1])
        ky: Wavenumber array in y (shape: [Ny])
        kz: Wavenumber array in z (shape: [Nz], optional)
            If None: solve ∇²⊥φ = ω (perpendicular Laplacian)
            If provided: solve ∇²φ = ω (full 3D Laplacian)

    Returns:
        Fourier-space solution φ̂ (shape: [Nz, Ny, Nx//2+1])

    Physics context:
        In KRMHD with B₀ = B₀ẑ, the Poisson bracket formulation requires
        solving the perpendicular Poisson equation:
            ∇²⊥φ = ∇²⊥A∥
        This determines φ at each z-plane independently when kz=None.

        For problems requiring full 3D coupling, use kz=grid.kz.

    Example:
        >>> # KRMHD: Solve ∇²⊥φ = ∇²⊥A∥ (perpendicular only)
        >>> omega_fourier = laplacian(A_parallel_fourier, grid.kx, grid.ky, kz=None)
        >>> phi_fourier = poisson_solve_3d(omega_fourier, grid.kx, grid.ky, kz=None)

        >>> # Alternative: Full 3D Poisson equation
        >>> phi_fourier_3d = poisson_solve_3d(omega_fourier, grid.kx, grid.ky, grid.kz)

    Note:
        - Default (kz=None) uses perpendicular Laplacian for KRMHD
        - k=0 mode is fixed at zero for each k∥ mode
        - Spectrally accurate (no truncation error)
    """
    if omega_fourier.ndim != 3:
        raise ValueError(
            f"Expected 3D field, got {omega_fourier.ndim}D with shape {omega_fourier.shape}"
        )

    # Broadcast wavenumbers to match 3D field shape
    kx_3d = _broadcast_wavenumber_x(kx, ndim=3)
    ky_3d = _broadcast_wavenumber_y(ky, ndim=3)

    # Compute perpendicular k²⊥ = kx² + ky²
    k_squared = kx_3d**2 + ky_3d**2

    # Add parallel component if requested
    if kz is not None:
        kz_3d = kz[:, jnp.newaxis, jnp.newaxis]  # Shape: [Nz, 1, 1]
        k_squared = k_squared + kz_3d**2

    # Solve: φ̂ = -ω̂/k²
    # Since laplacian() returns F{∇²φ} = -k²·φ̂, we have -k²·φ̂ = ω̂, thus φ̂ = -ω̂/k²
    # Use jnp.where to handle k=0 mode (set to zero)
    phi_fourier = jnp.where(
        k_squared > 0.0,
        -omega_fourier / k_squared,
        0.0 + 0.0j  # k=0 mode set to zero
    )

    return phi_fourier
