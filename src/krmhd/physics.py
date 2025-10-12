"""
KRMHD physics operations for nonlinear dynamics.

This module implements the core physics operators for Kinetic Reduced MHD:
- Poisson bracket: {f,g} = ẑ·(∇f × ∇g) for perpendicular advection
- Future: RHS functions for time evolution of Alfvén and passive fields

The Poisson bracket is the fundamental nonlinearity in KRMHD, appearing in:
- Alfvénic advection: ∂A∥/∂t + {φ, A∥} = ...
- Passive scalar advection: ∂δB∥/∂t + {φ, δB∥} = ...
- Vorticity evolution: ∂ω/∂t + {φ, ω} = ...

All functions use JAX for GPU acceleration and are JIT-compiled for performance.
"""

from functools import partial
from typing import Union
import jax
import jax.numpy as jnp
from jax import Array

from krmhd.spectral import (
    SpectralGrid2D,
    SpectralGrid3D,
    derivative_x,
    derivative_y,
    rfft2_forward,
    rfft2_inverse,
    rfftn_forward,
    rfftn_inverse,
    dealias,
)


@partial(jax.jit, static_argnums=(4, 5))
def poisson_bracket_2d(
    f_fourier: Array,
    g_fourier: Array,
    kx: Array,
    ky: Array,
    Ny: int,
    Nx: int,
    dealias_mask: Array,
) -> Array:
    """
    Compute 2D Poisson bracket {f,g} = ẑ·(∇f × ∇g) = ∂f/∂x · ∂g/∂y - ∂f/∂y · ∂g/∂x.

    This is the fundamental nonlinear operator in KRMHD for perpendicular advection.
    The computation is performed in spectral space for derivatives, then transformed
    to real space for multiplication, and back to spectral space with dealiasing.

    Algorithm:
        1. Compute spectral derivatives: ∂f/∂x, ∂f/∂y, ∂g/∂x, ∂g/∂y
        2. Transform derivatives to real space
        3. Compute cross product in real space: ∂f/∂x · ∂g/∂y - ∂f/∂y · ∂g/∂x
        4. Transform result back to Fourier space
        5. Apply 2/3 dealiasing to prevent aliasing errors

    Args:
        f_fourier: Fourier-space field f (shape: [Ny, Nx//2+1])
        g_fourier: Fourier-space field g (shape: [Ny, Nx//2+1])
        kx: Wavenumber array in x (shape: [Nx//2+1])
        ky: Wavenumber array in y (shape: [Ny])
        Ny: Number of grid points in y direction
        Nx: Number of grid points in x direction
        dealias_mask: Boolean mask for 2/3 rule dealiasing (shape: [Ny, Nx//2+1])

    Returns:
        Fourier-space Poisson bracket {f,g} (shape: [Ny, Nx//2+1])

    Properties:
        - Anti-symmetric: {f, g} = -{g, f}
        - Bilinear: {af + bh, g} = a{f,g} + b{h,g}
        - Vanishes for constants: {f, c} = 0
        - Conserves L2 norm: ∫ f·{g,h} dx = 0 for periodic boundaries

    Example:
        >>> grid = SpectralGrid2D.create(Nx=128, Ny=128)
        >>> # For f = sin(x), g = cos(y): {f,g} = -cos(x)·sin(y)
        >>> x = jnp.linspace(0, grid.Lx, grid.Nx, endpoint=False)
        >>> y = jnp.linspace(0, grid.Ly, grid.Ny, endpoint=False)
        >>> X, Y = jnp.meshgrid(x, y, indexing='ij')
        >>> f = jnp.sin(X).T  # Transpose for [Ny, Nx] ordering
        >>> g = jnp.cos(Y).T
        >>> f_k = rfft2_forward(f)
        >>> g_k = rfft2_forward(g)
        >>> bracket_k = poisson_bracket_2d(f_k, g_k, grid.kx, grid.ky, grid.Ny, grid.Nx, grid.dealias_mask)
        >>> bracket = rfft2_inverse(bracket_k, grid.Ny, grid.Nx)
        >>> # bracket ≈ -cos(X)·sin(Y)

    Note:
        The 2/3 dealiasing is CRITICAL for stability. Without it, aliasing errors
        from the nonlinear product will cause spurious energy growth and eventual
        numerical instability. The dealiasing zeros all modes where
        max(|kx|/kx_max, |ky|/ky_max) > 2/3.
    """
    # 1. Compute derivatives in Fourier space
    df_dx_fourier = derivative_x(f_fourier, kx)
    df_dy_fourier = derivative_y(f_fourier, ky)
    dg_dx_fourier = derivative_x(g_fourier, kx)
    dg_dy_fourier = derivative_y(g_fourier, ky)

    # 2. Transform to real space for multiplication
    df_dx = rfft2_inverse(df_dx_fourier, Ny, Nx)
    df_dy = rfft2_inverse(df_dy_fourier, Ny, Nx)
    dg_dx = rfft2_inverse(dg_dx_fourier, Ny, Nx)
    dg_dy = rfft2_inverse(dg_dy_fourier, Ny, Nx)

    # 3. Compute cross product: ẑ·(∇f × ∇g) = ∂f/∂x · ∂g/∂y - ∂f/∂y · ∂g/∂x
    bracket_real = df_dx * dg_dy - df_dy * dg_dx

    # 4. Transform back to Fourier space
    bracket_fourier = rfft2_forward(bracket_real)

    # 5. Apply 2/3 dealiasing (CRITICAL!)
    bracket_fourier = dealias(bracket_fourier, dealias_mask)

    return bracket_fourier


@partial(jax.jit, static_argnums=(4, 5, 6))
def poisson_bracket_3d(
    f_fourier: Array,
    g_fourier: Array,
    kx: Array,
    ky: Array,
    Nz: int,
    Ny: int,
    Nx: int,
    dealias_mask: Array,
) -> Array:
    """
    Compute 3D Poisson bracket {f,g} = ẑ·(∇f × ∇g) = ∂f/∂x · ∂g/∂y - ∂f/∂y · ∂g/∂x.

    This is identical to the 2D version since the Poisson bracket only involves
    perpendicular derivatives (∂/∂x and ∂/∂y). The operation is applied at each
    z-plane independently. The z-dependence enters only through the fields f and g.

    Algorithm:
        1. Compute spectral derivatives: ∂f/∂x, ∂f/∂y, ∂g/∂x, ∂g/∂y
        2. Transform derivatives to real space
        3. Compute cross product in real space: ∂f/∂x · ∂g/∂y - ∂f/∂y · ∂g/∂x
        4. Transform result back to Fourier space
        5. Apply 2/3 dealiasing to prevent aliasing errors

    Args:
        f_fourier: Fourier-space field f (shape: [Nz, Ny, Nx//2+1])
        g_fourier: Fourier-space field g (shape: [Nz, Ny, Nx//2+1])
        kx: Wavenumber array in x (shape: [Nx//2+1])
        ky: Wavenumber array in y (shape: [Ny])
        Nz: Number of grid points in z direction
        Ny: Number of grid points in y direction
        Nx: Number of grid points in x direction
        dealias_mask: Boolean mask for 2/3 rule dealiasing (shape: [Nz, Ny, Nx//2+1])

    Returns:
        Fourier-space Poisson bracket {f,g} (shape: [Nz, Ny, Nx//2+1])

    Properties:
        - Anti-symmetric: {f, g} = -{g, f}
        - Bilinear: {af + bh, g} = a{f,g} + b{h,g}
        - Vanishes for constants: {f, c} = 0
        - Conserves L2 norm: ∫ f·{g,h} dx = 0 for periodic boundaries
        - Perpendicular operator: Only involves ∂/∂x and ∂/∂y, not ∂/∂z

    Example:
        >>> grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)
        >>> # For f = sin(x), g = cos(y): {f,g} = -cos(x)·sin(y) at all z
        >>> x = jnp.linspace(0, grid.Lx, grid.Nx, endpoint=False)
        >>> y = jnp.linspace(0, grid.Ly, grid.Ny, endpoint=False)
        >>> z = jnp.linspace(0, grid.Lz, grid.Nz, endpoint=False)
        >>> X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
        >>> # Transpose to [Nz, Ny, Nx] ordering
        >>> f = jnp.sin(X).transpose(2, 1, 0)
        >>> g = jnp.cos(Y).transpose(2, 1, 0)
        >>> f_k = rfftn_forward(f)
        >>> g_k = rfftn_forward(g)
        >>> bracket_k = poisson_bracket_3d(f_k, g_k, grid.kx, grid.ky, grid.Nz, grid.Ny, grid.Nx, grid.dealias_mask)
        >>> bracket = rfftn_inverse(bracket_k, grid.Nz, grid.Ny, grid.Nx)
        >>> # bracket ≈ -cos(X)·sin(Y) at all z-planes

    Note:
        For KRMHD with B₀ = B₀ẑ, different k∥ (kz) modes evolve independently
        through the perpendicular Poisson bracket. However, we use full 3D grids
        for field line following and to track parallel structure in turbulence.

        The 2/3 dealiasing is CRITICAL for stability in the same way as 2D.
    """
    # 1. Compute derivatives in Fourier space (perpendicular only)
    df_dx_fourier = derivative_x(f_fourier, kx)
    df_dy_fourier = derivative_y(f_fourier, ky)
    dg_dx_fourier = derivative_x(g_fourier, kx)
    dg_dy_fourier = derivative_y(g_fourier, ky)

    # 2. Transform to real space for multiplication
    df_dx = rfftn_inverse(df_dx_fourier, Nz, Ny, Nx)
    df_dy = rfftn_inverse(df_dy_fourier, Nz, Ny, Nx)
    dg_dx = rfftn_inverse(dg_dx_fourier, Nz, Ny, Nx)
    dg_dy = rfftn_inverse(dg_dy_fourier, Nz, Ny, Nx)

    # 3. Compute cross product: ẑ·(∇f × ∇g) = ∂f/∂x · ∂g/∂y - ∂f/∂y · ∂g/∂x
    bracket_real = df_dx * dg_dy - df_dy * dg_dx

    # 4. Transform back to Fourier space
    bracket_fourier = rfftn_forward(bracket_real)

    # 5. Apply 2/3 dealiasing (CRITICAL!)
    bracket_fourier = dealias(bracket_fourier, dealias_mask)

    return bracket_fourier
