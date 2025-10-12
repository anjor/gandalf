"""
KRMHD physics operations for nonlinear dynamics.

This module implements the core physics operators for Kinetic Reduced MHD:
- Poisson bracket: {f,g} = ẑ·(∇f × ∇g) for perpendicular advection
- KRMHD state representation with fluid and kinetic fields
- Initialization functions for Alfvén waves and turbulent spectra
- Energy calculations for diagnostics

The Poisson bracket is the fundamental nonlinearity in KRMHD, appearing in:
- Alfvénic advection: ∂A∥/∂t + {φ, A∥} = ...
- Passive scalar advection: ∂δB∥/∂t + {φ, δB∥} = ...
- Vorticity evolution: ∂ω/∂t + {φ, ω} = ...

All functions use JAX for GPU acceleration and are JIT-compiled for performance.
"""

from functools import partial
from typing import Optional, Dict
import jax
import jax.numpy as jnp
from jax import Array
from pydantic import BaseModel, Field, field_validator, ConfigDict

from krmhd.spectral import (
    SpectralGrid3D,
    derivative_x,
    derivative_y,
    rfft2_forward,
    rfft2_inverse,
    rfftn_forward,
    rfftn_inverse,
    dealias,
)
from krmhd.hermite import hermite_basis_function


class KRMHDState(BaseModel):
    """
    Complete KRMHD state with fluid and kinetic fields.

    This dataclass represents the full state of the Kinetic Reduced MHD system,
    including both the Alfvénic (active) fields and the kinetic (Hermite moment)
    representation of the electron distribution function.

    All field arrays are stored in Fourier space for efficient spectral operations.
    The fields satisfy:
    - Reality condition: f(-k) = f*(k)
    - Divergence-free magnetic field: ∇·B = 0 (automatically satisfied)

    Attributes:
        phi: Stream function φ in Fourier space (shape: [Nz, Ny, Nx//2+1])
            Generates perpendicular velocity: v⊥ = ẑ × ∇φ
        A_parallel: Parallel vector potential A∥ in Fourier space (shape: [Nz, Ny, Nx//2+1])
            Generates perpendicular magnetic field: B⊥ = ẑ × ∇A∥
        B_parallel: Parallel magnetic field δB∥ (passive) in Fourier space (shape: [Nz, Ny, Nx//2+1])
        g: Hermite moments of electron distribution (shape: [Nz, Ny, Nx//2+1, M+1])
            Expansion: g(v∥) = Σ_m g_m · ψ_m(v∥/v_th)
        M: Number of Hermite moments (typically 20-30 for converged kinetics)
        beta_i: Ion plasma beta β_i = 8πn_i T_i / B₀²
        v_th: Electron thermal velocity v_th = √(T_e/m_e)
        nu: Collision frequency ν (Lenard-Bernstein operator)
        time: Simulation time
        grid: Reference to SpectralGrid3D for spatial dimensions

    Example:
        >>> grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)
        >>> state = KRMHDState(
        ...     phi=jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1), dtype=complex),
        ...     A_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1), dtype=complex),
        ...     B_parallel=jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1), dtype=complex),
        ...     g=jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1, 21), dtype=complex),
        ...     M=20,
        ...     beta_i=1.0,
        ...     v_th=1.0,
        ...     nu=0.01,
        ...     time=0.0,
        ...     grid=grid
        ... )

    Physics context:
        The KRMHD equations evolve these fields self-consistently:
        - Active (Alfvénic) sector: φ and A∥ couple through Poisson bracket
        - Passive sector: B∥ is advected by φ without back-reaction
        - Kinetic sector: g moments evolve with Landau damping and collisions
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    phi: Array = Field(description="Stream function in Fourier space")
    A_parallel: Array = Field(description="Parallel vector potential in Fourier space")
    B_parallel: Array = Field(description="Parallel magnetic field in Fourier space")
    g: Array = Field(description="Hermite moments of electron distribution")
    M: int = Field(gt=0, description="Number of Hermite moments")
    beta_i: float = Field(gt=0.0, description="Ion plasma beta")
    v_th: float = Field(gt=0.0, description="Electron thermal velocity")
    nu: float = Field(ge=0.0, description="Collision frequency")
    time: float = Field(ge=0.0, description="Simulation time")
    grid: SpectralGrid3D = Field(description="Spectral grid specification")

    @field_validator("phi", "A_parallel", "B_parallel")
    @classmethod
    def validate_field_shape(cls, v: Array, info) -> Array:
        """Validate that fields have correct 3D Fourier space shape."""
        if v.ndim != 3:
            raise ValueError(f"Field {info.field_name} must be 3D, got shape {v.shape}")
        return v

    @field_validator("g")
    @classmethod
    def validate_hermite_shape(cls, v: Array) -> Array:
        """Validate that Hermite moments have correct 4D shape and dtype."""
        if v.ndim != 4:
            raise ValueError(f"Hermite moments must be 4D [Nz, Ny, Nx//2+1, M+1], got shape {v.shape}")
        if not jnp.iscomplexobj(v):
            raise ValueError("Hermite moments must be complex-valued in Fourier space")
        return v


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


# =============================================================================
# State Initialization Functions
# =============================================================================


def initialize_hermite_moments(
    grid: SpectralGrid3D,
    M: int,
    v_th: float = 1.0,
    perturbation_amplitude: float = 0.0,
    seed: int = 42,
) -> Array:
    """
    Initialize Hermite moments for electron distribution function.

    Creates initial condition for Hermite moment array g with:
    - g_0 = equilibrium (Maxwellian, corresponding to constant in velocity space)
    - g_m>0 = small perturbations (for kinetic instability studies)

    The equilibrium Maxwellian corresponds to g_0 = constant and g_m>0 = 0.
    The zeroth moment g_0 represents the density perturbation.

    Args:
        grid: SpectralGrid3D defining spatial dimensions
        M: Number of Hermite moments (g_0, g_1, ..., g_M)
        v_th: Electron thermal velocity (default: 1.0)
        perturbation_amplitude: Amplitude of higher moment perturbations (default: 0.0)
        seed: Random seed for perturbations (default: 42, for reproducibility)

    Returns:
        Hermite moment array g (shape: [Nz, Ny, Nx//2+1, M+1])
        All modes initialized to near-Maxwellian with small perturbations

    Example:
        >>> grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)
        >>> g = initialize_hermite_moments(grid, M=20, v_th=1.0, perturbation_amplitude=0.01)
        >>> g.shape
        (64, 64, 33, 21)

    Note:
        For pure Maxwellian initial condition, set perturbation_amplitude=0.0.
        For kinetic instability studies (e.g., Landau damping tests), use small
        non-zero perturbation_amplitude to seed higher moments.
        Seed parameter ensures reproducible perturbations for testing.
    """
    shape = (grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1)

    # Initialize all moments to zero in Fourier space
    g = jnp.zeros(shape, dtype=jnp.complex64)

    # Add small perturbations to higher moments if requested
    # In practice, this would involve setting specific k modes
    # For now, return equilibrium (all zeros in Fourier space)
    # Real perturbations would be added based on physics (e.g., for Landau damping test)
    if perturbation_amplitude > 0:
        # Add small random perturbations to g_1 mode (velocity perturbation)
        # Must satisfy reality condition: f(-k) = f*(k) for real fields
        key = jax.random.PRNGKey(seed)

        # Generate random real-space field, then FFT to ensure reality condition
        key_real, key_imag = jax.random.split(key)
        perturbation_real = perturbation_amplitude * jax.random.normal(
            key_real, shape=(grid.Nz, grid.Ny, grid.Nx), dtype=jnp.float32
        )

        # Transform to Fourier space - rfftn automatically enforces reality condition
        perturbation_fourier = rfftn_forward(perturbation_real)

        # Apply perturbation to first moment (velocity perturbation)
        g = g.at[:, :, :, 1].set(perturbation_fourier)

    return g


def initialize_alfven_wave(
    grid: SpectralGrid3D,
    M: int,
    kx_mode: float = 1.0,
    ky_mode: float = 0.0,
    kz_mode: float = 1.0,
    amplitude: float = 0.1,
    v_th: float = 1.0,
    beta_i: float = 1.0,
    nu: float = 0.01,
) -> KRMHDState:
    """
    Initialize single-mode Alfvén wave for linear physics validation.

    Creates initial condition for Alfvén wave with wavenumber k = (kx, ky, kz).
    The Alfvén wave has the dispersion relation ω² = k∥² v_A² in the MHD limit.

    For KRMHD, the wave satisfies:
    - Perpendicular structure from E×B drift (phi)
    - Parallel structure from magnetic field (A_parallel)
    - Kinetic modifications from Hermite moments (g)

    The wave is initialized with correct phase relationship:
    - phi and A_parallel in quadrature (90° phase difference)
    - B_parallel = 0 (pure Alfvén wave has no compressibility)

    Args:
        grid: SpectralGrid3D defining spatial dimensions
        M: Number of Hermite moments
        kx_mode: Wavenumber in x direction (default: 1.0)
        ky_mode: Wavenumber in y direction (default: 0.0)
        kz_mode: Wavenumber in z (parallel) direction (default: 1.0)
        amplitude: Wave amplitude (default: 0.1, small for linear regime)
        v_th: Electron thermal velocity (default: 1.0)
        beta_i: Ion plasma beta (default: 1.0)
        nu: Collision frequency (default: 0.01)

    Returns:
        KRMHDState with Alfvén wave initial condition

    Example:
        >>> grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)
        >>> state = initialize_alfven_wave(grid, M=20, kz_mode=2.0, amplitude=0.1)
        >>> # Should propagate at Alfvén speed v_A with frequency ω = k∥·v_A

    Physics:
        For validation, measure the wave frequency ω from time evolution and
        compare with theoretical dispersion: ω² = k∥² v_A²
        In the kinetic regime, expect modifications from Landau damping when
        ω/(k∥·v_th) ~ 1.
    """
    # Initialize empty fields
    phi = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64)
    A_parallel = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64)
    B_parallel = jnp.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1), dtype=jnp.complex64)

    # Find indices corresponding to desired wavenumber
    # This is approximate - we pick the closest wavenumber on the grid
    ikx = jnp.argmin(jnp.abs(grid.kx - kx_mode))
    iky = jnp.argmin(jnp.abs(grid.ky - ky_mode))
    ikz = jnp.argmin(jnp.abs(grid.kz - kz_mode))

    # Set single Fourier mode for Alfvén wave
    # Convention: phi and A_parallel in quadrature (phase difference π/2)
    # This gives circularly polarized wave
    phi = phi.at[ikz, iky, ikx].set(amplitude * 1.0j)
    A_parallel = A_parallel.at[ikz, iky, ikx].set(amplitude * 1.0)

    # Initialize Hermite moments (equilibrium + small kinetic response)
    g = initialize_hermite_moments(grid, M, v_th, perturbation_amplitude=0.01 * amplitude)

    return KRMHDState(
        phi=phi,
        A_parallel=A_parallel,
        B_parallel=B_parallel,
        g=g,
        M=M,
        beta_i=beta_i,
        v_th=v_th,
        nu=nu,
        time=0.0,
        grid=grid,
    )


def initialize_kinetic_alfven_wave(
    grid: SpectralGrid3D,
    M: int,
    kx_mode: float = 1.0,
    ky_mode: float = 0.0,
    kz_mode: float = 1.0,
    amplitude: float = 0.1,
    v_th: float = 1.0,
    beta_i: float = 1.0,
    nu: float = 0.01,
) -> KRMHDState:
    """
    Initialize kinetic Alfvén wave with full kinetic response in Hermite moments.

    Similar to initialize_alfven_wave(), but includes proper kinetic response
    in the Hermite moments. The kinetic Alfvén wave (KAW) includes:
    - Finite Larmor radius (FLR) corrections at k⊥ρ_i ~ 1
    - Landau damping when ω/(k∥·v_th) ~ 1
    - Modified dispersion: ω² = k∥² v_A² (1 + k⊥²ρ_s²)

    The Hermite moments are initialized consistently with the wave fields to
    capture the kinetic response from the start.

    Args:
        grid: SpectralGrid3D defining spatial dimensions
        M: Number of Hermite moments
        kx_mode: Wavenumber in x direction (default: 1.0)
        ky_mode: Wavenumber in y direction (default: 0.0)
        kz_mode: Wavenumber in z (parallel) direction (default: 1.0)
        amplitude: Wave amplitude (default: 0.1)
        v_th: Electron thermal velocity (default: 1.0)
        beta_i: Ion plasma beta (default: 1.0)
        nu: Collision frequency (default: 0.01)

    Returns:
        KRMHDState with kinetic Alfvén wave initial condition

    Example:
        >>> grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)
        >>> state = initialize_kinetic_alfven_wave(
        ...     grid, M=30, kx_mode=2.0, kz_mode=1.0, v_th=1.0, beta_i=1.0
        ... )
        >>> # Should show Landau damping for appropriate parameters

    Physics:
        This initialization is for testing kinetic effects:
        - Use when k⊥ρ_s ~ 1 (kinetic regime)
        - Compare with fluid Alfvén wave to measure kinetic corrections
        - Validate Landau damping rate against linear theory
    """
    # Start with fluid Alfvén wave
    state = initialize_alfven_wave(
        grid, M, kx_mode, ky_mode, kz_mode, amplitude, v_th, beta_i, nu
    )

    # TODO: Add proper kinetic response to Hermite moments
    # This requires solving the linearized kinetic equation for the wave
    # For now, we use the fluid initialization with small moment perturbations
    # Future: Implement analytic kinetic Alfvén wave solution from linear theory

    return state


def initialize_random_spectrum(
    grid: SpectralGrid3D,
    M: int,
    alpha: float = 5.0 / 3.0,
    amplitude: float = 1.0,
    k_min: float = 1.0,
    k_max: Optional[float] = None,
    v_th: float = 1.0,
    beta_i: float = 1.0,
    nu: float = 0.01,
    seed: int = 42,
) -> KRMHDState:
    """
    Initialize turbulent spectrum with power law k^(-α) for decaying turbulence.

    Creates initial condition with random phases and specified energy spectrum.
    The energy spectrum follows E(k) ∝ k^(-α) where α is the spectral index:
    - α = 5/3: Kolmogorov spectrum (isotropic 3D turbulence)
    - α = 3/2: Kraichnan spectrum (2D turbulence or weak turbulence)
    - α = 2: Steep spectrum (viscous range)

    The initialization ensures:
    - Reality condition: f(-k) = f*(k)
    - Divergence-free magnetic field: ∇·B = 0
    - Random phases for statistical homogeneity
    - Energy concentrated in specified k-range [k_min, k_max]

    Args:
        grid: SpectralGrid3D defining spatial dimensions
        M: Number of Hermite moments
        alpha: Spectral index for E(k) ∝ k^(-α) (default: 5/3, Kolmogorov)
        amplitude: Overall amplitude scale (default: 1.0)
        k_min: Minimum wavenumber for energy injection (default: 1.0)
        k_max: Maximum wavenumber for energy injection (default: None, uses k_max/3 for dealiasing)
        v_th: Electron thermal velocity (default: 1.0)
        beta_i: Ion plasma beta (default: 1.0)
        nu: Collision frequency (default: 0.01)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        KRMHDState with random turbulent spectrum

    Example:
        >>> grid = SpectralGrid3D.create(Nx=128, Ny=128, Nz=128)
        >>> state = initialize_random_spectrum(
        ...     grid, M=20, alpha=5/3, amplitude=1.0, k_min=2.0, k_max=20.0
        ... )
        >>> # Let it evolve to study decaying turbulence cascade

    Physics:
        This initialization is for studying:
        - Decaying turbulence (no forcing)
        - Cascade dynamics and energy transfer
        - Approach to steady-state spectrum
        - Selective decay (magnetic energy dominance)

        For forced turbulence, add forcing term during time evolution (separate function).
    """
    key = jax.random.PRNGKey(seed)

    # If k_max not specified, use 2/3 of Nyquist for safety
    if k_max is None:
        k_max = 2.0 / 3.0 * jnp.max(grid.kx)

    # Create k-space grids
    kx_3d = grid.kx[jnp.newaxis, jnp.newaxis, :]
    ky_3d = grid.ky[jnp.newaxis, :, jnp.newaxis]
    kz_3d = grid.kz[:, jnp.newaxis, jnp.newaxis]

    # Compute |k| for each mode
    k_mag = jnp.sqrt(kx_3d**2 + ky_3d**2 + kz_3d**2)

    # Power spectrum: E(k) ∝ k^(-α)
    # Amplitude tapers smoothly from k_min to k_max
    spectrum = jnp.where(
        (k_mag >= k_min) & (k_mag <= k_max),
        amplitude * k_mag ** (-alpha / 2.0),  # sqrt because E ∝ |field|²
        0.0,
    )

    # Generate random phases
    key, subkey1, subkey2 = jax.random.split(key, 3)
    phase_phi = 2.0 * jnp.pi * jax.random.uniform(subkey1, shape=(grid.Nz, grid.Ny, grid.Nx // 2 + 1))
    phase_A = 2.0 * jnp.pi * jax.random.uniform(subkey2, shape=(grid.Nz, grid.Ny, grid.Nx // 2 + 1))

    # Create fields with random phases and power-law spectrum
    phi = spectrum * jnp.exp(1.0j * phase_phi).astype(jnp.complex64)
    A_parallel = spectrum * jnp.exp(1.0j * phase_A).astype(jnp.complex64)

    # Zero out k=0 mode (no DC component)
    phi = phi.at[0, 0, 0].set(0.0)
    A_parallel = A_parallel.at[0, 0, 0].set(0.0)

    # Passive scalar B_parallel (initially zero, will be excited by cascade)
    B_parallel = jnp.zeros_like(phi)

    # Initialize Hermite moments (equilibrium)
    g = initialize_hermite_moments(grid, M, v_th, perturbation_amplitude=0.0)

    return KRMHDState(
        phi=phi,
        A_parallel=A_parallel,
        B_parallel=B_parallel,
        g=g,
        M=M,
        beta_i=beta_i,
        v_th=v_th,
        nu=nu,
        time=0.0,
        grid=grid,
    )


# =============================================================================
# Energy Diagnostics
# =============================================================================


def energy(state: KRMHDState) -> Dict[str, float]:
    """
    Compute total energy and components for KRMHD state.

    Calculates energy contributions from:
    - Magnetic energy: E_mag = (1/2) ∫ |B⊥|² dx = (1/2) ∫ |∇A∥|² dx
    - Kinetic energy: E_kin = (1/2) ∫ |v⊥|² dx = (1/2) ∫ |∇φ|² dx
    - Compressive energy: E_comp = (1/2) ∫ |δB∥|² dx

    All energies are computed in Fourier space using Parseval's theorem:
        ∫ |f(x)|² dx = ∫ |f̂(k)|² dk

    Args:
        state: KRMHDState containing all fields

    Returns:
        Dictionary with energy components:
        - 'magnetic': Magnetic energy from perpendicular field
        - 'kinetic': Kinetic energy from perpendicular flow
        - 'compressive': Compressive energy from parallel field
        - 'total': Sum of all components

    Example:
        >>> grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)
        >>> state = initialize_alfven_wave(grid, M=20, amplitude=0.1)
        >>> E = energy(state)
        >>> print(f"Total energy: {E['total']:.6f}")
        >>> print(f"Magnetic fraction: {E['magnetic']/E['total']:.3f}")

    Physics:
        - For Alfvén waves: E_mag ≈ E_kin (equipartition)
        - For decaying turbulence: E_mag/E_kin increases (selective decay)
        - E_comp should remain small (passive, no back-reaction)

        Energy should be conserved in inviscid (nu=eta=0) simulations to
        within numerical precision (~1e-10 relative error).
    """
    grid = state.grid

    # Parseval's theorem for rfft: ∫ |f(x)|² dx = (1/N) Σ_k |f̂(k)|²
    # where N = Nx * Ny * Nz is the total number of grid points
    #
    # For rfft2/rfftn, we only store positive kx frequencies, so we need to:
    # 1. Double-count all modes except kx=0 (accounts for negative kx)
    # 2. Normalize by 1/N to match real-space integral
    #
    # The physical volume (Lx * Ly * Lz) cancels out because:
    # - Real space: ∫ |f|² dx has units [f²·volume]
    # - Fourier space: Σ |f̂|² has units [f²·volume/N]
    # - Factor of N makes them match

    Nx, Ny, Nz = grid.Nx, grid.Ny, grid.Nz
    N_total = Nx * Ny * Nz
    norm_factor = 1.0 / N_total

    # For rfft, we need to handle three cases:
    # - kx = 0: No negative counterpart, factor = 1
    # - 0 < kx < Nx/2: Has negative counterpart, factor = 2
    # - kx = Nx/2 (Nyquist, only if Nx even): Real-valued, factor = 1
    #
    # Create masks for each case
    kx_3d = grid.kx[jnp.newaxis, jnp.newaxis, :]
    ky_3d = grid.ky[jnp.newaxis, :, jnp.newaxis]

    kx_zero = (kx_3d == 0.0)
    kx_nyquist = (kx_3d == Nx // 2) if (Nx % 2 == 0) else jnp.zeros_like(kx_3d, dtype=bool)
    kx_middle = ~(kx_zero | kx_nyquist)  # All other positive kx values

    # Compute k² for gradient energy
    k_perp_squared = kx_3d**2 + ky_3d**2

    # Magnetic energy: E_mag = (1/2) ∫ |∇⊥A∥|² dx
    # Apply correct factors: 1 for kx=0 and Nyquist, 2 for middle modes
    A_mag_squared = k_perp_squared * jnp.abs(state.A_parallel) ** 2
    E_magnetic = 0.5 * norm_factor * (
        jnp.sum(jnp.where(kx_middle, 2.0 * A_mag_squared, A_mag_squared))
    ).real

    # Kinetic energy: E_kin = (1/2) ∫ |∇⊥φ|² dx
    phi_mag_squared = k_perp_squared * jnp.abs(state.phi) ** 2
    E_kinetic = 0.5 * norm_factor * (
        jnp.sum(jnp.where(kx_middle, 2.0 * phi_mag_squared, phi_mag_squared))
    ).real

    # Compressive energy: E_comp = (1/2) ∫ |δB∥|² dx
    B_mag_squared = jnp.abs(state.B_parallel) ** 2
    E_compressive = 0.5 * norm_factor * (
        jnp.sum(jnp.where(kx_middle, 2.0 * B_mag_squared, B_mag_squared))
    ).real

    # Total energy
    E_total = E_magnetic + E_kinetic + E_compressive

    # NOTE: We're not including kinetic (Hermite moment) energy here yet
    # That would require integrating over velocity space as well
    # For now, focus on fluid energy components

    return {
        "magnetic": float(E_magnetic),
        "kinetic": float(E_kinetic),
        "compressive": float(E_compressive),
        "total": float(E_total),
    }
