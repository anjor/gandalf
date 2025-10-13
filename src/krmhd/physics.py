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
    derivative_z,
    laplacian,
    rfft2_forward,
    rfft2_inverse,
    rfftn_forward,
    rfftn_inverse,
    dealias,
)


class KRMHDState(BaseModel):
    """
    Complete KRMHD state with Elsasser variables and kinetic fields.

    This dataclass represents the full state of the Kinetic Reduced MHD system,
    using Elsasser variables (z⁺, z⁻) for the Alfvénic sector and Hermite moments
    for the kinetic electron distribution function.

    All field arrays are stored in Fourier space for efficient spectral operations.
    The fields satisfy:
    - Reality condition: f(-k) = f*(k)
    - Divergence-free magnetic field: ∇·B = 0 (automatically satisfied)

    Attributes:
        z_plus: Elsasser z⁺ = φ + A∥ in Fourier space (shape: [Nz, Ny, Nx//2+1])
            Co-propagating Alfvén wave packet along +B₀
        z_minus: Elsasser z⁻ = φ - A∥ in Fourier space (shape: [Nz, Ny, Nx//2+1])
            Counter-propagating Alfvén wave packet along -B₀
        B_parallel: Parallel magnetic field δB∥ (passive) in Fourier space (shape: [Nz, Ny, Nx//2+1])
        g: Hermite moments of electron distribution (shape: [Nz, Ny, Nx//2+1, M+1])
            Expansion: g(v∥) = Σ_m g_m · ψ_m(v∥/v_th)
        M: Number of Hermite moments (typically 20-30 for converged kinetics)
        beta_i: Ion plasma beta β_i = 8πn_i T_i / B₀²
        v_th: Electron thermal velocity v_th = √(T_e/m_e)
        nu: Collision frequency ν (Lenard-Bernstein operator)
        time: Simulation time
        grid: Reference to SpectralGrid3D for spatial dimensions

    Properties (computed on demand):
        phi: Stream function φ = (z⁺ + z⁻)/2
            Generates perpendicular velocity: v⊥ = ẑ × ∇φ
        A_parallel: Parallel vector potential A∥ = (z⁺ - z⁻)/2
            Generates perpendicular magnetic field: B⊥ = ẑ × ∇A∥

    Example:
        >>> grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)
        >>> state = KRMHDState(
        ...     z_plus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1), dtype=complex),
        ...     z_minus=jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1), dtype=complex),
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
        The KRMHD equations in Elsasser form:
        - ∂z⁺/∂t = -∇²⊥{z⁻, z⁺} - ∇∥z⁻ + dissipation
        - ∂z⁻/∂t = -∇²⊥{z⁺, z⁻} + ∇∥z⁺ + dissipation
        - Passive sector: B∥ is advected by φ = (z⁺ + z⁻)/2
        - Kinetic sector: g moments evolve with Landau damping and collisions

        The Elsasser formulation simplifies the Alfvén wave dynamics compared
        to the coupled (φ, A∥) formulation used in earlier versions.

    Reference:
        - Elsasser (1950) Phys. Rev. 79:183 - Original Elsasser variables
        - Boldyrev (2006) PRL 96:115002 - RMHD turbulence
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    z_plus: Array = Field(description="Elsasser z+ (co-propagating wave) in Fourier space")
    z_minus: Array = Field(description="Elsasser z- (counter-propagating wave) in Fourier space")
    B_parallel: Array = Field(description="Parallel magnetic field in Fourier space")
    g: Array = Field(description="Hermite moments of electron distribution")
    M: int = Field(gt=0, description="Number of Hermite moments")
    beta_i: float = Field(gt=0.0, description="Ion plasma beta")
    v_th: float = Field(gt=0.0, description="Electron thermal velocity")
    nu: float = Field(ge=0.0, description="Collision frequency")
    time: float = Field(ge=0.0, description="Simulation time")
    grid: SpectralGrid3D = Field(description="Spectral grid specification")

    @field_validator("z_plus", "z_minus", "B_parallel")
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

    @property
    def phi(self) -> Array:
        """
        Stream function φ = (z⁺ + z⁻) / 2.

        Generates perpendicular velocity: v⊥ = ẑ × ∇φ

        This is a computed property for backward compatibility and diagnostics.
        The actual stored variables are z_plus and z_minus.

        Returns:
            Stream function in Fourier space (shape: [Nz, Ny, Nx//2+1])
        """
        return (self.z_plus + self.z_minus) / 2.0

    @property
    def A_parallel(self) -> Array:
        """
        Parallel vector potential A∥ = (z⁺ - z⁻) / 2.

        Generates perpendicular magnetic field: B⊥ = ẑ × ∇A∥

        This is a computed property for backward compatibility and diagnostics.
        The actual stored variables are z_plus and z_minus.

        Returns:
            Parallel vector potential in Fourier space (shape: [Nz, Ny, Nx//2+1])
        """
        return (self.z_plus - self.z_minus) / 2.0

    def __repr__(self) -> str:
        """
        Compact string representation for debugging.

        Shows key simulation parameters and state information without
        printing large arrays.

        Returns:
            String with time, grid size, and energies
        """
        # Import energy calculation (defined later in this file)
        from krmhd.physics import energy

        E_mag, E_kin, E_comp = energy(self)
        E_tot = E_mag + E_kin + E_comp

        return (
            f"KRMHDState(t={self.time:.3f}, "
            f"grid={self.grid.Nx}×{self.grid.Ny}×{self.grid.Nz}, "
            f"E_tot={E_tot:.3e}, E_mag={E_mag:.3e}, E_kin={E_kin:.3e}, "
            f"M={self.M}, β_i={self.beta_i:.2f})"
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


# =============================================================================
# Elsasser Variable Conversions
# =============================================================================


@jax.jit
def physical_to_elsasser(phi: Array, A_parallel: Array) -> tuple[Array, Array]:
    """
    Convert from physical variables (φ, A∥) to Elsasser variables (z⁺, z⁻).

    The Elsasser variables represent counter-propagating Alfvén wave packets:
    - z⁺ = φ + A∥  (co-propagating wave along +B₀)
    - z⁻ = φ - A∥  (counter-propagating wave along -B₀)

    This transformation diagonalizes the linear Alfvén wave dynamics and
    simplifies the nonlinear evolution equations in RMHD.

    Args:
        phi: Stream function φ in Fourier space (shape: [..., Ny, Nx//2+1])
            Generates perpendicular velocity: v⊥ = ẑ × ∇φ
        A_parallel: Parallel vector potential A∥ in Fourier space (same shape as phi)
            Generates perpendicular magnetic field: B⊥ = ẑ × ∇A∥

    Returns:
        Tuple of (z_plus, z_minus) in Fourier space with same shape as inputs

    Properties:
        - Linearity: The transformation is linear
        - Invertibility: Can recover (φ, A∥) via elsasser_to_physical()
        - Wave interpretation: z⁺ and z⁻ represent independent wave modes
        - Energy: E_total = (1/4)∫(|∇z⁺|² + |∇z⁻|²)dx

    Example:
        >>> grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)
        >>> phi = jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1), dtype=complex)
        >>> A_parallel = jnp.zeros_like(phi)
        >>> z_plus, z_minus = physical_to_elsasser(phi, A_parallel)
        >>> # z_plus and z_minus now contain the Elsasser fields

    Physics context:
        In the Elsasser formulation, the RMHD equations become:
        - ∂z⁺/∂t = -∇²⊥{z⁻, z⁺} - ∇∥z⁻ + dissipation
        - ∂z⁻/∂t = -∇²⊥{z⁺, z⁻} + ∇∥z⁺ + dissipation

        Notice that z⁺ is advected by z⁻ (and vice versa), representing
        wave packet interactions. This is more natural than the coupled
        (φ, A∥) formulation.

    Reference:
        - Elsasser (1950) Phys. Rev. 79:183 - Original Elsasser variables
        - Boldyrev (2006) PRL 96:115002 - Application to RMHD turbulence
    """
    # Defensive check: ensure shapes match
    assert phi.shape == A_parallel.shape, \
        f"Shape mismatch: phi {phi.shape} != A_parallel {A_parallel.shape}"

    z_plus = phi + A_parallel
    z_minus = phi - A_parallel
    return z_plus, z_minus


@jax.jit
def elsasser_to_physical(z_plus: Array, z_minus: Array) -> tuple[Array, Array]:
    """
    Convert from Elsasser variables (z⁺, z⁻) to physical variables (φ, A∥).

    This is the inverse transformation of physical_to_elsasser(), recovering
    the stream function φ and parallel vector potential A∥ from the Elsasser
    wave variables.

    Args:
        z_plus: Elsasser z⁺ variable in Fourier space (shape: [..., Ny, Nx//2+1])
            Represents co-propagating Alfvén wave packet along +B₀
        z_minus: Elsasser z⁻ variable in Fourier space (same shape as z_plus)
            Represents counter-propagating wave packet along -B₀

    Returns:
        Tuple of (phi, A_parallel) in Fourier space with same shape as inputs

    Relationships:
        - φ = (z⁺ + z⁻) / 2    (stream function, generates v⊥)
        - A∥ = (z⁺ - z⁻) / 2   (parallel vector potential, generates B⊥)

    Properties:
        - Linearity: The transformation is linear
        - Invertibility: Round-trip via physical_to_elsasser() is exact
        - Perpendicular fields:
          - v⊥ = ẑ × ∇φ = ẑ × ∇[(z⁺ + z⁻)/2]
          - B⊥ = ẑ × ∇A∥ = ẑ × ∇[(z⁺ - z⁻)/2]

    Example:
        >>> grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)
        >>> z_plus = jnp.ones((grid.Nz, grid.Ny, grid.Nx//2+1), dtype=complex)
        >>> z_minus = jnp.zeros_like(z_plus)
        >>> phi, A_parallel = elsasser_to_physical(z_plus, z_minus)
        >>> # phi = 0.5, A_parallel = 0.5 (single Elsasser mode → equal φ and A∥)

    Round-trip example:
        >>> phi_orig = jnp.ones((grid.Nz, grid.Ny, grid.Nx//2+1), dtype=complex)
        >>> A_orig = jnp.zeros_like(phi_orig)
        >>> z_p, z_m = physical_to_elsasser(phi_orig, A_orig)
        >>> phi_back, A_back = elsasser_to_physical(z_p, z_m)
        >>> assert jnp.allclose(phi_orig, phi_back)
        >>> assert jnp.allclose(A_orig, A_back)

    Use cases:
        - Convert Elsasser state back to physical fields for diagnostics
        - Initialize fields in physical space, then convert to Elsasser
        - Compute derived quantities (energy, spectra) that use φ and A∥
    """
    # Defensive check: ensure shapes match
    assert z_plus.shape == z_minus.shape, \
        f"Shape mismatch: z_plus {z_plus.shape} != z_minus {z_minus.shape}"

    phi = (z_plus + z_minus) / 2.0
    A_parallel = (z_plus - z_minus) / 2.0
    return phi, A_parallel


# =============================================================================
# Helper Functions
# =============================================================================


@jax.jit
def zero_k0_mode(field: Array) -> Array:
    """
    Zero out the k=0 mode (mean field) in Fourier space.

    The k=0 mode represents the spatial mean of a field. In RMHD, we typically
    want to suppress mean field evolution to focus on fluctuations. This is
    especially important for numerical stability over long integrations.

    Args:
        field: Field in Fourier space (shape: [..., Nz, Ny, Nx//2+1])
            The k=0 mode is at index [0, 0, 0]

    Returns:
        Field with k=0 mode set to exactly zero

    Note:
        This operation is defensive - initialization should already ensure k=0
        is zero, but numerical round-off could cause drift over time.
    """
    return field.at[0, 0, 0].set(0.0 + 0.0j)


# =============================================================================
# Elsasser RHS Functions (Time Evolution)
# =============================================================================


@partial(jax.jit, static_argnums=(7, 8, 9))
def z_plus_rhs(
    z_plus: Array,
    z_minus: Array,
    kx: Array,
    ky: Array,
    kz: Array,
    dealias_mask: Array,
    eta: float,
    Nz: int,
    Ny: int,
    Nx: int,
) -> Array:
    """
    Compute RHS for z⁺ evolution: ∂z⁺/∂t = -∇²⊥{z⁻, z⁺} - ∇∥z⁻ + η∇²z⁺.

    This implements the Elsasser formulation of RMHD for the co-propagating
    wave packet z⁺ = φ + A∥. The equation shows that z⁺ is:
    - Advected by z⁻ through the perpendicular Poisson bracket
    - Coupled to z⁻ through the parallel gradient ∇∥z⁻
    - Dissipated by resistivity η

    **Why ∇²⊥ appears in the Poisson bracket:**
    In RMHD, we evolve vorticity ω = ∇²⊥φ, not φ itself. The vorticity
    equation is: ∂ω/∂t + {φ, ω} = .... Substituting ω = ∇²⊥φ gives
    ∂(∇²⊥φ)/∂t + {φ, ∇²⊥φ} = ..., which becomes ∂z⁺/∂t = ...∇²⊥{z⁻, z⁺}...
    in Elsasser variables. This is why the Laplacian appears inside the
    Poisson bracket - it represents advection of vorticity, not of φ itself.

    Args:
        z_plus: Elsasser z⁺ in Fourier space (shape: [Nz, Ny, Nx//2+1])
        z_minus: Elsasser z⁻ in Fourier space (shape: [Nz, Ny, Nx//2+1])
        kx: Wavenumber array in x direction
        ky: Wavenumber array in y direction
        kz: Wavenumber array in z direction
        dealias_mask: 2/3 dealiasing mask
        eta: Resistivity coefficient
        Nz: Number of grid points in z direction (static)
        Ny: Number of grid points in y direction (static)
        Nx: Number of grid points in x direction (static)

    Returns:
        Time derivative ∂z⁺/∂t in Fourier space (shape: [Nz, Ny, Nx//2+1])

    Physics:
        The Elsasser equation for z⁺ in RMHD is:
        ∂z⁺/∂t + (z⁻·∇)z⁺ + ∇P = η∇²z⁺

        In the reduced MHD approximation with strong guide field B₀ẑ,
        the perpendicular advection (z⁻·∇)z⁺ splits into:
        - Perpendicular part: ∇²⊥{z⁻, z⁺} (vorticity advection)
        - Parallel part: ∂∥z⁻ (couples to opposite Elsasser variable)

        This gives:
        ∂z⁺/∂t = -∇²⊥{z⁻, z⁺} - ∂∥z⁻ + η∇²z⁺

        where ∂∥ = ∂/∂z is the parallel derivative. Note that z⁺ is coupled
        to the parallel gradient of z⁻, not its own gradient.

    Example:
        >>> grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)
        >>> z_plus = jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1), dtype=complex)
        >>> z_minus = jnp.zeros_like(z_plus)
        >>> dz_plus_dt = z_plus_rhs(z_plus, z_minus, grid.kx, grid.ky, grid.kz,
        ...                         grid.dealias_mask, 0.01, grid.Nz, grid.Ny, grid.Nx)

    Reference:
        - Boldyrev (2006) PRL 96:115002 - Elsasser formulation for RMHD
        - Maron & Goldreich (2001) ApJ 554:1175 - Reduced MHD equations
    """
    # Compute Poisson bracket: {z⁻, z⁺}
    # This represents the nonlinear advection
    bracket = poisson_bracket_3d(
        z_minus, z_plus,
        kx, ky,
        Nz, Ny, Nx,
        dealias_mask
    )

    # Apply perpendicular Laplacian to the bracket: ∇²⊥{z⁻, z⁺}
    # This gives the vorticity advection term
    bracket_with_lap = laplacian(bracket, kx, ky, kz=None)

    # Compute parallel derivative: ∇∥z⁻ = ∂z⁻/∂z
    parallel_grad_z_minus = derivative_z(z_minus, kz)

    # Assemble RHS: -∇²⊥{z⁻, z⁺} - ∇∥z⁻
    rhs = -bracket_with_lap - parallel_grad_z_minus

    # Add dissipation: η∇²z⁺ (always compute, multiply by eta)
    lap_z_plus = laplacian(z_plus, kx, ky, kz)
    rhs = rhs + eta * lap_z_plus

    # Zero out k=0 mode (mean field should not evolve)
    rhs = zero_k0_mode(rhs)

    return rhs


@partial(jax.jit, static_argnums=(7, 8, 9))
def z_minus_rhs(
    z_plus: Array,
    z_minus: Array,
    kx: Array,
    ky: Array,
    kz: Array,
    dealias_mask: Array,
    eta: float,
    Nz: int,
    Ny: int,
    Nx: int,
) -> Array:
    """
    Compute RHS for z⁻ evolution: ∂z⁻/∂t = -∇²⊥{z⁺, z⁻} + ∇∥z⁺ + η∇²z⁻.

    This implements the Elsasser formulation of RMHD for the counter-propagating
    wave packet z⁻ = φ - A∥. The equation shows that z⁻ is:
    - Advected by z⁺ through the perpendicular Poisson bracket
    - Coupled to z⁺ through the parallel gradient ∇∥z⁺ (opposite sign from z⁺)
    - Dissipated by resistivity η

    The key difference from z_plus_rhs is the sign of the parallel gradient term
    (+∂∥z⁺ vs -∂∥z⁻), reflecting the opposite propagation direction along B₀.

    **Vorticity formulation:** See z_plus_rhs docstring for explanation of why
    ∇²⊥ appears inside the Poisson bracket.

    Args:
        z_plus: Elsasser z⁺ in Fourier space (shape: [Nz, Ny, Nx//2+1])
        z_minus: Elsasser z⁻ in Fourier space (shape: [Nz, Ny, Nx//2+1])
        kx: Wavenumber array in x direction
        ky: Wavenumber array in y direction
        kz: Wavenumber array in z direction
        dealias_mask: 2/3 dealiasing mask
        eta: Resistivity coefficient
        Nz: Number of grid points in z direction (static)
        Ny: Number of grid points in y direction (static)
        Nx: Number of grid points in x direction (static)

    Returns:
        Time derivative ∂z⁻/∂t in Fourier space (shape: [Nz, Ny, Nx//2+1])

    Physics:
        The Elsasser equation for z⁻ in RMHD is:
        ∂z⁻/∂t + (z⁺·∇)z⁻ + ∇P = η∇²z⁻

        In the reduced MHD approximation with strong guide field B₀ẑ,
        the perpendicular advection (z⁺·∇)z⁻ splits into:
        - Perpendicular part: ∇²⊥{z⁺, z⁻} (vorticity advection)
        - Parallel part: -∂∥z⁺ (couples to opposite Elsasser variable)

        This gives:
        ∂z⁻/∂t = -∇²⊥{z⁺, z⁻} + ∂∥z⁺ + η∇²z⁻

        Note the OPPOSITE sign on the parallel gradient compared to z⁺
        (∂∥z⁺ vs -∂∥z⁻), reflecting counter-propagation. Both equations
        couple to the gradient of the OPPOSITE Elsasser variable.

    Example:
        >>> grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=64)
        >>> z_plus = jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1), dtype=complex)
        >>> z_minus = jnp.zeros_like(z_plus)
        >>> dz_minus_dt = z_minus_rhs(z_plus, z_minus, grid.kx, grid.ky, grid.kz,
        ...                           grid.dealias_mask, 0.01, grid.Nz, grid.Ny, grid.Nx)

    Reference:
        - Boldyrev (2006) PRL 96:115002 - Elsasser formulation for RMHD
        - Maron & Goldreich (2001) ApJ 554:1175 - Reduced MHD equations
    """
    # Compute Poisson bracket: {z⁺, z⁻}
    # This represents the nonlinear advection
    bracket = poisson_bracket_3d(
        z_plus, z_minus,
        kx, ky,
        Nz, Ny, Nx,
        dealias_mask
    )

    # Apply perpendicular Laplacian to the bracket: ∇²⊥{z⁺, z⁻}
    # This gives the vorticity advection term
    bracket_with_lap = laplacian(bracket, kx, ky, kz=None)

    # Compute parallel derivative: ∇∥z⁺ = ∂z⁺/∂z
    parallel_grad_z_plus = derivative_z(z_plus, kz)

    # Assemble RHS: -∇²⊥{z⁺, z⁻} + ∇∥z⁺
    # NOTE: Opposite sign on parallel gradient compared to z_plus_rhs
    rhs = -bracket_with_lap + parallel_grad_z_plus

    # Add dissipation: η∇²z⁻ (always compute, multiply by eta)
    lap_z_minus = laplacian(z_minus, kx, ky, kz)
    rhs = rhs + eta * lap_z_minus

    # Zero out k=0 mode (mean field should not evolve)
    rhs = zero_k0_mode(rhs)

    return rhs


# =============================================================================
# GANDALF-style Elsasser RHS (Energy-Conserving Formulation)
# =============================================================================


@partial(jax.jit, static_argnums=(7, 8, 9))
def z_plus_rhs_gandalf(
    z_plus: Array,
    z_minus: Array,
    kx: Array,
    ky: Array,
    kz: Array,
    dealias_mask: Array,
    eta: float,
    Nz: int,
    Ny: int,
    Nx: int,
) -> Array:
    """
    Compute RHS for z⁺ using GANDALF's energy-conserving formulation.

    This matches the original GANDALF Fortran/CUDA implementation from nonlin.cu:
        bracket1 = {z⁺, -k⊥²z⁻} + {z⁻, -k⊥²z⁺}
        bracket2 = -k⊥²{z⁺, z⁻}
        RHS_nonlinear = k⊥²^(-1) × [bracket1 - bracket2]

    Which expands to:
        ∂z⁺/∂t = k⊥²^(-1)[{z⁺, -k⊥²z⁻} + {z⁻, -k⊥²z⁺} + k⊥²{z⁺, z⁻}] - ∂∥z⁻ + η∇²z⁺

    This formulation exactly conserves perpendicular gradient energy:
        E = (1/4) ∫ (|∇⊥z⁺|² + |∇⊥z⁻|²) dx

    Args:
        z_plus: Elsasser z⁺ in Fourier space
        z_minus: Elsasser z⁻ in Fourier space
        kx, ky, kz: Wavenumber arrays
        dealias_mask: 2/3 dealiasing mask
        eta: Resistivity
        Nz, Ny, Nx: Grid dimensions (static)

    Returns:
        Time derivative ∂z⁺/∂t in Fourier space

    Reference:
        Original GANDALF: nonlin.cu and timestep.cu (alf_adv function)
    """
    # Compute perpendicular Laplacian terms
    lap_perp_z_plus = laplacian(z_plus, kx, ky, kz=None)   # -k⊥²z⁺
    lap_perp_z_minus = laplacian(z_minus, kx, ky, kz=None)  # -k⊥²z⁻

    # bracket1 = {z⁺, -k⊥²z⁻} + {z⁻, -k⊥²z⁺}
    bracket1a = poisson_bracket_3d(z_plus, lap_perp_z_minus, kx, ky, Nz, Ny, Nx, dealias_mask)
    bracket1b = poisson_bracket_3d(z_minus, lap_perp_z_plus, kx, ky, Nz, Ny, Nx, dealias_mask)
    bracket1 = bracket1a + bracket1b

    # bracket2 = -k⊥²{z⁺, z⁻}
    bracket_zm_zp = poisson_bracket_3d(z_plus, z_minus, kx, ky, Nz, Ny, Nx, dealias_mask)
    bracket2 = laplacian(bracket_zm_zp, kx, ky, kz=None)  # -k⊥²{z⁺,z⁻}

    # Compute: bracket1 - bracket2
    combined_bracket = bracket1 - bracket2

    # Apply inverse Laplacian: k⊥²^(-1) × [bracket1 - bracket2]
    # In Fourier space: multiply by -1/k⊥² (negative because laplacian returns -k⊥²f)
    kx_3d = kx[jnp.newaxis, jnp.newaxis, :]
    ky_3d = ky[jnp.newaxis, :, jnp.newaxis]
    k_perp_squared = kx_3d**2 + ky_3d**2

    # Avoid division by zero at k=0
    k_perp_squared_safe = jnp.where(k_perp_squared == 0, 1.0, k_perp_squared)
    inv_laplacian = combined_bracket / (-k_perp_squared_safe)
    inv_laplacian = jnp.where(k_perp_squared == 0, 0.0 + 0.0j, inv_laplacian)

    # Parallel gradient term: -∂∥z⁻
    parallel_grad_z_minus = derivative_z(z_minus, kz)

    # Assemble RHS
    rhs = -inv_laplacian - parallel_grad_z_minus

    # Add dissipation
    lap_z_plus = laplacian(z_plus, kx, ky, kz)
    rhs = rhs + eta * lap_z_plus

    # Zero k=0 mode
    rhs = zero_k0_mode(rhs)

    return rhs


@partial(jax.jit, static_argnums=(7, 8, 9))
def z_minus_rhs_gandalf(
    z_plus: Array,
    z_minus: Array,
    kx: Array,
    ky: Array,
    kz: Array,
    dealias_mask: Array,
    eta: float,
    Nz: int,
    Ny: int,
    Nx: int,
) -> Array:
    """
    Compute RHS for z⁻ using GANDALF's energy-conserving formulation.

    This matches the original GANDALF Fortran/CUDA implementation.
    For z⁻, the signs differ from z⁺ in the parallel gradient term:
        ∂z⁻/∂t = k⊥²^(-1)[{z⁻, -k⊥²z⁺} + {z⁺, -k⊥²z⁻} + k⊥²{z⁻, z⁺}] + ∂∥z⁺ + η∇²z⁻

    (Note: Opposite sign on parallel gradient compared to z⁺)

    Args:
        z_plus: Elsasser z⁺ in Fourier space
        z_minus: Elsasser z⁻ in Fourier space
        kx, ky, kz: Wavenumber arrays
        dealias_mask: 2/3 dealiasing mask
        eta: Resistivity
        Nz, Ny, Nx: Grid dimensions (static)

    Returns:
        Time derivative ∂z⁻/∂t in Fourier space

    Reference:
        Original GANDALF: nonlin.cu and timestep.cu (alf_adv function)
    """
    # Compute perpendicular Laplacian terms
    lap_perp_z_plus = laplacian(z_plus, kx, ky, kz=None)
    lap_perp_z_minus = laplacian(z_minus, kx, ky, kz=None)

    # bracket1 = {z⁻, -k⊥²z⁺} + {z⁺, -k⊥²z⁻}  (same as z⁺ but roles swapped)
    bracket1a = poisson_bracket_3d(z_minus, lap_perp_z_plus, kx, ky, Nz, Ny, Nx, dealias_mask)
    bracket1b = poisson_bracket_3d(z_plus, lap_perp_z_minus, kx, ky, Nz, Ny, Nx, dealias_mask)
    bracket1 = bracket1a + bracket1b

    # bracket2 = -k⊥²{z⁻, z⁺}
    bracket_zp_zm = poisson_bracket_3d(z_minus, z_plus, kx, ky, Nz, Ny, Nx, dealias_mask)
    bracket2 = laplacian(bracket_zp_zm, kx, ky, kz=None)

    # Compute: bracket1 - bracket2
    combined_bracket = bracket1 - bracket2

    # Apply inverse Laplacian
    kx_3d = kx[jnp.newaxis, jnp.newaxis, :]
    ky_3d = ky[jnp.newaxis, :, jnp.newaxis]
    k_perp_squared = kx_3d**2 + ky_3d**2

    k_perp_squared_safe = jnp.where(k_perp_squared == 0, 1.0, k_perp_squared)
    inv_laplacian = combined_bracket / (-k_perp_squared_safe)
    inv_laplacian = jnp.where(k_perp_squared == 0, 0.0 + 0.0j, inv_laplacian)

    # Parallel gradient term: +∂∥z⁺ (OPPOSITE sign from z_plus_rhs)
    parallel_grad_z_plus = derivative_z(z_plus, kz)

    # Assemble RHS
    rhs = -inv_laplacian + parallel_grad_z_plus  # Note: + sign!

    # Add dissipation
    lap_z_minus = laplacian(z_minus, kx, ky, kz)
    rhs = rhs + eta * lap_z_minus

    # Zero k=0 mode
    rhs = zero_k0_mode(rhs)

    return rhs


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
    In Fourier space, equilibrium means ALL g modes = 0 (no spatial variation).

    Note: g_0 represents density perturbation δn/n, g_1 represents parallel
    velocity perturbation u∥, and higher moments capture kinetic corrections.

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

    # Convert to Elsasser variables
    z_plus, z_minus = physical_to_elsasser(phi, A_parallel)

    # Initialize Hermite moments (equilibrium + small kinetic response)
    g = initialize_hermite_moments(grid, M, v_th, perturbation_amplitude=0.01 * amplitude)

    return KRMHDState(
        z_plus=z_plus,
        z_minus=z_minus,
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

    # TODO(Issue #TBD): Implement proper kinetic Alfvén wave solution
    # Currently uses fluid initialization. Need to add:
    # 1. Solve linearized kinetic equation for wave mode
    # 2. Set g moments from analytic solution: g_m(k) ∝ Z(ω/k∥v_th)
    # 3. Include FLR corrections: dispersion ω² = k∥²v_A²(1 + k⊥²ρ_s²)
    # 4. Validate Landau damping rate against linear theory
    #
    # References:
    # - Howes et al. (2006) ApJ 651:590 - KRMHD linear theory
    # - Schekochihin et al. (2009) ApJS 182:310 - Kinetic cascades
    #
    # For now, this returns fluid wave + small moment perturbations as placeholder.

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

    # Convert to Elsasser variables
    z_plus, z_minus = physical_to_elsasser(phi, A_parallel)

    # Passive scalar B_parallel (initially zero, will be excited by cascade)
    B_parallel = jnp.zeros_like(phi)

    # Initialize Hermite moments (equilibrium)
    g = initialize_hermite_moments(grid, M, v_th, perturbation_amplitude=0.0)

    return KRMHDState(
        z_plus=z_plus,
        z_minus=z_minus,
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
