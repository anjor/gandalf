"""
Time integration for KRMHD simulations.

This module implements time-stepping algorithms for evolving the KRMHD equations:
- RK4 (4th-order Runge-Kutta) for robust, fixed-timestep integration
- CFL condition calculator for numerical stability
- Unified RHS function combining all physics terms

The timestepping operates on KRMHDState objects and preserves:
- Field shapes and reality conditions
- Energy conservation (up to numerical precision and Issue #44)
- JIT compilation for performance

Example usage:
    >>> grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)
    >>> state = initialize_alfven_wave(grid, kz=1, amplitude=0.1)
    >>> dt = compute_cfl_timestep(state, v_A=1.0, cfl_safety=0.3)
    >>> new_state = rk4_step(state, dt, eta=0.01, v_A=1.0)

Physics context:
    The KRMHD equations in Elsasser form use GANDALF's energy-conserving formulation:
    - ∂z⁺/∂t = k⊥²⁻¹[{z⁺, -k⊥²z⁻} + {z⁻, -k⊥²z⁺} + k⊥²{z⁺, z⁻}] - ∇∥z⁻ + η∇²z⁺
    - ∂z⁻/∂t = k⊥²⁻¹[{z⁻, -k⊥²z⁺} + {z⁺, -k⊥²z⁻} + k⊥²{z⁻, z⁺}] + ∇∥z⁺ + η∇²z⁻

    This formulation conserves perpendicular gradient energy to ~0.0086% error.

    Time integration must respect:
    - CFL condition: dt < dx / max(v_A, |v_⊥|)
    - Energy conservation in inviscid limit (η=0)
    - Phase relationships for Alfvén waves

References:
    - Press et al. (2007) "Numerical Recipes" §17.1 - RK methods
    - Courant, Friedrichs, Lewy (1928) - CFL condition
"""

from functools import partial
from typing import Callable, Tuple, NamedTuple
import jax
import jax.numpy as jnp
from jax import Array

from krmhd.physics import KRMHDState, z_plus_rhs, z_minus_rhs
from krmhd.spectral import derivative_x, derivative_y, rfftn_inverse


class KRMHDFields(NamedTuple):
    """
    JAX-compatible lightweight container for KRMHD fields (hot path).

    This is a PyTree-compatible structure for JIT compilation.
    Use this in inner loops; convert to/from KRMHDState at boundaries.

    All fields are in Fourier space with shape [Nz, Ny, Nx//2+1].
    """
    z_plus: Array
    z_minus: Array
    B_parallel: Array
    g: Array  # Shape: [Nz, Ny, Nx//2+1, M+1]
    time: float


def _fields_from_state(state: KRMHDState) -> KRMHDFields:
    """Extract JAX-compatible fields from KRMHDState for hot path."""
    return KRMHDFields(
        z_plus=state.z_plus,
        z_minus=state.z_minus,
        B_parallel=state.B_parallel,
        g=state.g,
        time=state.time,
    )


def _state_from_fields(fields: KRMHDFields, state_template: KRMHDState) -> KRMHDState:
    """Reconstruct KRMHDState from JAX fields (validates at boundary)."""
    return KRMHDState(
        z_plus=fields.z_plus,
        z_minus=fields.z_minus,
        B_parallel=fields.B_parallel,
        g=fields.g,
        M=state_template.M,
        beta_i=state_template.beta_i,
        v_th=state_template.v_th,
        nu=state_template.nu,
        time=fields.time,
        grid=state_template.grid,
    )


@partial(jax.jit, static_argnames=["Nz", "Ny", "Nx"])
def _krmhd_rhs_jit(
    fields: KRMHDFields,
    kx: Array,
    ky: Array,
    kz: Array,
    dealias_mask: Array,
    eta: float,
    v_A: float,
    Nz: int,
    Ny: int,
    Nx: int,
) -> KRMHDFields:
    """
    JIT-compiled RHS function operating on lightweight KRMHDFields.

    This is the hot-path function that gets JIT-compiled for performance.
    All array operations happen here with no Pydantic overhead.
    """
    # Compute Elsasser RHS using GANDALF's energy-conserving formulation
    # (already JIT-compiled in physics.py)
    dz_plus_dt = z_plus_rhs(
        fields.z_plus,
        fields.z_minus,
        kx,
        ky,
        kz,
        dealias_mask,
        eta,
        Nz,
        Ny,
        Nx,
    )

    dz_minus_dt = z_minus_rhs(
        fields.z_plus,
        fields.z_minus,
        kx,
        ky,
        kz,
        dealias_mask,
        eta,
        Nz,
        Ny,
        Nx,
    )

    # Passive scalar B∥ evolution (Issue #7 - not yet implemented)
    dB_parallel_dt = jnp.zeros_like(fields.B_parallel)

    # Hermite moment evolution (Issues #22-24 - not yet implemented)
    dg_dt = jnp.zeros_like(fields.g)

    return KRMHDFields(
        z_plus=dz_plus_dt,
        z_minus=dz_minus_dt,
        B_parallel=dB_parallel_dt,
        g=dg_dt,
        time=0.0,  # Not a derivative
    )


def krmhd_rhs(
    state: KRMHDState,
    eta: float,
    v_A: float,
) -> KRMHDState:
    """
    Compute time derivatives for all KRMHD fields.

    This is a thin wrapper that converts KRMHDState to lightweight KRMHDFields,
    calls the JIT-compiled RHS function, and converts back.

    Currently implements:
    - Elsasser fields: z⁺, z⁻ (Alfvénic sector)
    - B∥: Passive parallel magnetic field (Issue #7 - set to zero for now)
    - g: Hermite moments (Issues #22-24 - set to zero for now)

    Args:
        state: Current KRMHD state with all fields
        eta: Resistivity coefficient for dissipation
        v_A: Alfvén velocity (for normalization)

    Returns:
        KRMHDState with time derivatives (time field set to 0.0)

    Example:
        >>> state = initialize_alfven_wave(grid, M=20, kz_mode=1)
        >>> derivatives = krmhd_rhs(state, eta=0.01, v_A=1.0)

    Performance:
        The inner computation is JIT-compiled via _krmhd_rhs_jit().
        Conversion overhead is minimal (boundary operation only).
    """
    grid = state.grid
    fields = _fields_from_state(state)

    # Call JIT-compiled kernel
    deriv_fields = _krmhd_rhs_jit(
        fields,
        grid.kx,
        grid.ky,
        grid.kz,
        grid.dealias_mask,
        eta,
        v_A,
        grid.Nz,
        grid.Ny,
        grid.Nx,
    )

    # Convert back to KRMHDState (Pydantic validation at boundary)
    return _state_from_fields(deriv_fields, state)


@partial(jax.jit, static_argnames=["Nz", "Ny", "Nx"])
def _rk4_step_jit(
    fields: KRMHDFields,
    dt: float,
    kx: Array,
    ky: Array,
    kz: Array,
    dealias_mask: Array,
    eta: float,
    v_A: float,
    Nz: int,
    Ny: int,
    Nx: int,
) -> KRMHDFields:
    """
    JIT-compiled RK4 kernel operating on lightweight KRMHDFields.

    This is the hot-path function with no Pydantic overhead.
    All 4 stages happen here in JIT-compiled code.

    Memory: Allocates 4 KRMHDFields structures (k1, k2, k3, k4) per call.
    For 256³ grid: ~1.1 GB temporary allocation.
    """
    # Stage 1: k1 = f(y_n)
    k1 = _krmhd_rhs_jit(fields, kx, ky, kz, dealias_mask, eta, v_A, Nz, Ny, Nx)

    # Stage 2: k2 = f(y_n + dt/2 * k1)
    fields_k2 = KRMHDFields(
        z_plus=fields.z_plus + 0.5 * dt * k1.z_plus,
        z_minus=fields.z_minus + 0.5 * dt * k1.z_minus,
        B_parallel=fields.B_parallel + 0.5 * dt * k1.B_parallel,
        g=fields.g + 0.5 * dt * k1.g,
        time=fields.time + 0.5 * dt,
    )
    k2 = _krmhd_rhs_jit(fields_k2, kx, ky, kz, dealias_mask, eta, v_A, Nz, Ny, Nx)

    # Stage 3: k3 = f(y_n + dt/2 * k2)
    fields_k3 = KRMHDFields(
        z_plus=fields.z_plus + 0.5 * dt * k2.z_plus,
        z_minus=fields.z_minus + 0.5 * dt * k2.z_minus,
        B_parallel=fields.B_parallel + 0.5 * dt * k2.B_parallel,
        g=fields.g + 0.5 * dt * k2.g,
        time=fields.time + 0.5 * dt,
    )
    k3 = _krmhd_rhs_jit(fields_k3, kx, ky, kz, dealias_mask, eta, v_A, Nz, Ny, Nx)

    # Stage 4: k4 = f(y_n + dt * k3)
    fields_k4 = KRMHDFields(
        z_plus=fields.z_plus + dt * k3.z_plus,
        z_minus=fields.z_minus + dt * k3.z_minus,
        B_parallel=fields.B_parallel + dt * k3.B_parallel,
        g=fields.g + dt * k3.g,
        time=fields.time + dt,
    )
    k4 = _krmhd_rhs_jit(fields_k4, kx, ky, kz, dealias_mask, eta, v_A, Nz, Ny, Nx)

    # Final update: y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return KRMHDFields(
        z_plus=fields.z_plus + (dt / 6.0) * (k1.z_plus + 2.0 * k2.z_plus + 2.0 * k3.z_plus + k4.z_plus),
        z_minus=fields.z_minus + (dt / 6.0) * (k1.z_minus + 2.0 * k2.z_minus + 2.0 * k3.z_minus + k4.z_minus),
        B_parallel=fields.B_parallel + (dt / 6.0) * (k1.B_parallel + 2.0 * k2.B_parallel + 2.0 * k3.B_parallel + k4.B_parallel),
        g=fields.g + (dt / 6.0) * (k1.g + 2.0 * k2.g + 2.0 * k3.g + k4.g),
        time=fields.time + dt,
    )


def rk4_step(
    state: KRMHDState,
    dt: float,
    eta: float,
    v_A: float,
) -> KRMHDState:
    """
    Advance KRMHD state by one timestep using 4th-order Runge-Kutta.

    The classic RK4 method computes:
        k1 = f(y_n)
        k2 = f(y_n + dt/2 * k1)
        k3 = f(y_n + dt/2 * k2)
        k4 = f(y_n + dt * k3)
        y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    This provides O(dt⁴) local truncation error.

    Args:
        state: Current KRMHD state at time t
        dt: Timestep size (should satisfy CFL condition)
        eta: Resistivity coefficient
        v_A: Alfvén velocity

    Returns:
        New KRMHDState at time t + dt

    Example:
        >>> state_new = rk4_step(state, dt=0.01, eta=0.01, v_A=1.0)
        >>> # Or in a loop:
        >>> for i in range(n_steps):
        ...     state = rk4_step(state, dt, eta, v_A)

    Performance:
        - All 4 RK4 stages run in JIT-compiled code via _rk4_step_jit()
        - Conversion overhead is minimal (boundary operation only)
        - Memory: ~1.1 GB temporary allocation for 256³ grid

    Physics:
        - Preserves symplectic structure of Hamiltonian systems
        - Energy conservation up to O(dt⁴) + Issue #44 drift
        - Reality condition f(-k) = f*(k) preserved automatically
        - Stable for dt < CFL limit
    """
    grid = state.grid
    fields = _fields_from_state(state)

    # Call JIT-compiled RK4 kernel (all 4 stages in compiled code)
    new_fields = _rk4_step_jit(
        fields,
        dt,
        grid.kx,
        grid.ky,
        grid.kz,
        grid.dealias_mask,
        eta,
        v_A,
        grid.Nz,
        grid.Ny,
        grid.Nx,
    )

    # Convert back to KRMHDState (Pydantic validation at boundary)
    return _state_from_fields(new_fields, state)


def compute_cfl_timestep(
    state: KRMHDState,
    v_A: float,
    cfl_safety: float = 0.3,
) -> float:
    """
    Compute maximum stable timestep from CFL condition.

    The CFL (Courant-Friedrichs-Lewy) condition ensures numerical stability
    by requiring that information propagates at most one grid cell per timestep:

        dt ≤ C * min(Δx, Δy, Δz) / max(v_A, |v_⊥|)

    where C is a safety factor (typically 0.1-0.5 for RK4).

    Args:
        state: Current KRMHD state (used to compute max velocity)
        v_A: Alfvén velocity (parallel wave speed)
        cfl_safety: Safety factor C ∈ (0, 1), default 0.3

    Returns:
        Maximum safe timestep dt

    Example:
        >>> dt = compute_cfl_timestep(state, v_A=1.0, cfl_safety=0.3)
        >>> # Use this dt for time integration
        >>> new_state = rk4_step(state, dt, eta=0.01, v_A=1.0)

    Physics:
        - Alfvén waves propagate at v_A along field lines (parallel)
        - Perpendicular flows from E×B drift: v_⊥ = ẑ × ∇φ
        - CFL violation → exponential instability

    Note:
        For typical KRMHD with strong guide field, v_A usually dominates
        over perpendicular velocities, so dt ~ Δz / v_A.
    """
    grid = state.grid

    # Grid spacings
    dx = grid.Lx / grid.Nx
    dy = grid.Ly / grid.Ny
    dz = grid.Lz / grid.Nz
    min_spacing = min(dx, dy, dz)

    # Maximum perpendicular velocity: |v_⊥| = |∇φ|
    # φ = (z⁺ + z⁻) / 2
    phi = (state.z_plus + state.z_minus) / 2.0

    # Compute gradients in Fourier space, then transform to real space
    dphi_dx = derivative_x(phi, grid.kx)
    dphi_dy = derivative_y(phi, grid.ky)

    # Transform to real space to find maximum
    dphi_dx_real = rfftn_inverse(dphi_dx, grid.Nz, grid.Ny, grid.Nx)
    dphi_dy_real = rfftn_inverse(dphi_dy, grid.Nz, grid.Ny, grid.Nx)

    # Maximum velocity magnitude
    v_perp_max = jnp.sqrt(dphi_dx_real**2 + dphi_dy_real**2).max()
    v_max = jnp.maximum(v_A, v_perp_max)

    # CFL timestep
    dt_cfl = cfl_safety * min_spacing / v_max

    return float(dt_cfl)
