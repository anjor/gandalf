"""
Time integration for KRMHD simulations using GANDALF integrating factor + RK2.

This module implements the original GANDALF time-stepping algorithm from the thesis
Chapter 2, Equations 2.13-2.19:
- Integrating factor for linear propagation term (analytically exact)
- RK2 (midpoint method) for nonlinear terms
- Exponential integration for dissipation

The integrating factor e^(±ikz*t) removes the stiff linear term, allowing the
nonlinear terms to be integrated with RK2 (2nd-order accurate).

Example usage:
    >>> grid = SpectralGrid3D.create(Nx=64, Ny=64, Nz=32)
    >>> state = initialize_alfven_wave(grid, kz=1, amplitude=0.1)
    >>> dt = compute_cfl_timestep(state, v_A=1.0, cfl_safety=0.3)
    >>> new_state = gandalf_step(state, dt, eta=0.01, v_A=1.0)

Physics context:
    The KRMHD equations in Elsasser form (thesis Eq. 2.12, UNCOUPLED linear terms):
    - ∂ξ⁺/∂t - ikz*ξ⁺ = (1/k²⊥)[NL] + η∇²ξ⁺
    - ∂ξ⁻/∂t + ikz*ξ⁻ = (1/k²⊥)[NL] + η∇²ξ⁻

    Note: Linear terms are UNCOUPLED (ξ⁺ uses ξ⁺, ξ⁻ uses ξ⁻, not crossed).
    The integrating factor e^(∓ikz*t) removes the linear propagation terms exactly.

References:
    - Thesis Chapter 2, §2.4 - GANDALF Algorithm
    - Eqs. 2.13-2.25 - Integrating factor + RK2 timestepping
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


# =============================================================================
# GANDALF Integrating Factor + RK2 Timestepping (Thesis Eq. 2.13-2.19)
# =============================================================================


@partial(jax.jit, static_argnames=["Nz", "Ny", "Nx"])
def _gandalf_step_jit(
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
    JIT-compiled GANDALF integrating factor + RK2 timestepper.

    Implements thesis Equations 2.13-2.25:
    1. Half-step: Apply integrating factor and advance with initial RHS
    2. Compute midpoint nonlinear terms
    3. Full step: Use midpoint RHS for final update
    4. Apply dissipation exactly using exponential factors

    The integrating factor e^(±ikz*dt) handles the linear propagation term
    ∓ikz*ξ∓ analytically, removing stiffness. RK2 (midpoint method) gives
    2nd-order accuracy for the nonlinear terms.

    Args:
        fields: Current KRMHD fields
        dt: Timestep
        kx, ky, kz: Wavenumbers
        dealias_mask: 2/3 dealiasing mask
        eta: Resistivity
        v_A: Alfvén velocity
        Nz, Ny, Nx: Grid dimensions

    Returns:
        Updated KRMHDFields after full timestep
    """
    # Build 3D arrays
    kz_3d = kz[:, jnp.newaxis, jnp.newaxis]
    kx_3d = kx[jnp.newaxis, jnp.newaxis, :]
    ky_3d = ky[jnp.newaxis, :, jnp.newaxis]
    k_perp_squared = kx_3d**2 + ky_3d**2
    k_squared = k_perp_squared + kz_3d**2

    # Integrating factors (thesis Eq. 2.13-2.14)
    # For ∂ξ⁺/∂t - ikz·ξ⁺ = [NL]: multiply by e^(+ikz*t)
    # For ∂ξ⁻/∂t + ikz·ξ⁻ = [NL]: multiply by e^(-ikz*t)
    phase_plus_half = jnp.exp(+1j * kz_3d * dt / 2.0)
    phase_minus_half = jnp.exp(-1j * kz_3d * dt / 2.0)
    phase_plus_full = jnp.exp(+1j * kz_3d * dt)
    phase_minus_full = jnp.exp(-1j * kz_3d * dt)

    # =========================================================================
    # Step 1: Half-step (thesis Eq. 2.14-2.17)
    # =========================================================================

    # Compute initial RHS (with dissipation temporarily set to 0)
    rhs_0 = _krmhd_rhs_jit(fields, kx, ky, kz, dealias_mask, 0.0, v_A, Nz, Ny, Nx)

    # Extract ONLY nonlinear terms by subtracting linear propagation terms
    # Full RHS includes: nonlinear + linear (∓ikz·ξ±)
    # We need: nonlinear only (equations are UNCOUPLED in linear term)
    nl_plus_0 = rhs_0.z_plus - (1j * kz_3d * fields.z_plus)    # Subtract +ikz·z⁺
    nl_minus_0 = rhs_0.z_minus + (1j * kz_3d * fields.z_minus)  # Subtract -ikz·z⁻

    # Half-step update: ξ±,n+1/2 = e^(±ikz·Δt/2) · [ξ±,n + e^(±ikz·Δt/2) · Δt/2 · NL^n]
    # Note: the e^(±ikz·Δt/2) factor appears twice (thesis Eq. 2.14)
    z_plus_half = phase_plus_half * (fields.z_plus + phase_plus_half * (dt / 2.0) * nl_plus_0)
    z_minus_half = phase_minus_half * (fields.z_minus + phase_minus_half * (dt / 2.0) * nl_minus_0)

    fields_half = KRMHDFields(
        z_plus=z_plus_half,
        z_minus=z_minus_half,
        B_parallel=fields.B_parallel,
        g=fields.g,
        time=fields.time + dt / 2.0,
    )

    # =========================================================================
    # Step 2: Compute midpoint RHS (thesis Eq. 2.18)
    # =========================================================================

    rhs_half = _krmhd_rhs_jit(fields_half, kx, ky, kz, dealias_mask, 0.0, v_A, Nz, Ny, Nx)

    # Extract ONLY nonlinear terms (UNCOUPLED)
    nl_plus_half = rhs_half.z_plus - (1j * kz_3d * fields_half.z_plus)
    nl_minus_half = rhs_half.z_minus + (1j * kz_3d * fields_half.z_minus)

    # =========================================================================
    # Step 3: Full step using midpoint RHS (thesis Eq. 2.19)
    # =========================================================================

    # Full step: ξ±,n+1 = e^(±ikz·Δt) · [ξ±,n + e^(±ikz·Δt) · Δt · NL^(n+1/2)]
    z_plus_new = phase_plus_full * (fields.z_plus + phase_plus_full * dt * nl_plus_half)
    z_minus_new = phase_minus_full * (fields.z_minus + phase_minus_full * dt * nl_minus_half)

    # =========================================================================
    # Step 4: Apply dissipation using exponential factors (thesis Eq. 2.23-2.25)
    # =========================================================================

    # Exponential dissipation: ξ± → ξ± * exp(-η*k²*Δt)
    dissipation_factor = jnp.exp(-eta * k_squared * dt)
    z_plus_new = z_plus_new * dissipation_factor
    z_minus_new = z_minus_new * dissipation_factor

    return KRMHDFields(
        z_plus=z_plus_new,
        z_minus=z_minus_new,
        B_parallel=fields.B_parallel,  # TODO: Issue #7
        g=fields.g,  # TODO: Issues #22-24
        time=fields.time + dt,
    )


def gandalf_step(
    state: KRMHDState,
    dt: float,
    eta: float,
    v_A: float,
) -> KRMHDState:
    """
    Advance KRMHD state using GANDALF integrating factor + RK2 method.

    This implements the original GANDALF algorithm from thesis Chapter 2:
    1. Integrating factor for linear propagation (analytically exact)
    2. RK2 (midpoint method) for nonlinear terms (2nd-order accurate)
    3. Exponential integration for dissipation (exact)

    The integrating factor e^(±ikz*t) removes the stiff linear term ∓ikz*ξ∓,
    allowing RK2 to integrate the nonlinear bracket terms efficiently.

    Args:
        state: Current KRMHD state at time t
        dt: Timestep size (should satisfy CFL for nonlinear terms)
        eta: Resistivity coefficient
        v_A: Alfvén velocity

    Returns:
        New KRMHDState at time t + dt

    Example:
        >>> state_new = gandalf_step(state, dt=0.01, eta=0.01, v_A=1.0)
        >>> # Or in a loop:
        >>> for i in range(n_steps):
        ...     state = gandalf_step(state, dt, eta, v_A)

    Physics:
        - Linear propagation: Handled exactly (unconditionally stable)
        - Nonlinear terms: O(dt²) accurate (RK2)
        - Dissipation: Exact exponential decay
        - Overall: O(dt²) convergence
        - Energy conservation: Excellent in inviscid limit

    Reference:
        - Thesis Chapter 2, §2.4 - GANDALF Algorithm
        - Eqs. 2.13-2.25 - Integrating factor + RK2 implementation
    """
    grid = state.grid
    fields = _fields_from_state(state)

    # Call JIT-compiled GANDALF kernel
    new_fields = _gandalf_step_jit(
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


# Alias for backward compatibility with tests
# NOTE: Despite the name, this is NOT plain RK4! It's GANDALF's integrating factor + RK2.
# The name is kept for backward compatibility but the algorithm is:
#   - Integrating factor: e^(±ikz·t) handles linear propagation exactly
#   - RK2 (midpoint method): 2nd-order for nonlinear terms
# This is MORE accurate than plain RK4 for linear waves (zero temporal error).
rk4_step = gandalf_step


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

    where C is a safety factor (typically 0.1-0.5 for RK2/RK4).

    Args:
        state: Current KRMHD state (used to compute max velocity)
        v_A: Alfvén velocity (parallel wave speed)
        cfl_safety: Safety factor C ∈ (0, 1), default 0.3

    Returns:
        Maximum safe timestep dt

    Example:
        >>> dt = compute_cfl_timestep(state, v_A=1.0, cfl_safety=0.3)
        >>> # Use this dt for time integration
        >>> new_state = gandalf_step(state, dt, eta=0.01, v_A=1.0)

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
