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
import warnings

import jax
import jax.numpy as jnp
from jax import Array

from krmhd.physics import (
    KRMHDState,
    z_plus_rhs,
    z_minus_rhs,
    g0_rhs,
    g1_rhs,
    gm_rhs,
)
from krmhd.spectral import derivative_x, derivative_y, rfftn_inverse


# =============================================================================
# Module Constants
# =============================================================================

# Maximum safe damping rate threshold for exp() operations
# Beyond this value, exp(-rate) underflows to zero (causes numerical issues)
# Used for both hyper-resistivity and hyper-collision validation
MAX_DAMPING_RATE_THRESHOLD = 50.0

# Warning threshold for moderate damping rates (triggers RuntimeWarning)
DAMPING_RATE_WARNING_THRESHOLD = 20.0


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
        Lambda=state_template.Lambda,
        time=fields.time,
        grid=state_template.grid,
    )


@partial(jax.jit, static_argnames=["Nz", "Ny", "Nx", "M"])
def _krmhd_rhs_jit(
    fields: KRMHDFields,
    kx: Array,
    ky: Array,
    kz: Array,
    dealias_mask: Array,
    eta: float,
    v_A: float,
    beta_i: float,
    nu: float,
    Lambda: float,
    M: int,
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

    # Hermite moment evolution (Issue #49 - now fully implemented!)
    # **OPTIMIZATION**: For M=0 (pure fluid mode), skip all kinetic physics computation.
    # This is critical for benchmarks like Orszag-Tang that test fluid-only dynamics.
    # When M=0, fields.g has shape (Nz, Ny, Nx//2+1, 1) and should remain zero.
    if M == 0:
        # Pure fluid mode: no kinetic response, g = 0 throughout evolution
        dg_dt = jnp.zeros_like(fields.g)
    else:
        # Full kinetic mode: evolve Hermite moment hierarchy
        # Compute g0 RHS (density moment, thesis Eq. 2.7)
        dg_dt_0 = g0_rhs(
            fields.g,
            fields.z_plus,
            fields.z_minus,
            kx,
            ky,
            kz,
            dealias_mask,
            beta_i,
            Nz,
            Ny,
            Nx,
        )

        # Compute g1 RHS (velocity moment, thesis Eq. 2.8)
        dg_dt_1 = g1_rhs(
            fields.g,
            fields.z_plus,
            fields.z_minus,
            kx,
            ky,
            kz,
            dealias_mask,
            beta_i,
            Lambda,
            Nz,
            Ny,
            Nx,
        )

        # Initialize dg_dt array and populate first two moments
        dg_dt = jnp.zeros_like(fields.g)
        dg_dt = dg_dt.at[:, :, :, 0].set(dg_dt_0)
        dg_dt = dg_dt.at[:, :, :, 1].set(dg_dt_1)

        # Compute higher moment RHS (m >= 2, thesis Eq. 2.9)
        # NOTE: Python for loop is correct here! Since M is in static_argnames, the JIT compiler
        # sees the iteration count at compile time and fully unrolls this loop into separate
        # gm_rhs() calls for each m. This is necessary because gm_rhs() requires m as a static
        # argument (for compile-time optimization of Hermite recurrence coefficients).
        # Using vmap/scan would make m dynamic, breaking the static_argnames contract.
        for m in range(2, M + 1):
            dg_dt_m = gm_rhs(
                fields.g,
                fields.z_plus,
                fields.z_minus,
                kx,
                ky,
                kz,
                dealias_mask,
                m,
                beta_i,
                nu,
                Nz,
                Ny,
                Nx,
            )
            dg_dt = dg_dt.at[:, :, :, m].set(dg_dt_m)

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
    - B∥: Passive parallel magnetic field (Issue #7 - not yet implemented)
    - g: Hermite moments (Issues #22-24, #49 - fully implemented!)

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
        state.beta_i,
        state.nu,
        state.Lambda,
        state.M,
        grid.Nz,
        grid.Ny,
        grid.Nx,
    )

    # Convert back to KRMHDState (Pydantic validation at boundary)
    return _state_from_fields(deriv_fields, state)


# =============================================================================
# GANDALF Integrating Factor + RK2 Timestepping (Thesis Eq. 2.13-2.19)
# =============================================================================


@partial(jax.jit, static_argnames=["Nz", "Ny", "Nx", "M", "hyper_r", "hyper_n"])
def _gandalf_step_jit(
    fields: KRMHDFields,
    dt: float,
    kx: Array,
    ky: Array,
    kz: Array,
    dealias_mask: Array,
    eta: float,
    v_A: float,
    beta_i: float,
    nu: float,
    Lambda: float,
    M: int,
    Nz: int,
    Ny: int,
    Nx: int,
    hyper_r: int = 1,
    hyper_n: int = 1,
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
        eta: Resistivity (or hyper-resistivity coefficient)
        v_A: Alfvén velocity
        beta_i: Ion plasma beta
        nu: Collision frequency (or hyper-collision coefficient)
        Lambda: Kinetic closure parameter
        M: Number of Hermite moments
        Nz, Ny, Nx: Grid dimensions (static)
        hyper_r: Hyper-resistivity order (default: 1)
            - r=1: Standard resistivity -ηk⊥² (default, backward compatible)
            - r=2: Moderate hyper-resistivity -ηk⊥⁴ (recommended for most cases)
            - r=4: Strong hyper-resistivity -ηk⊥⁸ (expert use, requires small eta)
            - r=8: Maximum hyper-resistivity -ηk⊥¹⁶ (expert use, requires tiny eta)
        hyper_n: Hyper-collision order (default: 1)
            - n=1: Standard collision -νm (default, backward compatible)
            - n=2: Moderate hyper-collision -νm⁴ (recommended for most cases)
            - n=4: Strong hyper-collision -νm⁸ (expert use, requires small nu)

    Returns:
        Updated KRMHDFields after full timestep
    """
    # Build 3D arrays
    kz_3d = kz[:, jnp.newaxis, jnp.newaxis]
    kx_3d = kx[jnp.newaxis, jnp.newaxis, :]
    ky_3d = ky[jnp.newaxis, :, jnp.newaxis]
    k_perp_squared = kx_3d**2 + ky_3d**2  # Perpendicular wavenumber (thesis uses k⊥² only)

    # Compute k_perp_max at 2/3 dealiasing boundary (matching original GANDALF normalization)
    # Original GANDALF uses idmax = (N-1)/3 for normalization (damping_kernel.cu:47-48)
    # This makes dissipation resolution-independent: exp(-η·(k⊥/k_max)^(2r)·dt)
    idx_max = (Nx - 1) // 3
    idy_max = (Ny - 1) // 3
    k_perp_max_squared = kx[idx_max]**2 + ky[idy_max]**2

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
    rhs_0 = _krmhd_rhs_jit(
        fields, kx, ky, kz, dealias_mask, 0.0, v_A, beta_i, nu, Lambda, M, Nz, Ny, Nx
    )

    # Extract ONLY nonlinear terms by subtracting linear propagation terms
    # Full RHS includes: nonlinear + linear (∓ikz·ξ±)
    # We need: nonlinear only (equations are UNCOUPLED in linear term)
    nl_plus_0 = rhs_0.z_plus - (1j * kz_3d * fields.z_plus)    # Subtract +ikz·z⁺
    nl_minus_0 = rhs_0.z_minus + (1j * kz_3d * fields.z_minus)  # Subtract -ikz·z⁻

    # Half-step update: ξ±,n+1/2 = e^(±ikz·Δt/2) · [ξ±,n + e^(±ikz·Δt/2) · Δt/2 · NL^n]
    # Note: the e^(±ikz·Δt/2) factor appears twice (thesis Eq. 2.14)
    z_plus_half = phase_plus_half * (fields.z_plus + phase_plus_half * (dt / 2.0) * nl_plus_0)
    z_minus_half = phase_minus_half * (fields.z_minus + phase_minus_half * (dt / 2.0) * nl_minus_0)

    # Hermite moment half-step: standard RK2, no integrating factor (thesis Eq. 2.15-2.17)
    g_half = fields.g + (dt / 2.0) * rhs_0.g

    fields_half = KRMHDFields(
        z_plus=z_plus_half,
        z_minus=z_minus_half,
        B_parallel=fields.B_parallel,
        g=g_half,
        time=fields.time + dt / 2.0,
    )

    # =========================================================================
    # Step 2: Compute midpoint RHS (thesis Eq. 2.18)
    # =========================================================================

    rhs_half = _krmhd_rhs_jit(
        fields_half, kx, ky, kz, dealias_mask, 0.0, v_A, beta_i, nu, Lambda, M, Nz, Ny, Nx
    )

    # Extract ONLY nonlinear terms (UNCOUPLED)
    nl_plus_half = rhs_half.z_plus - (1j * kz_3d * fields_half.z_plus)
    nl_minus_half = rhs_half.z_minus + (1j * kz_3d * fields_half.z_minus)

    # =========================================================================
    # Step 3: Full step using midpoint RHS (thesis Eq. 2.19-2.22)
    # =========================================================================

    # Elsasser full step: ξ±,n+1 = e^(±ikz·Δt) · [ξ±,n + e^(±ikz·Δt) · Δt · NL^(n+1/2)]
    z_plus_new = phase_plus_full * (fields.z_plus + phase_plus_full * dt * nl_plus_half)
    z_minus_new = phase_minus_full * (fields.z_minus + phase_minus_full * dt * nl_minus_half)

    # Hermite moment full-step: standard RK2 using midpoint RHS (thesis Eq. 2.20-2.22)
    g_new = fields.g + dt * rhs_half.g

    # =========================================================================
    # Step 4: Apply dissipation using exponential factors (thesis Eq. 2.23-2.25)
    # =========================================================================

    # Elsasser dissipation uses NORMALIZED k_perp^(2r) (perpendicular only, thesis Eq. 2.23)
    # Original GANDALF normalization: exp(-η·(k⊥²/k⊥²_max)^r·dt)
    # This makes the overflow constraint resolution-independent: η·dt < 50 (not η·k_max^(2r)·dt!)
    # Standard (r=1): ξ± → ξ± * exp(-η (k⊥²/k⊥²_max) Δt)
    # Hyper (r>1): ξ± → ξ± * exp(-η (k⊥²/k⊥²_max)^r Δt)
    # Note: Overflow validation is performed in gandalf_step() wrapper before JIT compilation
    k_perp_2r_normalized = (k_perp_squared / k_perp_max_squared) ** hyper_r
    perp_dissipation_factor = jnp.exp(-eta * k_perp_2r_normalized * dt)
    z_plus_new = z_plus_new * perp_dissipation_factor
    z_minus_new = z_minus_new * perp_dissipation_factor

    # Hermite moment dissipation (thesis Eq. 2.24-2.25)
    # DUAL DISSIPATION MECHANISMS:
    # 1. Resistive dissipation (all moments): g → g * exp(-η (k⊥²/k⊥²_max)^r δt)
    # 2. Collisional damping (m≥2 only): g_m → g_m * exp(-ν·m^(2n) δt)
    # Both mechanisms operate simultaneously on Hermite moments

    # (1) Resistive dissipation factor (from coupling to z± fields, NORMALIZED like Elsasser)
    g_resistive_damp = jnp.exp(-eta * k_perp_2r_normalized * dt)  # Shape: [Nz, Ny, Nx//2+1]

    # (2) Collisional damping factors (moment-dependent, NORMALIZED like original GANDALF)
    # Physics: Lenard-Bernstein collision operator (thesis Eq. 2.5)
    # Original GANDALF normalization: exp(-ν·(m/M)^(2n)·dt) (timestep.cu:111)
    # This makes the overflow constraint resolution-independent: ν·dt < 50 (not ν·M^(2n)·dt!)
    #   Standard (n=1): C[g_m] = -ν·(m/M)·g_m
    #   Hyper (n>1):    C[g_m] = -ν·(m/M)^(2n)·g_m
    # Time evolution:
    #   Standard (n=1): g_m → g_m * exp(-ν·(m/M)·δt)
    #   Hyper (n>1):    g_m → g_m * exp(-ν·(m/M)^(2n)·δt)
    # Conservation: m=0 (particle number) and m=1 (momentum) are exempt from collisions
    # Note: Overflow validation is performed in gandalf_step() wrapper before JIT compilation
    moment_indices = jnp.arange(M + 1)  # [0, 1, 2, ..., M]
    # For hyper-collisions: normalized by M to match original GANDALF
    collision_damping_rate = nu * ((moment_indices / M) ** (2 * hyper_n))
    collision_factors = jnp.where(
        moment_indices >= 2,
        jnp.exp(-collision_damping_rate * dt),  # m≥2: hyper-collision damping
        1.0,  # m=0,1: no collision (conserves particles and momentum)
    )  # Shape: [M+1]

    # Apply BOTH dissipation mechanisms: resistive (all m) AND collisional (m≥2)
    # Note: Multiplicative dissipation operators preserve reality condition f(-k) = f*(k)
    # since exp(-rate·dt) is real and applied uniformly to all modes
    g_new = g_new * g_resistive_damp[:, :, :, jnp.newaxis]  # (1) Resistive dissipation
    g_new = g_new * collision_factors[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]  # (2) Collisional damping

    return KRMHDFields(
        z_plus=z_plus_new,
        z_minus=z_minus_new,
        B_parallel=fields.B_parallel,  # TODO: Issue #7
        g=g_new,
        time=fields.time + dt,
    )


def gandalf_step(
    state: KRMHDState,
    dt: float,
    eta: float,
    v_A: float,
    nu: float | None = None,
    hyper_r: int = 1,
    hyper_n: int = 1,
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
        eta: Resistivity coefficient (or hyper-resistivity if hyper_r > 1)
        v_A: Alfvén velocity
        nu: Collision frequency coefficient (optional, defaults to state.nu)
            - **Precedence**: If provided, overrides state.nu for this timestep only
            - Allows runtime control of collision rate from config without recreating state
            - Used for hyper-collision damping: -ν·m^(2n)
            - **Note**: State is not mutated; next call uses state.nu unless nu is passed again
        hyper_r: Hyper-resistivity order (default: 1)
            - r=1: Standard resistivity -ηk⊥² (default, backward compatible)
            - r=2: Moderate hyper-resistivity -ηk⊥⁴ (recommended for most cases)
            - r=4: Strong hyper-resistivity -ηk⊥⁸ (expert use, requires small eta)
            - r=8: Maximum hyper-resistivity -ηk⊥¹⁶ (expert use, requires tiny eta)
        hyper_n: Hyper-collision order (default: 1)
            - n=1: Standard collision -νm (default, backward compatible)
            - n=2: Moderate hyper-collision -νm⁴ (recommended for most cases)
            - n=4: Maximum hyper-collision -νm⁸ (expert use, requires tiny nu)

    Returns:
        New KRMHDState at time t + dt

    Raises:
        ValueError: If hyper_r not in [1, 2, 4, 8]
        ValueError: If hyper_n not in [1, 2, 4]
        ValueError: If hyper-collision overflow risk detected (nu·dt >= 50, normalized)
        ValueError: If hyper-resistivity overflow risk detected (eta·dt >= 50, normalized)

    Example:
        >>> # Standard dissipation (backward compatible)
        >>> state_new = gandalf_step(state, dt=0.01, eta=0.01, v_A=1.0)
        >>>
        >>> # Hyper-dissipation for turbulence studies
        >>> state_new = gandalf_step(state, dt=0.01, eta=0.001, v_A=1.0,
        ...                          hyper_r=8, hyper_n=4)
        >>>
        >>> # In a loop:
        >>> for i in range(n_steps):
        ...     state = gandalf_step(state, dt, eta, v_A, hyper_r=8, hyper_n=4)

    Physics:
        - Linear propagation: Handled exactly (unconditionally stable)
        - Nonlinear terms: O(dt²) accurate (RK2)
        - Dissipation: Exact exponential decay
        - Overall: O(dt²) convergence
        - Energy conservation: Excellent in inviscid limit

        Hyper-dissipation (r>1) concentrates dissipation at small scales:
        - Standard (r=1): All scales affected equally ∝ k⊥²
        - Hyper (r=8): Sharp cutoff, negligible below k_max/2

    Safety Notes:
        Hyper-collision overflow risk (n>1):
        - For M moments, highest damping rate is nu·M^(2n)
        - Must satisfy: nu·M^(2n)·dt < 50 to avoid underflow
        - Safe ranges:
            * n=1, M=20: nu < 2.5/dt (standard, very safe)
            * n=4, M=10: nu < 5×10⁻⁸/dt (moderate risk)
            * n=4, M=20: nu < 2×10⁻⁹/dt (high risk, use with caution!)

        Hyper-resistivity overflow risk (r>1):
        - For grid with k_max, highest damping rate is eta·k_max^(2r)
        - Must satisfy: eta·k_max^(2r)·dt < 50 to avoid underflow
        - k_max ≈ Nx/2 for typical grids

    Reference:
        - Thesis Chapter 2, §2.4 - GANDALF Algorithm
        - Eqs. 2.13-2.25 - Integrating factor + RK2 implementation
        - Thesis §2.5.2 - Hyper-dissipation for inertial range studies
    """
    # Use provided nu or fall back to state.nu
    nu_effective = nu if nu is not None else state.nu

    # Input validation for hyper parameters
    if hyper_r not in [1, 2, 4, 8]:
        raise ValueError(
            f"hyper_r must be 1, 2, 4, or 8 (got {hyper_r}). "
            "Use r=1 for standard dissipation, r=2 for typical turbulence studies."
        )

    if hyper_n not in [1, 2, 4]:
        raise ValueError(
            f"hyper_n must be 1, 2, or 4 (got {hyper_n}). "
            "Use n=1 for standard collisions, n=2 for typical turbulence studies."
        )

    # Safety check for hyper-collision overflow with NORMALIZED dissipation
    # With normalization: exp(-ν·(m/M)^(2n)·dt), maximum rate at m=M is simply ν·dt
    # This is RESOLUTION-INDEPENDENT in moment space (matches original GANDALF)
    max_collision_rate = nu_effective * dt

    if max_collision_rate >= MAX_DAMPING_RATE_THRESHOLD:
        safe_nu = MAX_DAMPING_RATE_THRESHOLD / dt
        raise ValueError(
            f"Hyper-collision overflow risk detected!\n"
            f"  Parameter: nu·dt = {nu_effective}·{dt} = {max_collision_rate:.2e}\n"
            f"  Threshold: Must be < {MAX_DAMPING_RATE_THRESHOLD} to avoid exp() underflow\n"
            f"  Solution: Reduce nu to < {safe_nu:.2e} or reduce dt\n"
            f"  Note: With normalized dissipation, constraint is nu·dt < {MAX_DAMPING_RATE_THRESHOLD} (independent of M or n!)"
        )

    # Warning for moderate risk (20-50)
    if max_collision_rate >= DAMPING_RATE_WARNING_THRESHOLD:
        warnings.warn(
            f"Hyper-collision damping rate is high: nu·dt = {max_collision_rate:.2e}. "
            f"Consider reducing nu or dt to improve numerical stability.",
            RuntimeWarning
        )

    # Safety check for hyper-resistivity overflow with NORMALIZED dissipation
    # With normalization: exp(-η·(k⊥²/k⊥²_max)^r·dt), maximum rate at k⊥=k⊥_max is simply η·dt
    # This is RESOLUTION-INDEPENDENT in k-space (matches original GANDALF damping_kernel.cu:50)
    max_resistivity_rate = eta * dt

    if max_resistivity_rate >= MAX_DAMPING_RATE_THRESHOLD:
        safe_eta = MAX_DAMPING_RATE_THRESHOLD / dt
        raise ValueError(
            f"Hyper-resistivity overflow risk detected!\n"
            f"  Parameter: eta·dt = {eta}·{dt} = {max_resistivity_rate:.2e}\n"
            f"  Threshold: Must be < {MAX_DAMPING_RATE_THRESHOLD} to avoid exp() underflow\n"
            f"  Solution: Reduce eta to < {safe_eta:.2e} or reduce dt\n"
            f"  Note: With normalized dissipation, constraint is eta·dt < {MAX_DAMPING_RATE_THRESHOLD} (independent of resolution or r!)"
        )

    # Warning for moderate risk (20-50)
    if max_resistivity_rate >= DAMPING_RATE_WARNING_THRESHOLD:
        warnings.warn(
            f"Hyper-resistivity damping rate is high: eta·dt = {max_resistivity_rate:.2e}. "
            f"Consider reducing eta or dt to improve numerical stability.",
            RuntimeWarning
        )

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
        state.beta_i,
        nu_effective,
        state.Lambda,
        state.M,
        grid.Nz,
        grid.Ny,
        grid.Nx,
        hyper_r,
        hyper_n,
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
