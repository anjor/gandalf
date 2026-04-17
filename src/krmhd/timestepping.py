"""
Time integration for KRMHD simulations using a mixed integrating-factor scheme.

This module implements the original GANDALF time-stepping algorithm from the thesis
Chapter 2, Equations 2.13-2.19:
- Integrating factor for linear propagation term (analytically exact)
- RK2 (midpoint method) for Elsasser nonlinear terms
- RK4 (Lawson form) for Hermite nonlinear terms
- Exponential integration for dissipation

The integrating factor e^(±ikz*t) removes the stiff linear term. The Elsasser
sector retains the thesis midpoint RK2 update, while the passive Hermite
sector uses a Lawson-form RK4 update for stable externally driven advection.

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

import math
import warnings
from functools import partial
from typing import Callable, Tuple, NamedTuple

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
from krmhd.hermite import (
    compute_streaming_matrix,
    compute_streaming_eigensystem,
    build_implicit_operator,
    factor_imex_operator,
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
    time: float  # Python float at public API; JAX Array inside JIT context


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
    """Reconstruct KRMHDState from JAX fields (validates at boundary).

    This function MUST remain outside any jit-compiled scope because
    float() materializes a JAX scalar into a Python float for Pydantic.
    """
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
        time=float(fields.time),
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
def _gandalf_step_lawson_rk4_jit(
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
    streaming_eigenvalues: Array = None,
    streaming_P_T: Array = None,
    streaming_P_inv_T: Array = None,
) -> KRMHDFields:
    """
    JIT-compiled mixed integrating-factor timestepper.

    Implements thesis Equations 2.13-2.25:
    1. Elsasser half-step: Apply integrating factor and advance with initial RHS
    2. Elsasser midpoint: Compute midpoint nonlinear terms
    3. Hermite RK4: Advance passive Hermite moments in Lawson form
    4. Full step: Use midpoint/RK4 updates for final state
    5. Apply dissipation exactly using exponential factors

    The integrating factor e^(±ikz*dt) handles the linear propagation term
    ∓ikz*ξ∓ analytically, removing stiffness. The Elsasser sector keeps the
    original midpoint RK2 update, while the passive Hermite sector uses a
    classical RK4 update in the interaction picture to avoid the unconditional
    midpoint-instability for externally advected Hermite modes.

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
            - n=2: Moderate hyper-collision -νm² (recommended for most cases)
            - n=3: Strong hyper-collision -νm³ (matches original GANDALF alpha_m=3)
            - n=4: Very strong hyper-collision -νm⁴ (expert use, requires small nu)
            - n=6: Thesis Figure 3.3 benchmark value for strong high-m damping
        streaming_eigenvalues: Eigenvalues of streaming matrix T (from compute_streaming_eigensystem).
            Must be provided — None default is only for Python syntax (keyword after default args).
        streaming_P_T: Transposed right eigenvector matrix P.T (from compute_streaming_eigensystem).
        streaming_P_inv_T: Transposed inverse eigenvector matrix P_inv.T.

    Returns:
        Updated KRMHDFields after full timestep

    Note:
        This function evaluates _krmhd_rhs_jit 6x per step:
        2x with real kz for the Elsasser midpoint update and 4x with kz=0 for
        Hermite RK4 nonlinear stages. The kz=0 Hermite calls avoid float32
        cancellation from subtracting streaming computed two different ways.
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

    # Hermite helper: evaluate nonlinear-only RHS with streaming removed exactly.
    kz_zero = jnp.zeros_like(kz)

    def compute_nl_g(stage_fields: KRMHDFields) -> Array:
        """Evaluate the Hermite nonlinear RHS with kz forced to zero."""
        return _krmhd_rhs_jit(
            stage_fields,
            kx,
            ky,
            kz_zero,
            dealias_mask,
            0.0,
            v_A,
            beta_i,
            nu,
            Lambda,
            M,
            Nz,
            Ny,
            Nx,
        ).g

    # Streaming integrating-factor phases for Hermite moments:
    # exp(-i * sqrt(beta_i) * kz * eigenvalue * dt)
    kz_4d = kz[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    evals = streaming_eigenvalues[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    g_phase_half = jnp.exp(-1j * jnp.sqrt(beta_i) * kz_4d * evals * dt / 2.0)
    g_phase_full = g_phase_half ** 2

    def apply_streaming_if(g_arr, phase):
        """Apply streaming integrating factor: rotate to eigenspace, phase, rotate back.

        Note: At float32, P_inv @ P != I exactly (error ~6e-8), causing ~0.3% energy
        drift per application. This is acceptable for physics runs where collisional
        damping dominates the error. For precision-critical work, use float64 arrays.
        """
        g_eigen = g_arr @ streaming_P_inv_T  # project to eigenspace
        g_eigen = g_eigen * phase             # apply per-kz, per-eigenvalue phase
        return g_eigen @ streaming_P_T        # project back

    # =========================================================================
    # Step 1: Elsasser half-step (thesis Eq. 2.14-2.17)
    # =========================================================================

    # Compute initial RHS (with dissipation temporarily set to 0)
    # Note: rhs_0 and nl_g_1 both evaluate the same state, but nl_g_1 uses kz=0
    # to remove the linear streaming term exactly. Reusing rhs_0.g here would
    # reintroduce the float32 cancellation that motivated the split path.
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

    # Hermite RK4 stage 1: nonlinear-only RHS at t_n
    nl_g_1 = compute_nl_g(fields)
    g_base_half = apply_streaming_if(fields.g, g_phase_half)
    g_base_full = apply_streaming_if(fields.g, g_phase_full)
    nl_g_1_half = apply_streaming_if(nl_g_1, g_phase_half)
    nl_g_1_full = apply_streaming_if(nl_g_1, g_phase_full)
    g_stage_2 = g_base_half + (dt / 2.0) * nl_g_1_half

    fields_half = KRMHDFields(
        z_plus=z_plus_half,
        z_minus=z_minus_half,
        B_parallel=fields.B_parallel,
        g=g_stage_2,
        time=fields.time + dt / 2.0,
    )

    # =========================================================================
    # Step 2: Compute midpoint RHS / Hermite RK4 stage 2
    # =========================================================================

    rhs_half = _krmhd_rhs_jit(
        fields_half, kx, ky, kz, dealias_mask, 0.0, v_A, beta_i, nu, Lambda, M, Nz, Ny, Nx
    )

    # Extract ONLY nonlinear terms (UNCOUPLED)
    nl_plus_half = rhs_half.z_plus - (1j * kz_3d * fields_half.z_plus)
    nl_minus_half = rhs_half.z_minus + (1j * kz_3d * fields_half.z_minus)

    # Hermite RK4 stage 2 at t_n + dt/2, using the Elsasser midpoint fields.
    # Stages 2 and 3 deliberately share the same midpoint Elsasser state because
    # the Hermite hierarchy is passive: g samples Phi/Psi supplied by z±, but it
    # does not feed back on the Alfvénic sector during the sub-stages.
    nl_g_2 = compute_nl_g(fields_half)

    # =========================================================================
    # Step 3: Full step using midpoint RHS (Elsasser) and RK4 stage assembly (Hermite)
    # =========================================================================

    # Elsasser full step: ξ±,n+1 = e^(±ikz·Δt) · [ξ±,n + e^(±ikz·Δt) · Δt · NL^(n+1/2)]
    z_plus_new = phase_plus_full * (fields.z_plus + phase_plus_full * dt * nl_plus_half)
    z_minus_new = phase_minus_full * (fields.z_minus + phase_minus_full * dt * nl_minus_half)

    # Hermite RK4 stage 3: same midpoint Elsasser fields, updated Hermite state.
    g_stage_3 = g_base_half + (dt / 2.0) * nl_g_2
    fields_half_rk4 = KRMHDFields(
        z_plus=z_plus_half,
        z_minus=z_minus_half,
        B_parallel=fields.B_parallel,
        g=g_stage_3,
        time=fields.time + dt / 2.0,
    )
    nl_g_3 = compute_nl_g(fields_half_rk4)
    nl_g_2_half = apply_streaming_if(nl_g_2, g_phase_half)
    nl_g_3_half = apply_streaming_if(nl_g_3, g_phase_half)

    # Hermite RK4 stage 4 uses the full-step Elsasser state.
    g_stage_4 = g_base_full + dt * nl_g_3_half
    fields_full_rk4 = KRMHDFields(
        z_plus=z_plus_new,
        z_minus=z_minus_new,
        B_parallel=fields.B_parallel,
        g=g_stage_4,
        time=fields.time + dt,
    )
    nl_g_4 = compute_nl_g(fields_full_rk4)

    # Lawson-form RK4 update in the interaction picture.
    g_new = g_base_full + (dt / 6.0) * (
        nl_g_1_full + 2.0 * nl_g_2_half + 2.0 * nl_g_3_half + nl_g_4
    )

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
    # Original GANDALF normalization: exp(-ν·(m/M)^n·dt) (timestep.cu:111, alpha_m parameter)
    # This makes the overflow constraint resolution-independent: ν·dt < 50 (not ν·M^n·dt!)
    # IMPORTANT: Unlike spatial hyper-dissipation which uses k^(2r) due to ∇² ~ k²,
    #            moment collisions use just m^n to match original GANDALF alpha_m parameter
    #   Standard (n=1): C[g_m] = -ν·(m/M)·g_m
    #   Hyper (n>1):    C[g_m] = -ν·(m/M)^n·g_m
    # Time evolution:
    #   Standard (n=1): g_m → g_m * exp(-ν·(m/M)·δt)
    #   Hyper (n>1):    g_m → g_m * exp(-ν·(m/M)^n·δt)
    # Conservation: m=0 (particle number) and m=1 (momentum) are exempt from collisions
    # Note: gandalf_step() wrapper enforces M>=2 only when nu>0.
    # The JIT kernel independently guards collision application with `if M >= 2` below.

    # Apply BOTH dissipation mechanisms: resistive (all m) AND collisional (m≥2)
    # Note: Multiplicative dissipation operators preserve reality condition f(-k) = f*(k)
    # since exp(-rate·dt) is real and applied uniformly to all modes
    g_new = g_new * g_resistive_damp[:, :, :, jnp.newaxis]  # (1) Resistive dissipation

    # Static branch: M is in static_argnames, so this is resolved at JAX trace time
    if M >= 2:
        # Collision damping requires M>=2 to avoid division by zero in (m/M)^n
        moment_indices = jnp.arange(M + 1)  # [0, 1, 2, ..., M]
        # For hyper-collisions: normalized by M to match original GANDALF (requires M>=2)
        # Normalized by M: max damping rate at m=M is exactly ν·dt, independent of M
        collision_damping_rate = nu * ((moment_indices / M) ** hyper_n)
        collision_factors = jnp.where(
            moment_indices >= 2,
            jnp.exp(-collision_damping_rate * dt),  # m≥2: hyper-collision damping
            1.0,  # m=0,1: no collision (conserves particles and momentum)
        )  # Shape: [M+1]
        g_new = g_new * collision_factors[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]  # (2) Collisional damping

    # Project Hermite moments back to the resolved 3D spectral band. The Hermite
    # Poisson-bracket RHS is already dealiased in physics.py, so this is a
    # defensive guarantee against externally injected or checkpointed out-of-band
    # content rather than the primary nonlinear aliasing control.
    g_new = g_new * dealias_mask[:, :, :, jnp.newaxis]

    return KRMHDFields(
        z_plus=z_plus_new,
        z_minus=z_minus_new,
        B_parallel=fields.B_parallel,  # TODO: Issue #7
        g=g_new,
        time=fields.time + dt,
    )


# =============================================================================
# IMEX-RK222 Hermite Integrator (Issue #137)
# =============================================================================

# ARS(2,2,2) implicit-stage coefficient and stage-2 explicit scaling (Python floats).
_IMEX_GAMMA: float = (2.0 - math.sqrt(2.0)) / 2.0        # gamma = (2 - sqrt(2))/2
_IMEX_DELTA: float = 1.0 - 1.0 / (2.0 * _IMEX_GAMMA)     # delta = 1 - 1/(2*gamma)


@partial(jax.jit, static_argnames=["Nz", "Ny", "Nx", "M", "hyper_r"])
def _gandalf_step_imex222_jit(
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
    hyper_r: int,
    L_per_kz: Array,
    lu: Array,
    piv: Array,
) -> KRMHDFields:
    """
    JIT-compiled IMEX-RK222 (Ascher-Ruuth-Spiteri, gamma=(2-sqrt(2))/2) stepper.

    Linear Hermite streaming and hyper-collisional damping are treated
    implicitly via a per-kz batched LU solve; the Poisson-bracket nonlinear
    advection is treated explicitly. Elsasser z+/z- use the current
    integrating-factor RK2 midpoint (bit-identical to the Lawson-RK4 path).

    Scheme:
        Stage 1 (implicit):
            (I - dt*gamma*L) g^(1) = g^n + dt*gamma*N(g^n, z+/-^n)
        Stage 2 (implicit):
            (I - dt*gamma*L) g^(n+1) = g^n
                + dt*(1-gamma)*L*g^(1)
                + dt*delta*N(g^n, z+/-^n)
                + dt*(1-delta)*N(g^(1), z+/-^(n+dt/2))

    L and its factorization (lu, piv) are precomputed once per step outside
    this JIT kernel (see gandalf_step). Hyper-collisional damping is folded
    into L; the exponential collision factor is therefore NOT applied here.
    Resistive damping on g (from z+/- coupling) stays exponential.
    """
    kz_3d = kz[:, jnp.newaxis, jnp.newaxis]
    kx_3d = kx[jnp.newaxis, jnp.newaxis, :]
    ky_3d = ky[jnp.newaxis, :, jnp.newaxis]
    k_perp_squared = kx_3d**2 + ky_3d**2

    idx_max = (Nx - 1) // 3
    idy_max = (Ny - 1) // 3
    k_perp_max_squared = kx[idx_max]**2 + ky[idy_max]**2

    # -------------------------------------------------------------------------
    # Elsasser integrating-factor RK2 midpoint (unchanged from Lawson path)
    # -------------------------------------------------------------------------
    phase_plus_half = jnp.exp(+1j * kz_3d * dt / 2.0)
    phase_minus_half = jnp.exp(-1j * kz_3d * dt / 2.0)
    phase_plus_full = jnp.exp(+1j * kz_3d * dt)
    phase_minus_full = jnp.exp(-1j * kz_3d * dt)

    kz_zero = jnp.zeros_like(kz)

    def compute_nl_g(stage_fields: KRMHDFields) -> Array:
        """Evaluate Hermite nonlinear RHS with kz forced to zero (strips streaming)."""
        return _krmhd_rhs_jit(
            stage_fields,
            kx,
            ky,
            kz_zero,
            dealias_mask,
            0.0,
            v_A,
            beta_i,
            nu,
            Lambda,
            M,
            Nz,
            Ny,
            Nx,
        ).g

    # Elsasser half-step
    rhs_0 = _krmhd_rhs_jit(
        fields, kx, ky, kz, dealias_mask, 0.0, v_A, beta_i, nu, Lambda, M, Nz, Ny, Nx
    )
    nl_plus_0 = rhs_0.z_plus - (1j * kz_3d * fields.z_plus)
    nl_minus_0 = rhs_0.z_minus + (1j * kz_3d * fields.z_minus)
    z_plus_half = phase_plus_half * (fields.z_plus + phase_plus_half * (dt / 2.0) * nl_plus_0)
    z_minus_half = phase_minus_half * (fields.z_minus + phase_minus_half * (dt / 2.0) * nl_minus_0)

    # IMEX Stage 1: N at (g^n, z+/-^n)
    N_1 = compute_nl_g(fields)

    # Solve (I - dt*gamma*L) g^(1) = g^n + dt*gamma * N_1
    rhs_1 = fields.g + (dt * _IMEX_GAMMA) * N_1
    g_1 = _imex_solve(lu, piv, rhs_1)

    # Elsasser midpoint RHS (needed for the full-step update and stage-2 N)
    fields_half = KRMHDFields(
        z_plus=z_plus_half,
        z_minus=z_minus_half,
        B_parallel=fields.B_parallel,
        g=g_1,
        time=fields.time + dt / 2.0,
    )
    rhs_half = _krmhd_rhs_jit(
        fields_half, kx, ky, kz, dealias_mask, 0.0, v_A, beta_i, nu, Lambda, M, Nz, Ny, Nx
    )
    nl_plus_half = rhs_half.z_plus - (1j * kz_3d * fields_half.z_plus)
    nl_minus_half = rhs_half.z_minus + (1j * kz_3d * fields_half.z_minus)

    # Elsasser full-step (matches Lawson path exactly given same nl_plus/minus_half)
    z_plus_new = phase_plus_full * (fields.z_plus + phase_plus_full * dt * nl_plus_half)
    z_minus_new = phase_minus_full * (fields.z_minus + phase_minus_full * dt * nl_minus_half)

    # IMEX Stage 2: N at (g^(1), z+/-^mid)  — already computed above as rhs_half.g,
    # but rhs_half.g was evaluated with real kz (includes streaming). We need the
    # nonlinear-only Hermite RHS: reuse compute_nl_g on the same midpoint fields.
    N_2 = compute_nl_g(fields_half)

    # Explicit L*g^(1) mat-vec (per-kz; (M+1, M+1) @ (M+1,) for each (ky, kx))
    L_g1 = jnp.einsum("zij,zyxj->zyxi", L_per_kz, g_1)

    rhs_2 = (
        fields.g
        + (dt * (1.0 - _IMEX_GAMMA)) * L_g1
        + (dt * _IMEX_DELTA) * N_1
        + (dt * (1.0 - _IMEX_DELTA)) * N_2
    )
    g_new = _imex_solve(lu, piv, rhs_2)

    # -------------------------------------------------------------------------
    # Post-step dissipation (resistivity) and dealiasing
    # -------------------------------------------------------------------------
    k_perp_2r_normalized = (k_perp_squared / k_perp_max_squared) ** hyper_r
    perp_dissipation_factor = jnp.exp(-eta * k_perp_2r_normalized * dt)
    z_plus_new = z_plus_new * perp_dissipation_factor
    z_minus_new = z_minus_new * perp_dissipation_factor

    # Resistive damping on g (from coupling to z+/-, kept exponential).
    # Hyper-collisional damping is already inside L, so no collision factor here.
    g_resistive_damp = jnp.exp(-eta * k_perp_2r_normalized * dt)
    g_new = g_new * g_resistive_damp[:, :, :, jnp.newaxis]

    # Defensive dealias of Hermite moments
    g_new = g_new * dealias_mask[:, :, :, jnp.newaxis]

    return KRMHDFields(
        z_plus=z_plus_new,
        z_minus=z_minus_new,
        B_parallel=fields.B_parallel,
        g=g_new,
        time=fields.time + dt,
    )


def _imex_solve(lu: Array, piv: Array, rhs: Array) -> Array:
    """
    Per-kz batched linear solve for the ARS(2,2,2) implicit stages.

    Reshapes rhs of shape (Nz, Ny, Nx//2+1, M+1) into (Nz, M+1, Ny*Nxh) so
    that jax.scipy.linalg.lu_solve, vmapped over the leading Nz axis, solves
    one (M+1, M+1) system per k_z against Ny*Nxh RHS vectors at once.
    """
    Nz, Ny, Nxh, Mp1 = rhs.shape
    rhs_batched = jnp.transpose(rhs, (0, 3, 1, 2)).reshape(Nz, Mp1, Ny * Nxh)
    sol_batched = jax.vmap(jax.scipy.linalg.lu_solve)((lu, piv), rhs_batched)
    return sol_batched.reshape(Nz, Mp1, Ny, Nxh).transpose(0, 2, 3, 1)


def gandalf_step(
    state: KRMHDState,
    dt: float,
    eta: float,
    v_A: float,
    nu: float | None = None,
    hyper_r: int = 1,
    hyper_n: int = 1,
    scheme: str = "imex_rk222",
) -> KRMHDState:
    """
    Advance KRMHD state using the mixed GANDALF integrating-factor method.

    This implements the original GANDALF algorithm from thesis Chapter 2:
    1. Integrating factor for linear propagation (analytically exact)
    2. RK2 (midpoint method) for Elsasser nonlinear terms
    3. RK4 (Lawson form) for passive Hermite nonlinear terms
    4. Exponential integration for dissipation (exact)

    The integrating factor e^(±ikz*t) removes the stiff linear term ∓ikz*ξ∓,
    allowing the Alfvénic sector to retain the original midpoint update while
    the passive Hermite sector uses a stable higher-order advection update.

    Args:
        state: Current KRMHD state at time t
        dt: Timestep size (should satisfy CFL for nonlinear terms)
        eta: Resistivity coefficient (or hyper-resistivity if hyper_r > 1)
        v_A: Alfvén velocity
        nu: Collision frequency coefficient (optional, defaults to state.nu)
            - **Precedence**: If provided, overrides state.nu for this timestep only
            - Allows runtime control of collision rate from config without recreating state
            - Used for hyper-collision damping: -ν·m^n (matches GANDALF alpha_m)
            - **Note**: State is not mutated; next call uses state.nu unless nu is passed again
        hyper_r: Hyper-resistivity order (default: 1)
            - r=1: Standard resistivity -ηk⊥² (default, backward compatible)
            - r=2: Moderate hyper-resistivity -ηk⊥⁴ (recommended for most cases)
            - r=4: Strong hyper-resistivity -ηk⊥⁸ (expert use, requires small eta)
            - r=8: Maximum hyper-resistivity -ηk⊥¹⁶ (expert use, requires tiny eta)
        hyper_n: Hyper-collision order (default: 1)
            - n=1: Standard collision -νm (default, backward compatible)
            - n=2: Moderate hyper-collision -νm² (recommended for most cases)
            - n=3: Strong hyper-collision -νm³ (matches original GANDALF alpha_m=3)
            - n=4: Very strong hyper-collision -νm⁴ (expert use, requires small nu)
        scheme: Hermite integrator choice (default "imex_rk222" per Issue #137).
            - "imex_rk222": ARS(2,2,2) IMEX scheme. Streaming and
              hyper-collisional damping are implicit (unconditionally stable);
              nonlinear advection is explicit. Elsasser z+/z- advance is
              bit-identical to the Lawson path.
            - "lawson_rk4": legacy mixed integrating-factor + Lawson-RK4 path,
              retained for comparison and rollback. Known to be unstable at
              high M·k_z (see Issue #137).

    Returns:
        New KRMHDState at time t + dt

    Raises:
        ValueError: If hyper_r not in [1, 2, 4, 8]
        ValueError: If hyper_n not in [1, 2, 3, 4, 6]
        ValueError: If scheme not in {"lawson_rk4", "imex_rk222"}
        ValueError: If hyper-collision overflow risk detected (nu·dt >= 50, normalized;
            Lawson path only — the IMEX solve is unconditionally stable)
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
        - Nonlinear terms: Elsasser O(dt²), Hermite O(dt⁴) in Lawson form
        - Dissipation: Exact exponential decay
        - Overall: O(dt²) for the coupled mixed scheme
        - Energy conservation: Excellent in inviscid limit

        Hyper-dissipation (r>1) concentrates dissipation at small scales:
        - Standard (r=1): All scales affected equally ∝ k⊥²
        - Hyper (r=8): Sharp cutoff, negligible below k_max/2

    Safety Notes:
        Hyper-collision overflow risk (n>1):
        - NORMALIZED formulation: exp(-ν·(m/M)^n·dt) matches original GANDALF
        - Maximum damping rate at m=M is simply: ν·dt (independent of M!)
        - Must satisfy: ν·dt < 50 to avoid underflow
        - Safe ranges (resolution-independent!):
            * n=1: ν·dt < 50 (standard, very safe)
            * n=2: ν·dt < 50 (recommended, very safe)
            * n=3: ν·dt < 50 (matches GANDALF alpha_m=3, safe)
            * n=4: ν·dt < 50 (strong damping, safe with ν<1)
            * n=6: ν·dt < 50 (thesis Figure 3.3 benchmark, same normalized bound)

        Hyper-resistivity overflow risk (r>1):
        - NORMALIZED formulation: exp(-η·(k⊥²/k⊥²_max)^r·dt)
        - Maximum damping rate at k⊥=k⊥_max is simply: η·dt (independent of grid!)
        - Must satisfy: η·dt < 50 to avoid underflow

    Reference:
        - Thesis Chapter 2, §2.4 - GANDALF Algorithm
        - Eqs. 2.13-2.25 - Integrating factor implementation
        - Thesis §2.5.2 - Hyper-dissipation for inertial range studies
    """
    # Use provided nu or fall back to state.nu
    nu_effective = nu if nu is not None else state.nu

    # Input validation for hyper parameters
    if hyper_r not in (1, 2, 4, 8):
        raise ValueError(f"hyper_r must be 1, 2, 4, or 8 (got {hyper_r})")

    if hyper_n not in (1, 2, 3, 4, 6):
        raise ValueError(f"hyper_n must be 1, 2, 3, 4, or 6 (got {hyper_n})")

    if scheme not in ("lawson_rk4", "imex_rk222"):
        raise ValueError(
            f"scheme must be 'lawson_rk4' or 'imex_rk222' (got {scheme!r})"
        )

    # Validate M for collision operator (prevents division by zero)
    # Collision damping rate = ν·(m/M)^n requires M >= 2 for well-defined rates
    # M=0 (and M=1) are allowed for pure fluid RMHD runs when collisions are disabled (nu=0)
    # Strict > 0 check is intentional: even tiny nu (e.g. 1e-15) requires M>=2
    if state.M < 2 and nu_effective > 0:
        raise ValueError(
            f"M must be >= 2 when collisions are enabled (got M={state.M}, nu={nu_effective}). "
            "Collision damping uses (m/M)^n, requiring M >= 2 for meaningful rates. "
            "For pure fluid RMHD, set nu=0."
        )

    # Safety check for hyper-collision overflow with NORMALIZED dissipation.
    # Only the Lawson path applies damping via exp(-ν·(m/M)^n·dt), which can
    # underflow for large nu·dt. The IMEX path folds damping into an implicit
    # solve (unconditionally stable) and needs no such guard.
    max_collision_rate = nu_effective * dt

    if scheme == "lawson_rk4":
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

    if scheme == "lawson_rk4":
        # Precompute Hermite streaming eigensystem for integrating factor (cached)
        _, eigenvalues, P, P_inv = compute_streaming_eigensystem(
            state.M, state.Lambda
        )

        new_fields = _gandalf_step_lawson_rk4_jit(
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
            streaming_eigenvalues=eigenvalues,
            streaming_P_T=P.T,
            streaming_P_inv_T=P_inv.T,
        )
    else:  # scheme == "imex_rk222"
        # Build the implicit operator L(k_z) = -i*sqrt(beta_i)*kz*T + D(nu, M, hyper_n)
        # and factor (I - dt*gamma*L) once per step. L is cheap to rebuild because
        # only Nz distinct (M+1)x(M+1) matrices exist (shared across k_perp).
        L_per_kz = build_implicit_operator(
            grid.kz,
            state.beta_i,
            nu_effective,
            state.M,
            state.Lambda,
            hyper_n,
        )
        lu, piv = factor_imex_operator(L_per_kz, dt, _IMEX_GAMMA)

        new_fields = _gandalf_step_imex222_jit(
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
            L_per_kz,
            lu,
            piv,
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

    # Note: No Hermite streaming constraint needed for either scheme.
    # - Lawson path: the integrating factor handles oscillatory streaming
    #   terms exactly (unitary propagator).
    # - IMEX path (Issue #137): streaming is part of the implicit operator
    #   L and solved unconditionally stably each stage.

    return float(dt_cfl)
