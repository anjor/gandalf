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
    The KRMHD equations in Elsasser form evolve as:
    - ∂z⁺/∂t = -∇²⊥{z⁻, z⁺} - ∇∥z⁻ + η∇²z⁺
    - ∂z⁻/∂t = -∇²⊥{z⁺, z⁻} + ∇∥z⁺ + η∇²z⁻

    Time integration must respect:
    - CFL condition: dt < dx / max(v_A, |v_⊥|)
    - Energy conservation in inviscid limit (η=0)
    - Phase relationships for Alfvén waves

References:
    - Press et al. (2007) "Numerical Recipes" §17.1 - RK methods
    - Courant, Friedrichs, Lewy (1928) - CFL condition
"""

from functools import partial
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from jax import Array

from krmhd.physics import KRMHDState, z_plus_rhs, z_minus_rhs
from krmhd.spectral import derivative_x, derivative_y, rfftn_inverse


def krmhd_rhs(
    state: KRMHDState,
    eta: float,
    v_A: float,
) -> KRMHDState:
    """
    Compute time derivatives for all KRMHD fields.

    This function wraps the individual RHS functions for each field and returns
    a KRMHDState containing the time derivatives ∂/∂t of all fields.

    Note:
        This function is not JIT-compiled at the top level due to Pydantic models.
        The underlying physics functions (z_plus_rhs, z_minus_rhs) are JIT-compiled
        for performance.

    Currently implements:
    - Elsasser fields: z⁺, z⁻ (Alfvénic sector)
    - B∥: Passive parallel magnetic field (Issue #7 - set to zero for now)
    - g: Hermite moments (Issues #22-24 - set to zero for now)

    Args:
        state: Current KRMHD state with all fields
        eta: Resistivity coefficient for dissipation
        v_A: Alfvén velocity (for normalization)

    Returns:
        KRMHDState with time derivatives (not incrementing time field)

    Example:
        >>> state = initialize_alfven_wave(grid, kz=1)
        >>> derivatives = krmhd_rhs(state, eta=0.01, v_A=1.0)
        >>> # derivatives.z_plus contains ∂z⁺/∂t

    Note:
        - B_parallel evolution deferred to Issue #7 (passive scalar)
        - Hermite moment evolution deferred to Issues #22-24 (kinetic physics)
        - Time field in output is set to 0.0 (not a derivative, just placeholder)
    """
    # Extract grid parameters
    grid = state.grid
    Nz, Ny, Nx = grid.Nz, grid.Ny, grid.Nx

    # Compute Elsasser RHS (already implemented in physics.py)
    dz_plus_dt = z_plus_rhs(
        state.z_plus,
        state.z_minus,
        grid.kx,
        grid.ky,
        grid.kz,
        grid.dealias_mask,
        eta,
        Nz,
        Ny,
        Nx,
    )

    dz_minus_dt = z_minus_rhs(
        state.z_plus,
        state.z_minus,
        grid.kx,
        grid.ky,
        grid.kz,
        grid.dealias_mask,
        eta,
        Nz,
        Ny,
        Nx,
    )

    # Passive scalar B∥ evolution (Issue #7 - not yet implemented)
    # For now: no evolution (∂B∥/∂t = 0)
    dB_parallel_dt = jnp.zeros_like(state.B_parallel)

    # Hermite moment evolution (Issues #22-24 - not yet implemented)
    # For now: no kinetic evolution (∂g/∂t = 0)
    dg_dt = jnp.zeros_like(state.g)

    # Return state with derivatives (time=0.0 as placeholder)
    return KRMHDState(
        z_plus=dz_plus_dt,
        z_minus=dz_minus_dt,
        B_parallel=dB_parallel_dt,
        g=dg_dt,
        M=state.M,
        beta_i=state.beta_i,
        v_th=state.v_th,
        nu=state.nu,
        time=0.0,  # Not a derivative, just placeholder
        grid=state.grid,
    )


def rk4_step(
    state: KRMHDState,
    dt: float,
    eta: float,
    v_A: float,
) -> KRMHDState:
    """
    Advance KRMHD state by one timestep using 4th-order Runge-Kutta.

    Note:
        This function is not JIT-compiled at the top level due to Pydantic models.
        The underlying RHS functions are JIT-compiled for performance.

    The classic RK4 method computes:
        k1 = f(y_n)
        k2 = f(y_n + dt/2 * k1)
        k3 = f(y_n + dt/2 * k2)
        k4 = f(y_n + dt * k3)
        y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    This provides O(dt⁴) local truncation error and O(dt⁴) global error.

    Args:
        state: Current KRMHD state at time t
        dt: Timestep size (should satisfy CFL condition)
        eta: Resistivity coefficient
        v_A: Alfvén velocity

    Returns:
        New KRMHDState at time t + dt

    Example:
        >>> # Single timestep
        >>> state_new = rk4_step(state, dt=0.01, eta=0.01, v_A=1.0)
        >>>
        >>> # Multiple timesteps
        >>> for i in range(n_steps):
        ...     state = rk4_step(state, dt, eta, v_A)

    Physics:
        - Preserves symplectic structure of Hamiltonian systems
        - Energy conservation up to O(dt⁴) + Issue #44 drift
        - Stable for dt < CFL limit

    Note:
        All fields are in Fourier space, so RK4 operates on complex arrays.
        Reality condition f(-k) = f*(k) is preserved automatically by the RHS.
    """
    # Stage 1: k1 = f(y_n)
    k1 = krmhd_rhs(state, eta, v_A)

    # Stage 2: k2 = f(y_n + dt/2 * k1)
    state_k2 = KRMHDState(
        z_plus=state.z_plus + 0.5 * dt * k1.z_plus,
        z_minus=state.z_minus + 0.5 * dt * k1.z_minus,
        B_parallel=state.B_parallel + 0.5 * dt * k1.B_parallel,
        g=state.g + 0.5 * dt * k1.g,
        M=state.M,
        beta_i=state.beta_i,
        v_th=state.v_th,
        nu=state.nu,
        time=state.time + 0.5 * dt,
        grid=state.grid,
    )
    k2 = krmhd_rhs(state_k2, eta, v_A)

    # Stage 3: k3 = f(y_n + dt/2 * k2)
    state_k3 = KRMHDState(
        z_plus=state.z_plus + 0.5 * dt * k2.z_plus,
        z_minus=state.z_minus + 0.5 * dt * k2.z_minus,
        B_parallel=state.B_parallel + 0.5 * dt * k2.B_parallel,
        g=state.g + 0.5 * dt * k2.g,
        M=state.M,
        beta_i=state.beta_i,
        v_th=state.v_th,
        nu=state.nu,
        time=state.time + 0.5 * dt,
        grid=state.grid,
    )
    k3 = krmhd_rhs(state_k3, eta, v_A)

    # Stage 4: k4 = f(y_n + dt * k3)
    state_k4 = KRMHDState(
        z_plus=state.z_plus + dt * k3.z_plus,
        z_minus=state.z_minus + dt * k3.z_minus,
        B_parallel=state.B_parallel + dt * k3.B_parallel,
        g=state.g + dt * k3.g,
        M=state.M,
        beta_i=state.beta_i,
        v_th=state.v_th,
        nu=state.nu,
        time=state.time + dt,
        grid=state.grid,
    )
    k4 = krmhd_rhs(state_k4, eta, v_A)

    # Final update: y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return KRMHDState(
        z_plus=state.z_plus + (dt / 6.0) * (k1.z_plus + 2.0 * k2.z_plus + 2.0 * k3.z_plus + k4.z_plus),
        z_minus=state.z_minus + (dt / 6.0) * (k1.z_minus + 2.0 * k2.z_minus + 2.0 * k3.z_minus + k4.z_minus),
        B_parallel=state.B_parallel + (dt / 6.0) * (k1.B_parallel + 2.0 * k2.B_parallel + 2.0 * k3.B_parallel + k4.B_parallel),
        g=state.g + (dt / 6.0) * (k1.g + 2.0 * k2.g + 2.0 * k3.g + k4.g),
        M=state.M,
        beta_i=state.beta_i,
        v_th=state.v_th,
        nu=state.nu,
        time=state.time + dt,
        grid=state.grid,
    )


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
