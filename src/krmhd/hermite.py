"""
Hermite basis functions for kinetic velocity-space representation.

This module implements orthonormal Hermite functions (quantum harmonic oscillator
eigenfunctions) for expanding the ion compressive distribution G± of KRMHD in
parallel velocity:

    g(v∥) = Σ_m g_m · ψ_m(v∥/v_th)

where ψ_m(v) = N_m · H_m(v) · exp(-v²/2) are orthonormal Hermite functions.

The orthonormality condition is:
    ∫ ψ_m(v) · ψ_n(v) dv = δ_{mn}

This differs from bare Hermite polynomials H_m(v) which satisfy:
    ∫ H_m(v) · H_n(v) · exp(-v²) dv = √π · 2^m · m! · δ_{mn}

Key functions:
- hermite_polynomial: Evaluate physicist's Hermite polynomial H_m(v)
- hermite_normalization: Compute N_m = 1/√(2^m·m!·√π) for orthonormal basis
- hermite_basis_function: Evaluate ψ_m(v) = N_m · H_m(v) · exp(-v²/2)
- distribution_to_moments: Project g(v) onto orthonormal basis
- moments_to_distribution: Reconstruct g(v) from moments

All functions are JIT-compiled for performance and use JAX arrays for
automatic differentiation compatibility.

Physical context (Thesis §2.2, Eq 2.1):
The Hermite representation captures Landau damping and phase mixing in
velocity space, essential for kinetic plasma physics. The orthonormal basis
ensures proper treatment of the thermal distribution and enables efficient
moment truncation.

Normalization: v = v∥/v_th with v_th = √(2T_i/m_i), so the Maxwellian weight is
exp(-v∥²/v_th²) = exp(-v²). Each ψ_m carries half of it, exp(-v²/2). The
streaming coefficients √((m+1)/2), √(m/2) in physics.g_rhs follow from
v·H_m = H_{m+1}/2 + m·H_{m-1} in this normalization. The least-damped
eigenvalue of the hierarchy then converges to the root of Λ = 1 + ζZ(ζ).
"""

from functools import lru_cache, partial
from typing import Any, NamedTuple, Optional
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


@jax.jit
def hermite_normalization(m: int) -> Array:
    """
    Compute normalization constant for orthonormal Hermite functions.

    Returns 1/√(2^m · m! · √π) for Hermite functions ψ_m = N_m·H_m(v)·exp(-v²/2)
    (half of the Maxwellian weight exp(-v²), v = v∥/v_th, v_th = √(2T_i/m_i)).

    These are the standard quantum harmonic oscillator wavefunctions and satisfy:
        ∫ ψ_m(v) · ψ_n(v) dv = δ_{mn}

    where ψ_m(v) = N_m · H_m(v) · exp(-v²/2)

    Uses log-gamma for numerical stability.

    Args:
        m: Hermite function order (must be non-negative)

    Returns:
        Normalization constant 1/√(2^m · m! · √π) as a JAX scalar Array

    Example:
        >>> hermite_normalization(0)
        Array(0.751125..., dtype=float32)  # 1/π^(1/4)
        >>> hermite_normalization(1)
        Array(0.531126..., dtype=float32)  # 1/(√2 · π^(1/4))
    """
    # N_m = 1 / √(2^m · m! · √π)
    # log(N_m) = -0.5 * [m·ln(2) + ln(m!) + 0.5·ln(π)]
    log_norm = -0.5 * (
        m * jnp.log(2.0) +
        jax.scipy.special.gammaln(m + 1) +
        0.5 * jnp.log(jnp.pi)
    )
    return jnp.exp(log_norm)


@partial(jax.jit, static_argnames=['m'])
def hermite_polynomial(m: int, v: Array) -> Array:
    """
    Evaluate Hermite polynomial H_m(v) using recurrence relation.

    Uses the physicist's definition with recurrence:
        H_0(v) = 1
        H_1(v) = 2v
        H_{m+1}(v) = 2v·H_m(v) - 2m·H_{m-1}(v)

    Implementation uses jax.lax.fori_loop for efficient JIT compilation.

    Args:
        m: Polynomial order (non-negative integer)
        v: Velocity values (can be scalar or array)

    Returns:
        H_m(v) evaluated at input velocities (same shape as v)

    Raises:
        ValueError: If m < 0

    Example:
        >>> v = jnp.linspace(-2, 2, 5)
        >>> hermite_polynomial(0, v)  # All ones
        >>> hermite_polynomial(1, v)  # 2v
        >>> hermite_polynomial(2, v)  # 4v² - 2
    """
    if m < 0:
        raise ValueError(f"Hermite order must be non-negative, got m={m}")

    # Base cases
    if m == 0:
        return jnp.ones_like(v)
    if m == 1:
        return 2.0 * v

    # Recurrence relation: H_{m+1}(v) = 2v·H_m(v) - 2m·H_{m-1}(v)
    def recurrence_step(n, carry):
        H_prev, H_curr = carry
        H_next = 2.0 * v * H_curr - 2.0 * n * H_prev
        return (H_curr, H_next)

    H_0 = jnp.ones_like(v)
    H_1 = 2.0 * v

    # Apply recurrence from n=1 to n=m-1
    _, H_m = jax.lax.fori_loop(1, m, recurrence_step, (H_0, H_1))

    return H_m


@partial(jax.jit, static_argnames=['M'])
def hermite_polynomials_all(M: int, v: Array) -> Array:
    """
    Compute all Hermite polynomials H_0 through H_M simultaneously.

    More efficient than calling hermite_polynomial repeatedly, and works
    with JAX transformations like vmap.

    Uses the recurrence relation:
        H_0(v) = 1
        H_1(v) = 2v
        H_{m+1}(v) = 2v·H_m(v) - 2m·H_{m-1}(v)

    Args:
        M: Maximum polynomial order (will compute 0 through M inclusive)
        v: Velocity values, shape (..., Nv)

    Returns:
        Array of shape (M+1, ..., Nv) containing H_m(v) for m=0..M

    Example:
        >>> v = jnp.linspace(-2, 2, 100)
        >>> H_all = hermite_polynomials_all(5, v)  # shape (6, 100)
        >>> H_2 = H_all[2]  # Extract H_2(v)
    """
    if M < 0:
        raise ValueError(f"Maximum order M must be non-negative, got M={M}")

    # Initialize output array
    # Shape: (M+1, *v.shape)
    H_all = jnp.zeros((M + 1,) + v.shape, dtype=v.dtype)

    # Base cases
    H_all = H_all.at[0].set(jnp.ones_like(v))
    if M >= 1:
        H_all = H_all.at[1].set(2.0 * v)

    # Recurrence relation: H_{m+1} = 2v·H_m - 2m·H_{m-1}
    def recurrence_step(m, H_arr):
        H_next = 2.0 * v * H_arr[m] - 2.0 * m * H_arr[m - 1]
        return H_arr.at[m + 1].set(H_next)

    # Apply recurrence from m=1 to m=M-1
    if M >= 2:
        H_all = jax.lax.fori_loop(1, M, recurrence_step, H_all)

    return H_all


@partial(jax.jit, static_argnames=['m'])
def hermite_basis_function(m: int, v: Array, v_th: float = 1.0) -> Array:
    """
    Evaluate orthonormal Hermite basis function (Hermite function).

    Computes:
        ψ_m(v) = N_m · H_m(v/v_th) · exp(-v²/(2v_th²))

    where N_m = 1/√(2^m · m! · √π · v_th) ensures orthonormality:
        ∫ ψ_m(v) · ψ_n(v) dv = δ_{mn}

    These are the quantum harmonic oscillator eigenfunctions.

    Args:
        m: Basis function order
        v: Velocity values
        v_th: Ion thermal velocity √(2T_i/m_i) (default 1.0)

    Returns:
        Normalized Hermite function ψ_m(v)

    Example:
        >>> v = jnp.linspace(-10, 10, 1000)
        >>> psi_0 = hermite_basis_function(0, v, v_th=1.0)
        >>> integral = jnp.trapezoid(psi_0**2, v)  # Should be ~1.0
    """
    v_normalized = v / v_th

    # Gaussian weight exp(-v²/2)
    gaussian = jnp.exp(-0.5 * v_normalized**2)

    # Hermite polynomial
    H_m = hermite_polynomial(m, v_normalized)

    # Normalization (includes v_th scaling for correct integration)
    norm = hermite_normalization(m) / jnp.sqrt(v_th)

    return norm * H_m * gaussian


@partial(jax.jit, static_argnames=['M'])
def distribution_to_moments(
    g_v: Array,
    v_grid: Array,
    M: int,
    v_th: float = 1.0
) -> Array:
    """
    Project velocity distribution onto Hermite moments.

    Computes moments:
        g_m = ∫ g(v) · ψ_m(v) dv

    where ψ_m are orthonormal Hermite functions.

    Args:
        g_v: Distribution function values on velocity grid, shape (..., Nv)
        v_grid: Velocity grid points, shape (Nv,)
        M: Number of moments to compute (will return M+1 moments: 0 to M)
        v_th: Thermal velocity (must be > 0)

    Returns:
        Hermite moments array, shape (..., M+1)

    Notes:
        - Assumes v_grid has uniform spacing for trapezoidal rule
        - Integration performed via JAX's trapezoid function
        - Batch dimensions in g_v are preserved
        - For accurate results with M moments, recommend Nv ≥ 100*M grid points
        - Velocity grid should span at least ±5*v_th for exponential decay

    Example:
        >>> v = jnp.linspace(-10, 10, 1000)
        >>> # Maxwellian: should give g_0 ≈ √π^(1/4), others ≈ 0
        >>> g_v = jnp.exp(-0.5 * v**2) / jnp.sqrt(2 * jnp.pi)
        >>> moments = distribution_to_moments(g_v, v, M=5)
    """
    v_normalized = v_grid / v_th

    # Compute all Hermite polynomials at once: shape (M+1, Nv)
    H_all = hermite_polynomials_all(M, v_normalized)

    # Gaussian weight: shape (Nv,)
    gaussian = jnp.exp(-0.5 * v_normalized**2)

    # Normalization constants: shape (M+1,)
    m_values = jnp.arange(M + 1)
    log_norms = -0.5 * (
        m_values * jnp.log(2.0) +
        jax.scipy.special.gammaln(m_values + 1) +
        0.5 * jnp.log(jnp.pi)
    )
    norms = jnp.exp(log_norms) / jnp.sqrt(v_th)

    # Compute basis functions: ψ_m = N_m · H_m · exp(-v²/2)
    # Shape: (M+1, Nv)
    basis_functions = norms[:, None] * H_all * gaussian[None, :]

    # Compute integrands for all moments: shape (M+1, ..., Nv)
    # Add moment axis to front of g_v: (..., Nv) -> (1, ..., Nv)
    g_v_expanded = jnp.expand_dims(g_v, axis=0)

    # Reshape basis_functions to broadcast: (M+1, Nv) -> (M+1, 1, ..., 1, Nv)
    ndim_batch = g_v.ndim - 1
    basis_shape = (M + 1,) + (1,) * ndim_batch + (len(v_grid),)
    basis_bc = jnp.reshape(basis_functions, basis_shape)

    # Compute integrand: (M+1, ..., Nv)
    integrand = g_v_expanded * basis_bc

    # Integrate over velocity axis (last axis): (M+1, ...)
    moments = jnp.trapezoid(integrand, v_grid, axis=-1)

    # Move moment axis to end: (..., M+1)
    return jnp.moveaxis(moments, 0, -1)


@partial(jax.jit, static_argnames=['M'])
def moments_to_distribution(
    g_m: Array,
    v_grid: Array,
    v_th: float = 1.0,
    M: Optional[int] = None
) -> Array:
    """
    Reconstruct velocity distribution from Hermite moments.

    Computes:
        g(v) = Σ_m g_m · ψ_m(v)

    where ψ_m are orthonormal Hermite functions.

    Args:
        g_m: Hermite moments, shape (..., M+1)
        v_grid: Velocity grid for reconstruction, shape (Nv,)
        v_th: Thermal velocity (must be > 0)
        M: Number of moments (inferred from g_m if not provided)

    Returns:
        Reconstructed distribution, shape (..., Nv)

    Notes:
        - Truncation at M moments introduces approximation error
        - For accurate reconstruction, M should be large enough that
          |g_M| ≪ |g_0| (typically requiring M ≥ 10-20)

    Example:
        >>> v = jnp.linspace(-10, 10, 1000)
        >>> # Pure Maxwellian moments: only g_0 non-zero
        >>> moments = jnp.array([jnp.sqrt(jnp.sqrt(jnp.pi)), 0.0, 0.0])
        >>> g_v = moments_to_distribution(moments, v, v_th=1.0)
    """
    if M is None:
        M = g_m.shape[-1] - 1

    v_normalized = v_grid / v_th

    # Gaussian weight: shape (Nv,)
    gaussian = jnp.exp(-0.5 * v_normalized**2)

    # Compute all Hermite polynomials: shape (M+1, Nv)
    H_all = hermite_polynomials_all(M, v_normalized)

    # Compute normalization constants for all moments: shape (M+1,)
    m_values = jnp.arange(M + 1)
    log_norms = -0.5 * (
        m_values * jnp.log(2.0) +
        jax.scipy.special.gammaln(m_values + 1) +
        0.5 * jnp.log(jnp.pi)
    )
    norms = jnp.exp(log_norms) / jnp.sqrt(v_th)

    # Compute basis functions: ψ_m = N_m · H_m · exp(-v²/2)
    # Shape: (M+1, Nv)
    basis_functions = norms[:, None] * H_all * gaussian[None, :]

    # Contract moments with basis: (..., M+1) · (M+1, Nv) → (..., Nv)
    g_v = jnp.tensordot(g_m, basis_functions, axes=[[-1], [0]])

    return g_v


def check_orthogonality(
    m: int,
    n: int,
    v_grid: Array,
    v_th: float = 1.0,
    rtol: float = 1e-6
) -> dict[str, Array]:
    """
    Verify orthonormality condition for Hermite functions.

    Checks:
        ∫ ψ_m(v) · ψ_n(v) dv = δ_{mn}

    where ψ_m are the orthonormal Hermite functions (not polynomials).

    Args:
        m, n: Function orders to test
        v_grid: Velocity grid for integration
        v_th: Thermal velocity
        rtol: Relative tolerance for orthogonality check

    Returns:
        Dictionary with keys:
            - 'integral': Computed integral value
            - 'expected': Expected value (1 if m==n, else 0)
            - 'relative_error': |computed - expected| / max(|expected|, atol)
            - 'is_orthogonal': Boolean, True if error < rtol

    Example:
        >>> v = jnp.linspace(-10, 10, 1000)
        >>> result = check_orthogonality(2, 2, v)
        >>> result['is_orthogonal']  # Should be True
    """
    # Compute basis functions (not JIT since m, n are runtime values)
    psi_m = hermite_basis_function(m, v_grid, v_th)
    psi_n = hermite_basis_function(n, v_grid, v_th)

    # Integrate
    integrand = psi_m * psi_n
    integral = jnp.trapezoid(integrand, v_grid)

    # Expected value for orthonormal functions
    expected = 1.0 if m == n else 0.0

    # Relative error (handle expected = 0 case with absolute tolerance)
    atol = 1e-10
    if expected != 0.0:
        rel_error = jnp.abs(integral - expected) / jnp.abs(expected)
    else:
        rel_error = jnp.abs(integral - expected) / atol

    return {
        'integral': float(integral),
        'expected': expected,
        'relative_error': float(rel_error),
        'is_orthogonal': bool(rel_error < rtol)
    }


# ============================================================================
# Hermite Hierarchy Closures (Thesis §2.4)
# ============================================================================


@partial(jax.jit, static_argnames=['M'])
def closure_zero(g: Array, M: int) -> Array:
    """
    Simple truncation closure: gₘ₊₁ = 0.

    This is the simplest closure for the Hermite hierarchy, setting the
    (M+1)-th moment to zero. Valid when collision damping ensures gₘ is
    negligible (use check_hermite_convergence to verify).

    This is the default closure of physics.g_rhs for the highest
    retained moment. With finite collisions ν > 0, high moments are damped,
    making this closure reasonable for sufficiently large M.

    Args:
        g: Hermite moment array (shape: [Nz, Ny, Nx//2+1, M+1])
        M: Maximum moment index (highest retained moment)

    Returns:
        gₘ₊₁ = 0 with shape [Nz, Ny, Nx//2+1]

    Example:
        >>> grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        >>> g = jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1, 11), dtype=complex)
        >>> g_M_plus_1 = closure_zero(g, M=10)
        >>> jnp.all(g_M_plus_1 == 0.0)
        True

    Reference:
        Thesis §2.4 - Hermite hierarchy truncation and closure schemes
    """
    # Return zeros with correct shape
    return jnp.zeros_like(g[:, :, :, 0])


@partial(jax.jit, static_argnames=['M'])
def closure_symmetric(g: Array, M: int) -> Array:
    """
    Symmetric closure: gₘ₊₁ = gₘ₋₁.

    This closure assumes symmetry in the Hermite hierarchy, setting the
    (M+1)-th moment equal to the (M-1)-th moment. Generally provides better
    convergence properties than simple truncation.

    With finite collisions ν > 0 and sufficiently large M, results should be
    independent of closure choice (use check_hermite_convergence to verify).

    Physical motivation: In the Hermite recurrence relation, each moment couples
    to its neighbors. The symmetric closure preserves this coupling structure
    better than truncation.

    Args:
        g: Hermite moment array (shape: [Nz, Ny, Nx//2+1, M+1])
            Must contain at least M-1 moments (M ≥ 2)
        M: Maximum moment index (highest retained moment, must be ≥ 2)

    Returns:
        gₘ₊₁ = gₘ₋₁ with shape [Nz, Ny, Nx//2+1]

    Raises:
        ValueError: If M < 2 (need at least 3 moments for symmetric closure)

    Example:
        >>> grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        >>> g = jax.random.normal(jax.random.PRNGKey(42),
        ...                       (grid.Nz, grid.Ny, grid.Nx//2+1, 11), dtype=complex)
        >>> g_M_plus_1 = closure_symmetric(g, M=10)
        >>> jnp.allclose(g_M_plus_1, g[:, :, :, 9])
        True

    Reference:
        Thesis §2.4 - Symmetric closure for improved convergence
    """
    if M < 2:
        raise ValueError(
            f"Symmetric closure requires M ≥ 2 (need gₘ₋₁ for gₘ₊₁ = gₘ₋₁), got M={M}"
        )

    # Return gₘ₋₁ as the closure for gₘ₊₁
    return g[:, :, :, M - 1]


def check_hermite_convergence(
    g: Array,
    threshold: float = 1e-3,
    account_for_rfft: bool = True
) -> dict[str, Any]:
    """
    Check convergence of truncated Hermite hierarchy.

    Verifies that energy in the highest retained moment gₘ is negligible
    compared to total moment energy. If not converged, increasing M or
    collision frequency ν is recommended.

    Convergence criterion:
        E_M / E_total < threshold

    where E_M = |gₘ|² and E_total = Σ|gₙ|² over all moments.

    Note: E_total here is the unweighted per-moment spectrum sum, used only as
    a truncation metric — it is NOT the conserved free energy when Λ ≠ ∞ or
    under the symmetric closure (see diagnostics.hermite_free_energy, which
    weights m=0 by w₀ = 1 - 1/Λ and, for closure="symmetric", m=M by
    p_M = √M/(√M + √(M+1))).

    The energy calculation accounts for rfft format: modes with kx > 0 are
    counted twice (complex conjugate pairs), while kx = 0 is counted once.

    Args:
        g: Hermite moment array (shape: [Nz, Ny, Nx//2+1, M+1])
        threshold: Convergence threshold (default: 1e-3 = 0.1%)
        account_for_rfft: If True, properly weight rfft modes (default: True)

    Returns:
        Dictionary with keys:
            - 'is_converged': bool, True if E_M / E_total < threshold
            - 'energy_fraction': float, actual E_M / E_total ratio
            - 'max_moment_index': int, M (highest moment index)
            - 'energy_highest_moment': float, E_M
            - 'energy_total': float, E_total = Σ|gₙ|²
            - 'recommendation': str, human-readable guidance

    Example:
        >>> grid = SpectralGrid3D.create(Nx=32, Ny=32, Nz=16)
        >>> # Converged case: exponentially decaying moments
        >>> g = jnp.zeros((grid.Nz, grid.Ny, grid.Nx//2+1, 11), dtype=complex)
        >>> for m in range(11):
        ...     g = g.at[:, :, :, m].set(jnp.exp(-m) * (1 + 0j))
        >>> result = check_hermite_convergence(g)
        >>> result['is_converged']
        True
        >>> # Not converged case: uniform moments
        >>> g_uniform = jnp.ones_like(g)
        >>> result_bad = check_hermite_convergence(g_uniform)
        >>> result_bad['is_converged']
        False

    Reference:
        Thesis §2.4 - Convergence requirements for Hermite truncation
    """
    M = g.shape[3] - 1  # Maximum moment index

    if M < 1:
        raise ValueError(f"Need at least 2 moments (M ≥ 1) for convergence check, got M={M}")

    # Compute energy in each moment: |gₙ|² summed over spatial modes
    # Shape: [M+1]
    if account_for_rfft:
        # rfft format: kx=0 plane counted once, kx>0 counted twice
        # g has shape [Nz, Ny, Nx//2+1, M+1]

        # Vectorized calculation: compute |g|² for all moments at once
        g_squared = jnp.abs(g) ** 2

        # Energy from kx=0 plane (counted once): sum over (z, y) for each moment
        energy_kx0 = jnp.sum(g_squared[:, :, 0, :], axis=(0, 1))

        # Energy from kx>0 planes (counted twice): sum over (z, y, kx>0) for each moment
        energy_kx_pos = jnp.sum(g_squared[:, :, 1:, :], axis=(0, 1, 2))

        # Combine with proper weighting
        energy_per_moment = energy_kx0 + 2.0 * energy_kx_pos
    else:
        # Simple sum without rfft correction (for testing or debug)
        energy_per_moment = jnp.sum(jnp.abs(g)**2, axis=(0, 1, 2))

    # Total energy and highest moment energy
    energy_total = jnp.sum(energy_per_moment)
    energy_highest = energy_per_moment[M]

    # Avoid division by zero
    if energy_total == 0:
        return {
            'is_converged': True,
            'energy_fraction': 0.0,
            'max_moment_index': M,
            'energy_highest_moment': 0.0,
            'energy_total': 0.0,
            'recommendation': 'All moments are zero - trivially converged.'
        }

    # Compute energy fraction
    energy_fraction = energy_highest / energy_total

    # Check convergence
    is_converged = energy_fraction < threshold

    # Generate recommendation
    if is_converged:
        recommendation = (
            f"✓ Converged: Energy in g_{M} is {100*energy_fraction:.2f}% of total "
            f"(< {100*threshold:.2f}% threshold). Truncation at M={M} is valid."
        )
    else:
        # Estimate required M for convergence (assuming exponential decay)
        # E_m ~ exp(-α·m), want E_M / E_total < threshold
        # Very rough estimate: increase M by ~10-20% typically helps
        suggested_M = int(M * 1.2)
        recommendation = (
            f"✗ Not converged: Energy in g_{M} is {100*energy_fraction:.2f}% of total "
            f"(≥ {100*threshold:.2f}% threshold). "
            f"Recommendations: (1) Increase M to ~{suggested_M}, or "
            f"(2) Increase collision frequency ν to damp high moments."
        )

    return {
        'is_converged': bool(is_converged),
        'energy_fraction': float(energy_fraction),
        'max_moment_index': int(M),
        'energy_highest_moment': float(energy_highest),
        'energy_total': float(energy_total),
        'recommendation': recommendation
    }


# =============================================================================
# Hermite Streaming Matrix (for integrating factor timestepping)
# =============================================================================


HERMITE_CLOSURES: tuple[str, ...] = ("zero", "symmetric")

# Advancing g in place (Issue #186) makes the lawson_rk4 slice step,
# physics.g_rhs and imex_solve work on one slice, moment or k_z plane at a time,
# so their temporaries do not grow with M. On a GPU it costs time: at 128³,
# M = 128 on an A100, imex_rk222 takes 0.93 s per step in place against 0.25 s
# all at once, and lawson_rk4 0.37 s against 0.32 s. So g goes in place only
# when the all-at-once step does not fit:
#
# - None (default): on a device that reports its memory (GPU, TPU), in place
#   when ALL_AT_ONCE_G_BUFFERS g-sized buffers exceed its limit. On the CPU, in
#   place above CPU_IN_PLACE_ENTRIES entries, where it is as fast or faster.
# - An int: in place above that many entries, on any device. Tests and
#   benchmarks force a path with 0 or 2**62.
HERMITE_IN_PLACE_ENTRIES: Optional[int] = None

# The all-at-once imex_rk222 and lawson_rk4 steps both peak at 11.1 g-sized
# buffers on an A100 (22.7 GiB for a 2.05 GiB g at 128³, M = 128, complex128,
# through gandalf_step). 14 leaves a fifth of the memory for cuFFT workspace
# and fragmentation.
ALL_AT_ONCE_G_BUFFERS = 14
CPU_IN_PLACE_ENTRIES = 2**26  # 1 GiB in complex128


def advance_in_place(g: Any) -> bool:
    """Whether g (an array, tracer or jax.ShapeDtypeStruct) is advanced in place.

    Decided at trace time on the default device; see HERMITE_IN_PLACE_ENTRIES.
    On a mesh, g is one device's share.
    """
    size = int(np.prod(g.shape))
    if HERMITE_IN_PLACE_ENTRIES is not None:
        return size > HERMITE_IN_PLACE_ENTRIES
    device = jax.local_devices()[0]
    stats = None if device.platform == "cpu" else device.memory_stats()
    if not stats or "bytes_limit" not in stats:
        return size > CPU_IN_PLACE_ENTRIES
    return ALL_AT_ONCE_G_BUFFERS * size * jnp.dtype(g.dtype).itemsize > stats["bytes_limit"]


def validate_closure(closure: str, M: int) -> None:
    """Raise ValueError unless `closure` is a supported closure valid for this M."""
    if closure not in HERMITE_CLOSURES:
        raise ValueError(
            f"closure must be one of {HERMITE_CLOSURES} (got {closure!r})"
        )
    if closure == "symmetric" and M < 2:
        raise ValueError(
            f"closure='symmetric' requires M >= 2 (g_(M+1) = g_(M-1)), got M={M}"
        )


@lru_cache(maxsize=None)
def compute_streaming_matrix(M: int, Lambda: float = 1.0, closure: str = "zero"):
    """
    Build the (M+1)x(M+1) parallel streaming coupling matrix T.

    The linear streaming term in the Hermite hierarchy is:
        dg/dt = ... - sqrt(beta_i) * ikz * T @ g + ...

    Matrix elements follow from the Hermite recurrence (Thesis Eq. 2.7-2.9):
        T[0, 1] = sqrt(1/2)                    (g0 couples to g1)
        T[1, 0] = (1 - 1/Lambda) / sqrt(2)     (g1 kinetic correction)
        T[1, 2] = sqrt(2/2) = 1                (g1 couples to g2)
        T[m, m+1] = sqrt((m+1)/2)  for m >= 2  (standard recurrence)
        T[m, m-1] = sqrt(m/2)      for m >= 2  (standard recurrence)

    Note: T is NOT symmetric when Lambda != infinity due to the kinetic
    correction in the g1 equation.

    The closure for the unretained g_(M+1) sets the last row:
        "zero":      g_(M+1) = 0        -> T[M, M-1] = sqrt(M/2)
        "symmetric": g_(M+1) = g_(M-1)  -> T[M, M-1] = sqrt(M/2) + sqrt((M+1)/2)
    The symmetric closure breaks the symmetry of T in the last row, so the
    unweighted sum of |g_m|^2 is no longer conserved by streaming alone; the
    conserved free energy then weights g_M by p_M = sqrt(M)/(sqrt(M)+sqrt(M+1))
    (see free_energy_weights and diagnostics.hermite_free_energy).

    Args:
        M: Maximum Hermite moment index (array has M+1 moments: g0..gM)
        Lambda: Kinetic closure parameter (default 1.0)
        closure: Truncation closure, "zero" (default) or "symmetric"

    Returns:
        numpy.ndarray: (M+1, M+1) coupling matrix
    """
    import numpy as np

    validate_closure(closure, M)

    size = M + 1
    T = np.zeros((size, size), dtype=np.float64)

    if M >= 1:
        # g0 equation: couples to g1 with coefficient sqrt(1/2)
        T[0, 1] = np.sqrt(1.0 / 2.0)
        # g1 equation: couples to g0 with kinetic correction
        T[1, 0] = (1.0 - 1.0 / Lambda) / np.sqrt(2.0)

    if M >= 2:
        # g1 equation: couples to g2
        T[1, 2] = np.sqrt(2.0 / 2.0)  # = 1.0

    # Standard Hermite recurrence for m >= 2
    for m in range(2, M + 1):
        if m + 1 <= M:
            T[m, m + 1] = np.sqrt((m + 1) / 2.0)
        T[m, m - 1] = np.sqrt(m / 2.0)

    if closure == "symmetric":
        T[M, M - 1] += np.sqrt((M + 1) / 2.0)

    return T


def free_energy_weights(M: int, Lambda: float, closure: str = "zero") -> np.ndarray:
    """
    Per-moment weights p_m of the Hermite free energy W = sum_m p_m sum_k |g_m,k|^2.

    W is the quadratic form g^T P g, P = diag(p_0, ..., p_M), conserved by the
    collisionless streaming dynamics dg/dt = -i sqrt(beta_i) k_par T g. It is
    conserved iff P T is symmetric (detailed balance p_m T[m,n] = p_n T[n,m]),
    with T from compute_streaming_matrix(M, Lambda, closure). Normalising
    p_1 = 1 gives

        p_0 = 1 - 1/Lambda                          (T[1,0] kinetic correction)
        p_m = 1                  for 1 <= m < M
        p_M = 1                                     closure = "zero"
        p_M = sqrt(M) / (sqrt(M) + sqrt(M+1))       closure = "symmetric"

    The symmetric value follows from p_(M-1) T[M-1,M] = p_M T[M,M-1] with
    T[M-1,M] = sqrt(M/2) and T[M,M-1] = sqrt(M/2) + sqrt((M+1)/2); it is
    close to 1/2 (0.4495 at M = 2, 0.4881 at M = 10). The same weights make
    the {Psi, .} field-line part of grad_par conserve W, because the closure
    enters that term through the same combination of g_(M+1) and g_(M-1).

    Args:
        M: Maximum Hermite moment index (weights for g_0..g_M)
        Lambda: Kinetic closure parameter Lambda (alpha = -1/Lambda)
        closure: Truncation closure, "zero" (default) or "symmetric"

    Returns:
        numpy.ndarray: (M+1,) float64 weights p_m. For M = 0 this is [p_0].

    Raises:
        ValueError: If closure is unknown, or "symmetric" with M < 2 (the
            code defines the symmetric closure only for M >= 2; see
            validate_closure).
    """
    validate_closure(closure, M)

    weights = np.ones(M + 1, dtype=np.float64)
    weights[0] = 1.0 - 1.0 / Lambda
    if closure == "symmetric":
        weights[M] = np.sqrt(M) / (np.sqrt(M) + np.sqrt(M + 1))
    return weights


def compute_streaming_eigensystem(
    M: int, Lambda: float = 1.0, closure: str = "zero"
) -> tuple[Array, Array, Array, Array]:
    """
    Precompute eigendecomposition of the streaming matrix T = P @ diag(evals) @ P_inv.

    Used by the integrating factor timestepper to apply exact phase rotation
    for the linear streaming operator exp(-i * sqrt(beta_i) * kz * T * dt).

    Uses eigh (symmetric solver) when T is symmetric (guaranteed real eigenvalues,
    better conditioned), falling back to eig (general solver) for asymmetric T.

    The arrays are in the default precision of the call: float32, or float64
    under jax_enable_x64. The cache is keyed on that precision, so a process
    that switches it gets arrays of the new precision (Issue #194).

    Args:
        M: Maximum Hermite moment index
        Lambda: Kinetic closure parameter
        closure: Truncation closure, "zero" (default) or "symmetric"

    Returns:
        T_jax: (M+1, M+1) JAX array of the streaming matrix
        eigenvalues: (M+1,) array of eigenvalues
        P: (M+1, M+1) matrix of right eigenvectors
        P_inv: (M+1, M+1) inverse of P
    """
    return _streaming_eigensystem(M, Lambda, closure, jnp.result_type(float))


@lru_cache(maxsize=None)
def _streaming_eigensystem(
    M: int, Lambda: float, closure: str, default_float: Any
) -> tuple[Array, Array, Array, Array]:
    """compute_streaming_eigensystem, cached per default float dtype."""
    T = compute_streaming_matrix(M, Lambda, closure)

    # Use symmetric solver when T is symmetric (real eigenvalues guaranteed)
    if np.allclose(T, T.T, atol=1e-14):
        eigenvalues, P = np.linalg.eigh(T)
        P_inv = P.T  # Orthogonal for symmetric T
    else:
        eigenvalues, P = np.linalg.eig(T)
        P_inv = np.linalg.inv(P)
        if np.any(np.abs(np.imag(eigenvalues)) > 1e-6):
            raise ValueError(
                f"Streaming matrix has complex eigenvalues for Lambda={Lambda}. "
                f"The integrating factor requires real eigenvalues for unitarity. "
                f"RMHD ordering requires Lambda >= 1. Max |Im(eigenvalue)| = "
                f"{np.max(np.abs(np.imag(eigenvalues))):.2e}"
            )

    return jnp.array(T), jnp.array(eigenvalues), jnp.array(P), jnp.array(P_inv)


# =============================================================================
# IMEX implicit operator (Issues #137, #187)
# =============================================================================


# Bounded cache: (M, hyper_n) are small integers drawn from a handful of
# supported values, so 32 entries easily covers any realistic set of runs
# and guards against unbounded growth if someone does exploration sweeps.
@lru_cache(maxsize=32)
def _damping_diag(M: int, hyper_n: int) -> "np.ndarray":
    """
    Per-moment hyper-collisional damping rates normalized to max-rate unity.

    Returns a length-(M+1) float64 numpy array where entry m is:
        -(m/M)^hyper_n  for m >= 2
        0               for m = 0, 1 (conserves particle number and momentum)

    The damping operator D is diag(nu * _damping_diag(M, hyper_n)) in the
    implicit-stage matrix L. Matches the damping block the Lawson path
    applies as exp(-nu * (m/M)^n * dt).
    """
    m = np.arange(M + 1, dtype=np.float64)
    # Safe even for M=0,1: we only use m>=2 entries, and zero them out below.
    # Avoid division by zero for M=0: float division would produce inf/nan,
    # but the mask discards those values anyway.
    if M >= 2:
        rates = -((m / M) ** hyper_n)
    else:
        rates = np.zeros_like(m)
    rates[:2] = 0.0
    # Freeze the cached array: lru_cache returns the same object on every
    # call, so an accidental in-place edit by a caller would silently corrupt
    # future results.
    rates.setflags(write=False)
    return rates


def _streaming_diagonals(
    M: int, Lambda: float, closure: str, dtype: Any
) -> tuple[Array, Array]:
    """Off-diagonals of the streaming matrix T as device arrays.

    Returns (upper, lower), each of length M: upper[m] = T[m, m+1] and
    lower[m] = T[m+1, m]. T has a zero diagonal. Lambda may be a traced
    value: it enters only lower[0] = T[1, 0] = (1 - 1/Λ)/√2, and the rest
    comes from compute_streaming_matrix at trace time.
    """
    T = compute_streaming_matrix(M, 1.0, closure)  # Λ = 1 gives T[1, 0] = 0
    upper = jnp.asarray(np.diag(T, 1), dtype)
    lower = jnp.asarray(np.diag(T, -1), dtype)
    if M >= 1:
        lower = lower.at[0].set(jnp.asarray((1.0 - 1.0 / Lambda) / np.sqrt(2.0), dtype))
    return upper, lower


def apply_implicit_operator(
    g: Array,
    kz: Array,
    beta_i: float,
    nu: float,
    Lambda: float,
    M: int,
    hyper_n: int,
    closure: str = "zero",
) -> Array:
    """
    Return L·g for the IMEX implicit operator L = -i·√β_i·k_z·T + ν·D.

    T is the tridiagonal streaming matrix (compute_streaming_matrix) and D
    is diag(_damping_diag(M, hyper_n)), so L·g is a three-point stencil in
    m. g has the state layout (Nz, Ny, Nx//2+1, M+1). M, hyper_n and
    closure must be static under jit; the other arguments may be traced.
    """
    upper, lower = _streaming_diagonals(M, Lambda, closure, kz.dtype)
    pad = [(0, 0)] * (g.ndim - 1)
    T_g = jnp.pad(lower * g[..., :-1], pad + [(1, 0)]) + jnp.pad(
        upper * g[..., 1:], pad + [(0, 1)]
    )
    damping = nu * jnp.asarray(_damping_diag(M, hyper_n), kz.dtype)
    streaming = -1j * jnp.sqrt(beta_i) * kz[:, None, None, None]
    return streaming * T_g + damping * g


class ImexFactor(NamedTuple):
    """
    Factors of A = I - γ·dt·L for every k_z, built by imex_factor.

    A is tridiagonal in m with A[m, m] = 1 - γ·dt·ν·D_m ≥ 1 and
    A[m, m±1] = i·a·T[m, m±1], a = γ·dt·√β_i·k_z. Each field has shape
    (M+1, Nz). imex_solve eliminates upward from m = M and substitutes
    downward from m = 0:

        b'_M = b_M,   b'_m = b_m + eliminate[m]·b'_(m+1)          (m < M)
        x_0 = inv_pivot[0]·b'_0,   x_m = inv_pivot[m]·(b'_m + substitute[m]·x_(m-1))

    with eliminate[m] = -A[m, m+1]/q_(m+1), substitute[m] = -A[m, m-1] and
    pivots q_M = A[M, M], q_m = A[m, m] + a²·T[m, m+1]·T[m+1, m]/q_(m+1).
    """

    eliminate: Array  # complex, (M+1, Nz); eliminate[M] = 0
    substitute: Array  # complex, (M+1, Nz); substitute[0] = 0
    inv_pivot: Array  # real, (M+1, Nz); 1/q_m


def imex_factor(
    kz: Array,
    dt: Array | float,
    gamma: float,
    beta_i: float,
    nu: float,
    Lambda: float,
    M: int,
    hyper_n: int,
    closure: str = "zero",
) -> ImexFactor:
    """
    Factor A = I - γ·dt·L, L = -i·√β_i·k_z·T + ν·D, for each k_z.

    A is tridiagonal along m for both closures and every Λ, because T is
    tridiagonal and D diagonal, so the factorization costs O(M) per k_z.
    It runs on the device: M, hyper_n and closure must be static under jit,
    and kz, dt, beta_i, nu and Lambda may be traced (dt can change every
    step).

    Pivots. The elimination runs from m = M up to m = 0 without row
    exchanges. Every pivot q_m is real, because a²·T[m, m+1]·T[m+1, m] is
    real. For m ≥ 1 the product T[m, m+1]·T[m+1, m] is positive, so
    q_m ≥ A[m, m] ≥ 1 for every Λ. Λ enters only the last pivot,
    q_0 = 1 + a²·(1 - 1/Λ)/(2·q_1). It is ≥ 1 for Λ ≥ 1 and Λ < 0. For
    0 < Λ < 1 it can approach zero, but det A = q_0·q_1·…·q_M, so q_0
    vanishes only where A itself is singular, i.e. where 1/(γ·dt) is an
    eigenvalue of L. That range of Λ has a growing mode, so this can
    happen, and any solver fails there. A top-down sweep instead has the
    pivot 1 - a²·(1/Λ - 1)/2 at m = 1, which vanishes at a finite a even
    where A is nonsingular.

    Precision. The factors and the solve run in the dtype of kz:
    float32/complex64 by default and float64/complex128 when JAX's x64
    mode is on (JAX_ENABLE_X64=1 in the environment, or
    jax.config.update("jax_enable_x64", True) before any array is
    created). x64 mode makes the whole run float64; there is no separate
    switch for the solve. In complex64 at M = 128 the forward error of
    imex_solve is below 2e-7 (relative) for √β_i·k_z·dt ≤ 1 and below 7e-7
    up to √β_i·k_z·dt = 30, for both closures and Λ ∈ {1, ±√5, 0.5}. The
    batched dense LU it replaced reached 2.2e-6 there (Issue #140).

    Args:
        kz: (Nz,) parallel wavenumbers.
        dt: Timestep (may be traced).
        gamma: Diagonal coefficient of the implicit tableau.
        beta_i: Ion plasma beta.
        nu: Collision frequency (the rates in D are normalized to 1 at m = M).
        Lambda: Kinetic parameter Λ (enters T[1, 0]).
        M: Maximum Hermite moment index.
        hyper_n: Hyper-collision exponent.
        closure: Truncation closure, "zero" (default) or "symmetric".

    Returns:
        ImexFactor for imex_solve.
    """
    upper, lower = _streaming_diagonals(M, Lambda, closure, kz.dtype)
    dt_gamma = dt * gamma
    diagonal = 1.0 - (dt_gamma * nu) * jnp.asarray(_damping_diag(M, hyper_n), kz.dtype)
    a = (dt_gamma * jnp.sqrt(beta_i)) * kz
    a2_coupling = (a * a)[None, :] * (upper * lower)[:, None]  # (M, Nz)

    def pivot(q_next: Array, row: tuple[Array, Array]) -> tuple[Array, Array]:
        a_mm, a2_coupling_m = row
        q = a_mm + a2_coupling_m / q_next
        return q, q

    q_M = jnp.broadcast_to(diagonal[M], a.shape)
    _, q_below = jax.lax.scan(pivot, q_M, (diagonal[:M], a2_coupling), reverse=True)
    q = jnp.concatenate([q_below, q_M[None]])  # (M+1, Nz)

    zero = jnp.zeros_like(a)[None]
    eliminate = -1j * jnp.concatenate([a * upper[:, None] / q[1:], zero])
    substitute = -1j * jnp.concatenate([zero, a * lower[:, None]])
    return ImexFactor(eliminate, substitute, 1.0 / q)


def imex_solve(factor: ImexFactor, rhs: Array) -> Array:
    """
    Solve (I - γ·dt·L)·x = rhs for every Fourier point, with factors from imex_factor.

    rhs has the state layout (Nz, Ny, Nx//2+1, M+1). Both sweeps run over m
    on the moment-major transpose, so each step reads one contiguous
    (Nz, Ny, Nx//2+1) slab, and update that one buffer in place: the solve
    needs one g-sized temporary beyond rhs. Cost is O(M) per Fourier point.
    When advance_in_place(rhs) (Issue #186), the solve runs one k_z plane at a time and overwrites rhs in place, so the
    temporary is one plane instead.

    Returns:
        x with the shape and dtype of rhs.
    """
    if advance_in_place(rhs):

        def plane(iz: Array, b: Array) -> Array:
            f = ImexFactor(*(jax.lax.dynamic_slice_in_dim(a, iz, 1, axis=1) for a in factor))
            x = _imex_solve_all(f, jax.lax.dynamic_slice_in_dim(b, iz, 1, axis=0))
            return jax.lax.dynamic_update_slice_in_dim(b, x, iz, axis=0)

        rhs = rhs.astype(jnp.result_type(rhs, *factor))  # as the all-at-once solve
        return jax.lax.fori_loop(0, rhs.shape[0], plane, rhs)
    return _imex_solve_all(factor, rhs)


def _imex_solve_all(factor: ImexFactor, rhs: Array) -> Array:
    """imex_solve for all of rhs at once (factor fields (M+1, Nz) match rhs's Nz)."""
    x = jnp.moveaxis(rhs, -1, 0)
    M = x.shape[0] - 1

    def row(a: Array, m: Array) -> Array:
        return jax.lax.dynamic_index_in_dim(a, m, 0, keepdims=False)

    def eliminate(i: Array, x: Array) -> Array:
        m = M - 1 - i
        x_m = row(x, m) + row(factor.eliminate, m)[:, None, None] * row(x, m + 1)
        return jax.lax.dynamic_update_index_in_dim(x, x_m, m, 0)

    def substitute(m: Array, x: Array) -> Array:
        x_m = row(factor.inv_pivot, m)[:, None, None] * (
            row(x, m) + row(factor.substitute, m)[:, None, None] * row(x, m - 1)
        )
        return jax.lax.dynamic_update_index_in_dim(x, x_m, m, 0)

    x = jax.lax.fori_loop(0, M, eliminate, x)
    x = x.at[0].multiply(factor.inv_pivot[0][:, None, None])
    x = jax.lax.fori_loop(1, M + 1, substitute, x)
    return jnp.moveaxis(x, 0, -1)
