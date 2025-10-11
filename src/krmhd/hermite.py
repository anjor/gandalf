"""
Hermite basis functions for kinetic velocity-space representation.

This module implements orthonormal Hermite functions (quantum harmonic oscillator
eigenfunctions) for expanding the electron distribution function along the parallel
direction:

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
with Maxwellian weight exp(-v²/2) ensures proper treatment of the thermal
distribution and enables efficient moment truncation.
"""

from functools import partial
from typing import Optional
import jax
import jax.numpy as jnp
from jax import Array


@jax.jit
def hermite_normalization(m: int) -> Array:
    """
    Compute normalization constant for orthonormal Hermite functions.

    Returns 1/√(2^m · m! · √π) for Hermite functions with Maxwellian weight exp(-v²/2).

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
        v_th: Thermal velocity (default 1.0)

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
