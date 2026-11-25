"""
Multirate Lawson-RK4 over the phase-velocity slices (Issue #191, item 1).

scheme="lawson_rk4" advances every slice of the Hermite hierarchy with one
step, sized for the fastest slice. Slice j is a passive scalar advected by
Φ + √β_i λ_j Ψ (_slice_advection_rhs), and the largest |λ_j| grows like
√(2M), so most slices move much slower than the one that sets dt.
scheme="lawson_rk4_multirate" lets each slice take its own step:

- z± advance with the macro step H by lawson_rk4's own Lawson midpoint RK2
  (_elsasser_midpoint), giving z± at t, at t + H/2 (the Euler predictor) and
  at t + H.
- Slice j takes n_j Lawson-RK4 substeps of H/n_j, with
      n_j = ceil(H·v_j / (C·Δ)),  v_j = max over space of |∇⊥(Φ + √β_i λ_j Ψ)|
  at time t, Δ = min(Δx, Δy, Δz) and C the substep Courant number. This is
  compute_cfl_timestep's Courant rule applied slice by slice. The slices
  with the same n_j form one rate class.
- Φ and Ψ at the substep stage times come from the quadratic through the
  three z± states of the macro step, evaluated on the real-space gradients.
- Collisions, resistive and k_z damping act once per macro step, as in
  lawson_rk4.

Order. In lawson_rk4 the Hermite stages see z± at t, at the predictor and at
t + H. The predictor is off by O(H²), so each Hermite step makes an O(H³)
error and the coupled scheme is second order, like z± itself. The quadratic
interpolant is off by the predictor's O(H²) (weighted by 4θ(1 - θ) ≤ 1) plus
its own O(H³), so the multirate scheme keeps second order. For a flow that
does not change in time it is exact, and each slice is then integrated by
RK4 at fourth order. A linear interpolant would also keep second order, but
only the quadratic reproduces lawson_rk4's stage values: at n_j = 1 its
weights are exactly 0 and 1, and the step is lawson_rk4's to round-off.

Interpolating in real space costs no FFTs. It interpolates the Alfvén phase
exp(±i k_z t) of the high-k_z modes too, with a relative error below
0.008·(k_z H)³. A macro step within the v_A Courant limit has
k_z,max·H ≤ 2πC/3, so the error is at most 0.2% at C = 0.3, at k_z,max.

Stability. The interpolated Φ and Ψ are real band-limited fields, so each
substep's advection operator is skew-adjoint and RK4 is stable for
h_j·ρ_j ≤ 2√2 (compute_cfl_timestep). The Courant rule bounds ρ_j by
k⊥,max·v_j and k⊥,max·Δ ≤ 2√2·π/3, so C ≤ 0.955 keeps every substep inside the
RK4 limit for a frozen flow. The macro step H must still satisfy the Elsasser
and v_A terms of the Courant estimate (compute_cfl_timestep(mode="multirate")):
z± use midpoint RK2, which needs dissipation at its own Courant limit, as in
imex_rk222. docs/multirate_slices_study.md measures stability, order,
conservation and cost.

Exact radii. The Courant rule is loose: in developed turbulence the radius
ρ_j of every slice is about 1.5·v_j/Δ, half the worst case k⊥,max·v_j.
compute_cfl_timestep(mode="exact") uses ρ of the fastest slice. Its multirate
counterpart is exact_substep_cfl, which turns each slice's own radius into a
per-slice Courant number C_j, so that h_j·ρ_j ≤ cfl_safety·2√2 at the state it
is computed on. Between calls n_j follows v_j step by step, and C_j keeps the
ratio ρ_j/v_j of the last call.
"""

import math
import warnings
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from krmhd.hermite import compute_streaming_eigensystem
from krmhd.physics import KRMHDState
from krmhd.timestepping import (
    RK4_IMAGINARY_AXIS_LIMIT,
    KRMHDFields,
    _apply_moment_matrix,
    _collision_factors,
    _elsasser_midpoint,
    _perp_gradients,
    _post_step_dissipation,
    _slice_advection_rhs,
    slice_advection_radius,
)

# Cap on the substeps of one slice in one macro step. A macro step within the
# Elsasser Courant limit needs at most about 1 + √β_i·max|λ_j| (about 11 at
# β_i = 1, M = 64); the cap only bounds the loop when the fields blow up.
MAX_SUBSTEPS = 4096

# Relative slack on the Courant rule, so that round-off in v_j does not add a
# substep: a dt from compute_cfl_timestep(mode="courant") gives every slice one.
COURANT_RTOL = 1e-5

Gradients = Tuple[Array, Array, Array, Array]


def _slice_courant_speeds(grads: Gradients, slice_speeds: Array) -> Array:
    """v_j = max over space of |∇⊥(Φ + c_j Ψ)| for each slice speed c_j.

    grads are the real-space (∂xΦ, ∂yΦ, ∂xΨ, ∂yΨ) of _perp_gradients. One
    reduction per slice, in sequence, so no [S, Nz, Ny, Nx] temporary.
    """
    dphi_dx, dphi_dy, dpsi_dx, dpsi_dy = grads

    def speed(c: Array) -> Array:
        return jnp.sqrt(jnp.max((dphi_dx + c * dpsi_dx) ** 2 + (dphi_dy + c * dpsi_dy) ** 2))

    return jax.lax.map(speed, slice_speeds)


def _substep_counts(speeds: Array, dt: Array, substep_courant: Array) -> Array:
    """n_j = ceil(dt·v_j / substep_courant), clipped to [1, MAX_SUBSTEPS] (int32).

    The ratio is first reduced by COURANT_RTOL. A non-finite ratio (NaN
    fields) takes one substep, so a blown-up run fails as fast as lawson_rk4
    would.
    """
    ratio = dt * speeds / substep_courant * (1.0 - COURANT_RTOL)
    n = jnp.ceil(jnp.where(jnp.isfinite(ratio), ratio, 1.0))
    return jnp.clip(n, 1, MAX_SUBSTEPS).astype(jnp.int32)


@partial(jax.jit, static_argnames=["Nz", "Ny", "Nx"])
def multirate_substeps(
    z_plus: Array,
    z_minus: Array,
    slice_speeds: Array,
    dt: float,
    substep_courant: float,
    kx: Array,
    ky: Array,
    Nz: int,
    Ny: int,
    Nx: int,
) -> Array:
    """Substeps n_j that scheme="lawson_rk4_multirate" gives each slice in a step of dt.

    Args:
        z_plus, z_minus: Elsasser fields at the start of the step [Nz, Ny, Nx//2+1]
        slice_speeds: √β_i λ_j, one per slice [S]
        dt: Macro step
        substep_courant: C·min(Δx, Δy, Δz), the largest h_j·v_j of a substep
        kx, ky: Wavenumber arrays
        Nz, Ny, Nx: Grid dimensions (static)

    Returns:
        n_j [S] (int32). Σ_j n_j is the step's count of slice updates;
        lawson_rk4 at the same dt makes S.
    """
    grads = _perp_gradients(z_plus, z_minus, kx, ky, Nz, Ny, Nx)
    return _substep_counts(_slice_courant_speeds(grads, slice_speeds), dt, substep_courant)


def exact_substep_cfl(
    state: KRMHDState, cfl_safety: float = 0.8, closure: str = "zero", tol: float = 1e-3
) -> np.ndarray:
    """Per-slice Courant numbers C_j from the exact radius of every slice.

    C_j = cfl_safety·2√2·v_j / (ρ_j·Δ), with ρ_j from slice_advection_radius
    and v_j = max|∇⊥(Φ + √β_i λ_j Ψ)| on `state`. Passed as
    gandalf_step(substep_cfl=C) or StepParams(substep_cfl=tuple(C)), it gives
    h_j·ρ_j ≤ cfl_safety·2√2 for every slice on `state`: the multirate form of
    compute_cfl_timestep(mode="exact"). The entries follow the eigenvalues of
    compute_streaming_eigensystem(state.M, state.Lambda, closure).

    A slice whose Lanczos iteration does not converge, or whose v_j is zero,
    gets the rigorous Courant number cfl_safety·2√2/(k⊥,max·Δ) instead, with
    a warning for the former.

    Cost: one Lanczos run over all M + 1 slices, about 20-40 steps of 6 FFTs
    per slice. Call it every few hundred steps, as P1 recomputes its dt.
    """
    if not 0 < cfl_safety <= 1:
        raise ValueError(f"cfl_safety must be in (0, 1], got {cfl_safety}")
    _, lam, _, _ = compute_streaming_eigensystem(state.M, state.Lambda, closure)
    if jnp.iscomplexobj(lam):
        raise ValueError("exact_substep_cfl needs real eigenvalues of the streaming matrix T")
    grid = state.grid
    c = math.sqrt(state.beta_i) * np.asarray(lam, np.float64)
    spacing = min(grid.Lx / grid.Nx, grid.Ly / grid.Ny, grid.Lz / grid.Nz)
    kx_max = float(grid.kx[(grid.Nx - 1) // 3])
    ky_max = float(grid.ky[(grid.Ny - 1) // 3])
    courant = cfl_safety * RK4_IMAGINARY_AXIS_LIMIT / (math.hypot(kx_max, ky_max) * spacing)
    grads = _perp_gradients(state.z_plus, state.z_minus, grid.kx, grid.ky,
                            grid.Nz, grid.Ny, grid.Nx)
    v = np.asarray(_slice_courant_speeds(grads, jnp.asarray(c, jnp.real(state.z_plus).dtype)))
    radius = slice_advection_radius(
        state.z_plus, state.z_minus, jnp.asarray(c), grid.kx, grid.ky, grid.dealias_mask,
        grid.Nz, grid.Ny, grid.Nx, tol=tol,
    )
    rho = np.asarray(radius.rho, np.float64)
    converged = np.asarray(radius.converged)
    if not converged.all():
        warnings.warn(
            f"slice_advection_radius did not converge for {int((~converged).sum())} slices; "
            "they keep the Courant bound.", RuntimeWarning, stacklevel=2,
        )
    ok = converged & (rho > 0) & (v > 0)
    return np.where(
        ok, cfl_safety * RK4_IMAGINARY_AXIS_LIMIT * v / (np.where(ok, rho, 1.0) * spacing), courant
    )


def _stage_gradients(grads: Tuple[Gradients, Gradients, Gradients], theta: Array) -> Gradients:
    """Gradients of Φ and Ψ at t + θH from the quadratic through θ = 0, 1/2, 1.

    The Lagrange weights are exactly (1, 0, 0), (0, 1, 0) and (0, 0, 1) at
    the nodes, so a single substep sees lawson_rk4's stage values.
    """
    w0 = 2.0 * (theta - 0.5) * (theta - 1.0)
    w1 = 4.0 * theta * (1.0 - theta)
    w2 = 2.0 * theta * (theta - 0.5)
    return tuple(w0 * a + w1 * b + w2 * c for a, b, c in zip(*grads))


@partial(
    jax.jit,
    static_argnames=[
        "Nz", "Ny", "Nx", "M", "hyper_r", "hyper_n", "hyper_rz", "closure", "hermite_chunk"
    ],
)
def _gandalf_step_lawson_rk4_multirate_jit(
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
    eta_z: float = 0.0,
    hyper_rz: int = 1,
    streaming_eigenvalues: Array = None,
    streaming_P_T: Array = None,
    streaming_P_inv_T: Array = None,
    closure: str = "zero",
    substep_courant: float = float("inf"),
    hermite_chunk: int = 1,
) -> KRMHDFields:
    """
    JIT-compiled multirate step (scheme="lawson_rk4_multirate"); see the module docstring.

    Takes the arguments of _gandalf_step_lawson_rk4_jit, plus:
        substep_courant: C·min(Δx, Δy, Δz), a scalar or one value per slice
            in the order of streaming_eigenvalues. Slice j takes
            n_j = ceil(dt·v_j / substep_courant_j) substeps (_substep_counts).
            The default, inf, gives one substep per slice: the lawson_rk4 step.
        hermite_chunk: Slices advanced together (static, default 1). The
            slices are sorted by speed, and a chunk takes the substeps of its
            fastest slice, so a chunk larger than one buys batching with
            extra substeps.

    The step runs in place like lawson_rk4's in-place path: one g-sized buffer
    holds g, then the slices, then g^(n+1).

    The eigensystem must meet lawson_rk4's conditions: real and well
    conditioned. At Λ = 1 with odd M, T is defective and
    compute_streaming_eigensystem raises a ValueError (Issue #192).
    """
    # z± exactly as in lawson_rk4: at t + dt/2 (Euler predictor) and at t + dt, undamped
    z_plus_half, z_minus_half, z_plus_new, z_minus_new = _elsasser_midpoint(
        fields.z_plus, fields.z_minus, dt, kx, ky, kz, dealias_mask, Nz, Ny, Nx
    )

    if M == 0:
        g_new = fields.g
    else:
        P = streaming_P_T.T
        P_inv = streaming_P_inv_T.T
        lam = streaming_eigenvalues
        if any(jnp.iscomplexobj(a) for a in (lam, P, P_inv)):
            raise ValueError(
                "scheme='lawson_rk4_multirate' needs a real eigensystem of the "
                "streaming matrix T, but it is complex (T has complex eigenvalues "
                "for Lambda < 1). Use scheme='imex_rk222' or 'imex_ars343'."
            )
        sqrt_beta = jnp.sqrt(beta_i)
        kz_4d = kz[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
        real = jnp.real(fields.z_plus).dtype

        grads = tuple(
            _perp_gradients(zp, zm, kx, ky, Nz, Ny, Nx)
            for zp, zm in (
                (fields.z_plus, fields.z_minus),
                (z_plus_half, z_minus_half),
                (z_plus_new, z_minus_new),
            )
        )
        counts = _substep_counts(
            _slice_courant_speeds(grads[0], sqrt_beta * lam), dt, substep_courant
        )

        # Sort the slices by substeps, so that a chunk holds slices of one rate
        # class. Reordering the eigenpairs leaves T = P·diag(λ)·P⁻¹ unchanged.
        order = jnp.argsort(counts)
        lam, counts = lam[order], counts[order]
        P, P_inv = P[:, order], P_inv[order, :]

        if M >= 2:
            # Hyper-collisions once per macro step, in the back-projection as in lawson_rk4
            P = P * _collision_factors(M, nu, hyper_n, dt, P.dtype)[:, jnp.newaxis]

        def substeps(h: Array, lam_c: Array, n: Array) -> Array:
            """n Lawson-RK4 substeps of dt/n for slices h [C, Nz, Ny, Nx//2+1]."""
            n_real = n.astype(real)
            step = dt / n_real
            slice_speeds = sqrt_beta * lam_c
            lam_4d = lam_c[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
            E_half = jnp.exp(-1j * sqrt_beta * kz_4d * lam_4d * step / 2.0)

            def advect(stage_grads: Gradients, h_stage: Array) -> Array:
                return _slice_advection_rhs(
                    stage_grads, h_stage, slice_speeds, kx, ky, dealias_mask, Nz, Ny, Nx
                )

            def substep(i: Array, h: Array) -> Array:
                # The Lawson-RK4 stages of _lawson_rk4_step, with
                # Φ and Ψ at t + (i, i + 1/2, i + 1)·dt/n.
                i_real = i.astype(real)
                grads_start = _stage_gradients(grads, i_real / n_real)
                grads_mid = _stage_gradients(grads, (i_real + 0.5) / n_real)
                grads_end = _stage_gradients(grads, (i_real + 1.0) / n_real)
                k = advect(grads_start, h)
                acc = E_half * E_half * (h + (step / 6.0) * k)
                k = advect(grads_mid, E_half * (h + (step / 2.0) * k))
                acc = acc + (step / 3.0) * E_half * k
                k = advect(grads_mid, E_half * h + (step / 2.0) * k)
                acc = acc + (step / 3.0) * E_half * k
                k = advect(grads_end, E_half * (E_half * h + step * k))
                return acc + (step / 6.0) * k

            return jax.lax.fori_loop(0, n, substep, h)

        chunk = min(hermite_chunk, M + 1)

        def advance(g: Array, j0: Array | int, size: int) -> Array:
            h = jnp.moveaxis(jax.lax.dynamic_slice_in_dim(g, j0, size, axis=3), 3, 0)
            n = jax.lax.dynamic_index_in_dim(counts, j0 + size - 1, keepdims=False)
            h = substeps(h, jax.lax.dynamic_slice_in_dim(lam, j0, size), n)
            return jax.lax.dynamic_update_slice_in_dim(g, jnp.moveaxis(h, 0, 3), j0, axis=3)

        dtype = jnp.result_type(fields.g, P, fields.z_plus, fields.z_minus, kx, ky, kz)
        g = _apply_moment_matrix(fields.g.astype(dtype), P_inv)
        n_full, rem = divmod(M + 1, chunk)
        g = jax.lax.fori_loop(0, n_full, lambda i, g: advance(g, i * chunk, chunk), g)
        if rem:
            g = advance(g, n_full * chunk, rem)
        g_new = _apply_moment_matrix(g, P)

    # Resistive and k_z damping of z± and g, and the 2/3 mask on g, once per
    # macro step; the Hermite stages above sample the undamped z^(n+1).
    z_plus_new, z_minus_new, g_new = _post_step_dissipation(
        z_plus_new, z_minus_new, g_new, dt, kx, ky, kz, dealias_mask, eta,
        Nz, Ny, Nx, hyper_r, eta_z, hyper_rz,
    )
    return KRMHDFields(
        z_plus=z_plus_new, z_minus=z_minus_new, g=g_new, time=fields.time + dt
    )
