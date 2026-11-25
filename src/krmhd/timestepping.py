"""
Time integration for KRMHD simulations using a mixed integrating-factor scheme.

This module implements the original GANDALF time-stepping algorithm from the thesis
Chapter 2, Equations 2.13-2.19:
- Integrating factor for linear propagation term (analytically exact)
- RK2 (midpoint method) for Elsasser nonlinear terms
- RK4 (Lawson form) for Hermite nonlinear terms
- Exponential integration for dissipation

The integrating factor e^(±ikz*t) removes the stiff linear term. The Elsasser
sector uses the midpoint RK2 update in the interaction picture (Lawson form):
with w(s) = e^(∓ikz*s)·ξ±(s), dw/ds = e^(∓ikz*s)·NL, so each nonlinear
increment is propagated by exactly one phase factor covering the remaining
sub-interval. The passive Hermite sector uses a Lawson-form RK4 update for
stable externally driven advection. It runs in the eigenbasis of the streaming
matrix (the phase-velocity slices), where both the streaming factor and the
perpendicular advection are diagonal (Issue #184).

gandalf_step(scheme=...) selects the Hermite integrator. The default
"imex_rk222" (ARS(2,2,2)) treats streaming and hyper-collisions implicitly.
"imex_ars343" (ARS(3,4,3), Issue #166) does the same with a third-order
pair and advances z± with the same explicit stages. Its explicit part is
stable for advection, but once g streams the pair is not, and like
imex_rk222 it needs hyper-collisions and hyper-resistivity (Issue #200).
"lawson_rk4" is the integrating-factor path above.

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

import functools
import math
import warnings
from functools import partial
from typing import Callable, Literal, Tuple, NamedTuple, get_args

import jax
import jax.numpy as jnp
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from krmhd.physics import (
    KRMHDState,
    z_plus_rhs,
    z_minus_rhs,
    g_rhs,
)
from krmhd.hermite import (
    advance_in_place,
    compute_streaming_matrix,
    compute_streaming_eigensystem,
    apply_implicit_operator,
    imex_factor,
    imex_solve,
    validate_closure,
)
from krmhd.spectral import SpectralGrid3D, derivative_x, derivative_y, rfftn_inverse


# =============================================================================
# Module Constants
# =============================================================================

# Closure for the unretained g_(M+1) (see hermite.compute_streaming_matrix).
HermiteClosure = Literal["zero", "symmetric"]

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
    g: Array  # Shape: [Nz, Ny, Nx//2+1, M+1]
    time: float  # Python float at public API; JAX Array inside JIT context


def _fields_from_state(state: KRMHDState) -> KRMHDFields:
    """Extract JAX-compatible fields from KRMHDState for hot path."""
    return KRMHDFields(
        z_plus=state.z_plus,
        z_minus=state.z_minus,
        g=state.g,
        time=state.time,
    )


# shard_map is jax.shard_map from JAX 0.6 and jax.experimental.shard_map before.
# Both take mesh, in_specs and out_specs as keywords.
_shard_map = getattr(jax, "shard_map", None)
if _shard_map is None:
    from jax.experimental.shard_map import shard_map as _shard_map


def _mesh_specs(mesh: Mesh) -> KRMHDFields:
    """How the fields lie on a mesh (Issue #191): g split along k_z, z± and time whole."""
    whole = PartitionSpec()
    return KRMHDFields(
        z_plus=whole, z_minus=whole, g=PartitionSpec(mesh.axis_names[0]), time=whole
    )


def place_on_mesh(fields: KRMHDFields, mesh: Mesh) -> KRMHDFields:
    """fields with z± replicated on every device of mesh and g split along k_z."""
    specs = _mesh_specs(mesh)
    return fields._replace(
        z_plus=jax.device_put(fields.z_plus, NamedSharding(mesh, specs.z_plus)),
        z_minus=jax.device_put(fields.z_minus, NamedSharding(mesh, specs.z_minus)),
        g=jax.device_put(fields.g, NamedSharding(mesh, specs.g)),
    )


def validate_mesh(mesh: Mesh | None, scheme: str, M: int, Nz: int) -> None:
    """Raise ValueError unless a step can shard its Hermite slices over mesh."""
    if mesh is None:
        return
    if scheme != "lawson_rk4":
        raise ValueError(f"mesh needs scheme='lawson_rk4' (got {scheme!r})")
    if len(mesh.axis_names) != 1:
        raise ValueError(f"mesh must have one axis (got {mesh.axis_names})")
    if M < 1:
        raise ValueError(f"mesh shards the M + 1 Hermite slices, so it needs M >= 1 (got M={M})")
    if Nz % mesh.size:
        raise ValueError(f"mesh needs Nz divisible by its {mesh.size} devices (got Nz={Nz})")


def _state_from_fields(fields: KRMHDFields, state_template: KRMHDState) -> KRMHDState:
    """Reconstruct KRMHDState from JAX fields (validates at boundary).

    This function MUST remain outside any jit-compiled scope because
    float() materializes a JAX scalar into a Python float for Pydantic.
    """
    return KRMHDState(
        z_plus=fields.z_plus,
        z_minus=fields.z_minus,
        g=fields.g,
        M=state_template.M,
        beta_i=state_template.beta_i,
        v_th=state_template.v_th,
        nu=state_template.nu,
        Lambda=state_template.Lambda,
        time=float(fields.time),
        grid=state_template.grid,
    )


@partial(jax.jit, static_argnames=["Nz", "Ny", "Nx", "M", "closure"])
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
    closure: str = "zero",
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

    # Hermite moment evolution (Issue #49 - now fully implemented!)
    # **OPTIMIZATION**: For M=0 (pure fluid mode), skip all kinetic physics computation.
    # This is critical for benchmarks like Orszag-Tang that test fluid-only dynamics.
    # When M=0, fields.g has shape (Nz, Ny, Nx//2+1, 1) and should remain zero.
    if M == 0:
        # Pure fluid mode: no kinetic response, g = 0 throughout evolution
        dg_dt = jnp.zeros_like(fields.g)
    else:
        # Full kinetic mode: all moments of the Hermite hierarchy (thesis Eqs. 2.7-2.10)
        dg_dt = g_rhs(
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
            closure=closure,
        )

    return KRMHDFields(
        z_plus=dz_plus_dt,
        z_minus=dz_minus_dt,
        g=dg_dt,
        time=0.0,  # Not a derivative
    )


def krmhd_rhs(
    state: KRMHDState,
    eta: float,
    v_A: float,
    closure: HermiteClosure = "zero",
) -> KRMHDState:
    """
    Compute time derivatives for all KRMHD fields.

    This is a thin wrapper that converts KRMHDState to lightweight KRMHDFields,
    calls the JIT-compiled RHS function, and converts back.

    Currently implements:
    - Elsasser fields: z⁺, z⁻ (Alfvénic sector)
    - g: Hermite moments (Issues #22-24, #49 - fully implemented!)

    Args:
        state: Current KRMHD state with all fields
        eta: Resistivity coefficient for dissipation
        v_A: Alfvén velocity (for normalization)
        closure: Hermite closure for g_(M+1), "zero" (default) or "symmetric"

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
        closure=closure,
    )

    # Convert back to KRMHDState (Pydantic validation at boundary)
    return _state_from_fields(deriv_fields, state)


# =============================================================================
# GANDALF Integrating Factor + RK2 Timestepping (Thesis Eq. 2.13-2.19)
# =============================================================================


def _to_slices(g: Array, P_inv: Array) -> Array:
    """
    Project Hermite moments onto the slices, the eigenvectors of the streaming matrix T.

    At every Fourier point h_j = Σ_m P⁻¹[j, m] g_m, where T = P·diag(λ)·P⁻¹
    (hermite.compute_streaming_eigensystem). The input g has the state layout
    [Nz, Ny, Nx//2+1, M+1]; the output is moment-major, [M+1, Nz, Ny, Nx//2+1],
    so that each slice is a contiguous 3D field for the batched FFTs.

    P⁻¹ must be real (the Lawson kernel checks this). It is applied to the
    real and imaginary parts of g separately, which takes half the flops of a
    complex matmul, and each slice stays the transform of a real field.
    Precision.HIGHEST keeps float32 runs off reduced-precision (TF32) matmul
    paths on GPUs.
    """
    hi = jax.lax.Precision.HIGHEST
    return jax.lax.complex(
        jnp.einsum("jm,zyxm->jzyx", P_inv, jnp.real(g), precision=hi),
        jnp.einsum("jm,zyxm->jzyx", P_inv, jnp.imag(g), precision=hi),
    )


def _from_slices(h: Array, P: Array) -> Array:
    """
    Inverse of _to_slices: g_m = Σ_j P[m, j] h_j at every Fourier point.

    Takes moment-major slices [M+1, Nz, Ny, Nx//2+1] and returns Hermite
    moments in the state layout [Nz, Ny, Nx//2+1, M+1]. Passing P with its
    rows scaled by per-moment factors applies those factors (the
    hyper-collisions) in the same matmul. P must be real.
    """
    hi = jax.lax.Precision.HIGHEST
    return jax.lax.complex(
        jnp.einsum("mj,jzyx->zyxm", P, jnp.real(h), precision=hi),
        jnp.einsum("mj,jzyx->zyxm", P, jnp.imag(h), precision=hi),
    )


def _apply_moment_matrix(g: Array, A: Array) -> Array:
    """
    Replace g_m by Σ_n A[m, n]·g_n at every Fourier point, one k_z plane at a time.

    The in-place form of _to_slices / _from_slices for large g (Issue #186):
    g keeps the state layout [Nz, Ny, Nx//2+1, M+1] and each plane is
    overwritten in place, so the matmul needs one plane of workspace instead
    of a second g-sized array. A = P⁻¹ projects onto the slices and A = P
    projects back; A must be real.
    """
    hi = jax.lax.Precision.HIGHEST

    def plane(iz: Array, g: Array) -> Array:
        p = jax.lax.dynamic_index_in_dim(g, iz, axis=0, keepdims=False)
        p = jax.lax.complex(
            jnp.einsum("yxn,mn->yxm", jnp.real(p), A, precision=hi),
            jnp.einsum("yxn,mn->yxm", jnp.imag(p), A, precision=hi),
        )
        return jax.lax.dynamic_update_index_in_dim(g, p, iz, axis=0)

    return jax.lax.fori_loop(0, g.shape[0], plane, g)


def _perp_gradients(
    z_plus: Array, z_minus: Array, kx: Array, ky: Array, Nz: int, Ny: int, Nx: int
) -> Tuple[Array, Array, Array, Array]:
    """Real-space ∂xΦ, ∂yΦ, ∂xΨ, ∂yΨ, with Φ = (ξ⁺ + ξ⁻)/2 and Ψ = (ξ⁺ - ξ⁻)/2."""
    phi = (z_plus + z_minus) / 2.0
    psi = (z_plus - z_minus) / 2.0
    kx_3d = kx[jnp.newaxis, jnp.newaxis, :]
    ky_3d = ky[jnp.newaxis, :, jnp.newaxis]
    return tuple(
        jnp.fft.irfftn(1j * k_3d * f, s=(Nz, Ny, Nx), axes=(-3, -2, -1))
        for f in (phi, psi)
        for k_3d in (kx_3d, ky_3d)
    )


def _slice_advection_rhs(
    grads: Tuple[Array, Array, Array, Array],
    h: Array,
    slice_speeds: Array,
    kx: Array,
    ky: Array,
    dealias_mask: Array,
    Nz: int,
    Ny: int,
    Nx: int,
) -> Array:
    """
    Explicit Hermite right-hand side in the slice basis, for a group of slices.

    Physics: the Hermite hierarchy (physics.g_rhs) reads
        ∂g/∂t = -{Φ, g} - √β_i (∂z + {Ψ, ·})(T g),
    with T the streaming matrix (hermite.compute_streaming_matrix, including
    the (1 - 1/Λ) coupling and the closure). T acts on the moment index and
    the Poisson bracket on space, so they commute. In the eigenbasis of T,
    h = P⁻¹g, the explicit (perpendicular) part is therefore diagonal: slice j
    is a passive scalar advected by the stream function Φ + √β_i λ_j Ψ,
        N_j(h) = -{Φ + √β_i λ_j Ψ, h_j}.
    The streaming part -i√β_i k_z λ_j h_j is diagonal too and is integrated
    exactly by the Lawson integrating factor in _gandalf_step_lawson_rk4_jit.

    Cost: the gradients of Φ and Ψ (4 inverse FFTs, _perp_gradients) are
    shared by all slices; each slice then needs ∂x h_j and ∂y h_j (2 inverse
    FFTs) and one forward FFT of its bracket, batched over the group. The
    Hermite-basis RHS needs 6 FFTs per moment instead (two brackets per
    moment).

    Args:
        grads: Real-space (∂xΦ, ∂yΦ, ∂xΨ, ∂yΨ) from _perp_gradients
        h: Slices in Fourier space, moment-major [C, Nz, Ny, Nx//2+1]
        slice_speeds: √β_i λ_j, the coefficient of Ψ in slice j's stream
            function [C] (real)
        kx, ky: Wavenumber arrays
        dealias_mask: 2/3 dealiasing mask [Nz, Ny, Nx//2+1]
        Nz, Ny, Nx: Grid dimensions (static)

    Returns:
        ∂h/∂t from perpendicular advection, dealiased, with the k = 0 mode of
        every slice set to zero (as in physics.g_rhs).
    """
    dphi_dx, dphi_dy, dpsi_dx, dpsi_dy = grads
    kx_3d = kx[jnp.newaxis, jnp.newaxis, :]
    ky_3d = ky[jnp.newaxis, :, jnp.newaxis]

    def to_real(field_fourier: Array) -> Array:
        """Inverse rfft over the last three axes (any leading batch axis)."""
        return jnp.fft.irfftn(field_fourier, s=(Nz, Ny, Nx), axes=(-3, -2, -1))

    dh_dx = to_real(1j * kx_3d * h)
    dh_dy = to_real(1j * ky_3d * h)

    # {Φ + c_j Ψ, h_j} = ∂x(Φ + c_j Ψ)·∂y h_j - ∂y(Φ + c_j Ψ)·∂x h_j
    c = slice_speeds[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    bracket = (dphi_dx + c * dpsi_dx) * dh_dy - (dphi_dy + c * dpsi_dy) * dh_dx

    rhs = -jnp.fft.rfftn(bracket, axes=(-3, -2, -1)) * dealias_mask
    return rhs.at[:, 0, 0, 0].set(0.0)


@partial(
    jax.jit,
    static_argnames=[
        "Nz", "Ny", "Nx", "M", "hyper_r", "hyper_n", "hyper_rz", "closure", "hermite_chunk",
        "mesh",
    ],
)
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
    eta_z: float = 0.0,
    hyper_rz: int = 1,
    streaming_eigenvalues: Array = None,
    streaming_P_T: Array = None,
    streaming_P_inv_T: Array = None,
    closure: str = "zero",
    hermite_chunk: int | None = None,
    mesh: Mesh | None = None,
) -> KRMHDFields:
    """
    JIT-compiled mixed integrating-factor timestepper (scheme="lawson_rk4").

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

    Slice basis (Issue #184): the Hermite sector is advanced in the eigenbasis
    of the streaming matrix, T = P·diag(λ)·P⁻¹. There the streaming
    integrating factor exp(-i√β_i k_z λ_j t) is diagonal, and so is the
    explicit term: slice j is advected by Φ + √β_i λ_j Ψ (see
    _slice_advection_rhs). The step projects g onto the slices once, runs the
    four RK4 stages there, and projects back once, folding the hyper-collision
    factors into the back-projection. The Hermite-basis form of the same scheme
    rotated into and out of the eigenbasis at every stage (10 dense (M+1)²
    matmuls per step) and evaluated the right-hand side moment by moment (6
    FFTs per moment per stage, a graph unrolled M+1 times). The slice form
    makes 2 matmuls and 3 FFTs per slice per stage, batched over slices, and
    adds each RK4 stage into the result as soon as it is computed. It is the
    same scheme: the two forms agree to round-off.

    Memory (Issue #186): small g goes through all at once, and the step then
    holds about six g-sized temporaries. When hermite.advance_in_place(g)
    (the all-at-once step does not fit the device), or hermite_chunk is smaller
    than M+1, the step runs in place instead: the projections go one k_z
    plane at a time, and because the slices are independent, each group of
    hermite_chunk slices (default one) runs all four stages and is written
    back to the moment slots it was read from. One g-sized array then holds
    g, the slices and g^(n+1) in turn, and the other temporaries do not grow
    with M: about twenty real-space fields, the gradients of Φ and Ψ at the
    three stage times and the stage arrays of one group.

    Conditions on the eigensystem:
    - It must be real. Each slice is then the transform of a real field, which
      the real FFTs of _slice_advection_rhs assume. A complex eigensystem
      (T has complex eigenvalues for Λ < 1) raises a ValueError at trace time.
    - It must be well conditioned: the step loses about cond(P)·eps. cond(P)
      ≈ 1.4 at Λ = √5 for every M ≤ 768. For odd M, cond(P) grows like
      (Λ - 1)^(-1/2) as Λ → 1 (about 10 at Λ = 1.01 and 100 at Λ = 1.0001). At
      Λ = 1 with odd M, T is defective (λ = 0 is a double root with one
      eigenvector) and cond(P) is 4e14 or more, so compute_streaming_eigensystem
      raises a ValueError for cond(P) > 1e3 (Issue #192). Use an IMEX scheme
      there, or Λ ≠ 1.

    At M = 1 the step integrates the system defined by T with the zero
    closure, (Tg)_1 = T[1,0]·g_0, as _krmhd_rhs_jit and the IMEX schemes do
    (Issue #193).

    Args:
        fields: Current KRMHD fields
        dt: Timestep
        kx, ky, kz: Wavenumbers
        dealias_mask: 2/3 dealiasing mask
        eta: Resistivity (or hyper-resistivity coefficient)
        v_A: Alfvén velocity
        beta_i: Ion plasma beta
        nu: Collision frequency (or hyper-collision coefficient)
        Lambda: Kinetic closure parameter (enters through the eigensystem of T)
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
        eta_z: Parallel (kz) hyper-resistivity coefficient (default: 0.0, disabled).
            Post-step multiplicative damping exp(-η_z·(kz²/kz²_max)^rz·dt) applied
            to ξ± AND all Hermite moments, with kz_max at the 2/3 dealias boundary
            (idz_max = (Nz-1)//3). Matches original GANDALF dampz
            (damping_kernel.cu:4-33, applied every step in timestep.cu:94-108).
        hyper_rz: Parallel hyper-resistivity order (default: 1, static)
            - rz=1: Standard parallel dissipation ∝ kz²
            - rz=2: Moderate parallel hyper-dissipation ∝ kz⁴
            - rz=4, 8: Sharper parallel cutoffs (expert use)
        streaming_eigenvalues: Eigenvalues λ of the streaming matrix T (from
            compute_streaming_eigensystem). Must be provided — None default is
            only for Python syntax (keyword after default args).
        streaming_P_T: Transposed right eigenvector matrix P.T.
        streaming_P_inv_T: Transposed inverse eigenvector matrix P_inv.T.
        closure: Hermite closure (static). It enters only through the
            eigensystem of T passed in, which gandalf_step builds for it.
        hermite_chunk: Slices advanced together in place (static). None
            (default): one slice at a time when hermite.advance_in_place(g),
            all at once otherwise.
            With a mesh, both apply to each device's share of g and slices.
            Results do not depend on it beyond round-off.
        mesh: One-axis jax.sharding.Mesh to shard the slices over (static;
            Issue #191). None (default): one device. With a mesh, g is
            sharded along k_z and z± are replicated, in and out; see
            _lawson_rk4_step. Needs M >= 1 and Nz divisible by the mesh size
            (validate_mesh).

    Returns:
        Updated KRMHDFields after full timestep

    Note:
        Per step: two Elsasser RHS evaluations (real kz, η = 0) and four
        Hermite stages in the slice basis, each stage one batched advection
        per group of slices. For M = 0 (pure fluid) the Hermite sector is only
        damped and dealiased, as before.
    """
    step = partial(
        _lawson_rk4_step, M=M, Nz=Nz, Ny=Ny, Nx=Nx, hyper_r=hyper_r, hyper_n=hyper_n,
        hyper_rz=hyper_rz, hermite_chunk=hermite_chunk, mesh=mesh,
    )
    args = (
        fields, dt, kx, ky, kz, dealias_mask, eta, beta_i, nu, eta_z,
        streaming_eigenvalues, streaming_P_T, streaming_P_inv_T,
    )
    if mesh is None:
        return step(*args)
    specs = _mesh_specs(mesh)
    return _shard_map(
        step, mesh=mesh, in_specs=(specs,) + (PartitionSpec(),) * (len(args) - 1), out_specs=specs
    )(*args)


def _collision_factors(M: int, nu: float, hyper_n: int, dt: float, dtype) -> Array:
    """Hyper-collision factors exp(-ν·(m/M)^n·dt) for m ≥ 2 and 1 for m = 0, 1, [M+1].

    The Lawson steps fold them into the back-projection from the slices
    (see the collision comments in _lawson_rk4_step). M ≥ 2.
    """
    moment_indices = jnp.arange(M + 1, dtype=dtype)  # [0, 1, 2, ..., M]
    collision_damping_rate = nu * ((moment_indices / M) ** hyper_n)
    return jnp.where(
        moment_indices >= 2,
        jnp.exp(-collision_damping_rate * dt),  # m≥2: hyper-collision damping
        1.0,  # m=0,1: no collision (conserves particles and momentum)
    )


def _elsasser_midpoint(
    z_plus: Array,
    z_minus: Array,
    dt: float,
    kx: Array,
    ky: Array,
    kz: Array,
    dealias_mask: Array,
    Nz: int,
    Ny: int,
    Nx: int,
) -> Tuple[Array, Array, Array, Array]:
    """
    The Elsasser part of the Lawson steps: midpoint RK2 in the interaction picture.

    Returns z± at t + dt/2 (the Euler predictor) and at t + dt, both undamped.
    The Hermite stages of lawson_rk4 sample Φ and Ψ at t, at the predictor and
    at t + dt; lawson_rk4_multirate interpolates between the same three states.
    """
    kz_3d = kz[:, jnp.newaxis, jnp.newaxis]

    # Integrating factors (thesis Eq. 2.13-2.14)
    # For ∂ξ⁺/∂t - ikz·ξ⁺ = [NL]: multiply by e^(+ikz*t)
    # For ∂ξ⁻/∂t + ikz·ξ⁻ = [NL]: multiply by e^(-ikz*t)
    # Note: v_A is absent by convention — KRMHD uses thesis normalization with
    # τ_A = L_z/v_A = 1, so v_A enters only through the nonlinear Poisson
    # bracket (via {ψ, …}) and the CFL timestep, not the linear Alfven phase.
    phase_plus_half = jnp.exp(+1j * kz_3d * dt / 2.0)
    phase_minus_half = jnp.exp(-1j * kz_3d * dt / 2.0)
    phase_plus_full = jnp.exp(+1j * kz_3d * dt)
    phase_minus_full = jnp.exp(-1j * kz_3d * dt)

    def elsasser_nl(z_plus: Array, z_minus: Array) -> Tuple[Array, Array]:
        """Elsasser nonlinear terms: the RHS with η = 0 minus the linear ±ikz·ξ± terms.

        Same formula as the Elsasser part of _krmhd_rhs_jit, which the IMEX
        kernels use, so the z± trajectory is unchanged.
        """
        rhs_plus = z_plus_rhs(z_plus, z_minus, kx, ky, kz, dealias_mask, 0.0, Nz, Ny, Nx)
        rhs_minus = z_minus_rhs(z_plus, z_minus, kx, ky, kz, dealias_mask, 0.0, Nz, Ny, Nx)
        return rhs_plus - 1j * kz_3d * z_plus, rhs_minus + 1j * kz_3d * z_minus

    # =========================================================================
    # Step 1: Elsasser half-step (integrating-factor predictor, cf. thesis §2.4)
    # =========================================================================

    # Half-step update in the interaction picture (Lawson form): define
    # w(s) = e^(∓ikz·s)·ξ±(s), so that dw/ds = e^(∓ikz·s)·NL(ξ±(s)).
    # Forward-Euler predictor over [0, Δt/2] in w, mapped back to ξ±:
    #   ξ±,n+1/2 = e^(±ikz·Δt/2) · [ξ±,n + Δt/2 · NL^n]
    # The phase multiplies the whole bracket exactly ONCE. (An earlier version
    # applied the phase a second time to the nonlinear increment, citing
    # thesis Eq. 2.14; the original CUDA GANDALF (timestep.cu, alf_adv)
    # applies linstep to the NL term exactly once, matching this form.)
    nl_plus_0, nl_minus_0 = elsasser_nl(z_plus, z_minus)
    z_plus_half = phase_plus_half * (z_plus + (dt / 2.0) * nl_plus_0)
    z_minus_half = phase_minus_half * (z_minus + (dt / 2.0) * nl_minus_0)

    # =========================================================================
    # Step 2: Elsasser midpoint RHS and full step
    # =========================================================================

    # Midpoint rule in the interaction picture:
    #   w^(n+1) = w^n + Δt · e^(∓ikz·Δt/2) · NL^(n+1/2)
    #   =>  ξ±,n+1 = e^(±ikz·Δt) · ξ±,n + e^(±ikz·Δt/2) · Δt · NL^(n+1/2)
    # The half phase on the increment propagates the midpoint nonlinear
    # transfer over the remaining half interval [Δt/2, Δt]. Note: the original
    # CUDA GANDALF applied the full phase e^(±ikz·Δt) (once) to the corrector
    # increment; the consistent Lawson midpoint uses e^(±ikz·Δt/2). The
    # difference is an O(kz·Δt²) local term, so both are formally 2nd order.
    nl_plus_half, nl_minus_half = elsasser_nl(z_plus_half, z_minus_half)
    z_plus_new = phase_plus_full * z_plus + phase_plus_half * dt * nl_plus_half
    z_minus_new = phase_minus_full * z_minus + phase_minus_half * dt * nl_minus_half

    return z_plus_half, z_minus_half, z_plus_new, z_minus_new


def _lawson_rk4_step(
    fields: KRMHDFields,
    dt: float,
    kx: Array,
    ky: Array,
    kz: Array,
    dealias_mask: Array,
    eta: float,
    beta_i: float,
    nu: float,
    eta_z: float,
    streaming_eigenvalues: Array,
    streaming_P_T: Array,
    streaming_P_inv_T: Array,
    *,
    M: int,
    Nz: int,
    Ny: int,
    Nx: int,
    hyper_r: int,
    hyper_n: int,
    hyper_rz: int,
    hermite_chunk: int | None,
    mesh: Mesh | None,
) -> KRMHDFields:
    """
    The step of _gandalf_step_lawson_rk4_jit (see there), on one device or on
    one device of a mesh.

    With a mesh (Issue #191) it runs inside shard_map. fields.g is then this
    device's k_z planes, [Nz/D, Ny, Nx//2+1, M+1] for D devices, and every
    other input is whole. The device computes the Elsasser update and the
    gradients of Φ and Ψ itself, since every slice needs them whole. It
    projects its planes onto all the slices, then one all-to-all gives it all
    k_z planes of its own ⌈(M+1)/D⌉ slices (zero slices pad M + 1 to a
    multiple of D). It advances those slices, and a second all-to-all and the
    back-projection return its k_z planes of g^(n+1). These two all-to-alls are
    the step's only communication.
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

    # Steps 1-2: Elsasser Lawson midpoint RK2 (_elsasser_midpoint)
    z_plus_half, z_minus_half, z_plus_new, z_minus_new = _elsasser_midpoint(
        fields.z_plus, fields.z_minus, dt, kx, ky, kz, dealias_mask, Nz, Ny, Nx
    )

    # =========================================================================
    # Step 3: Dissipation factors (thesis Eq. 2.23-2.25)
    # =========================================================================

    # Elsasser dissipation uses NORMALIZED k_perp^(2r) (perpendicular only, thesis Eq. 2.23)
    # Original GANDALF normalization: exp(-η·(k⊥²/k⊥²_max)^r·dt)
    # This makes the overflow constraint resolution-independent: η·dt < 50 (not η·k_max^(2r)·dt!)
    # Standard (r=1): ξ± → ξ± * exp(-η (k⊥²/k⊥²_max) Δt)
    # Hyper (r>1): ξ± → ξ± * exp(-η (k⊥²/k⊥²_max)^r Δt)
    # Note: Overflow validation is performed in gandalf_step() wrapper before JIT compilation
    k_perp_2r_normalized = (k_perp_squared / k_perp_max_squared) ** hyper_r
    perp_dissipation_factor = jnp.exp(-eta * k_perp_2r_normalized * dt)

    # Factors that are the same for every Hermite moment commute with P, so
    # they are applied to the slices before the back-projection:
    # (1) resistive dissipation, g → g·exp(-η (k⊥²/k⊥²_max)^r δt) (thesis Eq. 2.24);
    # (2) parallel (kz) hyper-dissipation, below;
    # (3) the projection onto the resolved 2/3 band. The Hermite RHS is
    #     already dealiased, so this is a defensive guarantee against
    #     externally injected or checkpointed out-of-band content.
    uniform_damping = perp_dissipation_factor

    # Parallel (kz) hyper-dissipation, the parallel analogue of the normalized
    # perpendicular hyper-resistivity above. Matches original GANDALF dampz
    # (damping_kernel.cu:4-33), applied every step to the Elsasser fields and
    # every Hermite moment (timestep.cu:94-108):
    #   ξ±, g_m → exp(-η_z·(kz²/kz²_max)^rz·dt) with kz_max at the 2/3 dealias
    #   boundary, idz_max = (Nz-1)//3 (same pattern as idx_max/idy_max above).
    # Multiplicative factors commute, so ordering within this section is exact.
    idz_max = (Nz - 1) // 3  # Python int: Nz is static, resolved at trace time
    if idz_max >= 1:
        # Static branch: skipped entirely for tiny Nz (e.g. Nz=2 used by 2D-style
        # tests) where kz[idz_max]=kz[0]=0 would give a divide-by-zero inf, and
        # 0·inf = NaN inside exp even for eta_z=0. The original GANDALF likewise
        # guarded dampz with `if (Nz > 1)`.
        # fftfreq ordering: idz_max < Nz/2 lands in the positive-kz range.
        kz_max_squared = kz[idz_max] ** 2
        kz_2rz_normalized = (kz_3d**2 / kz_max_squared) ** hyper_rz
        z_dissipation_factor = jnp.exp(-eta_z * kz_2rz_normalized * dt)
        uniform_damping = uniform_damping * z_dissipation_factor

    uniform_damping = uniform_damping * dealias_mask  # Shape: [Nz, Ny, Nx//2+1]

    # =========================================================================
    # Step 4: Hermite Lawson-RK4 in the slice basis
    # =========================================================================

    if M == 0:
        # Static branch: M is in static_argnames. For M = 0 (pure fluid RMHD)
        # the Hermite RHS is identically zero and g is only damped.
        g_new = fields.g * uniform_damping[:, :, :, jnp.newaxis]
    else:
        P = streaming_P_T.T
        P_inv = streaming_P_inv_T.T
        if any(
            jnp.iscomplexobj(a) for a in (streaming_eigenvalues, P, P_inv)
        ):
            # Dtypes are static, so this check runs at trace time.
            raise ValueError(
                "scheme='lawson_rk4' needs a real eigensystem of the streaming "
                "matrix T, but it is complex (T has complex eigenvalues for "
                "Lambda < 1). Use scheme='imex_rk222' or 'imex_ars343'."
            )
        lam = streaming_eigenvalues  # real, checked above
        sqrt_beta = jnp.sqrt(beta_i)
        kz_4d = kz[jnp.newaxis, :, jnp.newaxis, jnp.newaxis]

        # Φ and Ψ gradients at the three Elsasser stage times, shared by all slices.
        grads_n = _perp_gradients(fields.z_plus, fields.z_minus, kx, ky, Nz, Ny, Nx)
        grads_half = _perp_gradients(z_plus_half, z_minus_half, kx, ky, Nz, Ny, Nx)
        grads_new = _perp_gradients(z_plus_new, z_minus_new, kx, ky, Nz, Ny, Nx)

        def rk4(h: Array, lam_c: Array) -> Array:
            """One Lawson-RK4 step of the slices h [C, Nz, Ny, Nx//2+1] with
            eigenvalues lam_c [C]; returns h^(n+1) with the uniform damping."""
            slice_speeds = sqrt_beta * lam_c
            # Streaming integrating factor over half a step, diagonal in the
            # slices: exp(-i·√β_i·kz·λ_j·dt/2), shape [C, Nz, 1, 1].
            lam_4d = lam_c[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
            E_half = jnp.exp(-1j * sqrt_beta * kz_4d * lam_4d * dt / 2.0)

            def advect(grads: Tuple[Array, ...], h_stage: Array) -> Array:
                return _slice_advection_rhs(
                    grads, h_stage, slice_speeds, kx, ky, dealias_mask, Nz, Ny, Nx
                )

            # Lawson-form RK4 in the interaction picture, with E = E_half:
            #   k1 = N(h, z^n)
            #   k2 = N(E·(h + dt/2·k1),     z^(n+1/2))
            #   k3 = N(E·h + dt/2·k2,       z^(n+1/2))
            #   k4 = N(E·(E·h + dt·k3),     z^(n+1))
            #   h^(n+1) = E²·h + dt/6·(E²·k1 + 2E·k2 + 2E·k3 + k4)
            # Stages 2 and 3 share the Elsasser midpoint state because the
            # Hermite hierarchy is passive: g samples Φ and Ψ but does not feed
            # back on the Alfvénic sector. Each stage is added into `acc` as
            # soon as it is computed.
            k = advect(grads_n, h)
            acc = E_half * E_half * (h + (dt / 6.0) * k)
            k = advect(grads_half, E_half * (h + (dt / 2.0) * k))
            acc = acc + (dt / 3.0) * E_half * k
            k = advect(grads_half, E_half * h + (dt / 2.0) * k)
            acc = acc + (dt / 3.0) * E_half * k
            k = advect(grads_new, E_half * (E_half * h + dt * k))
            acc = acc + (dt / 6.0) * k
            return acc * uniform_damping[jnp.newaxis, :, :, :]

        # Collisional damping (thesis Eq. 2.24-2.25), diagonal in m, applied in
        # the back-projection g_m = c_m Σ_j P[m, j] h_j.
        # Physics: Lenard-Bernstein collision operator (thesis Eq. 2.5)
        # Original GANDALF normalization: exp(-ν·(m/M)^n·dt) (timestep.cu:111, alpha_m parameter)
        # This makes the overflow constraint resolution-independent: ν·dt < 50 (not ν·M^n·dt!)
        # IMPORTANT: Unlike spatial hyper-dissipation which uses k^(2r) due to ∇² ~ k²,
        #            moment collisions use just m^n to match original GANDALF alpha_m parameter
        #   Standard (n=1): g_m → g_m * exp(-ν·(m/M)·δt)
        #   Hyper (n>1):    g_m → g_m * exp(-ν·(m/M)^n·δt)
        # Conservation: m=0 (particle number) and m=1 (momentum) are exempt from collisions
        # Note: gandalf_step() wrapper enforces M>=2 only when nu>0.
        if M >= 2:
            # Static branch: collision damping requires M>=2 to avoid division by zero in (m/M)^n
            P = P * _collision_factors(M, nu, hyper_n, dt, P.dtype)[:, jnp.newaxis]

        # The slices this device advances: all M + 1 on one device. On a mesh
        # of D devices (Issue #191), zero slices pad M + 1 to D·n_own and stay
        # zero, and device d advances slices d·n_own to (d + 1)·n_own - 1.
        g = fields.g
        n_own, lam_own, pad = M + 1, lam, 0
        if mesh is not None:
            n_own = -(-(M + 1) // mesh.size)
            pad = n_own * mesh.size - (M + 1)
            if pad:
                P_inv = jnp.pad(P_inv, ((0, pad), (0, pad)))
                P = jnp.pad(P, ((0, pad), (0, pad)))
                g = jnp.pad(g, ((0, 0), (0, 0), (0, 0), (0, pad)))
            lam_own = jax.lax.dynamic_slice_in_dim(
                jnp.pad(lam, (0, pad)), jax.lax.axis_index(mesh.axis_names[0]) * n_own, n_own
            )

        def exchange(h: Array, split_axis: int, concat_axis: int) -> Array:
            """On a mesh, the all-to-all that splits h's slice axis (split_axis)
            over the devices and gathers its k_z axis (concat_axis), or the
            reverse; the identity on one device."""
            if mesh is None:
                return h
            return jax.lax.all_to_all(h, mesh.axis_names[0], split_axis, concat_axis, tiled=True)

        if hermite_chunk is None:
            in_place = advance_in_place(fields.g)
            chunk = 1
        else:
            chunk = min(hermite_chunk, n_own)
            in_place = chunk < n_own
        if not in_place:
            h = exchange(_to_slices(g, P_inv), 0, 1)
            g_new = _from_slices(exchange(rk4(h, lam_own), 1, 0), P)
        else:
            # One g-sized buffer holds g, then the slices, then g^(n+1)
            # (Issue #186); on a mesh each all-to-all needs a second one. The
            # slices are independent, so each group runs all four stages and
            # goes back to the moment slots it came from.

            def advance(g: Array, j0: Array | int, size: int) -> Array:
                h = jnp.moveaxis(jax.lax.dynamic_slice_in_dim(g, j0, size, axis=3), 3, 0)
                acc = rk4(h, jax.lax.dynamic_slice_in_dim(lam_own, j0, size))
                return jax.lax.dynamic_update_slice_in_dim(
                    g, jnp.moveaxis(acc, 0, 3), j0, axis=3
                )

            # The dtype the all-at-once path would give: g promoted with the
            # eigensystem, the Elsasser fields and the wavenumbers.
            dtype = jnp.result_type(fields.g, P, fields.z_plus, fields.z_minus, kx, ky, kz)
            g = exchange(_apply_moment_matrix(g.astype(dtype), P_inv), 3, 0)
            n_full, rem = divmod(n_own, chunk)
            g = jax.lax.fori_loop(0, n_full, lambda i, g: advance(g, i * chunk, chunk), g)
            if rem:
                g = advance(g, n_full * chunk, rem)
            g_new = _apply_moment_matrix(exchange(g, 0, 3), P)
        if pad:
            g_new = g_new[..., : M + 1]

    # =========================================================================
    # Step 5: Damp the Elsasser fields (after the Hermite stages, which sample
    # the undamped z^(n+1))
    # =========================================================================

    z_plus_new = z_plus_new * perp_dissipation_factor
    z_minus_new = z_minus_new * perp_dissipation_factor
    if idz_max >= 1:
        z_plus_new = z_plus_new * z_dissipation_factor
        z_minus_new = z_minus_new * z_dissipation_factor

    return KRMHDFields(
        z_plus=z_plus_new,
        z_minus=z_minus_new,
        g=g_new,
        time=fields.time + dt,
    )


# =============================================================================
# IMEX-RK222 Hermite Integrator (Issue #137)
# =============================================================================

# Public allow-list of integrator schemes accepted by gandalf_step().
# `Scheme` is the single source of truth; `SUPPORTED_SCHEMES` is the
# runtime-introspectable frozenset derived from it so adding a new
# integrator is a one-line edit.
Scheme = Literal["imex_rk222", "imex_ars343", "lawson_rk4", "lawson_rk4_multirate"]
SUPPORTED_SCHEMES: frozenset[str] = frozenset(get_args(Scheme))

# ARS(2,2,2) implicit-stage coefficient and stage-2 explicit scaling (Python floats).
# Note delta is NEGATIVE (~ -0.7071); the dt*delta*N_1 contribution in
# stage B's RHS therefore *subtracts* the initial-state nonlinear evaluation.
# This is correct per Ascher-Ruuth-Spiteri 1997 Table IV.
_IMEX_GAMMA: float = (2.0 - math.sqrt(2.0)) / 2.0        # gamma = (2 - sqrt(2))/2
_IMEX_DELTA: float = 1.0 - 1.0 / (2.0 * _IMEX_GAMMA)     # delta = 1 - 1/(2*gamma) ~ -0.7071

# Sentinel passed as nu to _krmhd_rhs_jit on the IMEX path. Hyper-collisional
# damping is folded into the implicit operator L (hermite.imex_factor), so
# the explicit RHS must NOT double-count it.
# Using this named constant makes the "deliberately zero" intent readable at
# every call site (otherwise a bare literal `0.0` looks like a mistake).
_NU_IN_IMPLICIT_OPERATOR: float = 0.0


@partial(
    jax.jit,
    static_argnames=["Nz", "Ny", "Nx", "M", "hyper_r", "hyper_n", "hyper_rz", "closure"],
)
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
    hyper_n: int,
    eta_z: float = 0.0,
    hyper_rz: int = 1,
    closure: str = "zero",
) -> KRMHDFields:
    """
    JIT-compiled IMEX-RK222 (Ascher-Ruuth-Spiteri 1997, ARS(2,2,2)) stepper.

    Linear Hermite streaming and hyper-collisional damping are treated
    implicitly via a tridiagonal solve along m per Fourier point
    (hermite.imex_factor, hermite.imex_solve); the Poisson-bracket nonlinear
    advection is treated explicitly. Elsasser z+/z- use the current
    integrating-factor RK2 midpoint. The Elsasser formula is identical to
    the Lawson path (z_plus_rhs / z_minus_rhs do not depend on g), but the
    two schemes produce a different JIT graph, so the realized z+/- trajectory
    agrees only to ~float32 accumulation error (~5e-4 over 200 steps, see
    test_imex222_nontrivial_vA_matches_lawson), not to machine precision.

    Butcher tableau (γ=(2-√2)/2 ≈ 0.2929, δ=1-1/(2γ) ≈ -0.7071; see Ascher,
    Ruuth, Spiteri 1997, Table IV). Below "g" in the Hermite field and "γ"
    the tableau coefficient are distinct — tableau rows use γ, not g:

        Implicit A:          Explicit Ã:
          c  |                c  |
          0  | 0    0    0     0  | 0     0     0
          γ  | 0    γ    0     γ  | γ     0     0
          1  | 0   1-γ   γ     1  | δ    1-δ    0
          --------------------    --------------------
             | 0   1-γ   γ         | δ    1-δ    0

    The tableau has three stages, but stage 1 is trivial (c=0, A[0,*]=0)
    so g^(1) = g^n, and N(g^(1)) = N(g^n). This kernel implements the two
    non-trivial stages directly:

        Stage A (tableau stage 2, c=γ):
            (I - dt·γ·L) g^(1) = g^n + dt·γ·N(g^n, z±^n)
        Stage B (tableau stage 3, c=1):
            (I - dt·γ·L) g^(n+1) = g^n
                + dt·(1-γ)·L·g^(1)
                + dt·δ·N(g^n, z±^n)
                + dt·(1-δ)·N(g^(1), z±^(γ))

    where z±^(γ) = e^(±ikz·γ·dt)·[z±^n + γ·dt·NL±^n] is the Lawson
    predictor to t+γ·dt.

    The factors of (I - dt·γ·L) are built inside this kernel from
    (kz, dt, beta_i, nu, Lambda, hyper_n, closure), so dt may be traced.
    Hyper-collisional damping is folded into L; the exponential collision
    factor is therefore NOT applied here.

    Parallel (kz) hyper-dissipation (eta_z, hyper_rz): applied post-step as
    exp(-eta_z*(kz^2/kz_max^2)^hyper_rz*dt) to z+/-, and to all Hermite
    moments, with kz_max at the 2/3 dealias boundary (idz_max = (Nz-1)//3).
    Matches original GANDALF dampz (damping_kernel.cu:4-33, applied every
    step in timestep.cu:94-108) and is applied at the same point as in the
    Lawson kernel so the two schemes stay equivalent. Default eta_z=0.0 is
    an exact no-op.

    Known limitation — resistivity splitting:
        Resistive damping on g is applied as an exponential post-step factor
        exp(-eta*(k_perp^2/k_perp_max^2)^r*dt), identical to the Lawson path.
        This is a 1st-order Strang-style operator split with the IMEX solve —
        the combined scheme is formally 2nd-order for the streaming +
        collision block and 1st-order for the streaming + resistivity
        interaction. For eta*dt*|k_perp|^(2r) << 1 (the typical turbulence
        regime), the extra splitting error is negligible.

    Memory (Issue #186): when hermite.advance_in_place(g), g_rhs goes through
    the moments in groups of eight and imex_solve one k_z plane at a time, each
    in place. The stage-B right-hand side absorbs N_1 before N_2 is computed,
    and g_rhs adds N_2 into it group by group, so N_2 never exists whole. On an
    A100 the compiled step then holds 4.6 copies of g (arguments, output and
    temporaries) instead of 7.0.

    z+/- coupling time (Issue #158):
        Stage B evaluates N(g^(1), z+/-^(γ)) with z+/- sampled at t+γ·dt,
        the same time as g^(1). z+/-^(γ) comes from an interaction-picture
        (Lawson) forward-Euler predictor over [0, γ·dt]. That is the
        explicit tableau's own stage-2 update applied to z+/-. Before #158,
        stage B reused the Elsasser midpoint at t+dt/2. That left a local
        error of O(dt²·(1/2-γ)·∂_t z·∂N/∂z), so the coupled g-z+/-
        trajectory converged at first order only (see
        test_imex222_coupled_g_zpm_second_order). The Elsasser full step
        still uses its own midpoint predictor at t+dt/2. The stage-B fields
        carry time=t+γ·dt, so a time-dependent term in `_krmhd_rhs_jit`
        would also be sampled at the correct stage time.
    """
    kz_3d = kz[:, jnp.newaxis, jnp.newaxis]
    kx_3d = kx[jnp.newaxis, jnp.newaxis, :]
    ky_3d = ky[jnp.newaxis, :, jnp.newaxis]
    k_perp_squared = kx_3d**2 + ky_3d**2

    idx_max = (Nx - 1) // 3
    idy_max = (Ny - 1) // 3
    k_perp_max_squared = kx[idx_max]**2 + ky[idy_max]**2

    # -------------------------------------------------------------------------
    # Elsasser integrating-factor RK2 midpoint (unchanged from Lawson path).
    # Phase factors match _gandalf_step_lawson_rk4_jit byte-for-byte. v_A does
    # not appear here: KRMHD uses thesis normalization with τ_A = L_z/v_A = 1,
    # so v_A enters only through the nonlinear Poisson bracket and CFL.
    # -------------------------------------------------------------------------
    phase_plus_half = jnp.exp(+1j * kz_3d * dt / 2.0)
    phase_minus_half = jnp.exp(-1j * kz_3d * dt / 2.0)
    phase_plus_full = jnp.exp(+1j * kz_3d * dt)
    phase_minus_full = jnp.exp(-1j * kz_3d * dt)
    # Stage-B coupling time t+γ·dt (Issue #158); see the docstring.
    phase_plus_gamma = jnp.exp(+1j * kz_3d * (_IMEX_GAMMA * dt))
    phase_minus_gamma = jnp.exp(-1j * kz_3d * (_IMEX_GAMMA * dt))

    # In g's RHS (physics.g_rhs), kz enters only through the
    # linear streaming term i*kz*(T g) that couples m to m+/-1. The
    # perpendicular Poisson brackets {Phi, g_m} and {Psi, (T g)_m} and
    # the (1-1/Lambda) kinetic correction are all kz-independent. Forcing
    # kz to zero therefore strips ONLY the streaming contribution and leaves
    # the nonlinear advection intact — which is exactly what N must be.
    # No double-counting of collisions: g_rhs does NOT take nu, and its
    # docstring states that collisions are applied in the
    # timestepper, not in the RHS. The nu argument below is a no-op for
    # g's RHS; collisions enter only via D inside the implicit operator L.
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
            _NU_IN_IMPLICIT_OPERATOR,
            Lambda,
            M,
            Nz,
            Ny,
            Nx,
            closure=closure,
        ).g

    # Elsasser half-step. Passing nu=0.0 explicitly (not nu) makes it visible
    # that _krmhd_rhs_jit's nu argument is intentionally a no-op on the IMEX
    # path: collisional damping lives in L, not in the RHS. If someone later
    # adds a nu-dependent term to the RHS, the zero here is a signpost that
    # the IMEX path needs separate handling.
    rhs_0 = _krmhd_rhs_jit(
        fields, kx, ky, kz, dealias_mask, 0.0, v_A, beta_i,
        _NU_IN_IMPLICIT_OPERATOR, Lambda, M, Nz, Ny, Nx, closure=closure,
    )
    nl_plus_0 = rhs_0.z_plus - (1j * kz_3d * fields.z_plus)
    nl_minus_0 = rhs_0.z_minus + (1j * kz_3d * fields.z_minus)
    # Interaction-picture (Lawson) forward-Euler predictor over [0, Δt/2]:
    # ξ±,n+1/2 = e^(±ikz·Δt/2)·[ξ±,n + Δt/2·NL^n] — single phase application;
    # see the derivation in _gandalf_step_lawson_rk4_jit (identical block).
    z_plus_half = phase_plus_half * (fields.z_plus + (dt / 2.0) * nl_plus_0)
    z_minus_half = phase_minus_half * (fields.z_minus + (dt / 2.0) * nl_minus_0)

    # IMEX tableau stage 2 (c=gamma): N evaluated at (g^n, z+/-^n); u^(1)=g^n is trivial
    N_1 = compute_nl_g(fields)

    # Solve (I - dt*gamma*L) g^(1) = g^n + dt*gamma * N_1. Both stages share
    # the diagonal gamma, so one factorization serves both solves.
    factor = imex_factor(kz, dt, _IMEX_GAMMA, beta_i, nu, Lambda, M, hyper_n, closure)
    rhs_1 = fields.g + (dt * _IMEX_GAMMA) * N_1
    # The g^n and N_1 terms of the stage-B right-hand side, formed now so
    # that N_1 is dead before N_2 exists (Issue #186).
    rhs_2 = fields.g + (dt * _IMEX_DELTA) * N_1
    g_1 = imex_solve(factor, rhs_1)

    # Elsasser midpoint RHS (needed for the z+/- full-step update only)
    fields_half = KRMHDFields(
        z_plus=z_plus_half,
        z_minus=z_minus_half,
        g=g_1,
        time=fields.time + dt / 2.0,
    )
    # nu=0.0 here for the same reason as in the half-step call above.
    rhs_half = _krmhd_rhs_jit(
        fields_half, kx, ky, kz, dealias_mask, 0.0, v_A, beta_i,
        _NU_IN_IMPLICIT_OPERATOR, Lambda, M, Nz, Ny, Nx, closure=closure,
    )
    nl_plus_half = rhs_half.z_plus - (1j * kz_3d * fields_half.z_plus)
    nl_minus_half = rhs_half.z_minus + (1j * kz_3d * fields_half.z_minus)

    # Elsasser full-step: midpoint rule in the interaction picture,
    # ξ±,n+1 = e^(±ikz·Δt)·ξ±,n + e^(±ikz·Δt/2)·Δt·NL^(n+1/2)
    # (matches Lawson path exactly given same nl_plus/minus_half; see the
    # derivation and CUDA-comparison note in _gandalf_step_lawson_rk4_jit)
    z_plus_new = phase_plus_full * fields.z_plus + phase_plus_half * dt * nl_plus_half
    z_minus_new = phase_minus_full * fields.z_minus + phase_minus_half * dt * nl_minus_half

    # IMEX tableau stage 3 (c=1): N at (g^(1), z+/-^(γ)), both at t+γ·dt
    # (Issue #158). z+/-^(γ) is the Lawson forward-Euler predictor over
    # [0, γ·dt], i.e. the explicit tableau's stage-2 row applied to z+/-.
    # Sampling z+/- at the midpoint t+dt/2 instead drops the coupled g-z+/-
    # trajectory to first order. Reuses nl_plus_0/nl_minus_0, so no extra
    # RHS evaluation. compute_nl_g strips streaming (kz=0), unlike rhs_half.g.
    z_plus_gamma = phase_plus_gamma * (fields.z_plus + (dt * _IMEX_GAMMA) * nl_plus_0)
    z_minus_gamma = phase_minus_gamma * (fields.z_minus + (dt * _IMEX_GAMMA) * nl_minus_0)
    fields_gamma = KRMHDFields(
        z_plus=z_plus_gamma,
        z_minus=z_minus_gamma,
        g=g_1,
        time=fields.time + _IMEX_GAMMA * dt,
    )

    # Explicit L*g^(1): a three-point stencil in m
    L_g1 = apply_implicit_operator(g_1, kz, beta_i, nu, Lambda, M, hyper_n, closure)

    if advance_in_place(fields.g) and M > 0:
        # N_2 is added into rhs_2 one group of moments at a time, so it never
        # exists whole (Issue #186). Same terms, summed in the same order. At
        # M = 0, _krmhd_rhs_jit gives N = 0, so that case stays below.
        rhs_2 = g_rhs(
            g_1, fields_gamma.z_plus, fields_gamma.z_minus, kx, ky, kz_zero, dealias_mask,
            beta_i, Lambda, Nz, Ny, Nx, closure=closure,
            acc=rhs_2 + (dt * (1.0 - _IMEX_GAMMA)) * L_g1, scale=dt * (1.0 - _IMEX_DELTA),
        )
    else:
        N_2 = compute_nl_g(fields_gamma)
        rhs_2 = rhs_2 + (dt * (1.0 - _IMEX_GAMMA)) * L_g1 + (dt * (1.0 - _IMEX_DELTA)) * N_2
    g_new = imex_solve(factor, rhs_2)

    # -------------------------------------------------------------------------
    # Post-step dissipation (resistivity) and dealiasing
    # -------------------------------------------------------------------------
    k_perp_2r_normalized = (k_perp_squared / k_perp_max_squared) ** hyper_r
    perp_dissipation_factor = jnp.exp(-eta * k_perp_2r_normalized * dt)
    z_plus_new = z_plus_new * perp_dissipation_factor
    z_minus_new = z_minus_new * perp_dissipation_factor

    # Resistive damping on g (from coupling to z+/-, kept exponential).
    # Hyper-collisional damping is already inside L, so no collision factor
    # here. Reuses perp_dissipation_factor computed above for the z+/- damping
    # — the resistive kernel is identical across all three fields. The
    # [:, :, :, jnp.newaxis] broadcast appends the m-axis so the (Nz, Ny, Nxh)
    # kernel multiplies uniformly across all Hermite moments of g.
    g_new = g_new * perp_dissipation_factor[:, :, :, jnp.newaxis]

    # Parallel (kz) hyper-dissipation, the parallel analogue of the normalized
    # perpendicular hyper-resistivity above. Matches original GANDALF dampz
    # (damping_kernel.cu:4-33), applied every step to the Elsasser fields and
    # every Hermite moment (timestep.cu:94-108):
    #   z±, g_m → exp(-η_z·(kz²/kz²_max)^rz·dt) with kz_max at the 2/3 dealias
    #   boundary, idz_max = (Nz-1)//3 (same pattern as idx_max/idy_max above).
    # Same placement as in _gandalf_step_lawson_rk4_jit (post-step dissipation,
    # before the defensive dealias) so the two schemes stay equivalent.
    idz_max = (Nz - 1) // 3  # Python int: Nz is static, resolved at trace time
    if idz_max >= 1:
        # Static branch: skipped entirely for tiny Nz (e.g. Nz=2 used by 2D-style
        # tests) where kz[idz_max]=kz[0]=0 would give a divide-by-zero inf, and
        # 0·inf = NaN inside exp even for eta_z=0. The original GANDALF likewise
        # guarded dampz with `if (Nz > 1)`.
        # fftfreq ordering: idz_max < Nz/2 lands in the positive-kz range.
        kz_max_squared = kz[idz_max] ** 2
        kz_2rz_normalized = (kz_3d**2 / kz_max_squared) ** hyper_rz
        z_dissipation_factor = jnp.exp(-eta_z * kz_2rz_normalized * dt)
        z_plus_new = z_plus_new * z_dissipation_factor
        z_minus_new = z_minus_new * z_dissipation_factor
        g_new = g_new * z_dissipation_factor[:, :, :, jnp.newaxis]

    # Defensive dealias of Hermite moments. z+/- are NOT dealiased here by
    # design: their dealiasing is already done inside _krmhd_rhs_jit via the
    # Poisson-bracket evaluation (spectral.poisson_bracket_3d applies the
    # mask after the nonlinear product). The Lawson kernel follows the same
    # convention, so this asymmetry is intentional, not an omission.
    g_new = g_new * dealias_mask[:, :, :, jnp.newaxis]

    return KRMHDFields(
        z_plus=z_plus_new,
        z_minus=z_minus_new,
        g=g_new,
        time=fields.time + dt,
    )


# =============================================================================
# IMEX-ARS(3,4,3) Integrator (Issue #166)
# =============================================================================


def _ars343_gamma() -> float:
    """Diagonal coefficient γ of ARS(3,4,3).

    γ is the root in (1/3, 1/2) of 6γ³ - 18γ² + 9γ - 1 = 0. It makes the
    3-stage SDIRK implicit part third order and L-stable (Alexander 1977).
    Newton's method refines the 10-digit table value to double precision.
    """
    g = 0.4358665215
    for _ in range(3):
        g -= (6.0 * g**3 - 18.0 * g**2 + 9.0 * g - 1.0) / (18.0 * g**2 - 36.0 * g + 9.0)
    return g


_ARS343_GAMMA: float = _ars343_gamma()                       # ≈ 0.4358665215
_ARS343_C3: float = (1.0 + _ARS343_GAMMA) / 2.0              # stage-3 time ≈ 0.7179
_ARS343_B1: float = -1.5 * _ARS343_GAMMA**2 + 4.0 * _ARS343_GAMMA - 0.25
_ARS343_B2: float = 1.5 * _ARS343_GAMMA**2 - 5.0 * _ARS343_GAMMA + 1.25


def _ars343_explicit_a4() -> float:
    """Explicit coefficient â42 = â43 of ARS(3,4,3).

    ARS (1997, §2.7) leave â42 = â43 free and tabulate 0.5529291479. That
    value makes the explicit stability polynomial equal to RK4's,
    1 + z + z²/2 + z³/6 + z⁴/24, which is stable on the imaginary axis for
    |z| ≤ 2√2. The z⁴ coefficient is γ²·â43·â32, and â32 is linear in â43
    (â32 = p·â43 + q below), so the condition γ²·a·(p·a + q) = 1/24 is a
    quadratic in a. Its root near the table value is returned.
    """
    g = _ARS343_GAMMA
    p = -3.75 + 15.0 * g - 5.25 * g**2
    q = 4.0 - 12.5 * g + 4.5 * g**2
    c0 = -1.0 / (24.0 * g**2)
    disc = math.sqrt(q * q - 4.0 * p * c0)
    roots = ((-q + disc) / (2.0 * p), (-q - disc) / (2.0 * p))
    return min(roots, key=lambda a: abs(a - 0.5529291479))


_ARS343_A42: float = _ars343_explicit_a4()                   # ≈ 0.5529291479
_ARS343_A43: float = _ARS343_A42
_ARS343_A31: float = (
    (1.0 - 4.5 * _ARS343_GAMMA + 1.5 * _ARS343_GAMMA**2) * _ARS343_A42
    + (2.75 - 10.5 * _ARS343_GAMMA + 3.75 * _ARS343_GAMMA**2) * _ARS343_A43
    - 3.5 + 13.0 * _ARS343_GAMMA - 4.5 * _ARS343_GAMMA**2
)
_ARS343_A32: float = (
    (-1.0 + 4.5 * _ARS343_GAMMA - 1.5 * _ARS343_GAMMA**2) * _ARS343_A42
    + (-2.75 + 10.5 * _ARS343_GAMMA - 3.75 * _ARS343_GAMMA**2) * _ARS343_A43
    + 4.0 - 12.5 * _ARS343_GAMMA + 4.5 * _ARS343_GAMMA**2
)
_ARS343_A41: float = 1.0 - _ARS343_A42 - _ARS343_A43


def _post_step_dissipation(
    z_plus: Array,
    z_minus: Array,
    g: Array,
    dt: float,
    kx: Array,
    ky: Array,
    kz: Array,
    dealias_mask: Array,
    eta: float,
    Nz: int,
    Ny: int,
    Nx: int,
    hyper_r: int,
    eta_z: float,
    hyper_rz: int,
) -> Tuple[Array, Array, Array]:
    """Post-step resistive and parallel damping, then the defensive g dealias.

    Same factors, in the same order, as the end of _gandalf_step_imex222_jit:
    exp(-η·(k⊥²/k⊥²_max)^r·dt) on z± and every g_m, then
    exp(-η_z·(kz²/kz²_max)^rz·dt) on all fields when (Nz-1)//3 ≥ 1, then
    the 2/3 mask on g. Hyper-collisions live in the implicit operator L.
    Must be called inside a jitted kernel (Nx, Ny, Nz, hyper_r and hyper_rz
    static).
    """
    kz_3d = kz[:, jnp.newaxis, jnp.newaxis]
    kx_3d = kx[jnp.newaxis, jnp.newaxis, :]
    ky_3d = ky[jnp.newaxis, :, jnp.newaxis]
    k_perp_squared = kx_3d**2 + ky_3d**2

    idx_max = (Nx - 1) // 3
    idy_max = (Ny - 1) // 3
    k_perp_max_squared = kx[idx_max]**2 + ky[idy_max]**2
    k_perp_2r_normalized = (k_perp_squared / k_perp_max_squared) ** hyper_r
    perp_dissipation_factor = jnp.exp(-eta * k_perp_2r_normalized * dt)
    z_plus = z_plus * perp_dissipation_factor
    z_minus = z_minus * perp_dissipation_factor
    g = g * perp_dissipation_factor[:, :, :, jnp.newaxis]

    # Static branch; see the matching block in _gandalf_step_imex222_jit for
    # why tiny Nz skips the parallel damping.
    idz_max = (Nz - 1) // 3
    if idz_max >= 1:
        kz_max_squared = kz[idz_max] ** 2
        kz_2rz_normalized = (kz_3d**2 / kz_max_squared) ** hyper_rz
        z_dissipation_factor = jnp.exp(-eta_z * kz_2rz_normalized * dt)
        z_plus = z_plus * z_dissipation_factor
        z_minus = z_minus * z_dissipation_factor
        g = g * z_dissipation_factor[:, :, :, jnp.newaxis]

    g = g * dealias_mask[:, :, :, jnp.newaxis]
    return z_plus, z_minus, g


@partial(
    jax.jit,
    static_argnames=["Nz", "Ny", "Nx", "M", "hyper_r", "hyper_n", "hyper_rz", "closure"],
)
def _gandalf_step_imex343_jit(
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
    hyper_n: int,
    eta_z: float = 0.0,
    hyper_rz: int = 1,
    closure: str = "zero",
) -> KRMHDFields:
    """
    JIT-compiled IMEX-ARS(3,4,3) stepper (Ascher, Ruuth, Spiteri 1997, §2.7).

    Why (Issue #166): the explicit part of ARS(2,2,2) has the stability
    function 1 + z + z²/2, so |R(iy)|² = 1 + y⁴/4 > 1 for every y ≠ 0. Pure
    advection is therefore unstable at any dt, and imex_rk222 relies on
    hyper-collisions and hyper-resistivity to stay bounded. The explicit
    part of ARS(3,4,3) has RK4's stability polynomial (see
    _ars343_explicit_a4), stable for |dt·ρ| ≤ 2√2, where ρ is the spectral
    radius of the advection operator.

    That bound holds for z± and for g on the k_z = 0 plane, not for g that
    streams (Issue #200). The implicit stages treat streaming and the
    explicit stages treat advection, and together they amplify some modes
    at every dt > 0. Like imex_rk222, g then needs hyper-collisions and
    hyper-resistivity, though at the same dt 3 to 17 times less of them
    in the worst case. docs/numerical_methods.md has the numbers.

    Tableau (γ ≈ 0.4358665215, c₃ = (1+γ)/2, b₁, b₂ from _ARS343_*):

        Implicit A:                    Explicit Â:
          c  |                           c  |
          0  | 0   0        0   0        0  | 0    0    0    0
          γ  | 0   γ        0   0        γ  | γ    0    0    0
          c₃ | 0  (1-γ)/2   γ   0        c₃ | â31  â32  0    0
          1  | 0   b₁       b₂  γ        1  | â41  â42  â43  0
          -------------------------      -----------------------
             | 0   b₁       b₂  γ           | 0    b₁   b₂   γ

    Hermite moments g: linear streaming and hyper-collisions (L, implicit)
    and the Poisson-bracket advection N (explicit). The three implicit
    stages share the diagonal γ, so one factorization of the tridiagonal
    (I - γ·dt·L) serves all of them. The implicit part is stiffly accurate, so
    gⁿ⁺¹ = G₄ + dt·Σⱼ (b̂ⱼ - â₄ⱼ)·Nⱼ and needs no extra solve. The terms
    dt·L·Gᵢ that later stages need come from the stage equation itself,
    dt·L·Gᵢ = (Gᵢ - rhsᵢ)/γ, which avoids a mat-vec.

    Elsasser z±: the same explicit tableau in Lawson (interaction-picture)
    form. Stage values are
        Zᵢ = e^(±ikz·cᵢ·dt)·zⁿ + dt·Σⱼ âᵢⱼ·e^(±ikz·(cᵢ-cⱼ)·dt)·NLⱼ
    and zⁿ⁺¹ = e^(±ikz·dt)·zⁿ + dt·Σⱼ b̂ⱼ·e^(±ikz·(1-cⱼ)·dt)·NLⱼ. z± advection
    is then stable on the same interval, and every g stage samples z± at
    its own stage time. Unlike imex_rk222 and lawson_rk4, the z± trajectory
    is not the GANDALF midpoint RK2 one.

    Each stage makes one _krmhd_rhs_jit call with kz = 0 and η = 0, which
    returns the nonlinear terms of z± and g together (the linear
    propagation and streaming terms are the only kz-dependent parts). The
    step costs four RHS calls and three tridiagonal solves; imex_rk222 makes
    four RHS calls and two solves.

    Order: third order for the coupled (z±, g) system with η = η_z = 0.
    Resistive and parallel damping are applied after the step with exact
    exponential factors (_post_step_dissipation), the same first-order
    splitting as imex_rk222; see its "resistivity splitting" note.
    """
    gam = _ARS343_GAMMA
    c3 = _ARS343_C3
    kz_3d = kz[:, jnp.newaxis, jnp.newaxis]
    kz_zero = jnp.zeros_like(kz)

    def nonlinear(stage_fields: KRMHDFields) -> Tuple[Array, Array, Array]:
        """Nonlinear RHS (NL⁺, NL⁻, N_g) at one stage; kz = 0 strips the linear terms."""
        rhs = _krmhd_rhs_jit(
            stage_fields, kx, ky, kz_zero, dealias_mask, 0.0, v_A, beta_i,
            _NU_IN_IMPLICIT_OPERATOR, Lambda, M, Nz, Ny, Nx, closure=closure,
        )
        return rhs.z_plus, rhs.z_minus, rhs.g

    def propagator(s: float) -> Tuple[Array, Array]:
        """Elsasser linear propagators e^(±ikz·s·dt) over the fraction s of the step."""
        theta = kz_3d * (s * dt)
        return jnp.exp(1j * theta), jnp.exp(-1j * theta)

    # The phase offsets cᵢ - cⱼ and 1 - cⱼ that occur: γ, c₃, (1-γ)/2
    # (= c₃ - γ = 1 - c₃), 1 - γ and 1.
    ep_gam, em_gam = propagator(gam)
    ep_c3, em_c3 = propagator(c3)
    ep_half, em_half = propagator((1.0 - gam) / 2.0)
    ep_1mg, em_1mg = propagator(1.0 - gam)
    ep_one, em_one = propagator(1.0)

    zp_n, zm_n, g_n = fields.z_plus, fields.z_minus, fields.g
    factor = imex_factor(kz, dt, gam, beta_i, nu, Lambda, M, hyper_n, closure)

    # Stage 1 (c = 0): Z₁ = zⁿ, G₁ = gⁿ.
    nlp_1, nlm_1, N_1 = nonlinear(fields)

    # Stage 2 (c = γ).
    zp_2 = ep_gam * (zp_n + (dt * gam) * nlp_1)
    zm_2 = em_gam * (zm_n + (dt * gam) * nlm_1)
    rhs_2 = g_n + (dt * gam) * N_1
    G_2 = imex_solve(factor, rhs_2)
    dtLG_2 = (G_2 - rhs_2) / gam
    nlp_2, nlm_2, N_2 = nonlinear(
        KRMHDFields(z_plus=zp_2, z_minus=zm_2, g=G_2, time=fields.time + gam * dt)
    )

    # Stage 3 (c = c₃).
    zp_3 = ep_c3 * (zp_n + (dt * _ARS343_A31) * nlp_1) + (dt * _ARS343_A32) * ep_half * nlp_2
    zm_3 = em_c3 * (zm_n + (dt * _ARS343_A31) * nlm_1) + (dt * _ARS343_A32) * em_half * nlm_2
    rhs_3 = (
        g_n
        + ((1.0 - gam) / 2.0) * dtLG_2
        + (dt * _ARS343_A31) * N_1
        + (dt * _ARS343_A32) * N_2
    )
    G_3 = imex_solve(factor, rhs_3)
    dtLG_3 = (G_3 - rhs_3) / gam
    nlp_3, nlm_3, N_3 = nonlinear(
        KRMHDFields(z_plus=zp_3, z_minus=zm_3, g=G_3, time=fields.time + c3 * dt)
    )

    # Stage 4 (c = 1).
    zp_4 = (
        ep_one * (zp_n + (dt * _ARS343_A41) * nlp_1)
        + (dt * _ARS343_A42) * ep_1mg * nlp_2
        + (dt * _ARS343_A43) * ep_half * nlp_3
    )
    zm_4 = (
        em_one * (zm_n + (dt * _ARS343_A41) * nlm_1)
        + (dt * _ARS343_A42) * em_1mg * nlm_2
        + (dt * _ARS343_A43) * em_half * nlm_3
    )
    rhs_4 = (
        g_n
        + _ARS343_B1 * dtLG_2
        + _ARS343_B2 * dtLG_3
        + (dt * _ARS343_A41) * N_1
        + (dt * _ARS343_A42) * N_2
        + (dt * _ARS343_A43) * N_3
    )
    G_4 = imex_solve(factor, rhs_4)
    nlp_4, nlm_4, N_4 = nonlinear(
        KRMHDFields(z_plus=zp_4, z_minus=zm_4, g=G_4, time=fields.time + dt)
    )

    # Final update with b̂ = (0, b₁, b₂, γ).
    z_plus_new = ep_one * zp_n + dt * (
        _ARS343_B1 * ep_1mg * nlp_2 + _ARS343_B2 * ep_half * nlp_3 + gam * nlp_4
    )
    z_minus_new = em_one * zm_n + dt * (
        _ARS343_B1 * em_1mg * nlm_2 + _ARS343_B2 * em_half * nlm_3 + gam * nlm_4
    )
    g_new = G_4 + dt * (
        -_ARS343_A41 * N_1
        + (_ARS343_B1 - _ARS343_A42) * N_2
        + (_ARS343_B2 - _ARS343_A43) * N_3
        + gam * N_4
    )

    z_plus_new, z_minus_new, g_new = _post_step_dissipation(
        z_plus_new, z_minus_new, g_new, dt, kx, ky, kz, dealias_mask,
        eta, Nz, Ny, Nx, hyper_r, eta_z, hyper_rz,
    )

    return KRMHDFields(
        z_plus=z_plus_new,
        z_minus=z_minus_new,
        g=g_new,
        time=fields.time + dt,
    )


def _validate_step_args(
    state: KRMHDState,
    dt: float,
    eta: float,
    nu_effective: float,
    hyper_r: int,
    hyper_n: int,
    scheme: str,
    eta_z: float,
    hyper_rz: int,
    closure: str,
    substep_cfl: float | Tuple[float, ...] | Array = 0.3,
) -> None:
    """Check gandalf_step's arguments; raises or warns as its docstring lists.

    Shared with the multi-step loop (krmhd.runloop), which checks once per run.
    """
    # Input validation for hyper parameters
    if hyper_r not in (1, 2, 4, 8):
        raise ValueError(f"hyper_r must be 1, 2, 4, or 8 (got {hyper_r})")

    if hyper_n not in (1, 2, 3, 4, 6):
        raise ValueError(f"hyper_n must be 1, 2, 3, 4, or 6 (got {hyper_n})")

    if hyper_rz not in (1, 2, 4, 8):
        raise ValueError(f"hyper_rz must be 1, 2, 4, or 8 (got {hyper_rz})")

    if scheme not in SUPPORTED_SCHEMES:
        raise ValueError(
            f"scheme must be one of {sorted(SUPPORTED_SCHEMES)} (got {scheme!r})"
        )

    validate_closure(closure, state.M)

    if scheme == "lawson_rk4_multirate":
        c = jnp.asarray(substep_cfl, dtype=float)
        if c.shape not in ((), (state.M + 1,)) or not bool(jnp.all(c > 0)):
            raise ValueError(
                f"substep_cfl must be positive, a scalar or one value per slice "
                f"(M + 1 = {state.M + 1}), got {substep_cfl}"
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

    if scheme in ("lawson_rk4", "lawson_rk4_multirate"):
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

    # Safety check for parallel (kz) hyper-resistivity overflow, mirroring eta.
    # NORMALIZED formulation exp(-η_z·(kz²/kz²_max)^rz·dt): maximum rate at
    # kz=kz_max is simply η_z·dt (matches original GANDALF damping_kernel.cu:4-33)
    max_kz_resistivity_rate = eta_z * dt

    if max_kz_resistivity_rate >= MAX_DAMPING_RATE_THRESHOLD:
        safe_eta_z = MAX_DAMPING_RATE_THRESHOLD / dt
        raise ValueError(
            f"Parallel hyper-resistivity overflow risk detected!\n"
            f"  Parameter: eta_z·dt = {eta_z}·{dt} = {max_kz_resistivity_rate:.2e}\n"
            f"  Threshold: Must be < {MAX_DAMPING_RATE_THRESHOLD} to avoid exp() underflow\n"
            f"  Solution: Reduce eta_z to < {safe_eta_z:.2e} or reduce dt\n"
            f"  Note: With normalized dissipation, constraint is eta_z·dt < {MAX_DAMPING_RATE_THRESHOLD} (independent of resolution or rz!)"
        )

    # Warning for moderate risk (20-50)
    if max_kz_resistivity_rate >= DAMPING_RATE_WARNING_THRESHOLD:
        warnings.warn(
            f"Parallel hyper-resistivity damping rate is high: eta_z·dt = {max_kz_resistivity_rate:.2e}. "
            f"Consider reducing eta_z or dt to improve numerical stability.",
            RuntimeWarning
        )

    # Normalized kz damping needs at least one nonzero kz mode inside the 2/3
    # dealias boundary: idz_max = (Nz-1)//3 >= 1. For tiny Nz (e.g. Nz=2,
    # the 2D-style grids) kz[idz_max] = 0 and the normalization is undefined.
    # The original GANDALF guarded dampz with `if (Nz > 1)`; the JIT kernels
    # skip the kz damping block entirely (static branch) in that case.
    if eta_z > 0 and (state.grid.Nz - 1) // 3 < 1:
        raise ValueError(
            f"eta_z > 0 requires a grid with (Nz-1)//3 >= 1 (got Nz={state.grid.Nz}). "
            f"Normalized parallel dissipation exp(-eta_z·(kz²/kz²_max)^rz·dt) needs "
            f"a nonzero kz_max at the 2/3 dealias boundary; use Nz >= 4 or set eta_z=0."
        )


def gandalf_step(
    state: KRMHDState,
    dt: float,
    eta: float,
    v_A: float,
    nu: float | None = None,
    hyper_r: int = 1,
    hyper_n: int = 1,
    scheme: Scheme = "imex_rk222",
    eta_z: float = 0.0,
    hyper_rz: int = 1,
    closure: HermiteClosure = "zero",
    hermite_chunk: int | None = None,
    mesh: Mesh | None = None,
    substep_cfl: float | Tuple[float, ...] | Array = 0.3,
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
              nonlinear advection is explicit. Elsasser z+/z- formula matches
              the Lawson path (z_plus_rhs/z_minus_rhs do not depend on g);
              realized trajectories agree to ~float32 accumulation error.
              Its explicit stage has |R(iy)|² = 1 + y⁴/4, so advection grows
              at every dt unless hyper-collisions and hyper-resistivity damp
              it faster (Issue #166).
            - "imex_ars343": ARS(3,4,3) IMEX scheme (Issue #166). Same
              implicit operator as imex_rk222, with an L-stable third-order
              implicit part. Its explicit part has RK4's stability
              polynomial, so z± and g on the k_z = 0 plane advect stably
              for |dt·ρ| ≤ 2√2 without dissipation. z± use the same
              explicit tableau in Lawson form, so the z± trajectory differs
              from the midpoint RK2 of the other two schemes. Third order
              for the coupled system with η = η_z = 0. Costs one more
              tridiagonal solve per step than imex_rk222.
              Once g streams (k_z ≠ 0), g grows at every dt unless
              hyper-collisions and hyper-resistivity damp it (Issue #200),
              as in imex_rk222, though at the same dt imex_ars343 needs 3
              to 17 times less of them in the worst case.
              docs/numerical_methods.md has the numbers and the settings it
              needs.
            - "lawson_rk4": legacy mixed integrating-factor + Lawson-RK4 path,
              retained for comparison and rollback. Known to be unstable at
              high M·k_z (see Issue #137).
            - "lawson_rk4_multirate": lawson_rk4 with a step per slice
              (Issue #191, krmhd.multirate). z± take the step dt; slice j
              takes ceil(dt·v_j / (substep_cfl·Δ)) Lawson-RK4 substeps, with
              Φ and Ψ interpolated in time. Take dt from
              compute_cfl_timestep(mode="multirate"). With one substep per
              slice it is lawson_rk4. z± then run at their own Courant limit
              with midpoint RK2, which needs hyper-resistivity, as in
              imex_rk222. docs/multirate_slices_study.md has its stability,
              order and cost.

            NOTE: scheme is a per-call argument and is NOT stored on
            KRMHDState. Callers looping over time must pass it explicitly
            each step. Mixing a checkpoint written by one scheme into a
            subsequent run under the other scheme is valid — state carries
            only field data — but the evolution cadence is per-call.

            REPRODUCIBILITY: the scheme default changed from "lawson_rk4"
            to "imex_rk222" in #138. Existing drivers that omitted `scheme`
            will silently pick up the new integrator. The Elsasser formula
            is mathematically identical under both schemes, but the two
            JIT graphs produce different float32 arithmetic orderings, so
            a mid-run switch (or checkpoint resumed under a different
            default) will drift from the original trajectory at roughly
            1e-6 per step (~5e-4 over 200 steps; see
            test_imex222_nontrivial_vA_matches_lawson). For bit-level
            reproducibility across reruns, pin `scheme=` explicitly.
            Issue #158 moved the IMEX stage-B z+/- sample from t+dt/2 to
            t+γ·dt. IMEX g trajectories therefore differ from pre-#158
            runs by O(dt); z+/- trajectories are unchanged.
        eta_z: Parallel (kz) hyper-resistivity coefficient (default: 0.0, disabled).
            Post-step multiplicative damping exp(-η_z·(kz²/kz²_max)^rz·dt)
            applied to z⁺, z⁻ AND all Hermite moments g, with kz_max at the
            2/3 dealias boundary (idz_max = (Nz-1)//3). This is the exact
            parallel analogue of the normalized perpendicular hyper-resistivity
            eta, and matches original GANDALF dampz (damping_kernel.cu:4-33),
            which was applied every step to the Elsasser fields and every
            Hermite moment (timestep.cu:94-108). It regulates the parallel
            (kz) spectrum, which otherwise has no dissipation channel
            (relevant to Issues #136 and #137).

            DELIBERATE DEVIATION from original GANDALF: the original used a
            separate coefficient nu_kz_g (with order alpha_kz_g) for the
            Hermite moments; here a single coefficient eta_z (and single
            order hyper_rz) is shared by z± and g, exactly as the existing
            perpendicular eta is shared across all three fields.

            Default 0.0 gives exactly zero behavior change (the damping
            factor is identically 1).
        hyper_rz: Parallel hyper-resistivity order (default: 1)
            - rz=1: Standard parallel dissipation ∝ kz²
            - rz=2: Moderate parallel hyper-dissipation ∝ kz⁴
            - rz=4: Strong parallel hyper-dissipation ∝ kz⁸ (expert use)
            - rz=8: Maximum parallel hyper-dissipation ∝ kz¹⁶ (expert use)
        closure: Hermite truncation closure for the unretained g_(M+1)
            (default: "zero").
            - "zero": g_(M+1) = 0 (original GANDALF behaviour)
            - "symmetric": g_(M+1) = g_(M-1); requires M >= 2
            Applied consistently to the explicit RHS (streaming and the
            {Ψ, ·} field-line term at m = M), the Lawson streaming
            eigensystem, and the IMEX implicit operator. Like `scheme`, this
            is a per-call argument and is NOT stored on the state.
        hermite_chunk: Number of Hermite slices that scheme="lawson_rk4"
            advances at once, in place (Issue #186). The step's temporaries
            then scale with it, not with M. The default (None) advances one
            slice at a time when hermite.advance_in_place(g), that is when
            the all-at-once step does not fit the device's memory, and all
            of them at once otherwise. Results do not depend on it beyond round-off.
            For "lawson_rk4_multirate" the default is one slice at a time.
        mesh: One-axis jax.sharding.Mesh over which scheme="lawson_rk4"
            shards its Hermite slices (Issue #191), e.g.
            Mesh(jax.devices(), ("slices",)). Default None: one device. The
            returned state has g split along k_z over the mesh and z±
            replicated; a state on one device is placed so first. Each step
            then makes two all-to-alls of g's size over the devices, and
            results match one device to round-off. hermite_chunk and the
            in-place threshold apply to each device's share. Needs M >= 1 and
            Nz divisible by the number of devices. See
            docs/multi_gpu_slices.md for the tradeoff.
        substep_cfl: Courant number C of the slice substeps of
            "lawson_rk4_multirate" (default 0.3, ignored by the other
            schemes). Slice j takes the smallest number of substeps h_j with
            h_j·max|∇⊥(Φ + √β_i λ_j Ψ)| ≤ C·min(Δx, Δy, Δz), the Courant rule
            of compute_cfl_timestep. C ≤ 0.95 keeps every substep within the
            RK4 stability limit for a frozen flow. One value per slice, in
            the order of compute_streaming_eigensystem's eigenvalues, sets
            C_j per slice: multirate.exact_substep_cfl calibrates them to
            the exact radius of each slice.

    Returns:
        New KRMHDState at time t + dt

    Raises:
        ValueError: If hyper_r not in [1, 2, 4, 8]
        ValueError: If hyper_n not in [1, 2, 3, 4, 6]
        ValueError: If closure is unknown, or "symmetric" with M < 2
        ValueError: If hyper_rz not in [1, 2, 4, 8]
        ValueError: If scheme not in {"imex_rk222", "imex_ars343", "lawson_rk4",
            "lawson_rk4_multirate"}
        ValueError: If mesh is given with another scheme, more than one axis,
            M = 0, or Nz not divisible by its size
        ValueError: If hyper-collision overflow risk detected (nu·dt >= 50, normalized;
            Lawson path only — the IMEX solve is unconditionally stable)
        ValueError: If hyper-resistivity overflow risk detected (eta·dt >= 50, normalized)
        ValueError: If parallel hyper-resistivity overflow risk detected
            (eta_z·dt >= 50, normalized)
        ValueError: If eta_z > 0 but the grid is too small in z for normalized
            kz damping ((Nz-1)//3 < 1, i.e. no nonzero kz mode inside the 2/3
            dealias boundary; the original GANDALF guarded dampz with `if (Nz > 1)`)

    Example:
        >>> # Standard dissipation (backward compatible)
        >>> state_new = gandalf_step(state, dt=0.01, eta=0.01, v_A=1.0)
        >>>
        >>> # Hyper-dissipation for turbulence studies
        >>> state_new = gandalf_step(state, dt=0.01, eta=0.001, v_A=1.0,
        ...                          hyper_r=8, hyper_n=4)
        >>>
        >>> # In a loop — pass scheme explicitly if you want to pin an
        >>> # integrator regardless of future default changes:
        >>> for i in range(n_steps):
        ...     state = gandalf_step(state, dt, eta, v_A, hyper_r=8, hyper_n=4,
        ...                          scheme="imex_rk222")

    Performance:
        - The IMEX implicit operator (I - γ·dt·L) is tridiagonal along m
          (Issue #187). Each step factors it on the device in O(M·Nz) and
          solves it with two O(M) sweeps per Fourier point
          (hermite.imex_factor, hermite.imex_solve).
        - IMEX calls `_krmhd_rhs_jit` 4 times per step (2 real k_z for
          the Elsasser midpoint + 2 k_z=0 for the IMEX nonlinear stages).
          XLA drops the unused half of each call, so the compiled step
          evaluates the Hermite RHS twice: 6 FFTs per moment (Issue #141).
        - lawson_rk4 evaluates the Elsasser RHS twice and advances g in the
          eigenbasis of T (Issue #184): 4 stages of 3 FFTs per slice,
          batched over all slices, plus one projection onto the slices and
          one back (2 dense (M+1)x(M+1) matmuls per step). Its compile time
          does not grow with M.
        - imex_ars343 also evaluates `_krmhd_rhs_jit` 4 times per step, all
          with k_z=0, one per stage. It makes three tridiagonal solves to
          imex_rk222's two and skips the `L*g` stencil.
        - Resistive damping is applied as a post-step exp() factor, the same
          as the Lawson convention. This is a 1st-order Strang split with
          the IMEX solve; `eta*dt*|k_perp|^(2r) << 1` in typical turbulence
          runs makes the extra splitting error negligible and sets a soft
          upper bound on `dt` (though streaming+collisions are themselves
          unconditionally stable under IMEX).

    Physics:
        - Linear propagation: Handled exactly (unconditionally stable)
        - Nonlinear terms: Elsasser O(dt²), Hermite O(dt⁴) in Lawson form
        - Dissipation: Exact exponential decay
        - Overall: O(dt²) for the coupled mixed scheme; O(dt³) for
          imex_ars343 with η = η_z = 0
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

        Parallel hyper-resistivity overflow risk (eta_z > 0):
        - NORMALIZED formulation: exp(-η_z·(kz²/kz²_max)^rz·dt)
        - Maximum damping rate at kz=kz_max is simply: η_z·dt (independent of Nz!)
        - Must satisfy: η_z·dt < 50 to avoid underflow (same thresholds as eta)

    Reference:
        - Thesis Chapter 2, §2.4 - GANDALF Algorithm
        - Eqs. 2.13-2.25 - Integrating factor implementation
        - Thesis §2.5.2 - Hyper-dissipation for inertial range studies
        - Original GANDALF: damping_kernel.cu:4-33 (dampz kernel) and
          timestep.cu:94-108 (per-step dampz on z± and every Hermite moment)
    """
    # Use provided nu or fall back to state.nu
    nu_effective = nu if nu is not None else state.nu

    _validate_step_args(
        state, dt, eta, nu_effective, hyper_r, hyper_n, scheme, eta_z, hyper_rz, closure,
        substep_cfl,
    )
    validate_mesh(mesh, scheme, state.M, state.grid.Nz)

    grid = state.grid
    fields = _fields_from_state(state)
    if mesh is not None:
        fields = place_on_mesh(fields, mesh)

    if scheme == "lawson_rk4":
        # Precompute Hermite streaming eigensystem for integrating factor (cached)
        _, eigenvalues, P, P_inv = compute_streaming_eigensystem(
            state.M, state.Lambda, closure
        )
        # The slice-basis step needs a real eigenbasis (Issue #184).
        if any(jnp.iscomplexobj(a) for a in (eigenvalues, P, P_inv)):
            raise ValueError(
                f"scheme='lawson_rk4' needs a real eigensystem of the streaming "
                f"matrix T, but it is complex at Lambda = {state.Lambda} (T has "
                f"complex eigenvalues for Lambda < 1). Use scheme='imex_rk222' "
                f"or 'imex_ars343'."
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
            eta_z=eta_z,
            hyper_rz=hyper_rz,
            streaming_eigenvalues=eigenvalues,
            streaming_P_T=P.T,
            streaming_P_inv_T=P_inv.T,
            closure=closure,
            hermite_chunk=hermite_chunk,
            mesh=mesh,
        )
    elif scheme == "imex_ars343":
        new_fields = _gandalf_step_imex343_jit(
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
            eta_z=eta_z,
            hyper_rz=hyper_rz,
            closure=closure,
        )
    elif scheme == "lawson_rk4_multirate":
        # Imported here: krmhd.multirate builds on this module's helpers.
        from krmhd.multirate import _gandalf_step_lawson_rk4_multirate_jit

        if hermite_chunk is not None and hermite_chunk < 1:
            raise ValueError(f"hermite_chunk must be >= 1 (got {hermite_chunk})")
        _, eigenvalues, P, P_inv = compute_streaming_eigensystem(
            state.M, state.Lambda, closure
        )
        if any(jnp.iscomplexobj(a) for a in (eigenvalues, P, P_inv)):
            raise ValueError(
                f"scheme='lawson_rk4_multirate' needs a real eigensystem of the "
                f"streaming matrix T, but it is complex at Lambda = {state.Lambda} "
                f"(T has complex eigenvalues for Lambda < 1). Use "
                f"scheme='imex_rk222' or 'imex_ars343'."
            )
        min_spacing = min(grid.Lx / grid.Nx, grid.Ly / grid.Ny, grid.Lz / grid.Nz)
        new_fields = _gandalf_step_lawson_rk4_multirate_jit(
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
            eta_z=eta_z,
            hyper_rz=hyper_rz,
            streaming_eigenvalues=eigenvalues,
            streaming_P_T=P.T,
            streaming_P_inv_T=P_inv.T,
            closure=closure,
            substep_courant=jnp.asarray(substep_cfl, dtype=float) * min_spacing,
            hermite_chunk=1 if hermite_chunk is None else hermite_chunk,
        )
    else:  # scheme == "imex_rk222"
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
            hyper_n,
            eta_z=eta_z,
            hyper_rz=hyper_rz,
            closure=closure,
        )

    # Convert back to KRMHDState (Pydantic validation at boundary)
    return _state_from_fields(new_fields, state)


@functools.lru_cache(maxsize=64)
def _streaming_eigenvalue_extremes(
    M: int, Lambda: float, closure: str
) -> Tuple[float, float, float]:
    """Extreme eigenvalues of the Hermite ladder matrix T (Issue #157).

    Returns (λ_lo, λ_hi, |λ|_complex): the smallest and largest real
    eigenvalues of compute_streaming_matrix(M, Λ, closure), and the largest
    modulus among complex eigenvalues (0.0 if all are real). Complex pairs
    appear only for Λ in (0, 1), where the g0-g1 block turns oscillatory;
    they sit near zero and never set the bound in practice.
    """
    import numpy as np

    eigs = np.linalg.eigvals(compute_streaming_matrix(M, Lambda, closure))
    is_real = np.abs(eigs.imag) <= 1e-10 * max(1.0, np.abs(eigs).max())
    real_eigs = eigs.real[is_real]
    lam_lo = float(real_eigs.min()) if real_eigs.size else 0.0
    lam_hi = float(real_eigs.max()) if real_eigs.size else 0.0
    lam_complex = float(np.abs(eigs[~is_real]).max()) if (~is_real).any() else 0.0
    return lam_lo, lam_hi, lam_complex


# compute_cfl_timestep modes: the Courant estimate, the exact slice-advection
# radius for the RK4-type schemes (Issue #189), or the macro step of the
# multirate scheme, which steps each slice to its own limit (Issue #191).
CFLMode = Literal["courant", "exact", "multirate"]


def compute_cfl_timestep(
    state: KRMHDState,
    v_A: float,
    cfl_safety: float = 0.3,
    closure: HermiteClosure = "zero",
    mode: CFLMode = "courant",
) -> float:
    """
    Compute maximum stable timestep from CFL condition.

    The CFL (Courant-Friedrichs-Lewy) condition ensures numerical stability
    by requiring that information propagates at most one grid cell per timestep:

        dt ≤ C * min(Δx, Δy, Δz) / v_max

    where C is a safety factor (typically 0.1-0.5 for RK2/RK4) and v_max is
    the maximum signal speed in the system:

        v_max = max(v_A, max|∇⊥z⁺|, max|∇⊥z⁻|, v_g)

    where, when Hermite moments are evolved (M > 0),

        v_g = max_j max|∇⊥(Φ + √β_i λ_j Ψ)|

    with λ_j the eigenvalues of the Hermite ladder matrix T
    (hermite.compute_streaming_matrix). See Physics below.

    With mode="exact" (Issue #189, for the RK4-type schemes lawson_rk4 and
    imex_ars343) the Hermite term v_g is replaced by the exact limit of the
    explicit RK4 stages,

        dt ≤ C * min(min(Δx, Δy, Δz) / max(v_A, max|∇⊥z±|), 2√2 / ρ),

    where ρ = max(ρ(√β_i λ_min), ρ(√β_i λ_max)) is the spectral radius of the
    slice advection operator a ↦ P{Φ + √β_i λ Ψ, a} (slice_advection_radius).

    With mode="multirate" (for scheme="lawson_rk4_multirate", Issue #191) the
    Hermite term is left out: dt is the macro step of z±, and the step itself
    gives each slice the substeps its own Courant limit needs.

    Args:
        state: Current KRMHD state (used to compute max velocity)
        v_A: Alfvén velocity (parallel wave speed)
        cfl_safety: Safety factor C ∈ (0, 1), default 0.3
        closure: Hermite closure used by the time stepper, "zero" (default)
            or "symmetric". It changes the eigenvalues of T and hence v_g.
        mode: "courant" (default), "exact" or "multirate". The exact mode runs a Lanczos
            iteration on the device, about 20-40 steps of 6 FFTs of two
            fields each; Courant takes 8 FFTs of one field. It falls back to
            the Courant term, with a warning, if Lanczos does not converge.

    Returns:
        Maximum safe timestep dt

    Example:
        >>> dt = compute_cfl_timestep(state, v_A=1.0, cfl_safety=0.3)
        >>> # Use this dt for time integration
        >>> new_state = gandalf_step(state, dt, eta=0.01, v_A=1.0)

    Physics:
        - Alfvén waves propagate at v_A along field lines (parallel).
        - In the Elsasser system, z⁺ is advected by ẑ×∇z⁻ and z⁻ by ẑ×∇z⁺,
          so the perpendicular advection speeds are |∇⊥z∓|, NOT |∇⊥φ|.
          Since φ = (z⁺+z⁻)/2 is their average, |∇⊥φ| ≤ max(|∇⊥z±|), with
          equality only for purely balanced states. A φ-based bound
          underestimates the advection by up to 2x when b⊥ ~ u⊥, and by an
          arbitrary factor in imbalanced states: for z⁺ = -z⁻, φ ≡ 0
          identically while the nonlinear advection (from Ψ = (z⁺-z⁻)/2)
          remains finite.
        - Hermite moments g obey ∂g/∂t + {Φ, g} + √β_i {Ψ, T g} + ... = 0.
          In the eigenbasis of T each eigencombination is advected by the
          stream function Φ + √β_i λ_j Ψ. The largest |λ_j| grows like
          √(2M) (≈ 4.9 at M = 16, 15.4 at M = 128), so a bound using
          √β_i·|∇⊥Ψ| alone underestimates the speed by up to λ_max
          (Issue #157). Since |∇⊥(Φ + λΨ)|² is convex in λ, the max over
          real λ_j is attained at the extreme eigenvalues, so only two
          gradient magnitudes are evaluated. Complex eigenvalues (Λ < 1)
          use the triangle bound |∇⊥Φ| + √β_i |λ| |∇⊥Ψ|.
        - CFL violation → exponential instability. The late-time blowups of
          Issue #136 were not one: dt stayed at the Alfvén limit, and the
          cause was the integrating-factor phase fixed in #148
          (docs/long_run_stability_study.md).
        - dt_cfl does not by itself guarantee a stable step (Issue #166).
          The explicit advection stage of scheme="imex_rk222" amplifies
          every advected mode by about (dt·ρ)⁴/8 per step, where ρ is the
          advection rate of that mode. It stays bounded only when
          hyper-collisions and hyper-resistivity damp the mode faster. In
          pure advection without dissipation it grows even at dt_cfl.
          scheme="lawson_rk4" is stable for |dt·ρ| ≤ 2√2 without
          dissipation, and so is "imex_ars343" when there is no streaming
          (see the exact mode below).
        - Why 2√2 (mode="exact"): for real band-limited Φ and Ψ the slice
          operator a ↦ P{Φ + cΨ, a} is skew-adjoint, so its eigenvalues
          are ±iω with |ω| ≤ ρ. The explicit Hermite stages of lawson_rk4
          are classical RK4 in the interaction picture, and the explicit
          tableau of imex_ars343 has the same stability polynomial
          R(z) = 1 + z + z²/2 + z³/6 + z⁴/24 (_ars343_explicit_a4). On the
          imaginary axis |R(iy)|² = 1 - y⁶(8 - y²)/576, which is ≤ 1 exactly
          for |y| ≤ 2√2. The bound is sharp: with frozen fields, lawson_rk4
          at dt = 1.05·2√2/ρ grows the top mode by 1.39 per step.
        - The Elsasser and v_A terms stay in the exact mode. lawson_rk4
          advances z± with midpoint RK2, whose |R(iy)|² = 1 + y⁴/4 has no
          stability interval, so z± at dt ≈ 2√2/ρ_z would grow fast without
          dissipation.
        - imex_ars343 meets the 2√2 bound only without streaming. With
          streaming the implicit stages couple imaginary streaming and
          imaginary advection, and the amplification exceeds 1 at every
          dt > 0 (Issue #200). In a frozen shear flow without dissipation
          the most amplified mode grows by 1.0011, 1.005 and 1.017 per
          step at dt·ρ = 0.54, 0.85 and 1.41, which are the Courant and
          exact dt at cfl_safety=0.3 and half the RK4 limit. Like
          imex_rk222, it then relies on hyper-collisions and
          hyper-resistivity, and the exact mode's larger dt raises the
          per-step growth. In that flow ν = η = 1 (r = n = 2) keep g
          bounded up to dt·ρ = 2.69. docs/numerical_methods.md has the
          worst case and the settings the scheme needs.

    Note:
        For typical KRMHD with strong guide field, v_A usually dominates
        over perpendicular velocities, so dt ~ Δz / v_A.
    """
    if mode not in get_args(CFLMode):
        raise ValueError(f"mode must be one of {get_args(CFLMode)}, got {mode!r}")
    dt_slices = None
    if state.M > 0 and mode == "exact":
        dt_slices = _exact_slice_timestep(state, closure)

    dt_cfl = cfl_timestep_estimate(
        state.z_plus, state.z_minus, state.grid, state.M, state.beta_i, state.Lambda,
        v_A, cfl_safety, closure, hermite=dt_slices is None and mode != "multirate",
    )
    if dt_slices is not None:
        dt_cfl = jnp.minimum(dt_cfl, cfl_safety * dt_slices)
    return float(dt_cfl)


def cfl_timestep_estimate(
    z_plus: Array,
    z_minus: Array,
    grid: SpectralGrid3D,
    M: int,
    beta_i: float,
    Lambda: float,
    v_A: float,
    cfl_safety: float = 0.3,
    closure: HermiteClosure = "zero",
    hermite: bool = True,
) -> Array:
    """compute_cfl_timestep's Courant estimate as a device scalar, traceable under jit.

    Same arithmetic as compute_cfl_timestep(mode="courant"), which calls it
    and converts the result to a Python float. z_plus and z_minus may be
    tracers; grid, M, beta_i, Lambda, v_A, cfl_safety and closure are
    trace-time constants (the grid may also be a traced pytree: only its
    arrays are read as data). hermite=False leaves out the Hermite advection
    term, which mode="exact" replaces. The device run loop (krmhd.runloop)
    evaluates it inside its step loop.
    """
    # Grid spacings
    dx = grid.Lx / grid.Nx
    dy = grid.Ly / grid.Ny
    dz = grid.Lz / grid.Nz
    min_spacing = min(dx, dy, dz)

    def perp_gradient(field: Array) -> Tuple[Array, Array]:
        """Real-space (∂x f, ∂y f) for a Fourier-space field f."""
        df_dx = derivative_x(field, grid.kx)
        df_dy = derivative_y(field, grid.ky)
        return (
            rfftn_inverse(df_dx, grid.Nz, grid.Ny, grid.Nx),
            rfftn_inverse(df_dy, grid.Nz, grid.Ny, grid.Nx),
        )

    def max_perp_gradient(field: Array) -> Array:
        """Max over real space of |∇⊥f| for a Fourier-space field f."""
        df_dx_real, df_dy_real = perp_gradient(field)
        return jnp.sqrt(df_dx_real**2 + df_dy_real**2).max()

    # Elsasser cross-advection speeds: z± is advected by ẑ×∇z∓, so both
    # |∇⊥z⁺| and |∇⊥z⁻| bound the perpendicular dynamics. This dominates
    # the |∇⊥φ| estimate (φ is the average of z±, so its gradient never
    # exceeds the larger of the two).
    v_z_plus_max = max_perp_gradient(z_plus)
    v_z_minus_max = max_perp_gradient(z_minus)

    v_max = jnp.maximum(v_A, jnp.maximum(v_z_plus_max, v_z_minus_max))

    if M > 0 and hermite:
        # Hermite advection (Issue #157): each eigencombination of the ladder
        # matrix T moves with stream function Φ + √β_i λ_j Ψ. |∇⊥(Φ + cΨ)|²
        # is convex in c, so over real λ_j the max sits at λ_lo or λ_hi.
        validate_closure(closure, M)
        lam_lo, lam_hi, lam_complex = _streaming_eigenvalue_extremes(
            M, float(Lambda), closure
        )
        sqrt_beta = math.sqrt(beta_i)
        Phi = (z_plus + z_minus) / 2.0
        Psi = (z_plus - z_minus) / 2.0
        dPhi_dx, dPhi_dy = perp_gradient(Phi)
        dPsi_dx, dPsi_dy = perp_gradient(Psi)

        def max_speed(c: float) -> Array:
            return jnp.sqrt(
                (dPhi_dx + c * dPsi_dx) ** 2 + (dPhi_dy + c * dPsi_dy) ** 2
            ).max()

        v_hermite = jnp.maximum(
            max_speed(sqrt_beta * lam_lo), max_speed(sqrt_beta * lam_hi)
        )
        if lam_complex > 0.0:
            v_complex = (
                jnp.sqrt(dPhi_dx**2 + dPhi_dy**2).max()
                + sqrt_beta * lam_complex * jnp.sqrt(dPsi_dx**2 + dPsi_dy**2).max()
            )
            v_hermite = jnp.maximum(v_hermite, v_complex)
        v_max = jnp.maximum(v_max, v_hermite)

    # CFL timestep
    dt_cfl = cfl_safety * min_spacing / v_max

    # Note: No parallel Hermite streaming constraint needed for either scheme.
    # - Lawson path: the integrating factor handles oscillatory streaming
    #   terms exactly (unitary propagator).
    # - IMEX path (Issue #137): streaming is part of the implicit operator
    #   L and solved unconditionally stably each stage.

    return dt_cfl


# =============================================================================
# Exact slice-advection radius (Issue #189)
# =============================================================================

# RK4 is stable on the imaginary axis for |dt·ρ| ≤ 2√2: with y = dt·ρ,
# |R(iy)|² = 1 - y⁶(8 - y²)/576 for R(z) = 1 + z + z²/2 + z³/6 + z⁴/24.
RK4_IMAGINARY_AXIS_LIMIT: float = 2.0 * math.sqrt(2.0)


class SliceAdvectionRadius(NamedTuple):
    """Result of slice_advection_radius, one entry per slice speed c."""

    rho: Array  # [S] spectral radius of a ↦ P{Φ + cΨ, a}
    iterations: Array  # Lanczos steps taken (int scalar, shared by all c)
    converged: Array  # [S] bool: Ritz residual ≤ tol·ρ² was reached


@partial(jax.jit, static_argnames=["Nz", "Ny", "Nx", "max_iter", "check_every"])
def slice_advection_radius(
    z_plus: Array,
    z_minus: Array,
    slice_speeds: Array,
    kx: Array,
    ky: Array,
    dealias_mask: Array,
    Nz: int,
    Ny: int,
    Nx: int,
    tol: float = 1e-3,
    max_iter: int = 150,
    check_every: int = 10,
    seed: int = 0,
) -> SliceAdvectionRadius:
    """
    Spectral radius ρ_c of the dealiased slice advection operator, on the device.

    Slice j of the Hermite hierarchy is advected by the stream function
    Φ + c_j Ψ with c_j = √β_i λ_j (see _slice_advection_rhs). The explicit
    stages of lawson_rk4 and imex_ars343 apply
        A_c: a ↦ P{Φ + cΨ, a},
    with P the projection onto the 2/3 band without the k = 0 mode. This
    function returns ρ_c for every c in slice_speeds, batched.

    Why the norm: Φ and Ψ are first projected onto the band. For band-limited
    f and a, every alias of the product ∂f·∂a lands outside the band, so
    P{f, a} is the exact Galerkin truncation of the continuum bracket, and
    ∫ b {f, a} = -∫ a {f, b} makes A_c skew-adjoint on the band. Its
    eigenvalues are then ±iω, and ρ_c = ‖A_c‖ = √λ_max(-A_c²). The
    projection matters only for z± content outside the band, which the
    nonlinear terms never create (initial conditions or forcing can). The
    stepper's operator built from such z± is not skew: aliasing puts its
    eigenvalues off the imaginary axis.

    Method: Lanczos on the symmetric operator -A_c², all c at once, inside
    lax.while_loop. Each step costs two calls to _slice_advection_rhs, the
    stepper's own operator: 6 FFTs of the S slices. The gradients of Φ and Ψ
    are computed once, outside the loop. Vectors stay in the half spectrum, with
    the real-space inner product (modes with 0 < kx index < Nx/2 count
    twice). There is no reorthogonalization, so a long run grows ghost
    copies of converged Ritz values, but none exceeds λ_max. Every
    check_every steps the top Ritz pair (θ, s) of the tridiagonal matrix is
    computed, and c is converged when β_k·|s_k| ≤ tol·θ, ARPACK's criterion
    (scipy eigsh's tol). θ approaches ρ_c² from below and lies within tol·θ
    of an eigenvalue, so √θ is within tol/2 of a singular value of A_c. On
    developed turbulence at 64³ and 128³ (M = 64) it converges in 20-40
    steps at tol = 1e-3 and agrees with eigsh to about 1e-7.

    Use it as dt ≤ RK4_IMAGINARY_AXIS_LIMIT / max_c ρ_c; see
    compute_cfl_timestep(mode="exact").

    Args:
        z_plus, z_minus: Elsasser fields in Fourier space [Nz, Ny, Nx//2+1]
        slice_speeds: Coefficients c of Ψ [S], real. ‖A_c‖ is convex in c,
            so the two extremes √β_i λ_min and √β_i λ_max bound every slice.
        kx, ky: Wavenumber arrays
        dealias_mask: 2/3 dealiasing mask [Nz, Ny, Nx//2+1]
        Nz, Ny, Nx: Grid dimensions (static)
        tol: Relative tolerance on ρ_c² (traced)
        max_iter: Iteration cap (static; rounded up to a multiple of
            check_every). It sizes the stored tridiagonal matrix.
        check_every: Lanczos steps between convergence checks (static)
        seed: PRNG seed of the random start vector (traced)

    Returns:
        SliceAdvectionRadius(rho, iterations, converged). An entry with
        converged False underestimates ρ_c.
    """
    real_dtype = jnp.real(z_plus).dtype
    complex_dtype = z_plus.dtype
    S = slice_speeds.shape[0]
    K = -(-max_iter // check_every) * check_every

    zp = z_plus * dealias_mask
    zm = z_minus * dealias_mask
    c = jnp.asarray(slice_speeds, real_dtype)
    kx = jnp.asarray(kx, real_dtype)
    ky = jnp.asarray(ky, real_dtype)
    band = dealias_mask.at[0, 0, 0].set(False)

    # Real-space inner product on the half spectrum: a mode with
    # 0 < kx index < Nx/2 stands for itself and its conjugate partner.
    weight = jnp.full(Nx // 2 + 1, 2.0, real_dtype).at[0].set(1.0).at[-1].set(1.0)

    def dot(u: Array, v: Array) -> Array:
        return jnp.sum(weight * (u.real * v.real + u.imag * v.imag), axis=(1, 2, 3))

    def per_slice(x: Array) -> Array:
        return x[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]

    grads = _perp_gradients(zp, zm, kx, ky, Nz, Ny, Nx)

    def minus_A_squared(v: Array) -> Array:
        # _slice_advection_rhs returns -A_c v; two applications give A_c².
        a = _slice_advection_rhs(grads, v, c, kx, ky, dealias_mask, Nz, Ny, Nx)
        return -_slice_advection_rhs(grads, a, c, kx, ky, dealias_mask, Nz, Ny, Nx)

    noise = jax.random.normal(jax.random.PRNGKey(seed), (Nz, Ny, Nx), real_dtype)
    v0 = (jnp.fft.rfftn(noise) * band).astype(complex_dtype)
    v0 = jnp.broadcast_to(v0, (S,) + v0.shape)
    v0 = v0 / per_slice(jnp.sqrt(dot(v0, v0)))

    def lanczos_step(k: Array, carry: Tuple) -> Tuple:
        v_prev, v, beta_prev, alphas, betas = carry
        w = minus_A_squared(v)
        alpha = dot(v, w)
        w = w - per_slice(alpha) * v - per_slice(beta_prev) * v_prev
        beta = jnp.sqrt(dot(w, w))
        # beta = 0: the Krylov space is invariant (zero fields, tiny grids)
        # and the Ritz values are exact; later steps then add zero rows.
        v_next = w / per_slice(jnp.where(beta > 0, beta, 1.0))
        return v, v_next, beta, alphas.at[:, k].set(alpha), betas.at[:, k].set(beta)

    idx = jnp.arange(K)
    upper = jnp.eye(K, k=1, dtype=real_dtype)

    def top_ritz_pair(k: Array, alphas: Array, betas: Array) -> Tuple[Array, Array]:
        """Top eigenvalue θ of T_k and the residual β_k·|s_k| of its Ritz vector.

        T_k sits in the leading k×k block of a K×K matrix that is zero
        elsewhere, which adds only zero eigenvalues below θ.
        """
        off = jnp.where(idx < k - 1, betas, 0.0)
        T = (
            alphas[:, :, jnp.newaxis] * jnp.eye(K, dtype=real_dtype)
            + off[:, :, jnp.newaxis] * upper
            + off[:, jnp.newaxis, :] * upper.T
        )
        evals, evecs = jnp.linalg.eigh(T)
        return evals[:, -1], betas[:, k - 1] * jnp.abs(evecs[:, k - 1, -1])

    def not_done(state: Tuple) -> Array:
        k, *_, converged = state
        return (k < K) & ~jnp.all(converged)

    def check_block(state: Tuple) -> Tuple:
        k, carry, _, converged = state
        carry = jax.lax.fori_loop(
            0, check_every, lambda i, cr: lanczos_step(k + i, cr), carry
        )
        k = k + check_every
        theta, residual = top_ritz_pair(k, carry[3], carry[4])
        return k, carry, theta, converged | (residual <= tol * theta)

    zeros = jnp.zeros(S, real_dtype)
    carry0 = (
        jnp.zeros_like(v0),
        v0,
        zeros,
        jnp.zeros((S, K), real_dtype),
        jnp.zeros((S, K), real_dtype),
    )
    k, _, theta, converged = jax.lax.while_loop(
        not_done, check_block, (jnp.asarray(0), carry0, zeros, jnp.zeros(S, bool))
    )
    return SliceAdvectionRadius(
        rho=jnp.sqrt(jnp.maximum(theta, 0.0)), iterations=k, converged=converged
    )


def _exact_slice_timestep(state: KRMHDState, closure: str) -> float | None:
    """2√2/ρ for the slice advection of state (compute_cfl_timestep, mode="exact").

    ρ is the larger radius of the two extreme slices, c = √β_i λ_min and
    √β_i λ_max; the norm is convex in c, so they bound every slice. Returns
    None, with a warning, when Lanczos does not converge, and the caller
    falls back to the Courant estimate.
    """
    validate_closure(closure, state.M)
    lam_lo, lam_hi, lam_complex = _streaming_eigenvalue_extremes(
        state.M, float(state.Lambda), closure
    )
    if lam_complex > 0.0:
        raise ValueError(
            "mode='exact' needs real eigenvalues of the streaming matrix T, "
            f"but T has complex ones for Lambda = {state.Lambda} (Lambda < 1). "
            "Use mode='courant'."
        )
    grid = state.grid
    sqrt_beta = math.sqrt(state.beta_i)
    radius = slice_advection_radius(
        state.z_plus,
        state.z_minus,
        jnp.array([sqrt_beta * lam_lo, sqrt_beta * lam_hi]),
        grid.kx,
        grid.ky,
        grid.dealias_mask,
        grid.Nz,
        grid.Ny,
        grid.Nx,
    )
    if not bool(jnp.all(radius.converged)):
        warnings.warn(
            f"slice_advection_radius did not converge in {int(radius.iterations)} "
            "Lanczos steps; using the Courant estimate for the Hermite term.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    rho = float(jnp.max(radius.rho))
    return RK4_IMAGINARY_AXIS_LIMIT / rho if rho > 0.0 else math.inf
