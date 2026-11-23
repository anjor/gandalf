"""
Tests for the slice-basis Lawson-RK4 step (scheme="lawson_rk4", Issue #184).

The Hermite sector of the Lawson-RK4 step runs in the eigenbasis of the
streaming matrix T, where the streaming integrating factor and the explicit
perpendicular advection are both diagonal. These tests compare it with the
same scheme written in the Hermite basis, using an independent propagator
(scipy.linalg.expm of -i√β_i k_z T t, no eigendecomposition) and an explicit
term built moment by moment from poisson_bracket_3d and T. They also check
that this reference equals GANDALF's own Hermite right-hand side, that the
ideal step conserves the free energy to O(dt⁶), that out-of-band input is
projected away, that a float32 run stays complex64, and that the traced graph no longer
grows with the number of Hermite moments.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64
from scipy.linalg import expm

from krmhd.hermite import compute_streaming_matrix, free_energy_weights
from krmhd.physics import (
    KRMHDState,
    enforce_hermitian_state,
    poisson_bracket_3d,
    z_minus_rhs,
    z_plus_rhs,
)
from krmhd.spectral import SpectralGrid3D
from krmhd.timestepping import (
    KRMHDFields,
    _fields_from_state,
    _gandalf_step_lawson_rk4_jit,
    _krmhd_rhs_jit,
    gandalf_step,
)

# Fourier amplitude of the random fields. GANDALF's forward FFT is unnormalised,
# so on these small grids 50 gives real-space fields of order 1e-1 and an
# advection step of order 1e-2 at dt = 3e-3: strong enough that an error in the
# stage algebra shows well above round-off.
AMPLITUDE = 50.0


def _random_state(
    grid: SpectralGrid3D,
    M: int,
    Lambda: float,
    nu: float,
    beta_i: float = 0.7,
    amplitude: float = AMPLITUDE,
    seed: int = 0,
    in_band: bool = True,
) -> KRMHDState:
    """Random z± and g with Hermitian pairs on the kx = 0 and Nyquist planes.

    With in_band=False, g also has content outside the 2/3 band.
    """
    rng = np.random.default_rng(seed)
    shape = (grid.Nz, grid.Ny, grid.Nx // 2 + 1)
    mask = np.asarray(grid.dealias_mask)

    def draw(*s: int, band: bool = True) -> jnp.ndarray:
        a = (rng.standard_normal(s) + 1j * rng.standard_normal(s)) * amplitude
        if band:
            a = a * mask.reshape(shape + (1,) * (len(s) - 3))
        return jnp.asarray(a)

    state = KRMHDState(
        z_plus=draw(*shape),
        z_minus=draw(*shape),
        g=draw(*shape, M + 1, band=in_band),
        M=M,
        beta_i=beta_i,
        v_th=1.0,
        nu=nu,
        Lambda=Lambda,
        time=0.0,
        grid=grid,
    )
    return enforce_hermitian_state(state)


def _hermite_advection(
    z_plus: jnp.ndarray,
    z_minus: jnp.ndarray,
    g: jnp.ndarray,
    T: np.ndarray,
    beta_i: float,
    grid: SpectralGrid3D,
) -> jnp.ndarray:
    """-{Φ, g_m} - √β_i {Ψ, (T g)_m} for every m, moment by moment in the Hermite basis.

    Built from poisson_bracket_3d and the matrix T, so it is defined for every
    M including M = 1, and is independent of the slice code. Dealiased, with
    the k = 0 mode set to zero, like g0_rhs, g1_rhs and gm_rhs.
    """
    Nz, Ny, Nx = grid.Nz, grid.Ny, grid.Nx
    phi = (z_plus + z_minus) / 2.0
    psi = (z_plus - z_minus) / 2.0
    Tg = jnp.einsum("mn,zyxn->zyxm", jnp.asarray(T, dtype=jnp.real(g).dtype), g)
    s = math.sqrt(beta_i)
    out = []
    for m in range(g.shape[-1]):
        b = -poisson_bracket_3d(phi, g[..., m], grid.kx, grid.ky, Nz, Ny, Nx, grid.dealias_mask)
        b = b - s * poisson_bracket_3d(psi, Tg[..., m], grid.kx, grid.ky, Nz, Ny, Nx, grid.dealias_mask)
        out.append((b * grid.dealias_mask).at[0, 0, 0].set(0.0))
    return jnp.stack(out, axis=-1)


def _reference_lawson_rk4(
    state: KRMHDState,
    dt: float,
    eta: float,
    nu: float,
    hyper_r: int,
    hyper_n: int,
    eta_z: float,
    hyper_rz: int,
    closure: str,
) -> KRMHDFields:
    """
    One Lawson-RK4 step written in the Hermite basis, independently of the kernel.

    Elsasser sector: the integrating-factor midpoint update of thesis §2.4.
    Hermite sector: classical RK4 in the interaction picture, with the
    streaming propagator E = exp(-i√β_i k_z T dt/2) from scipy's expm and the
    explicit term N(g) = -{Φ, g} - √β_i {Ψ, T g} from _hermite_advection:
        k1 = N(g, z^n),  k2 = N(E(g + dt/2 k1), z^(n+1/2)),
        k3 = N(E g + dt/2 k2, z^(n+1/2)),  k4 = N(E(E g + dt k3), z^(n+1)),
        g^(n+1) = E²(g + dt/6 k1) + dt/3 E(k2 + k3) + dt/6 k4.
    Then hyper-resistivity, hyper-collisions, parallel damping and the 2/3
    projection, as exponential factors after the step.
    """
    grid = state.grid
    M = state.M
    Nz, Ny, Nx = grid.Nz, grid.Ny, grid.Nx
    kx, ky, kz, mask = grid.kx, grid.ky, grid.kz, grid.dealias_mask
    kz_3d = kz[:, None, None]
    zp, zm, g = state.z_plus, state.z_minus, state.g

    def elsasser_nl(a, b):
        rp = z_plus_rhs(a, b, kx, ky, kz, mask, 0.0, Nz, Ny, Nx)
        rm = z_minus_rhs(a, b, kx, ky, kz, mask, 0.0, Nz, Ny, Nx)
        return rp - 1j * kz_3d * a, rm + 1j * kz_3d * b

    nlp, nlm = elsasser_nl(zp, zm)
    zp_h = jnp.exp(1j * kz_3d * dt / 2) * (zp + dt / 2 * nlp)
    zm_h = jnp.exp(-1j * kz_3d * dt / 2) * (zm + dt / 2 * nlm)
    nlp, nlm = elsasser_nl(zp_h, zm_h)
    zp_n = jnp.exp(1j * kz_3d * dt) * zp + jnp.exp(1j * kz_3d * dt / 2) * dt * nlp
    zm_n = jnp.exp(-1j * kz_3d * dt) * zm + jnp.exp(-1j * kz_3d * dt / 2) * dt * nlm

    if M > 0:
        T = compute_streaming_matrix(M, state.Lambda, closure)
        s = math.sqrt(state.beta_i)
        E_half = jnp.asarray(
            np.stack([expm(-1j * s * float(k) * T * dt / 2) for k in np.asarray(kz)]),
            dtype=g.dtype,
        )

        def E(field):
            return jnp.einsum("zmn,zyxn->zyxm", E_half, field)

        def N(a, b, field):
            return _hermite_advection(a, b, field, T, state.beta_i, grid)

        k1 = N(zp, zm, g)
        k2 = N(zp_h, zm_h, E(g + dt / 2 * k1))
        k3 = N(zp_h, zm_h, E(g) + dt / 2 * k2)
        k4 = N(zp_n, zm_n, E(E(g) + dt * k3))
        g = E(E(g + dt / 6 * k1)) + dt / 3 * E(k2 + k3) + dt / 6 * k4

    kperp2 = kx[None, None, :] ** 2 + ky[None, :, None] ** 2
    kperp2_max = kx[(Nx - 1) // 3] ** 2 + ky[(Ny - 1) // 3] ** 2
    damp = jnp.exp(-eta * (kperp2 / kperp2_max) ** hyper_r * dt)
    zp_n, zm_n, g = zp_n * damp, zm_n * damp, g * damp[..., None]
    if M >= 2:
        m = np.arange(M + 1)
        coll = np.where(m >= 2, np.exp(-nu * (m / M) ** hyper_n * dt), 1.0)
        g = g * jnp.asarray(coll, dtype=jnp.real(g).dtype)
    idz = (Nz - 1) // 3
    if idz >= 1:
        dz = jnp.exp(-eta_z * (kz_3d ** 2 / kz[idz] ** 2) ** hyper_rz * dt)
        zp_n, zm_n, g = zp_n * dz, zm_n * dz, g * dz[..., None]
    g = g * mask[..., None]
    return KRMHDFields(zp_n, zm_n, g, state.time + dt)


def _eigensystem(M: int, Lambda: float, closure: str):
    """Eigendecomposition of T in the current precision, bypassing the lru_cache of
    compute_streaming_eigensystem (whose arrays keep the dtype of their first call)."""
    T = compute_streaming_matrix(M, Lambda, closure)
    evals, P = np.linalg.eig(T)
    return jnp.asarray(evals.real), jnp.asarray(P.real), jnp.asarray(np.linalg.inv(P).real)


def _kernel_step(state, dt, eta, nu, hyper_r, hyper_n, eta_z, hyper_rz, closure, eigensystem):
    grid = state.grid
    evals, P, P_inv = eigensystem
    return _gandalf_step_lawson_rk4_jit(
        _fields_from_state(state), dt, grid.kx, grid.ky, grid.kz, grid.dealias_mask,
        eta, 1.0, state.beta_i, nu, state.Lambda, state.M, grid.Nz, grid.Ny, grid.Nx,
        hyper_r, hyper_n, eta_z=eta_z, hyper_rz=hyper_rz,
        streaming_eigenvalues=evals, streaming_P_T=P.T, streaming_P_inv_T=P_inv.T,
        closure=closure,
    )


def _rel_diff(a: jnp.ndarray, b: jnp.ndarray) -> float:
    scale = float(jnp.max(jnp.abs(b)))
    return float(jnp.max(jnp.abs(a - b))) / (scale if scale > 0 else 1.0)


# (M, Lambda, closure, nu, hyper_n, eta_z): fluid M = 0, M = 1 (zero closure only),
# the smallest M with collisions, both closures, Λ from 1 (even M) to the large-Λ⁺
# branch, hyper-collisions and parallel damping.
CASES = [
    (0, 1.0, "zero", 0.0, 1, 0.0),
    (1, 2.0, "zero", 0.0, 1, 0.3),
    (2, math.sqrt(5.0), "zero", 3.0, 6, 0.0),
    (3, 1.7, "symmetric", 1.0, 2, 0.5),
    (8, math.sqrt(5.0), "zero", 2.0, 6, 1.0),
    (8, 1.0, "zero", 0.5, 3, 0.0),
    (11, 39.1, "symmetric", 5.0, 4, 0.0),
]
STEP = dict(dt=3e-3, eta=2.0, hyper_r=2, hyper_rz=2)


@pytest.mark.parametrize("M,Lambda,closure", [
    (2, math.sqrt(5.0), "zero"),
    (3, 1.7, "symmetric"),
    (8, 1.0, "zero"),
    (11, 39.1, "symmetric"),
])
def test_reference_advection_matches_gandalf_rhs(M, Lambda, closure):
    """The reference's explicit term, built from T, equals GANDALF's moment-by-moment
    Hermite RHS with k_z = 0 (g0_rhs, g1_rhs, gm_rhs) for M ≥ 2."""
    with enable_x64():
        grid = SpectralGrid3D.create(Nx=12, Ny=10, Nz=8)
        state = _random_state(grid, M, Lambda, 0.0, seed=20 + M)
        ref = _hermite_advection(state.z_plus, state.z_minus, state.g,
                                 compute_streaming_matrix(M, Lambda, closure), state.beta_i, grid)
        rhs = _krmhd_rhs_jit(
            _fields_from_state(state), grid.kx, grid.ky, jnp.zeros_like(grid.kz), grid.dealias_mask,
            0.0, 1.0, state.beta_i, 0.0, Lambda, M, grid.Nz, grid.Ny, grid.Nx, closure=closure,
        ).g
        assert _rel_diff(ref, rhs) < 1e-13


@pytest.mark.parametrize("M,Lambda,closure,nu,hyper_n,eta_z", CASES)
def test_slice_step_matches_hermite_basis_reference_float64(M, Lambda, closure, nu, hyper_n, eta_z):
    """In float64 the slice-basis step equals the Hermite-basis scheme to round-off."""
    with enable_x64():
        grid = SpectralGrid3D.create(Nx=12, Ny=10, Nz=8)
        state = _random_state(grid, M, Lambda, nu, seed=M)
        kw = dict(STEP, nu=nu, hyper_n=hyper_n, eta_z=eta_z, closure=closure)
        new = _kernel_step(state, eigensystem=_eigensystem(M, Lambda, closure), **kw)
        ref = _reference_lawson_rk4(state, **kw)
        assert new.g.dtype == jnp.complex128
        assert _rel_diff(new.z_plus, ref.z_plus) < 1e-13
        assert _rel_diff(new.z_minus, ref.z_minus) < 1e-13
        assert _rel_diff(new.g, ref.g) < 1e-12, f"g differs by {_rel_diff(new.g, ref.g):.2e}"


@pytest.mark.parametrize("M,Lambda,closure,nu,hyper_n,eta_z", CASES)
def test_gandalf_step_lawson_matches_reference_float32(M, Lambda, closure, nu, hyper_n, eta_z):
    """Through the public gandalf_step in the default precision (float32/complex64)."""
    grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8)
    state = _random_state(grid, M, Lambda, nu, seed=10 + M)
    kw = dict(STEP, nu=nu, hyper_n=hyper_n, eta_z=eta_z, closure=closure)
    new = gandalf_step(state, v_A=1.0, scheme="lawson_rk4", **kw)
    ref = _reference_lawson_rk4(state, **kw)
    assert new.g.dtype == state.g.dtype
    assert _rel_diff(new.z_plus, ref.z_plus) < 1e-5
    assert _rel_diff(new.g, ref.g) < 1e-5, f"g differs by {_rel_diff(new.g, ref.g):.2e}"
    assert new.time == pytest.approx(STEP["dt"])


def test_slice_step_projects_out_of_band_input():
    """g with content outside the 2/3 band: the step removes it and the in-band part
    still matches the reference."""
    with enable_x64():
        grid = SpectralGrid3D.create(Nx=12, Ny=10, Nz=8)
        M, Lambda, closure = 6, math.sqrt(5.0), "zero"
        state = _random_state(grid, M, Lambda, 1.0, seed=5, in_band=False)
        outside = ~grid.dealias_mask[..., None]
        assert float(jnp.max(jnp.abs(state.g * outside))) > 0.0
        kw = dict(STEP, nu=1.0, hyper_n=6, eta_z=0.0, closure=closure)
        new = _kernel_step(state, eigensystem=_eigensystem(M, Lambda, closure), **kw)
        ref = _reference_lawson_rk4(state, **kw)
        assert float(jnp.max(jnp.abs(new.g * outside))) == 0.0
        assert _rel_diff(new.g, ref.g) < 1e-12


def test_slice_step_conserves_free_energy_without_dissipation():
    """Ideal step (η = ν = 0): streaming and advection conserve the free energy
    W = Σ_m p_m Σ_k w_k |g_m,k|² (w_k the rfft half-space weights). The streaming factor is
    exact and RK4 on a skew operator has |R(iy)|² = 1 - y⁶/72 + ..., so W changes by
    O(dt⁶) per step: halving dt divides the change by 2⁶ = 64."""
    with enable_x64():
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8)
        M, Lambda, closure = 10, math.sqrt(5.0), "zero"
        state = _random_state(grid, M, Lambda, 0.0, seed=3)
        p = jnp.asarray(free_energy_weights(M, Lambda, closure))
        wx = np.full(grid.Nx // 2 + 1, 2.0)
        wx[0] = 1.0
        wx[-1] = 1.0  # Nx even: the Nyquist plane is its own partner
        w = jnp.asarray(wx)[None, None, :, None]

        def W(g):
            return float(jnp.sum(p * jnp.sum(w * jnp.abs(g) ** 2, axis=(0, 1, 2))))

        eig = _eigensystem(M, Lambda, closure)
        errs = []
        for dt in (8e-3, 4e-3):
            new = _kernel_step(state, dt, 0.0, 0.0, 1, 1, 0.0, 1, closure, eig)
            errs.append(abs(W(new.g) - W(state.g)) / W(state.g))
        assert errs[0] < 1e-6, errs
        assert 48.0 < errs[0] / errs[1] < 80.0, errs


def test_float64_eigensystem_in_float32_run_keeps_complex64():
    """An eigensystem built with x64 enabled (as a cache filled in a float64 run would be)
    does not promote a later float32 run: the step returns complex64."""
    with enable_x64():
        eig = _eigensystem(4, math.sqrt(5.0), "zero")
    assert eig[1].dtype == jnp.float64
    grid = SpectralGrid3D.create(Nx=8, Ny=8, Nz=8)
    state = _random_state(grid, 4, math.sqrt(5.0), 1.0, seed=7)
    assert state.g.dtype == jnp.complex64
    new = _kernel_step(state, 1e-3, 1.0, 1.0, 2, 6, 0.0, 1, "zero", eig)
    assert new.g.dtype == jnp.complex64
    assert bool(jnp.all(jnp.isfinite(new.g)))


def test_complex_eigensystem_raises():
    """The slice step uses real FFTs, so a complex eigenbasis must be rejected."""
    grid = SpectralGrid3D.create(Nx=8, Ny=8, Nz=8)
    M, Lambda = 4, math.sqrt(5.0)
    state = _random_state(grid, M, Lambda, 1.0, seed=8)
    evals, P, P_inv = _eigensystem(M, Lambda, "zero")
    with pytest.raises(ValueError, match="real eigensystem"):
        _kernel_step(state, 1e-3, 1.0, 1.0, 2, 6, 0.0, 1, "zero",
                     (evals, P.astype(jnp.complex64), P_inv.astype(jnp.complex64)))


def _count_primitive(jaxpr, name: str) -> int:
    """Number of equations with primitive `name` in a jaxpr, including nested jaxprs."""
    count = 0
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == name:
            count += 1
        for value in eqn.params.values():
            for sub in value if isinstance(value, (list, tuple)) else (value,):
                inner = getattr(sub, "jaxpr", sub)
                if hasattr(inner, "eqns"):
                    count += _count_primitive(inner, name)
    return count


def test_traced_graph_does_not_grow_with_M():
    """The Hermite stages are batched over slices, so the number of FFT calls in the traced
    step is the same at every M (the moment-by-moment form grew linearly with M)."""
    grid = SpectralGrid3D.create(Nx=8, Ny=8, Nz=8)
    counts = {}
    for M in (4, 12):
        state = _random_state(grid, M, math.sqrt(5.0), 1.0, seed=M)
        evals, P, P_inv = _eigensystem(M, math.sqrt(5.0), "zero")

        def step(fields, M=M, evals=evals, P=P, P_inv=P_inv):
            return _gandalf_step_lawson_rk4_jit(
                fields, 1e-3, grid.kx, grid.ky, grid.kz, grid.dealias_mask,
                1.0, 1.0, state.beta_i, 1.0, state.Lambda, M, grid.Nz, grid.Ny, grid.Nx,
                2, 6, eta_z=0.0, hyper_rz=1, streaming_eigenvalues=evals,
                streaming_P_T=P.T, streaming_P_inv_T=P_inv.T, closure="zero",
            )

        jaxpr = jax.make_jaxpr(step)(_fields_from_state(state))
        counts[M] = _count_primitive(jaxpr.jaxpr, "fft")
    assert counts[4] == counts[12], counts
    assert counts[4] > 0
