"""
Tests for the multirate slice step (scheme="lawson_rk4_multirate", Issue #191).

Slice j takes n_j = ceil(dt·v_j / (C·Δ)) Lawson-RK4 substeps, with Φ and Ψ
interpolated in time; z± take the macro step as in lawson_rk4. The tests check
that one substep per slice is lawson_rk4, that n substeps in a frozen flow are
n lawson_rk4 steps of dt/n slice by slice, the 2√2 stability limit of every
slice's own substep, the convergence order (4 in a frozen flow, 2 with
evolving z±), the free energy of the ideal step, the device loop, the CFL
mode and the argument checks. docs/multirate_slices_study.md has the
turbulence study.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64

from krmhd.hermite import (
    compute_streaming_eigensystem,
    compute_streaming_matrix,
    free_energy_weights,
)
from krmhd.multirate import exact_substep_cfl, multirate_substeps
from krmhd.physics import KRMHDState, enforce_hermitian_state
from krmhd.runloop import CflDt, DeviceLoop, FixedDt, RunConfig, StepParams
from krmhd.spectral import SpectralGrid3D
from krmhd.timestepping import (
    RK4_IMAGINARY_AXIS_LIMIT,
    _perp_gradients,
    cfl_timestep_estimate,
    compute_cfl_timestep,
    gandalf_step,
    slice_advection_radius,
)

LAMBDA = math.sqrt(5.0)


def _eigensystem(M: int, Lambda: float, closure: str = "zero"):
    """Real eigenpairs of T, unordered (as np.linalg.eig returns them)."""
    evals, P = np.linalg.eig(compute_streaming_matrix(M, Lambda, closure))
    return evals.real, P.real, np.linalg.inv(P).real


def _turbulent_state(
    grid: SpectralGrid3D, M: int, amplitude: float, seed: int = 0, nu: float = 0.0,
    beta_i: float = 1.0, Lambda: float = LAMBDA,
) -> KRMHDState:
    """z± with a k⊥^(-5/3)-like spectrum and random g, all in the 2/3 band."""
    rng = np.random.default_rng(seed)
    nx = np.fft.rfftfreq(grid.Nx, 1.0 / grid.Nx)[None, None, :]
    ny = np.fft.fftfreq(grid.Ny, 1.0 / grid.Ny)[None, :, None]
    nz = np.fft.fftfreq(grid.Nz, 1.0 / grid.Nz)[:, None, None]
    n_perp = np.maximum(np.sqrt(nx**2 + ny**2), 1.0)
    shape = n_perp ** (-11.0 / 6.0) * np.exp(-np.abs(nz) / (1.0 + n_perp ** (2.0 / 3.0)))
    mask = np.asarray(grid.dealias_mask)
    z = [np.fft.rfftn(rng.standard_normal((grid.Nz, grid.Ny, grid.Nx))) * shape * mask * amplitude
         for _ in range(2)]
    g_shape = z[0].shape + (M + 1,)
    g = (rng.standard_normal(g_shape) + 1j * rng.standard_normal(g_shape)) * mask[..., None]
    return enforce_hermitian_state(KRMHDState(
        z_plus=jnp.asarray(z[0]), z_minus=jnp.asarray(z[1]), g=jnp.asarray(g), M=M,
        beta_i=beta_i, v_th=1.0, nu=nu, Lambda=Lambda, time=0.0, grid=grid,
    ))


def _shear_state(M: int = 8, beta_i: float = 1.0) -> KRMHDState:
    """A frozen flow: z± = (1 ± 1/2)Φ with Φ = Φ(x), and random g.

    Every Elsasser bracket has a ∂y z± = 0 factor, so z± stay exactly fixed and
    the interpolated Φ and Ψ are exact. Slice j is advected by (1 + c_j/2)Φ,
    which commutes with streaming, so every slice keeps its own norm.
    """
    grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8)
    x = np.arange(16) / 16
    phi = np.broadcast_to(np.cos(2 * np.pi * x) + 0.5 * np.sin(4 * np.pi * x), (8, 16, 16))
    phi = jnp.fft.rfftn(phi)
    rng = np.random.default_rng(0)
    shape = (8, 16, 9, M + 1)
    g = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * np.asarray(
        grid.dealias_mask
    )[..., None]
    return enforce_hermitian_state(KRMHDState(
        z_plus=1.5 * phi, z_minus=0.5 * phi, g=jnp.asarray(g), M=M, beta_i=beta_i, v_th=1.0,
        nu=0.0, Lambda=LAMBDA, time=0.0, grid=grid,
    ))


def _counts(state: KRMHDState, dt: float, substep_cfl: float, closure: str = "zero") -> np.ndarray:
    """The substeps the step gives each slice, in the order of np.linalg.eig's eigenvalues."""
    grid = state.grid
    lam, _, _ = _eigensystem(state.M, state.Lambda, closure)
    spacing = min(grid.Lx / grid.Nx, grid.Ly / grid.Ny, grid.Lz / grid.Nz)
    return np.asarray(multirate_substeps(
        state.z_plus, state.z_minus, jnp.asarray(math.sqrt(state.beta_i) * lam), dt,
        substep_cfl * spacing, grid.kx, grid.ky, grid.Nz, grid.Ny, grid.Nx,
    ))


def _slices(state: KRMHDState) -> np.ndarray:
    """Real-space slices h_j = Σ_m P⁻¹[j, m] g_m, [M+1, Nz, Ny, Nx]."""
    _, _, P_inv = _eigensystem(state.M, state.Lambda)
    h = np.einsum("jm,zyxm->jzyx", P_inv, np.asarray(state.g))
    return np.fft.irfftn(h, s=(state.grid.Nz, state.grid.Ny, state.grid.Nx), axes=(1, 2, 3))


def _slice_norms(state: KRMHDState) -> np.ndarray:
    return np.sqrt(np.sum(_slices(state) ** 2, axis=(1, 2, 3)))


def _rel(a: jnp.ndarray, b: jnp.ndarray) -> float:
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


# (M, closure, nu, hyper_n, eta_z, hermite_chunk)
CASES = [
    (0, "zero", 0.0, 1, 0.0, None),
    (1, "zero", 0.0, 1, 0.3, None),
    (6, "zero", 3.0, 6, 0.0, None),
    (7, "symmetric", 1.0, 2, 0.5, 3),
    (10, "zero", 0.5, 4, 0.0, 11),
]


@pytest.mark.parametrize("M, closure, nu, hyper_n, eta_z, chunk", CASES)
def test_one_substep_per_slice_is_lawson_rk4(M, closure, nu, hyper_n, eta_z, chunk):
    """At the Courant dt of lawson_rk4 every slice takes one substep, and the step
    is lawson_rk4's to float64 round-off: the stage values are the same."""
    with enable_x64():
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8, Lx=1.0, Ly=1.0, Lz=1.0)
        state = _turbulent_state(grid, M, amplitude=0.5, nu=nu)
        dt = compute_cfl_timestep(state, v_A=1.0, cfl_safety=0.3, closure=closure)
        assert np.all(_counts(state, dt, 0.3, closure) == 1)
        kw = dict(eta=0.7, v_A=1.0, hyper_r=2, hyper_n=hyper_n, eta_z=eta_z, closure=closure)
        ref = gandalf_step(state, dt, scheme="lawson_rk4", **kw)
        out = gandalf_step(state, dt, scheme="lawson_rk4_multirate", substep_cfl=0.3,
                           hermite_chunk=chunk, **kw)
        for name in ("z_plus", "z_minus", "g"):
            assert _rel(getattr(out, name), getattr(ref, name)) < 1e-14, name
        assert out.time == ref.time


def test_substep_counts_follow_the_courant_rule():
    """n_j is the smallest n with (dt/n)·max|∇⊥(Φ + √β_i λ_j Ψ)| ≤ C·Δ, computed here in
    numpy; the macro step of mode="multirate" makes the fastest slices substep."""
    with enable_x64():
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8, Lx=1.0, Ly=1.0, Lz=2.0)
        state = _turbulent_state(grid, 12, amplitude=0.5, beta_i=2.0)
        dt = compute_cfl_timestep(state, v_A=1.0, mode="multirate")
        lam, _, _ = _eigensystem(12, LAMBDA)
        dphi_dx, dphi_dy, dpsi_dx, dpsi_dy = (
            np.asarray(a) for a in _perp_gradients(
                state.z_plus, state.z_minus, grid.kx, grid.ky, grid.Nz, grid.Ny, grid.Nx
            )
        )
        c = math.sqrt(2.0) * lam
        v = np.array([np.hypot(dphi_dx + cj * dpsi_dx, dphi_dy + cj * dpsi_dy).max() for cj in c])
        spacing = 1.0 / 16
        expected = np.maximum(np.ceil(dt * v / (0.3 * spacing)), 1)
        counts = _counts(state, dt, 0.3)
        np.testing.assert_array_equal(counts, expected)
        assert counts.max() >= 3 and counts.min() == 1
        # The fastest slice substeps down to lawson_rk4's own Courant dt
        dt_courant = compute_cfl_timestep(state, v_A=1.0)
        assert dt / counts.max() <= dt_courant * (1 + 1e-9)


@pytest.mark.parametrize("chunk", [1, 9])
def test_frozen_flow_substeps_are_lawson_steps(chunk):
    """In a frozen flow without dissipation the slices are independent. Slice j of one
    multirate step equals slice j of n_j lawson_rk4 steps of dt/n_j. With chunk = M+1 all
    slices take the fastest slice's substeps."""
    with enable_x64():
        state = _shear_state()
        dt = 4.0 * compute_cfl_timestep(state, v_A=1.0)
        counts = _counts(state, dt, 0.3)
        assert len(set(counts)) >= 3
        if chunk == state.M + 1:
            counts = np.full_like(counts, counts.max())
        out = gandalf_step(state, dt, eta=0.0, v_A=1.0, nu=0.0, scheme="lawson_rk4_multirate",
                           substep_cfl=0.3, hermite_chunk=chunk)
        assert float(jnp.max(jnp.abs(out.z_plus - state.z_plus))) == 0.0
        h_out = _slices(out)
        for n in sorted(set(counts)):
            ref = state
            for _ in range(n):
                ref = gandalf_step(ref, dt / n, eta=0.0, v_A=1.0, nu=0.0, scheme="lawson_rk4")
            h_ref = _slices(ref)
            for j in np.flatnonzero(counts == n):
                err = np.linalg.norm(h_out[j] - h_ref[j]) / np.linalg.norm(h_ref[j])
                assert err < 1e-12, (n, j, err)


def test_frozen_flow_stability_per_slice():
    """Each slice is stable when its own substep is inside the RK4 limit, h_j·ρ_j ≤ 2√2,
    and grows when it is outside, whatever the other slices do.

    Slice j is advected by (1 + c_j/2)Φ, so ρ_j/v_j is the same for every slice. The
    substep Courant number is set so that the largest h_j·ρ_j is 0.95·2√2 (all stable)
    or 1.15·2√2 (the slices whose h_j·ρ_j ends above 1.05·2√2 grow).
    """
    with enable_x64():
        state0 = _shear_state()
        grid = state0.grid
        lam, _, _ = _eigensystem(state0.M, LAMBDA)
        rho = np.asarray(slice_advection_radius(
            state0.z_plus, state0.z_minus, jnp.asarray(lam), grid.kx, grid.ky,
            grid.dealias_mask, grid.Nz, grid.Ny, grid.Nx, tol=1e-10,
        ).rho)
        spacing = min(grid.Lx / grid.Nx, grid.Ly / grid.Ny, grid.Lz / grid.Nz)
        dphi = np.asarray(_perp_gradients(
            state0.z_plus, state0.z_minus, grid.kx, grid.ky, grid.Nz, grid.Ny, grid.Nx
        )[0])
        v = np.abs(1.0 + lam / 2.0) * np.abs(dphi).max()
        fastest = np.argmax(rho)
        kappa = rho[fastest] / v[fastest]  # ρ_j / v_j, the same for every slice
        dt = 5.0 * RK4_IMAGINARY_AXIS_LIMIT / rho.max()
        n0 = _slice_norms(state0)
        for factor in (0.95, 1.15):
            substep_cfl = factor * RK4_IMAGINARY_AXIS_LIMIT / (kappa * spacing)
            counts = _counts(state0, dt, substep_cfl)
            y = dt / counts * rho  # each slice's own h_j·ρ_j
            state = state0
            # Few enough steps that round-off in the projections does not carry the
            # growing slices into the stable ones.
            for _ in range(12):
                state = gandalf_step(state, dt, eta=0.0, v_A=1.0, nu=0.0,
                                     scheme="lawson_rk4_multirate", substep_cfl=substep_cfl)
            ratio = _slice_norms(state) / n0
            stable = y <= 0.95 * RK4_IMAGINARY_AXIS_LIMIT * (1 + 1e-4)
            unstable = y >= 1.05 * RK4_IMAGINARY_AXIS_LIMIT
            assert np.all(ratio[stable] <= 1.0 + 1e-6), (factor, ratio, y)
            assert np.all(ratio[unstable] > 100.0), (factor, ratio, y)
            if factor == 0.95:
                assert stable.all()
            else:
                assert stable.any() and unstable.any()


def test_convergence_order():
    """Refining dt and the substep Courant number together keeps every n_j and halves
    every substep. In a frozen flow the stages see the exact Φ and Ψ, and the slices
    converge at RK4's fourth order. With evolving z± the coupled step is second order,
    as lawson_rk4: the Euler predictor of z± at the midpoint is off by O(dt²)."""
    with enable_x64():
        frozen = _shear_state()
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8, Lx=1.0, Ly=1.0, Lz=1.0)
        turbulent = _turbulent_state(grid, 12, amplitude=0.5)
        for state, order, dt0 in (
            (frozen, 4, 4.0 * compute_cfl_timestep(frozen, v_A=1.0)),
            (turbulent, 2, compute_cfl_timestep(turbulent, v_A=1.0, mode="multirate")),
        ):
            T = 2 * dt0

            def run(scheme, dt, substep_cfl=0.3):
                s = state
                for _ in range(round(T / dt)):
                    s = gandalf_step(s, dt, eta=0.0, v_A=1.0, nu=0.0, scheme=scheme,
                                     substep_cfl=substep_cfl)
                return s

            ref = run("lawson_rk4", dt0 / 64)
            counts = _counts(state, dt0, 0.2)
            assert counts.max() >= 3 and len(set(counts)) >= 3
            errs = []
            for k in range(3):
                assert np.array_equal(_counts(state, dt0 / 2**k, 0.2 / 2**k), counts)
                out = run("lawson_rk4_multirate", dt0 / 2**k, 0.2 / 2**k)
                errs.append(_rel(out.g, ref.g))
            rates = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
            assert np.all(np.abs(rates - order) < 0.35), (order, errs, rates)


def test_ideal_step_conserves_free_energy():
    """η = ν = 0 in a frozen flow: streaming is exact and RK4 on a skew operator has
    |R(iy)|² = 1 - y⁶/72 + ..., so the free energy W changes by O(h⁶) per substep. Halving
    dt and the substep Courant number together divides the change per step by 2⁶."""
    with enable_x64():
        state = _shear_state(M=10)
        p = jnp.asarray(free_energy_weights(10, LAMBDA, "zero"))
        wx = np.full(9, 2.0)
        wx[0] = wx[-1] = 1.0
        w = jnp.asarray(wx)[None, None, :, None]

        def W(g):
            return float(jnp.sum(p * jnp.sum(w * jnp.abs(g) ** 2, axis=(0, 1, 2))))

        dt = 4.0 * compute_cfl_timestep(state, v_A=1.0)
        errs = []
        for k in range(2):
            out = gandalf_step(state, dt / 2**k, eta=0.0, v_A=1.0, nu=0.0,
                               scheme="lawson_rk4_multirate", substep_cfl=0.3 / 2**k)
            errs.append(abs(W(out.g) - W(state.g)) / W(state.g))
        assert errs[0] < 1e-5, errs
        assert 48.0 < errs[0] / errs[1] < 80.0, errs


@pytest.mark.parametrize("dt_policy", ["fixed", "cfl"])
def test_device_loop_matches_gandalf_step(dt_policy):
    """DeviceLoop runs the same kernel, with CflDt giving the multirate macro step."""
    with enable_x64():
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8, Lx=1.0, Ly=1.0, Lz=1.0)
        state = _turbulent_state(grid, 6, amplitude=0.5, nu=0.5)
        params = StepParams(eta=0.5, v_A=1.0, hyper_r=2, hyper_n=4,
                            scheme="lawson_rk4_multirate", substep_cfl=0.2)
        ref, dts = state, []
        for _ in range(3):
            dt = 1e-2 if dt_policy == "fixed" else compute_cfl_timestep(
                ref, v_A=1.0, cfl_safety=0.3, mode="multirate"
            )
            assert _counts(ref, dt, 0.2).max() >= 2
            dts.append(dt)
            ref = gandalf_step(ref, dt, **params._asdict())
        policy = FixedDt(1e-2) if dt_policy == "fixed" else CflDt(0.3)
        loop = DeviceLoop(state, RunConfig(step=params, dt=policy))
        loop.advance(3)
        for name in ("z_plus", "z_minus", "g"):
            assert _rel(getattr(loop.state, name), getattr(ref, name)) < 1e-12, name
        assert loop.state.time == pytest.approx(sum(dts), rel=1e-12)


def test_cfl_multirate_mode_leaves_out_the_hermite_term():
    with enable_x64():
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8, Lx=1.0, Ly=1.0, Lz=1.0)
        state = _turbulent_state(grid, 8, amplitude=0.5)
        dt = compute_cfl_timestep(state, v_A=1.0, mode="multirate")
        expected = cfl_timestep_estimate(
            state.z_plus, state.z_minus, grid, 8, 1.0, LAMBDA, 1.0, 0.3, hermite=False
        )
        assert dt == pytest.approx(float(expected), rel=1e-12)
        assert dt > 2 * compute_cfl_timestep(state, v_A=1.0)
        fluid = state.model_copy(update=dict(M=0, g=state.g[..., :1]))
        assert compute_cfl_timestep(fluid, v_A=1.0, mode="multirate") == pytest.approx(
            compute_cfl_timestep(fluid, v_A=1.0), rel=1e-12
        )


def test_rejects_bad_arguments():
    grid = SpectralGrid3D.create(Nx=8, Ny=8, Nz=8)
    state = _turbulent_state(grid, 4, amplitude=0.5)
    with pytest.raises(ValueError, match="substep_cfl"):
        gandalf_step(state, 1e-3, eta=0.0, v_A=1.0, scheme="lawson_rk4_multirate", substep_cfl=0.0)
    with pytest.raises(ValueError, match="hermite_chunk"):
        gandalf_step(state, 1e-3, eta=0.0, v_A=1.0, scheme="lawson_rk4_multirate", hermite_chunk=0)
    complex_state = _turbulent_state(grid, 16, amplitude=0.5, Lambda=0.5)
    with pytest.raises(ValueError, match="real eigensystem|complex eigenvalues"):
        gandalf_step(complex_state, 1e-3, eta=0.0, v_A=1.0, scheme="lawson_rk4_multirate")
    with pytest.raises(ValueError, match="substep_cfl"):
        DeviceLoop(state, RunConfig(
            step=StepParams(eta=0.0, v_A=1.0, scheme="lawson_rk4_multirate", substep_cfl=-1.0),
            dt=FixedDt(1e-3),
        ))


@pytest.mark.parametrize("z_dtype", [jnp.complex64, jnp.complex128])
def test_dtypes_follow_lawson_rk4(z_dtype):
    """A complex64 state under float64 steps in float32, and complex64 g next to complex128
    z± is promoted, as in lawson_rk4."""
    with enable_x64():
        grid = SpectralGrid3D.create(Nx=12, Ny=10, Nz=8, Lx=1.0, Ly=1.0, Lz=1.0)
        state = _turbulent_state(grid, 5, amplitude=0.5, nu=0.7)
        state = state.model_copy(update=dict(
            z_plus=state.z_plus.astype(z_dtype), z_minus=state.z_minus.astype(z_dtype),
            g=state.g.astype(jnp.complex64),
        ))
        dt = compute_cfl_timestep(state, v_A=1.0, mode="multirate")
        assert _counts(state, dt, 0.3).max() >= 2
        kw = dict(eta=0.4, v_A=1.0, hyper_r=2, hyper_n=3, eta_z=0.3)
        ref = gandalf_step(state, dt, scheme="lawson_rk4", **kw)
        out = gandalf_step(state, dt, scheme="lawson_rk4_multirate", **kw)
        for name in ("z_plus", "z_minus", "g"):
            assert getattr(out, name).dtype == getattr(ref, name).dtype, name


def test_exact_substep_cfl_puts_every_slice_at_its_own_rk4_limit():
    """exact_substep_cfl(state, s) gives n_j = ceil(dt·ρ_j / (s·2√2)) on the state it is
    computed on, with ρ_j each slice's own exact radius."""
    with enable_x64():
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8, Lx=1.0, Ly=1.0, Lz=1.0)
        state = _turbulent_state(grid, 10, amplitude=0.5)
        C = exact_substep_cfl(state, 0.8)
        lam, _, _ = _eigensystem(10, LAMBDA)
        # exact_substep_cfl follows compute_streaming_eigensystem's order; match by value
        _, lam_c, _, _ = compute_streaming_eigensystem(10, LAMBDA, "zero")
        order = [int(np.argmin(np.abs(np.asarray(lam_c) - x))) for x in lam]
        rho = np.asarray(slice_advection_radius(
            state.z_plus, state.z_minus, jnp.asarray(lam), grid.kx, grid.ky,
            grid.dealias_mask, grid.Nz, grid.Ny, grid.Nx, tol=1e-3,
        ).rho)
        dt = compute_cfl_timestep(state, v_A=1.0, mode="multirate")
        counts = _counts(state, dt, C[order])
        expected = np.maximum(np.ceil(dt * rho / (0.8 * RK4_IMAGINARY_AXIS_LIMIT)), 1)
        assert np.abs(counts - expected).max() <= 1
        assert np.mean(counts == expected) > 0.8
        # ρ_j ≤ k⊥,max·v_j and k⊥,max·Δ ≤ 2√2π/3: never below the rigorous Courant number
        assert np.all(C >= 0.8 * 3 / np.pi)


def test_per_slice_courant_numbers():
    """substep_cfl may be one value per slice: equal values give the scalar step, and
    the device loop takes them as data."""
    with enable_x64():
        grid = SpectralGrid3D.create(Nx=16, Ny=16, Nz=8, Lx=1.0, Ly=1.0, Lz=1.0)
        state = _turbulent_state(grid, 6, amplitude=0.5, nu=0.5)
        dt = compute_cfl_timestep(state, v_A=1.0, mode="multirate")
        kw = dict(eta=0.5, v_A=1.0, hyper_r=2, hyper_n=4, scheme="lawson_rk4_multirate")
        scalar = gandalf_step(state, dt, substep_cfl=0.2, **kw)
        array = gandalf_step(state, dt, substep_cfl=np.full(7, 0.2), **kw)
        assert _rel(array.g, scalar.g) < 1e-15
        C = tuple(float(c) for c in np.linspace(0.15, 0.4, 7))
        ref = gandalf_step(state, dt, substep_cfl=C, **kw)
        assert _rel(ref.g, scalar.g) > 1e-8
        params = StepParams(eta=0.5, v_A=1.0, hyper_r=2, hyper_n=4,
                            scheme="lawson_rk4_multirate", substep_cfl=C)
        loop = DeviceLoop(state, RunConfig(step=params, dt=FixedDt(dt)))
        loop.advance(1)
        assert _rel(loop.state.g, ref.g) < 1e-12
        with pytest.raises(ValueError, match="substep_cfl"):
            gandalf_step(state, dt, substep_cfl=(0.2, 0.3), **kw)
